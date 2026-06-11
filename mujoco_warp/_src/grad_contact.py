# Copyright 2026 The Newton Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Implicit-function-theorem gradients through the constraint solver.

The constraint solver minimizes (Newton/CG, both paths)::

    L(qacc) = 0.5 * qacc^T M qacc - qacc^T qfrc_smooth
              + 0.5 * sum_active D_i * (J_i qacc - aref_i)^2

At convergence the optimality condition is::

    F(qacc) = M qacc - qfrc_smooth + J_A^T diag(D_A) (J_A qacc - aref_A) = 0

where ``A`` is the active (QUADRATIC-state) constraint set. Differentiating
implicitly with the active set frozen (valid almost everywhere; the solution
is non-smooth exactly at activation boundaries)::

    dqacc/dtheta = -H^{-1} dF/dtheta,   H = M + J_A^T diag(D_A) J_A

so the adjoint of any upstream quantity follows from one linear solve with
the same Hessian the Newton solver already factorizes::

    lam = H^{-1} dL/dqacc
    dL/dqfrc_smooth = lam
    dL/daref_i      = D_i * (J_i lam)               (i in A)
    dL/dD_i         = (J_i qacc - aref_i) * (J_i lam)  (i in A)
    dL/dJ_i         = D_i * (lam (J_i qacc - aref_i) + (J_i lam) qacc)  (i in A)

This module provides a research-grade implementation: forward state is read
back to the host and the per-world linear systems are solved with NumPy
(H is (nv, nv), symmetric positive definite). A GPU tile-Cholesky backward
is the natural follow-up once the interface settles.
"""

import numpy as np

from mujoco_warp._src import types

__all__ = ["solve_ift_backward"]


def _gather_world_state(m: types.Model, d: types.Data, worldid: int):
  """Reads one world's converged solver state back to the host.

  Returns (H, J_active, D_active, residual_active, active_efcids, qacc).
  """
  nv = m.nv
  nefc = int(d.nefc.numpy()[worldid])
  qacc = d.qacc.numpy()[worldid].astype(np.float64)

  efc_D = d.efc.D.numpy()[worldid, :nefc].astype(np.float64)
  efc_aref = d.efc.aref.numpy()[worldid, :nefc].astype(np.float64)
  efc_state = d.efc.state.numpy()[worldid, :nefc]

  # Dense J rows (works for both jacobian modes).
  J = np.zeros((nefc, nv), dtype=np.float64)
  if m.is_sparse:
    rownnz = d.efc.J_rownnz.numpy()[worldid, :nefc]
    rowadr = d.efc.J_rowadr.numpy()[worldid, :nefc]
    colind = d.efc.J_colind.numpy()[worldid, 0]
    Jval = d.efc.J.numpy()[worldid, 0]
    for i in range(nefc):
      sl = slice(rowadr[i], rowadr[i] + rownnz[i])
      J[i, colind[sl]] = Jval[sl]
  else:
    J = d.efc.J.numpy()[worldid, :nefc, :nv].astype(np.float64)

  # Dense M. In dense-jacobian mode d.M is stored densely (nv_pad, nv_pad);
  # in sparse mode it uses the CSR-like (1, nM) layout described by M_rownnz/
  # M_rowadr/M_colind (lower triangle).
  M = np.zeros((nv, nv), dtype=np.float64)
  if m.is_sparse:
    M_rownnz = m.M_rownnz.numpy()
    M_rowadr = m.M_rowadr.numpy()
    M_colind = m.M_colind.numpy()
    M_val = d.M.numpy()[worldid, 0]
    for i in range(nv):
      sl = slice(M_rowadr[i], M_rowadr[i] + M_rownnz[i])
      cols = M_colind[sl]
      M[i, cols] = M_val[sl]
      M[cols, i] = M_val[sl]  # stored triangle -> symmetrize
  else:
    M = d.M.numpy()[worldid, :nv, :nv].astype(np.float64)
    M = 0.5 * (M + M.T)  # symmetrize against padding artifacts

  active = efc_state == types.ConstraintState.QUADRATIC.value
  J_A = J[active]
  D_A = efc_D[active]
  res_A = J_A @ qacc - efc_aref[active]

  H = M + J_A.T @ (D_A[:, None] * J_A)
  return H, J_A, D_A, res_A, np.nonzero(active)[0], qacc


def solve_ift_backward(
  m: types.Model,
  d: types.Data,
  dL_dqacc: np.ndarray,
) -> dict:
  """Backpropagates a loss gradient through the converged constraint solve.

  Implicit-function-theorem adjoint at the solver fixed point with the active
  constraint set frozen. Call after `step`/`forward` has converged `d.qacc`.

  Args:
    m: Model.
    d: Data holding the converged solve (qacc, efc.*).
    dL_dqacc: Upstream gradient, shape (nworld, nv).

  Returns:
    dict with per-world host arrays:
      qfrc_smooth: (nworld, nv) - dL/dqfrc_smooth
      aref: list of (n_active,) - dL/defc_aref, active rows only
      D: list of (n_active,) - dL/defc_D, active rows only
      J: list of (n_active, nv) - dL/defc_J, active rows only
      active_efcids: list of (n_active,) - row indices the entries refer to
      lam: (nworld, nv) - the adjoint solve H^-1 dL/dqacc
  """
  dL_dqacc = np.asarray(dL_dqacc, dtype=np.float64)
  if dL_dqacc.shape != (d.nworld, m.nv):
    raise ValueError(f"dL_dqacc must have shape ({d.nworld}, {m.nv}), got {dL_dqacc.shape}")

  out = {
    "qfrc_smooth": np.zeros((d.nworld, m.nv)),
    "lam": np.zeros((d.nworld, m.nv)),
    "aref": [],
    "D": [],
    "J": [],
    "active_efcids": [],
  }

  for w in range(d.nworld):
    H, J_A, D_A, res_A, efcids, qacc = _gather_world_state(m, d, w)
    lam = np.linalg.solve(H, dL_dqacc[w])

    Jlam = J_A @ lam
    out["lam"][w] = lam
    # F = M qacc - qfrc_smooth + J^T D (J qacc - aref); dqacc/dtheta = -H^-1 dF/dtheta
    # => dL/dtheta = -lam^T dF/dtheta with signs folded below.
    out["qfrc_smooth"][w] = lam  # dF/dqfrc_smooth = -I
    out["aref"].append(D_A * Jlam)  # dF/daref_i = -J_i^T D_i
    out["D"].append(-res_A * Jlam)  # dF/dD_i = J_i^T res_i
    out["J"].append(-(np.outer(Jlam * D_A, qacc) + lam[None, :] * (D_A * res_A)[:, None]))
    out["active_efcids"].append(efcids)

  return out

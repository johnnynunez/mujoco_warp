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

"""Tests for implicit-function-theorem gradients through the constraint solver."""

import mujoco
import numpy as np
import warp as wp
from absl.testing import absltest
from absl.testing import parameterized

import mujoco_warp as mjw
from mujoco_warp._src import grad_contact

# A sphere resting on a plane: persistent contact, active constraint set
# stable under small perturbations (required for IFT validity).
_CONTACT_XML = """
<mujoco>
  <option timestep="0.002" tolerance="1e-12" iterations="200" ls_iterations="100"/>
  <worldbody>
    <geom type="plane" size="5 5 .1"/>
    <body pos="0 0 0.099">
      <freejoint/>
      <geom type="sphere" size="0.1" mass="1"/>
    </body>
  </worldbody>
</mujoco>
"""


def _make_converged(jacobian: str):
  spec = mujoco.MjSpec.from_string(_CONTACT_XML)
  mjm = spec.compile()
  mjm.opt.jacobian = {
    "SPARSE": mujoco.mjtJacobian.mjJAC_SPARSE,
    "DENSE": mujoco.mjtJacobian.mjJAC_DENSE,
  }[jacobian]
  mjd = mujoco.MjData(mjm)
  # settle into persistent contact
  for _ in range(50):
    mujoco.mj_step(mjm, mjd)
  mujoco.mj_forward(mjm, mjd)
  m = mjw.put_model(mjm)
  d = mjw.put_data(mjm, mjd)
  mjw.forward(m, d)
  wp.synchronize()
  return mjm, mjd, m, d


class IFTBackwardTest(parameterized.TestCase):
  """Validates IFT adjoints against finite differences of the full solve."""

  @parameterized.parameters("SPARSE", "DENSE")
  def test_qfrc_smooth_gradient_matches_finite_differences(self, jacobian):
    mjm, mjd, m, d = _make_converged(jacobian)
    nv = mjm.nv

    self.assertGreater(int(d.nefc.numpy()[0]), 0, "Test scene must have active constraints")

    # Loss: L = g^T qacc with fixed random g => dL/dqacc = g.
    rng = np.random.default_rng(0)
    g = rng.normal(size=nv)

    grads = grad_contact.solve_ift_backward(m, d, g[None])
    analytic = grads["qfrc_smooth"][0]

    # Finite differences through the full GPU solve: perturb qfrc_smooth via
    # qfrc_applied (additive in the smooth force) and re-solve. eps must stay
    # well above f32 solver noise (~1e-7 relative): central differences with
    # eps=1e-3 give truncation error O(eps^2)=1e-6 and noise error
    # O(1e-7/eps)=1e-4 on O(1) losses.
    eps = 1e-3
    fd = np.zeros(nv)
    for k in range(nv):
      vals = []
      for sign in (+1.0, -1.0):
        mjd2 = mujoco.MjData(mjm)
        mjd2.qpos[:] = mjd.qpos
        mjd2.qvel[:] = mjd.qvel
        mjd2.qfrc_applied[:] = 0.0
        mjd2.qfrc_applied[k] = sign * eps
        mujoco.mj_forward(mjm, mjd2)
        m2 = mjw.put_model(mjm)
        d2 = mjw.put_data(mjm, mjd2)
        mjw.forward(m2, d2)
        wp.synchronize()
        vals.append(float(g @ d2.qacc.numpy()[0]))
      fd[k] = (vals[0] - vals[1]) / (2 * eps)

    # IFT freezes the active set; FD goes through the full solver. Agreement
    # to ~1e-4 relative confirms the adjoint formula and active-set handling.
    np.testing.assert_allclose(analytic, fd, rtol=1e-3, atol=1e-4)

  @parameterized.parameters("SPARSE", "DENSE")
  def test_lam_solves_hessian_system(self, jacobian):
    """H @ lam must reproduce dL/dqacc (internal consistency of the solve)."""
    mjm, mjd, m, d = _make_converged(jacobian)
    nv = mjm.nv

    rng = np.random.default_rng(1)
    g = rng.normal(size=nv)
    grads = grad_contact.solve_ift_backward(m, d, g[None])

    H, _, _, _, _, _ = grad_contact._gather_world_state(m, d, 0)
    np.testing.assert_allclose(H @ grads["lam"][0], g, rtol=1e-9, atol=1e-9)

  def test_aref_gradient_shape_and_activity(self):
    """aref/D/J grads must cover exactly the active constraint rows."""
    mjm, mjd, m, d = _make_converged("SPARSE")
    nv = mjm.nv

    grads = grad_contact.solve_ift_backward(m, d, np.ones((1, nv)))
    n_active = len(grads["active_efcids"][0])
    self.assertGreater(n_active, 0)
    self.assertEqual(grads["aref"][0].shape, (n_active,))
    self.assertEqual(grads["D"][0].shape, (n_active,))
    self.assertEqual(grads["J"][0].shape, (n_active, nv))

  def test_rejects_bad_gradient_shape(self):
    mjm, mjd, m, d = _make_converged("SPARSE")
    with self.assertRaisesRegex(ValueError, "shape"):
      grad_contact.solve_ift_backward(m, d, np.ones(3))


if __name__ == "__main__":
  absltest.main()

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
"""Tests for GPU determinism (contact sorting, constraint rows, full-pipeline bitwise stability)."""

import mujoco
import numpy as np
import warp as wp
from absl.testing import absltest
from absl.testing import parameterized

import mujoco_warp as mjw
from mujoco_warp import test_data


def _heightfield_fixture(nworld, deterministic, jacobian):
  """Build a branching articulated body contacting a small slope, without external assets."""
  mjm, mjd, m, _ = test_data.fixture(
    xml="""
      <mujoco>
        <option timestep="0.005" integrator="Euler" cone="pyramidal" iterations="50"/>
        <default>
          <joint damping="0.3" armature="0.01"/>
          <geom density="777" friction="0.8 0.02 0.001"/>
        </default>
        <asset><hfield name="slope" nrow="9" ncol="9" size="1 1 0.1 0.1"/></asset>
        <worldbody>
          <geom type="hfield" hfield="slope"/>
          <body pos="0 0 0.56" euler="0 3 1">
            <freejoint/>
            <geom type="box" size="0.12 0.09 0.12" mass="2.3"/>
            <body pos="0 0.12 -0.1">
              <joint name="left" axis="0 1 0"/>
              <geom type="capsule" fromto="0 0 0 0 0 -0.17" size="0.025"/>
              <body pos="0 0 -0.17">
                <joint axis="0 1 0"/>
                <geom type="capsule" fromto="0 0 0 0 0 -0.17" size="0.025"/>
                <geom type="box" pos="0.025 0 -0.19" size="0.065 0.045 0.025"/>
              </body>
            </body>
            <body pos="0 -0.12 -0.1">
              <joint name="right" axis="0 1 0"/>
              <geom type="capsule" fromto="0 0 0 0 0 -0.17" size="0.028"/>
              <body pos="0 0 -0.17">
                <joint axis="0 1 0"/>
                <geom type="capsule" fromto="0 0 0 0 0 -0.17" size="0.028"/>
                <geom type="box" pos="0.025 0 -0.19" size="0.065 0.045 0.025"/>
              </body>
            </body>
            <body pos="0 0.11 0.1">
              <joint axis="1 0 0"/>
              <geom type="capsule" fromto="0 0 0 0 0.2 0" size="0.018"/>
            </body>
            <body pos="0 -0.11 0.1">
              <joint axis="1 0 0"/>
              <geom type="capsule" fromto="0 0 0 0 -0.2 0" size="0.021"/>
            </body>
          </body>
        </worldbody>
        <actuator><motor joint="left"/><motor joint="right"/></actuator>
      </mujoco>
    """,
    nworld=1,
    overrides={"opt.jacobian": jacobian},
  )
  mjm.hfield_data[:] = np.tile(np.linspace(0, 1, 9), 9)
  mjd.qvel[:] = np.linspace(-0.05, 0.08, mjm.nv)
  mjd.ctrl[:] = [0.1, -0.15]
  mujoco.mj_forward(mjm, mjd)
  m = mjw.put_model(mjm)
  m.opt.deterministic = deterministic
  d = mjw.put_data(mjm, mjd, nworld=nworld, nconmax=128, njmax=256)
  return m, d


class HeightfieldBatchDeterminismTest(parameterized.TestCase):
  """The same world must not change trajectory when repeated or placed in another batch."""

  @parameterized.product(jacobian=("DENSE", "SPARSE"), nworld=(2, 9))
  def test_heightfield_batch_invariance(self, jacobian, nworld):
    """Compare independent replays and different batch sizes at every physics step."""
    m, single = _heightfield_fixture(1, True, jacobian)
    _, batch = _heightfield_fixture(nworld, True, jacobian)
    _, repeat = _heightfield_fixture(nworld, True, jacobian)
    contacts_seen = False
    for step in range(80):
      for d in (single, batch, repeat):
        mjw.step(m, d)
        self.assertFalse(d.overflow.numpy().any(), f"Overflow at step {step}")
        contacts_seen |= bool(d.nacon.numpy()[0])
      for name in ("qpos", "qvel", "qacc", "qfrc_constraint"):
        reference = getattr(single, name).numpy()
        self.assertTrue(np.isfinite(reference).all())
        for d in (batch, repeat):
          values = getattr(d, name).numpy()
          self.assertTrue(np.isfinite(values).all())
          expected = np.broadcast_to(reference, values.shape)
          self.assertEqual(values.tobytes(), expected.tobytes(), f"{name}: step {step}, nworld={nworld}")
    self.assertTrue(contacts_seen, "The regression must exercise contact dynamics")


if __name__ == "__main__":
  wp.init()
  absltest.main()

"""Simulator: determinism, label sanity, GPU render (skipped without EGL)."""

import numpy as np
import pytest

from drjepa.config import GridConfig
from drjepa.labels import sim_targets
from drjepa.sim import RoverSim
from drjepa.sim.expert import ArcPlanner


def _gl_available():
    try:
        from drjepa.sim.render import gl_context
        gl_context()
        return True
    except Exception:
        return False


def test_seed_reproduces_world():
    a, b = RoverSim(seed=7), RoverSim(seed=7)
    assert a.scenario == b.scenario and len(a.props) == len(b.props)
    assert np.allclose(a.obs_xz, b.obs_xz) and a.camera == b.camera
    for _ in range(30):
        ea, eb = ArcPlanner(a), ArcPlanner(b)
        a.step(*ea.plan())
        b.step(*eb.plan())
    assert a.x == pytest.approx(b.x) and a.z == pytest.approx(b.z)


def test_expert_reaches_open_goal():
    sim = RoverSim(seed=3, scenario="open", spawn_mode="normal")
    ex = ArcPlanner(sim, np.random.default_rng(3))
    for _ in range(sim.cfg.max_frames):
        info = sim.step(*ex.plan())
        if info["reached"] or info["tipped"]:
            break
    assert info["reached"]


@pytest.mark.skipif(not _gl_available(), reason="no headless OpenGL")
def test_render_and_labels_agree():
    sim = RoverSim(seed=21)
    rgb, depth, cls, (R, t) = sim.render(gt=True)
    c = sim.camera
    assert rgb.shape == (c.height, c.width, 3) and rgb.dtype == np.uint8
    assert depth.shape == (c.height, c.width)
    ground = cls == 0
    assert ground.mean() > 0.2
    # rendered ground depth reprojects onto the terrain the physics uses
    P = c.unproject(depth, R, t)[ground & (depth < 15)]
    from drjepa.geo import robot_to_world
    wx, wz = robot_to_world(P[:, 0], P[:, 2], sim.yaw)
    h = sim.terrain.height(sim.x + wx, sim.z + wz) - sim.terrain.height(sim.x, sim.z)
    assert np.median(np.abs(P[:, 1] - h)) < 0.03
    L = sim_targets(sim, depth, cls, R, t, GridConfig())
    assert L["vis"].any() and set(L) >= {"vis", "occ", "obs_h", "elev", "grade", "sand"}
    sim.close()

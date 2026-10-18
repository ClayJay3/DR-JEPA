"""Belief map, planner and navigator (no GPU / simulator needed)."""

import numpy as np

from drjepa.camera import Camera
from drjepa.config import GridConfig, RobotProfile
from drjepa.nav.mapping import BeliefMap, MapConfig
from drjepa.nav.navigator import Navigator
from drjepa.nav.planning import Planner, astar


def test_astar_goes_around_a_wall():
    cost = np.ones((30, 30))
    lethal = np.zeros((30, 30), bool)
    lethal[15, 0:25] = True                         # wall with a gap at j >= 25
    path = astar(cost, lethal, (5, 5), (25, 5))
    assert path is not None and path[0] == (5, 5) and path[-1] == (25, 5)
    assert all(not lethal[p] for p in path)
    assert max(j for _, j in path) >= 25


def grids(occ_cells=(), C=48):
    occ = np.full((C, C), -6.0, np.float32)
    for i, j in occ_cells:
        occ[i, j] = 3.0
    z = np.zeros((C, C), np.float32)
    return {"occ_logit": occ, "occ": 1 / (1 + np.exp(-occ)), "vis": np.ones((C, C), np.float32),
            "obs_h": z, "elev": z, "grade": z, "sand": z + 0.01}


def test_fusion_marks_obstacle_in_world_frame():
    grid = GridConfig()
    m = BeliefMap(MapConfig(vo=False), grid)
    m.init_at(np.zeros(2))
    for _ in range(10):        # cell (24, 10) = 0.25 m right, 5.25 m ahead; facing east
        m.fuse(grids([(24, 10)]), np.zeros(2), 90.0)
    i, j = m.index(np.array(5.25), np.array(-0.25))  # east 5.25, south 0.25
    assert m.prob()[int(i), int(j)] > 0.9
    i2, j2 = m.index(np.array(0.25), np.array(5.25))  # same distance north: free
    assert m.prob()[int(i2), int(j2)] < 0.1


def test_slope_is_direction_aware():
    grid = GridConfig()
    m = BeliefMap(MapConfig(vo=False), grid)
    m.init_at(np.zeros(2))
    g = grids()
    g["elev"] = (m._gx * 0.5).astype(np.float32)     # rises to the robot's right
    g["grade"] = np.full_like(g["elev"], 0.5)
    for _ in range(5):                                # facing east: right = south
        m.fuse(g, np.zeros(2), 90.0)
    i, j = m.index(np.array(5.0), np.array(0.0))
    pitch, roll = m.slope_along(i, j, np.radians([0.0, 90.0, 180.0]))
    np.testing.assert_allclose(pitch, [-0.5, 0.0, 0.5], atol=0.02)   # north = downhill
    np.testing.assert_allclose(roll, [0.0, 0.5, 0.0], atol=0.02)


def test_planner_routes_around_believed_obstacle():
    grid, rb = GridConfig(), RobotProfile()
    m = BeliefMap(MapConfig(vo=False), grid)
    m.init_at(np.zeros(2))
    wall = [(i, 16) for i in range(14, 35)]          # 6 m ahead, x in [-5, +5.5]
    for _ in range(15):
        m.fuse(grids(wall), np.zeros(2), 0.0)
    p = Planner(rb)
    p.replan(m, np.zeros(2), np.array([0.0, 20.0]))
    assert p.path is not None
    assert np.abs(p.path[:, 0]).max() > 4.0          # detours past the wall's end


class FlatWorld:
    """Perception stub: open ground everywhere."""

    def __call__(self, rgb, camera, R=None, t=None, body_pitch=0.0, body_roll=0.0):
        return grids()


def test_navigator_drives_toward_goal():
    nav = Navigator(FlatWorld(), camera=Camera.from_fov(320, 240, 90))
    s = {"x": 0.0, "z": 0.0, "heading": 0.0, "speed": 2.0, "goal_x": 0.0, "goal_z": 40.0}
    cmd = None
    for _ in range(5):
        cmd = nav.step(np.zeros((240, 320, 3), np.uint8), s)
    assert cmd.throttle > 0.5 and abs(cmd.steer) < 0.2
    s["goal_x"], s["goal_z"] = 40.0, 0.0               # goal due east -> steer right
    for _ in range(4):
        cmd = nav.step(np.zeros((240, 320, 3), np.uint8), s)
    assert cmd.steer > 0.5


def test_navigator_rejects_bad_inputs():
    import pytest
    nav = Navigator(FlatWorld())
    s = {"x": 0.0, "z": 0.0, "heading": 0.0, "speed": 0.0, "goal_x": 0.0, "goal_z": 10.0}
    with pytest.raises(ValueError, match="Camera"):
        nav.step(np.zeros((240, 320, 3), np.uint8), s)
    with pytest.raises(ValueError, match="uint8"):
        nav.step(np.zeros((240, 320, 3), np.float32), s, camera=Camera.from_fov(320, 240, 90))

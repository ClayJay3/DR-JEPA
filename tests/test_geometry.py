"""Camera model, robot-frame poses, and the depth -> grid labeller."""

import math

import numpy as np
import pytest

from drjepa.camera import Camera, attitude
from drjepa.config import GridConfig
from drjepa.data.recordings import quat_to_mat, robot_frame_pose
from drjepa.labels import cell_centers, depth_targets


def test_attitude_conventions():
    # nose up: forward tilts toward +y; right side down: right axis toward -y
    f = attitude(10, 0) @ np.array([0, 0, 1.0])
    assert f[1] == pytest.approx(math.sin(math.radians(10)))
    r = attitude(0, 10) @ np.array([1.0, 0, 0])
    assert r[1] == pytest.approx(-math.sin(math.radians(10)))


@pytest.mark.parametrize("hfov,h,pitch,roll", [(90, 0.6, 0, 0), (120, 0.3, -20, 5), (55, 2.1, 4, -6)])
def test_project_unproject_roundtrip(hfov, h, pitch, roll):
    cam = Camera.from_fov(640, 400, hfov, mount_height=h, mount_pitch=pitch, mount_roll=roll)
    R, t = cam.pose(body_pitch=2.0, body_roll=-1.0)
    rng = np.random.default_rng(0)
    depth = rng.uniform(1, 30, (cam.height, cam.width)).astype(np.float32)
    P = cam.unproject(depth, R, t)
    uv, z = cam.project(P, R, t)
    v, u = np.mgrid[0:cam.height, 0:cam.width]
    assert np.allclose(uv[..., 0], u + 0.5, atol=1e-3)
    assert np.allclose(uv[..., 1], v + 0.5, atol=1e-3)
    assert np.allclose(z, depth, atol=1e-3)


def test_level_camera_sees_ground_ahead():
    cam = Camera.from_fov(640, 480, 90, mount_height=1.0)
    R, t = cam.pose()
    uv, z = cam.project(np.array([[0.0, 0.0, 5.0]]), R, t)
    # 5 m ahead on the ground: centred horizontally, below the principal point
    assert uv[0, 0] == pytest.approx(320.0)
    assert uv[0, 1] == pytest.approx(240.0 + cam.fy * 1.0 / 5.0)
    assert z[0] == pytest.approx(5.0)


def test_resize_crop_keep_geometry():
    cam = Camera.from_fov(1280, 720, 100, mount_height=0.4)
    R, t = cam.pose()
    p = np.array([[1.0, 0.2, 7.0]])
    uv, _ = cam.project(p, R, t)
    uv2, _ = cam.resized(640, 360).project(p, R, t)
    assert np.allclose(uv2, uv * 0.5, atol=1e-6)
    uv3, _ = cam.cropped(100, 50, 800, 500).project(p, R, t)
    assert np.allclose(uv3, uv - [100, 50], atol=1e-6)


def test_robot_frame_pose_identity_quaternion():
    # GL convention: identity = camera looking along -z, level
    R, t = robot_frame_pose(np.array([0, 0, 0, 1.0]), 0.5)
    assert np.allclose(R, np.diag([1, -1, 1]))
    assert np.allclose(t, [0, 0.5, 0])


def test_robot_frame_pose_pitched():
    # camera pitched 10 deg down (rotation about +x by -10 deg in GL frame)
    a = math.radians(-10) / 2
    R, _ = robot_frame_pose(np.array([math.sin(a), 0, 0, math.cos(a)]), 0.5)
    fwd = R[:, 2]
    assert fwd[1] == pytest.approx(-math.sin(math.radians(10)), abs=1e-6)
    assert np.allclose(quat_to_mat(0, 0, 0, 1), np.eye(3))


def synthetic_scene(cam, R, t, obstacle=(1.0, 3.0, 0.4, 1.0)):
    """Flat ground at y = 0 plus a box (x, z, half-width, height)."""
    rays = cam.rays() @ R.T                       # ray dirs in the robot frame (z_cam = 1)
    o = t
    # ground: o_y + s * ray_y = 0
    with np.errstate(divide="ignore", invalid="ignore"):
        s_g = np.where(rays[..., 1] < -1e-6, -o[1] / rays[..., 1], np.inf)
        ox, oz, hw, hh = obstacle
        s_f = (oz - o[2]) / rays[..., 2]           # box front face at z = oz
        px, py = o[0] + s_f * rays[..., 0], o[1] + s_f * rays[..., 1]
        hit = (s_f > 0) & (np.abs(px - ox) < hw) & (py >= 0) & (py <= hh)
    s = np.where(hit & (s_f < s_g), s_f, s_g)
    s[~np.isfinite(s)] = 0.0
    return s.astype(np.float32)                    # depth along the optical axis (z_cam = 1 rays)


def test_depth_targets_flat_ground_and_box():
    grid = GridConfig()
    cam = Camera.from_fov(640, 480, 90, mount_height=1.2, mount_pitch=-10)
    R, t = cam.pose()
    depth = synthetic_scene(cam, R, t)
    L = depth_targets(depth, cam, R, t, grid, dmax=30)
    X, Z = cell_centers(grid)
    near = L["vis"] & (Z < 8) & ~L["occ"]
    assert near.sum() > 50
    assert np.nanmax(np.abs(L["elev"][near])) < 0.06          # flat ground reads flat
    # box front at z = 3 m, x in [0.6, 1.4], 1 m tall
    i = int((1.0 + grid.half) / grid.res)
    j = int(3.0 / grid.res)
    assert L["occ"][i - 1:i + 2, j:j + 2].any()
    assert L["obs_h"][i - 1:i + 2, j:j + 2].max() == pytest.approx(1.0, abs=0.15)
    # nothing occupied on the far side of the robot
    assert not L["occ"][:grid.cells // 2 - 2].any()

"""Camera model shared by the simulator, the label builders and the robot.

Frames
------
Robot (BEV) frame: origin on the ground directly under the robot's
reference point, gravity-aligned: x right, y up, z forward along the
heading. Everything perception predicts lives on this frame's ground grid.

Camera frame (OpenCV): x right, y down, z along the optical axis.

A camera is fully described by its pinhole intrinsics plus a mount
(height above the ground plane, forward offset, pitch and roll relative
to the robot body). At run time the body's own pitch/roll (IMU) composes
with the mount, so the network always knows exactly where every image
pixel looks -- for any lens, any mast, any robot.
"""

import math
from dataclasses import dataclass, replace

import numpy as np


def rot_x(a):
    """Rotation about +x by a (rad)."""
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rot_z(a):
    """Rotation about +z by a (rad)."""
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def attitude(pitch_deg, roll_deg):
    """Body/camera attitude in the robot frame (x right, y up, z fwd).

    Pitch is nose-up positive, roll is right-side-down positive, matching
    what an IMU reports for a ground vehicle. Applied roll-then-pitch.
    """
    # nose up = forward axis (z) tilts toward +y: rotation about +x by -pitch
    return rot_x(-math.radians(pitch_deg)) @ rot_z(-math.radians(roll_deg))


# OpenCV camera axes expressed in an un-rotated robot frame:
# cam x = right (+x), cam y = down (-y), cam z = forward (+z)
CAM_FROM_LEVEL = np.diag([1.0, -1.0, 1.0])


@dataclass
class Camera:
    """Pinhole intrinsics + mount extrinsics (see module docstring)."""
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    mount_height: float = 0.6     # optical centre above the ground plane (m)
    mount_pitch: float = 0.0      # deg, + = looking up
    mount_roll: float = 0.0       # deg
    mount_forward: float = 0.0    # optical centre ahead of the reference point (m)

    # ------------------------------------------------------------ builders
    @staticmethod
    def from_fov(width, height, hfov_deg, **mount):
        """Square-pixel camera with the given horizontal field of view."""
        f = (width / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
        return Camera(width, height, f, f, width / 2.0, height / 2.0, **mount)

    @property
    def hfov_deg(self):
        """Horizontal field of view (deg)."""
        return math.degrees(2 * math.atan(self.width / (2 * self.fx)))

    @property
    def vfov_deg(self):
        """Vertical field of view (deg)."""
        return math.degrees(2 * math.atan(self.height / (2 * self.fy)))

    def K(self):
        """3x3 intrinsic matrix."""
        return np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1.0]])

    def resized(self, width, height):
        """Same camera after resizing the image to (width, height)."""
        sx, sy = width / self.width, height / self.height
        return replace(self, width=int(width), height=int(height),
                       fx=self.fx * sx, fy=self.fy * sy,
                       cx=self.cx * sx, cy=self.cy * sy)

    def cropped(self, x0, y0, width, height):
        """Same camera after cropping the window [x0, x0+w) x [y0, y0+h)."""
        return replace(self, width=int(width), height=int(height),
                       cx=self.cx - x0, cy=self.cy - y0)

    # ------------------------------------------------------------ pose
    def pose(self, body_pitch=0.0, body_roll=0.0):
        """(R, t): camera -> robot frame for the given body attitude (deg).

        R maps OpenCV camera axes to the robot frame; t is the optical
        centre in the robot frame. A point p_cam maps to R @ p_cam + t.
        """
        Rb = attitude(body_pitch, body_roll)
        Rm = attitude(self.mount_pitch, self.mount_roll)
        R = Rb @ Rm @ CAM_FROM_LEVEL
        t = Rb @ np.array([0.0, self.mount_height, self.mount_forward])
        return R, t

    # ------------------------------------------------------------ geometry
    def rays(self):
        """(H, W, 3) unit-z ray directions in the camera frame (pixel centres)."""
        v, u = np.mgrid[0:self.height, 0:self.width].astype(np.float32)
        return np.stack([(u + 0.5 - self.cx) / self.fx,
                         (v + 0.5 - self.cy) / self.fy,
                         np.ones_like(u)], -1)

    def unproject(self, depth, R, t):
        """Depth (H, W) along the optical axis -> robot-frame points (H, W, 3)."""
        p = self.rays() * depth[..., None]
        return p @ R.T + t

    def project(self, pts, R, t):
        """Robot-frame points (..., 3) -> (uv (..., 2), z_cam (...,))."""
        pc = (np.asarray(pts) - t) @ R          # R^T (p - t), row-vector form
        z = pc[..., 2]
        zs = np.where(np.abs(z) < 1e-6, 1e-6, z)
        u = pc[..., 0] / zs * self.fx + self.cx
        v = pc[..., 1] / zs * self.fy + self.cy
        return np.stack([u, v], -1), z

    def to_dict(self):
        """JSON-friendly form."""
        return {k: (float(v) if isinstance(v, (float, np.floating)) else v)
                for k, v in self.__dict__.items()}

    @staticmethod
    def from_dict(d):
        """Inverse of to_dict."""
        return Camera(**d)

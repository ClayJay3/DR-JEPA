"""Real recordings (ZED 2i SVO exports) -> labelled episodes.

`tools/zed/export_svo.py` writes rgb/depth/conf/poses per recording. This
module turns one into the SAME episode layout the simulator writes
(rgb/*.jpg, labels.npz with robot-frame grid targets and camera poses,
meta.json with the camera), so sim and real data flow through one loader.

The robot frame of every frame is built from the gravity-aligned VIO pose:
origin on the ground under the camera, y up, z = the camera's heading.
The mount height is measured from the stereo ground plane rather than
trusted from a config (the old ZED labels assumed 0.6 m; the stereo ground
says 0.39-0.40 m, which shifts every elevation label by 0.2 m).

    python -m drjepa.data.recordings recordings/zed/*
"""

import argparse
import glob
import json
import os

import cv2
import numpy as np

from ..camera import Camera
from ..config import GridConfig
from ..labels import depth_targets
from ..sim.datagen import pack_labels


def quat_to_mat(qx, qy, qz, qw):
    """Unit quaternion -> 3x3 rotation matrix."""
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)]])


def robot_frame_pose(q, height):
    """(R, t) camera -> robot frame from a gravity-aligned GL-convention pose.

    The VIO world is y-up; the camera looks along its local -z. The robot
    frame shares the camera's heading (forward projected on the horizontal)
    and sits on the ground `height` below the optical centre.
    """
    Rw = quat_to_mat(*q)
    right_c, up_c, back_c = Rw[:, 0], Rw[:, 1], Rw[:, 2]
    fwd = -back_c
    up = np.array([0.0, 1.0, 0.0])
    f = fwd - up * fwd[1]
    f /= max(np.linalg.norm(f), 1e-9)
    right = np.cross(f, up)
    B = np.stack([right, up, f])                     # world -> robot rows
    # OpenCV axes (x right, y down, z forward) in robot coordinates
    R = np.stack([B @ right_c, B @ (-up_c), B @ fwd], 1)
    return R, np.array([0.0, height, 0.0])


class Recording:
    """Lazy access to one exported recording directory."""

    def __init__(self, path):
        self.path = path.rstrip("/")
        with open(os.path.join(self.path, "meta.json")) as f:
            self.meta = json.load(f)
        rows = np.loadtxt(os.path.join(self.path, "frames.csv"), delimiter=",", skiprows=1)
        self.rows = rows.reshape(-1, 9)
        m = self.meta
        self.camera = Camera(m["width"], m["height"], m["fx"], m["fy"], m["cx"], m["cy"])

    def __len__(self):
        return len(self.rows)

    def rgb(self, k):
        """(H, W, 3) uint8 RGB."""
        return cv2.imread(os.path.join(self.path, "rgb", f"{k:05d}.jpg"))[:, :, ::-1].copy()

    def depth(self, k):
        """(H, W) float32 metres, 0 = no return."""
        return cv2.imread(os.path.join(self.path, "depth", f"{k:05d}.png"), -1).astype(np.float32) / 1000.0

    def conf(self, k):
        """(H, W) uint8 ZED confidence (0 best .. 100 worst)."""
        return cv2.imread(os.path.join(self.path, "conf", f"{k:05d}.png"), -1)

    def quat(self, k):
        """Camera orientation quaternion (x, y, z, w) in the VIO world."""
        return self.rows[k, 5:9]

    def position(self, k):
        """Camera position (x, y, z) in the VIO world."""
        return self.rows[k, 2:5]

    def estimate_height(self, n=60):
        """Camera height above the local ground from stereo (median of a
        per-frame robust ground height 1.2-4 m ahead, |x| < 1.5 m)."""
        hs = []
        for k in np.linspace(0, len(self) - 1, min(n, len(self))).astype(int):
            R, t = robot_frame_pose(self.quat(k), 0.0)
            d = self.depth(k)
            ok = (d > 0.5) & (d < 6) & (self.conf(k) < 40)
            P = self.camera.unproject(d, R, t)[ok]
            sel = (P[:, 2] > 1.2) & (P[:, 2] < 4.0) & (np.abs(P[:, 0]) < 1.5)
            if sel.sum() > 500:
                hs.append(-np.percentile(P[sel, 1], 30))
        return float(np.median(hs)) if hs else 0.6


def level_to_ground(cam, depth, valid, R, h0, rng, max_tilt=15.0, h_tol=0.12):
    """Re-level a frame against the stereo ground plane 1-6 m ahead.

    VIO gravity on long ZED recordings drifts by several degrees (measured:
    median 4 deg, p90 31 deg on one clip) -- at 12 m that alone is a metre
    of fake elevation. A RANSAC plane through the near ground gives the true
    local 'up' and camera height. Returns (R, t) or None if no sane ground
    plane exists (e.g. a wall fills the view)."""
    R0, t0 = R, np.zeros(3)
    P = cam.unproject(depth, R0, t0)[valid & (depth > 0.5) & (depth < 8)]
    sel = (P[:, 2] > 1.0) & (P[:, 2] < 6.0) & (np.abs(P[:, 0]) < 3.0)
    Q = P[sel][::7]
    if len(Q) < 400:
        return None
    best, bc = None, 0
    for _ in range(200):
        s = Q[rng.choice(len(Q), 3, replace=False)]
        nrm = np.cross(s[1] - s[0], s[2] - s[0])
        nn = np.linalg.norm(nrm)
        if nn < 1e-9:
            continue
        nrm = nrm / nn
        nrm = nrm if nrm[1] > 0 else -nrm
        if nrm[1] < np.cos(np.radians(max_tilt)):
            continue
        c = int((np.abs((Q - s[0]) @ nrm) < 0.04).sum())
        if c > bc:
            bc, best = c, (nrm, s[0])
    if best is None or bc < 0.35 * len(Q):
        return None
    nrm, p0 = best
    inl = Q[np.abs((Q - p0) @ nrm) < 0.04]
    c0 = inl.mean(0)
    _, _, vt = np.linalg.svd(inl - c0)
    nrm = vt[-1] if vt[-1][1] > 0 else -vt[-1]
    h = float(-(c0 @ nrm))                       # camera height above the plane
    if abs(h - h0) > h_tol:
        return None
    # rotate so the plane normal becomes +y, keep the heading
    up = np.array([0.0, 1.0, 0.0])
    v = np.cross(nrm, up)
    sn, cs = np.linalg.norm(v), float(nrm @ up)
    if sn < 1e-9:
        Rl = np.eye(3)
    else:
        vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        Rl = np.eye(3) + vx + vx @ vx * ((1 - cs) / sn ** 2)
    R1 = Rl @ R0
    f = R1[:, 2] - up * R1[1, 2]                  # re-project heading
    f /= max(np.linalg.norm(f), 1e-9)
    yaw = np.arctan2(f[0], f[2])
    cy_, sy_ = np.cos(-yaw), np.sin(-yaw)
    Ry = np.array([[cy_, 0, sy_], [0, 1, 0], [-sy_, 0, cy_]])
    return Ry @ R1, np.array([0.0, h, 0.0])


def build_episode(rec_dir, grid=None, conf_max=40, dmax=15.0, step=0.25):
    """Write labels.npz + episode.json (episode format) into a recording dir.

    Frames without a trustworthy ground plane are skipped; `frame_ids` in
    labels.npz maps label rows back to recording frame numbers."""
    grid = grid or GridConfig()
    rec = Recording(rec_dir)
    h = rec.estimate_height()
    cam = rec.camera
    rng = np.random.default_rng(0)
    labels, Rs, ts, pose, ids = [], [], [], [], []
    for k in range(len(rec)):
        R, _ = robot_frame_pose(rec.quat(k), h)
        d = rec.depth(k)
        valid = rec.conf(k) < conf_max
        lv = level_to_ground(cam, d, valid, R, h, rng)
        if lv is None:
            continue
        R, t = lv
        L = depth_targets(d, cam, R, t, grid, valid=valid, dmax=dmax, step=step)
        labels.append(L)
        Rs.append(R)
        ts.append(t)
        pose.append(rec.position(k))
        ids.append(k)
    p = np.array(pose, np.float32)
    ids = np.array(ids, np.int64)
    # speed / yaw-rate from consecutive VIO poses (for logging; not a label)
    tt = rec.rows[ids, 1] / 1e9
    sp = np.r_[0.0, np.linalg.norm(np.diff(p[:, [0, 2]], axis=0), axis=1) / np.maximum(np.diff(tt), 1e-3)]
    pitch = np.degrees(np.arcsin(np.clip([R[1, 2] for R in Rs], -1, 1)))
    roll = np.degrees(np.arcsin(np.clip([-R[1, 0] for R in Rs], -1, 1)))
    np.savez_compressed(os.path.join(rec_dir, "labels.npz"), **pack_labels(labels),
                        R=np.stack(Rs).astype(np.float32), t=np.stack(ts).astype(np.float32),
                        speed=sp.astype(np.float32), yaw_rate=np.zeros(len(p), np.float32),
                        imu_pitch=pitch.astype(np.float32), imu_roll=roll.astype(np.float32),
                        pose=p, frame_ids=ids, cells=grid.cells, res=grid.res)
    cam_d = cam.to_dict()
    cam_d["mount_height"] = h
    ep_meta = dict(rec.meta)
    ep_meta.update({"camera": cam_d, "domain": "real", "frames": len(ids),
                    "label_source": f"ZED stereo depth, conf < {conf_max}, <= {dmax} m"})
    with open(os.path.join(rec_dir, "episode.json"), "w") as f:
        json.dump(ep_meta, f, indent=1)
    occ = np.mean([L["occ"][L["vis"]].mean() if L["vis"].any() else 0 for L in labels])
    print(f"  {os.path.basename(rec_dir)}: {len(ids)}/{len(rec)} frames levelled, camera height {h:.3f} m, "
          f"visible {np.mean([L['vis'].mean() for L in labels]):.2f}, obstacle rate {occ:.4f}")
    return h


def main():
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("recordings", nargs="+")
    a = ap.parse_args()
    for d in sorted(x for p in a.recordings for x in glob.glob(p)):
        if os.path.exists(os.path.join(d, "frames.csv")):
            build_episode(d)


if __name__ == "__main__":
    main()

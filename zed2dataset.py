"""Stereolabs ZED SVO recordings -> DR-JEPA training episodes.

The ZED 2i we run at the University Rover Challenge (Hanksville, UT) logs
SVO files with synchronized left image + neural depth + IMU. This is the
best real-world data source for the model: dense metric depth to ~20 m
(vs a phone's ~8 m and 160x90) and rectified wide-FOV rectilinear images
that match the rover's actual camera mount.

Each SVO becomes one (mp4, csv, npz) episode -- the same format
`generate_synth_data.py` and `real2dataset.py` emit -- so ZED, phone, and
sim data all mix in `drjepa.py preprocess` with no trainer changes. The
occupancy / visibility / elevation wedge labels come from the ZED depth
unprojected through the ZED positional-tracking pose, exactly the geometry
`real2dataset.py` uses (its `wedge_from_depth` is reused verbatim).

Unlike the phone path, the depth is cropped to the SAME square window the
model sees (with the principal point shifted to match), so labels never
cover cells outside the image the network is given -- important here
because the ZED's native FOV is much wider than the square model input.

Needs the ZED SDK + `pyzed`. On an unsupported host (e.g. Fedora) the
easy path is the bundled Docker wrapper, which carries the SDK + pyzed +
CUDA and passes args straight through (Docker + NVIDIA container toolkit
required):
    ./zed_docker.sh --svo ~/svos/*.svo --output data_zed --cam_height 0.6
    ./zed_docker.sh --selftest
Or, with the SDK installed natively:
    python zed2dataset.py --svo run1.svo run2.svo --output data_zed \
        --cam_height 0.6
    python zed2dataset.py --selftest        # geometry check, no SDK needed
Then:
    python drjepa.py preprocess --data_dir data_sim,data_zed --output packed
"""

import argparse
import glob
import math
import os

import cv2
import numpy as np

from drjepa.config import Config
from drjepa.simulator import M_PER_DEG
# reuse the verified geometry from the phone converter
from real2dataset import (WEDGE_CELLS, WEDGE_RES, quat_to_mat,
                          wedge_from_depth)

ZED_DEPTH_MIN_M = 0.3    # ZED 2i minimum trustworthy depth
ZED_DEPTH_MAX_M = 15.0   # trust depth this far (2i resolves ~20 m, but
#                          labels only need the near wedge; noise grows)


def convert_svo(svo_path, out_dir, img_size, cam_height, target_hz=5.0,
                depth_mode="NEURAL", max_depth=ZED_DEPTH_MAX_M):
    """One SVO recording -> one (mp4, csv, npz) episode in out_dir."""
    import pyzed.sl as sl

    cam = sl.Camera()
    init = sl.InitParameters()
    init.set_from_svo_file(svo_path)
    init.svo_real_time_mode = False
    init.coordinate_units = sl.UNIT.METER
    # OpenGL-style frame (x right, y up, -z look) == what wedge_from_depth
    # and the ARCore phone path assume, so the same math applies unchanged
    init.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
    init.depth_mode = getattr(sl.DEPTH_MODE, depth_mode)
    if cam.open(init) != sl.ERROR_CODE.SUCCESS:
        print(f"  {svo_path}: could not open, skipped")
        return 0
    cam.enable_positional_tracking(sl.PositionalTrackingParameters())

    info = cam.get_camera_information()
    calib = info.camera_configuration.calibration_parameters.left_cam
    fx, fy, cx, cy = calib.fx, calib.fy, calib.cx, calib.cy   # at SVO res
    fps = info.camera_configuration.fps or 30.0
    stride = max(1, round(fps / target_hz))

    base = os.path.splitext(os.path.basename(svo_path))[0]
    os.makedirs(out_dir, exist_ok=True)
    writer = cv2.VideoWriter(os.path.join(out_dir, base + ".mp4"),
                             cv2.VideoWriter_fourcc(*"mp4v"), target_hz,
                             (img_size, img_size))

    left, depth, pose = sl.Mat(), sl.Mat(), sl.Pose()
    runtime = sl.RuntimeParameters()
    frames = []           # (pose7, rgb_square, occ, vis, elev, t_ns)
    half_c = WEDGE_CELLS // 2
    while cam.grab(runtime) == sl.ERROR_CODE.SUCCESS:
        if cam.get_svo_position() % stride:
            continue
        state = cam.get_position(pose, sl.REFERENCE_FRAME.WORLD)
        if state != sl.POSITIONAL_TRACKING_STATE.OK:
            continue                                  # VIO still initializing
        cam.retrieve_image(left, sl.VIEW.LEFT)
        cam.retrieve_measure(depth, sl.MEASURE.DEPTH)
        rgb = left.get_data()[:, :, :3]               # BGRA -> BGR
        dmap = depth.get_data().astype(np.float32)    # metres, HxW

        # crop depth AND image to the same centered square, and shift the
        # principal point so the wedge labels cover only what the model sees
        H, W = dmap.shape
        s = min(H, W)
        y0, x0 = (H - s) // 2, (W - s) // 2
        dcrop = dmap[y0:y0 + s, x0:x0 + s]
        t = pose.get_translation().get()
        q = pose.get_orientation().get()              # [x, y, z, w]
        pose7 = [float(t[0]), float(t[1]), float(t[2]),
                 float(q[0]), float(q[1]), float(q[2]), float(q[3])]
        occ, vis, elev = wedge_from_depth(
            dcrop, fx, fy, cx - x0, cy - y0, pose7, cam_height,
            dmin=ZED_DEPTH_MIN_M, dmax=max_depth)

        rgb_sq = cv2.resize(rgb[y0:y0 + s, x0:x0 + s], (img_size, img_size))
        frames.append((pose7, rgb_sq, occ, vis, elev,
                       cam.get_timestamp(sl.TIME_REFERENCE.IMAGE)
                       .get_nanoseconds()))
    cam.close()
    if len(frames) < 12:
        print(f"  {base}: only {len(frames)} tracked frames, skipped")
        writer.release()
        os.remove(os.path.join(out_dir, base + ".mp4"))
        return 0

    _write_episode(frames, writer, out_dir, base, half_c)
    print(f"  {base}: {len(frames)} frames")
    return len(frames)


def _write_episode(frames, writer, out_dir, base, half_c):
    """Shared episode writer: mp4 already open, plus csv + wedge npz.

    Telemetry mirrors real2dataset.py: a virtual goal a few metres beyond
    the drive's end so goal-context features look goal-directed, heading
    and pose in the VIO planar frame, throttle/steer synthesized from the
    motion (ignored as training targets; only the wedge labels supervise).
    """
    import pandas as pd

    last_pose = frames[-1][0]
    Rl = quat_to_mat(*last_pose[3:])
    lk = Rl @ np.array([0.0, 0.0, -1.0])
    b_last = math.atan2(lk[0], -lk[2])
    goal_e = last_pose[0] + 8.0 * math.sin(b_last)
    goal_n = -last_pose[2] + 8.0 * math.cos(b_last)
    goal_lat, goal_lon = goal_n / M_PER_DEG, goal_e / M_PER_DEG

    rows, occ_bits, vis_bits, elev_q, sand_q = [], [], [], [], []
    prev_b, prev_t = None, None
    t0 = frames[0][5]
    for pose7, rgb, occ, vis, elev, t_ns in frames:
        writer.write(rgb)
        occ_bits.append(np.packbits(occ))
        vis_bits.append(np.packbits(vis))
        elev_q.append(np.clip(elev / 3.5 * 127, -127, 127).astype(np.int8))
        sand_q.append(np.zeros((WEDGE_CELLS, WEDGE_CELLS), np.uint8))

        R = quat_to_mat(*pose7[3:])
        lk = R @ np.array([0.0, 0.0, -1.0])
        b = math.atan2(lk[0], -lk[2])
        t_s = t_ns / 1e9
        yaw_rate = 0.0
        if prev_b is not None and t_s > prev_t:
            yaw_rate = math.degrees(
                (b - prev_b + math.pi) % (2 * math.pi) - math.pi) \
                / (t_s - prev_t)
        speed = 0.0
        if prev_t is not None and t_s > prev_t:
            de = pose7[0] - prev_e
            dn = -pose7[2] - prev_n
            speed = math.hypot(de, dn) / (t_s - prev_t)
        prev_b, prev_t, prev_e, prev_n = b, t_s, pose7[0], -pose7[2]

        steer = float(np.clip(yaw_rate / 45.0, -1.0, 1.0))
        thr = float(np.clip(speed / 2.0, 0.0, 1.0))
        occ_ij = np.argwhere(occ)
        trav = 1.0
        if len(occ_ij):
            d = np.hypot(occ_ij[:, 0] - half_c, occ_ij[:, 1]) * WEDGE_RES
            trav = float(np.clip(d.min() / 6.0, 0.0, 1.0))
        rows.append({
            "timestamp_ms": int((t_ns - t0) / 1e6),
            "lat": (-pose7[2]) / M_PER_DEG, "lon": pose7[0] / M_PER_DEG,
            "goal_lat": goal_lat, "goal_lon": goal_lon,
            "throttle": thr, "steer": steer,
            "exec_throttle": thr, "exec_steer": steer,
            "heading": (math.degrees(b) + 360.0) % 360.0,
            "speed": speed, "altitude": 0.0,
            "trav_score": trav, "collision": 0,
            "true_x": pose7[0], "true_z": -pose7[2],
            "true_yaw": (math.degrees(b) + 360.0) % 360.0,
        })

    writer.release()
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, base + ".csv"),
                              index=False)
    np.savez_compressed(os.path.join(out_dir, base + ".npz"),
                        occ=np.stack(occ_bits), vis=np.stack(vis_bits),
                        elev=np.stack(elev_q), sand=np.stack(sand_q),
                        cells=WEDGE_CELLS, res=WEDGE_RES)


def selftest():
    """Synthetic full-res depth through the crop + wedge path (no SDK).

    A wide 16:9 depth image of flat ground + a 1 m obstacle 3 m ahead and
    1 m right; after the centered square crop and principal-point shift,
    the obstacle must still land in the correct wedge cell and no phantom
    obstacles appear off to the sides.
    """
    W, H = 640, 360                      # wide, like a ZED frame
    fx = fy = 360.0
    cx, cy = W / 2, H / 2
    cam_h = 0.6
    v, u = np.mgrid[0:H, 0:W].astype(np.float32)
    rx = (u + 0.5 - cx) / fx
    ry = -(v + 0.5 - cy) / fy
    depth = np.full((H, W), 0.0, np.float32)
    falling = ry < -1e-4
    depth[falling] = cam_h / -ry[falling]
    ox, oy = rx * 3.0, ry * 3.0
    hit = (ox > 0.8) & (ox < 1.2) & (oy > -cam_h) & (oy < 1.0 - cam_h)
    depth[hit & ((depth > 3.0) | (depth == 0))] = 3.0
    depth[depth > ZED_DEPTH_MAX_M] = 0.0

    s = min(H, W)
    y0, x0 = (H - s) // 2, (W - s) // 2
    dcrop = depth[y0:y0 + s, x0:x0 + s]
    pose = [0.0, cam_h, 0.0, 0.0, 0.0, 0.0, 1.0]
    occ, vis, elev = wedge_from_depth(
        dcrop, fx, fy, cx - x0, cy - y0, pose, cam_h,
        dmin=ZED_DEPTH_MIN_M, dmax=ZED_DEPTH_MAX_M)

    half = WEDGE_CELLS // 2
    assert vis[half, 5:8].any(), "ground ahead not visible after crop"
    oi, oj = half + 2, 6
    assert occ[oi - 1:oi + 2, oj - 1:oj + 2].any(), "obstacle cell missed"
    assert not occ[:half - 2, :].any(), "phantom obstacle left of view"
    print(f"selftest OK: crop+wedge correct, obstacle at ~i={oi} j={oj}, "
          "principal-point shift verified, no phantoms")


def main():
    """CLI entry: convert SVO files or run the geometry self-test."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--svo", nargs="+", default=[],
                    help="SVO files (globs ok)")
    ap.add_argument("--output", default="data_zed")
    ap.add_argument("--cam_height", type=float, default=0.6,
                    help="ZED height above the ground the rover drives on "
                         "(m) -- set to your actual mast/mount height")
    ap.add_argument("--hz", type=float, default=5.0,
                    help="target sampling rate (SVO is subsampled to this)")
    ap.add_argument("--depth_mode", default="NEURAL",
                    choices=["NEURAL", "ULTRA", "QUALITY", "PERFORMANCE"])
    ap.add_argument("--max_depth", type=float, default=ZED_DEPTH_MAX_M)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        selftest()
        return
    files = []
    for pat in args.svo:
        files += sorted(glob.glob(pat))
    files = [f for f in files if f.lower().endswith(".svo")
             or f.lower().endswith(".svo2")]
    if not files:
        raise SystemExit("no .svo files matched --svo")
    img_size = Config().model.img_size
    total = 0
    for f in files:
        total += convert_svo(f, args.output, img_size, args.cam_height,
                             args.hz, args.depth_mode, args.max_depth)
    print(f"wrote {total} frames from {len(files)} SVO(s) -> {args.output}")


if __name__ == "__main__":
    main()

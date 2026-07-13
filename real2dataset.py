"""Phone recordings -> DR-JEPA training episodes.

Converts sessions captured by the Android app's Record mode
(collect/rec_*/ with frames/, meta.jsonl, session.json) into the exact
(mp4, csv, npz) episode triplets that `drjepa.py preprocess` consumes, so
real-world data mixes with simulator episodes with ZERO trainer changes:

  * DEPTH IS ESTIMATED OFFLINE with a Depth Anything model on the recorded
    RGB -- the app no longer saves ARCore depth, which was motion-stereo
    with no ToF and far too noisy on the ground plane at range (it labelled
    flat lawn as ~80% obstacle: grazing-angle depth error faked +-1 m of
    height, tripping the obstacle threshold everywhere). A monocular metric
    model yields smooth, geometrically consistent depth instead.
  * occ / vis / elev wedge labels are then built by unprojecting that depth
    through the recorded camera poses -- the same geometry the simulator's
    wedge_ground_truth computes analytically.
  * sand is all-zero (no real label; lawns/forests genuinely have none --
    revisit for desert sessions), and the tip-hazard channel needs no
    label here because the trainer derives it from the elevation wedge.
  * there are no gt_* grids, so these episodes contribute nothing to the
    map completer (SeqDataset already handles that path).
  * telemetry rows synthesize a virtual goal ahead of the walk so the
    goal-context features look like goal-directed driving, and motion
    conditioning uses VIO speed + a yaw-rate steer proxy.

Monocular depth is only APPROXIMATELY metric; --depth_scale applies a
global multiplier if a systematic offset is measured. Absolute scale
mostly shifts distances/elevations -- the obstacle test is per-cell
relative, so smooth depth already fixes the false-obstacle problem.

Usage (needs `pip install transformers` for the depth model):
    python real2dataset.py --sessions collect/rec_* --output data_real
    python real2dataset.py --selftest        # geometry check, no model
Then:
    python drjepa.py preprocess --data_dir data_sim,data_real --output packed
"""

import argparse
import glob
import json
import math
import os

import cv2
import numpy as np

from drjepa.config import Config, fnorm_from_intrinsics
from drjepa.simulator import M_PER_DEG

WEDGE_CELLS = 48        # keep identical to generate_synth_data.py
WEDGE_RES = 0.5
DEPTH_MIN_M = 0.15      # trust estimated depth in this range
DEPTH_MAX_M = 20.0      # (a monocular metric model sees much farther than
#                         the phone's old ~8 m ARCore depth)
OBST_LO = 0.25          # above-ground band that counts as an obstacle:
OBST_HI = 2.5           # matches the sim's drive-over rule at the bottom,
#                         ignores overhanging canopy at the top
MIN_PTS_VIS = 3         # depth returns before a cell counts as observed
MIN_PTS_OCC = 4         # in-band returns before a cell counts as occupied

# Depth Anything V2, metric outdoor. Configurable via --depth_model; must
# be a metric (not relative) checkpoint so the output is in metres.
DEFAULT_DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf"

# upright <-> sensor rotation pairs (cv2 codes), for running the depth
# model on an upright image then mapping depth back to the sensor frame
_FWD_ROT = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180,
            270: cv2.ROTATE_90_COUNTERCLOCKWISE}
_INV_ROT = {90: cv2.ROTATE_90_COUNTERCLOCKWISE, 180: cv2.ROTATE_180,
            270: cv2.ROTATE_90_CLOCKWISE}


class DepthEstimator:
    """Lazy Depth Anything wrapper: BGR image -> metric depth (m), HxW.

    Runs the model on an UPRIGHT image (what it was trained on) and rotates
    the depth back to the sensor frame so it lines up with the recorded
    sensor-orientation intrinsics + pose.
    """

    def __init__(self, model_name=DEFAULT_DEPTH_MODEL, device="cuda"):
        from transformers import pipeline
        self.pipe = pipeline("depth-estimation", model=model_name,
                             device=0 if device == "cuda" else -1)

    def __call__(self, bgr_sensor, rot):
        from PIL import Image
        up = cv2.rotate(bgr_sensor, _FWD_ROT[rot]) if rot else bgr_sensor
        rgb = cv2.cvtColor(up, cv2.COLOR_BGR2RGB)
        pred = self.pipe(Image.fromarray(rgb))["predicted_depth"]
        d = pred.squeeze().cpu().numpy().astype(np.float32)
        d = cv2.resize(d, (up.shape[1], up.shape[0]))     # back to input res
        if rot:
            d = cv2.rotate(d, _INV_ROT[rot])              # -> sensor frame
        return d


def quat_to_mat(qx, qy, qz, qw):
    """Unit quaternion -> 3x3 rotation matrix (ARCore convention)."""
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw),
         2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz),
         2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw),
         1 - 2 * (qx * qx + qy * qy)]])


def look_pitch_deg(pose7):
    """Camera pitch (deg, + = looking up) from a pose quaternion.

    Matches RoverSim.camera_angles' convention so sim and real frames
    describe their camera the same way. The look direction is -z in the
    ARCore/ZED right-handed y-up frame, and the y component of that unit
    vector is the sine of its elevation above horizontal.
    """
    R = quat_to_mat(*pose7[3:])
    look = R @ np.array([0.0, 0.0, -1.0])
    return math.degrees(math.asin(float(np.clip(look[1], -1.0, 1.0))))


def unproject(depth_m, fx, fy, cx, cy, dmin=DEPTH_MIN_M, dmax=DEPTH_MAX_M):
    """Depth image -> camera-frame points (ARCore: x right, y up, -z look).

    Depth is distance along the optical axis; image v grows downward.
    NaN/inf (invalid returns) fail the finite-range test and drop out.
    Returns (N, 3) for valid pixels plus the validity mask.
    """
    dh, dw = depth_m.shape
    v, u = np.mgrid[0:dh, 0:dw].astype(np.float32)
    ok = (depth_m > dmin) & (depth_m < dmax)
    d = depth_m[ok]
    x = (u[ok] + 0.5 - cx) / fx * d
    y = -(v[ok] + 0.5 - cy) / fy * d
    z = -d
    return np.stack([x, y, z], axis=1), ok


def wedge_from_depth(depth_m, fx, fy, cx, cy, pose7, cam_height,
                     cells=WEDGE_CELLS, res=WEDGE_RES,
                     dmin=DEPTH_MIN_M, dmax=DEPTH_MAX_M):
    """One frame's depth -> (occ bool, vis bool, elev f32) wedge labels.

    The wedge matches wedge_ground_truth's frame: [i, j] covers
    x_right = (i+0.5)*res - cells*res/2, z_forward = (j+0.5)*res, and
    elevation is relative to the ground under the rover (camera minus
    cam_height). Cell ground = 20th-percentile point height (robust to
    obstacle points above it); occupied = enough returns in the
    [OBST_LO, OBST_HI] band above that ground. dmin/dmax bound the trusted
    depth range (a phone tops out ~8 m; a ZED sees much farther).
    """
    pts_cam, _ = unproject(depth_m, fx, fy, cx, cy, dmin, dmax)
    occ = np.zeros((cells, cells), bool)
    vis = np.zeros((cells, cells), bool)
    elev = np.zeros((cells, cells), np.float32)
    if len(pts_cam) == 0:
        return occ, vis, elev

    R = quat_to_mat(*pose7[3:])
    t = np.asarray(pose7[:3], np.float32)
    pts = pts_cam @ R.T + t

    # rover frame: planar bearing of the camera look direction
    look = R @ np.array([0.0, 0.0, -1.0])
    b = math.atan2(look[0], -look[2])          # planar east/north bearing
    e = pts[:, 0] - t[0]                       # planar east offset
    n = -(pts[:, 2] - t[2])                    # planar north offset
    z_fwd = e * math.sin(b) + n * math.cos(b)
    x_rgt = e * math.cos(b) - n * math.sin(b)
    h = pts[:, 1] - (t[1] - cam_height)        # height above rover ground

    half = cells * res / 2.0
    i = np.floor((x_rgt + half) / res).astype(np.int32)
    j = np.floor(z_fwd / res).astype(np.int32)
    inb = (i >= 0) & (i < cells) & (j >= 0) & (j < cells)
    if not inb.any():
        return occ, vis, elev
    cell = i[inb] * cells + j[inb]
    h = h[inb]

    order = np.lexsort((h, cell))
    cell, h = cell[order], h[order]
    starts = np.flatnonzero(np.r_[True, cell[1:] != cell[:-1]])
    ends = np.r_[starts[1:], len(cell)]
    for s, epd in zip(starts, ends):
        npt = epd - s
        if npt < MIN_PTS_VIS:
            continue
        c = cell[s]
        ci, cj = divmod(int(c), cells)
        ground = h[s + int(0.2 * (npt - 1))]   # sorted within cell
        vis[ci, cj] = True
        elev[ci, cj] = ground
        band = (h[s:epd] > ground + OBST_LO) & (h[s:epd] < ground + OBST_HI)
        if band.sum() >= MIN_PTS_OCC:
            occ[ci, cj] = True

    # depth noise at range fakes terrain grades; the trainer differentiates
    # elev into the tip-hazard label, so smooth it (visibility-normalized)
    visf = vis.astype(np.float32)
    num = cv2.boxFilter(elev * visf, -1, (3, 3), normalize=False)
    den = cv2.boxFilter(visf, -1, (3, 3), normalize=False)
    elev = np.where(vis, num / np.maximum(den, 1e-6), 0.0).astype(np.float32)
    return occ, vis, elev


def convert_session(sess_dir, out_dir, img_size, estimator,
                    cam_height_default=1.4, depth_scale=1.0):
    """One rec_* session -> one (mp4, csv, npz) episode in out_dir."""
    import pandas as pd

    with open(os.path.join(sess_dir, "session.json")) as f:
        sess = json.load(f)
    cam_height = float(sess.get("cam_height_m", cam_height_default))

    metas = []
    with open(os.path.join(sess_dir, "meta.jsonl")) as f:
        for line in f:
            line = line.strip()
            if line:
                metas.append(json.loads(line))
    if len(metas) < 12:
        print(f"  {sess_dir}: only {len(metas)} frames, skipped")
        return 0

    base = os.path.basename(os.path.normpath(sess_dir))
    os.makedirs(out_dir, exist_ok=True)
    writer = cv2.VideoWriter(os.path.join(out_dir, base + ".mp4"),
                             cv2.VideoWriter_fourcc(*"mp4v"), 5.0,
                             (img_size, img_size))

    # virtual goal: a few metres beyond the walk's end, so goal-context
    # features (dist shrinking, bearing ~ahead) look like driving at it
    lastm = metas[-1]
    R_last = quat_to_mat(*[lastm["pose"][k] for k in range(3, 7)])
    lk = R_last @ np.array([0.0, 0.0, -1.0])
    b_last = math.atan2(lk[0], -lk[2])
    goal_e = lastm["pose"][0] + 8.0 * math.sin(b_last)
    goal_n = -lastm["pose"][2] + 8.0 * math.cos(b_last)
    goal_lat = goal_n / M_PER_DEG
    goal_lon = goal_e / M_PER_DEG

    rows = []
    occ_bits, vis_bits, elev_q, sand_q = [], [], [], []
    prev_b, prev_t = None, None
    half_c = WEDGE_CELLS // 2
    for m in metas:
        name = "%05d" % m["i"]
        img = cv2.imread(os.path.join(sess_dir, "frames", name + ".jpg"))
        if img is None:
            continue
        rot = int(m.get("rot", 0))

        # estimate depth from the RGB (sensor frame, matching the recorded
        # sensor intrinsics + pose); scale corrects any systematic offset
        depth_m = estimator(img, rot) * depth_scale
        occ, vis, elev = wedge_from_depth(
            depth_m, sess["fx"], sess["fy"], sess["cx"], sess["cy"],
            m["pose"], cam_height, dmin=DEPTH_MIN_M, dmax=DEPTH_MAX_M)
        occ_bits.append(np.packbits(occ))
        vis_bits.append(np.packbits(vis))
        elev_q.append(np.clip(elev / 3.5 * 127, -127, 127).astype(np.int8))
        sand_q.append(np.zeros((WEDGE_CELLS, WEDGE_CELLS), np.uint8))

        # upright + square-crop + resize, matching the on-device pilot feed
        if rot:
            img = cv2.rotate(img, _FWD_ROT[rot])
        hgt, wid = img.shape[:2]
        crop = min(hgt, wid)
        y0, x0 = (hgt - crop) // 2, (wid - crop) // 2
        writer.write(cv2.resize(img[y0:y0 + crop, x0:x0 + crop],
                                (img_size, img_size)))

        # the camera the model is conditioned on is the one it actually sees:
        # this upright square crop resized to img_size, NOT the raw sensor.
        # A 90/270 upright rotation swaps which sensor axis is horizontal.
        f_horiz = sess["fy"] if rot in (90, 270) else sess["fx"]
        cam_fnorm = fnorm_from_intrinsics(f_horiz, crop)

        # telemetry row in the VIO planar frame (consistent lat/lon/heading)
        Rm = quat_to_mat(*[m["pose"][k] for k in range(3, 7)])
        lk = Rm @ np.array([0.0, 0.0, -1.0])
        b = math.atan2(lk[0], -lk[2])
        t_s = m["t_ns"] / 1e9
        yaw_rate = 0.0
        if prev_b is not None and t_s > prev_t:
            yaw_rate = math.degrees(
                (b - prev_b + math.pi) % (2 * math.pi) - math.pi) \
                / (t_s - prev_t)
        prev_b, prev_t = b, t_s
        steer = float(np.clip(yaw_rate / 45.0, -1.0, 1.0))
        thr = float(np.clip(m["speed"] / 2.0, 0.0, 1.0))
        occ_ij = np.argwhere(occ)
        trav = 1.0
        if len(occ_ij):
            d = np.hypot(occ_ij[:, 0] - half_c, occ_ij[:, 1]) * WEDGE_RES
            trav = float(np.clip(d.min() / 6.0, 0.0, 1.0))
        e_pos, n_pos = m["pose"][0], -m["pose"][2]
        rows.append({
            "timestamp_ms": int((m["t_ns"] - metas[0]["t_ns"]) / 1e6),
            "lat": n_pos / M_PER_DEG, "lon": e_pos / M_PER_DEG,
            "goal_lat": goal_lat, "goal_lon": goal_lon,
            "throttle": thr, "steer": steer,
            "exec_throttle": thr, "exec_steer": steer,
            "heading": (math.degrees(b) + 360.0) % 360.0,
            "speed": m["speed"], "altitude": 0.0,
            "trav_score": trav, "collision": 0,
            "cam_fnorm": cam_fnorm, "cam_height": cam_height,
            "cam_pitch": look_pitch_deg(m["pose"]),
            "true_x": e_pos, "true_z": n_pos,
            "true_yaw": (math.degrees(b) + 360.0) % 360.0,
        })

    writer.release()
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, base + ".csv"),
                              index=False)
    np.savez_compressed(os.path.join(out_dir, base + ".npz"),
                        occ=np.stack(occ_bits), vis=np.stack(vis_bits),
                        elev=np.stack(elev_q), sand=np.stack(sand_q),
                        cells=WEDGE_CELLS, res=WEDGE_RES)
    print(f"  {base}: {len(rows)} frames")
    return len(rows)


def selftest():
    """Synthetic scene through the full label path; asserts geometry.

    Camera 1.4 m above a flat ground plane, looking level along -z with
    a 1 m-tall obstacle 3 m dead ahead and 1 m to the right: the wedge
    must mark ground cells visible at elevation ~0 and the obstacle cell
    occupied at the right (i > center) location.
    """
    dw, dh = 160, 120
    fx = fy = 120.0
    cx, cy = dw / 2, dh / 2
    cam_h = 1.4
    v, u = np.mgrid[0:dh, 0:dw].astype(np.float32)
    rx = (u + 0.5 - cx) / fx            # camera-frame ray dirs (per unit -z)
    ry = -(v + 0.5 - cy) / fy
    # ground plane y = -cam_h (camera at origin): depth where ray hits it
    depth = np.full((dh, dw), np.inf, np.float32)
    falling = ry < -1e-4
    depth[falling] = cam_h / -ry[falling]
    # obstacle: vertical slab z = -3, x in [0.8, 1.2], y in [-cam_h, 1-cam_h]
    ox = rx * 3.0
    oy = ry * 3.0
    hit = (ox > 0.8) & (ox < 1.2) & (oy > -cam_h) & (oy < 1.0 - cam_h)
    depth[hit & (depth > 3.0)] = 3.0
    depth[~np.isfinite(depth)] = 0.0

    pose = [0.0, cam_h, 0.0, 0.0, 0.0, 0.0, 1.0]   # identity: look -z
    occ, vis, elev = wedge_from_depth(depth, fx, fy, cx, cy, pose, cam_h,
                                      dmax=8.2)

    half = WEDGE_CELLS // 2
    # ground ahead: visible, flat. (With a LEVEL camera this synthetic
    # 53-degree vertical FOV first sees ground ~2.8 m out, so probe at
    # 3.5-4 m; in real captures the phone is pitched slightly down.)
    assert vis[half, 7], "ground ahead not visible"
    assert abs(elev[half, 7]) < 0.15, f"ground elev {elev[half, 7]}"
    # obstacle at x_right ~ +1.0, z ~ 3.0 -> i = half + 2, j = 6
    oi, oj = half + 2, 6
    assert occ[oi - 1:oi + 2, oj - 1:oj + 2].any(), "obstacle not marked"
    # nothing occupied on the empty left side
    assert not occ[:half - 2, :].any(), "phantom obstacles on the left"
    # far cells beyond depth range are invisible
    assert not vis[:, 20:].any(), "cells beyond depth range marked visible"
    print("selftest OK: ground visible/flat, obstacle cell occupied at "
          f"i={oi} j={oj}, no phantoms, range mask correct")


def main():
    """CLI entry: convert sessions or run the geometry self-test."""
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sessions", nargs="+", default=[],
                    help="rec_* session dirs (globs ok)")
    ap.add_argument("--output", default="data_real")
    ap.add_argument("--cam_height", type=float, default=1.4,
                    help="fallback camera height (m) if a session omits it")
    ap.add_argument("--depth_model", default=DEFAULT_DEPTH_MODEL,
                    help="HF depth-estimation model (must be METRIC)")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--depth_scale", type=float, default=1.0,
                    help="global multiplier on estimated depth to correct a "
                         "systematic metric offset")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        selftest()
        return
    dirs = []
    for pat in args.sessions:
        dirs += sorted(glob.glob(pat))
    dirs = [d for d in dirs if os.path.isdir(d)]
    if not dirs:
        raise SystemExit("no session dirs matched --sessions")
    print(f"loading depth model {args.depth_model} ...")
    estimator = DepthEstimator(args.depth_model, args.device)
    img_size = Config().model.img_size
    total = 0
    for d in dirs:
        total += convert_session(d, args.output, img_size, estimator,
                                 args.cam_height, args.depth_scale)
    print(f"wrote {total} frames from {len(dirs)} sessions -> {args.output}")


if __name__ == "__main__":
    main()

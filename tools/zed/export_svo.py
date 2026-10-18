"""Export ZED SVO recordings to plain files (no SDK needed downstream).

Runs inside the ZED SDK container (see run.sh). For every SVO it writes a
"recording" directory that the rest of DR-JEPA reads with numpy + OpenCV:

    <out>/<name>/
        meta.json       intrinsics (left, rectified), image size, serial,
                        depth mode, export rate
        frames.csv      i, t_ns, tx ty tz, qx qy qz qw   (camera pose in a
                        gravity-aligned world frame, RIGHT_HANDED_Y_UP:
                        x right, y up, -z look -- the OpenGL convention)
        rgb/NNNNN.jpg   rectified left image
        depth/NNNNN.png uint16 depth in millimetres along the optical axis
                        (0 = no return)
        conf/NNNNN.png  uint8 ZED depth confidence (0 = best, 100 = worst)

Only frames where positional tracking reports OK are written, so every
exported frame has a trustworthy gravity-aligned pose.

    ./tools/zed/run.sh --svo zed_data/*.svo2 --output recordings/zed --hz 1
"""

import argparse
import glob
import json
import os

import cv2
import numpy as np


def export_svo(svo_path, out_root, hz, depth_mode, max_frames=0):
    """One SVO -> one recording directory. Returns frames written."""
    import pyzed.sl as sl

    cam = sl.Camera()
    init = sl.InitParameters()
    init.set_from_svo_file(svo_path)
    init.svo_real_time_mode = False
    init.coordinate_units = sl.UNIT.METER
    init.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
    init.depth_mode = getattr(sl.DEPTH_MODE, depth_mode)
    init.depth_maximum_distance = 40.0
    if cam.open(init) != sl.ERROR_CODE.SUCCESS:
        print(f"  {svo_path}: could not open, skipped")
        return 0
    tracking = sl.PositionalTrackingParameters()
    tracking.set_gravity_as_origin = True
    cam.enable_positional_tracking(tracking)

    info = cam.get_camera_information()
    conf_ = info.camera_configuration
    calib = conf_.calibration_parameters.left_cam
    fps = conf_.fps or 30.0
    stride = max(1, round(fps / hz))

    name = os.path.splitext(os.path.basename(svo_path))[0].replace(" ", "_")
    out = os.path.join(out_root, name)
    for sub in ("rgb", "depth", "conf"):
        os.makedirs(os.path.join(out, sub), exist_ok=True)

    left, depth, conf, pose = sl.Mat(), sl.Mat(), sl.Mat(), sl.Pose()
    runtime = sl.RuntimeParameters()
    runtime.confidence_threshold = 100          # keep everything; we store conf
    rows = []
    while cam.grab(runtime) == sl.ERROR_CODE.SUCCESS:
        if cam.get_svo_position() % stride:
            continue
        state = cam.get_position(pose, sl.REFERENCE_FRAME.WORLD)
        if state != sl.POSITIONAL_TRACKING_STATE.OK:
            continue
        cam.retrieve_image(left, sl.VIEW.LEFT)
        cam.retrieve_measure(depth, sl.MEASURE.DEPTH)
        cam.retrieve_measure(conf, sl.MEASURE.CONFIDENCE)
        k = len(rows)
        rgb = left.get_data()[:, :, :3]
        d = depth.get_data().astype(np.float32)
        d = np.where(np.isfinite(d) & (d > 0), d, 0.0)
        c = conf.get_data().astype(np.float32)
        c = np.where(np.isfinite(c), c, 100.0)
        cv2.imwrite(os.path.join(out, "rgb", f"{k:05d}.jpg"), rgb,
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
        cv2.imwrite(os.path.join(out, "depth", f"{k:05d}.png"),
                    np.clip(d * 1000.0, 0, 65535).astype(np.uint16))
        cv2.imwrite(os.path.join(out, "conf", f"{k:05d}.png"),
                    np.clip(c, 0, 255).astype(np.uint8))
        t = pose.get_translation().get()
        q = pose.get_orientation().get()           # x, y, z, w
        rows.append([k, cam.get_timestamp(sl.TIME_REFERENCE.IMAGE)
                     .get_nanoseconds(), *map(float, t), *map(float, q)])
        if max_frames and len(rows) >= max_frames:
            break
    cam.close()

    with open(os.path.join(out, "frames.csv"), "w") as f:
        f.write("i,t_ns,tx,ty,tz,qx,qy,qz,qw\n")
        for r in rows:
            f.write(",".join(str(v) for v in r) + "\n")
    meta = {"source": os.path.basename(svo_path), "sensor": "ZED 2i",
            "serial": int(info.serial_number),
            "width": int(conf_.resolution.width),
            "height": int(conf_.resolution.height),
            "fx": float(calib.fx), "fy": float(calib.fy),
            "cx": float(calib.cx), "cy": float(calib.cy),
            "depth_mode": depth_mode, "export_hz": hz, "svo_fps": float(fps),
            "frames": len(rows),
            "pose": "camera-to-world, gravity-aligned RIGHT_HANDED_Y_UP"}
    with open(os.path.join(out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  {name}: {len(rows)} frames @ {hz} Hz "
          f"({meta['width']}x{meta['height']}, fx {calib.fx:.1f})")
    return len(rows)


def main():
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--svo", nargs="+", required=True, help="SVO files (globs ok)")
    ap.add_argument("--output", required=True)
    ap.add_argument("--hz", type=float, default=1.0, help="export rate")
    ap.add_argument("--depth_mode", default="NEURAL",
                    choices=["NEURAL", "NEURAL_PLUS", "ULTRA", "QUALITY",
                             "PERFORMANCE"])
    ap.add_argument("--max_frames", type=int, default=0)
    args = ap.parse_args()
    files = sorted(f for pat in args.svo for f in glob.glob(pat)
                   if f.lower().endswith((".svo", ".svo2")))
    if not files:
        raise SystemExit("no .svo/.svo2 files matched --svo")
    total = sum(export_svo(f, args.output, args.hz, args.depth_mode,
                           args.max_frames) for f in files)
    print(f"exported {total} frames from {len(files)} recording(s) -> {args.output}")


if __name__ == "__main__":
    main()

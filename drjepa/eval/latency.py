"""Isolated latency: perception (PyTorch GPU, ONNX Runtime CPU) and the full
Navigator.step, one frame at a time. Run on an otherwise idle machine.

    python -m drjepa bench-latency --checkpoint models/geobev.pt --onnx models/geobev.onnx \
        --out docs/results/latency/geobev.json

Frames come from a labelled real recording when given (--recording), else a
synthetic 1280 x 720 image through a ZED-like camera.
"""

import argparse
import json
import time

import numpy as np

from ..camera import Camera
from ..nav.navigator import Navigator


def timed(fn, n, warmup=10, sync=None):
    """Median and p95 wall time (ms) of fn(i)."""
    for i in range(warmup):
        fn(i)
    ts = []
    for i in range(n):
        t0 = time.perf_counter()
        fn(i)
        if sync:
            sync()
        ts.append(1000 * (time.perf_counter() - t0))
    return {"median_ms": float(np.median(ts)), "p95_ms": float(np.percentile(ts, 95))}


def nav_step(nav, frames, R, t):
    def f(i):
        # drive north 0.1 m per step toward a goal that stays 60 m ahead
        nav.step(frames[i % len(frames)], {"x": 0.0, "z": 0.1 * i, "heading": 0.0, "speed": 1.0,
                                           "goal_x": 0.0, "goal_z": 0.1 * i + 60.0}, R=R, t=t)
    return f


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default=None, help=".pt checkpoint (GPU / PyTorch timings)")
    ap.add_argument("--onnx", default=None, help="exported .onnx (CPU / ONNX Runtime timings)")
    ap.add_argument("--recording", default=None, help="labelled recording to draw frames from")
    ap.add_argument("--threads", type=int, nargs="+", default=[0, 4, 1], help="ORT threads (0 = default)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    if a.recording:
        from .perception import Episode
        ep = Episode(a.recording)
        frames = [ep.rgb(k) for k in range(0, min(len(ep), 60), 2)]
        camera, R, t = ep.camera, ep.R[0], ep.t[0]
    else:
        rng = np.random.default_rng(0)
        frames = [rng.integers(0, 255, (720, 1280, 3), dtype=np.uint8) for _ in range(8)]
        camera = Camera.from_fov(1280, 720, 101.5, mount_height=0.40)
        R, t = camera.pose(0.0, 0.0)
    res = {"image": list(frames[0].shape)}

    if a.checkpoint:
        import torch

        from ..perception.infer import Perceiver
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        sync = torch.cuda.synchronize if dev == "cuda" else None
        p = Perceiver(a.checkpoint, device=dev)
        res["torch_device"] = torch.cuda.get_device_name() if dev == "cuda" else "cpu"
        img, K = p.prepare(frames[0], camera, R, t)
        x = (torch.from_numpy(img).to(dev).permute(2, 0, 1)[None].float() / 255.0,
             torch.from_numpy(K).to(dev)[None],
             torch.as_tensor(np.asarray(R), dtype=torch.float32, device=dev)[None],
             torch.as_tensor(np.asarray(t), dtype=torch.float32, device=dev)[None])
        with torch.no_grad():
            res["torch_network"] = timed(lambda i: p.model(*x), 300, sync=sync)
        res["torch_call"] = timed(lambda i: p(frames[i % len(frames)], camera, R, t), 200)
        nav = Navigator(p, camera=camera)
        step = nav_step(nav, frames, R, t)
        res["torch_nav_step"] = timed(step, 300)
        nav.timing = {k: 0.0 for k in nav.timing}          # per-stage means (A* amortized)
        for i in range(300):
            step(400 + i)
        res["torch_nav_stage_mean_ms"] = {k: 1000 * v / nav.timing["steps"]
                                          for k, v in nav.timing.items() if k != "steps"}

    if a.onnx:
        from ..perception.onnx_runtime import OnnxPerceiver
        for th in a.threads:
            p = OnnxPerceiver(a.onnx, threads=th or None)
            key = f"onnx_cpu_{th}t" if th else "onnx_cpu_default_threads"
            res[key + "_call"] = timed(lambda i, p=p: p(frames[i % len(frames)], camera, R, t),
                                       40 if th != 1 else 15, 5)
            if th == 4:
                res[key + "_nav_step"] = timed(nav_step(Navigator(p, camera=camera), frames, R, t), 120)

    print(json.dumps(res, indent=1))
    if a.out:
        with open(a.out, "w") as f:
            json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()

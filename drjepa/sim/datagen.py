"""Synthetic perception dataset generation (parallel, one GL context per worker).

Each episode: a randomized world + camera rig, driven by the privileged
expert (with injected steering perturbations so the views include the
skewed approaches a learned pilot produces). Every `stride`-th control
step is rendered and labelled. Output per episode `ep_XXXXX/`:

    rgb/NNNN.jpg     native-resolution camera image
    depth.npy        (N, H/4, W/4) float16 depth (m, 0 = sky)
    labels.npz       per-frame robot-frame grid targets (see drjepa.labels)
                     + R (N,3,3), t (N,3) camera->robot pose
                     + speed, yaw_rate, imu pitch/roll
    episode.json     camera (intrinsics + mount), scenario, seed, domain

    python -m drjepa.sim.datagen --out data/sim --episodes 1200 --workers 20
"""

import argparse
import json
import multiprocessing as mp
import os
import time

import numpy as np

from ..config import GridConfig, RobotProfile, SimConfig
from ..labels import sim_targets

_ARGS = None


def _init(args):
    global _ARGS
    _ARGS = args
    import cv2
    cv2.setNumThreads(1)


def pack_labels(frames):
    """List of per-frame label dicts -> compact arrays."""
    out = {}
    for k in ("vis", "occ"):
        out[k] = np.packbits(np.stack([f[k] for f in frames]).astype(bool), axis=-1)
    for k in ("obs_h", "elev", "grade"):
        out[k] = np.stack([f[k] for f in frames]).astype(np.float16)
    out["sand"] = (np.clip(np.stack([f["sand"] for f in frames]), 0, 1) * 255).astype(np.uint8)
    return out


def episode(ep):
    """Generate one episode; returns a small stats dict."""
    import cv2
    from .expert import ArcPlanner, NoiseInjector
    from .rover import RoverSim

    a = _ARGS
    seed = a.seed + ep
    grid = GridConfig()
    sim = RoverSim(SimConfig(max_frames=a.max_frames), RobotProfile(), seed=seed,
                   augment=a.augment and (ep % 2 == 1))
    rng = np.random.default_rng(seed + 991)
    expert, noise = ArcPlanner(sim, rng), NoiseInjector(rng, p_start=0.02)
    d = os.path.join(a.out, f"ep_{ep:05d}")
    os.makedirs(os.path.join(d, "rgb"), exist_ok=True)
    labels, depths, Rs, ts, extra = [], [], [], [], []
    offset = int(rng.integers(0, a.stride))
    status = "timeout"
    while True:
        if (sim.frame - offset) % a.stride == 0 and sim.frame >= offset:
            rgb, depth, cls, (R, t) = sim.render(gt=True)
            L = sim_targets(sim, depth, cls, R, t, grid)
            k = len(labels)
            cv2.imwrite(os.path.join(d, "rgb", f"{k:04d}.jpg"), rgb[:, :, ::-1],
                        [cv2.IMWRITE_JPEG_QUALITY, 93])
            depths.append(cv2.resize(depth, (depth.shape[1] // 4, depth.shape[0] // 4),
                                     interpolation=cv2.INTER_NEAREST).astype(np.float16))
            labels.append(L)
            Rs.append(R)
            ts.append(t)
            m = sim.meas
            extra.append((m["speed"], sim.yaw_rate, m["pitch"], m["roll"], sim.x, sim.z, sim.yaw))
            sim._prev_frame = None      # frames are not consecutive: no blur carry-over
        thr, st = expert.plan()
        thr, st = noise.apply(thr, st, sim.clearance())
        info = sim.step(thr, st)
        if info["tipped"]:
            status = "tipped"
            break
        if info["reached"]:
            status = "reached"
            break
        if info["timeout"] or len(labels) >= a.max_saved:
            break
    sim.close()
    if not labels:
        return {"ep": ep, "frames": 0}
    np.save(os.path.join(d, "depth.npy"), np.stack(depths))
    ex = np.array(extra, np.float32)
    np.savez_compressed(os.path.join(d, "labels.npz"), **pack_labels(labels),
                        R=np.stack(Rs).astype(np.float32), t=np.stack(ts).astype(np.float32),
                        speed=ex[:, 0], yaw_rate=ex[:, 1], imu_pitch=ex[:, 2], imu_roll=ex[:, 3],
                        pose=ex[:, 4:7], cells=grid.cells, res=grid.res)
    with open(os.path.join(d, "episode.json"), "w") as f:
        json.dump({"camera": sim.camera.to_dict(), "scenario": sim.scenario,
                   "spawn": sim.spawn_mode, "seed": seed, "status": status,
                   "augment": bool(sim.look["sat_scale"] != 1.0 or sim.look["hue_shift"] != 0.0),
                   "frames": len(labels), "domain": "sim"}, f, indent=1)
    return {"ep": ep, "frames": len(labels), "status": status, "scenario": sim.scenario}


def main():
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--episodes", type=int, default=1200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=max(1, os.cpu_count() - 4))
    ap.add_argument("--stride", type=int, default=4, help="render every Nth control step")
    ap.add_argument("--max_saved", type=int, default=220, help="cap saved frames per episode")
    ap.add_argument("--max_frames", type=int, default=1200)
    ap.add_argument("--augment", action="store_true",
                    help="colour/photometric augmentation on half the episodes")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    stats = []
    ctx = mp.get_context("spawn")       # each worker owns its own GL context
    with ctx.Pool(a.workers, initializer=_init, initargs=(a,), maxtasksperchild=40) as pool:
        for i, s in enumerate(pool.imap_unordered(episode, range(a.episodes))):
            stats.append(s)
            if (i + 1) % 25 == 0 or i + 1 == a.episodes:
                n = sum(x["frames"] for x in stats)
                print(f"  {i + 1}/{a.episodes} episodes, {n} frames, "
                      f"{n / (time.time() - t0):.0f} frames/s", flush=True)
    with open(os.path.join(a.out, "summary.json"), "w") as f:
        json.dump(stats, f)


if __name__ == "__main__":
    main()

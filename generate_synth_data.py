"""Generate synthetic rover driving episodes (video + telemetry CSV).

Each episode is a domain-randomized 3D world. By default the arc-sampling
expert drives (with occasional injected perturbations); with --dagger the
*trained model* drives while the expert only supplies labels, so the dataset
covers exactly the states the learned policy visits (DAgger).

The CSV logs what a real rover would log: noisy GPS, compass and odometry,
plus the expert's clean action labels and the actually-executed commands.

Usage:
    python generate_synth_data.py --episodes 300 --output data_synth
    python generate_synth_data.py --episodes 200 --output data_dagger \
        --dagger runs/best.pth --seed 5000
"""

import argparse
import os

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import multiprocessing as mp

from drjepa.simulator import (RoverSim, SimConfig, wedge_ground_truth,
                              episode_gt_grids)
from drjepa.expert import ArcPlanner, NoiseInjector

WEDGE_CELLS = 48
WEDGE_RES = 0.5

_ARGS = None    # populated in workers via initializer
_PILOT = None   # lazy per-worker model pilot (DAgger mode)


def _init_worker(args):
    """Pool initializer: share CLI args; single-threaded OpenCV per worker."""
    global _ARGS
    _ARGS = args
    cv2.setNumThreads(0)


def _get_pilot():
    """Lazily build one model pilot per worker (DAgger mode only)."""
    global _PILOT
    if _PILOT is None:
        from drjepa.pilot import Pilot
        _PILOT = Pilot(_ARGS.dagger, shield=False)
    return _PILOT


def generate_episode(ep_id):
    """Simulate one episode and write its video + CSV + wedge-GT npz.

    Per frame, in order: render the camera, read the noisy sensors, compute
    the wedge ground truth, query the expert for the clean action LABEL,
    pick the EXECUTED action (noise-injected expert, or the model itself in
    DAgger mode), log the row, then step the physics. Frame t therefore
    always pairs with the command chosen at frame t.
    """
    args = _ARGS
    seed = args.seed + ep_id
    sim = RoverSim(SimConfig(), scenario=args.scenario, seed=seed,
                   augment=getattr(args, "augment", False))
    rng = np.random.default_rng(seed + 777)
    expert = ArcPlanner(sim, rng)
    noise = NoiseInjector(rng)
    pilot = _get_pilot() if args.dagger else None
    if pilot:
        pilot.reset()

    prefix = "dag_" if args.dagger else ""
    base = f"{prefix}{sim.scenario}_{sim.spawn_mode}_{ep_id:05d}"
    vid_path = os.path.join(args.output, base + ".mp4")
    csv_path = os.path.join(args.output, base + ".csv")

    writer = cv2.VideoWriter(vid_path, cv2.VideoWriter_fourcc(*"mp4v"),
                             1.0 / sim.cfg.dt, (sim.cfg.img_w, sim.cfg.img_h))
    rows = []
    occ_bits, vis_bits, elev_q, sand_q = [], [], [], []
    reached = False
    tipped = False
    while True:
        frame = sim.render()
        sensors = sim.sensor_readout()
        clearance = sim.clearance()
        occ, vis, elev, sand = wedge_ground_truth(sim, WEDGE_CELLS, WEDGE_RES)
        occ_bits.append(np.packbits(occ))
        vis_bits.append(np.packbits(vis))
        # elevation quantized to int8 (:: 3.5 m / 127 per unit)
        elev_q.append(np.clip(elev / 3.5 * 127, -127, 127).astype(np.int8))
        sand_q.append((np.clip(sand, 0, 1) * 255).astype(np.uint8))
        label_thr, label_steer = expert.plan()
        if pilot:
            # DAgger: the model drives, the expert labels. A small expert
            # mixture keeps episodes progressing toward the goal.
            if rng.random() < args.expert_mix:
                exec_thr, exec_steer = label_thr, label_steer
            else:
                dist, rel = sim.goal_vector_measured()
                out = pilot.step(frame, dist, rel, sim.meas["speed"])
                exec_thr, exec_steer = out["throttle"], out["steer"]
        else:
            exec_thr, exec_steer = noise.apply(label_thr, label_steer, clearance)
        # traversability = worst of obstacle clearance and terrain margin
        # (steep grades and deep sand read as danger, same as obstacles)
        trav = float(np.clip(min(clearance / 6.0, sim.terrain_margin()),
                             0.0, 1.0))
        if sim.collided_now:
            trav = 0.0

        writer.write(frame)
        rows.append({
            "timestamp_ms": int(sim.frame * sim.cfg.dt * 1000),
            "lat": sensors["lat"], "lon": sensors["lon"],
            "goal_lat": sensors["goal_lat"], "goal_lon": sensors["goal_lon"],
            "throttle": label_thr, "steer": label_steer,
            "exec_throttle": exec_thr, "exec_steer": exec_steer,
            "heading": sensors["heading"], "speed": sensors["speed"],
            "altitude": sensors["altitude"],
            "trav_score": trav, "collision": int(sim.collided_now),
            # true pose: training supervision only, never a model input
            "true_x": sim.x, "true_z": sim.z, "true_yaw": sim.yaw,
        })

        info = sim.step(exec_thr, exec_steer)
        if info["tipped"]:
            tipped = True
            break
        if info["reached"]:
            reached = True
            break
        if info["timeout"]:
            break

    writer.release()
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    gt = episode_gt_grids(sim)
    np.savez_compressed(os.path.join(args.output, base + ".npz"),
                        occ=np.stack(occ_bits), vis=np.stack(vis_bits),
                        elev=np.stack(elev_q), sand=np.stack(sand_q),
                        gt_occ=gt["occ"], gt_hazard=gt["hazard"],
                        gt_sand=gt["sand"], gt_origin=gt["origin"],
                        gt_res=gt["res"],
                        cells=WEDGE_CELLS, res=WEDGE_RES)
    return {"ep": ep_id, "frames": len(rows), "reached": reached,
            "tipped": tipped, "collisions": sim.collision_count,
            "scenario": sim.scenario, "spawn": sim.spawn_mode}


def main():
    """Parse args, run the worker pool, print the dataset summary table."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--episodes", type=int, default=300)
    ap.add_argument("--output", default="data_synth")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 2))
    ap.add_argument("--scenario", default=None,
                    help="force one scenario (open/dense/wall/boulders)")
    ap.add_argument("--augment", action="store_true",
                    help="domain-randomize appearance: decorrelate class "
                         "colours + photometric jitter (hue/saturation/gamma, "
                         "some grayscale) so the model learns geometry, not "
                         "colour -- for generalizing across terrains")
    ap.add_argument("--dagger", default=None,
                    help="checkpoint: the model drives, the expert labels")
    ap.add_argument("--expert_mix", type=float, default=0.2,
                    help="DAgger: fraction of steps driven by the expert")
    args = ap.parse_args()

    os.makedirs(args.output, exist_ok=True)
    if args.dagger:
        # CUDA in workers requires spawn; keep worker count modest
        ctx = mp.get_context("spawn")
        args.workers = min(args.workers, 6)
    else:
        ctx = mp.get_context("fork")
    print(f"Generating {args.episodes} episodes -> {args.output} "
          f"({args.workers} workers{', DAgger' if args.dagger else ''}"
          f"{', augment' if args.augment else ''})")

    stats = []
    with ctx.Pool(args.workers, initializer=_init_worker, initargs=(args,)) as pool:
        for st in tqdm(pool.imap_unordered(generate_episode, range(args.episodes)),
                       total=args.episodes):
            stats.append(st)

    df = pd.DataFrame(stats)
    n = len(df)
    print(f"\nDone. {n} episodes, {df['frames'].sum()} frames total.")
    print(f"  expert success rate : {df['reached'].mean() * 100:.1f}%")
    print(f"  episodes w/ contact : {(df['collisions'] > 0).mean() * 100:.1f}%")
    print(f"  tipped episodes     : {df['tipped'].mean() * 100:.1f}%")
    print(f"  mean frames/episode : {df['frames'].mean():.0f}")
    print(df.groupby("scenario")["reached"].agg(["count", "mean"]))


if __name__ == "__main__":
    main()

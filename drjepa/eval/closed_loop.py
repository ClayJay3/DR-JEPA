"""Closed-loop driving benchmark in the simulator (parallel workers).

    python -m drjepa.eval.closed_loop --policy model --checkpoint runs/geobev/best.pt \
        --episodes 144 --seed 10000 --workers 8 --out bench/geobev.json
    python -m drjepa.eval.closed_loop --policy oracle ...     # perfect perception, same nav stack
    python -m drjepa.eval.closed_loop --policy expert ...     # privileged upper bound
    ... --camera zed      # pin the rig (ZED 2i: 101.5 deg HFOV, 0.4 m) instead of random rigs

Episodes end on reaching the goal, tipping over, `--no_progress_s` seconds
without getting closer than ever before (a slow-but-closing robot is not a
failure), or the `--max_frames` backstop. Reported: success, tip-overs,
contact events / episode, collision-free episodes, terrain stalls, SPL
(success weighted by straight-line / driven distance), per-scenario
breakdown and per-stage latency. Use >= 120 episodes: at n = 30 the same
model measured 93 % and 80 % success on two runs.
"""

import argparse
import json
import multiprocessing as mp
import os
import time

import numpy as np

CAMERAS = {
    "zed": dict(width=640, height=360, hfov=101.5, h=0.40, pitch=0.0),
    "phone": dict(width=640, height=480, hfov=66.0, h=1.4, pitch=-8.0),
    "low-wide": dict(width=640, height=400, hfov=120.0, h=0.30, pitch=-5.0),
    "mast": dict(width=640, height=480, hfov=70.0, h=2.0, pitch=-15.0),
    "rolled": dict(width=640, height=480, hfov=90.0, h=0.8, pitch=-8.0, roll=6.0),
}


def make_camera(spec):
    from ..camera import Camera
    if spec in (None, "random"):
        return None
    d = CAMERAS[spec] if isinstance(spec, str) else spec
    return Camera.from_fov(d["width"], d["height"], d["hfov"], mount_height=d["h"],
                           mount_pitch=d.get("pitch", 0.0), mount_roll=d.get("roll", 0.0))


class OraclePerception:
    """Ground-truth grids in the network's output format (diagnostics)."""

    def __init__(self):
        self.sim = None
        self.gt = None

    def __call__(self, rgb, camera, R=None, t=None, body_pitch=0.0, body_roll=0.0):
        g = self.gt
        return {"occ_logit": np.where(g["occ"], 2.0, -6.0).astype(np.float32),
                "occ": g["occ"].astype(np.float32), "vis": g["vis"].astype(np.float32),
                "obs_h": g["obs_h"], "elev": g["elev"], "grade": g["grade"], "sand": g["sand"]}


def _episode(seed, policy, nav, oracle, a, record_dir=None):
    from ..config import GridConfig, SimConfig
    from ..labels import sim_targets
    from ..sim import RoverSim
    from ..sim.expert import ArcPlanner

    sim = RoverSim(SimConfig(max_frames=a.max_frames, no_progress_s=a.no_progress_s),
                   seed=seed, camera=make_camera(a.camera))
    d0 = sim.goal_dist_true()
    expert = ArcPlanner(sim, np.random.default_rng(seed)) if policy == "expert" else None
    if nav is not None:
        nav.reset()
    writer = None
    if record_dir:
        import cv2
        os.makedirs(record_dir, exist_ok=True)
        writer = cv2.VideoWriter(os.path.join(record_dir, f"ep_{seed}_{sim.scenario}.mp4"),
                                 cv2.VideoWriter_fourcc(*"mp4v"), 10.0,
                                 (sim.camera.width + 360, max(sim.camera.height, 360)))
    status, min_clear = "timeout", 99.0
    grid = GridConfig()
    while True:
        if policy == "expert":
            thr, st = expert.plan()
        else:
            if policy == "oracle":
                rgb, depth, cls, (R, t) = sim.render(gt=True)
                oracle.gt = sim_targets(sim, depth, cls, R, t, grid)
            else:
                rgb = sim.render()
            cmd = nav.step(rgb, sim.sensor_readout(), camera=sim.camera)
            thr, st = cmd.throttle, cmd.steer
            if writer is not None:
                from ..viz.hud import compose_frame
                writer.write(compose_frame(rgb, nav, sim, thr, st)[:, :, ::-1])
        info = sim.step(thr, st)
        min_clear = min(min_clear, info["clearance"])
        if info["tipped"]:
            status = "tipped"
            break
        if info["reached"]:
            status = "reached"
            break
        if info["no_progress"]:
            status = "no_progress"
            break
        if info["timeout"]:
            break
    if writer is not None:
        writer.release()
    c = sim.camera
    res = {"seed": seed, "scenario": sim.scenario, "spawn": sim.spawn_mode, "status": status,
           "frames": sim.frame, "collisions": sim.collision_count, "stalls": sim.terrain_stalls,
           "spl": (d0 / max(sim.path_len, d0)) if status == "reached" else 0.0,
           "min_clearance": round(min_clear, 2), "goal_dist": round(d0, 1),
           "camera": {"hfov": round(c.hfov_deg, 1), "h": round(c.mount_height, 2),
                      "pitch": round(c.mount_pitch, 1), "roll": round(c.mount_roll, 1),
                      "size": [c.width, c.height]}}
    sim.close()
    return res


def nav_config(overrides):
    """NavConfig with 'section.key=value' overrides (section = map /
    planner / control, or a top-level NavConfig field)."""
    from ..nav.navigator import NavConfig
    cfg = NavConfig()
    for o in overrides or []:
        k, v = o.split("=", 1)
        obj, name = (getattr(cfg, k.split(".")[0]), k.split(".")[1]) if "." in k else (cfg, k)
        cur = getattr(obj, name)
        setattr(obj, name, type(cur)(v) if not isinstance(cur, bool) else v.lower() in ("1", "true"))
    return cfg


def _worker(job):
    seeds, a, wid = job
    import torch
    torch.set_num_threads(2)
    nav, oracle = None, None
    if a.policy != "expert":
        from ..nav.navigator import Navigator
        cfg = nav_config(a.nav)
        if a.policy == "oracle":
            oracle = OraclePerception()
            nav = Navigator(oracle, cfg=cfg)
        else:
            nav = Navigator(a.checkpoint, cfg=cfg)
    out = []
    for i, s in enumerate(seeds):
        rec = a.record_dir if (wid == 0 and i < a.record) else None
        out.append(_episode(s, a.policy, nav, oracle, a, rec))
    timing = nav.timing_ms() if nav is not None else {}
    return out, timing


def summarize(eps):
    n = max(len(eps), 1)
    f = lambda cond: sum(cond(e) for e in eps) / n
    s = {"n": len(eps), "success": f(lambda e: e["status"] == "reached"),
         "tipped": f(lambda e: e["status"] == "tipped"),
         "no_progress": f(lambda e: e["status"] == "no_progress"),
         "timeout": f(lambda e: e["status"] == "timeout"),
         "collision_free": f(lambda e: e["collisions"] == 0),
         "contacts_per_ep": sum(e["collisions"] for e in eps) / n,
         "stalls_per_ep": sum(e["stalls"] for e in eps) / n,
         "spl": float(np.mean([e["spl"] for e in eps])) if eps else 0.0}
    by = {}
    for e in eps:
        by.setdefault(e["scenario"], []).append(e)
    s["by_scenario"] = {k: {"n": len(v), "success": sum(x["status"] == "reached" for x in v) / len(v),
                            "tipped": sum(x["status"] == "tipped" for x in v) / len(v),
                            "contacts": sum(x["collisions"] for x in v) / len(v),
                            "spl": float(np.mean([x["spl"] for x in v]))} for k, v in sorted(by.items())}
    return s


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", choices=["model", "oracle", "expert"], default="model")
    ap.add_argument("--checkpoint", default="runs/geobev/best.pt")
    ap.add_argument("--episodes", type=int, default=144)
    ap.add_argument("--seed", type=int, default=10000)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--camera", default="random", help="random | " + " | ".join(CAMERAS))
    ap.add_argument("--max_frames", type=int, default=3000)
    ap.add_argument("--no_progress_s", type=float, default=60.0)
    ap.add_argument("--record", type=int, default=0, help="record the first N episodes of worker 0")
    ap.add_argument("--record_dir", default="bench/videos")
    ap.add_argument("--nav", nargs="*", default=[],
                    help="navigation overrides, e.g. control.roll_reject=0.7 planner.fatal_slope=0.85")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    seeds = list(range(a.seed, a.seed + a.episodes))
    jobs = [(seeds[w::a.workers], a, w) for w in range(a.workers)]
    t0 = time.time()
    with mp.get_context("spawn").Pool(a.workers) as pool:
        parts = pool.map(_worker, jobs)
    eps = sorted([e for p, _ in parts for e in p], key=lambda e: e["seed"])
    timing = {}
    for _, tm in parts:
        for k, v in tm.items():
            timing[k] = timing.get(k, 0.0) + v / len(parts)
    s = summarize(eps)
    s["latency_ms"] = timing
    s["wall_s"] = time.time() - t0
    s["policy"], s["camera"] = a.policy, a.camera
    s["checkpoint"] = a.checkpoint if a.policy == "model" else None
    s["nav_overrides"] = a.nav
    print(json.dumps({k: v for k, v in s.items() if k != "by_scenario"}, indent=1))
    for k, v in s["by_scenario"].items():
        print(f"  {k:9s} n={v['n']:3d} success {v['success']:.3f} tipped {v['tipped']:.3f} "
              f"contacts {v['contacts']:.2f} spl {v['spl']:.3f}")
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        with open(a.out, "w") as f:
            json.dump({"summary": s, "episodes": eps}, f, indent=1)


if __name__ == "__main__":
    main()

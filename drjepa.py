"""DR-JEPA -- training and evaluation pipeline.

Commands:
    preprocess  pack (video, csv) episodes into frozen-DINOv2 feature memmaps
    train       train RoverJEPA on a packed dataset (minutes, not hours)
    eval        closed-loop evaluation in the simulator (model or expert)
    viz         open-loop HUD visualization over one recorded episode

Standalone demo scripts live at the repo root: live_inference_test.py
(endless closed-loop run) and fsd_viz.py (cinematic belief-world video).
"""

import argparse
import json
import math
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from drjepa.config import Config
from drjepa.dataset import preprocess, SeqDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==========================================================================
# TRAIN
# ==========================================================================
def train(args):
    from drjepa.model import RoverJEPA

    cfg = Config()
    tc = cfg.train
    if args.epochs:
        tc.epochs = args.epochs
    torch.manual_seed(0)
    np.random.seed(0)

    train_ds = SeqDataset(args.dataset, cfg, is_val=False)
    val_ds = SeqDataset(args.dataset, cfg, is_val=True)
    if np.ptp(train_ds.weights) > 0:
        sampler = WeightedRandomSampler(train_ds.weights, len(train_ds))
        shuffle = None
    else:
        sampler, shuffle = None, True
    train_loader = DataLoader(train_ds, batch_size=tc.batch_size,
                              shuffle=shuffle, sampler=sampler,
                              num_workers=tc.num_workers, pin_memory=True,
                              drop_last=True, persistent_workers=tc.num_workers > 0)
    val_loader = DataLoader(val_ds, batch_size=tc.batch_size, shuffle=False,
                            num_workers=max(2, tc.num_workers // 2),
                            pin_memory=True)

    model = RoverJEPA(cfg.model).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"RoverJEPA trainable parameters: {n_params / 1e6:.2f}M")

    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=tc.lr, weight_decay=tc.weight_decay, fused=device.type == "cuda")

    steps_per_epoch = max(1, len(train_loader))
    warmup = tc.warmup_epochs * steps_per_epoch
    total_steps = tc.epochs * steps_per_epoch

    def lr_lambda(step):
        if step < warmup:
            return (step + 1) / warmup
        p = (step - warmup) / max(1, total_steps - warmup)
        return tc.min_lr / tc.lr + (1 - tc.min_lr / tc.lr) * 0.5 * (
            1 + math.cos(math.pi * p))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    os.makedirs(args.save_dir, exist_ok=True)
    best_score = float("inf")
    patience = 0
    amp = torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda")

    for epoch in range(tc.epochs):
        model.train()
        t0 = time.time()
        agg = {}
        for batch in train_loader:
            tokens, label, execu, ctx, danger, dist, occ, vis, motion = [
                b.to(device, non_blocking=True) for b in batch]
            opt.zero_grad(set_to_none=True)
            with amp:
                loss, parts = model.compute_losses(tokens, label, execu, ctx,
                                                   danger, dist, occ, vis,
                                                   motion, tc)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            model.update_ema(tc.ema_decay)
            for k, v in parts.items():
                agg[k] = agg.get(k, 0.0) + v
        n = len(train_loader)
        tr = {k: v / n for k, v in agg.items()}

        # ------------- validation -------------
        model.eval()
        agg = {}
        with torch.no_grad():
            for batch in val_loader:
                tokens, label, execu, ctx, danger, dist, occ, vis, motion = [
                    b.to(device) for b in batch]
                with amp:
                    _, parts = model.compute_losses(tokens, label, execu, ctx,
                                                    danger, dist, occ, vis,
                                                    motion, tc)
                for k, v in parts.items():
                    agg[k] = agg.get(k, 0.0) + v
        va = {k: v / max(1, len(val_loader)) for k, v in agg.items()}
        # mapping quality is what drives navigation; the BC heads overfit
        # earlier and must not veto a better perception checkpoint
        score = va["map"]

        print(f"ep {epoch + 1:3d}/{tc.epochs} [{time.time() - t0:5.1f}s] "
              f"train act {tr['act']:.3f} map {tr['map']:.3f} | "
              f"val act {va['act']:.3f} map {va['map']:.3f} "
              f"IoU {va['iou']:.3f} safe {va['safe']:.3f} "
              f"jepa {va['jepa']:.3f} | score {score:.4f}"
              + ("  *best*" if score < best_score else ""))

        ckpt = {"model": model.state_dict(), "config": cfg.to_dict(),
                "epoch": epoch, "val": va}
        torch.save(ckpt, os.path.join(args.save_dir, "latest.pth"))
        if score < best_score:
            best_score = score
            patience = 0
            torch.save(ckpt, os.path.join(args.save_dir, "best.pth"))
        else:
            patience += 1
            if patience >= tc.patience:
                print(f"Early stop at epoch {epoch + 1}")
                break
    print(f"Best functional val score: {best_score:.4f}")


# ==========================================================================
# CLOSED-LOOP EVAL
# ==========================================================================
def evaluate(args):
    import cv2
    from drjepa.simulator import RoverSim, SimConfig
    from drjepa.expert import ArcPlanner
    from drjepa.pilot import Pilot, MapPilot, draw_hud

    pilot = None
    if args.policy == "model":
        if args.pilot == "map":
            pilot = MapPilot(args.checkpoint, device=device.type,
                             vo=not args.no_vo)
        else:
            pilot = Pilot(args.checkpoint, device=device.type,
                          shield=not args.no_shield)
        print(f"Loaded {args.checkpoint} (pilot={args.pilot})")

    os.makedirs(args.record_dir, exist_ok=True)
    results = []
    for ep in range(args.episodes):
        seed = args.seed + ep
        sim = RoverSim(SimConfig(), seed=seed)
        expert = ArcPlanner(sim, np.random.default_rng(seed))
        if pilot:
            pilot.reset()
        d0 = sim.goal_dist_true()
        writer = None
        if ep < args.record:
            path = os.path.join(args.record_dir,
                                f"eval_{args.policy}_{ep:02d}_{sim.scenario}.mp4")
            writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"),
                                     10.0, (sim.cfg.img_w, sim.cfg.img_h))
        status = "timeout"
        min_clear = 99.0
        while True:
            frame = sim.render()
            if isinstance(pilot, MapPilot):
                out = pilot.step(frame, sim.sensor_readout())
                thr, st = out["throttle"], out["steer"]
            elif pilot:
                dist, rel = sim.goal_vector_measured()
                out = pilot.step(frame, dist, rel, sim.meas["speed"])
                thr, st = out["throttle"], out["steer"]
            else:
                thr, st = expert.plan()
                out = {"danger": 0.0, "risk": None}
            if writer is not None:
                hud = draw_hud(frame.copy(), thr, st, out["danger"],
                               out.get("risk"), sim.goal_dist_true(),
                               extra=f"{args.policy} | {sim.scenario}")
                if isinstance(pilot, MapPilot):
                    mv = pilot.map_view()
                    hud[8:8 + mv.shape[0], -mv.shape[1] - 8:-8] = mv
                writer.write(hud)
            info = sim.step(thr, st)
            min_clear = min(min_clear, info["clearance"])
            if info["reached"]:
                status = "reached"
                break
            if info["timeout"]:
                break
        if writer is not None:
            writer.release()
        spl = (d0 / max(sim.path_len, d0)) if status == "reached" else 0.0
        results.append({"seed": seed, "scenario": sim.scenario,
                        "spawn": sim.spawn_mode, "status": status,
                        "frames": sim.frame, "collisions": sim.collision_count,
                        "spl": spl, "min_clearance": round(min_clear, 2),
                        "goal_dist": round(d0, 1)})
        r = results[-1]
        print(f"  ep {ep:3d} seed {seed} {r['scenario']:9s}/{r['spawn']:8s} "
              f"{status:8s} frames {r['frames']:4d} coll {r['collisions']:2d} "
              f"spl {spl:.2f}")

    n = len(results)
    succ = sum(r["status"] == "reached" for r in results) / n
    coll_free = sum(r["collisions"] == 0 for r in results) / n
    tot_coll = sum(r["collisions"] for r in results)
    spl = np.mean([r["spl"] for r in results])
    print("\n================ CLOSED-LOOP RESULTS ================")
    print(f" policy            : {args.policy}"
          + ("" if args.policy == "expert" else
             f" (shield {'off' if args.no_shield else 'on'})"))
    print(f" episodes          : {n}")
    print(f" success rate      : {succ * 100:.1f}%")
    print(f" collision-free eps: {coll_free * 100:.1f}%")
    print(f" contact events/ep : {tot_coll / n:.2f}")
    print(f" SPL               : {spl:.3f}")
    out_path = os.path.join(args.record_dir, f"results_{args.policy}.json")
    with open(out_path, "w") as f:
        json.dump({"summary": {"success": succ, "collision_free": coll_free,
                               "contacts_per_ep": tot_coll / n, "spl": spl},
                   "episodes": results}, f, indent=2)
    print(f" saved -> {out_path}")


# ==========================================================================
# OPEN-LOOP VIZ
# ==========================================================================
def visualize(args):
    import cv2
    import pandas as pd
    from drjepa.simulator import goal_vector
    from drjepa.pilot import Pilot, draw_hud

    pilot = Pilot(args.checkpoint, device=device.type, shield=not args.no_shield)
    csv_path = os.path.splitext(args.video)[0] + ".csv"
    df = pd.read_csv(csv_path)
    cap = cv2.VideoCapture(args.video)
    out_path = args.out or os.path.splitext(args.video)[0] + "_viz.mp4"
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (w, h))

    i = 0
    while i < len(df):
        ok, frame = cap.read()
        if not ok:
            break
        row = df.iloc[i]
        dist, rel = goal_vector(row["lat"], row["lon"], row["heading"],
                                row["goal_lat"], row["goal_lon"])
        out = pilot.step(frame, dist, rel, row["speed"])
        hud = draw_hud(frame, out["throttle"], out["steer"], out["danger"],
                       out["risk"], dist,
                       ref=(row["throttle"], row["steer"]),
                       extra="green: model | orange: logged")
        writer.write(hud)
        if args.show:
            cv2.imshow("DR-JEPA viz", hud)
            if cv2.waitKey(1) == ord("q"):
                break
        i += 1
    cap.release()
    writer.release()
    print(f"Wrote {i} frames -> {out_path}")


# ==========================================================================
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="mode", required=True)

    p = sub.add_parser("preprocess", help="pack episodes into feature memmaps")
    p.add_argument("--data_dir", required=True)
    p.add_argument("--output", required=True)

    p = sub.add_parser("train", help="train RoverJEPA on a packed dataset")
    p.add_argument("--dataset", required=True)
    p.add_argument("--save_dir", default="runs")
    p.add_argument("--epochs", type=int, default=None)

    p = sub.add_parser("eval", help="closed-loop evaluation in the simulator")
    p.add_argument("--policy", choices=["model", "expert"], default="model")
    p.add_argument("--pilot", choices=["map", "bc"], default="map",
                   help="map: perceive->map->plan navigator; bc: reactive policy")
    p.add_argument("--checkpoint", default="runs/best.pth")
    p.add_argument("--episodes", type=int, default=40)
    p.add_argument("--seed", type=int, default=1000)
    p.add_argument("--record", type=int, default=0, help="record first N episodes")
    p.add_argument("--record_dir", default="eval_out")
    p.add_argument("--no_shield", action="store_true")
    p.add_argument("--no_vo", action="store_true",
                   help="disable visual-odometry map alignment")

    p = sub.add_parser("viz", help="open-loop HUD over a recorded episode")
    p.add_argument("--video", required=True)
    p.add_argument("--checkpoint", default="runs/best.pth")
    p.add_argument("--out", default=None)
    p.add_argument("--show", action="store_true")
    p.add_argument("--no_shield", action="store_true")

    args = ap.parse_args()
    if args.mode == "preprocess":
        preprocess(args.data_dir, args.output, Config(), device=device.type)
    elif args.mode == "train":
        train(args)
    elif args.mode == "eval":
        evaluate(args)
    elif args.mode == "viz":
        visualize(args)

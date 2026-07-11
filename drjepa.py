"""DR-JEPA -- training and evaluation pipeline.

Commands:
    preprocess       pack (video, csv) episodes into frozen-DINOv2 memmaps
    train            train RoverJEPA on a packed dataset (minutes, not hours)
    eval             closed-loop evaluation in the simulator (model or expert)
    collect_beliefs  log real belief maps from pilot runs (completer data)
    tune_completer   fine-tune + calibrate the map completer on real beliefs

Standalone demo scripts live at the repo root: live_inference_test.py
(endless closed-loop run) and fsd_viz.py (cinematic belief-world video).
"""

import argparse
import glob
import json
import math
import os
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from drjepa.config import Config
from drjepa.dataset import preprocess, SeqDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==========================================================================
# TRAIN
# ==========================================================================
def train(args):
    """Train RoverJEPA on a packed dataset.

    AdamW + cosine schedule with warmup, bf16 autocast, weighted sampling
    when DAgger episodes are present. Checkpoints are selected on the
    perception losses (wedge + completion + danger is reported but does
    not vote). Saves best.pth / latest.pth with the full config embedded.
    """
    from drjepa.model import RoverJEPA

    cfg = Config()
    tc = cfg.train
    if args.epochs:
        tc.epochs = args.epochs
    rw = str(getattr(args, "real_weight", "auto")).lower()
    tc.real_weight = -1.0 if rw in ("auto", "-1") else float(rw)
    torch.manual_seed(args.train_seed)
    np.random.seed(args.train_seed)

    train_ds = SeqDataset(args.dataset, cfg, is_val=False)
    # one combined val set (sim + real): the model is validated and selected
    # on both together, and the stratified split guarantees real is in it
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
        """Linear warmup then cosine decay to min_lr."""
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
            batch = {k: v.to(device, non_blocking=True)
                     for k, v in batch.items()}
            opt.zero_grad(set_to_none=True)
            with amp:
                loss, parts = model.compute_losses(batch, tc)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            for k, v in parts.items():
                agg[k] = agg.get(k, 0.0) + v
        n = len(train_loader)
        tr = {k: v / n for k, v in agg.items()}

        # ------------- validation -------------
        def run_val(loader):
            """Mean per-loss over a val loader."""
            agg = {}
            with torch.no_grad():
                for batch in loader:
                    batch = {k: v.to(device) for k, v in batch.items()}
                    with amp:
                        _, parts = model.compute_losses(batch, tc)
                    for k, v in parts.items():
                        agg[k] = agg.get(k, 0.0) + v
            return {k: v / max(1, len(loader)) for k, v in agg.items()}

        model.eval()
        va = run_val(val_loader)
        # perception quality is what drives navigation: all wedge +
        # completion losses vote; hazard weighs like occupancy scale-wise
        # -- missing a bank is the one terminal perception failure. Scored
        # over the combined sim+real val set (optimize/generalize both).
        score = va["map"] + 0.5 * va["elev"] + 0.25 * va["sand"] + \
            0.5 * va["comp"] + 0.5 * va["haz"]

        print(f"ep {epoch + 1:3d}/{tc.epochs} [{time.time() - t0:5.1f}s] "
              f"train map {tr['map']:.3f} comp {tr['comp']:.3f} | "
              f"val map {va['map']:.3f} IoU {va['iou']:.3f} "
              f"elev {va['elev']:.3f} sand {va['sand']:.3f} "
              f"haz {va['haz']:.3f} "
              f"comp {va['comp']:.3f} cIoU {va['ciou']:.3f} "
              f"safe {va['safe']:.3f} | score {score:.4f}"
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
    """Closed-loop evaluation over fresh randomized worlds.

    Drives N seeded episodes with the chosen policy (map pilot or the
    privileged expert) and reports success rate, collision-free
    rate, contact events, and SPL (success-weighted path efficiency).
    Optionally records the first episodes as HUD videos; the full
    per-episode table is written to results_<policy>.json.
    """
    import cv2
    from drjepa.simulator import RoverSim, SimConfig
    from drjepa.expert import ArcPlanner
    from drjepa.pilot import MapPilot, draw_hud

    pilot = None
    if args.policy == "model":
        pilot = MapPilot(args.checkpoint, device=device.type,
                         vo=not args.no_vo, complete=args.complete)
        pilot.use_invite = not getattr(args, "no_invite", False)
        pilot.use_governor = not getattr(args, "no_governor", False)
        print(f"Loaded {args.checkpoint}")

    os.makedirs(args.record_dir, exist_ok=True)
    results = []
    for ep in range(args.episodes):
        seed = args.seed + ep
        sim = RoverSim(SimConfig(max_frames=args.max_frames), seed=seed)
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
            if info["tipped"]:
                status = "tipped"
                break
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
                        "stalls": sim.terrain_stalls,
                        "spl": spl, "min_clearance": round(min_clear, 2),
                        "goal_dist": round(d0, 1)})
        r = results[-1]
        print(f"  ep {ep:3d} seed {seed} {r['scenario']:9s}/{r['spawn']:8s} "
              f"{status:8s} frames {r['frames']:4d} coll {r['collisions']:2d} "
              f"spl {spl:.2f}")

    n = len(results)
    succ = sum(r["status"] == "reached" for r in results) / n
    tipped = sum(r["status"] == "tipped" for r in results) / n
    coll_free = sum(r["collisions"] == 0 for r in results) / n
    tot_coll = sum(r["collisions"] for r in results)
    tot_stall = sum(r["stalls"] for r in results)
    spl = np.mean([r["spl"] for r in results])
    label = args.policy
    if args.policy == "model":
        label += (" +vo" if not args.no_vo else " -vo")
        if args.complete:
            mech = [m for m, on in
                    [("inv", not getattr(args, "no_invite", False)),
                     ("gov", not getattr(args, "no_governor", False))] if on]
            label += f" +complete[{'+'.join(mech) or 'none'}]"
        else:
            label += " -complete"
    print("\n================ CLOSED-LOOP RESULTS ================")
    print(f" policy            : {label}")
    print(f" episodes          : {n}")
    print(f" success rate      : {succ * 100:.1f}%")
    print(f" tipped episodes   : {tipped * 100:.1f}%")
    print(f" collision-free eps: {coll_free * 100:.1f}%")
    print(f" contact events/ep : {tot_coll / n:.2f}")
    print(f" terrain stalls/ep : {tot_stall / n:.2f}")
    print(f" SPL               : {spl:.3f}")
    out_path = os.path.join(args.record_dir, f"results_{args.policy}.json")
    with open(out_path, "w") as f:
        json.dump({"summary": {"success": succ, "tipped": tipped,
                               "collision_free": coll_free,
                               "contacts_per_ep": tot_coll / n,
                               "stalls_per_ep": tot_stall / n, "spl": spl},
                   "episodes": results}, f, indent=2)
    print(f" saved -> {out_path}")


# ==========================================================================
# OPEN-LOOP VIZ
# ==========================================================================
# ==========================================================================
# MAP-COMPLETER: REAL-BELIEF DATA + FINE-TUNE + CALIBRATION
# ==========================================================================
def collect_beliefs(args):
    """Log real belief-map snapshots from closed-loop pilot runs.

    The completer was originally trained on synthetic trajectory-fan masks
    over GT grids -- but at deployment it reads the pilot's actual fused
    belief (perception noise, VO-corrected pose, decay and all). This
    collects (belief crop, GT crop) pairs from real runs so the completer
    can be fine-tuned on exactly its deployment input distribution.
    One .npz per episode: inputs (S,4,G,G) f16, targets occ/haz/sand
    (S,G,G) u8, valid (S,G,G) bool.
    """
    from drjepa.simulator import RoverSim, SimConfig, episode_gt_grids
    from drjepa.pilot import MapPilot

    pilot = MapPilot(args.checkpoint, device=device.type)
    os.makedirs(args.output, exist_ok=True)
    G = pilot.cfg.model.comp_cells
    cres = pilot.cfg.model.comp_res
    tot = 0
    for ep in range(args.episodes):
        seed = args.seed + ep
        # over-sample walls: the weakest scenario and the one where
        # anticipating hidden structure has the most to offer
        scenario = "wall" if (ep % 10) < 3 else None
        sim = RoverSim(SimConfig(), seed=seed, scenario=scenario)
        gt = episode_gt_grids(sim)
        go, gres = gt["origin"], float(gt["res"])
        pilot.reset()
        ins, t_occ, t_haz, t_sand, valids = [], [], [], [], []
        while True:
            frame = sim.render()
            out = pilot.step(frame, sim.sensor_readout())
            if pilot.step_i >= 30 and pilot.step_i % args.snap_every == 0:
                comp_in, origin = pilot._completion_input()
                # window cell centers -> GT grid indices
                gi = ((origin[0] + (np.arange(G) + 0.5) * cres - go[0])
                      / gres).astype(int)
                gj = ((origin[1] + (np.arange(G) + 0.5) * cres - go[1])
                      / gres).astype(int)
                valid = ((gi >= 0) & (gi < gt["occ"].shape[0]))[:, None] & \
                        ((gj >= 0) & (gj < gt["occ"].shape[1]))[None, :]
                gi = np.clip(gi, 0, gt["occ"].shape[0] - 1)
                gj = np.clip(gj, 0, gt["occ"].shape[1] - 1)
                ins.append(comp_in.astype(np.float16))
                t_occ.append(gt["occ"][np.ix_(gi, gj)])
                t_haz.append(gt["hazard"][np.ix_(gi, gj)])
                t_sand.append(gt["sand"][np.ix_(gi, gj)])
                valids.append(valid)
            info = sim.step(out["throttle"], out["steer"])
            if info["reached"] or info["tipped"] or info["timeout"]:
                break
        if ins:
            np.savez_compressed(
                os.path.join(args.output, f"ep{ep:04d}.npz"),
                inputs=np.stack(ins), occ=np.stack(t_occ),
                hazard=np.stack(t_haz), sand=np.stack(t_sand),
                valid=np.stack(valids), seed=seed, scenario=sim.scenario)
            tot += len(ins)
        print(f"  ep {ep:3d} seed {seed} {sim.scenario:9s} "
              f"snaps {len(ins):3d} (total {tot})", flush=True)
    print(f"Collected {tot} belief snapshots -> {args.output}")


def _rank_auc(scores, labels):
    """AUC via the Mann-Whitney rank statistic (no sklearn dependency)."""
    pos = labels > 0.5
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(1, len(scores) + 1)
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2)
                 / (n_pos * n_neg))


def tune_completer(args):
    """Fine-tune the map completer on real belief snapshots + calibrate.

    Freezes everything except map_completer, trains on collect_beliefs
    output (BCE on valid cells: full weight where UNOBSERVED -- the
    anticipation task -- and 0.15 weight where observed, which teaches
    mild denoising of perception false positives). Then fits a per-channel
    temperature on held-out episodes so downstream planning thresholds are
    probability-calibrated, and writes a new checkpoint with the updated
    weights + calibration. Reports hidden-cell AUC before and after.
    """
    from drjepa.model import RoverJEPA

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = Config.from_dict(ckpt["config"])
    model = RoverJEPA(cfg.model).to(device)
    model.load_state_dict(ckpt["model"])

    files = sorted(glob.glob(os.path.join(args.belief_dir, "*.npz")))
    if not files:
        raise SystemExit(f"no belief snapshots in {args.belief_dir}")
    n_val = max(2, len(files) // 10)
    splits = {"train": files[:-n_val], "val": files[-n_val:]}
    data = {}
    for name, fl in splits.items():
        ins, tgt, val, obs = [], [], [], []
        for f in fl:
            z = np.load(f)
            ins.append(z["inputs"].astype(np.float32))
            tgt.append(np.stack([z["occ"], z["hazard"],
                                 z["sand"].astype(np.float32) / 255.0], 1))
            val.append(z["valid"])
            obs.append(z["inputs"][:, 3] > 0.5)
        data[name] = (np.concatenate(ins),
                      np.concatenate(tgt).astype(np.float32),
                      np.concatenate(val), np.concatenate(obs))
        print(f"{name}: {len(data[name][0])} snapshots from {len(fl)} eps")

    pos_w = torch.tensor([4.0, 4.0, 2.0], device=device).view(1, 3, 1, 1)

    def batch_loss(mdl, xb, tb, vb, ob):
        """Weighted BCE over valid cells: hidden 1.0, observed 0.15."""
        logit = mdl.map_completer(xb)
        w = vb[:, None] * torch.where(ob[:, None], 0.15, 1.0)
        bce = nn.functional.binary_cross_entropy_with_logits(
            logit, tb, reduction="none",
            pos_weight=pos_w)
        return (bce * w).sum() / w.sum().clamp(min=1.0)

    def evaluate_split(mdl, name):
        """Mean val loss + hidden-cell AUC (occ, hazard) + logits cache."""
        xs, ts, vs, os_ = data[name]
        losses, logits = [], []
        mdl.eval()
        with torch.no_grad():
            for k in range(0, len(xs), 64):
                xb = torch.from_numpy(xs[k:k + 64]).to(device)
                tb = torch.from_numpy(ts[k:k + 64]).to(device)
                vb = torch.from_numpy(vs[k:k + 64]).float().to(device)
                ob = torch.from_numpy(os_[k:k + 64]).to(device)
                losses.append(float(batch_loss(mdl, xb, tb, vb, ob)))
                logits.append(mdl.map_completer(xb).cpu().numpy())
        logits = np.concatenate(logits)
        hidden = vs & ~os_
        aucs = [_rank_auc(logits[:, c][hidden], ts[:, c][hidden] > 0.5)
                for c in (0, 1)]
        return float(np.mean(losses)), aucs, logits

    _, auc0, _ = evaluate_split(model, "val")
    print(f"pre-tune hidden AUC: occ {auc0[0]:.3f} hazard {auc0[1]:.3f}")

    for p in model.parameters():
        p.requires_grad = False
    for p in model.map_completer.parameters():
        p.requires_grad = True
    opt = torch.optim.AdamW(model.map_completer.parameters(),
                            lr=args.lr, weight_decay=1e-4)
    xs, ts, vs, os_ = data["train"]
    best, best_state, bad = float("inf"), None, 0
    for epoch in range(args.epochs):
        model.train()
        perm = np.random.default_rng(epoch).permutation(len(xs))
        for k in range(0, len(perm), 64):
            idx = perm[k:k + 64]
            xb = torch.from_numpy(xs[idx]).to(device)
            tb = torch.from_numpy(ts[idx]).to(device)
            vb = torch.from_numpy(vs[idx]).float().to(device)
            ob = torch.from_numpy(os_[idx]).to(device)
            loss = batch_loss(model, xb, tb, vb, ob)
            opt.zero_grad()
            loss.backward()
            opt.step()
        vl, aucs, _ = evaluate_split(model, "val")
        marker = ""
        if vl < best - 1e-4:
            best, bad = vl, 0
            best_state = {k: v.detach().clone() for k, v in
                          model.map_completer.state_dict().items()}
            marker = " *"
        else:
            bad += 1
        print(f"epoch {epoch:2d} val {vl:.4f} "
              f"AUC occ {aucs[0]:.3f} haz {aucs[1]:.3f}{marker}", flush=True)
        if bad >= args.patience:
            break
    if best_state is not None:
        model.map_completer.load_state_dict(best_state)

    # per-channel temperature on held-out hidden cells (NLL grid search)
    _, auc1, logits = evaluate_split(model, "val")
    print(f"post-tune hidden AUC: occ {auc1[0]:.3f} hazard {auc1[1]:.3f}")
    _, ts_v, vs_v, os_v = data["val"]
    hidden = vs_v & ~os_v
    temps = []
    for c in range(3):
        lg = torch.from_numpy(logits[:, c][hidden])
        tg = torch.from_numpy(ts_v[:, c][hidden])
        cands = np.exp(np.linspace(np.log(0.25), np.log(4.0), 33))
        nll = [float(nn.functional.binary_cross_entropy_with_logits(
            lg / t, tg)) for t in cands]
        temps.append(float(cands[int(np.argmin(nll))]))
    print(f"calibration temperatures (occ, hazard, sand): "
          f"{[round(t, 2) for t in temps]}")

    ckpt["model"] = model.state_dict()
    ckpt["comp_calib"] = {"t": temps}
    out = args.out or os.path.join(os.path.dirname(args.checkpoint),
                                   "best_tuned.pth")
    torch.save(ckpt, out)
    print(f"Saved fine-tuned + calibrated checkpoint -> {out}")


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
    p.add_argument("--train_seed", type=int, default=0,
                   help="init/shuffle seed (contacts are heavy-tailed; "
                        "train-twice-and-select is legitimate variance control)")
    p.add_argument("--real_weight", default="auto",
                   help="sampling weight for real (phone) episodes: 'auto' "
                        "(default) balances them to a fixed share of the "
                        "signal, capped; or a float for a fixed weight")

    p = sub.add_parser("eval", help="closed-loop evaluation in the simulator")
    p.add_argument("--policy", choices=["model", "expert"], default="model")
    p.add_argument("--checkpoint", default="runs/best.pth")
    p.add_argument("--episodes", type=int, default=40)
    p.add_argument("--seed", type=int, default=1000)
    p.add_argument("--record", type=int, default=0, help="record first N episodes")
    p.add_argument("--record_dir", default="eval_out")
    p.add_argument("--max_frames", type=int, default=900,
                   help="episode time cap in frames (10 fps)")
    p.add_argument("--no_vo", action="store_true",
                   help="disable visual-odometry map alignment")
    p.add_argument("--complete", action="store_true",
                   help="use map-space JEPA predictions while driving: "
                        "invite-only planner costs + predicted-hazard "
                        "speed governor")
    p.add_argument("--no_invite", action="store_true",
                   help="ablation: with --complete, disable the invite-only "
                        "planner-cost integration")
    p.add_argument("--no_governor", action="store_true",
                   help="ablation: with --complete, disable the "
                        "predicted-hazard speed governor")

    p = sub.add_parser("collect_beliefs",
                       help="log real belief-map snapshots from pilot runs "
                            "for completer fine-tuning")
    p.add_argument("--checkpoint", default="runs/best.pth")
    p.add_argument("--episodes", type=int, default=240)
    p.add_argument("--seed", type=int, default=200000)
    p.add_argument("--snap_every", type=int, default=20)
    p.add_argument("--output", default="belief_data")

    p = sub.add_parser("tune_completer",
                       help="fine-tune + calibrate the map completer on "
                            "collected real belief snapshots")
    p.add_argument("--checkpoint", default="runs/best.pth")
    p.add_argument("--belief_dir", default="belief_data")
    p.add_argument("--out", default=None)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--patience", type=int, default=6)
    p.add_argument("--lr", type=float, default=3e-4)

    args = ap.parse_args()
    if args.mode == "preprocess":
        preprocess(args.data_dir, args.output, Config(), device=device.type)
    elif args.mode == "train":
        train(args)
    elif args.mode == "eval":
        evaluate(args)
    elif args.mode == "collect_beliefs":
        collect_beliefs(args)
    elif args.mode == "tune_completer":
        tune_completer(args)

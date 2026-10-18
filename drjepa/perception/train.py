"""Train GeoBEV on sim episodes + real labelled recordings.

    python -m drjepa.perception.train --sim data/sim \
        --real "recordings/zed/ZED_2i_31237348@0:0.85" "recordings/zed/ZED_2i_39163798@0:0.85" \
        --val_real "recordings/zed/ZED_2i_31237348@0.85:1" "recordings/zed/ZED_2i_39163798@0.85:1" \
        --encoder efficientnet_b0 --out runs/geobev

Selection metric (higher = better), averaged over the sim and real val
sets: obstacle average precision minus half the 0-12 m elevation MAE.
Average precision rather than AUC because obstacles are 2-5 % of visible
cells -- AUC stays high while precision collapses, and precision is what
keeps phantom walls out of the map.
"""

import argparse
import glob
import json
import math
import os
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

from .dataset import FrameDataset
from .model import GeoBEV, ModelConfig, count_params

LOSS_W = {"vis": 1.0, "occ": 2.0, "obs_h": 1.0, "elev": 1.0, "grade": 1.0, "sand": 0.5, "depth": 0.5,
          "feat": 1.0, "kd": 1.0}


GRADE_Q = 0.5            # pinball quantile for the grade head (0.5 = L1)
GRADE_STEEP_W = 0.0      # extra weight on steep cells (grade 0.25 -> 0.55)


def _pinball(pred, target, q):
    """Quantile loss: q > 0.5 penalizes UNDER-estimation more."""
    d = target - pred
    return torch.maximum(q * d, (q - 1) * d)


def compute_losses(out, b, cells, teacher=None):
    """Masked per-channel losses; returns (total, parts dict).

    `teacher` (optional): outputs of a frozen teacher GeoBEV on the same
    inputs, incl. its encoder features -> feature + output distillation.
    Unlabeled frames (b["is_labeled"] = 0) only feed the distillation terms."""
    dev = out["occ"].device
    real = b["is_real"].view(-1, 1, 1)
    lab = b.get("is_labeled", torch.ones_like(b["is_real"])).view(-1, 1, 1)
    vis = b["vis"]
    zw = torch.linspace(1.5, 0.6, cells, device=dev).view(1, 1, cells)      # near rows matter more
    m = vis * zw * lab
    msum = m.sum().clamp(min=1.0)
    # elevation / grade can be unknown (NaN) on real cells that only see an
    # object: supervise them nowhere rather than at a made-up value
    ev = torch.isfinite(b["elev"]).float()
    me = m * ev
    elev_gt = torch.nan_to_num(b["elev"], nan=0.0)
    grade_gt = torch.nan_to_num(b["grade"], nan=0.0)
    sim_m = m * (1 - real) * torch.isfinite(b["grade"]).float()
    smsum = sim_m.sum().clamp(min=1.0)
    occ = b["occ"]
    parts = {
        "vis": (F.binary_cross_entropy_with_logits(out["vis"], vis, reduction="none") * lab).sum()
               / (lab.sum() * cells * cells).clamp(min=1.0),
        "occ": (F.binary_cross_entropy_with_logits(out["occ"], occ, reduction="none",
                                                   pos_weight=torch.tensor(3.0, device=dev)) * m).sum() / msum,
        "obs_h": (F.smooth_l1_loss(out["obs_h"], b["obs_h"].clamp(0, 2.5), beta=0.1, reduction="none")
                  * m * (1 + 2 * occ)).sum() / msum,
        "elev": (F.smooth_l1_loss(out["elev"], elev_gt.clamp(-3.4, 3.4), beta=0.1, reduction="none")
                 * me).sum() / me.sum().clamp(min=1.0),
        # slopes that tip a robot are rare; plain regression shrinks them toward
        # the flat majority (measured: every sim tip-over happened on observed
        # ground believed 0.12-0.42 where the truth was 0.53-0.73). Weight the
        # steep cells up and, optionally, err toward steeper when unsure.
        "grade": ((_pinball(out["grade"], grade_gt.clamp(0, 1.5), GRADE_Q) if GRADE_Q != 0.5 else
                   F.smooth_l1_loss(out["grade"], grade_gt.clamp(0, 1.5), beta=0.05, reduction="none"))
                  * (1 + GRADE_STEEP_W * ((grade_gt - 0.25) / 0.3).clamp(0, 1)) * sim_m).sum() / smsum,
        "sand": (F.binary_cross_entropy_with_logits(out["sand"], b["sand"], reduction="none")
                 * sim_m).sum() / smsum,
    }
    if "depth_log" in out and "depth" in b:
        d = b["depth"]
        dv = ((d > 0.2) & (d < 60)).float()
        err = (out["depth_log"] - torch.log(d.clamp(min=0.2))).abs()
        parts["depth"] = (err * dv).sum() / dv.sum().clamp(min=1.0)
    if teacher is not None:
        tf, sf = teacher["feat"], out["feat"]
        parts["feat"] = ((sf - tf) ** 2).mean() / tf.var().clamp(min=1e-6)
        tv = torch.sigmoid(teacher["vis"]) * zw                    # trust the teacher where it sees
        tvs = tv.sum().clamp(min=1.0)
        kd = 0.0
        for k in ("occ", "sand"):
            kd = kd + (F.binary_cross_entropy_with_logits(out[k], torch.sigmoid(teacher[k]), reduction="none")
                       * tv).sum() / tvs
        kd = kd + F.binary_cross_entropy_with_logits(out["vis"], torch.sigmoid(teacher["vis"]))
        for k in ("obs_h", "elev", "grade"):
            kd = kd + (F.smooth_l1_loss(out[k], teacher[k], beta=0.1, reduction="none") * tv).sum() / tvs
        parts["kd"] = kd
    total = sum(LOSS_W[k] * v for k, v in parts.items())
    return total, {k: float(v.detach()) for k, v in parts.items()}


class Meter:
    """Streaming detection / regression metrics over visible cells."""

    def __init__(self, bins=400):
        self.bins = bins
        self.pos = np.zeros(bins)
        self.neg = np.zeros(bins)
        self.e_err = 0.0
        self.e_n = 0
        self.h_err = 0.0
        self.h_n = 0

    def add(self, out, b, near_cells):
        with torch.no_grad():
            v = b["vis"] > 0.5
            p = torch.sigmoid(out["occ"].float())[v]
            y = b["occ"][v] > 0.5
            idx = (p * (self.bins - 1)).long().clamp(0, self.bins - 1).cpu().numpy()
            yy = y.cpu().numpy()
            self.pos += np.bincount(idx[yy], minlength=self.bins)
            self.neg += np.bincount(idx[~yy], minlength=self.bins)
            near = v.clone()
            near[..., near_cells:] = False
            ne = near & torch.isfinite(b["elev"])
            self.e_err += float((out["elev"].float() - b["elev"]).abs()[ne].sum())
            self.e_n += int(ne.sum())
            self.h_err += float((out["obs_h"].float() - b["obs_h"]).abs()[near].sum())
            self.h_n += int(near.sum())

    def result(self):
        tp = np.cumsum(self.pos[::-1])
        fp = np.cumsum(self.neg[::-1])
        prec = tp / np.maximum(tp + fp, 1)
        rec = tp / max(self.pos.sum(), 1)
        ap = float(np.sum(np.diff(np.r_[0.0, rec]) * prec))
        auc = float(np.sum(np.diff(np.r_[0.0, fp / max(self.neg.sum(), 1)]) *
                           (tp / max(self.pos.sum(), 1)))) if self.neg.sum() else float("nan")
        f1 = float(np.max(2 * prec * rec / np.maximum(prec + rec, 1e-9)))
        return {"ap": ap, "auc": auc, "best_f1": f1, "elev_mae": self.e_err / max(self.e_n, 1),
                "obs_h_mae": self.h_err / max(self.h_n, 1), "occ_rate": float(self.pos.sum() /
                                                                          max(self.pos.sum() + self.neg.sum(), 1))}


def to_dev(b, dev):
    out = {k: v.to(dev, non_blocking=True) for k, v in b.items()}
    out["img"] = out["img"].float() / 255.0
    return out


@torch.no_grad()
def validate(model, loader, dev, near_cells):
    model.eval()
    meter, agg, n = Meter(), {}, 0
    for b in loader:
        b = to_dev(b, dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(b["img"], b["K"], b["R"], b["t"])
        out = {k: v.float() for k, v in out.items()}
        _, parts = compute_losses(out, b, model.cfg.cells)
        for k, v in parts.items():
            agg[k] = agg.get(k, 0.0) + v
        n += 1
        meter.add(out, b, near_cells)
    r = meter.result()
    r.update({f"loss_{k}": v / max(n, 1) for k, v in agg.items()})
    return r


def sim_split(root, val_every=20):
    eps = sorted(glob.glob(os.path.join(root, "ep_*")))
    eps = [e for e in eps if os.path.exists(os.path.join(e, "labels.npz"))]
    val = [e for i, e in enumerate(eps) if i % val_every == 0]
    tr = [e for i, e in enumerate(eps) if i % val_every != 0]
    return tr, val


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sim", default="data/sim")
    ap.add_argument("--real", nargs="*", default=[])
    ap.add_argument("--val_real", nargs="*", default=[])
    ap.add_argument("--real_share", type=float, default=0.15)
    ap.add_argument("--encoder", default="efficientnet_b0")
    ap.add_argument("--lift", default="geometric", choices=["geometric", "implicit"])
    ap.add_argument("--depth_aux", action="store_true", help="training-only depth supervision")
    ap.add_argument("--teacher", default=None, help="frozen GeoBEV checkpoint to distill from")
    ap.add_argument("--grade_q", type=float, default=0.5,
                    help="grade pinball quantile (>0.5 = prefer over-estimating slopes)")
    ap.add_argument("--grade_steep_w", type=float, default=0.0,
                    help="extra grade-loss weight on steep cells (0.25-0.55 grade ramp)")
    ap.add_argument("--unlabeled", nargs="*", default=[], help="unlabeled real episodes (distillation only)")
    ap.add_argument("--init_head_from_teacher", action="store_true",
                    help="start the lift + top-down head from the teacher's weights")
    ap.add_argument("--input_w", type=int, default=512)
    ap.add_argument("--input_h", type=int, default=320)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--epoch_samples", type=int, default=60000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--enc_lr_mult", type=float, default=0.3)
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--out", default="runs/geobev")
    ap.add_argument("--max_sim_eps", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    global GRADE_Q, GRADE_STEEP_W
    GRADE_Q, GRADE_STEEP_W = a.grade_q, a.grade_steep_w
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    dev = torch.device("cuda")
    mcfg = ModelConfig(encoder=a.encoder, input_w=a.input_w, input_h=a.input_h, lift=a.lift,
                       depth_aux=a.depth_aux)
    sim_tr, sim_val = sim_split(a.sim)
    if a.max_sim_eps:
        sim_tr = sim_tr[:a.max_sim_eps]
    train_ds = FrameDataset(sim_tr + list(a.real) + list(a.unlabeled), mcfg, train=True, depth=a.depth_aux)
    val_sim = FrameDataset(sim_val, mcfg, train=False, stride=3)
    val_real = FrameDataset(a.val_real, mcfg, train=False) if a.val_real else None
    n_real = int(train_ds.is_real.sum())
    n_sim = len(train_ds) - n_real
    w = np.ones(len(train_ds))
    if n_real and a.real_share > 0:
        w[train_ds.is_real] = a.real_share / (1 - a.real_share) * n_sim / n_real
    sampler = WeightedRandomSampler(w, a.epoch_samples, replacement=True)
    tl = DataLoader(train_ds, a.batch, sampler=sampler, num_workers=a.workers, pin_memory=True,
                    drop_last=True, persistent_workers=True, prefetch_factor=4)
    vls = DataLoader(val_sim, 32, num_workers=6, pin_memory=True)
    vlr = DataLoader(val_real, 32, num_workers=4, pin_memory=True) if val_real else None
    print(f"train: {n_sim} sim + {n_real} real frames (real share {a.real_share:.0%}); "
          f"val: {len(val_sim)} sim + {len(val_real) if val_real else 0} real")

    model = GeoBEV(mcfg).to(dev)
    teacher = None
    if a.teacher:
        tck = torch.load(a.teacher, map_location="cpu", weights_only=False)
        tcfg = ModelConfig.from_dict(tck["model_cfg"])
        tcfg.pretrained = False
        teacher = GeoBEV(tcfg)
        teacher.load_state_dict(tck["model"])
        teacher.to(dev).eval()
        for p in teacher.parameters():
            p.requires_grad = False
        if a.init_head_from_teacher:
            head = {k: v for k, v in tck["model"].items() if k.startswith("bev.")}
            model.load_state_dict(head, strict=False)
            print(f"  student head initialised from teacher ({len(head)} tensors)")
        print(f"  distilling from {a.teacher} [{tcfg.encoder}]")
    print(f"GeoBEV[{a.encoder}] trainable params {count_params(model) / 1e6:.2f} M")
    enc = [p for p in model.encoder.parameters() if p.requires_grad]
    rest = [p for n, p in model.named_parameters() if not n.startswith("encoder.") and p.requires_grad]
    opt = torch.optim.AdamW([{"params": enc, "lr": a.lr * a.enc_lr_mult},
                             {"params": rest, "lr": a.lr}], weight_decay=0.02)
    steps = a.epochs * (a.epoch_samples // a.batch)
    warm = min(1000, steps // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.03 + 0.97 * 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm))))
    os.makedirs(a.out, exist_ok=True)
    near_cells = int(12.0 / mcfg.res)
    best, hist = -1e9, []
    for ep in range(a.epochs):
        model.train()
        t0, agg, n = time.time(), {}, 0
        for b in tl:
            b = to_dev(b, dev)
            tout = None
            with torch.autocast("cuda", dtype=torch.bfloat16):
                if teacher is not None:
                    with torch.no_grad():
                        tout = {k: v.float() for k, v in
                                teacher(b["img"], b["K"], b["R"], b["t"], return_feat=True).items()}
                out = model(b["img"], b["K"], b["R"], b["t"], return_feat=teacher is not None)
            out = {k: v.float() for k, v in out.items()}
            loss, parts = compute_losses(out, b, mcfg.cells, tout)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            sched.step()
            for k, v in parts.items():
                agg[k] = agg.get(k, 0.0) + v
            n += 1
        tr = {k: v / n for k, v in agg.items()}
        vs = validate(model, vls, dev, near_cells)
        vr = validate(model, vlr, dev, near_cells) if vlr else None
        doms = [vs] + ([vr] if vr else [])
        score = float(np.mean([d["ap"] - 0.5 * d["elev_mae"] for d in doms]))
        rec = {"epoch": ep + 1, "train": tr, "val_sim": vs, "val_real": vr, "score": score,
               "time_s": time.time() - t0}
        hist.append(rec)
        line = (f"ep {ep + 1:2d} [{time.time() - t0:5.0f}s] train occ {tr['occ']:.3f} elev {tr['elev']:.3f} | "
                f"SIM AP {vs['ap']:.3f} AUC {vs['auc']:.3f} F1 {vs['best_f1']:.3f} elev {vs['elev_mae']:.3f} "
                f"obsh {vs['obs_h_mae']:.3f}")
        if vr:
            line += (f" | REAL AP {vr['ap']:.3f} AUC {vr['auc']:.3f} F1 {vr['best_f1']:.3f} "
                     f"elev {vr['elev_mae']:.3f} obsh {vr['obs_h_mae']:.3f}")
        print(line + f" | score {score:.4f}" + (" *" if score > best else ""), flush=True)
        ck = {"model": model.state_dict(), "model_cfg": mcfg.to_dict(), "epoch": ep + 1,
              "val": {"sim": vs, "real": vr}, "args": vars(a)}
        torch.save(ck, os.path.join(a.out, "last.pt"))
        if score > best:
            best = score
            torch.save(ck, os.path.join(a.out, "best.pt"))
        with open(os.path.join(a.out, "history.json"), "w") as f:
            json.dump(hist, f, indent=1)


if __name__ == "__main__":
    main()

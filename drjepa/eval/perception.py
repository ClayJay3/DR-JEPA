"""Open-loop perception benchmark on labelled episodes (sim or real).

A method is any callable `predict(episode, k) -> dict` returning robot-
frame grid predictions for frame k of an episode:

    occ    (C, C) obstacle probability (or any monotone score)
    elev   (C, C) terrain elevation (m)            [optional]
    vis    (C, C) predicted visibility / confidence [optional]
    occ_threshold  scalar operating point for `occ` [optional, default 0.5]

Scores are computed on GT-visible cells only, split by forward range:
obstacle AUC (threshold-free), best-F1 (oracle threshold, same procedure
for every method), F1 / precision / recall at the method's own operating
point, and elevation MAE.
"""

import json
import os

import cv2
import numpy as np

from ..camera import Camera

BANDS = ((0.0, 6.0), (6.0, 12.0), (12.0, 24.0))


class Episode:
    """One labelled episode directory (sim `ep_*` or real recording)."""

    def __init__(self, path):
        self.path = path.rstrip("/")
        with open(os.path.join(self.path, "episode.json")) as f:
            self.meta = json.load(f)
        self.camera = Camera.from_dict(self.meta["camera"])
        z = np.load(os.path.join(self.path, "labels.npz"))
        self.n = len(z["R"])
        C = int(z["cells"])
        self.cells, self.res = C, float(z["res"])
        unpack = lambda a: np.unpackbits(a, axis=-1)[..., :C * C].reshape(self.n, C, C).astype(bool)
        self.vis, self.occ = unpack(z["vis"]), unpack(z["occ"])
        self.obs_h = z["obs_h"].astype(np.float32)
        self.elev = z["elev"].astype(np.float32)
        self.grade = z["grade"].astype(np.float32)
        self.sand = z["sand"].astype(np.float32) / 255.0
        self.R, self.t = z["R"].astype(np.float64), z["t"].astype(np.float64)
        self.speed = z["speed"]
        self.frame_ids = z["frame_ids"] if "frame_ids" in z.files else np.arange(self.n)
        self.domain = self.meta.get("domain", "sim")
        self._real = os.path.isdir(os.path.join(self.path, "depth"))

    def __len__(self):
        return self.n

    def rgb(self, k):
        """(H, W, 3) uint8 RGB."""
        fid = int(self.frame_ids[k])
        f = os.path.join(self.path, "rgb", f"{fid:05d}.jpg" if self._real else f"{fid:04d}.jpg")
        return cv2.imread(f)[:, :, ::-1].copy()

    def pitch_roll(self, k):
        """Camera pitch / roll (deg) w.r.t. gravity from the pose."""
        R = self.R[k]
        return (float(np.degrees(np.arcsin(np.clip(R[1, 2], -1, 1)))),
                float(np.degrees(np.arcsin(np.clip(-R[1, 0], -1, 1)))))


def _auc(score, label):
    pos = label > 0.5
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score))
    ranks[order] = np.arange(1, len(score) + 1)
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _f1(pred, gt):
    tp = float((pred & gt).sum())
    fp = float((pred & ~gt).sum())
    fn = float((~pred & gt).sum())
    p = tp / max(tp + fp, 1.0)
    r = tp / max(tp + fn, 1.0)
    return 2 * p * r / max(p + r, 1e-9), p, r


def evaluate(method, episodes, frames=None, stride=1, extra_mask=None):
    """Run a method over episodes and score it. Returns a nested dict."""
    acc = {b: {"s": [], "y": [], "pred": [], "e_err": [], "pred_vis": [], "vis": []} for b in BANDS}
    for ep in episodes:
        ks = range(0, len(ep), stride) if frames is None else frames
        C, res = ep.cells, ep.res
        zc = (np.arange(C) + 0.5) * res
        for k in ks:
            out = method(ep, k)
            vis = ep.vis[k].copy()
            if extra_mask is not None:
                vis &= extra_mask(ep, k)
            occ_gt = ep.occ[k]
            thr = out.get("occ_threshold", 0.5)
            for (lo, hi) in BANDS:
                band = np.zeros((C, C), bool)
                band[:, (zc >= lo) & (zc < hi)] = True
                m = vis & band
                if not m.any():
                    continue
                a = acc[(lo, hi)]
                a["s"].append(out["occ"][m].astype(np.float32))
                a["y"].append(occ_gt[m])
                a["pred"].append(out["occ"][m] >= thr)
                if "elev" in out:
                    me = m & np.isfinite(ep.elev[k])
                    a["e_err"].append(np.abs(out["elev"][me] - ep.elev[k][me]))
                if "vis" in out:
                    a["pred_vis"].append(out["vis"][band] > 0.5)
                    a["vis"].append(ep.vis[k][band])
    res = {}
    for b, a in acc.items():
        if not a["s"]:
            continue
        s, y, p = np.concatenate(a["s"]), np.concatenate(a["y"]), np.concatenate(a["pred"])
        f1, pr, rc = _f1(p, y)
        best = 0.0
        if y.any():
            for q in np.quantile(s, np.linspace(0.5, 0.999, 60)):
                best = max(best, _f1(s >= q, y)[0])
        r = {"n_cells": int(len(y)), "obstacle_rate": float(y.mean()), "auc": _auc(s, y),
             "f1": f1, "precision": pr, "recall": rc, "best_f1": best}
        if a["e_err"]:
            r["elev_mae"] = float(np.concatenate(a["e_err"]).mean())
        if a["vis"]:
            pv, gv = np.concatenate(a["pred_vis"]), np.concatenate(a["vis"])
            r["vis_iou"] = float((pv & gv).sum() / max((pv | gv).sum(), 1))
        res[f"{b[0]:.0f}-{b[1]:.0f}m"] = r
    return res


def format_table(name, res):
    """One printable block per method."""
    lines = [f"== {name}"]
    for band, r in res.items():
        lines.append(f"   {band:>7s}  cells {r['n_cells']:7d} occ {r['obstacle_rate']*100:5.2f}%  "
                     f"AUC {r['auc']:.3f}  bestF1 {r['best_f1']:.3f}  F1 {r['f1']:.3f} "
                     f"(P {r['precision']:.2f} R {r['recall']:.2f})"
                     + (f"  elevMAE {r['elev_mae']:.3f} m" if "elev_mae" in r else ""))
    return "\n".join(lines)

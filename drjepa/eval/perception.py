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
    steep = {"score": [], "true": []}
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
            if "grade" in out and ep.domain == "sim":
                gm = vis & np.isfinite(ep.grade[k])
                gm[:, zc >= 12.0] = False
                steep["score"].append(out["grade"][gm])
                steep["true"].append(ep.grade[k][gm])
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
    if steep["true"]:
        sc, tr = np.concatenate(steep["score"]), np.concatenate(steep["true"])
        st = tr > 0.45
        res["steep"] = {"n_steep": int(st.sum()), "auc": _auc(sc, st),
                        "recall_at_0.46": float((sc[st] > 0.46).mean()) if st.any() else float("nan"),
                        "bias_on_steep": float((sc[st] - tr[st]).mean()) if st.any() else float("nan"),
                        "false_steep_rate": float((sc[~st & (tr < 0.3)] > 0.46).mean())}
    return res


def format_table(name, res):
    """One printable block per method."""
    lines = [f"== {name}"]
    if "steep" in res:
        s = res["steep"]
        lines.append(f"   steep ground (grade > 0.45, <= 12 m): n {s['n_steep']}  AUC {s['auc']:.3f}  "
                     f"recall@0.46 {s['recall_at_0.46']:.2f}  bias {s['bias_on_steep']:+.3f}  "
                     f"false-steep {s['false_steep_rate']:.3f}")
    for band, r in res.items():
        if band == "steep":
            continue
        lines.append(f"   {band:>7s}  cells {r['n_cells']:7d} occ {r['obstacle_rate']*100:5.2f}%  "
                     f"AUC {r['auc']:.3f}  bestF1 {r['best_f1']:.3f}  F1 {r['f1']:.3f} "
                     f"(P {r['precision']:.2f} R {r['recall']:.2f})"
                     + (f"  elevMAE {r['elev_mae']:.3f} m" if "elev_mae" in r else ""))
    return "\n".join(lines)


def main():
    """CLI: score a GeoBEV checkpoint (or the depth teacher) on episodes."""
    import argparse
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default=None, help="GeoBEV checkpoint")
    ap.add_argument("--teacher", default=None, choices=["Small", "Base"],
                    help="score the Depth Anything V2 teacher instead")
    ap.add_argument("--episodes", nargs="+", required=True)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--square_crop", action="store_true",
                    help="only score cells inside the centred square crop's view")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.teacher:
        from ..perception.teacher import DepthTeacher
        t = DepthTeacher(a.teacher)

        def method(ep, k):
            g = t(ep.rgb(k), ep.camera, R=ep.R[k], t=ep.t[k])
            return {"occ": g["score"], "elev": g["elev"], "vis": g["vis"], "occ_threshold": t.step}
        name = f"teacher DA-V2-{a.teacher}"
    else:
        from ..perception.infer import Perceiver
        method = Perceiver(a.checkpoint).benchmark_method()
        name = a.checkpoint
    mask = square_crop_mask if a.square_crop else None
    results = {}
    for e in a.episodes:
        ep = Episode(e)
        r = evaluate(method, [ep], stride=a.stride, extra_mask=mask)
        results[os.path.basename(e.rstrip("/"))] = r
        print(format_table(f"{name} | {os.path.basename(e.rstrip('/'))}", r), flush=True)
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        with open(a.out, "w") as f:
            json.dump({"method": name, "results": results}, f, indent=1)


def square_crop_mask(ep, k):
    """Cells whose ground point projects inside the centred square crop
    (the legacy model's field of view), for like-for-like comparisons."""
    cam, C, res = ep.camera, ep.cells, ep.res
    xs = (np.arange(C) + 0.5) * res - C * res / 2
    zs = (np.arange(C) + 0.5) * res
    X, Z = np.meshgrid(xs, zs, indexing="ij")
    P = np.stack([X, np.nan_to_num(ep.elev[k]), Z], -1).reshape(-1, 3)
    uv, zc = cam.project(P, ep.R[k], ep.t[k])
    s = min(cam.width, cam.height)
    x0, y0 = (cam.width - s) / 2, (cam.height - s) / 2
    ok = (zc > 0.2) & (uv[:, 0] >= x0) & (uv[:, 0] < x0 + s) & (uv[:, 1] >= y0) & (uv[:, 1] < y0 + s)
    return ok.reshape(C, C)


if __name__ == "__main__":
    main()

"""Feature preprocessing and the training dataset.

`preprocess` runs every episode (video + telemetry CSV) through the frozen
DINOv2 backbone ONCE and stores per-frame token features (original and
horizontally-mirrored variants) in a memory-mapped fp16 array. Training then
never touches pixels: it is bottlenecked only by the ~4M-parameter temporal
model, which makes full training runs a matter of minutes.

Works with synthetic data out of the box and with real rover logs, as long
as they are (video, CSV) pairs with the same column layout.
"""

import glob
import os

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from .config import Config
from .simulator import goal_vector

# meta column indices
(DIST, SINB, COSB, SPEED, LTHR, LSTEER, ETHR, ESTEER, DANGER, EP,
 TRUE_X, TRUE_Z) = range(12)
META_COLS = 12

DANGER_LOOKAHEAD = 10  # frames: "danger" = worst traversability soon


def _episode_meta(df, cfg: Config):
    """Per-frame context + labels from a telemetry dataframe -> (N, 12)."""
    n = len(df)
    m = np.zeros((n, META_COLS), dtype=np.float32)
    for i, row in enumerate(df.itertuples(index=False)):
        dist, rel = goal_vector(row.lat, row.lon, row.heading,
                                row.goal_lat, row.goal_lon)
        m[i, DIST] = min(dist / cfg.model.dist_norm, 1.5)
        m[i, SINB] = np.sin(np.radians(rel))
        m[i, COSB] = np.cos(np.radians(rel))
        m[i, SPEED] = row.speed / cfg.model.speed_norm
    m[:, LTHR] = df["throttle"].values
    m[:, LSTEER] = df["steer"].values
    m[:, ETHR] = df.get("exec_throttle", df["throttle"]).values
    m[:, ESTEER] = df.get("exec_steer", df["steer"]).values
    # true pose: only consumed to crop map-completion training targets
    if "true_x" in df.columns:
        m[:, TRUE_X] = df["true_x"].values
        m[:, TRUE_Z] = df["true_z"].values

    trav = df.get("trav_score", pd.Series(np.ones(n))).values.astype(np.float32)
    coll = df.get("collision", pd.Series(np.zeros(n))).values.astype(np.float32)
    # sharp danger: trav_score is clearance/6, so trav < 0.3 means the
    # nearest obstacle surface is closer than ~1.8 m. A soft "any obstacle
    # within 6 m" label saturates (>80% positives) and drowns the signal.
    raw = np.clip((0.3 - trav) / 0.3 + coll, 0.0, 1.0)
    # danger_t = worst raw danger within the lookahead window
    pad = np.concatenate([raw, np.full(DANGER_LOOKAHEAD, raw[-1])])
    m[:, DANGER] = np.max(np.lib.stride_tricks.sliding_window_view(
        pad, DANGER_LOOKAHEAD + 1)[:n], axis=1)
    return m


def preprocess(data_dir, out_dir, cfg: Config = None, batch_size=160,
               device="cuda"):
    """Pack raw episodes into training-ready memory-mapped arrays.

    For every (video, csv) pair found in the comma-separated `data_dir`
    list: decode frames, run the frozen DINOv2 backbone once per frame, and
    store per-frame token grids (fp16), per-frame context/label metadata,
    and the bit-packed occupancy/visibility wedge targets from the episode
    .npz (zeros when absent, e.g. real logs without geometry labels).

    Because the backbone is frozen this is the ONLY time pixels are
    touched; training afterwards runs entirely from these features.
    """
    from .model import Backbone

    cfg = cfg or Config()
    videos = []
    for d in str(data_dir).split(","):
        videos += sorted(glob.glob(os.path.join(d, "*.mp4")) +
                         glob.glob(os.path.join(d, "*.avi")))
    pairs = []
    for v in videos:
        c = os.path.splitext(v)[0] + ".csv"
        if os.path.exists(c):
            pairs.append((v, c))
    if not pairs:
        raise SystemExit(f"No (video, csv) pairs found in {data_dir}")

    counts = [len(pd.read_csv(c, usecols=["timestamp_ms"])) for _, c in pairs]
    total = int(sum(counts))
    nt, fd = cfg.model.n_tokens, cfg.model.feat_dim
    C = cfg.model.wedge_cells
    n_bits = C * C // 8
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "gt"), exist_ok=True)
    feats = np.lib.format.open_memmap(
        os.path.join(out_dir, "feats.npy"), mode="w+",
        dtype=np.float16, shape=(total, nt, fd))
    occ_all = np.zeros((total, n_bits), dtype=np.uint8)
    vis_all = np.zeros((total, n_bits), dtype=np.uint8)
    elev_all = np.zeros((total, C, C), dtype=np.int8)
    sand_all = np.zeros((total, C, C), dtype=np.uint8)
    meta = np.zeros((total, META_COLS), dtype=np.float32)

    backbone = Backbone(cfg.model, device=device)
    write = 0
    print(f"Packing {len(pairs)} episodes / {total} frames -> {out_dir}")
    for ep_id, (vp, cp) in enumerate(tqdm(pairs)):
        df = pd.read_csv(cp)
        m = _episode_meta(df, cfg)
        m[:, EP] = ep_id
        n = len(df)

        wedge_path = os.path.splitext(vp)[0] + ".npz"
        occ_ep = vis_ep = elev_ep = sand_ep = None
        if os.path.exists(wedge_path):
            z = np.load(wedge_path)
            occ_ep, vis_ep = z["occ"], z["vis"]
            elev_ep = z["elev"] if "elev" in z.files else None
            sand_ep = z["sand"] if "sand" in z.files else None
            if "gt_occ" in z.files:
                # episode ground-truth grids feed map-completion training
                np.savez_compressed(
                    os.path.join(out_dir, "gt", f"ep{ep_id}.npz"),
                    occ=z["gt_occ"], hazard=z["gt_hazard"],
                    sand=z["gt_sand"], origin=z["gt_origin"],
                    res=z["gt_res"])

        cap = cv2.VideoCapture(vp)
        buf = []
        got = 0
        while got < n:
            ok, frame = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            buf.append(torch.from_numpy(rgb).permute(2, 0, 1))
            got += 1
            if len(buf) == batch_size or got == n:
                tok = backbone(torch.stack(buf))           # (b, nt, fd)
                b = len(buf)
                feats[write:write + b] = tok.cpu().numpy().astype(np.float16)
                write += b
                buf = []
        cap.release()
        if got < n:  # video shorter than CSV: truncate meta
            m = m[:got]
        meta[write - got:write] = m
        if occ_ep is not None:
            occ_all[write - got:write] = occ_ep[:got]
            vis_all[write - got:write] = vis_ep[:got]
        if elev_ep is not None:
            elev_all[write - got:write] = elev_ep[:got]
            sand_all[write - got:write] = sand_ep[:got]

    feats.flush()
    meta = meta[:write]
    np.save(os.path.join(out_dir, "meta.npy"), meta)
    np.save(os.path.join(out_dir, "occ.npy"), occ_all[:write])
    np.save(os.path.join(out_dir, "vis.npy"), vis_all[:write])
    np.save(os.path.join(out_dir, "elev.npy"), elev_all[:write])
    np.save(os.path.join(out_dir, "sand.npy"), sand_all[:write])
    ep_names = np.array([os.path.basename(v) for v, _ in pairs])
    np.savez(os.path.join(out_dir, "info.npz"), count=write,
             n_tokens=nt, feat_dim=fd, episodes=len(pairs),
             ep_names=ep_names, wedge_cells=cfg.model.wedge_cells)
    print(f"Done: {write} frames, feats {feats.nbytes / 1e9:.2f} GB")


# ==========================================================================
class SeqDataset(Dataset):
    """Sliding windows of precomputed features for RoverJEPA training."""

    def __init__(self, data_dir, cfg: Config, is_val=False):
        """Index sliding windows over a packed dataset.

        Windows never straddle episode boundaries, the train/val split is
        by whole episodes (seeded, deterministic -- no leakage), and DAgger
        episodes get a reduced sampling weight.
        """
        self.cfg = cfg
        mc = cfg.model
        self.S = mc.seq_len
        self.k_max = max(mc.jepa_offsets)
        self.W = self.S + self.k_max
        assert mc.action_horizon <= self.k_max + 1, \
            "action horizon must fit inside the feature window"
        self.is_val = is_val

        self.data_dir = data_dir
        self.feats = np.load(os.path.join(data_dir, "feats.npy"), mmap_mode="r")
        self.meta = np.load(os.path.join(data_dir, "meta.npy"))
        self.occ = np.load(os.path.join(data_dir, "occ.npy"), mmap_mode="r")
        self.vis = np.load(os.path.join(data_dir, "vis.npy"), mmap_mode="r")
        self.elev = np.load(os.path.join(data_dir, "elev.npy"), mmap_mode="r")
        self.sand = np.load(os.path.join(data_dir, "sand.npy"), mmap_mode="r")
        self.cells = cfg.model.wedge_cells
        self._gt_cache = {}   # per-worker LRU-ish cache of episode GT grids
        ep_ids = self.meta[:, EP].astype(np.int64)
        info = np.load(os.path.join(data_dir, "info.npz"), allow_pickle=True)
        ep_names = info["ep_names"] if "ep_names" in info.files else None

        # deterministic episode-level split (no leakage between train/val)
        eps = np.unique(ep_ids)
        rng = np.random.default_rng(42)
        eps = rng.permutation(eps)
        n_val = max(1, int(len(eps) * cfg.train.val_split))
        chosen = set((eps[-n_val:] if is_val else eps[:-n_val]).tolist())

        self.starts = []
        weights = []
        stride = cfg.train.window_stride
        for ep in np.unique(ep_ids):
            if int(ep) not in chosen:
                continue
            # DAgger episodes carry corrective labels for off-policy states;
            # downweight them so they inform without dominating
            w = 1.0
            if ep_names is not None and str(ep_names[int(ep)]).startswith("dag_"):
                w = cfg.train.dagger_weight
            idx = np.flatnonzero(ep_ids == ep)
            for s in range(idx[0], idx[-1] - self.W + 2, stride):
                self.starts.append(s)
                weights.append(w)
        self.starts = np.array(self.starts, dtype=np.int64)
        self.weights = np.array(weights, dtype=np.float64)
        print(f"  {'val' if is_val else 'train'}: {len(chosen)} episodes, "
              f"{len(self.starts)} windows")

    def __len__(self):
        """Number of training windows."""
        return len(self.starts)

    # ---------------- map-completion sample construction ----------------
    def _gt_grids(self, ep):
        """Episode ground-truth grids (occ, hazard, sand, origin), cached."""
        ep = int(ep)
        if ep not in self._gt_cache:
            if len(self._gt_cache) > 64:
                self._gt_cache.pop(next(iter(self._gt_cache)))
            path = os.path.join(self.data_dir, "gt", f"ep{ep}.npz")
            if os.path.exists(path):
                z = np.load(path)
                self._gt_cache[ep] = {k: z[k] for k in
                                      ("occ", "hazard", "sand", "origin")}
            else:
                self._gt_cache[ep] = None
        return self._gt_cache[ep]

    def _completion_sample(self, ep, px, pz, rng):
        """Build one masked-map training sample around true pose (px, pz).

        The observed mask mimics a fused belief map: FOV fans painted along
        a synthetic trajectory that ends at the rover, plus value noise so
        the completer tolerates imperfect real maps. Targets are the full
        ground-truth grids; only hidden cells are graded.
        """
        G = self.cfg.model.comp_cells
        res = self.cfg.model.comp_res
        gt = self._gt_grids(ep)
        comp_in = np.zeros((4, G, G), np.float32)
        comp_tgt = np.zeros((3, G, G), np.float32)
        comp_mask = np.zeros((1, G, G), np.float32)
        if gt is None:
            return comp_in, comp_tgt, comp_mask

        # crop GT around the pose (north-up; [i]=x, [j]=z), edge-padded zero
        half = G // 2
        gi = int((px - gt["origin"][0]) / res) - half
        gj = int((pz - gt["origin"][1]) / res) - half
        tgt = np.zeros((3, G, G), np.float32)
        for c, key in enumerate(("occ", "hazard", "sand")):
            src = gt[key].astype(np.float32)
            if key == "sand":
                src = src / 255.0
            i0, j0 = max(0, gi), max(0, gj)
            i1 = min(src.shape[0], gi + G)
            j1 = min(src.shape[1], gj + G)
            if i1 > i0 and j1 > j0:
                tgt[c, i0 - gi:i1 - gi, j0 - gj:j1 - gj] = src[i0:i1, j0:j1]
        comp_tgt[:] = tgt

        # synthetic observation mask: FOV fans along a random walk that
        # ends at the crop center (where the rover is).
        # grid convention: axis 0 = x (east), axis 1 = z (north)
        obs = np.zeros((G, G), np.float32)
        ii, jj = np.mgrid[0:G, 0:G].astype(np.float32)
        pos = np.array([half, half], np.float32)
        heading = rng.uniform(0, 2 * np.pi)
        n_fans = rng.integers(5, 18)
        step = rng.uniform(3, 9)
        centers = [(pos.copy(), heading)]
        for _ in range(n_fans - 1):
            heading += rng.uniform(-0.8, 0.8)
            # walk backward along the fake trajectory; fans face travel
            pos = pos - step * np.array([np.sin(heading), np.cos(heading)])
            centers.append((pos.copy(), heading))
        for c, hd in centers:
            dx = ii - c[0]
            dz = jj - c[1]
            r = np.sqrt(dx * dx + dz * dz)
            ang = np.arctan2(dx, dz)
            fan_dir = hd + rng.uniform(-0.4, 0.4)
            dang = np.abs((ang - fan_dir + np.pi) % (2 * np.pi) - np.pi)
            obs = np.maximum(obs, ((r < rng.uniform(10, 17)) &
                                   (dang < 0.9)).astype(np.float32))
            obs = np.maximum(obs, (r < 3.0).astype(np.float32))
        comp_mask[0] = 1.0 - obs                 # grade the HIDDEN cells

        # observed input = ground truth + corruption (imperfect perception)
        noisy = tgt.copy()
        flip = rng.random(tgt.shape) < 0.04
        noisy = np.where(flip, 1.0 - noisy, noisy)
        comp_in[:3] = noisy * obs[None]
        comp_in[3] = obs
        return comp_in, comp_tgt, comp_mask

    def __getitem__(self, i):
        """One window as a dict; shapes documented in compute_losses."""
        s0 = self.starts[i]
        S, W, C = self.S, self.W, self.cells
        rng = np.random.default_rng(
            (s0 * 9973 + (0 if self.is_val else np.random.randint(1 << 30)))
            % (1 << 31))

        tokens = torch.from_numpy(
            np.ascontiguousarray(self.feats[s0:s0 + W])).float()
        m = self.meta[s0:s0 + W]

        label = np.stack([m[:, LTHR], m[:, LSTEER]], axis=1)
        execu = np.stack([m[:, ETHR], m[:, ESTEER]], axis=1)
        ctx = np.stack([m[:S, DIST], m[:S, SINB],
                        m[:S, COSB], m[:S, SPEED]], axis=1)
        danger = m[:, DANGER:DANGER + 1]
        dist = m[:, DIST:DIST + 1]
        motion = np.stack([m[:, SPEED], m[:, ESTEER]], axis=1)
        occ = np.unpackbits(self.occ[s0:s0 + W], axis=-1)
        occ = occ.reshape(W, C, C).astype(np.float32)
        vis = np.unpackbits(self.vis[s0:s0 + W], axis=-1)
        vis = vis.reshape(W, C, C).astype(np.float32)
        elev = self.elev[s0:s0 + W].astype(np.float32) * (3.5 / 127.0)
        sand = self.sand[s0:s0 + W].astype(np.float32) / 255.0

        # completion sample anchored at the last context frame's true pose
        anchor = m[S - 1]
        comp_in, comp_tgt, comp_mask = self._completion_sample(
            anchor[EP], anchor[TRUE_X], anchor[TRUE_Z], rng)

        return {"tokens": tokens,
                "label": torch.from_numpy(label.astype(np.float32)),
                "execu": torch.from_numpy(execu.astype(np.float32)),
                "ctx": torch.from_numpy(ctx.astype(np.float32)),
                "danger": torch.from_numpy(danger.astype(np.float32)),
                "dist": torch.from_numpy(dist.astype(np.float32)),
                "occ": torch.from_numpy(occ),
                "vis": torch.from_numpy(vis),
                "elev": torch.from_numpy(elev),
                "sand": torch.from_numpy(sand),
                "motion": torch.from_numpy(motion.astype(np.float32)),
                "comp_in": torch.from_numpy(comp_in),
                "comp_tgt": torch.from_numpy(comp_tgt),
                "comp_mask": torch.from_numpy(comp_mask)}

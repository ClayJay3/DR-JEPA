"""Frame dataset over labelled episodes (sim `ep_*` dirs and real recordings).

Every sample is one camera frame resized to the model input, its intrinsics
scaled to match, its camera -> robot pose, and the robot-frame grid targets.
Augmentations are geometry-consistent: a random zoom/crop changes the
intrinsics exactly (so the lift still projects correctly), and a horizontal
flip mirrors the image, the camera pose and the grid together.
"""

import json
import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

FLIP = np.diag([-1.0, 1.0, 1.0])


def load_episode_meta(d):
    """(camera dict, domain, n_frames, rgb filename pattern) for an episode dir."""
    with open(os.path.join(d, "episode.json")) as f:
        m = json.load(f)
    real = os.path.isdir(os.path.join(d, "depth"))
    return m["camera"], m.get("domain", "real" if real else "sim"), int(m["frames"]), \
        "{:05d}.jpg" if real else "{:04d}.jpg"


def parse_spec(spec):
    """'path' or 'path@a:b' (a, b = fractions of the episode's frames)."""
    if "@" in spec:
        p, r = spec.rsplit("@", 1)
        a, b = (float(x) for x in r.split(":"))
        return p, (a, b)
    return spec, (0.0, 1.0)


class FrameDataset(Dataset):
    """All frames of the given episode directories. A directory may carry a
    fractional frame range ('path@0.0:0.8') for temporal train/val splits
    inside one long real recording."""

    def __init__(self, specs, mcfg, train=True, stride=1, max_frames=None, depth=False):
        self.mcfg, self.train = mcfg, train
        self.with_depth = depth         # auxiliary depth targets at feature resolution
        parsed = [parse_spec(s) for s in specs]
        self.dirs = [p for p, _ in parsed]
        self.meta = [load_episode_meta(d) for d in self.dirs]
        self.index = []
        for e, (_, _, n, _) in enumerate(self.meta):
            a, b = parsed[e][1]
            ks = list(range(int(a * n), int(b * n), stride))
            if max_frames:
                ks = ks[:max_frames]
            self.index += [(e, k) for k in ks]
        self.index = np.array(self.index, np.int64).reshape(-1, 2)
        self.is_real = np.array([self.meta[e][1] == "real" for e, _ in self.index], bool)
        self._cache = {}

    def __len__(self):
        return len(self.index)

    def _labels(self, e):
        if e not in self._cache:
            if len(self._cache) > 48:
                self._cache.pop(next(iter(self._cache)))
            z = np.load(os.path.join(self.dirs[e], "labels.npz"))
            self._cache[e] = {k: z[k] for k in ("vis", "occ", "obs_h", "elev", "grade", "sand", "R", "t")}
            self._cache[e]["cells"] = int(z["cells"])
            # real recordings drop frames without a ground plane: label row
            # k is recording frame frame_ids[k]
            self._cache[e]["ids"] = z["frame_ids"] if "frame_ids" in z.files else None
            dp = os.path.join(self.dirs[e], "depth.npy")
            self._cache[e]["depth"] = np.load(dp, mmap_mode="r") if os.path.exists(dp) else None
        return self._cache[e]

    def __getitem__(self, i):
        e, k = self.index[i]
        cam, dom, _, pat = self.meta[e]
        L = self._labels(e)
        C = L["cells"]
        fid = k if L["ids"] is None else int(L["ids"][k])
        img = cv2.cvtColor(cv2.imread(os.path.join(self.dirs[e], "rgb", pat.format(fid))), cv2.COLOR_BGR2RGB)
        H0, W0 = img.shape[:2]
        fx, fy, cx, cy = cam["fx"] * W0 / cam["width"], cam["fy"] * H0 / cam["height"], \
            cam["cx"] * W0 / cam["width"], cam["cy"] * H0 / cam["height"]
        R = L["R"][k].astype(np.float64)
        t = L["t"][k].astype(np.float64)
        unpack = lambda a: np.unpackbits(a[k])[:C * C].reshape(C, C).astype(np.float32)
        y = {"vis": unpack(L["vis"]), "occ": unpack(L["occ"]),
             "obs_h": L["obs_h"][k].astype(np.float32), "elev": L["elev"][k].astype(np.float32),
             "grade": L["grade"][k].astype(np.float32),
             "sand": L["sand"][k].astype(np.float32) / 255.0}
        rng = np.random.default_rng() if self.train else None
        x0, y0, cw, ch = 0, 0, W0, H0
        if self.train and rng.random() < 0.7:
            # random zoom-crop: keep 72-100 % of each side (changes the FOV the
            # network sees -- the geometry follows exactly)
            s = rng.uniform(0.72, 1.0)
            a = rng.uniform(0.9, 1.1)
            cw = int(min(W0, W0 * s * a))
            ch = int(min(H0, H0 * s / a))
            x0 = int(rng.integers(0, W0 - cw + 1))
            y0 = int(rng.integers(0, H0 - ch + 1))
            img = img[y0:y0 + ch, x0:x0 + cw]
        Wi, Hi = self.mcfg.input_w, self.mcfg.input_h
        img = cv2.resize(img, (Wi, Hi), interpolation=cv2.INTER_AREA)
        sx, sy = Wi / cw, Hi / ch
        K = np.array([[fx * sx, 0, (cx - x0) * sx], [0, fy * sy, (cy - y0) * sy], [0, 0, 1]])
        dep = self._depth(e, fid, k, L, (W0, H0), (x0, y0, cw, ch)) if self.with_depth else None
        if self.train:
            if rng.random() < 0.5:                          # horizontal mirror
                img = img[:, ::-1]
                K[0, 2] = Wi - K[0, 2]
                R = FLIP @ R @ FLIP
                t = FLIP @ t
                y = {kk: v[::-1].copy() for kk, v in y.items()}
                if dep is not None:
                    dep = dep[:, ::-1].copy()
            img = self._photometric(np.ascontiguousarray(img), rng)
        if dep is not None:
            y["depth"] = dep
        out = {"img": torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1),
               "K": torch.tensor(K, dtype=torch.float32),
               "R": torch.tensor(R, dtype=torch.float32),
               "t": torch.tensor(t, dtype=torch.float32),
               "is_real": torch.tensor(float(dom == "real"))}
        out.update({kk: torch.from_numpy(np.ascontiguousarray(v)) for kk, v in y.items()})
        return out

    def _depth(self, e, fid, k, L, wh0, crop):
        """Depth target (Hi/8, Wi/8) in metres for the same crop (0 = none)."""
        W0, H0 = wh0
        x0, y0, cw, ch = crop
        if L["depth"] is not None:                          # sim: (N, H/4, W/4)
            d = np.asarray(L["depth"][k], np.float32)
        else:                                               # real: stereo png + confidence
            d = cv2.imread(os.path.join(self.dirs[e], "depth", f"{fid:05d}.png"), -1)
            if d is None:
                return np.zeros((self.mcfg.input_h // 8, self.mcfg.input_w // 8), np.float32)
            d = d.astype(np.float32) / 1000.0
            cf = cv2.imread(os.path.join(self.dirs[e], "conf", f"{fid:05d}.png"), -1)
            d[(cf >= 40) | (d < 0.3) | (d > 20.0)] = 0.0
        sx, sy = d.shape[1] / W0, d.shape[0] / H0
        d = d[int(y0 * sy):int((y0 + ch) * sy), int(x0 * sx):int((x0 + cw) * sx)]
        return cv2.resize(d, (self.mcfg.input_w // 8, self.mcfg.input_h // 8),
                          interpolation=cv2.INTER_NEAREST)

    @staticmethod
    def _photometric(img, rng):
        """Colour / blur / noise / compression jitter (uint8 RGB in, out)."""
        f = img.astype(np.float32)
        if rng.random() < 0.8:
            f = f * rng.uniform(0.7, 1.3) + rng.uniform(-20, 20)           # exposure
            m = f.mean()
            f = (f - m) * rng.uniform(0.75, 1.3) + m                        # contrast
            f = f * rng.uniform(0.9, 1.1, 3)                                # white balance
            g = f.mean(-1, keepdims=True)
            f = g + (f - g) * rng.uniform(0.5, 1.4)                         # saturation
        if rng.random() < 0.15:
            f = np.repeat(f.mean(-1, keepdims=True), 3, -1)
        f = np.clip(f, 0, 255)
        if rng.random() < 0.3:
            f = (np.clip(f / 255.0, 0, 1) ** rng.uniform(0.75, 1.35)) * 255.0
        if rng.random() < 0.25:
            k = int(rng.choice([3, 5]))
            f = cv2.GaussianBlur(f, (k, k), 0)
        if rng.random() < 0.4:
            f = f + rng.normal(0, rng.uniform(1, 6), f.shape)
        img = np.clip(f, 0, 255).astype(np.uint8)
        if rng.random() < 0.3:
            _, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(35, 90))])
            img = cv2.imdecode(enc, cv2.IMREAD_UNCHANGED)
        return img

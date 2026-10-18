"""Engineering dashboard: everything the navigator perceives, believes,
decides and measures, on one 1920 x 1080 frame.

    camera + AR overlay    predicted obstacle voxels, route, chosen arc,
                           range rings and the IMU horizon projected through
                           the real camera model
    perception heads       all six network outputs for this frame
    vs ground truth        truth, hit / false-alarm / miss map, error maps
                           (sim: exact; real recordings: stereo labels)
    belief map             fused memory: hillshaded terrain, obstacles,
                           slopes, sand, field of view, trail, route and
                           all candidate arcs scored by the controller
    accuracy / control     live metrics, binding speed cap, mission stats
    telemetry / latency    20 s history of speed / throttle / steer and
                           per-stage step latency

Used by ../../dashboard.py (sim or real-recording replay).
"""

import collections
import math
import os

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ..geo import world_to_robot

# ------------------------------------------------------------------ theme
BG = (11, 15, 20)
PANEL = (18, 24, 33)
BORDER = (36, 47, 61)
GRIDC = (30, 40, 52)
TEXT = (232, 238, 245)
MUTED = (139, 152, 168)
DIM = (88, 101, 118)
CYAN = (60, 200, 230)
TEAL = (43, 217, 168)
AMBER = (245, 179, 66)
RED = (255, 90, 95)
VIOLET = (157, 124, 255)
SAND = (232, 199, 122)
WHITE = (255, 255, 255)
MODE_COL = {"drive": TEAL, "creep": AMBER, "reverse": RED, "escape": VIOLET}

_FONT_DIR = os.path.join(os.path.dirname(__file__), "fonts")
_FONT_FILES = {"regular": "Inter-Regular.ttf", "medium": "Inter-Medium.ttf",
               "semibold": "Inter-SemiBold.ttf", "mono": "Hack-Regular.ttf"}
_fonts = {}


def font(weight, size):
    key = (weight, size)
    if key not in _fonts:
        try:
            _fonts[key] = ImageFont.truetype(os.path.join(_FONT_DIR, _FONT_FILES[weight]), size)
        except OSError:
            _fonts[key] = ImageFont.load_default(size)
    return _fonts[key]


def _lut(stops):
    """Piecewise-linear 256-entry RGB colormap from (pos, rgb) stops."""
    xs = np.linspace(0, 1, 256)
    pos = np.array([p for p, _ in stops])
    cols = np.array([c for _, c in stops], float)
    return np.stack([np.interp(xs, pos, cols[:, k]) for k in range(3)], 1).astype(np.uint8)


def _cv_lut(cmap):
    lut = cv2.applyColorMap(np.arange(256, dtype=np.uint8)[:, None], cmap)[:, 0, ::-1]
    return np.ascontiguousarray(lut)


CMAPS = {
    "inferno": _cv_lut(cv2.COLORMAP_INFERNO),
    "viridis": _cv_lut(cv2.COLORMAP_VIRIDIS),
    "magma": _cv_lut(cv2.COLORMAP_MAGMA),
    "cividis": _cv_lut(cv2.COLORMAP_CIVIDIS),
    "gray": _lut([(0, (20, 26, 34)), (1, (220, 228, 236))]),
    "elev": _lut([(0, (40, 90, 200)), (0.5, (228, 232, 236)), (1, (170, 100, 40))]),
    "err": _lut([(0, (20, 26, 34)), (0.4, (120, 70, 160)), (1, (255, 120, 90))]),
}


# ------------------------------------------------------------------ canvas
class Canvas:
    """Numpy RGB canvas with cv2 shapes and deferred PIL text."""

    def __init__(self, w, h):
        self.w, self.h = w, h
        self.img = np.zeros((h, w, 3), np.uint8)
        self.img[:] = BG
        self.texts = []

    def text(self, xy, s, size=14, color=TEXT, weight="regular", anchor="la"):
        self.texts.append((xy, str(s), size, color, weight, anchor))

    def panel(self, x, y, w, h, title=None, right=None):
        cv2.rectangle(self.img, (x, y), (x + w, y + h), PANEL, -1)
        cv2.rectangle(self.img, (x, y), (x + w, y + h), BORDER, 1, cv2.LINE_AA)
        if title:
            self.text((x + 12, y + 10), title.upper(), 11, MUTED, "semibold")
        if right:
            self.text((x + w - 12, y + 10), right, 11, DIM, "medium", "ra")

    def blit(self, img, x, y):
        h, w = img.shape[:2]
        self.img[y:y + h, x:x + w] = img

    def finish(self):
        pil = Image.fromarray(self.img)
        d = ImageDraw.Draw(pil)
        for xy, s, size, col, weight, anchor in self.texts:
            d.text(xy, s, fill=col, font=font(weight, size), anchor=anchor)
        return np.asarray(pil)


def heat(grid, cmap, vmin, vmax, mask=None, size=152, flip=True):
    """Robot-frame grid [i = x, j = z] -> forward-up RGB tile."""
    g = np.nan_to_num(np.asarray(grid, np.float32), nan=vmin)
    v = np.clip((g - vmin) / max(vmax - vmin, 1e-9), 0, 1)
    rgb = CMAPS[cmap][(v * 255).astype(np.uint8)]
    if mask is not None:
        rgb = np.where(mask[..., None], rgb, (rgb * 0.18 + np.array(PANEL) * 0.82).astype(np.uint8))
    if flip:
        rgb = np.flipud(rgb.transpose(1, 0, 2))
    return cv2.resize(np.ascontiguousarray(rgb), (size, size), interpolation=cv2.INTER_NEAREST)


def colorbar(c, x, y, w, cmap, lo, hi, unit=""):
    bar = np.repeat(CMAPS[cmap][None, np.linspace(0, 255, w).astype(int)], 7, 0)
    c.blit(bar, x, y)
    c.text((x, y + 10), f"{lo:g}{unit}", 10, DIM, "mono")
    c.text((x + w, y + 10), f"{hi:g}{unit}", 10, DIM, "mono", "ra")


def plot_lines(c, x, y, w, h, series, ylim, colors, labels, grid_y=None):
    """Small line chart: series = list of 1-D arrays (oldest first)."""
    lo, hi = ylim
    for gy in (grid_y or []):
        py = int(y + h - (gy - lo) / (hi - lo) * h)
        cv2.line(c.img, (x, py), (x + w, py), GRIDC, 1, cv2.LINE_AA)
        c.text((x - 6, py - 6), f"{gy:g}", 10, DIM, "mono", "ra")
    for s, col in zip(series, colors):
        s = np.asarray(s, float)
        if len(s) < 2:
            continue
        xs = x + np.linspace(0, w, len(s))
        ys = y + h - (np.clip(s, lo, hi) - lo) / (hi - lo) * h
        pts = np.stack([xs, ys], 1).astype(np.int32)
        cv2.polylines(c.img, [pts], False, col, 2, cv2.LINE_AA)
    lx = x + w
    for lab, col, s in reversed(list(zip(labels, colors, series))):
        txt = lab + (f"  {s[-1]:+.2f}" if len(s) else "")
        tw = int(font("medium", 11).getlength(txt))
        lx -= tw
        c.text((lx, y - 26), txt, 11, MUTED, "medium")
        cv2.line(c.img, (lx - 20, y - 19), (lx - 6, y - 19), col, 2, cv2.LINE_AA)
        lx -= 40


# ------------------------------------------------------------------ dashboard
class Dashboard:
    """Stateful renderer bound to one Navigator (keeps history for plots)."""

    W, H = 1920, 1080

    def __init__(self, nav, source="SIM"):
        self.nav = nav
        self.source = source
        self.hist = collections.deque(maxlen=200)
        self.lat = collections.deque(maxlen=200)
        self.trail = collections.deque(maxlen=3000)
        self.conf = np.zeros(3)                 # running TP / FP / FN (<= 12 m)
        self.e_err = [0.0, 0]
        self.g_err = [0.0, 0]
        self.t0 = None

    # ------------------------------------------------------------ entry
    def render(self, rgb, cmd, sensors, R, t, gt=None, info=None):
        """One dashboard frame (RGB uint8, 1920 x 1080)."""
        nav = self.nav
        self.trail.append(nav.pose.copy())
        self.hist.append((float(sensors.get("speed", 0.0)), cmd.throttle, cmd.steer))
        self.lat.append({k: 1000 * v for k, v in nav.last_timing.items()})
        c = Canvas(self.W, self.H)
        self._header(c, info or {})
        self._camera(c, rgb, R, t, nav.camera if nav.camera else None, sensors)
        self._heads(c, gt)
        self._belief(c)
        self._metrics(c, gt, info or {})
        self._telemetry(c)
        self._latency(c)
        return c.finish()

    # ------------------------------------------------------------ header
    def _header(self, c, info):
        nav = self.nav
        cv2.rectangle(c.img, (0, 0), (self.W, 56), (14, 19, 26), -1)
        cv2.line(c.img, (0, 56), (self.W, 56), BORDER, 1)
        c.text((20, 15), "DR-JEPA", 22, TEXT, "semibold")
        c.text((128, 21), "navigation diagnostics", 15, MUTED, "regular")
        mode = nav.controller.mode
        chips = [(mode.upper(), MODE_COL.get(mode, TEAL)), (self.source, CYAN)]
        if info.get("scenario"):
            chips.append((info["scenario"].upper(), MUTED))
        x = 360
        for label, col in chips:
            tw = int(font("semibold", 12).getlength(label)) + 22
            cv2.rectangle(c.img, (x, 16), (x + tw, 40), tuple(int(v * 0.22) for v in col), -1)
            cv2.rectangle(c.img, (x, 16), (x + tw, 40), col, 1, cv2.LINE_AA)
            c.text((x + tw // 2, 28), label, 12, col, "semibold", "mm")
            x += tw + 10
        lat = sum(nav.last_timing.values()) * 1000 if nav.last_timing else 0.0
        right = [f"step {nav.step_i}", f"t {nav.step_i * nav.cfg.dt:.1f} s",
                 f"network {1 / max(nav.last_timing.get('perception', 1e-3), 1e-3):.0f} Hz",
                 f"loop {lat:.1f} ms"]
        x = self.W - 20
        for s in reversed(right):
            c.text((x, 20), s, 13, MUTED, "mono", "ra")
            x -= int(font("mono", 13).getlength(s)) + 26

    # ------------------------------------------------------------ camera
    def _camera(self, c, rgb, R, t, cam, sensors):
        x0, y0, pw, ph = 16, 68, 1024, 572
        cs = cam
        rig = (f"{cs.width}x{cs.height}  HFOV {cs.hfov_deg:.0f}°  h {cs.mount_height:.2f} m  "
               f"pitch {math.degrees(math.asin(np.clip(R[1, 2], -1, 1))):+.1f}°  "
               f"roll {math.degrees(math.asin(np.clip(-R[1, 0], -1, 1))):+.1f}°") if cs else ""
        c.panel(x0, y0, pw, ph, "camera · AR overlay", rig)
        aw, ah = pw - 16, ph - 64             # leave a legend strip below the image
        H0, W0 = rgb.shape[:2]
        s = min(aw / W0, ah / H0)
        dw, dh = int(W0 * s), int(H0 * s)
        ox, oy = x0 + 8 + (aw - dw) // 2, y0 + 32 + (ah - dh) // 2
        img = cv2.resize(rgb, (dw, dh), interpolation=cv2.INTER_AREA)
        self._labels = []
        if cs is not None:
            img = self._ar(img, s * W0 / cs.width, cs, R, t)
        c.blit(img, ox, oy)
        for (u, v), lab in self._labels:
            c.text((ox + int(u), oy + int(v)), lab, 11, (210, 218, 228), "medium")
        cv2.rectangle(c.img, (ox - 1, oy - 1), (ox + dw, oy + dh), BORDER, 1)
        # legend
        lx, ly = x0 + 14, y0 + ph - 24
        for lab, col in (("obstacle voxels", RED), ("route", CYAN), ("chosen arc", WHITE),
                         ("range rings 5 m", MUTED), ("IMU horizon", AMBER)):
            cv2.circle(c.img, (lx + 5, ly + 7), 4, col, -1, cv2.LINE_AA)
            c.text((lx + 14, ly), lab, 11, MUTED)
            lx += int(font("regular", 11).getlength(lab)) + 34

    def _ar(self, img, scale, cam, R, t):
        """Project robot-frame geometry into the (scaled) camera image."""
        nav = self.nav
        g = nav.grids
        over = img.copy()
        self._labels = []

        def proj(P):
            uv, z = cam.project(np.asarray(P, float), R, t)
            return uv * scale, z

        # range rings on the ground (dashed)
        placed = []
        for r in (5, 10, 15, 20):
            a = np.radians(np.linspace(-70, 70, 61))
            P = np.stack([r * np.sin(a), np.zeros_like(a), r * np.cos(a)], 1)
            uv, z = proj(P)
            for k in range(0, len(a) - 1, 2):
                if z[k] > 0.3 and z[k + 1] > 0.3:
                    cv2.line(over, tuple(uv[k].astype(int)), tuple(uv[k + 1].astype(int)),
                             (150, 160, 175), 1, cv2.LINE_AA)
            k = int(len(a) * 0.72)
            p = uv[k] + [6, -14]
            if (z[k] > 0.3 and 0 <= p[0] < img.shape[1] - 30 and 0 <= p[1] < img.shape[0] - 14
                    and all(np.hypot(*(p - q)) > 22 for q in placed)):
                placed.append(p)
                self._labels.append((p, f"{r} m"))
        # IMU horizon: directions parallel to the ground plane at camera height
        a = np.radians(np.linspace(-80, 80, 9))
        P = np.stack([1e4 * np.sin(a), np.full_like(a, t[1]), 1e4 * np.cos(a)], 1)
        uv, z = proj(P)
        ok = z > 0
        if ok.sum() >= 2:
            pts = uv[ok].astype(np.int32)
            cv2.polylines(over, [pts], False, AMBER, 1, cv2.LINE_AA)
        if g is not None:
            # obstacle voxels: predicted-occupied visible cells as 3D boxes
            res = nav.grid.res
            occ = (g["occ"] > 0.5) & (g["vis"] > 0.5)
            occ[:, int(20 / res):] = False
            ii, jj = np.nonzero(occ)
            boxes = []
            for i, j in zip(ii, jj):
                xa = i * res - nav.grid.half
                za = j * res
                e = float(g["elev"][i, j])
                hgt = max(float(g["obs_h"][i, j]), 0.3)
                foot = np.array([[xa, e, za], [xa + res, e, za], [xa + res, e, za + res], [xa, e, za + res]])
                top = foot + [0, hgt, 0]
                uvf, zf = proj(foot)
                uvt, zt = proj(top)
                if (zf < 0.4).any() or (zt < 0.4).any():
                    continue
                boxes.append((float(zf.mean()), uvf, uvt, float(g["occ"][i, j])))
            boxes.sort(key=lambda b: -b[0])
            for _, uvf, uvt, p in boxes:
                col = tuple(int(v) for v in np.array(RED) * (0.55 + 0.45 * p))
                cv2.fillPoly(over, [uvf.astype(np.int32)], col, cv2.LINE_AA)
                cv2.fillPoly(over, [np.concatenate([uvf[[0, 1]], uvt[[1, 0]]]).astype(np.int32)],
                             tuple(int(v * 0.8) for v in col), cv2.LINE_AA)
                cv2.polylines(over, [uvt.astype(np.int32)], True, (255, 170, 170), 1, cv2.LINE_AA)
        img = cv2.addWeighted(over, 0.62, img, 0.38, 0)
        # route + chosen arc (drawn crisp on top)
        for pts_w, col, th in ((nav.planner.path, CYAN, 3), (nav.controller.last_arc, WHITE, 2)):
            if pts_w is None or len(pts_w) < 2:
                continue
            dx, dz = pts_w[:, 0] - nav.pose[0], pts_w[:, 1] - nav.pose[1]
            xr, zr = world_to_robot(dx, dz, nav.heading)
            keep = (zr > 0.5) & (zr < 40)
            if keep.sum() < 2:
                continue
            P = np.stack([xr[keep], np.zeros(keep.sum()), zr[keep]], 1)
            uv, z = proj(P)
            ok = z > 0.4
            if ok.sum() >= 2:
                pts = uv[ok].astype(np.int32)
                if col == CYAN:
                    cv2.polylines(img, [pts], False, (10, 60, 80), th + 4, cv2.LINE_AA)
                cv2.polylines(img, [pts], False, col, th, cv2.LINE_AA)
        return img

    # ------------------------------------------------------------ heads
    def _heads(self, c, gt):
        nav = self.nav
        g = nav.grids
        x0, y0, pw, ph = 16, 652, 1024, 412
        c.panel(x0, y0, pw, ph, "perception heads · this frame",
                "robot frame · 24 × 24 m · forward ↑")
        if g is None:
            return
        rb = nav.robot
        vis = g["vis"] > 0.5
        tiles = [("P(obstacle) · where visible", g["occ"], "inferno", 0, 1, "", vis),
                 ("obstacle height", g["obs_h"], "viridis", 0, 1.5, " m", vis),
                 ("terrain elevation", g["elev"], "elev", -1, 1, " m", vis),
                 ("slope grade", g["grade"], "magma", 0, round(rb.tip_roll * 1.2, 2), "", vis),
                 ("soft ground", g["sand"], "cividis", 0, 1, "", vis),
                 ("visibility", g["vis"], "gray", 0, 1, "", None)]
        sz, gap = 140, 26
        tx = x0 + 14
        for k, (name, grid, cmap, lo, hi, unit, mask) in enumerate(tiles):
            xx = tx + k * (sz + gap)
            c.text((xx, y0 + 30), name, 12, TEXT, "medium")
            c.blit(heat(grid, cmap, lo, hi, mask, sz), xx, y0 + 48)
            colorbar(c, xx, y0 + 48 + sz + 6, sz, cmap, lo, hi, unit)
        yy = y0 + 48 + sz + 36
        if gt is None:
            c.text((tx, yy + 50), "no ground truth for this source", 13, DIM)
            return
        v = gt["vis"]
        occ_p = g["occ"] > 0.5
        conf = np.zeros(gt["occ"].shape + (3,), np.uint8)
        conf[:] = (26, 33, 44)
        conf[v] = (40, 52, 66)
        conf[v & occ_p & gt["occ"]] = TEAL
        conf[v & occ_p & ~gt["occ"]] = RED
        conf[v & ~occ_p & gt["occ"]] = AMBER
        conf = cv2.resize(np.ascontiguousarray(np.flipud(conf.transpose(1, 0, 2))), (sz, sz),
                          interpolation=cv2.INTER_NEAREST)
        ge = np.nan_to_num(gt["elev"], nan=0.0)
        egood = v & np.isfinite(gt["elev"])
        rows = [("truth: obstacles", heat(gt["occ"].astype(float), "inferno", 0, 1, v, sz), None),
                ("hit / false alarm / miss", conf, "conf"),
                ("truth: elevation", heat(ge, "elev", -1, 1, egood, sz), ("elev", -1, 1, " m")),
                ("|elevation error|", heat(np.abs(g["elev"] - ge), "err", 0, 0.5, egood, sz), ("err", 0, 0.5, " m")),
                ("truth: slope", heat(np.nan_to_num(gt["grade"]), "magma", 0, round(rb.tip_roll * 1.2, 2),
                                      v & np.isfinite(gt["grade"]), sz), None),
                ("|slope error|", heat(np.abs(g["grade"] - np.nan_to_num(gt["grade"])), "err", 0, 0.3,
                                       v & np.isfinite(gt["grade"]), sz), ("err", 0, 0.3, ""))]
        sz2 = sz
        for k, (name, tile, bar) in enumerate(rows):
            xx = tx + k * (sz + gap)
            c.text((xx, yy), name, 12, TEXT, "medium")
            c.blit(cv2.resize(tile, (sz, sz2), interpolation=cv2.INTER_NEAREST), xx, yy + 18)
            if bar == "conf":
                lx = xx
                for lab, col in (("hit", TEAL), ("false", RED), ("miss", AMBER)):
                    cv2.rectangle(c.img, (lx, yy + 18 + sz2 + 8), (lx + 8, yy + 18 + sz2 + 16), col, -1)
                    c.text((lx + 12, yy + 18 + sz2 + 5), lab, 10, MUTED)
                    lx += 50
            elif bar:
                colorbar(c, xx, yy + 18 + sz2 + 6, sz, bar[0], bar[1], bar[2], bar[3])

    # ------------------------------------------------------------ belief map
    def _belief(self, c):
        nav = self.nav
        bm = nav.map
        x0, y0, pw, ph = 1056, 68, 540, 604
        c.panel(x0, y0, pw, ph, "belief map · fused memory", "north ↑ · 40 m")
        if bm.corner is None:
            return
        size = 516
        span = 40.0
        r = bm.cfg.res
        n = int(span / r)
        ci, cj = bm.index(nav.pose[0], nav.pose[1])
        i0, j0 = int(ci) - n // 2, int(cj) - n // 2
        N = bm.cfg.cells
        sl_i = np.clip(np.arange(i0, i0 + n), 0, N - 1)
        sl_j = np.clip(np.arange(j0, j0 + n), 0, N - 1)
        L = bm.L[np.ix_(sl_i, sl_j)]
        E, EW = bm.E[np.ix_(sl_i, sl_j)], bm.EW[np.ix_(sl_i, sl_j)]
        G, GW = bm.G[np.ix_(sl_i, sl_j)], bm.GW[np.ix_(sl_i, sl_j)]
        S = 1 / (1 + np.exp(-bm.LS[np.ix_(sl_i, sl_j)]))
        known = (np.abs(L) > 0.4) | (EW > 1.0)
        prob = 1 / (1 + np.exp(-L))
        # hillshade from fused elevation
        ex, ez = np.gradient(E, r)
        shade = np.clip(0.75 - 0.9 * (ex * 0.6 + ez * 0.8), 0.35, 1.15)
        base = np.array((44, 56, 70), float) * shade[..., None]
        img = np.where(known[..., None], base, np.array((16, 22, 30), float))
        rb = nav.robot
        steep = known & (GW > 0.3) & (G > rb.grade_drive)
        a = np.clip((G - rb.grade_drive) / max(rb.tip_roll - rb.grade_drive, 1e-3), 0, 1)[..., None]
        img = np.where(steep[..., None], img * (1 - a) + np.array(VIOLET) * a, img)
        sandy = known & (S > 0.5)
        img = np.where(sandy[..., None], img * 0.6 + np.array(SAND) * 0.4, img)
        occ = known & (prob > 0.5)
        a = ((prob - 0.5) * 2)[..., None]
        img = np.where(occ[..., None], img * (1 - a) + np.array(RED) * a, img)
        img = np.flipud(img.transpose(1, 0, 2)).astype(np.uint8)
        img = cv2.resize(np.ascontiguousarray(img), (size, size), interpolation=cv2.INTER_NEAREST)
        s = size / n

        def px(w):
            w = np.asarray(w, float).reshape(-1, 2)
            return np.stack([((w[:, 0] - bm.corner[0]) / r - i0) * s,
                             size - ((w[:, 1] - bm.corner[1]) / r - j0) * s], 1)

        over = img.copy()
        # field-of-view wedge of the camera
        if nav.camera is not None:
            hf = math.radians(nav.camera.hfov_deg / 2)
            hd = math.radians(nav.heading)
            pts = [nav.pose] + [nav.pose + 24 * np.array([math.sin(hd + a), math.cos(hd + a)])
                                for a in np.linspace(-hf, hf, 12)]
            cv2.fillPoly(over, [px(pts).astype(np.int32)], (40, 120, 150), cv2.LINE_AA)
        img = cv2.addWeighted(over, 0.25, img, 0.75, 0)
        # trail
        if len(self.trail) > 1:
            cv2.polylines(img, [px(np.array(self.trail)).astype(np.int32)], False, (120, 130, 145), 1, cv2.LINE_AA)
        # candidate arcs coloured by score
        dbg = nav.controller.debug
        if "arcs" in dbg:
            arcs, feas = dbg["arcs"], dbg["feasible"]
            sc = dbg.get("score")
            if sc is not None:
                f = sc[feas]
                lo, hi = (f.min(), f.max()) if f.size else (0, 1)
            for k in range(len(arcs)):
                pts = px(np.vstack([nav.pose, arcs[k]])).astype(np.int32)
                if not feas[k]:
                    cv2.polylines(img, [pts], False, (110, 45, 55), 1, cv2.LINE_AA)
                else:
                    q = 0.5 if sc is None else (sc[k] - lo) / max(hi - lo, 1e-6)
                    col = tuple(int(v) for v in np.array((30, 90, 80)) * (1 - q) + np.array(TEAL) * q)
                    cv2.polylines(img, [pts], False, col, 1, cv2.LINE_AA)
        path = nav.planner.path
        if path is not None and len(path) > 1:
            pp = px(path).astype(np.int32)
            cv2.polylines(img, [pp], False, (10, 60, 80), 6, cv2.LINE_AA)
            cv2.polylines(img, [pp], False, CYAN, 2, cv2.LINE_AA)
        if nav.controller.last_arc is not None:
            cv2.polylines(img, [px(np.vstack([nav.pose, nav.controller.last_arc])).astype(np.int32)],
                          False, WHITE, 2, cv2.LINE_AA)
        # robot
        hd = math.radians(nav.heading)
        f = np.array([math.sin(hd), math.cos(hd)])
        rt = np.array([math.cos(hd), -math.sin(hd)])
        tri = px([nav.pose + f * 1.6, nav.pose - f * 0.9 + rt * 0.9, nav.pose - f * 0.9 - rt * 0.9]).astype(np.int32)
        cv2.fillPoly(img, [tri], WHITE, cv2.LINE_AA)
        # goal (ring if inside, edge arrow otherwise)
        gp = px(nav.goal)[0]
        gd = float(np.linalg.norm(nav.goal - nav.pose))
        if 0 <= gp[0] < size and 0 <= gp[1] < size:
            cv2.circle(img, tuple(gp.astype(int)), 9, AMBER, 2, cv2.LINE_AA)
        else:
            ctr = np.array([size / 2, size / 2])
            d = (gp - ctr) / max(np.linalg.norm(gp - ctr), 1e-6)
            tip = ctr + d * (size / 2 - 14)
            cv2.arrowedLine(img, tuple((tip - d * 26).astype(int)), tuple(tip.astype(int)), AMBER, 2,
                            cv2.LINE_AA, tipLength=0.45)
        mx, my = x0 + 12, y0 + 32
        c.blit(img, mx, my)
        cv2.rectangle(c.img, (mx - 1, my - 1), (mx + size, my + size), BORDER, 1)
        c.text((mx + size - 8, my + 8), f"goal {gd:5.1f} m", 12, AMBER, "mono", "ra")
        # scale bar
        bx, by = mx + 14, my + size - 18
        cv2.line(c.img, (bx, by), (bx + int(10 / r * s), by), TEXT, 2, cv2.LINE_AA)
        c.text((bx, by - 18), "10 m", 11, TEXT, "medium")
        lx, ly = x0 + 14, y0 + ph - 24
        for lab, col in (("obstacle", RED), ("steep", VIOLET), ("sand", SAND), ("route", CYAN),
                         ("arcs (score)", TEAL), ("blocked arc", (170, 60, 70))):
            cv2.circle(c.img, (lx + 5, ly + 7), 4, col, -1, cv2.LINE_AA)
            c.text((lx + 14, ly), lab, 11, MUTED)
            lx += int(font("regular", 11).getlength(lab)) + 30

    # ------------------------------------------------------------ metrics
    def _metrics(self, c, gt, info):
        nav = self.nav
        x0, y0, pw, ph = 1608, 68, 296, 604
        c.panel(x0, y0, pw, ph, "accuracy · control · mission")
        y = y0 + 34

        def row(label, val, col=TEXT, unit=""):
            nonlocal y
            c.text((x0 + 14, y), label, 12, MUTED)
            c.text((x0 + pw - 14, y), f"{val}{unit}", 13, col, "mono", "ra")
            y += 21

        def section(name):
            nonlocal y
            y += 6
            c.text((x0 + 14, y), name, 11, CYAN, "semibold")
            cv2.line(c.img, (x0 + 14, y + 18), (x0 + pw - 14, y + 18), GRIDC, 1)
            y += 26

        section("PERCEPTION VS TRUTH (≤ 12 m)")
        g = nav.grids
        if gt is not None and g is not None:
            C = gt["occ"].shape[1]
            near = np.zeros_like(gt["vis"])
            near[:, :int(12 / nav.grid.res)] = True
            v = gt["vis"] & near
            p, y_ = (g["occ"] > 0.5) & v, gt["occ"] & v
            tp, fp, fn = float((p & y_).sum()), float((p & ~y_).sum()), float((~p & y_).sum())
            self.conf += (tp, fp, fn)
            T, Fp, Fn = self.conf
            iou = tp / max(tp + fp + fn, 1)
            row("obstacle IoU (frame / run)", f"{iou:.2f} / {T / max(T + Fp + Fn, 1):.2f}")
            row("precision (run)", f"{T / max(T + Fp, 1):.2f}")
            row("recall (run)", f"{T / max(T + Fn, 1):.2f}")
            ev = v & np.isfinite(gt["elev"])
            if ev.any():
                self.e_err[0] += float(np.abs(g["elev"] - gt["elev"])[ev].sum())
                self.e_err[1] += int(ev.sum())
            gv = v & np.isfinite(gt["grade"])
            if gv.any() and self.source == "SIM":
                self.g_err[0] += float(np.abs(g["grade"] - gt["grade"])[gv].sum())
                self.g_err[1] += int(gv.sum())
            row("elevation MAE (run)", f"{self.e_err[0] / max(self.e_err[1], 1):.3f}", unit=" m")
            row("slope MAE (run)", f"{self.g_err[0] / max(self.g_err[1], 1):.3f}" if self.g_err[1] else "n/a")
            row("visible cells", f"{int(gt['vis'].sum())}/{C * C}")
        else:
            row("ground truth", "n/a", DIM)
        section("CONTROL")
        dbg = nav.controller.debug
        mode = nav.controller.mode
        row("mode", mode, MODE_COL.get(mode, TEXT))
        if self.hist:
            sp, thr, st = self.hist[-1]
            row("speed", f"{sp:5.2f}", unit=" m/s")
            row("throttle / steer", f"{thr:+.2f} / {st:+.2f}")
        if "caps" in dbg:
            k = min(dbg["caps"], key=dbg["caps"].get)
            row("binding speed cap", f"{k} {dbg['caps'][k]:.2f}", AMBER if dbg["caps"][k] < 0.9 else TEXT)
            row("clear gap ahead", f"{min(dbg.get('gap_m', 99), 99):5.1f}", unit=" m")
        if "feasible" in dbg:
            row("feasible arcs", f"{int(dbg['feasible'].sum())}/{len(dbg['feasible'])}")
        section("MISSION")
        row("goal distance", f"{float(np.linalg.norm(nav.goal - nav.pose)):6.1f}", unit=" m")
        for k, lab in (("goals", "goals reached"), ("contacts", "contact events"), ("pose_err", "pose error")):
            if k in info:
                v = info[k]
                row(lab, f"{v:.2f} m" if isinstance(v, float) else v)
        bm = nav.map
        row("mapped cells", f"{int(((np.abs(bm.L) > 0.4) | (bm.EW > 1)).sum()):,}")
        row("believed obstacles", f"{int((bm.L > 0).sum()):,}")

    # ------------------------------------------------------------ telemetry
    def _telemetry(self, c):
        x0, y0, pw, ph = 1056, 684, 848, 190
        c.panel(x0, y0, pw, ph, "drive · last 20 s")
        if not self.hist:
            return
        h = np.array(self.hist)
        vmax = max(2.0, float(np.abs(h[:, 0]).max()) * 1.1)
        plot_lines(c, x0 + 46, y0 + 34, pw - 64, ph - 52, [h[:, 0] / vmax, h[:, 1], h[:, 2]],
                   (-1.05, 1.05), [CYAN, TEAL, AMBER],
                   [f"speed ÷ {vmax:.0f} m/s", "throttle", "steer"], grid_y=[-1, 0, 1])

    def _latency(self, c):
        x0, y0, pw, ph = 1056, 886, 848, 178
        c.panel(x0, y0, pw, ph, "latency · per stage", "ms per control step")
        if not self.lat:
            return
        keys = ("perception", "fusion", "planning", "control")
        cols = (CYAN, TEAL, VIOLET, AMBER)
        L = np.array([[d.get(k, 0.0) for k in keys] for d in self.lat])
        tot = L.sum(1)
        hi = max(10.0, float(np.percentile(tot, 98)) * 1.15)
        X, Y, Wd, Hd = x0 + 46, y0 + 34, pw - 260, ph - 52
        for gy in np.linspace(0, hi, 3):
            py = int(Y + Hd - gy / hi * Hd)
            cv2.line(c.img, (X, py), (X + Wd, py), GRIDC, 1)
            c.text((X - 6, py - 6), f"{gy:.0f}", 10, DIM, "mono", "ra")
        n = len(L)
        bw = Wd / 200
        for k in range(n):
            xb = int(X + (200 - n + k) * bw)
            acc = 0.0
            for j, col in enumerate(cols):
                y1 = int(np.clip(Y + Hd - (acc + L[k, j]) / hi * Hd, Y, Y + Hd))
                y2 = int(np.clip(Y + Hd - acc / hi * Hd, Y, Y + Hd))
                if y2 > y1:
                    cv2.rectangle(c.img, (xb, y1), (xb + max(int(bw) - 1, 1), y2), col, -1)
                acc += L[k, j]
        tx = X + Wd + 24
        ty = Y
        m = L.mean(0)
        for k, col, v, cur in zip(keys, cols, m, L[-1]):
            cv2.rectangle(c.img, (tx, ty + 4), (tx + 10, ty + 14), col, -1)
            c.text((tx + 18, ty), k, 12, MUTED)
            c.text((tx + 180, ty), f"{cur:5.1f}  μ {v:5.1f}", 12, TEXT, "mono", "ra")
            ty += 24
        c.text((tx + 18, ty + 4), "total", 12, TEXT, "semibold")
        c.text((tx + 180, ty + 4), f"{tot[-1]:5.1f}  μ {tot.mean():5.1f}", 12, TEXT, "mono", "ra")

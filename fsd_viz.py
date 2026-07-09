"""FSD-style cinematic visualization of the model's belief world.

Everything in the main view is rendered from what the MODEL believes -- the
persistent occupancy map, the planned route, the live neural detections --
never from simulator ground truth. A chase camera follows the rover through
a dark stylized world: detected obstacles rise as glowing boxes, explored
ground is shaded, the A* route flows ahead as an animated ribbon, and the
goal shines as a light beacon. Insets show the raw camera feed, the raw
neural occupancy wedge, and the full run-length memory map.

Output is tweened to 3 video frames per control step (30 fps).

    python fsd_viz.py --checkpoint runs/best.pth --frames 900 --output_video fsd_demo.mp4
    python fsd_viz.py --checkpoint runs/best.pth --scenario wall --show
"""

import argparse
import collections
import math

import cv2
import numpy as np

# ------------------------------ palette (BGR) ------------------------------
BG_TOP = np.array((14, 8, 4), np.float32)
BG_HOR = np.array((44, 26, 12), np.float32)
GROUND_UNKNOWN = (26, 18, 12)
GROUND_FREE = (46, 34, 20)
GRID_LINE = (62, 48, 30)
TRAIL = (110, 84, 48)
PATH_CORE = (255, 210, 60)      # cyan-ish blue
PATH_GLOW = (140, 90, 20)
ARC_CORE = (160, 255, 120)
GOAL_COL = (60, 170, 255)       # amber
GHOST_OCC = (200, 90, 170)      # violet: JEPA-predicted (unseen) obstacles
GHOST_OPEN = (54, 42, 34)       # faint tint: predicted-open unseen ground
BOX_FILL_LO = np.array((70, 44, 16), np.float32)
BOX_FILL_HI = np.array((200, 140, 40), np.float32)
BOX_EDGE = (255, 216, 120)
BOX_FRESH = (255, 250, 210)
ROVER_BODY = (235, 235, 235)
ROVER_DARK = (90, 90, 90)
HUD_TEXT = (196, 176, 150)
HUD_VALUE = (250, 248, 244)
HUD_PANEL = (26, 18, 12)
DANGER_COL = (70, 70, 255)


def _put(img, text, org, scale=0.5, col=HUD_TEXT, thick=1,
         font=cv2.FONT_HERSHEY_DUPLEX):
    """Anti-aliased text helper with the HUD's default styling."""
    cv2.putText(img, text, org, font, scale, col, thick, cv2.LINE_AA)


class FSDRenderer:
    """Composites one cinematic frame of the pilot's belief world.

    Pipeline per video frame: sky gradient -> homography-warped top-down
    belief layer (explored ground, grid, trail, route ribbon, goal ring)
    -> 3D obstacle boxes / rover / goal beacon -> glow pass -> HUD insets
    (camera PiP, neural wedge, memory minimap, drive cluster).
    """

    TWEENS = 3                  # video frames per control step (10 -> 30 fps)
    CAM_BACK = 9.0              # chase camera offset (m)
    CAM_UP = 4.6
    CAM_PITCH = -17.0           # degrees
    FOV = 62.0
    TD_SPAN = 150.0             # top-down source coverage (m)
    TD_PX = 4.0                 # px per metre in top-down source

    def __init__(self, pilot, size=(1280, 720), tweens=None):
        """Bind to a MapPilot and precompute static layers (sky, fade)."""
        self.pilot = pilot
        self.W, self.H = size
        if tweens is not None:
            self.TWEENS = max(1, int(tweens))
        self.f = 0.5 * self.W / math.tan(math.radians(self.FOV / 2))
        self.cx, self.cy = self.W / 2.0, self.H * 0.44
        self.trail = collections.deque(maxlen=4000)
        self.cam_yaw = None
        self._prev = None       # (pose, heading) of previous control step
        self.t = 0.0
        self._sky = self._make_sky()
        # static radial fade of the top-down source (soft belief horizon)
        n = int(self.TD_SPAN * self.TD_PX)
        yy, xx = np.mgrid[0:n, 0:n].astype(np.float32)
        r = np.sqrt((xx - n / 2) ** 2 + (yy - n / 2) ** 2) / (n / 2)
        self._td_alpha = np.clip((1.05 - r) * 2.2, 0, 1)

    # ------------------------------------------------------------------
    def _make_sky(self):
        """Static dark sky gradient with a soft haze band at the horizon."""
        g = (np.linspace(0, 1, self.H) ** 1.6)[:, None, None]
        img = BG_TOP[None, None] * (1 - g) + BG_HOR[None, None] * g
        # soft haze band just above the horizon line
        v = np.arange(self.H, dtype=np.float32)
        band = np.exp(-((v - self.H * 0.44) / (self.H * 0.05)) ** 2)
        img += band[:, None, None] * np.array((30, 20, 10), np.float32)
        return np.clip(img, 0, 255).astype(np.uint8).repeat(self.W, axis=1)

    # ------------------------------------------------------------------
    # camera helpers (world: x east, z north, y up; screen: u right, v down)
    def _cam(self, pose, heading):
        """Chase-camera pose: behind/above the rover, pitched down, using
        the lazily-smoothed cam_yaw for a cinematic swing through turns."""
        yaw = math.radians(self.cam_yaw)
        fwd = np.array([math.sin(yaw), 0.0, math.cos(yaw)])
        pos = np.array([pose[0], 0.0, pose[1]]) - fwd * self.CAM_BACK
        pos[1] = self.CAM_UP
        th = math.radians(self.CAM_PITCH)
        right = np.array([math.cos(yaw), 0.0, -math.sin(yaw)])
        up0 = np.array([0.0, 1.0, 0.0])
        f2 = fwd * math.cos(th) + up0 * math.sin(th)
        u2 = up0 * math.cos(th) - fwd * math.sin(th)
        R = np.stack([right, u2, f2], axis=1)
        return pos, R

    def _project(self, pts, pos, R):
        """World points -> (pixel uv, camera depth z)."""
        c = (pts - pos) @ R
        z = np.maximum(c[..., 2], 1e-3)
        u = c[..., 0] / z * self.f + self.cx
        v = self.cy - c[..., 1] / z * self.f
        return np.stack([u, v], -1), c[..., 2]

    # ------------------------------------------------------------------
    # top-down belief source (world-anchored; warped onto the ground plane)
    def _topdown(self, pose):
        """Build the world-anchored top-down belief layer for this step.

        Everything painted here (explored ground, grid, trail, animated
        route ribbon, committed arc, goal ring) lands on the 3D ground
        plane via a single homography warp -- one image, pixel-smooth
        perspective for free. Returns (image, edge-fade alpha, anchor).
        """
        p = self.pilot
        n = int(self.TD_SPAN * self.TD_PX)
        img = np.zeros((n, n, 3), np.uint8)
        img[:] = GROUND_UNKNOWN

        # world->pixel: px = (x - x0)*s ; py = (z0 - z)*s   (north up)
        s = self.TD_PX
        x0 = pose[0] - self.TD_SPAN / 2
        z0 = pose[1] + self.TD_SPAN / 2

        def to_px(w):
            """World (x, z) -> top-down pixel coordinates (north up)."""
            w = np.asarray(w, np.float32).reshape(-1, 2)
            return np.stack([(w[:, 0] - x0) * s, (z0 - w[:, 1]) * s], -1)

        # explored free space from the belief map
        i0 = int(np.clip((x0 - p.map_corner[0]) / p.res, 0, p.mc_cells - 1))
        j0 = int(np.clip((pose[1] - self.TD_SPAN / 2 - p.map_corner[1]) / p.res,
                         0, p.mc_cells - 1))
        cells = int(self.TD_SPAN / p.res)
        i1 = min(p.mc_cells, i0 + cells)
        j1 = min(p.mc_cells, j0 + cells)
        L = p.L[i0:i1, j0:j1]
        known_free = (L < -0.4)
        # map crop is (x, z) -> image needs (row=north-down flip, col=x)
        free_img = np.flipud(known_free.T).astype(np.uint8) * 255
        free_img = cv2.resize(free_img, (n, n), interpolation=cv2.INTER_NEAREST)
        free_img = cv2.GaussianBlur(free_img, (9, 9), 0)
        m = free_img > 90
        img[m] = GROUND_FREE

        # grid lines (2 m), brighter where explored
        step = int(2 * s)
        gx0 = int((math.ceil(x0 / 2) * 2 - x0) * s)
        for k in range(gx0, n, step):
            img[:, k] = np.where(m[:, k, None], np.array(GRID_LINE),
                                 img[:, k]).astype(np.uint8)
        gz0 = int((z0 - math.floor(z0 / 2) * 2) * s)
        for k in range(gz0, n, step):
            img[k, :] = np.where(m[k, :, None], np.array(GRID_LINE),
                                 img[k, :]).astype(np.uint8)

        # --- JEPA ghost layer: predicted content of UNSEEN space ---
        # violet = predicted obstacles/hazards, faint warm tint = predicted
        # open ground. Rendered from the completer's prediction layer, which
        # is kept strictly separate from the observed-evidence map: watch
        # ghosts solidify (guess confirmed) or dissolve (guess refuted) as
        # the camera reaches them. Computed for the viz even when the
        # planner is not consuming predictions.
        p._complete_map()
        if p.pred is not None:
            Gp = p.pred.shape[1]
            pres = p.cfg.model.comp_res
            gx0 = (p.pred_origin[0] - x0) * s
            cell_px = pres * s
            blocked = np.maximum(p.pred[0], p.pred[1])
            unobs = p.pred_observed < 0.5
            open_m = unobs & (blocked < 0.25)
            occ_m = unobs & (blocked > 0.45)
            # paint into a small grid image then resize into place
            gimg = np.zeros((Gp, Gp, 3), np.float32)
            gimg[open_m] = GHOST_OPEN
            strength = np.clip((blocked - 0.45) / 0.4, 0, 1)[occ_m, None]
            gimg[occ_m] = (np.array(GHOST_OCC, np.float32)[None]
                           * (0.45 + 0.55 * strength))
            galpha = np.zeros((Gp, Gp), np.float32)
            galpha[open_m] = 0.5
            galpha[occ_m] = 0.5 + 0.4 * strength[:, 0]
            # grid [i=x, j=z] -> image rows = -z, cols = x
            gimg = np.ascontiguousarray(np.flipud(gimg.transpose(1, 0, 2)))
            galpha = np.ascontiguousarray(np.flipud(galpha.T))
            gw = int(Gp * cell_px)
            gimg = cv2.resize(gimg, (gw, gw), interpolation=cv2.INTER_NEAREST)
            galpha = cv2.resize(galpha, (gw, gw),
                                interpolation=cv2.INTER_NEAREST)
            c0 = int(gx0)                                  # left column
            r0 = int((z0 - (p.pred_origin[1] + Gp * pres)) * s)  # top row
            rr0, cc0 = max(0, r0), max(0, c0)
            rr1 = min(n, r0 + gw)
            cc1 = min(n, c0 + gw)
            if rr1 > rr0 and cc1 > cc0:
                sub_i = slice(rr0 - r0, rr0 - r0 + (rr1 - rr0))
                sub_j = slice(cc0 - c0, cc0 - c0 + (cc1 - cc0))
                a = galpha[sub_i, sub_j][..., None]
                roi = img[rr0:rr1, cc0:cc1].astype(np.float32)
                img[rr0:rr1, cc0:cc1] = (roi * (1 - a) +
                                         gimg[sub_i, sub_j] * a
                                         ).astype(np.uint8)

        # trail
        if len(self.trail) > 2:
            tr = to_px(np.array(self.trail)).astype(np.int32)
            cv2.polylines(img, [tr], False, TRAIL, 2, cv2.LINE_AA)

        # planned route: glow pass + animated dashed core
        if p.path is not None and len(p.path) > 1:
            pts = to_px(p.path).astype(np.int32)
            cv2.polylines(img, [pts], False, PATH_GLOW, 9, cv2.LINE_AA)
            cv2.polylines(img, [pts], False, PATH_GLOW, 5, cv2.LINE_AA)
            # flowing dashes
            seg = np.asarray(p.path, np.float32)
            d = np.linalg.norm(np.diff(seg, axis=0), axis=1)
            cum = np.concatenate([[0], np.cumsum(d)])
            dash, gap = 1.6, 1.4
            phase = (self.t * 4.0) % (dash + gap)
            a = -phase
            while a < cum[-1]:
                b = a + dash
                lo, hi = max(a, 0), min(b, cum[-1])
                if hi > lo:
                    ts = np.linspace(lo, hi, max(2, int((hi - lo) * 2)))
                    px = to_px(np.stack([np.interp(ts, cum, seg[:, 0]),
                                         np.interp(ts, cum, seg[:, 1])], -1))
                    cv2.polylines(img, [px.astype(np.int32)], False,
                                  PATH_CORE, 2, cv2.LINE_AA)
                a += dash + gap
        # immediate arc (controller's committed motion)
        if p.last_arc is not None:
            pts = to_px(np.vstack([[pose], p.last_arc]))
            cv2.polylines(img, [pts.astype(np.int32)], False, ARC_CORE, 2,
                          cv2.LINE_AA)

        # goal ring
        g = to_px([p.goal])[0]
        rr = int(2.2 * s)
        cv2.circle(img, tuple(g.astype(int)), rr, GOAL_COL, 2, cv2.LINE_AA)
        cv2.circle(img, tuple(g.astype(int)),
                   max(2, int(rr * (0.35 + 0.25 * math.sin(self.t * 3)))),
                   GOAL_COL, 1, cv2.LINE_AA)

        # radial edge fade -> soft horizon when warped (precomputed)
        return img, self._td_alpha, (x0, z0, s)

    # ------------------------------------------------------------------
    def _ground_homography(self, anchor, pos, R):
        """Top-down pixel -> screen homography (ground plane y=0)."""
        x0, z0, s = anchor
        yaw = math.radians(self.cam_yaw)
        fwd = np.array([math.sin(yaw), math.cos(yaw)])
        right = np.array([math.cos(yaw), -math.sin(yaw)])
        gpos = np.array([pos[0], pos[2]])
        wpts = []
        for d, lat in ((4, -6), (4, 6), (55, -50), (55, 50)):
            wpts.append(gpos + fwd * d + right * lat)
        wpts = np.array(wpts, np.float32)
        w3 = np.stack([wpts[:, 0], np.zeros(4), wpts[:, 1]], -1)
        uv, _ = self._project(w3, pos, R)
        src = np.stack([(wpts[:, 0] - x0) * s, (z0 - wpts[:, 1]) * s], -1)
        return cv2.getPerspectiveTransform(src.astype(np.float32),
                                           uv.astype(np.float32))

    # ------------------------------------------------------------------
    def _obstacle_boxes(self, pose):
        """Discrete belief objects: connected components of occupied cells."""
        p = self.pilot
        rad = 55.0
        i0 = int(np.clip((pose[0] - rad - p.map_corner[0]) / p.res, 0,
                         p.mc_cells - 1))
        j0 = int(np.clip((pose[1] - rad - p.map_corner[1]) / p.res, 0,
                         p.mc_cells - 1))
        cells = int(2 * rad / p.res)
        L = p.L[i0:i0 + cells, j0:j0 + cells]
        occ = (L > 0.45).astype(np.uint8)
        nlab, lab, stats, cent = cv2.connectedComponentsWithStats(occ)
        boxes = []
        for k in range(1, nlab):
            # stats are (col, row, w_cols, h_rows): cols index z, rows index x
            zc, xr, wz, hx, area = stats[k]
            if area < 2:
                continue
            wx0 = p.map_corner[0] + (i0 + xr) * p.res
            wz0 = p.map_corner[1] + (j0 + zc) * p.res
            prob = float(1 / (1 + np.exp(-L[xr:xr + hx, zc:zc + wz].max())))
            boxes.append((wx0, wz0, hx * p.res, wz * p.res, prob))
        return boxes

    def _draw_boxes(self, canvas, boxes, pose, posR):
        """Render belief objects as shaded 3D boxes with luminous top edges
        (painter-sorted between and within boxes; distance-faded fill)."""
        pos, R = posR
        fwd = R[:, 2]
        queue = []
        for (bx, bz, bw, bd, prob) in boxes:
            cx, cz = bx + bw / 2, bz + bd / 2
            rel = np.array([cx - pos[0], 0, cz - pos[2]])
            if rel @ fwd < -6 or np.hypot(rel[0], rel[2]) > 60:
                continue
            hgt = 1.0 + 0.8 * prob
            xs = (bx, bx + bw)
            zs = (bz, bz + bd)
            v = np.array([[xs[0], 0, zs[0]], [xs[1], 0, zs[0]],
                          [xs[1], 0, zs[1]], [xs[0], 0, zs[1]],
                          [xs[0], hgt, zs[0]], [xs[1], hgt, zs[0]],
                          [xs[1], hgt, zs[1]], [xs[0], hgt, zs[1]]])
            uv, z = self._project(v, pos, R)
            if z.min() < 0.5:
                continue
            depth = float(z.mean())
            fill = BOX_FILL_LO + (BOX_FILL_HI - BOX_FILL_LO) * prob
            fade = float(np.clip(1.4 - depth / 55.0, 0.35, 1.0))
            faces = [((0, 1, 5, 4), 0.55), ((1, 2, 6, 5), 0.75),
                     ((2, 3, 7, 6), 0.55), ((3, 0, 4, 7), 0.75),
                     ((4, 5, 6, 7), 1.0)]
            # painter order within the box too, or far faces punch holes
            faces.sort(key=lambda fa: float(z[list(fa[0])].mean()),
                       reverse=True)
            queue.append((depth, uv, z, faces, fill * fade, prob))
        queue.sort(key=lambda q: q[0], reverse=True)
        for depth, uv, z, faces, fill, prob in queue:
            for idx, sh in faces:
                poly = uv[list(idx)].astype(np.int32)
                cv2.fillPoly(canvas, [poly], (fill * sh).tolist(), cv2.LINE_AA)
            # luminous top edge
            top = uv[[4, 5, 6, 7]].astype(np.int32)
            ec = BOX_FRESH if prob > 0.9 else BOX_EDGE
            cv2.polylines(canvas, [top], True, ec, 1, cv2.LINE_AA)

    # ------------------------------------------------------------------
    def _draw_rover(self, canvas, pose, heading, posR):
        """Low-poly rover model: grounding shadow, shaded body box, and an
        amber heading chevron on the roof."""
        pos, R = posR
        yaw = math.radians(heading)
        c, s = math.cos(yaw), math.sin(yaw)

        def rot(px, pz):
            """Rover-frame (right, forward) offset -> world 3D point."""
            return np.array([pose[0] + px * c + pz * s, 0,
                             pose[1] - px * s + pz * c])

        # grounding shadow (blend only inside its bounding box)
        sh = [rot(px, pz) for px, pz in
              [(-0.8, -1.1), (0.8, -1.1), (0.8, 1.1), (-0.8, 1.1)]]
        suv, sz = self._project(np.array(sh), pos, R)
        if sz.min() > 0.5:
            pts = suv.astype(np.int32)
            x0, y0 = np.clip(pts.min(0), 0, [self.W - 1, self.H - 1])
            x1, y1 = np.clip(pts.max(0) + 1, 1, [self.W, self.H])
            if x1 > x0 and y1 > y0:
                roi = canvas[y0:y1, x0:x1]
                over = roi.copy()
                cv2.fillPoly(over, [pts - [x0, y0]], (6, 4, 2), cv2.LINE_AA)
                cv2.addWeighted(over, 0.55, roi, 0.45, 0, dst=roi)

        body = [(-0.55, -0.8), (0.55, -0.8), (0.55, 0.8), (-0.55, 0.8)]
        hgt = 0.5
        v = np.array([rot(px, pz) for px, pz in body] +
                     [rot(px, pz) + [0, hgt, 0] for px, pz in body])
        uv, z = self._project(v, pos, R)
        if z.min() < 0.5:
            return
        for idx, sh in [((0, 1, 5, 4), 0.5), ((1, 2, 6, 5), 0.7),
                        ((2, 3, 7, 6), 0.5), ((3, 0, 4, 7), 0.7),
                        ((4, 5, 6, 7), 1.0)]:
            col = tuple(int(cc * sh) for cc in ROVER_BODY)
            cv2.fillPoly(canvas, [uv[list(idx)].astype(np.int32)], col,
                         cv2.LINE_AA)
        # amber heading chevron on the roof
        tip = rot(0, 0.72) + [0, hgt + 0.01, 0]
        l = rot(-0.34, 0.18) + [0, hgt + 0.01, 0]
        r = rot(0.34, 0.18) + [0, hgt + 0.01, 0]
        tuv, _ = self._project(np.stack([tip, l, r]), pos, R)
        cv2.fillPoly(canvas, [tuv.astype(np.int32)], GOAL_COL, cv2.LINE_AA)

    # ------------------------------------------------------------------
    def _draw_beacon(self, canvas, glow, posR):
        """Vertical pulsing light pillar at the goal (core + glow pass)."""
        pos, R = posR
        g = self.pilot.goal
        base = np.array([g[0], 0.0, g[1]])
        topp = np.array([g[0], 9.0, g[1]])
        uv, z = self._project(np.stack([base, topp]), pos, R)
        if z.min() < 0.5:
            return
        a, b = uv.astype(int)
        pulse = 0.7 + 0.3 * math.sin(self.t * 3.0)
        cv2.line(glow, tuple(a), tuple(b),
                 tuple(int(cc * pulse) for cc in GOAL_COL), 9, cv2.LINE_AA)
        cv2.line(canvas, tuple(a), tuple(b), GOAL_COL, 2, cv2.LINE_AA)

    # ------------------------------------------------------------------
    # HUD widgets
    def _panel(self, canvas, x, y, w, h, alpha=0.62):
        """Semi-transparent dark HUD panel with a thin border."""
        roi = canvas[y:y + h, x:x + w]
        fill = np.empty_like(roi)
        fill[:] = HUD_PANEL
        cv2.addWeighted(roi, 1 - alpha, fill, alpha, 0, dst=roi)
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (70, 56, 38), 1,
                      cv2.LINE_AA)

    def _hud(self, canvas, sim_frame, out, sensors):
        """Draw all 2D overlays: camera PiP, neural-wedge panel, memory
        minimap, status panel, drive cluster, and the danger vignette."""
        p = self.pilot
        W, H = self.W, self.H

        # ---- camera PiP (top right) ----
        pw = 252
        pip = cv2.resize(sim_frame, (pw, pw))
        x0, y0 = W - pw - 18, 18
        canvas[y0:y0 + pw, x0:x0 + pw] = pip
        cv2.rectangle(canvas, (x0 - 1, y0 - 1), (x0 + pw, y0 + pw),
                      (90, 74, 50), 1, cv2.LINE_AA)
        _put(canvas, "FRONT CAMERA", (x0, y0 - 6), 0.42)

        # ---- neural occupancy wedge (below PiP) ----
        if hasattr(p, "last_evidence"):
            ev = p.last_evidence[:, :p.PAINT_RANGE_CELLS]
            cf = p.last_conf[:, :p.PAINT_RANGE_CELLS]
            e = np.clip(ev / 2.0, 0, 1) * np.clip(cf * 2, 0, 1)
            img = np.zeros((*e.T.shape, 3), np.float32)
            img[..., 0] = 255 * np.flipud(e.T)          # blue
            img[..., 1] = 200 * np.flipud(e.T)
            img[..., 2] = 90 * np.flipud(e.T)
            free = np.flipud((np.clip(-ev / 4.0, 0, 1) * cf).T)
            img += free[..., None] * np.array((66, 50, 30))
            wpx = pw
            hpx = int(wpx * e.shape[1] / e.shape[0])
            img = cv2.resize(img.astype(np.uint8), (wpx, hpx),
                             interpolation=cv2.INTER_LINEAR)
            y1 = y0 + pw + 30
            canvas[y1:y1 + hpx, x0:x0 + wpx] = img
            cv2.rectangle(canvas, (x0 - 1, y1 - 1), (x0 + wpx, y1 + hpx),
                          (90, 74, 50), 1, cv2.LINE_AA)
            _put(canvas, "NEURAL OCCUPANCY", (x0, y1 - 6), 0.42)

        # ---- memory minimap (bottom left) ----
        mm = self._minimap(200)
        mx, my = 18, H - 200 - 18
        canvas[my:my + 200, mx:mx + 200] = mm
        cv2.rectangle(canvas, (mx - 1, my - 1), (mx + 200, my + 200),
                      (90, 74, 50), 1, cv2.LINE_AA)
        _put(canvas, "MEMORY MAP", (mx, my - 6), 0.42)

        # ---- status (top left) ----
        self._panel(canvas, 18, 18, 240, 116)
        _put(canvas, "DR-JEPA v10", (30, 44), 0.62, HUD_VALUE)
        _put(canvas, "BELIEF-SPACE NAVIGATION", (30, 62), 0.36)
        d = float(np.linalg.norm(p.goal - p.pose))
        _put(canvas, "GOAL", (30, 88), 0.4)
        _put(canvas, f"{d:5.0f} m", (96, 90), 0.55, HUD_VALUE)
        _put(canvas, "HAZARD", (30, 114), 0.4)
        w = int(np.clip(p.last_danger, 0, 1) * 120)
        cv2.rectangle(canvas, (96, 104), (216, 114), (52, 42, 30), -1)
        col = DANGER_COL if p.last_danger > 0.55 else (120, 200, 120)
        if w:
            cv2.rectangle(canvas, (96, 104), (96 + w, 114), col, -1)

        # ---- drive cluster (bottom center) ----
        cx = W // 2
        self._panel(canvas, cx - 150, H - 96, 300, 78)
        _put(canvas, f"{abs(sensors['speed']):4.1f}", (cx - 118, H - 44),
             1.05, HUD_VALUE, 2)
        _put(canvas, "m/s", (cx - 44, H - 44), 0.42)
        # steering arc
        scx, scy, r = cx + 62, H - 52, 26
        cv2.ellipse(canvas, (scx, scy), (r, r), 0, 180, 360, (80, 64, 44), 2,
                    cv2.LINE_AA)
        ang = math.radians(90 * out["steer"])
        cv2.line(canvas, (scx, scy),
                 (int(scx + r * math.sin(ang)), int(scy - r * math.cos(ang))),
                 HUD_VALUE, 2, cv2.LINE_AA)
        # throttle bar
        tb = int(np.clip(out["throttle"], -1, 1) * 30)
        cv2.rectangle(canvas, (cx + 116, H - 88), (cx + 128, H - 26),
                      (52, 42, 30), -1)
        ymid = H - 57
        c2 = (120, 220, 140) if tb >= 0 else (80, 110, 255)
        cv2.rectangle(canvas, (cx + 117, ymid - max(tb, 0)),
                      (cx + 127, ymid - min(tb, 0)), c2, -1)

        # danger vignette
        if p.last_danger > 0.55:
            k = (p.last_danger - 0.55) / 0.45
            pulse = 0.5 + 0.5 * math.sin(self.t * 8)
            edge = np.zeros_like(canvas)
            cv2.rectangle(edge, (0, 0), (W - 1, H - 1), DANGER_COL, 24)
            edge = cv2.GaussianBlur(edge, (61, 61), 0)
            cv2.addWeighted(canvas, 1.0, edge, 0.5 * k * pulse, 0, dst=canvas)

    def _minimap(self, size):
        """North-up overview of the ENTIRE persistent map: explored space,
        obstacle beliefs, driven trail, planned route, rover and goal.
        Auto-zooms to the explored extent."""
        p = self.pilot
        # auto-zoom to the explored extent (plus goal), min 70 m span
        ex = np.array(self.trail) if len(self.trail) > 1 else p.pose[None]
        lo = np.minimum(ex.min(axis=0), p.goal) - 15
        hi = np.maximum(ex.max(axis=0), p.goal) + 15
        span = float(max(70.0, (hi - lo).max()))
        mid = (lo + hi) / 2
        c = int(span / p.res)
        ci = int((mid[0] - p.map_corner[0]) / p.res)
        cj = int((mid[1] - p.map_corner[1]) / p.res)
        i0, j0 = max(0, ci - c // 2), max(0, cj - c // 2)
        crop = p.L[i0:i0 + c, j0:j0 + c]
        img = np.zeros((*crop.T.shape, 3), np.uint8)
        img[:] = GROUND_UNKNOWN
        prob = 1 / (1 + np.exp(-crop.T))
        known = np.abs(crop.T) > 0.4
        img[known & (prob < 0.5)] = GROUND_FREE
        occm = known & (prob > 0.6)
        img[occm] = BOX_EDGE
        img = np.ascontiguousarray(np.flipud(img))

        def mark(w, col, r=2):
            """Dot marker at a world position on the minimap."""
            i = int((w[0] - p.map_corner[0]) / p.res) - i0
            j = int((w[1] - p.map_corner[1]) / p.res) - j0
            if 0 <= i < img.shape[1] and 0 <= j < img.shape[0]:
                cv2.circle(img, (i, img.shape[0] - 1 - j), r, col, -1,
                           cv2.LINE_AA)

        if len(self.trail) > 2:
            pts = np.array([[int((w[0] - p.map_corner[0]) / p.res) - i0,
                             img.shape[0] - 1 -
                             (int((w[1] - p.map_corner[1]) / p.res) - j0)]
                            for w in self.trail], np.int32)
            cv2.polylines(img, [pts], False, TRAIL, 1, cv2.LINE_AA)
        if p.path is not None:
            pts = np.array([[int((w[0] - p.map_corner[0]) / p.res) - i0,
                             img.shape[0] - 1 -
                             (int((w[1] - p.map_corner[1]) / p.res) - j0)]
                            for w in p.path[::2]], np.int32)
            cv2.polylines(img, [pts], False, PATH_CORE, 1, cv2.LINE_AA)
        mark(p.pose, (255, 255, 255), 3)
        mark(p.goal, GOAL_COL, 3)
        return cv2.resize(img, (size, size), interpolation=cv2.INTER_NEAREST)

    # ------------------------------------------------------------------
    def step(self, sim_frame, sensors, out):
        """One control step -> TWEENS video frames."""
        p = self.pilot
        p.last_danger = out["danger"]
        pose = p.pose.copy()
        heading = sensors["heading"]
        self.trail.append(tuple(pose))
        if self.cam_yaw is None:
            self.cam_yaw = heading
        if self._prev is None:
            self._prev = (pose, heading)

        td, td_alpha, anchor = self._topdown(pose)
        p0, h0 = self._prev
        frames = []
        for k in range(self.TWEENS):
            a = (k + 1) / self.TWEENS
            self.t += 1.0 / (10 * self.TWEENS)
            tw_pose = p0 * (1 - a) + pose * a
            dh = (heading - h0 + 180) % 360 - 180
            tw_head = h0 + dh * a
            # lazy cinematic camera
            dyaw = (tw_head - self.cam_yaw + 180) % 360 - 180
            self.cam_yaw += dyaw * 0.10
            posR = self._cam(tw_pose, tw_head)

            Hm = self._ground_homography(anchor, *posR)
            warped = cv2.warpPerspective(td, Hm, (self.W, self.H))
            wa = cv2.warpPerspective(td_alpha, Hm, (self.W, self.H))
            # SIMD uint8 per-pixel alpha blend (no float round trips)
            canvas = cv2.blendLinear(warped, self._sky, wa, 1.0 - wa)

            glow = np.zeros_like(canvas)
            self._draw_beacon(canvas, glow, posR)
            self._draw_boxes(canvas, self._boxes_cache, tw_pose, posR)
            self._draw_rover(canvas, tw_pose, tw_head, posR)
            glow = cv2.GaussianBlur(glow, (41, 41), 0)
            cv2.addWeighted(canvas, 1.0, glow, 0.85, 0, dst=canvas)

            self._hud(canvas, sim_frame, out, sensors)
            frames.append(canvas)
        self._prev = (pose, heading)
        return frames

    @property
    def _boxes_cache(self):
        """Obstacle boxes recomputed at most once per control step (the
        map only changes between steps, not between tween frames)."""
        if getattr(self, "_bc_step", -1) != self.pilot.step_i:
            self._bc = self._obstacle_boxes(self.pilot.pose)
            self._bc_step = self.pilot.step_i
        return self._bc


# ==========================================================================
# Standalone run loop
# ==========================================================================
def main():
    """Drive the MapPilot through an endless world and record the FSD view."""
    from drjepa.simulator import RoverSim, SimConfig
    from drjepa.pilot import MapPilot

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", default="runs/best.pth")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--scenario", default=None,
                    choices=[None, "open", "dense", "wall", "boulders"])
    ap.add_argument("--frames", type=int, default=600, help="control steps")
    ap.add_argument("--output_video", default="fsd_demo.mp4")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--tweens", type=int, default=3,
                    help="video frames per control step (1 = fast preview "
                         "at 10 fps, 3 = smooth 30 fps)")
    ap.add_argument("--show", action="store_true", help="also open a window")
    args = ap.parse_args()

    pilot = MapPilot(args.checkpoint)
    sim = RoverSim(SimConfig(), scenario=args.scenario, seed=args.seed)
    renderer = FSDRenderer(pilot, size=(args.width, args.height),
                           tweens=args.tweens)
    writer = cv2.VideoWriter(args.output_video, cv2.VideoWriter_fourcc(*"mp4v"),
                             10.0 * renderer.TWEENS,
                             (args.width, args.height))
    print(f"Rendering {args.frames} control steps -> {args.output_video} "
          f"({10 * renderer.TWEENS} fps, q to quit)")
    goals = 0
    for n in range(args.frames):
        frame = sim.render()
        out = pilot.step(frame, sim.sensor_readout())
        for f in renderer.step(frame, sim.sensor_readout(), out):
            writer.write(f)
            if args.show:
                cv2.imshow("DR-JEPA FSD view", f)
                if cv2.waitKey(1) == ord("q"):
                    writer.release()
                    return
        info = sim.step(out["throttle"], out["steer"])
        if info["tipped"]:
            print(f"  [{n:4d}] tipped over -- ending run")
            break
        if info["reached"] or info["timeout"]:
            goals += info["reached"]
            print(f"  [{n:4d}] {'goal reached' if info['reached'] else 'timeout'}"
                  f" (total goals {goals})")
            sim.respawn_goal()
    writer.release()
    cv2.destroyAllWindows()
    print(f"Done: {goals} goals, video at {args.output_video}")


if __name__ == "__main__":
    main()

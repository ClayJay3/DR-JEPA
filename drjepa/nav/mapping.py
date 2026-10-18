"""Persistent world-anchored belief map (the robot's long-term memory).

Per-frame perception grids (robot frame) fuse into a 512 x 512 x 0.5 m map
anchored in the local world frame (x east, z north) that re-centres when
the robot nears its edge: O(1) cost per step for an unlimited run.

Channels
    L       occupancy log-odds (prior-corrected Bayesian update)
    E, EW   terrain elevation: confidence-weighted running mean (+ weight)
    G, GW   terrain grade: confidence-weighted running mean (+ weight)
    SX, SZ  world-frame slope vector (uphill, |S| ~ grade), same weights as G
    LS      soft-ground (sand) log-odds

Why each occupancy rule exists (all measured in v8-v12, kept verbatim):
  * prior correction -- a calibrated network's "no information" logit is
    the base rate (~2-3 %), not 0; fusing raw logits counts weak positive
    evidence as free space (success 75 % -> 100 % when fixed);
  * positive evidence scaled 0.45x, decay 0.9985/step, positive cap +3.5 --
    false positives are correlated frame to frame (a ridge misread once is
    misread every frame) and otherwise grow phantom walls;
  * fusion only within 12 m and where predicted visibility >= 0.55.
Slope and elevation are continuous, so they fuse as weighted means with
trust falling off with range (1.0 within 4 m, 0.5 to 8 m, 0.25 beyond):
far estimates never accumulate, and near views overwrite them on approach.
The slope vector takes its magnitude from the grade head and its direction
from the elevation head's gradient (median 9 deg error on steep sim cells):
robots tip sideways long before they tip forward, so traversability depends
on the direction of travel.
"""

import math
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class MapConfig:
    """Belief-map and fusion constants."""
    cells: int = 512
    res: float = 0.5
    paint_range: float = 12.0        # m: fuse perception only this close
    vis_gate: float = 0.55           # predicted-visibility gate
    occ_prior_logit: float = -2.5    # network logit meaning "base rate"
    pos_evidence_scale: float = 0.45
    evidence_gain: float = 0.55
    evidence_clip: float = 4.0
    decay: float = 0.9985
    lodds_min: float = -6.0
    lodds_max: float = 3.5
    sand_prior_logit: float = -2.4
    guard_evidence: float = 2.0      # single-frame evidence that blocks the guard
    vo: bool = True                  # scan-match pose correction
    vo_window: int = 2
    vo_gain: float = 0.2
    vo_margin: float = 10.0
    vo_min_l: float = 1.2


class BeliefMap:
    """Multi-channel world map + per-frame fusion (see module docstring)."""

    CHANNELS = ("L", "E", "EW", "G", "GW", "SX", "SZ", "LS")

    def __init__(self, cfg: MapConfig, grid):
        self.cfg, self.grid = cfg, grid
        C, r = grid.cells, grid.res
        xs = (np.arange(C) + 0.5) * r - grid.half
        zs = (np.arange(C) + 0.5) * r
        self._gx, self._gz = np.meshgrid(xs, zs, indexing="ij")     # robot-frame cell centres
        att = np.ones((C, C), np.float32)
        att[:, zs > 4.0] = 0.5
        att[:, zs > 8.0] = 0.25
        self._range_w = att
        self.reset()

    def reset(self):
        n = self.cfg.cells
        for c in self.CHANNELS:
            setattr(self, c, np.zeros((n, n), np.float32))
        self.corner = None              # world coords of map cell (0, 0)
        self.guard_dist = None
        self.guard_pose = None
        self.last_evidence = None

    # ------------------------------------------------------------ frame
    def init_at(self, pos):
        """Centre the map on a world position (first fix)."""
        half = self.cfg.cells * self.cfg.res / 2.0
        self.corner = np.asarray(pos, float) - half

    def maybe_recenter(self, pos):
        """Scroll every channel if the robot nears the map edge."""
        n, r = self.cfg.cells, self.cfg.res
        ij = (np.asarray(pos) - self.corner) / r
        margin = n * 0.15
        if np.all(ij > margin) and np.all(ij < n - margin):
            return
        s = np.floor(ij - n / 2.0).astype(int)
        x0, z0 = s
        xs = slice(max(0, x0), min(n, n + x0))
        xd = slice(max(0, -x0), max(0, -x0) + (xs.stop - xs.start))
        zs = slice(max(0, z0), min(n, n + z0))
        zd = slice(max(0, -z0), max(0, -z0) + (zs.stop - zs.start))
        for c in self.CHANNELS:
            src = getattr(self, c)
            new = np.zeros_like(src)
            new[xd, zd] = src[xs, zs]
            setattr(self, c, new)
        self.corner = self.corner + s * r

    def index(self, wx, wz):
        """World coords -> integer map indices (unclipped)."""
        r = self.cfg.res
        return (np.floor((wx - self.corner[0]) / r).astype(int),
                np.floor((wz - self.corner[1]) / r).astype(int))

    def _cells_world(self, pos, yaw_deg):
        c, s = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
        wx = pos[0] + self._gx * c + self._gz * s
        wz = pos[1] - self._gx * s + self._gz * c
        return wx, wz

    # ------------------------------------------------------------ fusion
    def fuse(self, g, pose, yaw_deg, recovering=False):
        """Fuse one perception frame (dict of robot-frame grids) at `pose`
        (world x, z) / heading. Returns the (possibly VO-corrected) pose."""
        cfg = self.cfg
        n = cfg.cells
        conf = g["vis"].astype(np.float32).copy()
        conf[:, self._gz[0] > cfg.paint_range] = 0.0
        conf[conf < cfg.vis_gate] = 0.0
        self.L *= cfg.decay
        ev = g["occ_logit"] - cfg.occ_prior_logit
        ev = np.where(ev > 0, ev * cfg.pos_evidence_scale, ev)
        upd = np.clip(ev, -cfg.evidence_clip, cfg.evidence_clip) * conf * cfg.evidence_gain
        self.last_evidence, self.last_conf = ev, conf
        pose = np.asarray(pose, float).copy()

        def cells(p):
            wx, wz = self._cells_world(p, yaw_deg)
            ii, jj = self.index(wx, wz)
            ok = (ii >= 0) & (ii < n) & (jj >= 0) & (jj < n) & (conf > 0)
            return ii, jj, ok

        ii, jj, ok = cells(pose)
        if cfg.vo and not recovering:
            shift = self._vo_shift(ii, jj, upd, ok)
            if shift is not None:
                pose += cfg.vo_gain * np.array(shift) * cfg.res
                ii, jj, ok = cells(pose)
        np.add.at(self.L, (ii[ok], jj[ok]), upd[ok])
        np.clip(self.L, cfg.lodds_min, cfg.lodds_max, out=self.L)

        # elevation: map stores ABSOLUTE height; wedge elevation is relative
        # to the ground under the robot, whose height comes from cells
        # already fused around it (0 at the start)
        ci, cj = self.index(pose[0], pose[1])
        w_h = self.EW[ci - 2:ci + 3, cj - 2:cj + 3]
        h0 = float((self.E[ci - 2:ci + 3, cj - 2:cj + 3] * w_h).sum() / w_h.sum()) if w_h.sum() > 0.5 else 0.0
        iv, jv = ii[ok], jj[ok]
        w = (conf * self._range_w)[ok]
        self._wmean(("E",), "EW", iv, jv, (g["elev"][ok] + h0,), w, cap=20.0)
        ex, ez = np.gradient(g["elev"].astype(np.float32), self.grid.res)
        k = g["grade"] / np.maximum(np.hypot(ex, ez), 1e-3)
        c, s = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
        sx, sz = (ex * c + ez * s) * k, (-ex * s + ez * c) * k     # robot -> world
        self._wmean(("G", "SX", "SZ"), "GW", iv, jv, (g["grade"][ok], sx[ok], sz[ok]), w, cap=20.0)
        sev = np.clip(np.log(np.clip(g["sand"], 1e-4, 1 - 1e-4) / np.clip(1 - g["sand"], 1e-4, 1)) - cfg.sand_prior_logit, -4, 4)
        sev = np.where(sev > 0, sev * 0.5, sev)
        np.add.at(self.LS, (iv, jv), (sev * conf)[ok] * cfg.evidence_gain)
        np.clip(self.LS, -6.0, 4.0, out=self.LS)

        # zero-lag near-field guard: distance field of THIS frame's blocked
        # cells, in the robot frame (immune to pose drift)
        blocked = ((ev > cfg.guard_evidence) & (conf > 0.5)).astype(np.uint8)
        self.guard_dist = cv2.distanceTransform(1 - blocked, cv2.DIST_L2, 3) * self.grid.res
        self.guard_pose = (pose.copy(), yaw_deg)
        return pose

    def _wmean(self, names, wname, i, j, vals, w, cap):
        """Weighted running mean of channels `names` sharing weight `wname`."""
        Wt = getattr(self, wname)
        # several grid cells can land in one map cell: accumulate first
        den = np.zeros_like(Wt)
        np.add.at(den, (i, j), w)
        m = den > 0
        tot = np.maximum(Wt[m] + den[m], 1e-6)
        for name, v in zip(names, vals):
            V = getattr(self, name)
            num = np.zeros_like(V)
            np.add.at(num, (i, j), v * w)
            V[m] = (V[m] * Wt[m] + num[m]) / tot
        Wt[m] = np.minimum(Wt[m] + den[m], cap)

    def _vo_shift(self, ii, jj, upd, ok):
        """Scan-match the new evidence against established map structure;
        returns the best (di, dj) cell shift or None."""
        cfg, n = self.cfg, self.cfg.cells
        iv, jv, uv = ii[ok], jj[ok], upd[ok]
        strong = uv > 0.4
        if strong.sum() < 25:
            return None
        iv, jv, uv = iv[strong], jv[strong], uv[strong]
        Lm = np.where(np.abs(self.L) > cfg.vo_min_l, self.L, 0.0)
        best, best_shift, zero = -1e18, (0, 0), 0.0
        w = cfg.vo_window
        for di in range(-w, w + 1):
            i2 = np.clip(iv + di, 0, n - 1)
            for dj in range(-w, w + 1):
                j2 = np.clip(jv + dj, 0, n - 1)
                s = float(np.dot(uv, Lm[i2, j2])) - 1.5 * (di * di + dj * dj)
                if di == 0 and dj == 0:
                    zero = s
                if s > best:
                    best, best_shift = s, (di, dj)
        if best_shift != (0, 0) and best - zero > cfg.vo_margin:
            return best_shift
        return None

    # ------------------------------------------------------------ queries
    def guard_clearance(self, dx, dz):
        """Clearance (m) along world offsets via the freshest frame (min per row)."""
        if self.guard_dist is None:
            return np.full(dx.shape[0], 99.0)
        yaw = math.radians(self.guard_pose[1])
        c, s = math.cos(yaw), math.sin(yaw)
        rx, rz = dx * c - dz * s, dx * s + dz * c
        G = self.grid
        ii = np.floor((rx + G.half) / G.res).astype(int)
        jj = np.floor(rz / G.res).astype(int)
        inside = (ii >= 0) & (ii < G.cells) & (jj >= 0) & (jj < G.cells)
        d = np.where(inside, self.guard_dist[np.clip(ii, 0, G.cells - 1), np.clip(jj, 0, G.cells - 1)], 99.0)
        return d.min(axis=1)

    def guard_ahead(self, half_w=2, depth=6):
        """Min fresh-frame clearance in a corridor straight ahead (m)."""
        if self.guard_dist is None:
            return 99.0
        ci = self.grid.cells // 2
        return float(self.guard_dist[ci - half_w:ci + half_w + 1, :depth].min())

    def slope_along(self, ai, aj, heading_rad):
        """Believed (pitch, roll) grade for travel at `heading_rad` (from
        north, clockwise) through map cells (ai, aj); pitch > 0 is uphill.
        Disagreement between fused directions counts as roll (|mean vector|
        < mean grade), so an uncertain direction is treated as side slope."""
        known = self.GW[ai, aj] > 0.3
        G = np.where(known, self.G[ai, aj], 0.0)
        vx = np.where(known, self.SX[ai, aj], 0.0)
        vz = np.where(known, self.SZ[ai, aj], 0.0)
        fx, fz = np.sin(heading_rad), np.cos(heading_rad)
        pitch = vx * fx + vz * fz
        roll = np.maximum(np.abs(vx * fz - vz * fx), G - np.hypot(vx, vz))
        return pitch, roll

    def prob(self, sl=None):
        """Occupancy probability (optionally of a window slice)."""
        L = self.L if sl is None else self.L[sl]
        return 1.0 / (1.0 + np.exp(-L))

    def observed(self, sl):
        """Cells with any real evidence."""
        return (np.abs(self.L[sl]) > 0.4) | (self.EW[sl] > 1.0)

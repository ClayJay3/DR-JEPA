"""Global route planning on the belief map (weighted A*).

Costmap per planning cell (2x-downsampled window around robot + goal):
    1 + 6 p(occ) + soft inflation near believed obstacles
      + 5 * slope penalty (grade between comfortable and tip limits)
      + 3 p(sand) + 0.3 on never-observed cells
Lethal: within `lethal_r` of a believed obstacle, or fused grade past the
robot's tip limit. If no route exists the planner retries with the slim
`hard_r` radius (the map can be wrong; the planner must not deadlock) --
but never across believed-fatal slopes: a tip-over is terminal, a graze
is not.

Plan-consistency prior (v12, measured): near-tie routes around an obstacle
otherwise flip sides every replan and the alternating waypoint drives the
robot AT the obstacle. Cells within 3 m of the incumbent route get a small
discount -- only where >1.6 m clear of believed obstacles (ungated, it
offsets the inflation penalty and the planner threads corridors).
"""

import heapq
import math
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class PlannerConfig:
    """Global planner constants (robot-specific radii come from the profile)."""
    occ_thresh: float = 0.6          # fused p(occ) treated as an obstacle
    lethal_margin: float = 0.15      # lethal radius = robot radius + this
    hard_margin: float = -0.25       # fallback radius = robot radius + this
    inflate: float = 1.1             # soft inflation beyond the lethal radius
    window_pad: float = 16.0         # planning window margin around robot+goal
    heuristic_w: float = 1.4         # weighted A* (fewer expansions)
    max_expansions: int = 60000
    consistency: float = 0.45
    consistency_r: float = 3.0
    consistency_gate: float = 1.6
    unknown_cost: float = 0.3
    fatal_slope: float = 0.92        # x tip_roll: believed slope the route never crosses
    fallback_slope: float = 1.0      # x tip_roll: ... even when no other route exists


class Planner:
    """Weighted A* over the believed costmap; keeps the incumbent route."""

    def __init__(self, robot, cfg: PlannerConfig = None):
        self.robot, self.cfg = robot, cfg or PlannerConfig()
        self.path = None
        self.dist_m = None
        self.win = None

    def reset(self):
        self.path = None

    def window(self, bmap, pose, goal):
        """Map index window covering robot + goal with padding."""
        r, n, pad = bmap.cfg.res, bmap.cfg.cells, self.cfg.window_pad
        lo = np.minimum(pose, goal) - pad
        hi = np.maximum(pose, goal) + pad
        i0 = int(max(0, (lo[0] - bmap.corner[0]) / r))
        j0 = int(max(0, (lo[1] - bmap.corner[1]) / r))
        i1 = int(min(n, (hi[0] - bmap.corner[0]) / r + 1))
        j1 = int(min(n, (hi[1] - bmap.corner[1]) / r + 1))
        return i0, j0, i1, j1

    def replan(self, bmap, pose, goal):
        """Recompute the route; stores self.path (world points) or None."""
        c, rb, res = self.cfg, self.robot, bmap.cfg.res
        i0, j0, i1, j1 = self.window(bmap, pose, goal)
        sl = (slice(i0, i1), slice(j0, j1))
        prob = bmap.prob(sl)
        occ = (prob > c.occ_thresh).astype(np.uint8)
        self.dist_m = cv2.distanceTransform((1 - occ).astype(np.uint8), cv2.DIST_L2, 3) * res
        self.win = (i0, j0, i1, j1)
        ph, pw = (i1 - i0) // 2, (j1 - j0) // 2
        if ph < 2 or pw < 2:
            self.path = None
            return
        ds = lambda a, f: f(a[:ph * 2, :pw * 2].reshape(ph, 2, pw, 2), axis=(1, 3))
        p2, d2 = ds(prob, np.max), ds(self.dist_m, np.min)
        lethal_r = rb.radius + c.lethal_margin
        infl = lethal_r + c.inflate
        cost = 1.0 + 6.0 * p2 + np.where(d2 < infl, (infl - d2) * 2.0, 0.0)
        # believed terrain: fused grade (confidence-weighted) and sand
        G, GW = bmap.G[sl], bmap.GW[sl]
        g2 = ds(np.where(GW > 0.3, G, 0.0), np.max)
        span = max(rb.tip_roll - rb.grade_drive, 1e-3)
        steep = np.clip((g2 - rb.grade_drive) / span, 0.0, 1.0)
        cost += 5.0 * steep
        fatal = g2 > c.fatal_slope * rb.tip_roll
        s2 = ds(1.0 / (1.0 + np.exp(-bmap.LS[sl])), np.max)
        cost += 3.0 * s2
        o2 = ds(bmap.observed(sl), np.max)
        cost += (~o2) * c.unknown_cost
        lethal = (d2 < lethal_r) | fatal

        if self.path is not None and len(self.path) > 1:
            pi = (self.path[:, 0] - bmap.corner[0]) / res - i0
            pj = (self.path[:, 1] - bmap.corner[1]) / res - j0
            inb = (pi >= 0) & (pi < ph * 2) & (pj >= 0) & (pj < pw * 2)
            if inb.sum() >= 2:
                on = np.ones((ph, pw), np.uint8)
                on[(pi[inb] // 2).astype(int), (pj[inb] // 2).astype(int)] = 0
                dp = cv2.distanceTransform(on, cv2.DIST_L2, 3) * (2 * res)
                disc = c.consistency * np.clip(1.0 - dp / c.consistency_r, 0.0, 1.0)
                disc *= d2 > c.consistency_gate
                cost = np.maximum(cost - disc, 0.8)

        def cell(p):
            return (int(np.clip((p[0] - bmap.corner[0]) / res - i0, 0, ph * 2 - 1) // 2),
                    int(np.clip((p[1] - bmap.corner[1]) / res - j0, 0, pw * 2 - 1) // 2))

        s, g = cell(pose), cell(goal)
        path = astar(cost, lethal, s, g, c.heuristic_w, c.max_expansions)
        if path is None:
            path = astar(cost, (d2 < rb.radius + c.hard_margin) | (g2 > c.fallback_slope * rb.tip_roll), s, g,
                         c.heuristic_w, c.max_expansions)
        if path is None:
            self.path = None
            return
        self.path = np.array([[bmap.corner[0] + (i0 + (a + 0.5) * 2) * res,
                               bmap.corner[1] + (j0 + (b + 0.5) * 2) * res] for a, b in path])

    def clearance(self, bmap, pts):
        """Believed clearance (m) at world points via the window distance field."""
        if self.dist_m is None:
            return np.full(pts.shape[:-1], 99.0)
        i0, j0, _, _ = self.win
        r = bmap.cfg.res
        ii = np.clip(((pts[..., 0] - bmap.corner[0]) / r - i0).astype(int), 0, self.dist_m.shape[0] - 1)
        jj = np.clip(((pts[..., 1] - bmap.corner[1]) / r - j0).astype(int), 0, self.dist_m.shape[1] - 1)
        return self.dist_m[ii, jj]


_STEPS = [(-1, -1, 1.414), (-1, 0, 1.0), (-1, 1, 1.414), (0, -1, 1.0),
          (0, 1, 1.0), (1, -1, 1.414), (1, 0, 1.0), (1, 1, 1.414)]


def astar(cost, lethal, start, goal, w_h=1.4, max_exp=60000):
    """Weighted A* on an 8-connected grid; returns a cell list or None."""
    h, w = cost.shape
    if lethal[start]:
        lethal = lethal.copy()
        lethal[start] = False
    if lethal[goal]:
        free = np.argwhere(~lethal)
        if len(free) == 0:
            return None
        goal = tuple(free[np.argmin(((free - np.array(goal)) ** 2).sum(1))])
    cost_l = cost.tolist()
    leth_l = lethal.tolist()
    gsc = {start: 0.0}
    came = {}
    hq = [(0.0, start)]
    gi, gj = goal
    n = 0
    while hq and n < max_exp:
        _, cur = heapq.heappop(hq)
        n += 1
        if cur == goal:
            p = [goal]
            while p[-1] != start:
                p.append(came[p[-1]])
            return p[::-1]
        g0 = gsc[cur]
        ci, cj = cur
        for di, dj, sl in _STEPS:
            i, j = ci + di, cj + dj
            if i < 0 or j < 0 or i >= h or j >= w or leth_l[i][j]:
                continue
            ng = g0 + sl * cost_l[i][j]
            nb = (i, j)
            if ng < gsc.get(nb, 1e18):
                gsc[nb] = ng
                came[nb] = cur
                heapq.heappush(hq, (ng + w_h * math.hypot(gi - i, gj - j), nb))
    return None

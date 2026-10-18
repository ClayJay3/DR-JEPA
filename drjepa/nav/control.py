"""Local control: track the A* route with arcs checked against the map.

Every step: pick a waypoint ~7 m along the route, roll out 17 constant-
curvature arcs (2.4 s), reject arcs that come too close to believed
obstacles (fused map AND the zero-lag fresh-frame guard) or cross believed
steep ground, and score the rest on progress + clearance + alignment +
smoothness -- the privileged expert's recipe on the believed world.

Recovery behaviours (all measured necessary in v9-v12):
  * creep: in a tight gap the conservative arc check may reject everything
    while A* says the corridor is passable -- creep along the route slowly
    instead of thrashing into recovery;
  * map-checked reverse: the map remembers what is behind; stop reversing
    when the rear closes or the front opens (a blind fixed-length reverse
    backed into obstacles the robot had already seen);
  * escape turn when boxed in both ways.
Throttle = min(clearance, turn, goal-distance, gap-width caps). There is
deliberately NO sand brake: in soft ground momentum is what gets you
through (measured: braking for sand quadrupled stalls).
"""

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class ControlConfig:
    """Arc-controller constants."""
    n_arcs: int = 17
    arc_t: float = 2.4
    arc_dt: float = 0.3
    lookahead: float = 7.0
    map_buffer: float = 0.2          # clearance buffer beyond the radius (map)
    guard_buffer: float = 0.1        # ... for the fresh-frame guard (no pose error)
    min_clear: float = 0.05
    haz_reject: float = 0.9          # x tip_roll: arc rejected on fused grade
    stuck_speed: float = 0.35
    stuck_frames: int = 12


class Controller:
    """Arc sampler + creep + recovery + throttle governors."""

    def __init__(self, robot, cfg: ControlConfig = None):
        self.robot, self.cfg = robot, cfg or ControlConfig()
        self.reset()

    def reset(self):
        self.prev_steer = 0.0
        self.recovery = 0
        self.recovery_steer = 0.0
        self.escape = 0
        self.stuck = 0
        self.last_arc = None
        self.last_target = None
        self.mode = "drive"
        self.debug = {}          # arcs, scores, feasibility, speed caps (for viz)

    @property
    def recovering(self):
        return self.recovery > 0 or self.escape > 0

    def target(self, path, pose, goal):
        """Waypoint ~lookahead metres along the route (or the goal)."""
        if path is None or len(path) < 2:
            return goal
        k = int(np.argmin(np.linalg.norm(path - pose, axis=1)))
        seg = path[k:]
        if len(seg) < 2:
            return goal
        along = np.linalg.norm(np.diff(seg, axis=0), axis=1).cumsum()
        w = int(np.searchsorted(along, self.cfg.lookahead))
        return seg[min(w + 1, len(seg) - 1)]

    def _reverse(self, steer):
        self.prev_steer = steer
        self.mode = "reverse"
        return -0.5, steer

    def step(self, bmap, planner, pose, heading_deg, goal, v_meas):
        """Returns (throttle, steer) in [-1, 1]."""
        c, rb = self.cfg, self.robot
        target = self.target(planner.path, pose, goal)
        self.last_target = target
        goal_d = float(np.linalg.norm(goal - pose))
        yaw = math.radians(heading_deg)
        R = rb.radius

        if self.recovery > 0:
            self.recovery -= 1
            back = pose[None] - np.array([math.sin(yaw), math.cos(yaw)])[None] * np.array([[0.9], [1.7], [2.5]])
            rear = float(planner.clearance(bmap, back).min())
            front = bmap.guard_ahead(2, 6)
            if rear < R + 0.3 or front > R + 1.45:
                self.recovery = 0
                if rear < R + 0.3 and front <= R + 1.45:
                    self.escape = 8
                    self.recovery_steer = -self.recovery_steer
            else:
                return self._reverse(self.recovery_steer)
        if self.escape > 0:
            self.escape -= 1
            self.prev_steer = self.recovery_steer
            self.mode = "escape"
            return 0.3, self.recovery_steer

        steers = np.linspace(-1, 1, c.n_arcs)
        yr = np.radians(steers * rb.yaw_rate_max)
        t = np.arange(1, int(c.arc_t / c.arc_dt) + 1) * c.arc_dt
        v_plan = float(np.clip(abs(v_meas), 1.5, rb.v_top))
        seg_yaw = yaw + yr[:, None] * (t - c.arc_dt / 2)[None, :]
        dx = np.cumsum(np.sin(seg_yaw) * v_plan * c.arc_dt, axis=1)
        dz = np.cumsum(np.cos(seg_yaw) * v_plan * c.arc_dt, axis=1)
        pts = np.stack([pose[0] + dx, pose[1] + dz], -1)
        clear = planner.clearance(bmap, pts).min(axis=1) - (R + c.map_buffer)
        clear = np.minimum(clear, bmap.guard_clearance(dx, dz) - (R + c.guard_buffer))
        feasible = clear > c.min_clear
        ai, aj = bmap.index(pts[..., 0], pts[..., 1])
        n = bmap.cfg.cells
        ai, aj = np.clip(ai, 0, n - 1), np.clip(aj, 0, n - 1)
        g_arc = np.where(bmap.GW[ai, aj] > 0.3, bmap.G[ai, aj], 0.0)
        steep = g_arc.max(axis=1) > c.haz_reject * rb.tip_roll
        feasible &= ~steep
        clear = np.where(steep, np.minimum(clear, 0.0), clear)

        self.stuck = self.stuck + 1 if abs(v_meas) < c.stuck_speed else 0
        if self.stuck > c.stuck_frames:
            half = c.n_arcs // 2
            self.recovery_steer = 0.8 if clear[:half].max() > clear[half + 1:].max() else -0.8
            if self.stuck > 40:
                self.escape = 10
                self.recovery_steer = -self.recovery_steer
                self.stuck = 0
            else:
                self.recovery = 14
            return self._reverse(self.recovery_steer)

        if not feasible.any():
            half = c.n_arcs // 2
            side = 0.8 if clear[:half].max() > clear[half + 1:].max() else -0.8
            if bmap.guard_ahead(2, 6) < R + 0.1:
                self.recovery, self.recovery_steer = 14, side
                return self._reverse(side)
            if planner.path is not None:
                bear = math.atan2(target[0] - pose[0], target[1] - pose[1])
                err = (bear - yaw + math.pi) % (2 * math.pi) - math.pi
                steer = float(np.clip(err / math.radians(35.0), -1, 1))
                self.prev_steer = steer
                self.mode = "creep"
                return 0.25, steer
            self.recovery, self.recovery_steer = 14, side
            return self._reverse(side)

        self.debug = {"arcs": pts, "feasible": feasible, "clear": clear, "steers": steers}
        end = pts[:, -1]
        wd = np.linalg.norm(target - pose)
        prog = wd - np.linalg.norm(target[None] - end, axis=1)
        bear = np.arctan2(target[0] - end[:, 0], target[1] - end[:, 1])
        align = np.cos(bear - seg_yaw[:, -1])
        score = 1.4 * prog + 2.2 * np.minimum(clear, 2.5) + 1.5 * align - 1.2 * np.abs(steers - self.prev_steer)
        score[~feasible] = -1e9
        best = int(np.argmax(score))
        steer = float(steers[best])
        self.last_arc = pts[best]
        caps = {"clearance": float(np.clip(clear[best] / 3.5, 0.35, 1.0)),
                "turn": 1.0 - 0.4 * abs(steer),
                "goal": float(np.clip(goal_d / 8.0, 0.3, 1.0))}
        # thread tight gaps slowly: with actuation latency, speed turns a
        # near-miss into a graze
        ahead = bmap.guard_ahead(3, 8)
        caps["gap"] = 0.35 if ahead < R + 1.25 else (0.55 if ahead < R + 2.75 else 1.0)
        throttle = float(np.clip(min(caps.values()), 0.25, 1.0))
        self.debug.update(score=score, best=best, caps=caps, gap_m=ahead)
        self.prev_steer = steer
        self.mode = "drive"
        return throttle, steer

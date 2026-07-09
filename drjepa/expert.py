"""Privileged expert planner (sees the true obstacle map).

Two layers, mirroring a classic autonomy stack:
  * GridPlanner -- coarse A* over an inflated occupancy grid, giving a
    globally-sensible route (finds wall gaps, escapes cul-de-sacs).
  * ArcPlanner  -- local arc sampling with the rover's kinematics, tracking
    a waypoint ~12 m ahead on the A* path while trading off clearance and
    smoothness. When boxed in, it reverses toward the more open side.

Used to generate demonstration labels for behavior cloning and as a
closed-loop baseline in evaluation.
"""

import heapq
import math

import numpy as np


class GridPlanner:
    """A* on an inflated occupancy grid; returns waypoints along the path."""

    RES = 1.5  # metres per cell

    def __init__(self, obs_xz, obs_rad, inflate, start, goal, margin=45.0):
        """Rasterize the (inflated) obstacles into a grid and plan once.

        `inflate` is added to every obstacle radius so the path keeps
        rover-radius clearance; `margin` pads the grid bounds beyond the
        start/goal bounding box so routes can swing around wide walls.
        """
        self.goal = np.asarray(goal, float)
        lo = np.minimum(start, goal) - margin
        hi = np.maximum(start, goal) + margin
        self.lo = lo
        self.hi = hi
        self.nx = int((hi[0] - lo[0]) / self.RES) + 1
        self.nz = int((hi[1] - lo[1]) / self.RES) + 1
        self._obs_xz, self._obs_rad = obs_xz, obs_rad
        self.inflate = inflate
        self.occ = self._build_occ(inflate)
        self.path = None
        self._wp_idx = 0
        self.replan(start)

    def _build_occ(self, inflate):
        """Occupancy grid: cells within `inflate` of any obstacle surface."""
        occ = np.zeros((self.nx, self.nz), bool)
        obs_xz, obs_rad, lo, hi = self._obs_xz, self._obs_rad, self.lo, self.hi
        if len(obs_rad):
            xs = lo[0] + (np.arange(self.nx) + 0.5) * self.RES
            zs = lo[1] + (np.arange(self.nz) + 0.5) * self.RES
            inside = ((obs_xz[:, 0] > lo[0] - 8) & (obs_xz[:, 0] < hi[0] + 8) &
                      (obs_xz[:, 1] > lo[1] - 8) & (obs_xz[:, 1] < hi[1] + 8))
            oxz, orad = obs_xz[inside], obs_rad[inside] + inflate
            for (ox, oz), r in zip(oxz, orad):
                i0 = max(0, int((ox - r - lo[0]) / self.RES))
                i1 = min(self.nx, int((ox + r - lo[0]) / self.RES) + 2)
                j0 = max(0, int((oz - r - lo[1]) / self.RES))
                j1 = min(self.nz, int((oz + r - lo[1]) / self.RES) + 2)
                if i1 <= i0 or j1 <= j0:
                    continue
                dx = xs[i0:i1, None] - ox
                dz = zs[None, j0:j1] - oz
                occ[i0:i1, j0:j1] |= (dx * dx + dz * dz) < r * r
        return occ

    # ---------------- helpers ----------------
    def _cell(self, p):
        """World point -> clamped grid cell (i, j)."""
        i = int(np.clip((p[0] - self.lo[0]) / self.RES, 0, self.nx - 1))
        j = int(np.clip((p[1] - self.lo[1]) / self.RES, 0, self.nz - 1))
        return i, j

    def _world(self, ij):
        """Grid cell -> world coordinates of its center."""
        return np.array([self.lo[0] + (ij[0] + 0.5) * self.RES,
                         self.lo[1] + (ij[1] + 0.5) * self.RES])

    def _free_near(self, ij, rmax=8):
        """Nearest unoccupied cell (spiral search) -- start/goal may sit
        inside the inflation ring of an obstacle."""
        if not self.occ[ij]:
            return ij
        for r in range(1, rmax):
            for di in range(-r, r + 1):
                for dj in range(-r, r + 1):
                    if max(abs(di), abs(dj)) != r:
                        continue
                    i, j = ij[0] + di, ij[1] + dj
                    if 0 <= i < self.nx and 0 <= j < self.nz and not self.occ[i, j]:
                        return (i, j)
        return ij

    # ---------------- A* ----------------
    def replan(self, start):
        """8-connected A* from `start` to the goal over the occupancy grid.

        If no route exists at full inflation the grid is rebuilt once with a
        slimmer safety margin (narrow gaps can disappear under inflation).
        Returns True and stores the path on success.
        """
        s = self._free_near(self._cell(start))
        g = self._free_near(self._cell(self.goal))
        occ = self.occ
        nx, nz = self.nx, self.nz
        gscore = {s: 0.0}
        came = {}
        h0 = math.hypot(g[0] - s[0], g[1] - s[1])
        heap = [(h0, s)]
        found = False
        steps = [(-1, -1, 1.414), (-1, 0, 1.0), (-1, 1, 1.414), (0, -1, 1.0),
                 (0, 1, 1.0), (1, -1, 1.414), (1, 0, 1.0), (1, 1, 1.414)]
        while heap:
            f, cur = heapq.heappop(heap)
            if cur == g:
                found = True
                break
            gc = gscore[cur]
            if f - math.hypot(g[0] - cur[0], g[1] - cur[1]) > gc + 1e-9:
                continue
            for di, dj, w in steps:
                i, j = cur[0] + di, cur[1] + dj
                if not (0 <= i < nx and 0 <= j < nz) or occ[i, j]:
                    continue
                ng = gc + w
                nb = (i, j)
                if ng < gscore.get(nb, 1e18):
                    gscore[nb] = ng
                    came[nb] = cur
                    heapq.heappush(heap, (ng + math.hypot(g[0] - i, g[1] - j), nb))
        if not found:
            # narrow passages can vanish under full inflation: retry once
            # with a slimmer safety margin before giving up
            if self.inflate > 0.9:
                self.inflate = 0.85
                self.occ = self._build_occ(self.inflate)
                return self.replan(start)
            self.path = None
            return False
        cells = [g]
        while cells[-1] != s:
            cells.append(came[cells[-1]])
        cells.reverse()
        self.path = np.array([self._world(c) for c in cells])
        self.path = np.vstack([self.path, self.goal[None]])
        self._wp_idx = 0
        return True

    # ---------------- queries ----------------
    def waypoint(self, x, z, lookahead=12.0):
        """Point ~lookahead metres ahead along the path (monotonic tracker)."""
        if self.path is None:
            return self.goal
        p = np.array([x, z])
        w = self.path[self._wp_idx:self._wp_idx + 40]
        self._wp_idx += int(np.argmin(np.linalg.norm(w - p, axis=1)))
        seg = self.path[self._wp_idx:]
        d = np.linalg.norm(np.diff(seg, axis=0), axis=1).cumsum() if len(seg) > 1 \
            else np.array([0.0])
        k = int(np.searchsorted(d, lookahead))
        return seg[min(k + 1, len(seg) - 1)]

    def deviation(self, x, z):
        """Distance (m) from the rover to the nearest stored path point."""
        if self.path is None:
            return 0.0
        w = self.path[max(0, self._wp_idx - 5):self._wp_idx + 40]
        return float(np.min(np.linalg.norm(w - np.array([x, z]), axis=1)))


class ArcPlanner:
    """Local controller: samples constant-curvature arcs, tracks the A* path.

    Runs every control step. Falls back to reversing recovery (and a
    forward-turn escape if reversing is also blocked) when boxed in.
    """

    N_ARCS = 17
    HORIZON_S = 2.6
    ARC_DT = 0.26

    def __init__(self, sim, rng=None):
        """Bind to a simulator (privileged access) and plan the global route."""
        self.sim = sim
        self.rng = rng or np.random.default_rng()
        self.prev_steer = 0.0
        self.recovery = 0          # frames of reversing left
        self.recovery_steer = 0.0
        self.escape = 0            # frames of slow forward-turn escape
        self.caution = 0           # frames of extra margin after a contact
        self.steers = np.linspace(-1.0, 1.0, self.N_ARCS)
        self.n_steps = int(self.HORIZON_S / self.ARC_DT)
        self.grid = GridPlanner(sim.obs_xz, sim.obs_rad,
                                sim.cfg.rover_radius + 0.45,
                                (sim.x, sim.z), (sim.goal_x, sim.goal_z))
        self._since_replan = 0

    # ------------------------------------------------------------------
    def _rollout(self, speed):
        """Positions of all candidate arcs: (N_ARCS, n_steps, 2)."""
        sim = self.sim
        yaw0 = math.radians(sim.yaw)
        yaw_rates = np.radians(self.steers * sim.cfg.yaw_rate_max)
        t = (np.arange(1, self.n_steps + 1) * self.ARC_DT)[None, :]
        yaws = yaw0 + yaw_rates[:, None] * t
        # integrate positions incrementally (mid-point yaw per segment)
        seg_yaws = yaw0 + yaw_rates[:, None] * (t - self.ARC_DT / 2)
        dx = np.cumsum(np.sin(seg_yaws) * speed * self.ARC_DT, axis=1)
        dz = np.cumsum(np.cos(seg_yaws) * speed * self.ARC_DT, axis=1)
        pts = np.stack([sim.x + dx, sim.z + dz], axis=-1)
        return pts, yaws

    def _arc_clearances(self, pts):
        """Min clearance along each arc: (N_ARCS,)."""
        sim = self.sim
        if len(sim.obs_rad) == 0:
            return np.full(len(pts), 99.0)
        # only nearby obstacles matter
        p0 = np.array([sim.x, sim.z])
        d0 = np.linalg.norm(sim.obs_xz - p0, axis=1)
        near = d0 < (self.HORIZON_S * max(abs(self.sim.v), 3.0) + 12.0 + sim.obs_rad)
        if not near.any():
            return np.full(len(pts), 99.0)
        oxz = sim.obs_xz[near]                       # (M, 2)
        # extra margin absorbs steering lag / wheel slip the arcs don't model
        orad = sim.obs_rad[near] + sim.cfg.rover_radius + 0.25
        d = np.linalg.norm(pts[:, :, None, :] - oxz[None, None, :, :], axis=-1)
        clear = d - orad[None, None, :]
        return clear.min(axis=(1, 2))

    # ------------------------------------------------------------------
    def plan(self):
        """Returns the expert (throttle, steer) command for the current state."""
        sim = self.sim
        gd = sim.goal_dist_true()

        # --- recovery: reverse out of a trap ---
        if self.recovery > 0:
            if sim.collided_now:
                # something behind us too: switch to a slow forward turn
                self.recovery = 0
                self.escape = int(self.rng.uniform(8, 14))
                self.recovery_steer = -self.recovery_steer
            else:
                self.recovery -= 1
                self.prev_steer = self.recovery_steer
                return -0.55, self.recovery_steer
        if self.escape > 0:
            if sim.collided_now:
                self.escape = 0    # blocked again: fall through and re-decide
            else:
                self.escape -= 1
                self.prev_steer = self.recovery_steer
                return 0.35, self.recovery_steer

        blocked_contact = sim.collided_now or sim.clearance() < 0.05
        speed_plan = float(np.clip(abs(sim.v), 2.0, sim.v_max))
        pts, yaws = self._rollout(speed_plan)
        clear = self._arc_clearances(pts)
        margin = 0.45 if self.caution > 0 else 0.10
        self.caution = max(0, self.caution - 1)
        feasible = clear > margin

        if blocked_contact or not feasible.any():
            # choose reverse turn direction toward the more open side
            left = clear[: self.N_ARCS // 2].max()
            right = clear[self.N_ARCS // 2 + 1:].max()
            # steering while reversing turns the nose the other way
            self.recovery_steer = 0.8 if left > right else -0.8
            self.recovery = int(self.rng.uniform(12, 22))
            self.caution = 40
            self.prev_steer = self.recovery_steer
            return -0.55, self.recovery_steer

        # --- score arcs against the A* waypoint ---
        self._since_replan += 1
        if self._since_replan > 40 or self.grid.deviation(sim.x, sim.z) > 6.0:
            self.grid.replan((sim.x, sim.z))
            self._since_replan = 0
        wx, wz = self.grid.waypoint(sim.x, sim.z, 12.0)

        end = pts[:, -1, :]                       # (N, 2)
        wd = math.hypot(wx - sim.x, wz - sim.z)
        d_end = np.hypot(wx - end[:, 0], wz - end[:, 1])
        progress = wd - d_end
        end_bearing = np.degrees(np.arctan2(wx - end[:, 0], wz - end[:, 1]))
        align = np.cos(np.radians(end_bearing) - yaws[:, -1])
        score = (1.4 * progress
                 + 2.2 * np.minimum(clear, 2.5)
                 + 1.5 * align
                 - 1.2 * np.abs(self.steers - self.prev_steer))
        score[~feasible] = -1e9
        best = int(np.argmax(score))
        steer = float(self.steers[best])

        # --- throttle: clearance-, curvature- and goal-aware ---
        v_clear = np.clip(clear[best] / 3.5, 0.50, 1.0)
        v_turn = 1.0 - 0.40 * abs(steer)
        v_goal = np.clip(gd / 8.0, 0.30, 1.0)
        throttle = float(np.clip(min(v_clear, v_turn, v_goal), 0.25, 1.0))

        self.prev_steer = steer
        return throttle, steer


class NoiseInjector:
    """Occasional steering perturbations during data generation.

    The *executed* command is perturbed while the *logged label* stays clean,
    so the dataset contains off-policy states with corrective labels
    (ChauffeurNet-style synthetic disturbances).
    """

    def __init__(self, rng, p_start=0.012):
        """`p_start` is the per-step chance of starting a perturbation burst."""
        self.rng = rng
        self.p_start = p_start
        self.left = 0
        self.offset = 0.0
        self.thr_scale = 1.0

    def apply(self, throttle, steer, clearance=99.0):
        """Return the (possibly perturbed) command to actually execute."""
        # never perturb while squeezed close to obstacles or reversing
        if clearance < 1.5 or throttle < 0:
            self.left = 0
            return throttle, steer
        if self.left > 0:
            self.left -= 1
            return (float(np.clip(throttle * self.thr_scale, -1, 1)),
                    float(np.clip(steer + self.offset, -1, 1)))
        if self.rng.random() < self.p_start:
            self.left = int(self.rng.uniform(4, 10))
            self.offset = self.rng.uniform(-0.45, 0.45)
            self.thr_scale = self.rng.uniform(0.7, 1.0)
        return throttle, steer

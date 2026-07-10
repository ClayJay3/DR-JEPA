"""Streaming closed-loop inference.

MapPilot is the navigator: one frozen-DINOv2 pass per frame feeds the
MapDecoder, whose per-frame terrain wedges (occupancy / visibility /
elevation / sand / tip-hazard, plus a danger scalar) fuse into a
PERSISTENT world-anchored log-odds map (the run-long spatial memory --
nothing observed is ever forgotten). A route is planned on that learned
map with A* and tracked with an arc-sampling local controller -- the same
plan-on-a-map loop the privileged expert uses, except the map comes from
the camera instead of ground truth.

(The reactive behavior-cloned Pilot and its JEPA latent-MPC shield were
removed with the temporal branch after v11; `git log` has them.)
"""

import collections
import heapq
import math

import cv2
import numpy as np
import torch

from .config import Config
from .model import RoverJEPA, Backbone
from .simulator import latlon_to_meters


# ==========================================================================
# Map-based navigator
# ==========================================================================
class MapPilot:
    """Perceive -> remember -> plan -> act.

    Memory: a world-anchored occupancy map in log-odds, updated Bayesian-
    style from the perception net's calibrated per-cell predictions. The map
    covers the whole run and persists indefinitely (grows by re-centering if
    the rover drives off the edge), giving unbounded temporal context at
    O(1) cost per step.
    """

    DT = 0.1
    GPS_GAIN = 0.08          # complementary pose filter: pull toward GPS
    LODDS_CLAMP = 6.0
    OCC_THRESH = 0.6         # fused probability considered blocking
    PRIOR_LOGIT = -3.5       # occupancy base rate ~2-3%: the network's
    #                          decision boundary sits here, not at logit 0.
    #                          Evidence = logit - prior (Bayesian update).
    HAZ_PRIOR_LOGIT = -2.2   # steep-ground base rate ~3-5%; same
    #                          prior-correction logic as occupancy (set just
    #                          above the near-field flat-ground p99 of -1.97)
    GUARD_EVIDENCE = 2.0     # single-frame evidence that blocks the guard
    REPLAN_EVERY = 3         # control steps between A* replans
    N_ARCS = 17
    ARC_T = 2.4              # seconds of arc rollout
    ARC_DT = 0.3

    def __init__(self, checkpoint, device="cuda", vo=True, complete=False):
        """Restore the model and precompute wedge-cell geometry.

        vo=False disables the scan-matching pose correction. complete=True
        lets map-space JEPA predictions bias the planner's costs for
        unobserved cells -- measured NEUTRAL-TO-NEGATIVE in this sim's
        closed loop (the wall-continuation prior also paints over unseen
        gaps, which is anti-exploratory), so it defaults OFF; the completer
        itself stays trained and drives the ghost visualization either way.
        """
        ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
        self.cfg = Config.from_dict(ckpt["config"])
        mc = self.cfg.model
        self.device = device
        self.vo = vo             # visual-odometry map alignment
        self.complete = complete # map-space JEPA in the planner
        self.model = RoverJEPA(mc).to(device)
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()
        self.backbone = Backbone(mc, device=device)
        # per-channel temperature calibration for the completer (fitted on
        # held-out real belief maps by `drjepa.py tune_completer`); T=1
        # when the checkpoint predates calibration
        calib = ckpt.get("comp_calib", {"t": [1.0, 1.0, 1.0]})
        self.comp_temp = torch.tensor(calib["t"], dtype=torch.float32,
                                      device=device).view(3, 1, 1)

        self.res = mc.wedge_res
        self.wc = mc.wedge_cells
        self.mc_cells = mc.map_cells
        # wedge cell centers in the rover frame (x right, z forward)
        half = self.wc * self.res / 2.0
        xs = (np.arange(self.wc) + 0.5) * self.res - half
        zs = (np.arange(self.wc) + 0.5) * self.res
        self._wx, self._wz = np.meshgrid(xs, zs, indexing="ij")

        self.reset()

    def reset(self):
        """Wipe the map, pose filter, and histories (new run = new memory)."""
        n = self.mc_cells
        self.L = np.zeros((n, n), np.float32)      # occupancy log-odds
        self.E = np.zeros((n, n), np.float32)      # fused elevation (m)
        self.EW = np.zeros((n, n), np.float32)     # elevation weight (conf.)
        self.LS = np.zeros((n, n), np.float32)     # sand log-odds
        self.LH = np.zeros((n, n), np.float32)     # terrain-hazard log-odds
        self.pred = None                            # completer output (probs)
        self.pred_origin = None
        self.origin = None       # lat/lon anchor of the local frame
        self.map_corner = None   # world coords of map cell (0, 0)
        self.pose = None         # filtered (x, z) in local metres
        self.step_i = 0
        self.path = None
        self.prev_steer = 0.0
        self.recovery = 0
        self.recovery_steer = 0.0
        self.escape = 0
        self._last_pos = None
        self._stuck_frames = 0
        self.last_arc = None
        self.last_danger = 0.0
        # multi-frame perception history: (tokens, (speed, steer)) per step
        m_off = max(self.cfg.model.frame_offsets)
        self._tok_hist = collections.deque(maxlen=m_off + 1)
        self._last_cmd_steer = 0.0

    # ---------------- pose ----------------
    def _update_pose(self, sensors):
        """Complementary pose filter: integrate odometry, pull toward GPS.

        Odometry gives smooth short-term motion; the weak GPS correction
        (GPS_GAIN per step) bounds the long-term drift. Also converts the
        goal fix into the local metric frame and re-centers the map if the
        rover nears its edge.
        """
        if self.origin is None:
            self.origin = (sensors["lat"], sensors["lon"])
        gx, gz = latlon_to_meters(sensors["lat"], sensors["lon"], *self.origin)
        head = math.radians(sensors["heading"])
        if self.pose is None:
            self.pose = np.array([gx, gz], float)
            # center the map on the start position
            c = self.mc_cells * self.res / 2.0
            self.map_corner = self.pose - c
        else:
            v = sensors["speed"]
            self.pose += v * self.DT * np.array([math.sin(head), math.cos(head)])
            self.pose += self.GPS_GAIN * (np.array([gx, gz]) - self.pose)
        self.heading = sensors["heading"]
        gx2, gz2 = latlon_to_meters(sensors["goal_lat"], sensors["goal_lon"],
                                    *self.origin)
        self.goal = np.array([gx2, gz2], float)
        self._maybe_recenter()

    def _maybe_recenter(self):
        """Grow (re-center) the map if the rover approaches its edge."""
        ij = (self.pose - self.map_corner) / self.res
        margin = self.mc_cells * 0.15
        if np.all(ij > margin) and np.all(ij < self.mc_cells - margin):
            return
        shift_cells = np.floor(ij - self.mc_cells / 2.0).astype(int)
        x0, z0 = shift_cells
        xs_src = slice(max(0, x0), min(self.mc_cells, self.mc_cells + x0))
        xs_dst = slice(max(0, -x0), max(0, -x0) + (xs_src.stop - xs_src.start))
        zs_src = slice(max(0, z0), min(self.mc_cells, self.mc_cells + z0))
        zs_dst = slice(max(0, -z0), max(0, -z0) + (zs_src.stop - zs_src.start))
        for name in ("L", "E", "EW", "LS", "LH"):  # scroll every belief channel
            src = getattr(self, name)
            new = np.zeros_like(src)
            new[xs_dst, zs_dst] = src[xs_src, zs_src]
            setattr(self, name, new)
        self.map_corner = self.map_corner + shift_cells * self.res

    # ---------------- perception + fusion ----------------
    @torch.no_grad()
    def _perceive(self, frame_bgr, sensors):
        """Camera frame -> (occ_logit, conf, danger). Overridable for
        oracle-perception testing."""
        mc = self.cfg.model
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        tokens = self.backbone(torch.from_numpy(rgb).permute(2, 0, 1)[None])
        mot = (sensors["speed"] / mc.speed_norm, self._last_cmd_steer)
        self._tok_hist.append((tokens, mot))

        # multi-frame stack: current frame + lookbacks (clamped in warm-up)
        hist = list(self._tok_hist)
        toks, mots = [], []
        for off in mc.frame_offsets:
            t, m = hist[max(0, len(hist) - 1 - off)]
            toks.append(t)
            mots.extend(m)
        tok_stack = torch.stack(toks, dim=1)              # (1, F, nt, fd)
        mot_t = torch.tensor([mots], dtype=torch.float32, device=tokens.device)
        occ_logit, conf_logit, elev, sand_logit, haz_logit, d_logit = \
            self.model.map_decoder(tok_stack, mot_t)
        occ_logit = occ_logit[0].float().cpu().numpy()
        cl = np.clip(conf_logit[0].float().cpu().numpy(), -30.0, 30.0)
        conf = 1.0 / (1.0 + np.exp(-cl))
        elev = elev[0].float().cpu().numpy()
        sand_logit = sand_logit[0].float().cpu().numpy()
        haz_logit = haz_logit[0].float().cpu().numpy()
        # danger rides the same trunk: no separate temporal branch to run
        danger = torch.sigmoid(d_logit[0].float()).item()
        return occ_logit, conf, elev, sand_logit, haz_logit, danger

    PAINT_RANGE_CELLS = 24   # trust the near 12 m for map fusion
    POS_EVIDENCE_SCALE = 0.45  # correlated false positives accumulate; slow
    #                            them down relative to free-space evidence
    DECAY = 0.9985           # per-step log-odds decay (half-life ~46 s):
    #                          phantom mass fades unless re-confirmed
    VO_WINDOW = 2            # scan-matching search radius (cells)
    VO_GAIN = 0.2            # fraction of the matched offset fed back to pose
    VO_MARGIN = 10.0         # required score edge over the zero shift
    VO_MIN_L = 1.2           # only match against well-established map cells

    def _vo_align(self, ii, jj, upd, ok):
        """Visual-odometry pose correction by scan matching.

        The GPS-filtered pose drifts, so a wedge painted now can land up to
        ~1 m off the same obstacle painted a minute ago, smearing the map.
        Before fusing, slide the wedge over the existing map and find the
        translation that best agrees with prior evidence (dot product of new
        evidence with stored log-odds); feed that offset back into the pose
        filter. GPS still anchors the absolute frame, this only removes the
        relative drift between observations.
        """
        if self.recovery > 0 or self.escape > 0:
            return                                # unreliable while thrashing
        iv, jv, uv = ii[ok], jj[ok], upd[ok]
        # only match where the new wedge is opinionated *positive* (obstacle
        # structure gives the correlation peak; free space is featureless)
        strong = uv > 0.4
        if strong.sum() < 25:
            return
        iv, jv, uv = iv[strong], jv[strong], uv[strong]
        w = self.VO_WINDOW
        n = self.mc_cells
        Lm = np.where(np.abs(self.L) > self.VO_MIN_L, self.L, 0.0)
        best, best_shift, zero = -1e18, (0, 0), 0.0
        for di in range(-w, w + 1):
            i2 = np.clip(iv + di, 0, n - 1)
            for dj in range(-w, w + 1):
                j2 = np.clip(jv + dj, 0, n - 1)
                s = float(np.dot(uv, Lm[i2, j2])) - 1.5 * (di * di + dj * dj)
                if di == 0 and dj == 0:
                    zero = s
                if s > best:
                    best, best_shift = s, (di, dj)
        if best_shift != (0, 0) and best - zero > self.VO_MARGIN:
            self.pose += self.VO_GAIN * np.array(best_shift) * self.res

    SAND_PRIOR = -2.4        # soft-ground base rate is ~8%

    def _paint(self, occ_logit, conf, elev=None, sand_logit=None,
               haz_logit=None):
        """Fuse one wedge into the persistent multi-channel belief map.

        Occupancy, sand, and steep-ground hazard use prior-corrected
        log-odds; elevation is a confidence-weighted running mean (it is a
        continuous quantity, not a probability).
        """
        conf = conf.copy()
        conf[:, self.PAINT_RANGE_CELLS:] = 0.0            # untrusted far field
        conf[conf < 0.55] = 0.0                           # low-vis cells
        self.L *= self.DECAY
        yaw = math.radians(self.heading)
        cy, sy = math.cos(yaw), math.sin(yaw)
        evidence = occ_logit - self.PRIOR_LOGIT
        evidence = np.where(evidence > 0, evidence * self.POS_EVIDENCE_SCALE,
                            evidence)
        upd = np.clip(evidence, -4.0, 4.0) * conf * 0.55
        self.last_evidence = evidence          # for visualization
        self.last_conf = conf

        def cells():
            """Wedge cell centers -> map indices at the CURRENT pose."""
            wx = self.pose[0] + self._wx * cy + self._wz * sy
            wz = self.pose[1] - self._wx * sy + self._wz * cy
            ii = np.floor((wx - self.map_corner[0]) / self.res).astype(int)
            jj = np.floor((wz - self.map_corner[1]) / self.res).astype(int)
            ok = ((ii >= 0) & (ii < self.mc_cells) &
                  (jj >= 0) & (jj < self.mc_cells) & (conf > 0.0))
            return ii, jj, ok

        ii, jj, ok = cells()
        if self.vo:
            pose_before = self.pose.copy()
            self._vo_align(ii, jj, upd, ok)
            if not np.array_equal(pose_before, self.pose):
                ii, jj, ok = cells()          # repaint at the corrected pose
        np.add.at(self.L, (ii[ok], jj[ok]), upd[ok])
        # positive cap below negative cap: occupied belief must stay
        # revisable when later evidence disagrees
        np.clip(self.L, -self.LODDS_CLAMP, 3.5, out=self.L)

        if elev is not None:
            # elevation in the map is ABSOLUTE (relative to the local
            # origin): wedge elevation is relative to the rover's ground,
            # whose absolute height we estimate from already-fused cells
            # near the rover (zero at start).
            ci = int((self.pose[0] - self.map_corner[0]) / self.res)
            cj = int((self.pose[1] - self.map_corner[1]) / self.res)
            w_here = self.EW[ci - 2:ci + 3, cj - 2:cj + 3]
            e_here = self.E[ci - 2:ci + 3, cj - 2:cj + 3]
            h0 = float((e_here * w_here).sum() / w_here.sum()) \
                if w_here.sum() > 0.5 else 0.0
            ev = elev[ok] + h0
            wgt = conf[ok]
            iv, jv = ii[ok], jj[ok]
            # running weighted mean: E <- (E*W + e*w) / (W + w)
            old = self.E[iv, jv] * self.EW[iv, jv]
            self.EW[iv, jv] = np.minimum(self.EW[iv, jv] + wgt, 20.0)
            self.E[iv, jv] = (old + ev * wgt) / np.maximum(self.EW[iv, jv],
                                                           1e-6)
        if sand_logit is not None:
            sev = np.clip(sand_logit - self.SAND_PRIOR, -4.0, 4.0)
            sev = np.where(sev > 0, sev * 0.5, sev)
            np.add.at(self.LS, (ii[ok], jj[ok]), sev[ok] * conf[ok] * 0.55)
            np.clip(self.LS, -6.0, 4.0, out=self.LS)

        if haz_logit is not None:
            # terrain-hazard channel from the DIRECTLY SUPERVISED steep-
            # ground head. Two dead ends are documented here so they are
            # not retried: (1) gradients of the FUSED elevation map have
            # pose-drift seams between frames that once produced phantom
            # lethal walls; (2) gradients of the per-wedge PREDICTED
            # elevation carry no steepness signal at all (regression
            # smooths banks flat -- measured: true 0.5-grade banks scored
            # lower than mild slopes). Classification from visual cues is
            # the only version that separates.
            hev = np.clip(haz_logit - self.HAZ_PRIOR_LOGIT, -4.0, 4.0)
            hev = np.where(hev > 0, hev * self.POS_EVIDENCE_SCALE, hev)
            # positive hazard evidence is only trustworthy NEAR: measured
            # flat-ground FP rates by range band are 2% (0-4 m), 4% (4-8 m)
            # and 10% (8-12 m) -- the far band alone floods the map with
            # phantom lethal terrain (v12 timeouts). Negative evidence is
            # safe at any range. The lower positive cap (vs occupancy's
            # +3.5) lets later clean views wash a phantom out quickly.
            att = np.ones_like(hev)
            att[:, 8:16] = 0.5
            att[:, 16:] = 0.25
            hev = np.where(hev > 0, hev * att, hev)
            np.add.at(self.LH, (ii[ok], jj[ok]), hev[ok] * conf[ok] * 0.55)
            np.clip(self.LH, -6.0, 2.0, out=self.LH)

        # zero-lag near-field guard: distance field of the freshest wedge in
        # the CURRENT rover frame (immune to pose-filter drift)
        blocked = ((evidence > self.GUARD_EVIDENCE) & (conf > 0.5)).astype(np.uint8)
        self._wedge_dist = cv2.distanceTransform(
            1 - blocked, cv2.DIST_L2, 3) * self.res
        self._wedge_yaw = math.radians(self.heading)
        self._wedge_pose = self.pose.copy()

    @torch.no_grad()
    def _complete_map(self):
        """Run the map-space JEPA completer on the current belief map.

        Builds the completer's input (occupancy prob, believed hazard,
        sand prob, observed mask) as a comp_cells x comp_cells crop at 1 m
        around the rover, and returns predicted (occ, hazard, sand)
        probability grids in the same frame. Cached per control step.
        The output is a PREDICTION layer: it shapes planning costs and the
        ghost visualization but is never fused into the evidence map.
        """
        if getattr(self, "_pred_step", -1) == self.step_i:
            return self.pred
        comp_in, origin = self._completion_input()
        with torch.no_grad(), \
                torch.autocast(self.device, enabled=self.device != "cpu"):
            logit = self.model.map_completer(
                torch.from_numpy(comp_in)[None].to(self.device))
            # temperature calibration: a planner acting on "p=0.7" needs
            # 0.7 to actually mean 70% (fitted per channel on real beliefs)
            logit = logit[0].float() / self.comp_temp
        pred = torch.sigmoid(logit).cpu().numpy()
        self.pred = pred
        self.pred_origin = origin
        self.pred_observed = comp_in[3]
        self._pred_step = self.step_i
        return pred

    def _completion_input(self):
        """Build the completer's input from the current belief map.

        Returns (comp_in, origin): a (4, G, G) float32 stack -- observed
        occupancy prob, believed hazard, observed sand prob, observed mask
        -- on the comp_cells x comp_cells 1 m grid centered on the rover,
        plus the world coords of grid cell (0, 0). Shared verbatim between
        deployment and the belief-snapshot logger so the completer trains
        on byte-identical inputs to what it sees when driving.
        """
        mc = self.cfg.model
        G, cres = mc.comp_cells, mc.comp_res
        half_m = G * cres / 2.0
        x0 = self.pose[0] - half_m
        z0 = self.pose[1] - half_m
        # sample belief channels at the 1 m completion grid
        xs = x0 + (np.arange(G) + 0.5) * cres
        zs = z0 + (np.arange(G) + 0.5) * cres
        ii = np.clip(((xs - self.map_corner[0]) / self.res).astype(int),
                     0, self.mc_cells - 1)
        jj = np.clip(((zs - self.map_corner[1]) / self.res).astype(int),
                     0, self.mc_cells - 1)
        Lc = self.L[np.ix_(ii, jj)]
        EWc = self.EW[np.ix_(ii, jj)]
        LSc = self.LS[np.ix_(ii, jj)]
        LHc = self.LH[np.ix_(ii, jj)]
        occ_p = 1.0 / (1.0 + np.exp(-np.clip(Lc, -10, 10)))
        known_e = EWc > 1.0
        hazard = (1.0 / (1.0 + np.exp(-np.clip(LHc, -10, 10)))
                  > 0.6).astype(np.float32)
        sand_p = 1.0 / (1.0 + np.exp(-np.clip(LSc, -10, 10)))
        observed = ((np.abs(Lc) > 0.4) | known_e).astype(np.float32)
        comp_in = np.stack([occ_p * observed, hazard, sand_p * observed,
                            observed]).astype(np.float32)
        return comp_in, np.array([x0, z0])

    # ---------------- planning ----------------
    def _plan_window(self):
        """Sub-window of the map covering rover + goal, at planning scale."""
        lo = np.minimum(self.pose, self.goal) - 16.0
        hi = np.maximum(self.pose, self.goal) + 16.0
        i0 = int(max(0, (lo[0] - self.map_corner[0]) / self.res))
        j0 = int(max(0, (lo[1] - self.map_corner[1]) / self.res))
        i1 = int(min(self.mc_cells, (hi[0] - self.map_corner[0]) / self.res + 1))
        j1 = int(min(self.mc_cells, (hi[1] - self.map_corner[1]) / self.res + 1))
        return i0, j0, i1, j1

    def _replan(self):
        """Plan a route to the goal over the believed map.

        Builds a costmap from the fused occupancy (soft cost rising with
        probability + inflation from the distance field, small penalty for
        unknown space) and runs weighted A* on a 2x-downsampled window.
        Falls back to a slimmer lethal radius if the map claims no route
        exists -- the map can be wrong, the planner should not deadlock.
        """
        i0, j0, i1, j1 = self._plan_window()
        L = self.L[i0:i1, j0:j1]
        prob = 1.0 / (1.0 + np.exp(-L))
        occ = (prob > self.OCC_THRESH).astype(np.uint8)
        # distance (m) to the nearest believed obstacle, for arcs + inflation
        free = (1 - occ).astype(np.uint8)
        dist_cells = cv2.distanceTransform(free, cv2.DIST_L2, 3)
        self._dist_m = dist_cells * self.res
        self._win = (i0, j0, i1, j1)

        # planning grid: 2x downsample for speed
        ph, pw = (i1 - i0) // 2, (j1 - j0) // 2
        if ph < 2 or pw < 2:
            self.path = None
            return
        p2 = prob[:ph * 2, :pw * 2].reshape(ph, 2, pw, 2).max(axis=(1, 3))
        d2 = self._dist_m[:ph * 2, :pw * 2].reshape(ph, 2, pw, 2).min(axis=(1, 3))
        lethal = d2 < 0.9                       # rover radius + margin
        cost = 1.0 + 6.0 * p2 + np.where(d2 < 2.0, (2.0 - d2) * 2.0, 0.0)

        # ---- believed terrain: hazard channel (per-wedge grades), sand ----
        EW = self.EW[i0:i1, j0:j1]
        known_e = EW > 1.0
        hazp = 1.0 / (1.0 + np.exp(-self.LH[i0:i1, j0:j1]))
        h2 = hazp[:ph * 2, :pw * 2].reshape(ph, 2, pw, 2).max(axis=(1, 3))
        lethal = lethal | (h2 > 0.7)
        cost += 5.0 * np.clip(h2 / 0.7, 0, 1)
        sandp = 1.0 / (1.0 + np.exp(-self.LS[i0:i1, j0:j1]))
        s2 = sandp[:ph * 2, :pw * 2].reshape(ph, 2, pw, 2).max(axis=(1, 3))
        cost += 3.0 * s2

        # ---- map-space JEPA: anticipate the unobserved cells ----
        observed = ((np.abs(L) > 0.4) | known_e)
        o2 = observed[:ph * 2, :pw * 2].reshape(ph, 2, pw, 2).max(axis=(1, 3))
        # mild flat penalty for unknown space (exploration is not free)
        cost += (~o2) * 0.3
        if self.complete and getattr(self, "use_invite", True):
            pred = self._complete_map()
            if pred is not None:
                # sample the completer's 1 m grid at planning-cell centers
                pi = ((self.map_corner[0] + (i0 + np.arange(ph) * 2 + 1) *
                       self.res) - self.pred_origin[0]).astype(int)
                pj = ((self.map_corner[1] + (j0 + np.arange(pw) * 2 + 1) *
                       self.res) - self.pred_origin[1]).astype(int)
                valid = ((pi >= 0) & (pi < pred.shape[1]))[:, None] & \
                        ((pj >= 0) & (pj < pred.shape[2]))[None, :]
                pi = np.clip(pi, 0, pred.shape[1] - 1)
                pj = np.clip(pj, 0, pred.shape[2] - 1)
                pocc = pred[0][np.ix_(pi, pj)]
                phaz = pred[1][np.ix_(pi, pj)]
                # INVITE-ONLY integration: predictions may make unknown
                # space CHEAPER (confident-open pulls the route toward the
                # most-likely-real gap) but never more expensive. Raising
                # costs on predicted walls measured anti-exploratory in
                # v10: the completer's wall-continuation prior paints over
                # exactly the unseen gap the rover should be probing for.
                # Scale: unknown cells implicitly cost ~4.3 (p=0.5 through
                # the 6*p occupancy term + the 0.3 exploration penalty) vs
                # ~1.2 for observed-free, so a meaningful invite must
                # cancel most of that implicit penalty; the 1.35 floor
                # keeps seen-free ground always (slightly) preferred.
                popen = 1.0 - np.maximum(pocc, phaz)
                invite = np.clip((popen - 0.55) / 0.35, 0.0, 1.0)
                inv_mask = ~o2 & valid
                cost = np.where(inv_mask,
                                np.maximum(cost - 2.9 * invite, 1.35), cost)

        # ---- plan-consistency prior (route hysteresis) ----
        # Near-tie left/right routes around an obstacle otherwise flip on
        # every replan (each evidence/decay jitter breaks the tie the
        # other way), the ~7 m waypoint jumps sides, and the alternating
        # steering integrates to driving AT the obstacle. Discount cells
        # near the incumbent route so a challenger must beat it by a real
        # margin, not by epsilon; genuinely better routes (or new
        # evidence blocking the old one) still win.
        if self.path is not None and len(self.path) > 1:
            pi_prev = (self.path[:, 0] - self.map_corner[0]) / self.res - i0
            pj_prev = (self.path[:, 1] - self.map_corner[1]) / self.res - j0
            inb = ((pi_prev >= 0) & (pi_prev < ph * 2) &
                   (pj_prev >= 0) & (pj_prev < pw * 2))
            if inb.sum() >= 2:
                on_prev = np.ones((ph, pw), np.uint8)
                on_prev[(pi_prev[inb] // 2).astype(int),
                        (pj_prev[inb] // 2).astype(int)] = 0
                d_prev = cv2.distanceTransform(on_prev, cv2.DIST_L2, 3) \
                    * (2 * self.res)
                # 0.45 within 3 m halves the flip rate; the clearance gate
                # (no discount within 1.6 m of believed obstacles) is
                # essential -- without it the prior partially offsets the
                # obstacle-inflation penalty and the planner keeps
                # threading tight corridors it should widen out of
                # (measured: contacts 1.31 -> 1.91/ep ungated)
                # 0.45 within 3 m of the incumbent route, only on cells
                # >1.6 m clear of believed obstacles. Measured (108 eps
                # per arm): flips halve; success +1.9 / SPL +0.035 / tips
                # equal vs no prior; grazes rise ~+0.6/ep because the old
                # flip-flop wobble was accidental clearance margin -- the
                # commitment threads gaps it previously dithered at. A
                # 2.2 m gate trades that back (contacts 1.48, cf 65%) at
                # -0.9 success / +1.8 tips; mission metrics won.
                disc = 0.45 * np.clip(1.0 - d_prev / 3.0, 0.0, 1.0)
                disc *= (d2 > 1.6)
                cost = np.maximum(cost - disc, 0.8)

        def to_cell(p):
            """World point -> clamped planning-grid cell."""
            return (int(np.clip((p[0] - self.map_corner[0]) / self.res - i0, 0,
                                ph * 2 - 1) // 2),
                    int(np.clip((p[1] - self.map_corner[1]) / self.res - j0, 0,
                                pw * 2 - 1) // 2))

        start = to_cell(self.pose)
        goal = to_cell(self.goal)
        path = self._astar(cost, lethal, start, goal, (ph, pw))
        if path is None:
            # believed-blocked: relax the OBSTACLE inflation once (maybe the
            # map is wrong about clearance) -- but never believed-fatal
            # slopes: a tip-over is terminal, an obstacle graze is not
            path = self._astar(cost, (d2 < 0.5) | (h2 > 0.8),
                               start, goal, (ph, pw))
        if path is None:
            self.path = None
            return
        pts = np.array([[self.map_corner[0] + (i0 + (a + 0.5) * 2) * self.res,
                         self.map_corner[1] + (j0 + (b + 0.5) * 2) * self.res]
                        for a, b in path])
        self.path = pts

    @staticmethod
    def _astar(cost, lethal, start, goal, shape):
        """Weighted A* (heuristic x1.4) over an 8-connected cost grid.

        The inflated heuristic trades a slightly suboptimal path for far
        fewer expansions; an expansion cap bounds worst-case latency.
        Returns the cell path or None.
        """
        h, w = shape
        if lethal[start]:
            lethal = lethal.copy()
            lethal[start] = False
        if lethal[goal]:
            # aim at the nearest non-lethal cell to the goal
            free = np.argwhere(~lethal)
            if len(free) == 0:
                return None
            goal = tuple(free[np.argmin(((free - np.array(goal)) ** 2).sum(1))])
        gsc = {start: 0.0}
        came = {}
        hq = [(0.0, start)]
        steps = [(-1, -1, 1.414), (-1, 0, 1.0), (-1, 1, 1.414), (0, -1, 1.0),
                 (0, 1, 1.0), (1, -1, 1.414), (1, 0, 1.0), (1, 1, 1.414)]
        W_H = 1.4                                # weighted A* for speed
        found = False
        expansions = 0
        while hq and expansions < 60000:
            _, cur = heapq.heappop(hq)
            expansions += 1
            if cur == goal:
                found = True
                break
            g = gsc[cur]
            for di, dj, sl in steps:
                i, j = cur[0] + di, cur[1] + dj
                if not (0 <= i < h and 0 <= j < w) or lethal[i, j]:
                    continue
                ng = g + sl * cost[i, j]
                if ng < gsc.get((i, j), 1e18):
                    gsc[(i, j)] = ng
                    came[(i, j)] = cur
                    hh = math.hypot(goal[0] - i, goal[1] - j)
                    heapq.heappush(hq, (ng + W_H * hh, (i, j)))
        if not found:
            return None
        p = [goal]
        while p[-1] != start:
            p.append(came[p[-1]])
        p.reverse()
        return p

    def _wedge_clearance(self, dx, dz):
        """Clearance (m) along arc offsets using the freshest wedge."""
        if not hasattr(self, "_wedge_dist"):
            return np.full(dx.shape[0], 99.0)
        yaw = self._wedge_yaw
        cy, sy = math.cos(yaw), math.sin(yaw)
        # world offsets -> rover frame (x right, z forward)
        rx = dx * cy - dz * sy
        rz = dx * sy + dz * cy
        half = self.wc * self.res / 2.0
        ii = np.floor((rx + half) / self.res).astype(int)
        jj = np.floor(rz / self.res).astype(int)
        inside = (ii >= 0) & (ii < self.wc) & (jj >= 0) & (jj < self.wc)
        ii, jj = np.clip(ii, 0, self.wc - 1), np.clip(jj, 0, self.wc - 1)
        d = np.where(inside, self._wedge_dist[ii, jj], 99.0)
        return d.min(axis=1)

    def _map_clearance(self, pts):
        """Believed clearance (m) at world points via the distance field."""
        i0, j0, i1, j1 = self._win
        ii = (pts[..., 0] - self.map_corner[0]) / self.res - i0
        jj = (pts[..., 1] - self.map_corner[1]) / self.res - j0
        ii = np.clip(ii, 0, self._dist_m.shape[0] - 1).astype(int)
        jj = np.clip(jj, 0, self._dist_m.shape[1] - 1).astype(int)
        return self._dist_m[ii, jj]

    # ---------------- local control ----------------
    def _control(self, danger, v_meas):
        """Local controller: track the A* path with believed-map arc checks.

        Mirrors the expert's arc sampler but every clearance lookup goes
        through the fused map + the zero-lag fresh-wedge guard instead of
        ground truth. Handles creep-through-gaps, reversing recovery, and
        speed governance (clearance, turn sharpness, goal proximity, gap
        width, and the learned danger probability all cap throttle).
        Returns (throttle, steer).
        """
        # waypoint ~7 m ahead on the A* path (or the goal directly)
        target = self.goal
        if self.path is not None and len(self.path) > 1:
            d = np.linalg.norm(self.path - self.pose, axis=1)
            k = int(np.argmin(d))
            seg = self.path[k:]
            along = np.linalg.norm(np.diff(seg, axis=0), axis=1).cumsum() \
                if len(seg) > 1 else np.array([0.0])
            w = int(np.searchsorted(along, 7.0))
            target = seg[min(w + 1, len(seg) - 1)]
        self.last_target = target              # for HUD / diagnostics

        goal_d = float(np.linalg.norm(self.goal - self.pose))
        yaw = math.radians(self.heading)

        if self.recovery > 0:
            self.recovery -= 1
            # map-checked reverse: the belief map remembers what is BEHIND.
            # A fixed-length blind reverse regularly backed into obstacles
            # the rover had already seen; stop early when the rear closes
            # or as soon as the front has opened enough to steer out.
            ryaw = math.radians(self.heading)
            back = self.pose[None] - np.array(
                [math.sin(ryaw), math.cos(ryaw)])[None] * \
                np.array([[0.9], [1.7], [2.5]])
            rear_clear = float(self._map_clearance(back).min()) \
                if hasattr(self, "_dist_m") else 99.0
            ci = self.wc // 2
            front_open = float(self._wedge_dist[ci - 2:ci + 3, :6].min()) \
                if hasattr(self, "_wedge_dist") else 0.0
            if rear_clear < 1.05 or front_open > 2.2:
                self.recovery = 0
                if rear_clear < 1.05 and front_open <= 2.2:
                    # boxed in both ways: forward escape turn instead
                    self.escape = 8
                    self.recovery_steer = -self.recovery_steer
            else:
                self.prev_steer = self.recovery_steer
                return -0.5, self.recovery_steer
        if self.escape > 0:
            self.escape -= 1
            self.prev_steer = self.recovery_steer
            return 0.3, self.recovery_steer

        # arc sampling against the BELIEVED map
        steers = np.linspace(-1, 1, self.N_ARCS)
        yr = np.radians(steers * 70.0)                    # yaw_rate_max
        t = np.arange(1, int(self.ARC_T / self.ARC_DT) + 1) * self.ARC_DT
        v_plan = float(np.clip(abs(v_meas), 1.5, 6.0))
        seg_yaw = yaw + yr[:, None] * (t - self.ARC_DT / 2)[None, :]
        dx = np.cumsum(np.sin(seg_yaw) * v_plan * self.ARC_DT, axis=1)
        dz = np.cumsum(np.cos(seg_yaw) * v_plan * self.ARC_DT, axis=1)
        pts = np.stack([self.pose[0] + dx, self.pose[1] + dz], axis=-1)
        # rover radius + buffer absorbing pose-filter error
        clear = self._map_clearance(pts).min(axis=1) - 0.95
        # fuse the zero-lag wedge guard (rover frame, no pose drift): use a
        # slimmer buffer since it has no pose error
        wclear = self._wedge_clearance(dx, dz) - 0.85
        clear = np.minimum(clear, wclear)
        feasible = clear > 0.05

        # believed-terrain feasibility along arcs: reject arcs crossing
        # cells the hazard channel believes are too steep
        ai = np.clip(((pts[..., 0] - self.map_corner[0]) / self.res)
                     .astype(int), 0, self.mc_cells - 1)
        aj = np.clip(((pts[..., 1] - self.map_corner[1]) / self.res)
                     .astype(int), 0, self.mc_cells - 1)
        haz_arc = 1.0 / (1.0 + np.exp(-self.LH[ai, aj]))
        terrain_bad = haz_arc.max(axis=1) > 0.65
        feasible &= ~terrain_bad
        clear = np.where(terrain_bad, np.minimum(clear, 0.0), clear)
        # believed sand ahead (for the throttle governor below)
        sand_arc = 1.0 / (1.0 + np.exp(-self.LS[ai, aj]))

        # stuck detection from filtered pose (contact isn't sensed directly)
        if self._last_pos is not None and abs(v_meas) < 0.35:
            self._stuck_frames += 1
        else:
            self._stuck_frames = 0
        self._last_pos = self.pose.copy()

        if self._stuck_frames > 12:
            left = clear[: self.N_ARCS // 2].max()
            right = clear[self.N_ARCS // 2 + 1:].max()
            self.recovery_steer = 0.8 if left > right else -0.8
            if self._stuck_frames > 40:                   # reversing failed too
                self.escape = 10
                self.recovery_steer = -self.recovery_steer
                self._stuck_frames = 0
            else:
                self.recovery = 14
            self.prev_steer = self.recovery_steer
            return -0.5, self.recovery_steer

        if not feasible.any():
            # imminent frontal block per the freshest wedge? back out now
            if hasattr(self, "_wedge_dist"):
                ci = self.wc // 2
                ahead = self._wedge_dist[ci - 2:ci + 3, :6].min()
                if ahead < 0.85:
                    self.recovery = 14
                    self.recovery_steer = 0.8 if clear[:self.N_ARCS // 2].max() \
                        > clear[self.N_ARCS // 2 + 1:].max() else -0.8
                    self.prev_steer = self.recovery_steer
                    return -0.5, self.recovery_steer
            # tight passage: the A* plan says the corridor is passable at its
            # own inflation, so creep along it instead of thrashing between
            # the arc check and recovery. Actual stuck-ness is caught above.
            if self.path is not None:
                bear = math.atan2(target[0] - self.pose[0],
                                  target[1] - self.pose[1])
                err = (bear - yaw + math.pi) % (2 * math.pi) - math.pi
                steer = float(np.clip(err / math.radians(35.0), -1, 1))
                self.prev_steer = steer
                return 0.25, steer
            self.recovery = 14
            self.recovery_steer = 0.8 if clear[:self.N_ARCS // 2].max() > \
                clear[self.N_ARCS // 2 + 1:].max() else -0.8
            self.prev_steer = self.recovery_steer
            return -0.5, self.recovery_steer

        end = pts[:, -1]
        wd = np.linalg.norm(target - self.pose)
        prog = wd - np.linalg.norm(target[None] - end, axis=1)
        bear = np.arctan2(target[0] - end[:, 0], target[1] - end[:, 1])
        align = np.cos(bear - seg_yaw[:, -1])
        score = (1.4 * prog + 2.2 * np.minimum(clear, 2.5) + 1.5 * align
                 - 1.2 * np.abs(steers - self.prev_steer))
        score[~feasible] = -1e9
        best = int(np.argmax(score))
        steer = float(steers[best])
        self.last_arc = pts[best]              # for visualization

        v_clear = np.clip(clear[best] / 3.5, 0.35, 1.0)
        v_turn = 1.0 - 0.4 * abs(steer)
        v_goal = np.clip(goal_d / 8.0, 0.3, 1.0)
        # believed soft ground ahead: slow before the wheels find out
        v_sand = 1.0 - 0.45 * float(sand_arc[best, :4].max())
        # predicted-terrain caution: JEPA-anticipated (still unseen) steep
        # ground or obstacles along the chosen arc lift the foot off the
        # gas. Speed caution cannot cause detours, so unlike cost-shaping
        # it cannot be anti-exploratory -- it only buys the perception
        # stack time to confirm or refute the prediction. Uses the cached
        # completion (refreshed at each replan), so no extra forwards.
        v_pred = 1.0
        if self.complete and getattr(self, "use_governor", True):
            # refresh at the replan cadence even when the invite path is
            # disabled (ablations), keeping one forward per replan interval
            if self.pred is None or (self.step_i -
                                     getattr(self, "_pred_step", -99)
                                     >= self.REPLAN_EVERY):
                self._complete_map()
        if self.complete and getattr(self, "use_governor", True) \
                and getattr(self, "pred", None) is not None:
            # sample the PLANNED PATH 5-20 m out rather than the local arc:
            # the arc's near field is inside wedge paint range and thus
            # almost always observed (measured: the arc-based governor
            # never fired once across 108 episodes)
            if self.path is not None and len(self.path) > 1:
                along = np.linalg.norm(np.diff(self.path, axis=0),
                                       axis=1).cumsum()
                k0, k1 = np.searchsorted(along, [5.0, 20.0])
                bp = self.path[k0 + 1:k1 + 1]
            else:
                bp = pts[best][-2:]
            if len(bp) == 0:
                bp = pts[best][-2:]
            cres = self.cfg.model.comp_res
            gi = ((bp[:, 0] - self.pred_origin[0]) / cres).astype(int)
            gj = ((bp[:, 1] - self.pred_origin[1]) / cres).astype(int)
            ok = (gi >= 0) & (gi < self.pred.shape[1]) & \
                 (gj >= 0) & (gj < self.pred.shape[2])
            gi, gj = np.clip(gi, 0, self.pred.shape[1] - 1), \
                np.clip(gj, 0, self.pred.shape[2] - 1)
            unseen = ok & (self.pred_observed[gi, gj] < 0.5)
            if unseen.any():
                ahead_haz = float(np.maximum(self.pred[1], self.pred[0])
                                  [gi, gj][unseen].max())
                v_pred = 1.0 - 0.55 * np.clip((ahead_haz - 0.45) / 0.35,
                                              0.0, 1.0)
        throttle = float(np.clip(min(v_clear, v_turn, v_goal, v_sand,
                                     v_pred), 0.25, 1.0))
        # thread tight passages slowly: with 200 ms actuation latency and
        # steering lag, speed is what turns a near-miss into a graze
        if hasattr(self, "_wedge_dist"):
            ci = self.wc // 2
            ahead = float(self._wedge_dist[ci - 3:ci + 4, :8].min())
            if ahead < 2.0:
                throttle = min(throttle, 0.35)
            elif ahead < 3.5:
                throttle = min(throttle, 0.55)
        if danger > 0.65:
            throttle = min(throttle, 0.4)
        self.prev_steer = steer
        return throttle, steer

    # ---------------- main entry ----------------
    def step(self, frame_bgr, sensors):
        """sensors: dict with lat, lon, heading, speed, goal_lat, goal_lon."""
        self._update_pose(sensors)
        occ_logit, conf, elev, sand_logit, haz_logit, danger = \
            self._perceive(frame_bgr, sensors)
        self._paint(occ_logit, conf, elev, sand_logit, haz_logit)
        if self.step_i % self.REPLAN_EVERY == 0 or self.path is None:
            self._replan()
        throttle, steer = self._control(danger, sensors["speed"])
        self.step_i += 1
        self._last_cmd_steer = steer
        return {"throttle": throttle, "steer": steer, "danger": danger,
                "risk": None, "path": self.path}

    # ---------------- introspection ----------------
    def map_view(self, size=112, span_m=40.0):
        """BGR image of the believed occupancy around the rover (for HUD).

        Layers, back to front: unknown grey, JEPA ghost predictions for
        unseen space (violet = predicted obstacle/hazard, light grey =
        predicted open -- same convention as fsd_viz), observed free/occ,
        then path/rover/goal markers.
        """
        c = int(span_m / self.res)
        ci = int((self.pose[0] - self.map_corner[0]) / self.res)
        cj = int((self.pose[1] - self.map_corner[1]) / self.res)
        i0, j0 = max(0, ci - c // 2), max(0, cj - c // 2)
        crop = self.L[i0:i0 + c, j0:j0 + c]
        prob = 1.0 / (1.0 + np.exp(-crop))
        img = np.full((*prob.shape, 3), 128, np.uint8)     # unknown = grey
        known = np.abs(crop) > 0.4

        # --- JEPA ghost layer: predicted content of UNSEEN cells ---
        # Computed for the HUD even when the planner is not consuming
        # predictions (mirrors fsd_viz). Cached per control step, so this
        # adds at most one completer forward per step while recording.
        if getattr(getattr(self, "model", None), "map_completer", None) is not None:
            pred = self._complete_map()
            if pred is not None:
                Gp = pred.shape[1]
                pres = self.cfg.model.comp_res
                # world position of each crop cell -> completer grid index
                xs = self.map_corner[0] + (i0 + np.arange(crop.shape[0]) + 0.5) * self.res
                zs = self.map_corner[1] + (j0 + np.arange(crop.shape[1]) + 0.5) * self.res
                pi = np.floor((xs - self.pred_origin[0]) / pres).astype(int)
                pj = np.floor((zs - self.pred_origin[1]) / pres).astype(int)
                valid = ((pi >= 0) & (pi < Gp))[:, None] & \
                        ((pj >= 0) & (pj < Gp))[None, :]
                pi = np.clip(pi, 0, Gp - 1)
                pj = np.clip(pj, 0, Gp - 1)
                blocked = np.maximum(pred[0], pred[1])[np.ix_(pi, pj)]
                unobs = valid & ~known & \
                    (self.pred_observed[np.ix_(pi, pj)] < 0.5)
                img[unobs & (blocked < 0.25)] = (150, 150, 150)
                occ_g = unobs & (blocked > 0.45)
                a = (0.55 + 0.45 * np.clip((blocked[occ_g] - 0.45) / 0.4,
                                           0, 1))[:, None]
                img[occ_g] = ((1 - a) * img[occ_g] +
                              a * np.array((200, 90, 170))).astype(np.uint8)

        img[known & (prob <= 0.5)] = (230, 230, 230)       # free = white
        occ_v = (np.clip(prob, 0.5, 1.0) - 0.5) * 2
        occ_mask = known & (prob > 0.5)
        img[occ_mask] = np.stack([40 + 0 * occ_v[occ_mask],
                                  40 + 0 * occ_v[occ_mask],
                                  120 + 135 * occ_v[occ_mask]], axis=-1)
        # rover + goal + path markers
        def mark(p, col):
            """Small square marker at a world position on the map view."""
            i = int((p[0] - self.map_corner[0]) / self.res) - i0
            j = int((p[1] - self.map_corner[1]) / self.res) - j0
            if 0 <= i < img.shape[0] and 0 <= j < img.shape[1]:
                img[max(0, i - 1):i + 2, max(0, j - 1):j + 2] = col
        if self.path is not None:
            for p in self.path[::2]:
                mark(p, (0, 200, 255))
        mark(self.pose, (0, 255, 0))
        mark(self.goal, (255, 100, 0))
        # north-up view: x (east) right, z (north) up
        img = cv2.resize(np.flipud(img.transpose(1, 0, 2)), (size, size),
                         interpolation=cv2.INTER_NEAREST)
        return img


# ==========================================================================
# Shared HUD drawing (viz, eval recordings, live test)
# ==========================================================================
def draw_hud(frame, throttle, steer, danger, risk=None, dist=None,
             ref=None, extra=None):
    """Overlay a compact telemetry HUD (danger bar, steering needle,
    throttle bar, labels) onto a camera frame. Used by eval recordings,
    the live test, and the open-loop viz."""
    h, w = frame.shape[:2]
    # danger bar
    cv2.rectangle(frame, (10, 10), (10 + int(120 * min(danger, 1.0)), 26),
                  (0, 0, 255) if danger > 0.5 else (0, 200, 0), -1)
    cv2.rectangle(frame, (10, 10), (130, 26), (255, 255, 255), 1)
    cv2.putText(frame, f"danger {danger:.2f}", (136, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    if risk is not None:
        cv2.putText(frame, f"imagined risk {risk:.2f}", (136, 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 220, 255), 1, cv2.LINE_AA)
    if dist is not None:
        cv2.putText(frame, f"goal {dist:.0f} m", (10, h - 64),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    if extra:
        cv2.putText(frame, extra, (10, 44),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)

    # steering needle(s)
    cx, cy, r = w // 2, h - 34, 26
    cv2.circle(frame, (cx, cy), r, (140, 140, 140), 1, cv2.LINE_AA)
    for val, col, th in ([(ref[1], (255, 120, 0), 1)] if ref else []) + \
            [(steer, (0, 255, 0), 2)]:
        a = float(val) * 1.2
        cv2.line(frame, (cx, cy),
                 (int(cx + r * np.sin(a)), int(cy - r * np.cos(a))), col, th,
                 cv2.LINE_AA)
    # throttle bar
    bx = cx + r + 14
    cv2.rectangle(frame, (bx, cy - 24), (bx + 10, cy + 24), (140, 140, 140), 1)
    t = int(np.clip(throttle, -1, 1) * 24)
    cv2.rectangle(frame, (bx + 1, cy - max(t, 0)), (bx + 9, cy - min(t, 0)),
                  (0, 255, 0) if throttle >= 0 else (0, 120, 255), -1)
    return frame

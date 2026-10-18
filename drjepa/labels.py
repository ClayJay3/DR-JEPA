"""Robot-frame ground-grid targets for perception training and evaluation.

Every perception output is a PHYSICAL quantity per ground cell, so one
model serves any robot (the robot profile decides what is blocking):

    vis    1 = the camera observes this cell (occlusion- and frame-aware)
    occ    1 = a rigid obstacle at least `step_height` tall stands here
    obs_h  rigid obstacle height above the local terrain (m, 0 = clear)
    elev   terrain height relative to the ground under the robot (m)
    grade  terrain slope magnitude (rise / run)
    sand   soft-ground intensity in [0, 1]

Two builders share the same grid and the same depth-based visibility:
`sim_targets` (exact: analytic terrain + prop footprints) and
`depth_targets` (from any metric depth map -- ZED stereo, lidar
projected into the image, ...). Grids are indexed [i = x right, j = z fwd].
"""

import math

import numpy as np

from .config import GridConfig
from .geo import robot_to_world

CHANNELS = ("vis", "occ", "obs_h", "elev", "grade", "sand")


def cell_centers(grid: GridConfig):
    """(X, Z) robot-frame centres of the grid cells, shape (C, C)."""
    xs = (np.arange(grid.cells) + 0.5) * grid.res - grid.half
    zs = (np.arange(grid.cells) + 0.5) * grid.res
    return np.meshgrid(xs, zs, indexing="ij")


def _cell_index(pts, grid):
    """Robot-frame points (N, 3) -> flat cell index (N,), -1 outside."""
    i = np.floor((pts[:, 0] + grid.half) / grid.res).astype(np.int64)
    j = np.floor(pts[:, 2] / grid.res).astype(np.int64)
    ok = (i >= 0) & (i < grid.cells) & (j >= 0) & (j < grid.cells)
    return np.where(ok, i * grid.cells + j, -1)


def reprojection_count(depth, cam, R, t, grid, valid=None):
    """How many depth pixels land in each cell (C, C)."""
    m = depth > 0 if valid is None else valid & (depth > 0)
    pts = cam.unproject(depth, R, t)[m]
    idx = _cell_index(pts, grid)
    idx = idx[idx >= 0]
    return np.bincount(idx, minlength=grid.cells ** 2).reshape(grid.cells, grid.cells)


def ground_visible(depth, cam, R, t, grid, elev, tol=0.04):
    """Cells whose ground point projects inside the image and is not
    hidden behind something closer in the depth map (C, C) bool."""
    X, Z = cell_centers(grid)
    P = np.stack([X, elev, Z], -1).reshape(-1, 3)
    uv, zc = cam.project(P, R, t)
    u = np.floor(uv[:, 0]).astype(np.int64)
    v = np.floor(uv[:, 1]).astype(np.int64)
    inside = (zc > 0.2) & (u >= 0) & (u < cam.width) & (v >= 0) & (v < cam.height)
    d = np.zeros(len(P), np.float32)
    d[inside] = depth[v[inside], u[inside]]
    ok = inside & (d > 0) & (d >= zc * (1 - tol) - 0.05)
    return ok.reshape(grid.cells, grid.cells)


# ==========================================================================
# simulator (exact)
# ==========================================================================
def sim_targets(sim, depth, cls, R, t, grid: GridConfig):
    """Exact targets for the CURRENT sim frame (rendered with pose R, t)."""
    C = grid.cells
    X, Z = cell_centers(grid)
    wx, wz = robot_to_world(X, Z, sim.yaw)
    WX, WZ = sim.x + wx, sim.z + wz
    h0 = float(sim.terrain.height(sim.x, sim.z))
    elev = (sim.terrain.height(WX.ravel(), WZ.ravel()).reshape(C, C) - h0).astype(np.float32)
    grade = sim.terrain.grade(WX.ravel(), WZ.ravel()).reshape(C, C).astype(np.float32)
    sand = sim.terrain.sand(WX.ravel(), WZ.ravel()).reshape(C, C).astype(np.float32)

    occ = np.zeros((C, C), bool)
    obs_h = np.zeros((C, C), np.float32)
    px, pz, pr, ph, cx, cz, cr = _prop_arrays(sim)
    reach = grid.depth * 1.5 + 10.0
    sel = (np.abs(px - sim.x) < reach) & (np.abs(pz - sim.z) < reach)
    _stamp(obs_h, WX, WZ, px[sel], pz[sel], pr[sel], ph[sel], sim, grid, mode="max")
    sel = (np.abs(cx - sim.x) < reach) & (np.abs(cz - sim.z) < reach)
    _stamp(occ, WX, WZ, cx[sel], cz[sel], cr[sel], None, sim, grid, mode="or")
    # an occupied cell carries at least the step height (colliders without a
    # matching footprint, e.g. vehicles, still read as tall)
    obs_h = np.where(occ, np.maximum(obs_h, sim.robot.step_height), obs_h)

    valid = cls < 3                                     # not sky
    count = reprojection_count(depth, sim.camera, R, t, grid, valid)
    vis = (count > 0) | ground_visible(depth, sim.camera, R, t, grid, elev)
    return {"vis": vis, "occ": occ, "obs_h": obs_h, "elev": elev,
            "grade": grade, "sand": sand}


def _prop_arrays(sim):
    """Cached flat arrays of prop footprints and colliders for one sim."""
    key = (id(sim.props), len(sim.props))
    cache = getattr(sim, "_label_cache", None)
    if cache is None or cache[0] != key:
        ps = [p for p in sim.props if p.kind != "tuft"]
        px = np.array([p.x for p in ps])
        pz = np.array([p.z for p in ps])
        pr = np.array([max(p.footprint, 1e-3) for p in ps])
        ph = np.array([p.rigid_h for p in ps])
        cols = np.array([c for p in ps for c in p.colliders]).reshape(-1, 3)
        cache = (key, (px, pz, pr, ph, cols[:, 0], cols[:, 1], cols[:, 2]))
        sim._label_cache = cache
    return cache[1]


def _stamp(out, WX, WZ, x, z, r, val, sim, grid, mode, chunk=256):
    """Rasterize discs into a cell grid: cells whose centre lies inside a
    disc, plus the cell containing each disc's centre (small objects)."""
    C = grid.cells
    for k in range(0, len(x), chunk):
        xs, zs, rs = x[k:k + chunk, None, None], z[k:k + chunk, None, None], r[k:k + chunk, None, None]
        hit = (WX[None] - xs) ** 2 + (WZ[None] - zs) ** 2 < rs * rs
        xr, zr = _to_robot(x[k:k + chunk] - sim.x, z[k:k + chunk] - sim.z, sim.yaw)
        ci = np.floor((xr + grid.half) / grid.res).astype(int)
        cj = np.floor(zr / grid.res).astype(int)
        ok = (ci >= 0) & (ci < C) & (cj >= 0) & (cj < C)
        hit[np.flatnonzero(ok), ci[ok], cj[ok]] = True
        if mode == "or":
            out |= hit.any(0)
        else:
            v = np.where(hit, val[k:k + chunk, None, None], 0.0).max(0)
            np.maximum(out, v, out=out)


def _to_robot(dx, dz, yaw_deg):
    c, s = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
    return dx * c - dz * s, dx * s + dz * c


# ==========================================================================
# real depth (stereo / lidar)
# ==========================================================================
def depth_targets(depth, cam, R, t, grid: GridConfig, valid=None, step=0.25,
                  min_pts=3, band_hi=2.5, dmin=0.3, dmax=20.0):
    """Targets from a metric depth map in the camera frame.

    Per cell: ground = 10th-percentile point height (robust to anything
    standing on it); obstacle height = 95th percentile above the local
    ground. Cells that see only an object (no ground) get elevation / grade
    NaN = unknown, never the object's height. Returns the same dict as
    sim_targets plus `n` (points per cell); `sand` is all-zero (not
    measurable from geometry).
    """
    C = grid.cells
    m = (depth > dmin) & (depth < dmax)
    if valid is not None:
        m &= valid
    pts = cam.unproject(depth, R, t)[m]
    idx = _cell_index(pts, grid)
    keep = idx >= 0
    idx, h = idx[keep], pts[keep, 1]
    n = np.bincount(idx, minlength=C * C)
    ground = np.full(C * C, np.nan, np.float32)
    top = np.full(C * C, np.nan, np.float32)
    if len(idx):
        order = np.lexsort((h, idx))
        idx, h = idx[order], h[order]
        starts = np.flatnonzero(np.r_[True, idx[1:] != idx[:-1]])
        ends = np.r_[starts[1:], len(idx)]
        cnt = ends - starts
        ok = cnt >= min_pts
        s, c = starts[ok], cnt[ok]
        cells = idx[s]
        ground[cells] = h[s + (0.10 * (c - 1)).astype(np.int64)]
        top[cells] = h[s + (0.95 * (c - 1)).astype(np.int64)]
    ground = ground.reshape(C, C)
    top = top.reshape(C, C)
    vis = ~np.isnan(ground)

    # Where is GROUND actually observed? A cell that only sees an object's
    # face has its low percentile ON the object. Neighbourhood ground = min
    # over the 3x3 cells (0.5 m radius: slopes up to ~0.4 grade still pass);
    # cells more than 0.15 m above it are object-covered -> elevation unknown.
    # Obstacle height is measured from the lower of the cell's own ground and
    # the neighbourhood ground (+5 cm), so a face-only cell still reads the
    # object's full height (a 1.0 m box read 0.84 m without this).
    gpad = np.pad(np.where(vis, ground, np.inf), 1, constant_values=np.inf)
    env = np.min(np.stack([gpad[1 + di:1 + di + C, 1 + dj:1 + dj + C]
                           for di in range(-1, 2) for dj in range(-1, 2)]), axis=0)
    env = np.where(np.isfinite(env), env, np.nan)
    ground_seen = vis & ~(ground > env + 0.15)
    base = np.fmin(ground, env + 0.05)
    obs_h = np.where(vis, np.clip(top - base, 0.0, band_hi), 0.0).astype(np.float32)
    obs_h = np.nan_to_num(obs_h, nan=0.0)
    occ = vis & (obs_h >= step)

    elev = np.where(ground_seen, ground, np.nan).astype(np.float32)
    filled = np.where(np.isnan(elev), np.nanmedian(elev) if ground_seen.any() else 0.0, elev)
    gx_, gz_ = np.gradient(filled, grid.res)
    grade = np.where(ground_seen, np.sqrt(gx_ ** 2 + gz_ ** 2), np.nan).astype(np.float32)
    return {"vis": vis, "occ": occ, "obs_h": obs_h, "elev": elev, "grade": grade,
            "sand": np.zeros((C, C), np.float32), "n": n.reshape(C, C)}

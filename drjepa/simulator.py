"""Domain-randomized 3D rover simulator.

Realism features aimed at sim-to-real transfer:
  * rolling heightfield terrain (camera pitch/roll follows the ground)
  * per-episode randomized lighting (sun direction, ambient/diffuse), sky
    gradient, ground/rock/vegetation palettes, distance fog
  * irregular rock/tree/bush meshes with lambertian shading (no cartoon
    outlines, no debug grid lines)
  * non-colliding ground clutter for parallax cues
  * camera post-processing: motion blur, exposure/white-balance shift,
    vignette, sensor noise, ride-bump camera shake
  * vehicle dynamics: acceleration limits, steering lag, actuation latency,
    wheel slip -- and the rover physically stops when it hits something
  * noisy sensors: GPS random walk + white noise, compass bias/noise

The goal marker is intentionally never rendered: navigation must rely on the
goal vector alone, exactly like the real rover.
"""

import math

import cv2
import numpy as np

from .config import SimConfig

# Approximate metres per degree of latitude.
M_PER_DEG = 111139.0


def meters_to_latlon(x, z, origin_lat, origin_lon):
    """Local metres (x=east, z=north) -> GPS degrees."""
    lat = origin_lat + z / M_PER_DEG
    lon = origin_lon + x / (M_PER_DEG * math.cos(math.radians(origin_lat)))
    return lat, lon


def latlon_to_meters(lat, lon, origin_lat, origin_lon):
    """GPS degrees -> local metres (x=east, z=north) about an origin fix."""
    z = (lat - origin_lat) * M_PER_DEG
    x = (lon - origin_lon) * M_PER_DEG * math.cos(math.radians(origin_lat))
    return x, z


def wedge_ground_truth(sim, cells=48, res=0.5, fov_deg=None):
    """Ground-truth occupancy + visibility of the wedge ahead of the rover.

    Returns (occ, vis) boolean arrays of shape (cells, cells) in the rover
    frame: index [i, j] covers x_right = (i+0.5)*res - cells*res/2,
    z_forward = (j+0.5)*res. Visibility is an occlusion-aware raycast: cells
    behind the first obstacle along a camera ray (or outside the FOV) are
    marked invisible, so the perception net is never asked to hallucinate
    what the camera cannot see. Used only at data-generation time.
    """
    fov = math.radians(fov_deg if fov_deg is not None else sim.cfg.fov_deg)
    half = cells * res / 2.0
    xs = (np.arange(cells) + 0.5) * res - half        # rover-frame right
    zs = (np.arange(cells) + 0.5) * res               # rover-frame forward
    X, Z = np.meshgrid(xs, zs, indexing="ij")
    yaw = math.radians(sim.yaw)
    cy, sy = math.cos(yaw), math.sin(yaw)
    WX = sim.x + X * cy + Z * sy
    WZ = sim.z - X * sy + Z * cy

    occ = np.zeros((cells, cells), bool)
    if len(sim.obs_rad):
        reach = cells * res * 1.5
        near = np.linalg.norm(sim.obs_xz - np.array([sim.x, sim.z]),
                              axis=1) < reach + sim.obs_rad
        for (ox, oz), r in zip(sim.obs_xz[near], sim.obs_rad[near]):
            occ |= (WX - ox) ** 2 + (WZ - oz) ** 2 < r * r

    # raycast visibility in the rover frame
    vis = np.zeros((cells, cells), bool)
    n_rays = 3 * cells
    angs = np.linspace(-fov / 2, fov / 2, n_rays)
    rs = np.arange(res / 2, cells * res, res / 2)
    px = np.sin(angs)[:, None] * rs[None, :]          # (rays, steps)
    pz = np.cos(angs)[:, None] * rs[None, :]
    ii = np.floor((px + half) / res).astype(int)
    jj = np.floor(pz / res).astype(int)
    inside = (ii >= 0) & (ii < cells) & (jj >= 0) & (jj < cells)
    ii, jj = np.clip(ii, 0, cells - 1), np.clip(jj, 0, cells - 1)
    hit = occ[ii, jj] & inside
    # first hit blocks everything past it (the hit cell itself is visible)
    blocked = np.cumsum(hit, axis=1) - hit.astype(int) > 0
    ok = inside & ~blocked
    vis[ii[ok], jj[ok]] = True
    return occ, vis


def goal_vector(lat, lon, heading_deg, goal_lat, goal_lon):
    """Distance (m) and relative bearing (deg, [-180,180]) to the goal."""
    dz = (goal_lat - lat) * M_PER_DEG
    dx = (goal_lon - lon) * M_PER_DEG * math.cos(math.radians(lat))
    dist = math.hypot(dx, dz)
    bearing = math.degrees(math.atan2(dx, dz))
    rel = (bearing - heading_deg + 180.0) % 360.0 - 180.0
    return dist, rel


# ==========================================================================
# Episode style (domain randomization)
# ==========================================================================
_GROUND_PALETTES = [  # (color_a, color_b) in BGR float
    ((70, 95, 130), (55, 75, 105)),    # red-brown desert
    ((90, 115, 140), (70, 90, 115)),   # sandy
    ((80, 90, 95), (60, 68, 75)),      # grey gravel
    ((60, 100, 110), (45, 80, 95)),    # dry mud
    ((70, 110, 90), (50, 90, 75)),     # sparse grass
]
_ROCK_PALETTES = [
    (95, 100, 110), (70, 75, 85), (85, 95, 120), (110, 115, 120), (60, 70, 90),
]
_CANOPY_PALETTES = [
    (55, 110, 60), (45, 90, 45), (60, 130, 90), (50, 95, 110), (40, 75, 55),
]
_SKY_PRESETS = [  # (zenith, horizon) BGR
    ((200, 140, 70), (235, 210, 175)),   # clear blue
    ((215, 180, 140), (235, 225, 210)),  # hazy
    ((190, 185, 180), (215, 210, 205)),  # overcast
    ((170, 150, 120), (200, 200, 220)),  # warm evening
]


class EpisodeStyle:
    """One bag of appearance/dynamics randomization, drawn per episode."""

    def __init__(self, rng: np.random.Generator):
        """Sample every appearance/dynamics parameter for one episode."""
        self.ground_a, self.ground_b = [np.array(c, float) for c in
                                        _GROUND_PALETTES[rng.integers(len(_GROUND_PALETTES))]]
        self.rock_color = np.array(_ROCK_PALETTES[rng.integers(len(_ROCK_PALETTES))], float)
        self.canopy_color = np.array(_CANOPY_PALETTES[rng.integers(len(_CANOPY_PALETTES))], float)
        self.trunk_color = np.array((45, 65, 90), float) * rng.uniform(0.8, 1.2)

        sky_i = rng.integers(len(_SKY_PRESETS))
        self.sky_zenith = np.array(_SKY_PRESETS[sky_i][0], float) * rng.uniform(0.9, 1.1)
        self.sky_horizon = np.array(_SKY_PRESETS[sky_i][1], float) * rng.uniform(0.9, 1.1)
        self.fog_color = self.sky_horizon.copy()
        self.fog_dist = float(np.exp(rng.uniform(np.log(70), np.log(400))))

        elev = math.radians(rng.uniform(20, 70))
        azim = math.radians(rng.uniform(0, 360))
        self.sun_dir = np.array([math.cos(elev) * math.sin(azim),
                                 math.sin(elev),
                                 math.cos(elev) * math.cos(azim)])
        self.sun_visible = sky_i in (0, 3) and rng.random() < 0.7
        self.ambient = rng.uniform(0.40, 0.55)
        self.diffuse = rng.uniform(0.45, 0.70)

        self.terrain_amp = rng.uniform(0.0, 1.6)
        self.terrain_wavelen = rng.uniform(25.0, 60.0)
        self.ground_noise_wavelen = rng.uniform(8.0, 20.0)

        self.bump_amp_deg = rng.uniform(0.2, 0.9)
        self.exposure = rng.normal(1.0, 0.05, size=3).clip(0.85, 1.15) * rng.uniform(0.88, 1.12)
        self.noise_sigma = rng.uniform(1.0, 5.0)
        self.vignette = rng.uniform(0.0, 0.35)
        self.motion_blur = rng.uniform(0.10, 0.40)
        self.v_max = rng.uniform(4.0, 8.0)


# ==========================================================================
# Terrain heightfield
# ==========================================================================
class Terrain:
    """Rolling heightfield built from a small sum of random sinusoids.

    Cheap to evaluate anywhere (no grid storage), smooth enough for the
    camera pitch/roll to follow, and different every episode.
    """

    def __init__(self, rng: np.random.Generator, amp: float, wavelen: float):
        """Draw random directions, wavelengths, and phases for 4 waves."""
        n = 4
        ang = rng.uniform(0, 2 * np.pi, n)
        wl = wavelen * rng.uniform(0.6, 1.6, n)
        self.kx = 2 * np.pi * np.cos(ang) / wl
        self.kz = 2 * np.pi * np.sin(ang) / wl
        self.phase = rng.uniform(0, 2 * np.pi, n)
        self.amp = amp * rng.uniform(0.5, 1.0, n) / n * 2.0

    def height(self, x, z):
        """Vectorized: x, z arrays or scalars -> heights."""
        x = np.asarray(x, float)[..., None]
        z = np.asarray(z, float)[..., None]
        h = np.sum(self.amp * np.sin(x * self.kx + z * self.kz + self.phase), axis=-1)
        return h if h.shape else float(h)

    def slope(self, x, z, d=0.75):
        """Finite-difference terrain gradient (dh/dx, dh/dz) at a point."""
        hx = (self.height(x + d, z) - self.height(x - d, z)) / (2 * d)
        hz = (self.height(x, z + d) - self.height(x, z - d)) / (2 * d)
        return hx, hz


# ==========================================================================
# Obstacle meshes
# ==========================================================================
class Obstacle:
    """A static world object: shaded triangle/quad mesh + collision circle."""

    __slots__ = ("x", "z", "radius", "kind", "verts", "faces",
                 "face_colors", "face_normals", "face_centers", "bound_r", "center")

    def __init__(self, x, z, radius, kind, verts, faces, base_colors, style):
        """Bake lambertian shading per face (the sun is fixed within an
        episode, so face colors are computed once at build time)."""
        self.x, self.z, self.radius, self.kind = x, z, radius, kind
        self.verts = verts
        self.faces = faces
        centroid = verts.mean(axis=0)
        normals, centers, colors = [], [], []
        for f, base in zip(faces, base_colors):
            pts = verts[f]
            # Newell's method normal, oriented away from the mesh centroid
            n = np.zeros(3)
            for i in range(len(pts)):
                a, b = pts[i], pts[(i + 1) % len(pts)]
                n += np.cross(a - centroid, b - centroid)
            norm = np.linalg.norm(n)
            n = n / norm if norm > 1e-9 else np.array([0.0, 1.0, 0.0])
            c = pts.mean(axis=0)
            if np.dot(n, c - centroid) < 0:
                n = -n
            shade = style.ambient + style.diffuse * max(0.0, float(np.dot(n, style.sun_dir)))
            normals.append(n)
            centers.append(c)
            colors.append(np.clip(base * shade, 0, 255))
        self.face_normals = np.array(normals)
        self.face_centers = np.array(centers)
        self.face_colors = np.array(colors)
        self.center = centroid
        self.bound_r = float(np.max(np.linalg.norm(verts - centroid, axis=1)))


_BOX_FACES = [np.array(f) for f in
              ([0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7], [4, 5, 6, 7])]


def _jittered_box(rng, w, h, d, jitter):
    """Box vertices with random per-vertex jitter -> irregular rock-like hull."""
    hw, hd = w / 2, d / 2
    v = np.array([[-hw, 0, -hd], [hw, 0, -hd], [hw, 0, hd], [-hw, 0, hd],
                  [-hw, h, -hd], [hw, h, -hd], [hw, h, hd], [-hw, h, hd]], float)
    v += rng.uniform(-jitter, jitter, v.shape) * np.array([w, h, d]) * 0.5
    return v


def make_rock(rng, x, z, scale, terrain, style, kind="rock"):
    """Irregular boulder mesh seated on the terrain; collision radius ~ size."""
    w = scale * rng.uniform(0.8, 1.3)
    d = scale * rng.uniform(0.8, 1.3)
    h = scale * rng.uniform(0.5, 0.9)
    v = _jittered_box(rng, w, h, d, 0.28)
    y0 = terrain.height(x, z) - 0.12 * h
    v += np.array([x, y0, z])
    base = style.rock_color * rng.uniform(0.82, 1.18)
    colors = [base * rng.uniform(0.92, 1.08) for _ in _BOX_FACES]
    radius = 0.55 * max(w, d)
    return Obstacle(x, z, radius, kind, v, _BOX_FACES, colors, style)


def make_tree(rng, x, z, scale, terrain, style):
    """Trunk + jittered canopy; collision radius follows the thin trunk."""
    y0 = terrain.height(x, z)
    trunk_h = scale * rng.uniform(1.2, 2.2)
    trunk_w = 0.12 * scale * rng.uniform(0.8, 1.4)
    tv = _jittered_box(rng, trunk_w, trunk_h, trunk_w, 0.15) + np.array([x, y0 - 0.05, z])
    can_w = scale * rng.uniform(0.9, 1.6)
    can_h = scale * rng.uniform(0.9, 1.5)
    cv_ = _jittered_box(rng, can_w, can_h, can_w, 0.30) + np.array([x, y0 + trunk_h * 0.75, z])
    verts = np.vstack([tv, cv_])
    faces = list(_BOX_FACES) + [f + 8 for f in _BOX_FACES]
    tc = style.trunk_color
    cc = style.canopy_color * rng.uniform(0.8, 1.2)
    colors = [tc * rng.uniform(0.9, 1.1) for _ in range(5)] + \
             [cc * rng.uniform(0.88, 1.12) for _ in range(5)]
    return Obstacle(x, z, min(0.8, max(0.35, trunk_w * 3)), "tree",
                    verts, faces, colors, style)


def make_bush(rng, x, z, scale, terrain, style):
    """Low ground-hugging shrub (a jittered canopy without a trunk)."""
    w = scale * rng.uniform(0.9, 1.5)
    h = scale * rng.uniform(0.5, 0.9)
    v = _jittered_box(rng, w, h, w, 0.35)
    v += np.array([x, terrain.height(x, z) - 0.08 * h, z])
    cc = style.canopy_color * rng.uniform(0.7, 1.1)
    colors = [cc * rng.uniform(0.85, 1.15) for _ in _BOX_FACES]
    return Obstacle(x, z, 0.5 * w, "bush", v, _BOX_FACES, colors, style)


def make_clutter(rng, x, z, terrain, style):
    """Tiny non-colliding pebbles/tufts for ground texture and parallax."""
    s = rng.uniform(0.08, 0.35)
    if rng.random() < 0.6:
        return make_rock(rng, x, z, s, terrain, style, kind="clutter")
    o = make_bush(rng, x, z, s, terrain, style)
    o.kind = "clutter"
    o.radius = 0.0
    return o


# ==========================================================================
# Render engine
# ==========================================================================
class Camera:
    """Pinhole camera: world -> camera-frame -> pixel projection.

    Conventions: world x=east, y=up, z=north; camera X=right, Y=up,
    Z=forward; yaw is compass-style (0 = north, positive clockwise).
    """

    def __init__(self, w, h, fov_deg):
        """Derive focal length from image width and horizontal FOV."""
        self.W, self.H = w, h
        self.cx, self.cy = w / 2.0, h / 2.0
        self.f = (w / 2.0) / math.tan(math.radians(fov_deg / 2.0))
        self.znear = 0.25

    def rotation(self, yaw_deg, pitch_deg, roll_deg):
        """Columns are the camera basis (right, up, forward) in world coords."""
        psi, th, ph = map(math.radians, (yaw_deg, pitch_deg, roll_deg))
        fwd0 = np.array([math.sin(psi), 0.0, math.cos(psi)])
        right0 = np.array([math.cos(psi), 0.0, -math.sin(psi)])
        up0 = np.array([0.0, 1.0, 0.0])
        fwd = fwd0 * math.cos(th) + up0 * math.sin(th)
        up1 = up0 * math.cos(th) - fwd0 * math.sin(th)
        right = right0 * math.cos(ph) + up1 * math.sin(ph)
        up = up1 * math.cos(ph) - right0 * math.sin(ph)
        return np.stack([right, up, fwd], axis=1)

    def to_cam(self, pts, cam_pos, R):
        """World points (N, 3) -> camera-frame points given rotation R."""
        return (pts - cam_pos) @ R

    def project(self, pts_cam):
        """Camera-frame points -> pixel (u, v); caller must handle z<=0."""
        z = np.maximum(pts_cam[..., 2], 1e-6)
        u = pts_cam[..., 0] / z * self.f + self.cx
        v = self.cy - pts_cam[..., 1] / z * self.f
        return np.stack([u, v], axis=-1)

    def clip_near(self, poly_cam):
        """Sutherland-Hodgman clip of a polygon against z >= znear."""
        out = []
        n = len(poly_cam)
        for i in range(n):
            a, b = poly_cam[i], poly_cam[(i + 1) % n]
            ain, bin_ = a[2] >= self.znear, b[2] >= self.znear
            if ain:
                out.append(a)
            if ain != bin_:
                t = (self.znear - a[2]) / (b[2] - a[2])
                out.append(a + t * (b - a))
        return np.array(out) if len(out) >= 3 else None


class Renderer:
    """Painter's-algorithm rasterizer for the synthetic camera.

    Draws, far to near: sky gradient, hazy far-ground rings, sun disc,
    depth-sorted terrain quads and obstacle faces (lambertian shading +
    distance fog), then applies camera post-processing (motion blur,
    exposure/white-balance, vignette, sensor noise).
    """

    def __init__(self, cfg: SimConfig, style: EpisodeStyle, terrain: Terrain,
                 rng: np.random.Generator):
        """Precompute the static sky, vignette mask, and ground-noise field."""
        self.cfg = cfg
        self.style = style
        self.terrain = terrain
        self.cam = Camera(cfg.img_w, cfg.img_h, cfg.fov_deg)
        self._sky = self._make_sky()
        self._vignette = self._make_vignette()
        self._prev_frame = None
        self._noise_rng = rng
        self.grid_step = 3.0
        self.grid_radius = 57.0
        # low-frequency ground color mixing field
        self._gk = 2 * np.pi / style.ground_noise_wavelen
        self._gphase = rng.uniform(0, 2 * np.pi, 2)

    # ---------------- static layers ----------------
    def _make_sky(self):
        """Vertical zenith->horizon gradient image for the episode's sky."""
        h = self.cfg.img_h
        t = (np.linspace(0, 1, h) ** 1.4)[:, None]
        col = self.style.sky_zenith[None, :] * (1 - t) + self.style.sky_horizon[None, :] * t
        return np.repeat(col[:, None, :], self.cfg.img_w, axis=1).astype(np.float32)

    def _make_vignette(self):
        """Radial brightness falloff mask (per-episode strength)."""
        yy, xx = np.mgrid[0:self.cfg.img_h, 0:self.cfg.img_w]
        r2 = (((xx - self.cam.cx) / self.cam.cx) ** 2 +
              ((yy - self.cam.cy) / self.cam.cy) ** 2)
        return (1.0 - self.style.vignette * (r2 / 2.0)).astype(np.float32)[..., None]

    def _fog(self, d):
        """Exponential fog blend factor in [0, 1) for distance d (metres)."""
        return 1.0 - np.exp(-np.asarray(d, float) / self.style.fog_dist)

    # ---------------- ground ----------------
    def _ground_color(self, x, z):
        """Vectorized ground albedo with low-freq patches + per-cell jitter."""
        m = 0.5 + 0.5 * np.sin(x * self._gk + self._gphase[0]) * \
            np.sin(z * self._gk * 1.31 + self._gphase[1])
        i, j = np.floor(x / self.grid_step), np.floor(z / self.grid_step)
        jit = np.modf(np.abs(np.sin(i * 127.1 + j * 311.7) * 43758.5453))[0]
        col = (self.style.ground_a[None] * (1 - m[..., None]) +
               self.style.ground_b[None] * m[..., None])
        return col * (0.84 + 0.32 * jit[..., None])

    def _terrain_faces(self, cam_pos, R, queue):
        """Append shaded, fogged terrain quads around the camera to the
        painter queue (heights, normals, and colors are all vectorized)."""
        st, rad = self.grid_step, self.grid_radius
        x0 = math.floor((cam_pos[0] - rad) / st) * st
        z0 = math.floor((cam_pos[2] - rad) / st) * st
        n = int(2 * rad / st) + 1
        xs = x0 + np.arange(n + 1) * st
        zs = z0 + np.arange(n + 1) * st
        X, Z = np.meshgrid(xs, zs, indexing="ij")
        H = self.terrain.height(X.ravel(), Z.ravel()).reshape(X.shape)
        corners = np.stack([X, H, Z], axis=-1)                    # (n+1, n+1, 3)
        cc = (corners.reshape(-1, 3) - cam_pos) @ R
        cc = cc.reshape(n + 1, n + 1, 3)

        # per-quad data
        c00, c10 = cc[:-1, :-1], cc[1:, :-1]
        c01, c11 = cc[:-1, 1:], cc[1:, 1:]
        zq = np.stack([c00[..., 2], c10[..., 2], c01[..., 2], c11[..., 2]], -1)
        depth = zq.mean(-1)
        vis = zq.max(-1) > self.cam.znear

        # lighting from heightfield gradient
        hx = (H[1:, :-1] + H[1:, 1:] - H[:-1, :-1] - H[:-1, 1:]) / (2 * st)
        hz = (H[:-1, 1:] + H[1:, 1:] - H[:-1, :-1] - H[1:, :-1]) / (2 * st)
        inv = 1.0 / np.sqrt(hx ** 2 + hz ** 2 + 1.0)
        ndl = np.clip((-hx * self.style.sun_dir[0] + self.style.sun_dir[1]
                       - hz * self.style.sun_dir[2]) * inv, 0, None)
        shade = self.style.ambient + self.style.diffuse * ndl

        qx = X[:-1, :-1] + st / 2
        qz = Z[:-1, :-1] + st / 2
        albedo = self._ground_color(qx, qz)
        fog = self._fog(np.maximum(depth, 0.0))[..., None]
        color = albedo * shade[..., None] * (1 - fog) + self.style.fog_color[None, None] * fog

        idx = np.argwhere(vis)
        for i, j in idx:
            poly = np.stack([c00[i, j], c10[i, j], c11[i, j], c01[i, j]])
            queue.append((float(depth[i, j]), poly, color[i, j]))

    def _far_ground(self, canvas, cam_pos, R, yaw):
        """Haze-colored ground rings between the terrain mesh and the horizon."""
        base = 0.5 * (self.style.ground_a + self.style.ground_b) * \
            (self.style.ambient + self.style.diffuse * self.style.sun_dir[1] * 0.8)
        for dist in (600.0, 160.0, 80.0):
            fog = float(self._fog(dist))
            col = base * (1 - fog) + self.style.fog_color * fog
            ang = np.radians(yaw + np.linspace(-80, 80, 17))
            pts = np.stack([cam_pos[0] + dist * np.sin(ang),
                            np.zeros(17),
                            cam_pos[2] + dist * np.cos(ang)], axis=1)
            pc = self.cam.to_cam(pts, cam_pos, R)
            keep = pc[:, 2] > self.cam.znear
            if keep.sum() < 2:
                continue
            uv = self.cam.project(pc[keep])
            poly = np.vstack([uv, [[self.cfg.img_w + 50, self.cfg.img_h + 50],
                                   [-50, self.cfg.img_h + 50]]]).astype(np.int32)
            cv2.fillPoly(canvas, [poly], col.tolist(), lineType=cv2.LINE_AA)

    def _sun(self, canvas, cam_pos, R):
        """Draw the sun disc + soft glare when it falls inside the frame."""
        if not self.style.sun_visible:
            return
        pc = (self.style.sun_dir * 5000.0) @ R
        if pc[2] <= 1.0:
            return
        u, v = self.cam.project(pc[None])[0]
        if not (-60 <= u <= self.cfg.img_w + 60 and -60 <= v <= self.cfg.img_h + 60):
            return
        overlay = canvas.copy()
        cv2.circle(overlay, (int(u), int(v)), 42, (255, 250, 235), -1, cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.25, canvas, 0.75, 0, dst=canvas)
        cv2.circle(canvas, (int(u), int(v)), 12, (255, 253, 245), -1, cv2.LINE_AA)

    # ---------------- main ----------------
    def render(self, x, z, yaw, pitch, roll, cam_h, obstacles, speed_frac=0.5):
        """Render one BGR camera frame from the given pose.

        speed_frac (0..1) scales the motion-blur accumulation so the image
        smears more at speed, like a real rolling camera.
        """
        cfg, cam, style = self.cfg, self.cam, self.style
        cam_pos = np.array([x, self.terrain.height(x, z) + cam_h, z])
        R = cam.rotation(yaw, pitch, roll)

        canvas = self._sky.copy()
        canvas_u8 = canvas.astype(np.uint8)
        self._far_ground(canvas_u8, cam_pos, R, yaw)
        self._sun(canvas_u8, cam_pos, R)

        queue = []  # (depth, cam-space poly, BGR color)
        self._terrain_faces(cam_pos, R, queue)

        view_dist = min(style.fog_dist * 2.5, 260.0)
        fwd = R[:, 2]
        for ob in obstacles:
            dx, dz = ob.x - x, ob.z - z
            d2 = dx * dx + dz * dz
            if d2 > view_dist * view_dist:
                continue
            if ob.kind == "clutter" and d2 > 45.0 ** 2:
                continue
            # frustum-ish cull: behind camera and not close
            if d2 > 20 ** 2 and (dx * fwd[0] + dz * fwd[2]) < 0:
                continue
            cc = cam.to_cam(ob.verts, cam_pos, R)
            if cc[:, 2].max() <= cam.znear:
                continue
            face_vis = ((ob.face_centers - cam_pos) * ob.face_normals).sum(1) < 0
            d_ob = math.sqrt(d2)
            fog = float(self._fog(d_ob))
            for fi, f in enumerate(ob.faces):
                if not face_vis[fi]:
                    continue
                poly = cc[f]
                depth = float(poly[:, 2].mean())
                col = ob.face_colors[fi] * (1 - fog) + style.fog_color * fog
                queue.append((depth, poly, col))

        queue.sort(key=lambda t: t[0], reverse=True)
        W, Hh = cfg.img_w, cfg.img_h
        for depth, poly, col in queue:
            if poly[:, 2].min() < cam.znear:
                poly = cam.clip_near(poly)
                if poly is None:
                    continue
            uv = cam.project(poly)
            if (uv[:, 0].max() < 0 or uv[:, 0].min() > W or
                    uv[:, 1].max() < 0 or uv[:, 1].min() > Hh):
                continue
            cv2.fillPoly(canvas_u8, [uv.astype(np.int32)],
                         np.clip(col, 0, 255).tolist(), lineType=cv2.LINE_AA)

        # ---------------- post-processing ----------------
        frame = canvas_u8.astype(np.float32)
        if self._prev_frame is not None:
            a = style.motion_blur * (0.5 + speed_frac)
            a = min(a, 0.65)
            frame = frame * (1 - a) + self._prev_frame * a
        self._prev_frame = frame.copy()

        frame *= style.exposure[None, None, :]
        frame *= self._vignette
        frame += self._noise_rng.normal(0, style.noise_sigma, frame.shape)
        return np.clip(frame, 0, 255).astype(np.uint8)


# ==========================================================================
# World generation
# ==========================================================================
SCENARIOS = ("open", "dense", "wall", "boulders")
SPAWN_MODES = ("normal", "uturn", "recovery")


def _scatter(rng, terrain, style, x_range, z_range, count, mix, ox=0.0, oz=0.0):
    """Uniformly scatter `count` obstacles; `mix` = (rock, tree, bush) odds."""
    obs = []
    for _ in range(count):
        x = ox + rng.uniform(*x_range)
        z = oz + rng.uniform(*z_range)
        r = rng.random()
        if r < mix[0]:
            obs.append(make_rock(rng, x, z, rng.uniform(0.5, 2.2), terrain, style))
        elif r < mix[0] + mix[1]:
            obs.append(make_tree(rng, x, z, rng.uniform(0.8, 2.0), terrain, style))
        else:
            obs.append(make_bush(rng, x, z, rng.uniform(0.6, 1.6), terrain, style))
    return obs


def build_world(rng, terrain, style, scenario, goal_x, goal_z, ox=0.0, oz=0.0):
    """Obstacles for the requested scenario. goal_x/z are relative to the
    world origin (ox, oz); obstacle meshes are built at absolute positions so
    terrain heights bake in correctly."""
    span = max(abs(goal_x), 60.0) + 40.0
    depth = max(goal_z + 45.0, 25.0)   # endless mode can place goals behind
    obs = []
    # corridor scatter along the spawn->goal line so the rover regularly
    # meets obstacles regardless of scenario (dense supplies its own)
    d_goal = max(math.hypot(goal_x, goal_z), 1.0)
    ux, uz = goal_x / d_goal, goal_z / d_goal
    n_corr = 0 if scenario == "dense" else int(d_goal * rng.uniform(0.30, 0.55))
    for _ in range(n_corr):
        along = rng.uniform(8.0, d_goal)
        lateral = rng.uniform(-22.0, 22.0)
        cx = ox + ux * along - uz * lateral
        cz = oz + uz * along + ux * lateral
        r = rng.random()
        if r < 0.55:
            obs.append(make_rock(rng, cx, cz, rng.uniform(0.6, 2.4), terrain, style))
        elif r < 0.8:
            obs.append(make_tree(rng, cx, cz, rng.uniform(0.8, 2.0), terrain, style))
        else:
            obs.append(make_bush(rng, cx, cz, rng.uniform(0.6, 1.6), terrain, style))
    if scenario == "open":
        n = int(320 * (span * depth) / (200 * 165) * rng.uniform(0.6, 1.2))
        obs += _scatter(rng, terrain, style, (-span, span), (-25, depth), n,
                        (0.55, 0.25, 0.20), ox, oz)
    elif scenario == "dense":
        n = int(0.018 * (2 * 38) * depth * rng.uniform(0.75, 1.15))
        obs += _scatter(rng, terrain, style, (-38, 38), (-15, depth), n,
                        (0.15, 0.5, 0.35), ox, oz)
    elif scenario == "boulders":
        n = int(0.006 * (2 * span) * depth * rng.uniform(0.7, 1.2))
        for _ in range(n):
            x = ox + rng.uniform(-span, span)
            z = oz + rng.uniform(-20, depth)
            obs.append(make_rock(rng, x, z, rng.uniform(1.8, 4.0), terrain, style))
        obs += _scatter(rng, terrain, style, (-span, span), (-20, depth), n // 2,
                        (0.6, 0.1, 0.3), ox, oz)
    elif scenario == "wall":
        d_goal = math.hypot(goal_x, goal_z)
        ux, uz = goal_x / d_goal, goal_z / d_goal      # unit vector spawn->goal
        px, pz = -uz, ux                               # perpendicular
        n_walls = rng.integers(1, 4)
        fracs = sorted(rng.uniform(0.25, 0.85, n_walls))
        for fr in fracs:
            wx, wz = ox + goal_x * fr, oz + goal_z * fr
            gap = rng.uniform(-18, 18)
            gap_half = rng.uniform(3.2, 5.5)
            for t in np.arange(-34, 34.1, 2.6):
                if abs(t - gap) < gap_half:
                    continue
                jx, jz = rng.uniform(-0.8, 0.8, 2)
                obs.append(make_rock(rng, wx + px * t + jx, wz + pz * t + jz,
                                     rng.uniform(1.1, 1.9), terrain, style))
        n = int(0.002 * (2 * span) * depth)
        obs += _scatter(rng, terrain, style, (-span, span), (-20, depth), n,
                        (0.5, 0.25, 0.25), ox, oz)
    # ground clutter everywhere (non-colliding)
    for _ in range(int(rng.uniform(120, 260))):
        cx = ox + rng.uniform(-span, span)
        cz = oz + rng.uniform(-25, depth)
        obs.append(make_clutter(rng, cx, cz, terrain, style))
    return obs


# ==========================================================================
# Rover simulation
# ==========================================================================
class RoverSim:
    """Full episode simulation: world + physics + sensors + renderer."""

    def __init__(self, cfg: SimConfig = None, scenario: str = None,
                 spawn_mode: str = None, seed: int = None,
                 origin=(35.0, -120.0)):
        """Build one fully-randomized episode.

        scenario/spawn_mode are drawn from the seeded RNG when omitted, so a
        seed alone reproduces the entire world, style, and noise sequence.
        `origin` is the lat/lon anchor for the local metric frame.
        """
        self.cfg = cfg or SimConfig()
        self.rng = np.random.default_rng(seed)
        self.origin = origin
        rng = self.rng

        self.scenario = scenario or rng.choice(SCENARIOS)
        if spawn_mode is None:
            r = rng.random()
            spawn_mode = "uturn" if r < 0.15 else ("recovery" if r < 0.30 else "normal")
        self.spawn_mode = spawn_mode

        self.style = EpisodeStyle(rng)
        self.terrain = Terrain(rng, self.style.terrain_amp, self.style.terrain_wavelen)
        self.renderer = Renderer(self.cfg, self.style, self.terrain, rng)
        self.v_max = self.style.v_max

        # --- goal & world ---
        gd = rng.uniform(110, 150) if rng.random() < 0.15 else rng.uniform(40, 110)
        gb = math.radians(rng.uniform(-40, 40))
        self.goal_x = gd * math.sin(gb)
        self.goal_z = gd * math.cos(gb)
        self.obstacles = build_world(rng, self.terrain, self.style,
                                     self.scenario, self.goal_x, self.goal_z)

        # --- spawn ---
        self.x, self.z = 0.0, 0.0
        self.yaw = math.degrees(gb) + rng.uniform(-30, 30)
        if spawn_mode == "uturn":
            self.yaw = math.degrees(gb) + 180 + rng.uniform(-45, 45)
        # clear zones around spawn and goal
        self.obstacles = [o for o in self.obstacles if not (
            (o.radius > 0 and math.hypot(o.x, o.z) < 7.0) or
            (o.radius > 0 and math.hypot(o.x - self.goal_x, o.z - self.goal_z) < 6.0))]
        if spawn_mode == "recovery":
            a = math.radians(self.yaw + rng.uniform(-15, 15))
            scale = rng.uniform(1.8, 3.0)
            # place a big rock dead ahead with a small but collision-free gap
            d = 0.72 * scale + self.cfg.rover_radius + rng.uniform(0.5, 1.5)
            rx, rz = self.x + d * math.sin(a), self.z + d * math.cos(a)
            rock = make_rock(rng, rx, rz, scale, self.terrain, self.style)
            rock.radius = min(rock.radius, d - self.cfg.rover_radius - 0.3)
            self.obstacles.append(rock)

        self._collider_arrays()

        # --- dynamics state ---
        self.v = 0.0
        self.yaw_rate = 0.0
        self.cmd_queue = [(0.0, 0.0)] * self.cfg.latency_steps
        self.bump_phase = rng.uniform(0, 2 * np.pi, 2)
        self.t = 0.0
        self.frame = 0
        self.path_len = 0.0
        self.collided_now = False
        self.collision_count = 0

        # --- sensor biases (OU processes) ---
        self.gps_bias = np.zeros(2)
        self.heading_bias = 0.0
        self._sensor_step()

    # ------------- colliders -------------
    def _collider_arrays(self):
        """Cache obstacle centers/radii as arrays for vectorized queries."""
        cols = [(o.x, o.z, o.radius) for o in self.obstacles if o.radius > 0]
        if cols:
            arr = np.array(cols)
            self.obs_xz, self.obs_rad = arr[:, :2], arr[:, 2]
        else:
            self.obs_xz = np.zeros((0, 2))
            self.obs_rad = np.zeros(0)

    def clearance(self, x=None, z=None):
        """Distance from rover edge to nearest obstacle surface (m)."""
        if len(self.obs_rad) == 0:
            return 99.0
        p = np.array([self.x if x is None else x, self.z if z is None else z])
        d = np.linalg.norm(self.obs_xz - p, axis=1) - self.obs_rad - self.cfg.rover_radius
        return float(d.min())

    def _hits(self, x, z):
        """True if the rover footprint at (x, z) overlaps any obstacle."""
        if len(self.obs_rad) == 0:
            return False
        d = np.linalg.norm(self.obs_xz - np.array([x, z]), axis=1)
        return bool(np.any(d < self.obs_rad + self.cfg.rover_radius))

    def _nearest_normal(self, x, z):
        """Unit vector pointing away from the closest obstacle."""
        p = np.array([x, z])
        d = np.linalg.norm(self.obs_xz - p, axis=1)
        i = int(np.argmin(d - self.obs_rad))
        n = p - self.obs_xz[i]
        norm = np.linalg.norm(n)
        return n / norm if norm > 1e-9 else np.array([1.0, 0.0])

    # ------------- sensors -------------
    def _sensor_step(self):
        """Advance sensor noise: OU-process GPS/compass biases + white noise.

        The Ornstein-Uhlenbeck pull-back keeps the biases bounded (like real
        GPS multipath drift) instead of random-walking to infinity.
        """
        cfg, dt, rng = self.cfg, self.cfg.dt, self.rng
        tau = 30.0
        self.gps_bias += (-self.gps_bias / tau) * dt + \
            cfg.gps_walk_sigma * math.sqrt(dt) * rng.normal(size=2)
        self.heading_bias += (-self.heading_bias / tau) * dt + \
            cfg.heading_bias_sigma * math.sqrt(dt) / 3.0 * rng.normal()
        mx = self.x + self.gps_bias[0] + rng.normal(0, cfg.gps_noise_sigma)
        mz = self.z + self.gps_bias[1] + rng.normal(0, cfg.gps_noise_sigma)
        mh = self.yaw + self.heading_bias + rng.normal(0, cfg.heading_noise_sigma)
        mv = self.v * (1 + rng.normal(0, 0.03)) + rng.normal(0, 0.05)
        self.meas = {"x": mx, "z": mz, "heading": mh % 360.0, "speed": mv}

    def sensor_readout(self):
        """Noisy GPS/compass/odometry, exactly what a real rover would log."""
        lat, lon = meters_to_latlon(self.meas["x"], self.meas["z"], *self.origin)
        glat, glon = meters_to_latlon(self.goal_x, self.goal_z, *self.origin)
        return {"lat": lat, "lon": lon, "goal_lat": glat, "goal_lon": glon,
                "heading": self.meas["heading"], "speed": self.meas["speed"],
                "altitude": float(self.terrain.height(self.x, self.z))}

    def goal_vector_measured(self):
        """(distance, relative bearing) to goal as the noisy sensors see it."""
        s = self.sensor_readout()
        return goal_vector(s["lat"], s["lon"], s["heading"], s["goal_lat"], s["goal_lon"])

    def goal_dist_true(self):
        """True metric distance to goal (evaluation/termination only)."""
        return math.hypot(self.goal_x - self.x, self.goal_z - self.z)

    # ------------- dynamics -------------
    def step(self, throttle, steer):
        """Apply one control command (with actuation latency). Returns info dict."""
        cfg, dt, rng = self.cfg, self.cfg.dt, self.rng
        self.cmd_queue.append((float(np.clip(throttle, -1, 1)), float(np.clip(steer, -1, 1))))
        thr, st = self.cmd_queue.pop(0)

        # speed with acceleration limit + slip noise
        v_target = thr * self.v_max
        dv = np.clip(v_target - self.v, -cfg.accel_max * dt, cfg.accel_max * dt)
        self.v = (self.v + dv) * (1 + rng.normal(0, 0.01))

        # steering first-order lag + yaw slip
        yr_target = st * cfg.yaw_rate_max
        alpha = dt / max(cfg.steer_tau, dt)
        self.yaw_rate += (yr_target - self.yaw_rate) * min(alpha, 1.0)
        self.yaw += (self.yaw_rate + rng.normal(0, 0.6)) * dt

        rad = math.radians(self.yaw)
        nx = self.x + math.sin(rad) * self.v * dt
        nz = self.z + math.cos(rad) * self.v * dt

        was_contact = self.collided_now
        self.collided_now = False
        if self._hits(nx, nz):
            moved = False
            if self._hits(self.x, self.z):
                # already overlapping (rare): allow clearance-improving moves
                if self.clearance(nx, nz) > self.clearance():
                    moved = True
            else:
                # contact: slide along the obstacle tangent with friction,
                # the way a real rover scrapes past a rock instead of
                # freezing in place
                n = self._nearest_normal(self.x, self.z)
                m = np.array([nx - self.x, nz - self.z])
                mt = m - np.dot(m, n) * n
                sx, sz = self.x + 0.6 * mt[0], self.z + 0.6 * mt[1]
                if np.linalg.norm(mt) > 1e-4 and not self._hits(sx, sz):
                    nx, nz = sx, sz
                    moved = True
                    self.v *= 0.5
            self.collided_now = True
            if not was_contact:
                self.collision_count += 1
            if moved:
                self.path_len += math.hypot(nx - self.x, nz - self.z)
                self.x, self.z = nx, nz
            else:
                self.v = 0.0
        else:
            self.path_len += math.hypot(nx - self.x, nz - self.z)
            self.x, self.z = nx, nz

        self.t += dt
        self.frame += 1
        self.bump_phase += self.v * dt * np.array([1.7, 2.3]) + dt * np.array([2.0, 3.1])
        self._sensor_step()

        return {"collided": self.collided_now,
                "reached": self.goal_dist_true() < cfg.goal_radius,
                "timeout": self.frame >= cfg.max_frames,
                "clearance": self.clearance()}

    # ------------- rendering -------------
    def render(self):
        """Render the camera frame at the current pose, with terrain-driven
        pitch/roll plus speed-scaled ride-bump oscillation."""
        hx, hz = self.terrain.slope(self.x, self.z)
        rad = math.radians(self.yaw)
        fx, fz = math.sin(rad), math.cos(rad)
        pitch = math.degrees(math.atan2(hx * fx + hz * fz, 1.0))
        roll = math.degrees(math.atan2(hx * fz - hz * fx, 1.0)) * 0.8
        sf = abs(self.v) / max(self.v_max, 1e-6)
        b = self.style.bump_amp_deg * (0.3 + sf)
        pitch += b * math.sin(self.bump_phase[0])
        roll += b * 0.7 * math.sin(self.bump_phase[1])
        return self.renderer.render(self.x, self.z, self.yaw, pitch, roll,
                                    self.cfg.cam_height, self.obstacles, sf)

    # ------------- endless mode (live testing) -------------
    def respawn_goal(self):
        """New goal + fresh obstacle field around the current pose (endless runs)."""
        rng = self.rng
        self.scenario = rng.choice(SCENARIOS)
        gd = rng.uniform(50, 130)
        gb = math.radians(self.yaw + rng.uniform(-60, 60))
        ox, oz = self.x, self.z
        self.goal_x = ox + gd * math.sin(gb)
        self.goal_z = oz + gd * math.cos(gb)
        keep = [o for o in self.obstacles
                if math.hypot(o.x - ox, o.z - oz) < 25.0]
        gx_rel, gz_rel = self.goal_x - ox, self.goal_z - oz
        new = build_world(rng, self.terrain, self.style, self.scenario,
                          gx_rel, gz_rel, ox, oz)
        new = [o for o in new if not (
            (o.radius > 0 and math.hypot(o.x - ox, o.z - oz) < 8.0) or
            (o.radius > 0 and math.hypot(o.x - self.goal_x, o.z - self.goal_z) < 6.0))]
        self.obstacles = keep + new
        self._collider_arrays()
        self.frame = 0

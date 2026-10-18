"""World generation: terrain heightfield, props, and colliders.

A world is static for an episode. Terrain is a gridded heightfield with
the hazards that actually end rover missions (steep mound flanks, carved
washes with un-climbable banks, soft sand); props are instanced meshes
with circle colliders for the physics, the expert and the labels.

Drive-over rule: a prop shorter than the robot's step height renders and
appears in the obstacle-height labels, but gets no collider -- the rover
rolls over curb-sized rocks and crushes small shrubs.
"""

import math
from dataclasses import dataclass, field

import numpy as np

from . import assets

# material ids understood by the object shader
ROCK, FOLIAGE, BARK, PAINT, CLOTH, METAL, TUFT = range(7)
# class ids written to the class buffer
CLS_TERRAIN, CLS_OBSTACLE, CLS_DRIVEOVER, CLS_SKY = 0, 1, 2, 3

SCENARIOS = ("open", "dense", "wall", "boulders", "camp")


# ==========================================================================
# terrain
# ==========================================================================
@dataclass
class TerrainParams:
    """Per-episode relief and hazard counts."""
    amp: float = 1.0
    wavelen: float = 50.0
    n_mounds: int = 2
    n_washes: int = 1
    n_sand: int = 3
    rough_amp: float = 0.08

    @staticmethod
    def sample(rng):
        """Draw relief + hazards (zero-hazard episodes stay in the mix)."""
        return TerrainParams(amp=rng.uniform(0.2, 3.0),
                             wavelen=rng.uniform(30.0, 70.0),
                             n_mounds=int(rng.integers(0, 6)),
                             n_washes=int(rng.integers(0, 4)),
                             n_sand=int(rng.integers(0, 7)),
                             rough_amp=rng.uniform(0.0, 0.15))


class Terrain:
    """Gridded heightfield (1 m, +-300 m) with mounds, washes and sand.

    All lookups are bilinear and edge-clamped; the GPU renderer samples the
    same grid with the same interpolation, so what the camera sees is
    exactly the surface the physics drives on and the labels describe.
    """

    EXTENT = 300.0
    RES = 1.0

    def __init__(self, rng, tp: TerrainParams):
        n = int(2 * self.EXTENT / self.RES) + 1
        self.n = n
        ax = np.linspace(-self.EXTENT, self.EXTENT, n)
        X, Z = np.meshgrid(ax, ax, indexing="ij")
        H = np.zeros((n, n), np.float32)
        for _ in range(4):
            ang = rng.uniform(0, 2 * np.pi)
            k = 2 * np.pi / (tp.wavelen * rng.uniform(0.6, 1.6))
            H += (tp.amp / 2.0 * rng.uniform(0.5, 1.0) *
                  np.sin(k * (X * np.cos(ang) + Z * np.sin(ang)) + rng.uniform(0, 2 * np.pi))
                  ).astype(np.float32)
        # small-scale roughness (drivable bumps at 3-8 m)
        for _ in range(6):
            ang = rng.uniform(0, 2 * np.pi)
            k = 2 * np.pi / rng.uniform(3.0, 8.0)
            H += (tp.rough_amp / 6.0 * 2.0 *
                  np.sin(k * (X * np.cos(ang) + Z * np.sin(ang)) + rng.uniform(0, 2 * np.pi))
                  ).astype(np.float32)
        for _ in range(tp.n_mounds):
            mx, mz = rng.uniform(-180, 180, 2)
            r = rng.uniform(5.0, 14.0)
            a = rng.uniform(1.2, 3.2)
            H += (a * np.exp(-((X - mx) ** 2 + (Z - mz) ** 2) / (2 * (r / 2.2) ** 2))).astype(np.float32)
        self.wash_paths = []
        for _ in range(tp.n_washes):
            pts = [np.array([rng.uniform(-70, 70), rng.uniform(0, 140)])]
            heading = rng.choice([-1, 1]) * (np.pi / 2) + rng.uniform(-0.7, 0.7)
            for _ in range(40):
                heading += rng.uniform(-0.35, 0.35)
                pts.append(pts[-1] + 6.0 * np.array([np.sin(heading), np.cos(heading)]))
            pts = np.array(pts)
            self.wash_paths.append(pts)
            depth, width = rng.uniform(0.9, 2.0), rng.uniform(2.0, 3.5)
            carve = np.zeros((n, n), np.float32)
            for k in range(len(pts) - 1):
                p = pts[k]
                taper = min(1.0, k / 5.0, (len(pts) - 1 - k) / 5.0)
                w = int(3 * width / self.RES) + 1
                ci, cj = int((p[0] + self.EXTENT) / self.RES), int((p[1] + self.EXTENT) / self.RES)
                i0, i1, j0, j1 = max(0, ci - w), min(n, ci + w + 1), max(0, cj - w), min(n, cj + w + 1)
                if i1 <= i0 or j1 <= j0:
                    continue
                d2 = (X[i0:i1, j0:j1] - p[0]) ** 2 + (Z[i0:i1, j0:j1] - p[1]) ** 2
                cut = depth * taper * np.exp(-d2 / (2 * (width / 2.6) ** 2))
                carve[i0:i1, j0:j1] = np.maximum(carve[i0:i1, j0:j1], cut.astype(np.float32))
            H -= carve
        self.H = H
        S = np.zeros((n, n), np.float32)
        for _ in range(tp.n_sand):
            sx, sz = rng.uniform(-160, 160, 2)
            r = rng.uniform(4.0, 12.0)
            S += np.exp(-((X - sx) ** 2 + (Z - sz) ** 2) / (2 * (r / 1.6) ** 2)).astype(np.float32)
        for pts in self.wash_paths:            # sand collects in wash bottoms
            for p in pts[::2]:
                ci, cj = int((p[0] + self.EXTENT) / self.RES), int((p[1] + self.EXTENT) / self.RES)
                i0, i1, j0, j1 = max(0, ci - 4), min(n, ci + 5), max(0, cj - 4), min(n, cj + 5)
                if i1 <= i0 or j1 <= j0:
                    continue
                d2 = (X[i0:i1, j0:j1] - p[0]) ** 2 + (Z[i0:i1, j0:j1] - p[1]) ** 2
                S[i0:i1, j0:j1] += 0.7 * np.exp(-d2 / 8.0).astype(np.float32)
        self.S = np.clip(S, 0.0, 1.0)

    def _bilinear(self, grid, x, z):
        gx = np.clip((np.asarray(x, np.float32) + self.EXTENT) / self.RES, 0, self.n - 1.001)
        gz = np.clip((np.asarray(z, np.float32) + self.EXTENT) / self.RES, 0, self.n - 1.001)
        i0, j0 = gx.astype(np.int32), gz.astype(np.int32)
        fx, fz = gx - i0, gz - j0
        v = (grid[i0, j0] * (1 - fx) * (1 - fz) + grid[i0 + 1, j0] * fx * (1 - fz) +
             grid[i0, j0 + 1] * (1 - fx) * fz + grid[i0 + 1, j0 + 1] * fx * fz)
        return v if v.shape else float(v)

    def height(self, x, z):
        """Terrain height (m) at world (x, z)."""
        return self._bilinear(self.H, x, z)

    def sand(self, x, z):
        """Sand intensity in [0, 1]."""
        return self._bilinear(self.S, x, z)

    def slope(self, x, z, d=0.75):
        """(dh/dx, dh/dz) by central differences."""
        hx = (self.height(x + d, z) - self.height(x - d, z)) / (2 * d)
        hz = (self.height(x, z + d) - self.height(x, z - d)) / (2 * d)
        return hx, hz

    def grade(self, x, z, d=0.75):
        """Slope magnitude (rise / run)."""
        hx, hz = self.slope(x, z, d)
        return np.sqrt(hx * hx + hz * hz)


# ==========================================================================
# props
# ==========================================================================
@dataclass
class Prop:
    """One placed object (rendered by instancing, collided as circles)."""
    kind: str
    proto: int
    x: float
    z: float
    y: float
    yaw: float
    size: tuple                     # (sx, sy, sz) applied to the unit proto
    colors: tuple                   # three linear-RGB colours (one per part)
    materials: tuple                # material id per part
    height: float                   # physical height above ground (m)
    colliders: list = field(default_factory=list)   # [(x, z, r), ...]
    cls: int = CLS_OBSTACLE
    rigid_h: float = 0.0            # rigid-equivalent height: what the
    #                                 chassis actually meets (foliage crushes,
    #                                 so a shrub counts for less than its
    #                                 height; grass counts for nothing)
    footprint: float = 0.0          # ground-contact radius (m), for labels


class PropFactory:
    """Per-episode prototype meshes + physically-sized prop placement."""

    N_PROTO = {"rock": 14, "tree": 8, "bush": 8, "tuft": 6, "cone": 1, "person": 6,
               "vehicle": 4, "building": 3, "tent": 3, "post": 1}

    def __init__(self, rng, terrain, look, step_height=0.25):
        self.rng, self.terrain, self.look = rng, terrain, look
        self.step = step_height
        self.protos = {k: [assets.BUILDERS[k](rng) for _ in range(n)]
                       for k, n in self.N_PROTO.items()}
        # unit-proto extents: horizontal radius and height
        self.ext = {k: [(float(np.sqrt((m.pos[:, 0] ** 2 + m.pos[:, 2] ** 2).max())),
                         float(m.pos[:, 1].max())) for m in ms]
                    for k, ms in self.protos.items()}

    CRUSH = {"bush": 1.8, "tuft": float("inf")}   # foliage compresses

    def _place(self, kind, x, z, size, colors, materials, sink=0.0, yaw=None,
               collide=True, collider_r=None, extra_colliders=None,
               height_target=None):
        rng = self.rng
        k = int(rng.integers(len(self.protos[kind])))
        r_unit, h_unit = self.ext[kind][k]
        sx, sy, sz = size
        if height_target is not None:
            sy = height_target / max(h_unit, 1e-6)
        height = h_unit * sy * (1.0 - sink)
        y = float(self.terrain.height(x, z)) - h_unit * sy * sink
        yaw = rng.uniform(0, 2 * np.pi) if yaw is None else yaw
        rigid = 0.0 if not collide else height / self.CRUSH.get(kind, 1.0)
        cls = CLS_OBSTACLE if rigid >= self.step else CLS_DRIVEOVER
        cols = []
        if cls == CLS_OBSTACLE:
            if extra_colliders is not None:
                cols = extra_colliders
            else:
                r = collider_r if collider_r is not None else 0.92 * r_unit * max(sx, sz)
                cols = [(x, z, r)]
        return Prop(kind, k, x, z, y, yaw, (sx, sy, sz), colors, materials,
                    height, cols, cls, rigid, 0.85 * r_unit * max(sx, sz))

    def _pick(self, key):
        pal = self.look[key]
        return pal[self.rng.integers(len(pal))] * self.rng.uniform(0.85, 1.15, 3)

    # ---- kinds ----
    def rock(self, x, z, scale):
        rng = self.rng
        s = scale * rng.uniform(0.8, 1.25)
        hf = rng.uniform(0.7, 1.3)
        c = self._pick("rock")
        return self._place("rock", x, z, (s, s * hf, s * rng.uniform(0.8, 1.25)),
                           (c, c, c), (ROCK, ROCK, ROCK), sink=0.1)

    def tree(self, x, z, scale):
        rng = self.rng
        h = scale * rng.uniform(2.0, 3.5)
        w = h * rng.uniform(0.7, 1.2)
        r = float(np.clip(0.05 * w * 3, 0.35, 0.8))
        prop = self._place("tree", x, z, (w, h, w), (self._pick("foliage"), self._pick("bark"), self._pick("bark")),
                           (FOLIAGE, BARK, BARK), collider_r=r)
        prop.footprint = r          # the canopy is overhead; the trunk is what you hit
        return prop

    def bush(self, x, z, scale):
        rng = self.rng
        w = scale * rng.uniform(0.9, 1.5)
        h = w * rng.uniform(0.45, 0.9)
        c = self._pick("foliage") * rng.uniform(0.8, 1.1)
        return self._place("bush", x, z, (w, h / 0.6, w), (c, c, c), (FOLIAGE,) * 3, sink=0.08,
                           collider_r=0.45 * w)

    def tuft(self, x, z):
        rng = self.rng
        s = rng.uniform(0.15, 0.55)
        # brightness jitter + a little hue drift (per-channel jitter in linear
        # space turns dry grass pink)
        base = self.look["grassc"] * rng.uniform(0.7, 1.2) * rng.uniform(0.95, 1.05, 3)
        tip = np.clip(base * rng.uniform(1.1, 1.4), 0, 1)
        return self._place("tuft", x, z, (s, s * rng.uniform(0.6, 1.3), s), (base, base, tip),
                           (TUFT,) * 3, collide=False)

    def cone(self, x, z):
        s = self.rng.uniform(0.6, 1.0)
        org = np.array([0.95, 0.18, 0.02]) * self.rng.uniform(0.85, 1.05)
        return self._place("cone", x, z, (s, s, s), (org, np.array([0.02, 0.02, 0.02]), np.array([0.8, 0.8, 0.8])),
                           (PAINT,) * 3, collider_r=0.2 * s)

    def person(self, x, z):
        rng = self.rng
        h = rng.uniform(1.55, 1.95)
        shirt = rng.uniform(0.02, 0.8, 3)
        pants = rng.choice([np.array([0.05, 0.06, 0.12]), np.array([0.3, 0.25, 0.18]), rng.uniform(0.02, 0.4, 3)])
        skin = np.array([0.6, 0.4, 0.3]) * rng.uniform(0.3, 1.1)
        return self._place("person", x, z, (h, h, h), (shirt, pants, skin), (CLOTH,) * 3, collider_r=0.35)

    def vehicle(self, x, z):
        rng = self.rng
        L = rng.uniform(4.5, 7.0)
        yaw = rng.uniform(0, 2 * np.pi)
        body = rng.choice([np.array([0.85, 0.85, 0.85]), np.array([0.05, 0.05, 0.05]),
                           rng.uniform(0.05, 0.8, 3)])
        cols = [(x + math.sin(yaw) * t, z + math.cos(yaw) * t, 0.48 * L * 0.42)
                for t in np.linspace(-0.38 * L, 0.38 * L, 4)]
        return self._place("vehicle", x, z, (L, L, L), (body, np.array([0.03, 0.03, 0.03]),
                                                         np.array([0.05, 0.07, 0.09])),
                           (PAINT, BARK, PAINT), yaw=yaw, extra_colliders=cols)

    def building(self, x, z):
        rng = self.rng
        sx, sz = rng.uniform(4, 12), rng.uniform(3, 8)
        h = rng.uniform(2.5, 5.0)
        yaw = rng.uniform(0, 2 * np.pi)
        cy, sy = math.cos(yaw), math.sin(yaw)
        cols = []
        r = min(sx, sz) / 2
        for a in np.arange(-sx / 2 + r, sx / 2 - r + 1e-6, r * 0.8) if sx > 2 * r else [0.0]:
            for b in np.arange(-sz / 2 + r, sz / 2 - r + 1e-6, r * 0.8) if sz > 2 * r else [0.0]:
                cols.append((x + a * cy + b * sy, z - a * sy + b * cy, r * 1.05))
        wall = rng.choice([np.array([0.75, 0.75, 0.72]), np.array([0.55, 0.6, 0.62]), rng.uniform(0.2, 0.8, 3)])
        return self._place("building", x, z, (sx, 1.0, sz),
                           (wall, wall * 0.8, np.array([0.3, 0.3, 0.32])), (METAL, METAL, PAINT),
                           yaw=yaw, extra_colliders=cols, height_target=h)

    def tent(self, x, z):
        rng = self.rng
        s = rng.uniform(2.5, 4.0)
        c = rng.choice([np.array([0.8, 0.8, 0.82]), np.array([0.1, 0.25, 0.6]), np.array([0.7, 0.1, 0.08]),
                        rng.uniform(0.05, 0.9, 3)])
        return self._place("tent", x, z, (s, s, s), (c, c * 0.9, c), (CLOTH,) * 3, collider_r=0.62 * s)

    def post(self, x, z):
        h = self.rng.uniform(1.0, 1.8)
        return self._place("post", x, z, (h, h, h), (np.array([0.4, 0.4, 0.42]), np.array([0.9, 0.9, 0.9]),
                                                     np.array([0.9, 0.9, 0.9])),
                           (BARK, PAINT, PAINT), collider_r=0.15)

    def scatter_mix(self, x, z, mix):
        """One random rock / tree / bush with the given (rock, tree) odds."""
        r = self.rng.random()
        if r < mix[0]:
            return self.rock(x, z, self.rng.uniform(0.5, 2.2))
        if r < mix[0] + mix[1]:
            return self.tree(x, z, self.rng.uniform(0.8, 2.0))
        return self.bush(x, z, self.rng.uniform(0.6, 1.6))


def build_world(rng, fac: PropFactory, scenario, goal_x, goal_z, ox=0.0, oz=0.0):
    """Props for one scenario around the spawn (ox, oz) -> goal line."""
    span = max(abs(goal_x), 60.0) + 40.0
    depth = max(goal_z + 45.0, 25.0)
    out = []
    d_goal = max(math.hypot(goal_x, goal_z), 1.0)
    ux, uz = goal_x / d_goal, goal_z / d_goal
    n_corr = 0 if scenario == "dense" else int(d_goal * rng.uniform(0.30, 0.55))
    for _ in range(n_corr):
        along, lat = rng.uniform(8.0, d_goal), rng.uniform(-22.0, 22.0)
        out.append(fac.scatter_mix(ox + ux * along - uz * lat, oz + uz * along + ux * lat, (0.55, 0.25)))
    U = lambda a, b: rng.uniform(a, b)
    if scenario == "open":
        n = int(320 * (span * depth) / (200 * 165) * U(0.6, 1.2))
        out += [fac.scatter_mix(ox + U(-span, span), oz + U(-25, depth), (0.55, 0.25)) for _ in range(n)]
    elif scenario == "dense":
        n = int(0.018 * 76 * depth * U(0.75, 1.15))
        out += [fac.scatter_mix(ox + U(-38, 38), oz + U(-15, depth), (0.15, 0.5)) for _ in range(n)]
    elif scenario == "boulders":
        n = int(0.006 * (2 * span) * depth * U(0.7, 1.2))
        out += [fac.rock(ox + U(-span, span), oz + U(-20, depth), U(1.8, 4.0)) for _ in range(n)]
        out += [fac.scatter_mix(ox + U(-span, span), oz + U(-20, depth), (0.6, 0.1)) for _ in range(n // 2)]
    elif scenario == "wall":
        px, pz = -uz, ux
        for fr in sorted(rng.uniform(0.25, 0.85, rng.integers(1, 4))):
            wx, wz = ox + goal_x * fr, oz + goal_z * fr
            gap, gap_half = U(-18, 18), U(3.2, 5.5)
            for t in np.arange(-34, 34.1, 2.6):
                if abs(t - gap) < gap_half:
                    continue
                jx, jz = rng.uniform(-0.8, 0.8, 2)
                out.append(fac.rock(wx + px * t + jx, wz + pz * t + jz, U(1.1, 1.9)))
        n = int(0.002 * (2 * span) * depth)
        out += [fac.scatter_mix(ox + U(-span, span), oz + U(-20, depth), (0.5, 0.25)) for _ in range(n)]
    elif scenario == "camp":
        # URC-style base camp: vehicles, tents, sheds, people, cones, posts
        n = int(0.0012 * (2 * span) * depth * U(0.6, 1.3)) + 6
        kinds = ["vehicle", "tent", "building", "person", "person", "cone", "cone", "post"]
        for _ in range(n):
            k = kinds[rng.integers(len(kinds))]
            out.append(getattr(fac, k)(ox + U(-span, span), oz + U(-20, depth)))
        out += [fac.scatter_mix(ox + U(-span, span), oz + U(-20, depth), (0.6, 0.15))
                for _ in range(int(n * 0.8))]
    # a few people / cones / posts anywhere: real sites are never empty
    for _ in range(rng.integers(0, 6)):
        k = ["person", "cone", "post", "vehicle"][rng.integers(4)]
        out.append(getattr(fac, k)(ox + U(-span * 0.7, span * 0.7), oz + U(5, depth)))
    # small clutter: mostly drive-over pebbles / shrubs (the same height rule
    # decides; a few come out just tall enough to matter)
    for _ in range(int(U(150, 320))):
        x, z = ox + U(-span, span), oz + U(-25, depth)
        out.append(fac.rock(x, z, U(0.12, 0.42)) if rng.random() < 0.65
                   else fac.bush(x, z, U(0.2, 0.5)))
    # grass tufts: purely visual, clumped by a smooth random density field
    area = (2 * span) * (depth + 25)
    n_t = int(min(30000, area * fac.look["tuft_density"]))
    if n_t:
        k = rng.uniform(0.02, 0.08, (3, 2)) * rng.choice([-1, 1], (3, 2))
        ph = rng.uniform(0, 2 * np.pi, 3)
        xs = ox + rng.uniform(-span, span, 3 * n_t)
        zs = oz + rng.uniform(-25, depth, 3 * n_t)
        dens = sum(np.sin(k[i, 0] * xs + k[i, 1] * zs + ph[i]) for i in range(3)) / 3.0
        keep = rng.random(len(xs)) < np.clip(0.35 + 0.65 * dens, 0.02, 1.0)
        for x, z in list(zip(xs[keep], zs[keep]))[:n_t]:
            out.append(fac.tuft(x, z))
    return out


def colliders(props):
    """Flatten prop colliders -> (obs_xz (k, 2), obs_rad (k,))."""
    cols = [c for p in props for c in p.colliders]
    if not cols:
        return np.zeros((0, 2)), np.zeros(0)
    a = np.array(cols, float)
    return a[:, :2], a[:, 2]


def render_groups(fac: PropFactory, props):
    """Group props by (kind, proto) into instanced draw batches."""
    from .render import instance_rows
    by = {}
    for p in props:
        by.setdefault((p.kind, p.proto), []).append(p)
    groups = []
    for (kind, k), ps in by.items():
        models = []
        for p in ps:
            c, s = math.cos(p.yaw), math.sin(p.yaw)
            M = np.array([[c * p.size[0], 0, s * p.size[2], p.x],
                          [0, p.size[1], 0, p.y],
                          [-s * p.size[0], 0, c * p.size[2], p.z],
                          [0, 0, 0, 1]], np.float32)
            models.append(M)
        rows = instance_rows(np.array(models),
                             np.array([p.colors[0] for p in ps]),
                             np.array([p.colors[1] for p in ps]),
                             np.array([p.colors[2] for p in ps]),
                             np.array([p.materials for p in ps]),
                             np.array([p.cls for p in ps]))
        groups.append((fac.protos[kind][k], rows, kind == "tuft"))
    return groups

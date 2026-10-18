"""Procedural meshes for the simulator.

Every function returns a `Mesh`: float32 positions (N, 3) and normals
(N, 3), a float32 per-vertex part index (N,) that selects one of the
instance's three colours, and uint32 triangle indices (M, 3). Local frame:
y up, the object's base sits on y = 0, roughly unit-sized so instances can
scale it to physical dimensions.

Shapes are randomized per episode (a handful of prototypes per kind, each
instanced many times with its own scale / rotation / colour), which keeps
draw calls low while still giving every rock its own silhouette.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class Mesh:
    """Indexed triangle mesh with per-vertex normals and part ids."""
    pos: np.ndarray
    nrm: np.ndarray
    part: np.ndarray
    idx: np.ndarray

    def vertex_bytes(self):
        """Interleaved (pos, nrm, part) float32 vertex buffer."""
        v = np.concatenate([self.pos, self.nrm, self.part[:, None]], axis=1)
        return v.astype(np.float32).tobytes()

    def index_bytes(self):
        """uint32 index buffer."""
        return self.idx.astype(np.uint32).tobytes()


def merge(meshes):
    """Concatenate meshes into one (indices re-based)."""
    pos, nrm, part, idx, off = [], [], [], [], 0
    for m in meshes:
        pos.append(m.pos)
        nrm.append(m.nrm)
        part.append(m.part)
        idx.append(m.idx + off)
        off += len(m.pos)
    return Mesh(np.concatenate(pos), np.concatenate(nrm),
                np.concatenate(part), np.concatenate(idx))


def vertex_normals(pos, idx):
    """Area-weighted smooth vertex normals."""
    a, b, c = pos[idx[:, 0]], pos[idx[:, 1]], pos[idx[:, 2]]
    fn = np.cross(b - a, c - a)
    n = np.zeros_like(pos)
    for k in range(3):
        np.add.at(n, idx[:, k], fn)
    return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)


def transform(m: Mesh, scale=(1, 1, 1), offset=(0, 0, 0), part=None):
    """Scale + translate a mesh (normals re-derived for the scale)."""
    s = np.asarray(scale, np.float32)
    pos = m.pos * s + np.asarray(offset, np.float32)
    nrm = m.nrm / s
    nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-9)
    p = m.part if part is None else np.full(len(pos), part, np.float32)
    return Mesh(pos.astype(np.float32), nrm.astype(np.float32), p, m.idx)


# --------------------------------------------------------------------------
# primitives
# --------------------------------------------------------------------------
def icosphere(subdiv=2):
    """Unit icosphere centred at the origin."""
    t = (1.0 + 5 ** 0.5) / 2.0
    v = [(-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0),
         (0, -1, t), (0, 1, t), (0, -1, -t), (0, 1, -t),
         (t, 0, -1), (t, 0, 1), (-t, 0, -1), (-t, 0, 1)]
    f = [(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11),
         (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
         (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9),
         (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1)]
    verts = [np.array(p, float) / np.linalg.norm(p) for p in v]
    faces = f
    for _ in range(subdiv):
        cache, nf = {}, []

        def mid(a, b, cache=cache):
            key = (min(a, b), max(a, b))
            if key not in cache:
                m = verts[a] + verts[b]
                verts.append(m / np.linalg.norm(m))
                cache[key] = len(verts) - 1
            return cache[key]
        for a, b, c in faces:
            ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
            nf += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
        faces = nf
    pos = np.array(verts, np.float32)
    idx = np.array(faces, np.uint32)
    return Mesh(pos, pos.copy(), np.zeros(len(pos), np.float32), idx)


def cylinder(sides=10, r0=1.0, r1=1.0, h=1.0, caps=True):
    """Tapered cylinder along +y from y=0 to y=h (smooth sides)."""
    ang = np.linspace(0, 2 * np.pi, sides, endpoint=False)
    ca, sa = np.cos(ang), np.sin(ang)
    bot = np.stack([r0 * ca, np.zeros(sides), r0 * sa], 1)
    top = np.stack([r1 * ca, np.full(sides, h), r1 * sa], 1)
    slope = (r0 - r1) / max(h, 1e-6)
    sn = np.stack([ca, np.full(sides, slope), sa], 1)
    sn /= np.linalg.norm(sn, axis=1, keepdims=True)
    pos = [bot, top]
    nrm = [sn, sn]
    idx = []
    for k in range(sides):
        a, b = k, (k + 1) % sides
        idx += [(a, sides + a, b), (b, sides + a, sides + b)]
    idx = np.array(idx)
    if caps:
        base = 2 * sides
        pos += [np.array([[0, h, 0]]), top, np.array([[0, 0, 0]]), bot]
        nrm += [np.array([[0, 1, 0]]), np.tile([0, 1, 0], (sides, 1)),
                np.array([[0, -1, 0]]), np.tile([0, -1, 0], (sides, 1))]
        tc, tr = base, base + 1
        bc, br = base + 1 + sides, base + 2 + sides
        cap = []
        for k in range(sides):
            a, b = k, (k + 1) % sides
            cap += [(tc, tr + b, tr + a), (bc, br + a, br + b)]
        idx = np.concatenate([idx, np.array(cap)])
    pos = np.concatenate(pos).astype(np.float32)
    nrm = np.concatenate(nrm).astype(np.float32)
    return Mesh(pos, nrm, np.zeros(len(pos), np.float32),
                idx.astype(np.uint32))


def box(sx=1.0, sy=1.0, sz=1.0):
    """Axis-aligned box with flat-shaded faces, base on y=0."""
    hx, hz = sx / 2, sz / 2
    faces = [((0, 1, 0), [(-hx, sy, -hz), (-hx, sy, hz), (hx, sy, hz), (hx, sy, -hz)]),
             ((0, -1, 0), [(-hx, 0, -hz), (hx, 0, -hz), (hx, 0, hz), (-hx, 0, hz)]),
             ((1, 0, 0), [(hx, 0, -hz), (hx, sy, -hz), (hx, sy, hz), (hx, 0, hz)]),
             ((-1, 0, 0), [(-hx, 0, -hz), (-hx, 0, hz), (-hx, sy, hz), (-hx, sy, -hz)]),
             ((0, 0, 1), [(-hx, 0, hz), (hx, 0, hz), (hx, sy, hz), (-hx, sy, hz)]),
             ((0, 0, -1), [(-hx, 0, -hz), (-hx, sy, -hz), (hx, sy, -hz), (hx, 0, -hz)])]
    pos, nrm, idx = [], [], []
    for n, quad in faces:
        b = len(pos)
        pos += quad
        nrm += [n] * 4
        idx += [(b, b + 1, b + 2), (b, b + 2, b + 3)]
    return Mesh(np.array(pos, np.float32), np.array(nrm, np.float32),
                np.zeros(len(pos), np.float32), np.array(idx, np.uint32))


def _displace(m: Mesh, rng, amp=0.25, n_waves=10, freq=(1.0, 3.5)):
    """Radial displacement by random directional waves (rock-like lumps)."""
    p = m.pos.astype(np.float64)
    r = np.ones(len(p))
    for _ in range(n_waves):
        d = rng.normal(size=3)
        d /= np.linalg.norm(d)
        f = rng.uniform(*freq)
        r += amp / n_waves * 2.0 * np.sin(f * np.pi * (p @ d) + rng.uniform(0, 2 * np.pi))
    p = p * r[:, None]
    return p


# --------------------------------------------------------------------------
# props
# --------------------------------------------------------------------------
def rock(rng):
    """Lumpy, partially buried boulder; unit footprint, height ~0.6."""
    m = icosphere(3)
    p = _displace(m, rng, amp=rng.uniform(0.18, 0.35))
    # facets: quantize a few random planes to give fractured faces
    for _ in range(rng.integers(2, 6)):
        d = rng.normal(size=3)
        d /= np.linalg.norm(d)
        cut = rng.uniform(0.55, 0.85)
        s = p @ d
        p -= np.maximum(s - cut, 0)[:, None] * d
    p[:, 1] *= rng.uniform(0.55, 0.85)
    p[:, 1] = np.maximum(p[:, 1], -0.15)          # flat buried base
    p[:, 1] += 0.12
    p /= max(np.abs(p[:, [0, 2]]).max(), 1e-6) * 2.0   # footprint ~1
    p[:, 1] -= p[:, 1].min() + 0.02
    pos = p.astype(np.float32)
    return Mesh(pos, vertex_normals(pos, m.idx), m.part, m.idx)


def foliage_clump(rng, n_blobs=(3, 6), flatten=1.0, part=0):
    """Several lumpy spheres merged into a canopy / bush volume (unit-ish)."""
    blobs = []
    for _ in range(rng.integers(*n_blobs)):
        s = icosphere(2)
        p = _displace(s, rng, amp=0.35, n_waves=14, freq=(2.0, 6.0))
        r = rng.uniform(0.35, 0.6)
        c = rng.normal(0, 0.28, 3) * np.array([1.0, 0.6 * flatten, 1.0])
        p = p * r + c
        p = p.astype(np.float32)
        blobs.append(Mesh(p, vertex_normals(p, s.idx),
                          np.full(len(p), part, np.float32), s.idx))
    m = merge(blobs)
    m.pos[:, 1] -= m.pos[:, 1].min()
    return m


def tree(rng):
    """Trunk (part 1) + clumped canopy (part 0); unit height."""
    th = rng.uniform(0.35, 0.6)
    trunk = cylinder(8, 0.05, 0.03, th + 0.15)
    trunk.part[:] = 1
    can = foliage_clump(rng, (4, 8))
    ext = max(np.ptp(can.pos[:, 0]), np.ptp(can.pos[:, 2]))
    can = transform(can, (0.75 / ext, 0.7 / max(np.ptp(can.pos[:, 1]), 1e-6) * rng.uniform(0.5, 0.75), 0.75 / ext),
                    (0, th, 0), part=0)
    m = merge([trunk, can])
    m.pos[:, 1] /= m.pos[:, 1].max()
    m.nrm = vertex_normals(m.pos, m.idx)
    return m


def bush(rng):
    """Low shrub: flattened foliage clump with unit footprint."""
    m = foliage_clump(rng, (2, 5), flatten=0.6)
    ext = max(np.ptp(m.pos[:, 0]), np.ptp(m.pos[:, 2]))
    return transform(m, (1 / ext, 0.6 / max(np.ptp(m.pos[:, 1]), 1e-6), 1 / ext))


def tuft(rng):
    """Grass tuft: thin blades fanning out from the centre (double-sided)."""
    pos, idx = [], []
    for _ in range(rng.integers(8, 16)):
        a = rng.uniform(0, 2 * np.pi)
        lean = rng.uniform(0.1, 0.6)
        h = rng.uniform(0.5, 1.0)
        w = rng.uniform(0.03, 0.06)
        base = np.array([np.cos(a), 0, np.sin(a)]) * rng.uniform(0, 0.15)
        tip = base + np.array([np.cos(a) * lean, h, np.sin(a) * lean])
        side = np.array([-np.sin(a), 0, np.cos(a)]) * w
        b = len(pos)
        pos += [base - side, base + side, tip]
        idx.append((b, b + 1, b + 2))
    pos = np.array(pos, np.float32)
    idx = np.array(idx, np.uint32)
    nrm = np.tile(np.array([[0, 1, 0]], np.float32), (len(pos), 1))
    part = np.tile(np.array([0, 0, 2], np.float32), len(idx))   # 2 = tip
    return Mesh(pos, nrm, part, idx)


def cone_marker(rng=None):
    """Traffic cone: orange cone (part 0) on a square base (part 1)."""
    body = cylinder(16, 0.16, 0.025, 0.7, caps=False)
    base = box(0.38, 0.03, 0.38)
    base.part[:] = 1
    stripe = cylinder(16, 0.115, 0.08, 0.12, caps=False)
    stripe = transform(stripe, offset=(0, 0.32, 0), part=2)
    stripe.pos[:, [0, 2]] *= 1.02
    return merge([base, transform(body, offset=(0, 0.03, 0)), stripe])


def person(rng):
    """Standing person ~1 unit tall: legs (part 1), torso (0), head (2)."""
    legs = [transform(cylinder(8, 0.07, 0.06, 0.47), offset=(dx, 0, 0), part=1)
            for dx in (-0.08, 0.08)]
    torso = transform(cylinder(10, 0.17, 0.15, 0.33), offset=(0, 0.47, 0), part=0)
    sph = icosphere(1)
    head = transform(sph, (0.11, 0.13, 0.11), (0, 0.9, 0), part=2)
    arms = [transform(cylinder(6, 0.045, 0.04, 0.32), offset=(dx, 0.48, 0), part=0)
            for dx in (-0.21, 0.21)]
    return merge(legs + [torso, head] + arms)


def vehicle(rng):
    """Pickup / box truck: body (part 0), cab windows (2), wheels (1).

    Unit length along z; width ~0.4, height ~0.38."""
    kind = rng.integers(0, 2)
    parts = []
    if kind == 0:                                     # pickup
        parts.append(transform(box(0.4, 0.17, 1.0), offset=(0, 0.08, 0)))
        parts.append(transform(box(0.38, 0.13, 0.35), offset=(0, 0.25, 0.12), part=2))
    else:                                             # box truck
        parts.append(transform(box(0.42, 0.33, 0.68), offset=(0, 0.09, -0.15)))
        parts.append(transform(box(0.4, 0.2, 0.3), offset=(0, 0.08, 0.34)))
        parts.append(transform(box(0.38, 0.09, 0.02), offset=(0, 0.2, 0.495), part=2))
    for wx in (-0.19, 0.19):
        for wz in (-0.32, 0.32):
            w = cylinder(12, 0.09, 0.09, 0.07)
            p = w.pos[:, [1, 0, 2]].copy()
            p[:, 0] -= 0.035
            n = w.nrm[:, [1, 0, 2]].copy()
            w = Mesh(p + np.array([wx, 0.09, wz], np.float32), n,
                     np.full(len(p), 1, np.float32), w.idx[:, [0, 2, 1]])
            parts.append(w)
    return merge(parts)


def building(rng):
    """Shed / trailer: walls (part 0) + roof (part 1); unit footprint."""
    h = rng.uniform(0.35, 0.6)
    walls = box(1.0, h, 1.0)
    roof = transform(box(1.04, 0.04, 1.04), offset=(0, h, 0), part=1)
    door = transform(box(0.2, 0.6 * h, 0.01), offset=(rng.uniform(-0.3, 0.3), 0, 0.505), part=2)
    return merge([walls, roof, door])


def tent(rng):
    """Enclosed camping / canopy tent: walls (part 1) under a roof (part 0).

    Solid on purpose: a rover should treat a tent as an obstacle, not a
    gap between four legs."""
    wall_h = rng.uniform(0.45, 0.75)
    walls = transform(box(0.96, wall_h, 0.96), part=1)
    roof = cylinder(4, 0.72, 0.02, rng.uniform(0.2, 0.35), caps=True)
    p = roof.pos.copy()
    c, s = np.cos(np.pi / 4), np.sin(np.pi / 4)
    p[:, [0, 2]] = p[:, [0, 2]] @ np.array([[c, -s], [s, c]])
    roof = Mesh(p + np.array([0, wall_h, 0], np.float32), roof.nrm, roof.part, roof.idx)
    roof.nrm = vertex_normals(roof.pos, roof.idx)
    return merge([walls, roof])


def post(rng):
    """Marker post with a small square sign (part 1) near the top."""
    pole = cylinder(8, 0.02, 0.02, 1.0)
    sign = transform(box(0.2, 0.2, 0.01), offset=(0, 0.72, 0.025), part=1)
    return merge([pole, sign])


def rover_body():
    """Blocky rover silhouette (shadow caster only): chassis, mast, wheels."""
    parts = [transform(box(0.9, 0.35, 1.2), offset=(0, 0.35, 0)),
             transform(box(0.08, 0.6, 0.08), offset=(0, 0.7, 0.35)),
             transform(box(0.35, 0.12, 0.12), offset=(0, 1.3, 0.38))]
    for wx in (-0.55, 0.55):
        for wz in (-0.45, 0.0, 0.45):
            parts.append(transform(box(0.18, 0.4, 0.4), offset=(wx, 0, wz)))
    return merge(parts)


# prototype builders keyed by prop kind; the renderer draws several random
# prototypes per kind per episode
BUILDERS = {
    "rock": rock, "tree": tree, "bush": bush, "tuft": tuft,
    "cone": cone_marker, "person": person, "vehicle": vehicle,
    "building": building, "tent": tent, "post": post,
}

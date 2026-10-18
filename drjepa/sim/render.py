"""GPU renderer for the simulator (OpenGL 3.3 via moderngl, headless EGL).

One `GLRenderer` per episode: it uploads the terrain heightfield / sand
field as float textures (the vertex shader displaces a camera-centred grid
by exact bilinear lookups, so the rendered ground is the same surface the
physics and labels use) and every prop as instanced prototype meshes.

Each `render()` returns, at the requested camera intrinsics:
    rgb    (H, W, 3) uint8   tone-mapped, vignetted image (RGB order)
    depth  (H, W)    float32 metric depth along the optical axis, 0 = sky
    cls    (H, W)    uint8   0 terrain, 1 obstacle, 2 drive-over, 3 sky

Rendering is supersampled (ss x ss) and resolved on the GPU; depth/class
come from the nearest sub-sample so object edges stay sharp. Sensor noise,
motion blur and compression are applied afterwards by the caller
(see drjepa.sim.camera_fx), since they are cheap on the CPU and easier to
randomize there.
"""

import math

import numpy as np

from . import shaders

_CTX = None


def gl_context():
    """Process-wide headless OpenGL context (created on first use)."""
    global _CTX
    if _CTX is None:
        import moderngl
        try:
            _CTX = moderngl.create_standalone_context(backend="egl", require=330)
        except Exception:                                   # pragma: no cover
            _CTX = moderngl.create_standalone_context(require=330)
    return _CTX


def _mat_bytes(m):
    """Row-major numpy 4x4 -> column-major float32 bytes for GLSL."""
    return np.ascontiguousarray(np.asarray(m, np.float32).T).tobytes()


def projection_from_intrinsics(fx, fy, cx, cy, w, h, near=0.08, far=4000.0):
    """OpenGL projection matrix for a pinhole camera (pixel v grows down)."""
    a = -(far + near) / (far - near)
    b = -2.0 * far * near / (far - near)
    return np.array([[2 * fx / w, 0.0, 1.0 - 2 * cx / w, 0.0],
                     [0.0, 2 * fy / h, 2 * cy / h - 1.0, 0.0],
                     [0.0, 0.0, a, b],
                     [0.0, 0.0, -1.0, 0.0]])


def view_matrix(cam_pos, R):
    """World -> eye. R columns are the camera right/up/forward axes in world."""
    right, up, fwd = R[:, 0], R[:, 1], R[:, 2]
    rot = np.stack([right, up, -fwd])
    m = np.eye(4)
    m[:3, :3] = rot
    m[:3, 3] = -rot @ np.asarray(cam_pos, float)
    return m


def _light_vp(center, sun_dir, half, texels):
    """Orthographic sun camera centred on `center`, snapped to shadow texels."""
    L = np.asarray(sun_dir, float) / np.linalg.norm(sun_dir)
    up = np.array([0.0, 1.0, 0.0]) if abs(L[1]) < 0.95 else np.array([1.0, 0.0, 0.0])
    fwd = -L
    right = np.cross(fwd, up)
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    rot = np.stack([right, up, -fwd])
    c = rot @ np.asarray(center, float)
    step = 2 * half / texels
    c[0] = math.floor(c[0] / step) * step                   # no shimmer
    c[1] = math.floor(c[1] / step) * step
    view = np.eye(4)
    view[:3, :3] = rot
    view[:3, 3] = -c
    view[2, 3] -= 250.0              # eye sits 250 m from the centre, sunward
    near, far = 1.0, 600.0
    proj = np.array([[1 / half, 0, 0, 0], [0, 1 / half, 0, 0],
                     [0, 0, -2 / (far - near), -(far + near) / (far - near)],
                     [0, 0, 0, 1]])
    return proj @ view


def _terrain_axis(near_step=0.5, near_half=40.0, growth=1.055, far=2000.0):
    """1-D vertex offsets: uniform near the camera, geometric further out."""
    xs = list(np.arange(0.0, near_half + 1e-6, near_step))
    s = near_step
    while xs[-1] < far:
        s *= growth
        xs.append(xs[-1] + s)
    xs = np.array(xs)
    return np.concatenate([-xs[:0:-1], xs]).astype(np.float32)


class GLRenderer:
    """Per-episode GPU renderer. See module docstring for outputs."""

    INST_FLOATS = 29        # mat4 + 3 colours + 3 part materials + class

    def __init__(self, width, height, terrain, look, groups, rover_mesh=None,
                 ss=2, shadow_res=2048, shadow_half=55.0):
        """Upload terrain textures, prop instances, and build GL programs.

        terrain : has H, S (n, n) [i=x, j=z] grids, EXTENT, RES
        look    : dict of shading parameters (see drjepa.sim.style)
        groups  : list of (Mesh, instances float32 (k, 27), double_sided)
        """
        import moderngl
        self.mgl = moderngl
        self.ctx = ctx = gl_context()
        self.W, self.H, self.ss = int(width), int(height), int(ss)
        self.look = look
        self.shadow_res, self.shadow_half = shadow_res, shadow_half
        self._res = []                                      # GL objects to free

        def keep(o):
            self._res.append(o)
            return o

        # ---- programs ----
        self.p_terrain = keep(ctx.program(vertex_shader=shaders.TERRAIN_VS,
                                          fragment_shader=shaders.TERRAIN_FS))
        self.p_obj = keep(ctx.program(vertex_shader=shaders.OBJ_VS,
                                      fragment_shader=shaders.OBJ_FS))
        self.p_sky = keep(ctx.program(vertex_shader=shaders.FULLSCREEN_VS,
                                      fragment_shader=shaders.SKY_FS))
        self.p_resolve = keep(ctx.program(vertex_shader=shaders.FULLSCREEN_VS,
                                          fragment_shader=shaders.RESOLVE_FS))
        self.p_sh_terrain = keep(ctx.program(vertex_shader=shaders.SHADOW_VS_TERRAIN,
                                             fragment_shader=shaders.SHADOW_FS))
        self.p_sh_obj = keep(ctx.program(vertex_shader=shaders.SHADOW_VS_OBJ,
                                         fragment_shader=shaders.SHADOW_FS))

        # ---- terrain textures (exact bilinear lookups in the shaders) ----
        n = terrain.H.shape[0]
        self.t_h = keep(ctx.texture((n, n), 1, np.ascontiguousarray(
            terrain.H.T, np.float32).tobytes(), dtype="f4"))
        self.t_s = keep(ctx.texture((n, n), 1, np.ascontiguousarray(
            terrain.S.T, np.float32).tobytes(), dtype="f4"))
        for t in (self.t_h, self.t_s):
            t.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self.t_h.use(location=0)
        self.t_s.use(location=1)
        for p in (self.p_terrain, self.p_sh_terrain):
            for k, v in (("u_hmap", 0), ("u_smap", 1)):
                self._set(p, k, v)
            for k, v in (("u_extent", terrain.EXTENT), ("u_res", terrain.RES),
                         ("u_n", float(n))):
                self._set(p, k, float(v))

        # ---- terrain grid ----
        ax = _terrain_axis()
        gx, gz = np.meshgrid(ax, ax, indexing="ij")
        xz = np.stack([gx.ravel(), gz.ravel()], 1).astype(np.float32)
        m = len(ax)
        ii, jj = np.meshgrid(np.arange(m - 1), np.arange(m - 1), indexing="ij")
        a = (ii * m + jj).ravel()
        tri = np.stack([a, a + 1, a + m, a + 1, a + m + 1, a + m], 1).reshape(-1, 3)
        self.vb_grid = keep(ctx.buffer(xz.tobytes()))
        self.ib_grid = keep(ctx.buffer(tri.astype(np.uint32).tobytes()))
        self.va_terrain = keep(ctx.vertex_array(
            self.p_terrain, [(self.vb_grid, "2f", "in_xz")], self.ib_grid))
        self.va_sh_terrain = keep(ctx.vertex_array(
            self.p_sh_terrain, [(self.vb_grid, "2f", "in_xz")], self.ib_grid))

        # ---- props ----
        self.groups = []
        for mesh, inst, double in groups:
            if len(inst) == 0:
                continue
            vb = keep(ctx.buffer(mesh.vertex_bytes()))
            ib = keep(ctx.buffer(mesh.index_bytes()))
            buf = keep(ctx.buffer(np.ascontiguousarray(inst, np.float32).tobytes()))
            va = keep(ctx.vertex_array(self.p_obj, [
                (vb, "3f 3f 1f", "in_pos", "in_nrm", "in_part"),
                (buf, "16f 3f 3f 3f 4f/i", "i_model", "i_c0", "i_c1", "i_c2", "i_mat")], ib))
            vs = keep(ctx.vertex_array(self.p_sh_obj, [
                (vb, "3f 16x", "in_pos"),
                (buf, "16f 52x/i", "i_model")], ib))
            self.groups.append((va, vs, len(inst), double))
        self.rover = None
        if rover_mesh is not None:
            vb = keep(ctx.buffer(rover_mesh.vertex_bytes()))
            ib = keep(ctx.buffer(rover_mesh.index_bytes()))
            self.rover_buf = keep(ctx.buffer(reserve=self.INST_FLOATS * 4))
            self.rover = keep(ctx.vertex_array(self.p_sh_obj, [
                (vb, "3f 16x", "in_pos"), (self.rover_buf, "16f 52x/i", "i_model")], ib))

        self.va_sky = keep(ctx.vertex_array(self.p_sky, []))
        self.va_resolve = keep(ctx.vertex_array(self.p_resolve, []))

        # ---- render targets ----
        Ws, Hs = self.W * self.ss, self.H * self.ss
        self.t_hdr = keep(ctx.texture((Ws, Hs), 4, dtype="f2"))
        self.t_lin = keep(ctx.texture((Ws, Hs), 1, dtype="f4"))
        self.t_cls = keep(ctx.texture((Ws, Hs), 1, dtype="f1"))
        for t in (self.t_hdr, self.t_lin, self.t_cls):
            t.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self.t_zbuf = keep(ctx.depth_texture((Ws, Hs)))
        self.fbo_main = keep(ctx.framebuffer([self.t_hdr, self.t_lin, self.t_cls], self.t_zbuf))
        self.t_out = keep(ctx.texture((self.W, self.H), 4, dtype="f1"))
        self.t_dout = keep(ctx.texture((self.W, self.H), 1, dtype="f4"))
        self.t_cout = keep(ctx.texture((self.W, self.H), 1, dtype="f1"))
        self.fbo_out = keep(ctx.framebuffer([self.t_out, self.t_dout, self.t_cout]))
        self.t_shadow = keep(ctx.depth_texture((shadow_res, shadow_res)))
        self.t_shadow.compare_func = "<="
        self.t_shadow.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.t_shadow.repeat_x = self.t_shadow.repeat_y = False
        self.fbo_shadow = keep(ctx.framebuffer(depth_attachment=self.t_shadow))

        self._set_look(look)

    # ------------------------------------------------------------------
    def _set(self, prog, name, value):
        u = prog.get(name, None)
        if u is None:
            return
        if isinstance(value, np.ndarray) and value.shape == (4, 4):
            u.write(_mat_bytes(value))
        elif isinstance(value, (tuple, list, np.ndarray)):
            u.value = tuple(float(x) for x in np.ravel(value))
        else:
            u.value = value

    def _set_look(self, look):
        light = ("u_sun_dir", "u_sun_col", "u_amb_sky", "u_amb_gnd",
                 "u_fog_col", "u_fog_dist")
        for p in (self.p_terrain, self.p_obj):
            for k in light:
                self._set(p, k, look[k[2:]])
            self._set(p, "u_shadow_texel", 1.0 / self.shadow_res)
        for k in ("ga", "gb", "gc", "sandc", "grassc", "grass_amt",
                  "grass_thr", "rough", "seed"):
            self._set(self.p_terrain, "u_" + k, look[k])
        for k in ("sun_dir", "sun_col", "zenith", "horizon", "fog_col",
                  "cloud_cover", "cloud_scale", "seed", "sun_vis",
                  "mtn_col", "mtn_amp", "mtn_freq"):
            self._set(self.p_sky, "u_" + k, look[k])
        self._set(self.p_resolve, "u_exposure", look["exposure"])
        self._set(self.p_resolve, "u_vignette", look["vignette"])
        self._set(self.p_resolve, "u_size", (self.W, self.H))
        self._set(self.p_resolve, "u_ss", self.ss)
        self._set(self.p_resolve, "u_noise", float(look.get("noise_sigma", 0.0)) / 255.0)
        self._frame = 0

    def release(self):
        """Free every GL object owned by this episode."""
        for o in reversed(self._res):
            try:
                o.release()
            except Exception:
                pass
        self._res = []

    # ------------------------------------------------------------------
    def render(self, cam_pos, R, intrinsics, rover_model=None):
        """Render one frame. intrinsics = (fx, fy, cx, cy) at (W, H) pixels;
        R columns = camera right / up / forward in world (y up).
        rover_model: 4x4 world transform of the rover body (shadow only)."""
        mgl, ctx = self.mgl, self.ctx
        fx, fy, cx, cy = intrinsics
        s = self.ss
        proj = projection_from_intrinsics(fx * s, fy * s, cx * s, cy * s,
                                          self.W * s, self.H * s)
        view = view_matrix(cam_pos, R)
        vp = proj @ view
        center = np.array([cam_pos[0], 0.0, cam_pos[2]])
        center[1] = cam_pos[1]
        lvp = _light_vp(center + R[:, 2] * self.shadow_half * 0.5,
                        self.look["sun_dir"], self.shadow_half, self.shadow_res)
        c2 = (math.floor(cam_pos[0] / 0.5) * 0.5, math.floor(cam_pos[2] / 0.5) * 0.5)

        # ---- shadow pass ----
        self.fbo_shadow.use()
        ctx.viewport = (0, 0, self.shadow_res, self.shadow_res)
        self.fbo_shadow.clear(depth=1.0)
        ctx.enable(mgl.DEPTH_TEST)
        ctx.disable(mgl.CULL_FACE)
        self.t_h.use(location=0)
        self.t_s.use(location=1)
        self._set(self.p_sh_terrain, "u_vp", lvp)
        self._set(self.p_sh_terrain, "u_center", c2)
        self.va_sh_terrain.render()
        self._set(self.p_sh_obj, "u_vp", lvp)
        for _, vs, n, _ in self.groups:
            vs.render(instances=n)
        if self.rover is not None and rover_model is not None:
            inst = np.zeros(self.INST_FLOATS, np.float32)
            inst[:16] = np.asarray(rover_model, np.float32).T.ravel()
            self.rover_buf.write(inst.tobytes())
            self.rover.render(instances=1)

        # ---- main pass ----
        self.fbo_main.use()
        ctx.viewport = (0, 0, self.W * s, self.H * s)
        self.fbo_main.clear(0.0, 0.0, 0.0, 0.0, depth=1.0)
        ctx.disable(mgl.DEPTH_TEST)
        self._set(self.p_sky, "u_inv_vp", np.linalg.inv(vp))
        self._set(self.p_sky, "u_cam_pos", tuple(cam_pos))
        self.va_sky.render(mgl.TRIANGLES, vertices=3)
        ctx.enable(mgl.DEPTH_TEST)
        self.t_shadow.use(location=2)
        for p in (self.p_terrain, self.p_obj):
            self._set(p, "u_vp", vp)
            self._set(p, "u_view", view)
            self._set(p, "u_light_vp", lvp)
            self._set(p, "u_cam_pos", tuple(cam_pos))
            self._set(p, "u_shadow", 2)
        self._set(self.p_terrain, "u_center", c2)
        # no back-face culling: meshes carry outward normals, so lighting
        # never depends on triangle winding (which mirrors with handedness)
        ctx.disable(mgl.CULL_FACE)
        self.va_terrain.render()
        for va, _, n, _ in self.groups:
            va.render(instances=n)

        # ---- resolve (supersample + tone map + GT depth/class) ----
        self.fbo_out.use()
        ctx.viewport = (0, 0, self.W, self.H)
        ctx.disable(mgl.DEPTH_TEST)
        self.t_hdr.use(location=3)
        self.t_lin.use(location=4)
        self.t_cls.use(location=5)
        self.p_resolve["u_hdr"].value = 3
        self.p_resolve["u_lin"].value = 4
        self.p_resolve["u_cls"].value = 5
        self._frame += 1
        self._set(self.p_resolve, "u_frame", float(self._frame % 9973))
        self.va_resolve.render(mgl.TRIANGLES, vertices=3)

        rgb = np.frombuffer(self.t_out.read(), np.uint8).reshape(self.H, self.W, 4)
        depth = np.frombuffer(self.t_dout.read(), np.float32).reshape(self.H, self.W)
        cls = np.frombuffer(self.t_cout.read(), np.uint8).reshape(self.H, self.W)
        return (np.ascontiguousarray(rgb[::-1, :, :3]),
                np.ascontiguousarray(depth[::-1]),
                np.ascontiguousarray(cls[::-1]))


def instance_rows(models, c0, c1, c2, materials, cls):
    """Pack per-instance data: models (k, 4, 4), colours (k, 3) each,
    materials (k, 3) = material id per mesh part, cls (k,) class id."""
    k = len(models)
    out = np.zeros((k, GLRenderer.INST_FLOATS), np.float32)
    out[:, :16] = np.transpose(np.asarray(models, np.float32), (0, 2, 1)).reshape(k, 16)
    out[:, 16:19] = c0
    out[:, 19:22] = c1
    out[:, 22:25] = c2
    out[:, 25:28] = materials
    out[:, 28] = cls
    return out

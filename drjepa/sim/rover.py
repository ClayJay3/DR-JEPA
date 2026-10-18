"""Episode simulation: world + rover physics + noisy sensors + camera.

Physics and sensors are carried over from the measured v12 simulator
(acceleration limits, steering lag, actuation latency, wheel slip, contact
with tangential sliding, terrain stalls and terminal tip-over, OU-process
GPS/compass biases). New here: every episode can draw its own camera rig
(resolution, FOV, mount height / pitch / roll, principal point), the image
comes from the GPU renderer, and the exact camera pose in the robot frame
is exposed for label generation.
"""

import math

import numpy as np

from ..camera import Camera, attitude
from ..config import RobotProfile, SimConfig
from ..geo import meters_to_latlon
from . import assets, world
from .style import camera_fx, sample_look

SPAWN_MODES = ("normal", "uturn", "recovery")


def random_camera(rng, sc: SimConfig):
    """Draw one camera rig from the SimConfig ranges."""
    w, h = sc.resolutions[rng.integers(len(sc.resolutions))]
    hfov = rng.uniform(*sc.hfov_range)
    cam = Camera.from_fov(w, h, hfov)
    cam.fy = cam.fx * rng.uniform(0.98, 1.02)
    cam.cx += rng.uniform(-sc.pp_jitter, sc.pp_jitter) * w
    cam.cy += rng.uniform(-sc.pp_jitter, sc.pp_jitter) * h
    # log-uniform height: low rovers are as common as tall masts
    lo, hi = sc.height_range
    cam.mount_height = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
    cam.mount_pitch = float(rng.uniform(*sc.pitch_range))
    cam.mount_roll = float(rng.uniform(*sc.roll_range) if rng.random() < 0.6 else 0.0)
    cam.mount_forward = float(rng.uniform(-0.2, 0.5))
    return cam


class RoverSim:
    """One fully randomized episode (a seed reproduces everything)."""

    def __init__(self, cfg: SimConfig = None, robot: RobotProfile = None,
                 scenario=None, spawn_mode=None, seed=None, camera: Camera = None,
                 origin=(38.4, -110.8), augment=False):
        self.cfg = cfg = cfg or SimConfig()
        self.robot = robot = robot or RobotProfile()
        self.rng = rng = np.random.default_rng(seed)
        self.origin = origin
        self.camera = camera if camera is not None else random_camera(rng, cfg)
        self.scenario = scenario or world.SCENARIOS[rng.integers(len(world.SCENARIOS))]
        if spawn_mode is None:
            r = rng.random()
            spawn_mode = "uturn" if r < 0.15 else ("recovery" if r < 0.30 else "normal")
        self.spawn_mode = spawn_mode

        self.look = sample_look(rng, augment)
        self.terrain = world.Terrain(rng, world.TerrainParams.sample(rng))
        vm = robot.v_max
        self.v_max = float(rng.uniform(*vm)) if isinstance(vm, (tuple, list)) else float(vm)
        self.factory = world.PropFactory(rng, self.terrain, self.look, robot.step_height)

        gd = rng.uniform(110, 150) if rng.random() < 0.15 else rng.uniform(40, 110)
        gb = math.radians(rng.uniform(-40, 40))
        self.goal_x, self.goal_z = self._find_drivable(gd * math.sin(gb), gd * math.cos(gb))
        self.props = world.build_world(rng, self.factory, self.scenario, self.goal_x, self.goal_z)
        self.x, self.z = self._find_drivable(0.0, 0.0)
        self.yaw = math.degrees(gb) + rng.uniform(-30, 30)
        if spawn_mode == "uturn":
            self.yaw = math.degrees(gb) + 180 + rng.uniform(-45, 45)
        self._clear_zone(self.x, self.z, 7.0)
        self._clear_zone(self.goal_x, self.goal_z, 6.0)
        if spawn_mode == "recovery":
            a = math.radians(self.yaw + rng.uniform(-15, 15))
            scale = rng.uniform(1.8, 3.0)
            d = 0.72 * scale + robot.radius + rng.uniform(0.5, 1.5)
            rock = self.factory.rock(self.x + d * math.sin(a), self.z + d * math.cos(a), scale)
            r = min(rock.colliders[0][2], d - robot.radius - 0.3) if rock.colliders else 0.0
            if r > 0:
                rock.colliders = [(rock.x, rock.z, r)]
                self.props.append(rock)
        self._collider_arrays()
        self._renderer = None
        self._prev_frame = None

        self.v = 0.0
        self.yaw_rate = 0.0
        self.cmd_queue = [(0.0, 0.0)] * robot.latency_steps
        self.bump_phase = rng.uniform(0, 2 * np.pi, 2)
        self.t = 0.0
        self.frame = 0
        self.path_len = 0.0
        self.collided_now = False
        self.collision_count = 0
        self.tipped = False
        self.terrain_stalls = 0
        self.best_goal_d = self.goal_dist_true()
        self.stagnant_frames = 0
        self.gps_bias = np.zeros(2)
        self.heading_bias = 0.0
        self._sensor_step()

    # ------------------------------------------------------------ world
    def _clear_zone(self, x, z, r):
        """Remove colliding props within r of (x, z) (spawn / goal pads)."""
        self.props = [p for p in self.props
                      if not (p.colliders and math.hypot(p.x - x, p.z - z) < r)]

    def _collider_arrays(self):
        self.obs_xz, self.obs_rad = world.colliders(self.props)

    def clearance(self, x=None, z=None):
        """Distance from the rover edge to the nearest collider surface (m)."""
        if len(self.obs_rad) == 0:
            return 99.0
        p = np.array([self.x if x is None else x, self.z if z is None else z])
        d = np.linalg.norm(self.obs_xz - p, axis=1) - self.obs_rad - self.robot.radius
        return float(d.min())

    def _hits(self, x, z):
        if len(self.obs_rad) == 0:
            return False
        d = np.linalg.norm(self.obs_xz - np.array([x, z]), axis=1)
        return bool(np.any(d < self.obs_rad + self.robot.radius))

    def _nearest_normal(self, x, z):
        p = np.array([x, z])
        d = np.linalg.norm(self.obs_xz - p, axis=1)
        i = int(np.argmin(d - self.obs_rad))
        n = p - self.obs_xz[i]
        nn = np.linalg.norm(n)
        return n / nn if nn > 1e-9 else np.array([1.0, 0.0])

    def _find_drivable(self, x, z, r_max=20.0):
        if self.terrain_margin(x, z) > 0.5:
            return x, z
        for r in np.arange(2.0, r_max, 2.0):
            for a in np.linspace(0, 2 * np.pi, 16, endpoint=False):
                cx, cz = x + r * np.sin(a), z + r * np.cos(a)
                if self.terrain_margin(cx, cz) > 0.5:
                    return float(cx), float(cz)
        return x, z

    def terrain_pose(self, x=None, z=None, yaw=None):
        """(pitch_grade, roll_grade) along / across the heading."""
        x = self.x if x is None else x
        z = self.z if z is None else z
        yaw = self.yaw if yaw is None else yaw
        hx, hz = self.terrain.slope(x, z)
        fx, fz = math.sin(math.radians(yaw)), math.cos(math.radians(yaw))
        return float(hx * fx + hz * fz), float(hx * fz - hz * fx)

    def terrain_margin(self, x=None, z=None):
        """0..1 traversability margin from terrain alone (1 = flat + firm)."""
        x = self.x if x is None else x
        z = self.z if z is None else z
        g = float(self.terrain.grade(x, z))
        rb = self.robot
        m_grade = np.clip((rb.grade_block - g) / rb.grade_block, 0.0, 1.0)
        return float(min(m_grade, 1.0 - 0.6 * float(self.terrain.sand(x, z))))

    # ------------------------------------------------------------ sensors
    def _sensor_step(self):
        cfg, dt, rng = self.cfg, self.cfg.dt, self.rng
        tau = 30.0
        self.gps_bias += (-self.gps_bias / tau) * dt + cfg.gps_walk_sigma * math.sqrt(dt) * rng.normal(size=2)
        self.heading_bias += (-self.heading_bias / tau) * dt + \
            cfg.heading_bias_sigma * math.sqrt(dt) / 3.0 * rng.normal()
        p, r = self.body_attitude()
        self.meas = {
            "x": self.x + self.gps_bias[0] + rng.normal(0, cfg.gps_noise_sigma),
            "z": self.z + self.gps_bias[1] + rng.normal(0, cfg.gps_noise_sigma),
            "heading": (self.yaw + self.heading_bias + rng.normal(0, cfg.heading_noise_sigma)) % 360.0,
            "speed": self.v * (1 + rng.normal(0, 0.03)) + rng.normal(0, 0.05),
            "pitch": p + rng.normal(0, cfg.imu_noise_deg),
            "roll": r + rng.normal(0, cfg.imu_noise_deg),
        }

    def sensor_readout(self):
        """What a real robot would report: GPS fix, compass, wheel speed,
        IMU pitch/roll, and the goal fix (the camera is self.camera)."""
        m = self.meas
        lat, lon = meters_to_latlon(m["x"], m["z"], *self.origin)
        glat, glon = meters_to_latlon(self.goal_x, self.goal_z, *self.origin)
        return {"lat": lat, "lon": lon, "goal_lat": glat, "goal_lon": glon,
                "heading": m["heading"], "speed": m["speed"],
                "pitch": m["pitch"], "roll": m["roll"]}

    def goal_dist_true(self):
        """True distance to the goal (evaluation only)."""
        return math.hypot(self.goal_x - self.x, self.goal_z - self.z)

    # ------------------------------------------------------------ dynamics
    def step(self, throttle, steer):
        """One control step (with actuation latency). Returns an info dict."""
        cfg, rb, dt, rng = self.cfg, self.robot, self.cfg.dt, self.rng
        self.cmd_queue.append((float(np.clip(throttle, -1, 1)), float(np.clip(steer, -1, 1))))
        thr, st = self.cmd_queue.pop(0)
        sand = float(self.terrain.sand(self.x, self.z))
        traction = 1.0 - rb.sand_drag * sand
        v_target = thr * self.v_max * traction
        dv = np.clip(v_target - self.v, -rb.accel_max * traction * dt, rb.accel_max * dt)
        self.v = (self.v + dv) * (1 + rng.normal(0, 0.01 + 0.02 * sand))
        alpha = dt / max(rb.steer_tau, dt)
        self.yaw_rate += (st * rb.yaw_rate_max - self.yaw_rate) * min(alpha, 1.0)
        self.yaw += (self.yaw_rate + rng.normal(0, 0.6)) * dt
        rad = math.radians(self.yaw)
        nx = self.x + math.sin(rad) * self.v * dt
        nz = self.z + math.cos(rad) * self.v * dt
        was_contact = self.collided_now
        self.collided_now = False

        pitch_g, roll_g = self.terrain_pose(nx, nz)
        if abs(roll_g) > rb.tip_roll or pitch_g < -rb.tip_roll * 1.4:
            self.tipped = True
            self.v = 0.0
        elif pitch_g > rb.grade_block and self.v > 0:
            self.collided_now = True
            if not was_contact:
                self.terrain_stalls += 1
            self.v = 0.0
            nx, nz = self.x, self.z

        if self._hits(nx, nz):
            moved = False
            if self._hits(self.x, self.z):
                moved = self.clearance(nx, nz) > self.clearance()
            else:
                n = self._nearest_normal(self.x, self.z)
                m = np.array([nx - self.x, nz - self.z])
                mt = m - np.dot(m, n) * n
                sx, sz = self.x + 0.6 * mt[0], self.z + 0.6 * mt[1]
                if np.linalg.norm(mt) > 1e-4 and not self._hits(sx, sz):
                    nx, nz, moved = sx, sz, True
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
        d = self.goal_dist_true()
        if d < self.best_goal_d - cfg.progress_eps:
            self.best_goal_d, self.stagnant_frames = d, 0
        else:
            self.stagnant_frames += 1
        return {"collided": self.collided_now, "tipped": self.tipped,
                "reached": d < cfg.goal_radius, "timeout": self.frame >= cfg.max_frames,
                "no_progress": cfg.no_progress_s > 0 and self.stagnant_frames * dt >= cfg.no_progress_s,
                "clearance": self.clearance()}

    # ------------------------------------------------------------ camera
    def body_attitude(self):
        """True body (pitch, roll) in degrees: terrain slope + ride bumps."""
        hx, hz = self.terrain.slope(self.x, self.z)
        fx, fz = math.sin(math.radians(self.yaw)), math.cos(math.radians(self.yaw))
        pitch = math.degrees(math.atan(hx * fx + hz * fz))
        roll = -math.degrees(math.atan(hx * fz - hz * fx)) * 0.8   # + = right side down
        b = self.look["bump_amp_deg"] * (0.3 + abs(self.v) / max(self.v_max, 1e-6))
        return (pitch + b * math.sin(self.bump_phase[0]),
                roll + b * 0.7 * math.sin(self.bump_phase[1]))

    def camera_pose(self):
        """(R, t) camera -> robot frame for the CURRENT frame (exact)."""
        p, r = self.body_attitude()
        return self.camera.pose(p, r)

    def _yaw_basis(self):
        """Columns: robot right / up / forward axes in world coordinates."""
        c, s = math.cos(math.radians(self.yaw)), math.sin(math.radians(self.yaw))
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

    def _ensure_renderer(self):
        if self._renderer is None:
            from .render import GLRenderer
            groups = world.render_groups(self.factory, self.props)
            self._renderer = GLRenderer(self.camera.width, self.camera.height, self.terrain,
                                        self.look, groups, rover_mesh=assets.rover_body(),
                                        ss=self.cfg.supersample)
        return self._renderer

    def render(self, gt=False):
        """Camera image (H, W, 3) uint8 RGB; with gt=True also (depth, cls)
        and the (R, t) camera pose the frame was rendered from."""
        r = self._ensure_renderer()
        R, t = self.camera_pose()
        M = self._yaw_basis()
        g = float(self.terrain.height(self.x, self.z))
        pos = np.array([self.x, g, self.z]) + M @ t
        Rw = M @ R                                   # OpenCV cam axes in world
        basis = np.stack([Rw[:, 0], -Rw[:, 1], Rw[:, 2]], 1)
        pb, rb = self.body_attitude()
        # the shadow-casting body is sized to the rig: chassis below and
        # behind the camera (a 0.4 m camera cannot sit on a 0.7 m chassis)
        h = self.camera.mount_height
        k = float(np.clip(h / 1.3, 0.35, 1.0))
        S = np.diag([k * 1.0, min(k, (h - 0.08) / 1.3), k * 1.0])
        body = np.eye(4)
        body[:3, :3] = M @ attitude(pb, rb) @ S
        body[:3, 3] = np.array([self.x, g, self.z]) + M @ attitude(pb, rb) @ np.array(
            [0.0, 0.0, self.camera.mount_forward - 0.45 * k])
        c = self.camera
        rgb, depth, cls = r.render(pos, basis, (c.fx, c.fy, c.cx, c.cy), rover_model=body)
        rgb, self._prev_frame = camera_fx(rgb, self.look, self.rng, self._prev_frame,
                                          abs(self.v) / max(self.v_max, 1e-6))
        if gt:
            return rgb, depth, cls, (R, t)
        return rgb

    def close(self):
        """Release GPU resources."""
        if self._renderer is not None:
            self._renderer.release()
            self._renderer = None

    # ------------------------------------------------------------ endless mode
    def respawn_goal(self):
        """New goal + fresh props around the current pose (endless runs)."""
        rng = self.rng
        self.scenario = world.SCENARIOS[rng.integers(len(world.SCENARIOS))]
        gd, gb = rng.uniform(50, 130), math.radians(self.yaw + rng.uniform(-60, 60))
        ox, oz = self.x, self.z
        self.goal_x, self.goal_z = self._find_drivable(ox + gd * math.sin(gb), oz + gd * math.cos(gb))
        keep = [p for p in self.props if math.hypot(p.x - ox, p.z - oz) < 25.0]
        new = world.build_world(rng, self.factory, self.scenario,
                                self.goal_x - ox, self.goal_z - oz, ox, oz)
        self.props = keep + new
        self._clear_zone(ox, oz, 8.0)
        self._clear_zone(self.goal_x, self.goal_z, 6.0)
        self._collider_arrays()
        self.close()
        self.frame = 0
        self.best_goal_d = self.goal_dist_true()
        self.stagnant_frames = 0

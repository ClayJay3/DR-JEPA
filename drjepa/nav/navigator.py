"""Navigator: the one object a robot talks to.

    from drjepa import Navigator, Camera, RobotProfile
    nav = Navigator("runs/geobev/best.pt", camera=Camera.from_fov(1280, 720, 101, mount_height=0.4),
                    robot=RobotProfile(radius=0.6, v_max=2.0))
    cmd = nav.step(rgb, {"lat": .., "lon": .., "heading": .., "speed": ..,
                         "pitch": .., "roll": .., "goal_lat": .., "goal_lon": ..})
    cmd.throttle, cmd.steer          # in [-1, 1]

Each step: pose update -> perception (image + camera + IMU attitude ->
robot-frame grids) -> belief-map fusion -> A* replan (every 3 steps) ->
arc controller. Sensors may instead carry "x"/"z" (local metres) from an
external pose estimator, and "goal_x"/"goal_z"; then GPS is not used.
"""

import collections
import math
import time
from dataclasses import dataclass, field

import numpy as np

from ..camera import Camera
from ..config import GridConfig, RobotProfile
from ..geo import latlon_to_meters
from .control import ControlConfig, Controller
from .mapping import BeliefMap, MapConfig
from .planning import Planner, PlannerConfig


@dataclass
class NavConfig:
    """Top-level navigation settings."""
    dt: float = 0.1
    gps_gain: float = 0.08           # complementary filter pull toward GPS
    replan_every: int = 3
    map: MapConfig = field(default_factory=MapConfig)
    planner: PlannerConfig = field(default_factory=PlannerConfig)
    control: ControlConfig = field(default_factory=ControlConfig)


@dataclass
class Command:
    throttle: float
    steer: float
    info: dict


class PoseFilter:
    """Wheel-speed x compass dead reckoning, pulled gently toward GPS."""

    def __init__(self, dt, gain):
        self.dt, self.gain = dt, gain
        self.reset()

    def reset(self):
        self.origin = None
        self.pose = None

    def update(self, s):
        """Returns (pose (x, z) metres, heading deg, goal (x, z) metres)."""
        if "x" in s:                                   # external pose estimator
            self.pose = np.array([s["x"], s["z"]], float)
            goal = np.array([s["goal_x"], s["goal_z"]], float)
            return self.pose, s["heading"], goal
        if self.origin is None:
            self.origin = (s["lat"], s["lon"])
        gx, gz = latlon_to_meters(s["lat"], s["lon"], *self.origin)
        h = math.radians(s["heading"])
        if self.pose is None:
            self.pose = np.array([gx, gz], float)
        else:
            self.pose += s["speed"] * self.dt * np.array([math.sin(h), math.cos(h)])
            self.pose += self.gain * (np.array([gx, gz]) - self.pose)
        goal = np.array(latlon_to_meters(s["goal_lat"], s["goal_lon"], *self.origin), float)
        return self.pose, s["heading"], goal


class Navigator:
    """Perceive -> remember -> plan -> act (see module docstring)."""

    def __init__(self, perception, camera: Camera = None, robot: RobotProfile = None,
                 cfg: NavConfig = None, grid: GridConfig = None):
        """`perception` is a checkpoint path, a Perceiver, or any callable
        (rgb, camera, body_pitch=, body_roll=) -> grids dict."""
        if isinstance(perception, str):
            from ..perception.infer import Perceiver
            perception = Perceiver(perception)
        self.perception = perception
        self.camera = camera
        self.robot = robot or RobotProfile()
        self.cfg = cfg or NavConfig()
        self.grid = grid or GridConfig()
        self.map = BeliefMap(self.cfg.map, self.grid)
        self.planner = Planner(self.robot, self.cfg.planner)
        self.controller = Controller(self.robot, self.cfg.control)
        self.pose_filter = PoseFilter(self.cfg.dt, self.cfg.gps_gain)
        self.timing = collections.defaultdict(float)
        self.reset()

    def reset(self):
        """New run = new memory."""
        self.map.reset()
        self.planner.reset()
        self.controller.reset()
        self.pose_filter.reset()
        self.step_i = 0
        self.last_timing = {}
        self.grids = None
        self.pose = None
        self.goal = None
        self.heading = 0.0

    def step(self, rgb, sensors, camera: Camera = None, R=None, t=None):
        """One control step. `rgb` is (H, W, 3) uint8 RGB. Optionally pass
        the exact camera pose (R, t) in the robot frame (else it is the
        camera mount composed with the IMU pitch / roll in `sensors`)."""
        cam = camera or self.camera
        t0 = time.perf_counter()
        pose, heading, goal = self.pose_filter.update(sensors)
        if self.map.corner is None:
            self.map.init_at(pose)
        self.map.maybe_recenter(pose)
        t1 = time.perf_counter()
        g = self.perception(rgb, cam, R=R, t=t, body_pitch=sensors.get("pitch", 0.0),
                            body_roll=sensors.get("roll", 0.0))
        t2 = time.perf_counter()
        new_pose = self.map.fuse(g, pose, heading, recovering=self.controller.recovering)
        self.pose_filter.pose[:] = new_pose
        pose = new_pose
        t3 = time.perf_counter()
        if self.step_i % self.cfg.replan_every == 0 or self.planner.path is None:
            self.planner.replan(self.map, pose, goal)
        t4 = time.perf_counter()
        thr, st = self.controller.step(self.map, self.planner, pose, heading, goal, sensors["speed"])
        t5 = time.perf_counter()
        self.last_timing = {"pose": t1 - t0, "perception": t2 - t1, "fusion": t3 - t2,
                            "planning": t4 - t3, "control": t5 - t4}
        for k, v in self.last_timing.items():
            self.timing[k] += v
        self.timing["steps"] += 1
        self.step_i += 1
        self.grids, self.pose, self.goal, self.heading = g, pose, goal, heading
        return Command(thr, st, {"mode": self.controller.mode, "path": self.planner.path,
                                 "target": self.controller.last_target})

    def timing_ms(self):
        """Mean per-step latency by stage (ms)."""
        n = max(self.timing["steps"], 1)
        return {k: 1000 * v / n for k, v in self.timing.items() if k != "steps"}

"""Every tunable in one place, grouped by concern.

RobotProfile   the vehicle: footprint, dynamics, and what terrain it can
               handle. Physics, the expert, labels and the planner all read
               it, so one profile describes one robot everywhere.
GridConfig     the perception grid (robot-frame ground cells ahead).
SimConfig      episode logic, sensor noise, and camera randomization.
"""

from dataclasses import asdict, dataclass, field


@dataclass
class RobotProfile:
    """Physical robot description (defaults = the simulated URC rover)."""
    radius: float = 0.75              # collision footprint (m)
    v_max: tuple = (4.0, 8.0)         # top speed (m/s); a range is drawn per
    #                                   sim episode, a real robot uses one value
    accel_max: float = 3.0            # m/s^2
    yaw_rate_max: float = 70.0        # deg/s at full steering
    steer_tau: float = 0.25           # steering first-order lag (s)
    latency_steps: int = 2            # actuation latency (control steps)
    step_height: float = 0.25         # rigid obstacles shorter than this are
    #                                   driven over (wheel radius ~ curb height)
    grade_drive: float = 0.35         # comfortable grade (rise/run)
    grade_block: float = 0.55         # climbing steeper than this stalls
    tip_roll: float = 0.50            # lateral grade that tips the robot
    sand_drag: float = 0.55           # fraction of speed lost in deep sand

    @property
    def v_top(self):
        """Single representative top speed (upper end of the range)."""
        return self.v_max[1] if isinstance(self.v_max, (tuple, list)) else float(self.v_max)


@dataclass
class GridConfig:
    """Robot-frame perception grid: `cells` x `cells` at `res` metres,
    covering x in [-cells*res/2, +cells*res/2] (right) and z in
    [0, cells*res] (forward). Indexed [i = x, j = z]."""
    cells: int = 48
    res: float = 0.5

    @property
    def half(self):
        """Half-width of the grid (m)."""
        return self.cells * self.res / 2.0

    @property
    def depth(self):
        """Forward reach of the grid (m)."""
        return self.cells * self.res


@dataclass
class SimConfig:
    """Simulator episode, sensor and camera-randomization parameters."""
    dt: float = 0.1                   # control period (10 Hz)
    goal_radius: float = 4.0
    max_frames: int = 900
    no_progress_s: float = 0.0        # 0 = off (data gen); eval sets 60 s
    progress_eps: float = 0.25

    # sensor noise
    gps_walk_sigma: float = 0.35
    gps_noise_sigma: float = 0.25
    heading_bias_sigma: float = 2.5
    heading_noise_sigma: float = 1.0
    imu_noise_deg: float = 0.3        # attitude noise on reported pitch/roll

    # camera: drawn per episode unless a fixed Camera is passed to RoverSim
    resolutions: tuple = ((640, 480), (640, 360), (512, 512), (640, 400),
                          (576, 432), (720, 405))
    hfov_range: tuple = (50.0, 125.0)
    height_range: tuple = (0.3, 2.2)
    pitch_range: tuple = (-28.0, 6.0)
    roll_range: tuple = (-6.0, 6.0)
    pp_jitter: float = 0.03           # principal point offset (fraction)
    supersample: int = 2


@dataclass
class Config:
    """Bundle stored inside checkpoints so inference rebuilds what trained."""
    robot: RobotProfile = field(default_factory=RobotProfile)
    grid: GridConfig = field(default_factory=GridConfig)
    sim: SimConfig = field(default_factory=SimConfig)

    def to_dict(self):
        """Plain dict (JSON-safe)."""
        return asdict(self)

    @staticmethod
    def from_dict(d):
        """Inverse of to_dict (tuples restored from lists)."""
        def tup(dc, cls):
            kw = {}
            for k, v in (d.get(dc) or {}).items():
                if k not in cls.__dataclass_fields__:
                    continue
                if isinstance(v, list):
                    v = tuple(tuple(x) if isinstance(x, list) else x for x in v)
                kw[k] = v
            return cls(**kw)
        return Config(robot=tup("robot", RobotProfile), grid=tup("grid", GridConfig),
                      sim=tup("sim", SimConfig))

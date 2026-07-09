"""Single source of truth for every tunable in the project."""

from dataclasses import dataclass, field, asdict


@dataclass
class SimConfig:
    """Simulation / rendering parameters (data generation + live testing)."""
    img_w: int = 448                 # render resolution; keep >= ModelConfig.img_size
    img_h: int = 448                 # or the extra model resolution buys nothing
    fov_deg: float = 90.0
    dt: float = 0.1                  # control period (10 Hz)
    cam_height: float = 1.1          # camera height above local terrain (m)
    rover_radius: float = 0.75       # collision footprint (m)

    # Vehicle dynamics
    v_max: float = 6.0               # nominal top speed (m/s), randomized per episode
    accel_max: float = 3.0           # m/s^2
    yaw_rate_max: float = 70.0       # deg/s at full steering
    steer_tau: float = 0.25          # steering first-order lag (s)
    latency_steps: int = 2           # actuation latency in control steps (200 ms)

    # Sensor noise
    gps_walk_sigma: float = 0.35     # random-walk (m per sqrt(s))
    gps_noise_sigma: float = 0.25    # white noise (m)
    heading_bias_sigma: float = 2.5  # slowly-varying compass bias (deg)
    heading_noise_sigma: float = 1.0 # white compass noise (deg)

    # Episode logic
    goal_radius: float = 4.0         # success distance (m)
    max_frames: int = 900            # 90 s cap per episode


@dataclass
class ModelConfig:
    """RoverJEPA architecture."""
    img_size: int = 448              # DINOv2 input resolution (must be a
    #                                  multiple of 14; raise/lower to trade
    #                                  perception sharpness vs speed; frames
    #                                  are resized to this before the backbone)
    backbone: str = "dinov2_vits14"  # frozen; features are precomputed
    feat_dim: int = 384              # DINOv2 ViT-S embedding width
    pool_rows: int = 12              # patch-token pooling grid (vertical)
    pool_cols: int = 12              # patch-token pooling grid (horizontal)
    n_tokens: int = 145              # pool_rows*pool_cols + CLS

    # Metric mapping (the persistent spatial memory)
    wedge_cells: int = 48            # per-frame occupancy wedge (48x48)
    wedge_res: float = 0.5           # metres per cell (24m x 24m ahead)
    wedge_range_cells: int = 32      # supervise/trust only the near 16 m
    map_cells: int = 512             # persistent world map (256m x 256m)
    # multi-frame perception: the wedge decoder sees the current frame plus
    # these lookbacks (control steps), giving it motion parallax
    frame_offsets: tuple = (0, 2, 4)

    embed_dim: int = 256             # frame embedding / belief state width
    seq_len: int = 12                # temporal context (1.2 s at 10 Hz)
    n_layers: int = 3                # causal transformer depth
    n_heads: int = 4
    dropout: float = 0.1

    action_horizon: int = 8          # predicted action chunk length
    steer_bins: int = 15             # steering is classified (multimodal), not regressed
    jepa_offsets: tuple = (1, 4, 8)  # future frames the world model predicts
    ctx_dim: int = 4                 # [dist, sin(bearing), cos(bearing), speed]

    # Context normalization
    dist_norm: float = 100.0         # metres mapped to 1.0 (capped)
    speed_norm: float = 10.0         # m/s mapped to 1.0


@dataclass
class TrainConfig:
    epochs: int = 60
    batch_size: int = 128
    lr: float = 3e-4
    min_lr: float = 1e-5
    warmup_epochs: int = 3
    weight_decay: float = 0.05
    ema_decay: float = 0.995         # JEPA target-encoder EMA
    patience: int = 12
    val_split: float = 0.15          # fraction of episodes held out
    window_stride: int = 6           # frames between training windows
    num_workers: int = 8

    # Loss weights
    w_action: float = 1.0
    w_safety: float = 0.5
    w_jepa: float = 0.5
    w_map: float = 2.0               # occupancy-wedge prediction (primary)
    occ_pos_weight: float = 1.5      # mild: the fusion prior handles the
    #                                  base rate; large values fatten the
    #                                  false-positive tail that pollutes maps
    w_progress: float = 0.5          # goal-progress regression on futures
    w_reg: float = 0.1               # VICReg variance+covariance anti-collapse
    w_jerk: float = 0.3              # action-chunk smoothness penalty
    dagger_weight: float = 0.5       # sampling weight for DAgger episodes


@dataclass
class Config:
    sim: SimConfig = field(default_factory=SimConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def to_dict(self):
        return asdict(self)

    @staticmethod
    def from_dict(d):
        m = dict(d.get("model", {}))
        for key, default in (("jepa_offsets", (1, 4, 8)),
                             ("frame_offsets", (0, 2, 4))):
            m[key] = tuple(m.get(key, default))
        return Config(
            sim=SimConfig(**d.get("sim", {})),
            model=ModelConfig(**m),
            train=TrainConfig(**d.get("train", {})),
        )


DEFAULT = Config()

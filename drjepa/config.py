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

    # Terrain hazard limits (shared by physics, expert costs, and GT labels)
    grade_drive: float = 0.35        # comfortable grade (rise/run)
    grade_block: float = 0.55        # climbing steeper than this stalls out
    tip_roll: float = 0.50           # lateral grade that tips the rover over
    sand_drag: float = 0.55          # fraction of speed lost in deep sand

    # Episode logic
    goal_radius: float = 4.0         # success distance (m)
    max_frames: int = 900            # hard backstop per episode (frames)
    no_progress_s: float = 0.0       # give up after this many seconds without
    #                                  getting closer to the goal than ever
    #                                  before. 0 = disabled (data generation
    #                                  keeps the plain max_frames cap). Eval
    #                                  uses it so a SLOW-but-closing rover is
    #                                  not scored the same as a stuck one.
    progress_eps: float = 0.25       # metres of improvement that count as
    #                                  progress (ignores GPS/pose jitter)


@dataclass
class ModelConfig:
    """RoverJEPA architecture."""
    img_size: int = 448              # backbone input resolution; must be a
    #                                  multiple of the patch size (14 for
    #                                  DINOv2, 16 for DINOv3 -- 448 works for
    #                                  both). Frames are resized to this.
    backbone: str = "dinov3_vits16"  # frozen; features are precomputed.
    #                                  dinov2_* loads from torch.hub (open);
    #                                  dinov3_* loads via HF transformers.
    backbone_weights: str = "facebook/dinov3-vits16-pretrain-lvd1689m"
    #                                  HF model id (or local dir) for DINOv3;
    #                                  weights are license-gated, so accept
    #                                  the licence + `huggingface-cli login`
    #                                  once and it downloads automatically
    #                                  (or override with DINOV3_WEIGHTS env)
    feat_dim: int = 384              # ViT-S embedding width (same v2 and v3)
    pool_rows: int = 24              # patch-token pooling grid (vertical) --
    pool_cols: int = 24              # 24x24 (was 12) for finer small-object
    #                                  detail; DINOv3@448 has a 28x28 grid to
    #                                  pool from, wedge (48) is a clean 2x up
    n_tokens: int = 577              # pool_rows*pool_cols + CLS

    # Metric mapping (the persistent spatial memory)
    wedge_cells: int = 48            # per-frame occupancy wedge (48x48)
    wedge_res: float = 0.5           # metres per cell (24m x 24m ahead)
    wedge_range_cells: int = 32      # supervise/trust only the near 16 m
    map_cells: int = 512             # persistent world map (256m x 256m)
    # multi-frame perception: the wedge decoder sees the current frame plus
    # these lookbacks (control steps), giving it motion parallax
    frame_offsets: tuple = (0, 2, 4)
    # map-space JEPA (completion of unobserved map areas)
    comp_cells: int = 80             # completion crop (80 m x 80 m @ 1 m)
    comp_res: float = 1.0

    # Training-window geometry (also used by the dataset loader). The
    # temporal/embedding JEPA branch was removed after v11; seq_len and
    # jepa_offsets survive only to size dataset windows (W = seq_len +
    # max(jepa_offsets)), and action_horizon/ctx_dim to shape logged
    # label/context arrays the loss no longer consumes.
    seq_len: int = 12
    jepa_offsets: tuple = (1, 4, 8)
    action_horizon: int = 8
    ctx_dim: int = 4                 # [dist, sin(bearing), cos(bearing), speed]

    # Context normalization
    dist_norm: float = 100.0         # metres mapped to 1.0 (capped)
    speed_norm: float = 10.0         # m/s mapped to 1.0


@dataclass
class TrainConfig:
    """Optimization schedule, dataset split, and loss weights."""
    epochs: int = 60
    batch_size: int = 64             # halved for the 24x24 token grid: 4x the
    #                                  tokens = 4x decoder activations, and
    #                                  128 no longer fits 16 GB VRAM alongside
    #                                  the desktop (peaks 12.8 GB vs 6.4 here)
    lr: float = 3e-4
    min_lr: float = 1e-5
    warmup_epochs: int = 3
    weight_decay: float = 0.05
    patience: int = 12
    val_split: float = 0.15          # fraction of episodes held out
    window_stride: int = 6           # frames between training windows
    num_workers: int = 8

    # Loss weights
    w_safety: float = 0.5            # danger head ("trouble within ~1 s")
    w_map: float = 2.0               # occupancy-wedge prediction (primary)
    w_elev: float = 1.0              # elevation regression (terrain wedge)
    w_sand: float = 0.5              # soft-ground classification
    w_haz: float = 1.0               # steep-ground classification (tip risk)
    w_complete: float = 1.0          # map-space JEPA (hidden-map prediction)
    occ_pos_weight: float = 1.5      # mild: the fusion prior handles the
    #                                  base rate; large values fatten the
    #                                  false-positive tail that pollutes maps
    dagger_weight: float = 0.5       # sampling weight for DAgger episodes
    real_weight: float = -1.0        # sampling weight for real (phone-
    #                                  captured) episodes; < 0 = auto-balance
    #                                  real up to REAL_TARGET_SHARE of the
    #                                  sampled signal (capped, see dataset.py),
    #                                  >= 0 = use this fixed weight


@dataclass
class Config:
    """Bundle of all three config groups.

    A snapshot of this (via `to_dict`) is stored inside every checkpoint so
    inference always reconstructs the exact architecture it was trained
    with, regardless of later edits to the defaults in this file.
    """
    sim: SimConfig = field(default_factory=SimConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def to_dict(self):
        """Plain-dict form for embedding in checkpoints (JSON-safe)."""
        return asdict(self)

    @staticmethod
    def from_dict(d):
        """Rebuild a Config from a checkpoint snapshot (inverse of to_dict).

        Tuple-valued fields are restored explicitly because asdict/torch
        serialization round-trips them as lists.
        """
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

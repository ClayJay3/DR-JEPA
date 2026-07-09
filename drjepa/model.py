"""RoverJEPA: an action-conditioned JEPA world model for goal-driven driving.

Design (all sizes from ModelConfig):

  frozen DINOv2 (precomputed)      7 tokens x 384 per frame
        |
  FrameEncoder  ------------------ 256-d frame embedding e_t
        |            \
  causal Transformer  EMA copy --- target embedding  ~e_t   (no grad)
        |
  belief state s_t (256)
        |-- PolicyHead(s_t, goal ctx)         -> action chunk (H x 2, tanh)
        |-- DangerHead(s_t)                   -> imminent-danger logit
        |-- JEPAPredictor(s_t, actions, k)    -> predicted e_{t+k}
                 |-- FutureDangerHead(e)      -> danger logit of a (predicted)
                                                 future embedding

The JEPA objective: the predictor must regress the EMA target embedding of
the frame k steps ahead, conditioned on the actions actually executed in
between. This forces the belief state to carry controllable scene dynamics
(an action-conditioned world model) rather than static appearance.
VICReg-style variance/covariance regularization prevents collapse.

At inference the predictor doubles as a one-step "imagination" engine: the
latent safety shield rolls candidate action chunks through it and vetoes
those whose predicted future embedding decodes to high danger.

Total trainable parameters: ~4M. The DINOv2 backbone stays frozen, which is
what preserves sim-to-real robustness of the visual features.
"""

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig


class FrameEncoder(nn.Module):
    """(B, T, n_tokens, feat_dim) DINOv2 tokens -> (B, T, embed_dim).

    The stored token grid is fine (for metric mapping); the frame embedding
    only needs coarse layout, so grid tokens are average-pooled to 4x4
    before projection.
    """

    POOL = 4

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.rows, self.cols = cfg.pool_rows, cfg.pool_cols
        self.norm = nn.LayerNorm(cfg.feat_dim)
        self.token_proj = nn.Linear(cfg.feat_dim, 128)
        n_eff = self.POOL * self.POOL + 1
        self.mlp = nn.Sequential(
            nn.Linear(n_eff * 128, 512),
            nn.GELU(),
            nn.Linear(512, cfg.embed_dim),
            nn.LayerNorm(cfg.embed_dim),
        )

    def forward(self, tokens):
        lead = tokens.shape[:-2]
        t = tokens.reshape(-1, tokens.shape[-2], tokens.shape[-1])
        cls, grid = t[:, :1], t[:, 1:]
        g = grid.view(-1, self.rows, self.cols, grid.shape[-1]).permute(0, 3, 1, 2)
        g = F.adaptive_avg_pool2d(g, (self.POOL, self.POOL))
        g = g.flatten(2).permute(0, 2, 1)
        x = self.token_proj(self.norm(torch.cat([cls, g], dim=1)))
        out = self.mlp(x.flatten(-2))
        return out.reshape(*lead, -1)


class MapDecoder(nn.Module):
    """Multi-frame metric perception: DINOv2 tokens -> occupancy wedge.

    Predicts, for the 24m x 24m area in front of the camera (48x48 cells),
    a per-cell occupancy logit and a per-cell visibility/confidence logit.
    The decoder sees the current frame plus cfg.frame_offsets lookbacks
    (channel-stacked per grid position) together with the ego-motion of
    those frames (speed + commanded steer), so it can exploit motion
    parallax -- the strongest monocular depth cue -- instead of texture
    scale alone. Outputs are fused into the persistent world map.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.rows, self.cols = cfg.pool_rows, cfg.pool_cols
        self.cells = cfg.wedge_cells
        self.n_frames = len(cfg.frame_offsets)
        assert self.cells % self.rows == 0, "wedge must be a multiple of grid"
        n_up = int(math.log2(self.cells // self.rows))
        self.cls_proj = nn.Linear(cfg.feat_dim + 2 * self.n_frames, 64)
        self.token_proj = nn.Linear(cfg.feat_dim * self.n_frames + 64, 256)
        ch = [256, 128, 64, 32]
        layers = [nn.Conv2d(256, 256, 3, 1, 1), nn.GELU()]
        for i in range(n_up):                    # e.g. 12x12 -> 48x48
            layers += [nn.ConvTranspose2d(ch[i], ch[i + 1], 4, 2, 1), nn.GELU(),
                       nn.Conv2d(ch[i + 1], ch[i + 1], 3, 1, 1), nn.GELU()]
        self.net = nn.Sequential(*layers, nn.Conv2d(ch[n_up], 2, 3, 1, 1))

    def forward(self, tokens, motion):
        """tokens (..., F, n_tokens, feat_dim) stacked [current, -2, -4, ...];
        motion (..., F*2) = (speed, steer) per stacked frame.
        Returns occ, conf logits (..., C, C) indexed [i = x_right,
        j = z_forward] (rover frame)."""
        lead = tokens.shape[:-3]
        F_, nt, fd = tokens.shape[-3:]
        t = tokens.reshape(-1, F_, nt, fd)
        m = motion.reshape(-1, motion.shape[-1])
        cls = t[:, 0, 0]                                   # current-frame CLS
        grid = t[:, :, 1:].permute(0, 2, 1, 3).reshape(-1, nt - 1, F_ * fd)
        c = self.cls_proj(torch.cat([cls, m], dim=-1))
        x = self.token_proj(torch.cat([grid, c[:, None].expand(-1, nt - 1, -1)],
                                      dim=-1))
        x = x.view(-1, self.rows, self.cols, 256).permute(0, 3, 1, 2)
        out = self.net(x)                        # (N, 2, cells, cells)
        # conv output is image-aligned (row ~ image-y ~ distance-from-far,
        # col ~ image-x ~ lateral); re-index to the rover-frame (i, j)
        # convention so convs only learn a local perspective warp, never a
        # global transpose
        out = out.flip(-2).transpose(-1, -2)
        occ, conf = out[:, 0], out[:, 1]
        return occ.reshape(*lead, self.cells, self.cells), \
            conf.reshape(*lead, self.cells, self.cells)


class TemporalEncoder(nn.Module):
    """Causal transformer over frame embeddings -> belief states."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.pos = nn.Parameter(torch.zeros(1, cfg.seq_len, cfg.embed_dim))
        nn.init.trunc_normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.embed_dim, nhead=cfg.n_heads,
            dim_feedforward=cfg.embed_dim * 4, dropout=cfg.dropout,
            batch_first=True, norm_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, num_layers=cfg.n_layers)
        self.out_norm = nn.LayerNorm(cfg.embed_dim)

    def forward(self, e):
        S = e.shape[1]
        x = e + self.pos[:, :S]
        mask = nn.Transformer.generate_square_subsequent_mask(S, device=e.device)
        return self.out_norm(self.encoder(x, mask=mask, is_causal=True))


class JEPAPredictor(nn.Module):
    """(s_t, executed actions t..t+k-1, offset k) -> predicted e_{t+k}."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.k_max = max(cfg.jepa_offsets)
        self.offset_emb = nn.Embedding(len(cfg.jepa_offsets), 32)
        self.net = nn.Sequential(
            nn.Linear(cfg.embed_dim + self.k_max * 2 + 32, 512),
            nn.GELU(),
            nn.Linear(512, 512),
            nn.GELU(),
            nn.Linear(512, cfg.embed_dim),
        )

    def forward(self, s, actions_flat, offset_idx):
        """s: (N, E); actions_flat: (N, k_max*2) zero-padded; offset_idx: int."""
        k = self.offset_emb.weight[offset_idx].expand(s.shape[0], -1)
        return self.net(torch.cat([s, actions_flat, k], dim=-1))


class RoverJEPA(nn.Module):
    def __init__(self, cfg: ModelConfig = None):
        super().__init__()
        self.cfg = cfg = cfg or ModelConfig()

        self.frame_encoder = FrameEncoder(cfg)
        self.target_encoder = copy.deepcopy(self.frame_encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad = False

        self.temporal = TemporalEncoder(cfg)
        self.predictor = JEPAPredictor(cfg)
        self.map_decoder = MapDecoder(cfg)

        # steering is CLASSIFIED over bins: obstacle dodging is multimodal
        # (left or right both valid) and regression averages the modes into
        # "drive straight at the rock". throttle stays a regression.
        self.policy = nn.Sequential(
            nn.Linear(cfg.embed_dim + cfg.ctx_dim, 512),
            nn.GELU(),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, cfg.action_horizon * (cfg.steer_bins + 1)),
        )
        self.register_buffer(
            "bin_centers", torch.linspace(-1.0, 1.0, cfg.steer_bins),
            persistent=False)
        self.danger_head = nn.Sequential(
            nn.Linear(cfg.embed_dim, 128), nn.GELU(), nn.Linear(128, 1))
        self.future_danger_head = nn.Sequential(
            nn.Linear(cfg.embed_dim, 128), nn.GELU(), nn.Linear(128, 1))
        # regresses 10 * (goal_dist_{t+k} - goal_dist_t) (normalized units)
        # from the embedding delta plus goal context, so imagined futures can
        # be scored on goal progress (visual motion alone can't know the
        # goal direction -- that comes from ctx)
        self.progress_head = nn.Sequential(
            nn.Linear(cfg.embed_dim + cfg.ctx_dim, 128), nn.GELU(),
            nn.Linear(128, 1))

    # ------------------------------------------------------------------
    @torch.no_grad()
    def update_ema(self, decay):
        for pt, po in zip(self.target_encoder.parameters(),
                          self.frame_encoder.parameters()):
            pt.mul_(decay).add_(po, alpha=1 - decay)

    def embed(self, tokens):
        return self.frame_encoder(tokens)

    def belief(self, e_seq):
        return self.temporal(e_seq)

    def policy_raw(self, s, ctx):
        """(steer logits (N, H, K), throttle (N, H))."""
        cfg = self.cfg
        out = self.policy(torch.cat([s, ctx], dim=-1))
        out = out.view(-1, cfg.action_horizon, cfg.steer_bins + 1)
        return out[..., :cfg.steer_bins], torch.tanh(out[..., cfg.steer_bins])

    def act(self, s, ctx):
        """Belief + goal context -> decoded action chunk (N, H, 2).

        Steering decodes as the argmax bin (the mode), which commits to one
        side of an obstacle instead of averaging the options.
        """
        logits, thr = self.policy_raw(s, ctx)
        steer = self.bin_centers[logits.argmax(dim=-1)]
        return torch.stack([thr, steer], dim=-1)

    def danger(self, s):
        return self.danger_head(s)

    def predict_future(self, s, exec_actions, offset_idx):
        return self.predictor(s, exec_actions, offset_idx)

    # ------------------------------------------------------------------
    def compute_losses(self, tokens, label_actions, exec_actions, ctx, danger,
                       dist, occ_gt, vis_gt, motion, train_cfg):
        """All training losses for one batch.

        tokens:        (B, W, n_tokens, feat_dim)  W = seq_len + k_max
        label_actions: (B, W, 2)   clean expert labels
        exec_actions:  (B, W, 2)   executed (noise-injected) actions
        ctx:           (B, S, ctx_dim)
        danger:        (B, W, 1) in [0, 1]
        dist:          (B, W, 1) normalized goal distance per frame
        occ_gt/vis_gt: (B, W, cells, cells) wedge occupancy / visibility
        motion:        (B, W, 2)   (speed, executed steer) per frame
        """
        cfg = self.cfg
        S = cfg.seq_len
        H = cfg.action_horizon
        k_max = max(cfg.jepa_offsets)
        B, W = tokens.shape[:2]

        e_all = self.frame_encoder(tokens)                    # (B, W, E)
        with torch.no_grad():
            e_tgt = self.target_encoder(tokens)               # (B, W, E)
        s = self.temporal(e_all[:, :S])                       # (B, S, E)
        s_flat = s.reshape(B * S, -1)

        # ---------------- policy (behavior cloning) ----------------
        logits, thr = self.policy_raw(s_flat, ctx.reshape(B * S, -1))
        tgt_chunks = label_actions.unfold(1, H, 1)            # (B, W-H+1, 2, H)
        tgt_chunks = tgt_chunks[:, :S].permute(0, 1, 3, 2).reshape(B * S, H, 2)
        tgt_steer = tgt_chunks[..., 1]
        K = cfg.steer_bins
        tgt_bin = ((tgt_steer + 1.0) / 2.0 * (K - 1)).round().long().clamp(0, K - 1)
        w_steer = 1.0 + 2.0 * tgt_steer.abs()
        ce = F.cross_entropy(logits.reshape(-1, K), tgt_bin.reshape(-1),
                             reduction="none", label_smoothing=0.05)
        l_str = (w_steer.reshape(-1) * ce).mean()
        l_thr = F.smooth_l1_loss(thr, tgt_chunks[..., 0])
        l_jerk = ((thr[:, 1:] - thr[:, :-1]) ** 2).mean()
        loss_action = l_thr + l_str + train_cfg.w_jerk * l_jerk

        # ---------------- danger heads ----------------
        d_logit = self.danger_head(s_flat)
        loss_safety = F.binary_cross_entropy_with_logits(
            d_logit, danger[:, :S].reshape(B * S, 1))

        # ---------------- metric perception (occupancy wedge) ----------------
        # stack the multi-frame input with strided slices; supervise every
        # second frame past the lookback warm-up
        offs = cfg.frame_offsets
        m_off = max(offs)
        tok_stack = torch.stack(
            [tokens[:, m_off - o:W - o:2] for o in offs], dim=2)
        mot_stack = torch.cat(
            [motion[:, m_off - o:W - o:2] for o in offs], dim=-1)
        occ_logit, conf_logit = self.map_decoder(tok_stack, mot_stack)
        occ_gt = occ_gt[:, m_off::2]
        vis_gt = vis_gt[:, m_off::2]
        C = occ_logit.shape[-1]
        # nearer rows matter more for driving; beyond wedge_range_cells the
        # monocular distance ambiguity is too large to supervise usefully
        row_w = torch.linspace(1.6, 0.7, C, device=occ_logit.device)
        row_w[cfg.wedge_range_cells:] = 0.0
        row_w = row_w[None, None, None, :]
        w = vis_gt * row_w
        loss_occ = F.binary_cross_entropy_with_logits(
            occ_logit, occ_gt, weight=w,
            pos_weight=torch.tensor(train_cfg.occ_pos_weight,
                                    device=occ_logit.device))
        loss_occ = loss_occ * (w.numel() / w.sum().clamp(min=1.0))
        loss_conf = F.binary_cross_entropy_with_logits(conf_logit, vis_gt)
        loss_map = loss_occ + 0.25 * loss_conf
        with torch.no_grad():
            R = cfg.wedge_range_cells
            pred = (occ_logit[..., :R] > 0) & (vis_gt[..., :R] > 0.5)
            gt = (occ_gt[..., :R] > 0.5) & (vis_gt[..., :R] > 0.5)
            inter = (pred & gt).sum()
            union = (pred | gt).sum().clamp(min=1)
            occ_iou = float(inter) / float(union)

        # ---------------- JEPA world model ----------------
        # exec action windows: (B, W-k_max+1, 2, k_max) -> (B, S, k_max*2)
        act_windows = exec_actions.unfold(1, k_max, 1)[:, :S]
        act_windows = act_windows.permute(0, 1, 3, 2).reshape(B, S, k_max * 2)

        loss_jepa = 0.0
        loss_fdanger = 0.0
        loss_prog = 0.0
        n_prog = 0
        for i, k in enumerate(cfg.jepa_offsets):
            acts = act_windows.clone()
            acts[:, :, 2 * k:] = 0.0                          # mask beyond k
            e_hat = self.predictor(s_flat, acts.reshape(B * S, -1), i)
            tgt = e_tgt[:, k:S + k].reshape(B * S, -1)
            loss_jepa = loss_jepa + F.smooth_l1_loss(e_hat, tgt)

            d_fut = danger[:, k:S + k].reshape(B * S, 1)
            loss_fdanger = loss_fdanger + 0.5 * (
                F.binary_cross_entropy_with_logits(
                    self.future_danger_head(e_all[:, k:S + k].reshape(B * S, -1)),
                    d_fut) +
                F.binary_cross_entropy_with_logits(
                    self.future_danger_head(e_hat.detach()), d_fut))

            if k >= 4:  # short offsets are dominated by GPS noise
                prog_tgt = 10.0 * (dist[:, k:S + k] - dist[:, :S]).reshape(B * S, 1)
                e_now = e_all[:, :S].reshape(B * S, -1).detach()
                ctx_flat = ctx.reshape(B * S, -1)
                loss_prog = loss_prog + 0.5 * (
                    F.smooth_l1_loss(self.progress_head(torch.cat(
                        [e_all[:, k:S + k].reshape(B * S, -1) - e_now,
                         ctx_flat], dim=-1)), prog_tgt) +
                    F.smooth_l1_loss(self.progress_head(torch.cat(
                        [e_hat.detach() - e_now, ctx_flat], dim=-1)), prog_tgt))
                n_prog += 1
        loss_jepa = loss_jepa / len(cfg.jepa_offsets)
        loss_fdanger = loss_fdanger / len(cfg.jepa_offsets)
        loss_prog = loss_prog / max(1, n_prog)

        # ---------------- anti-collapse regularization ----------------
        e_flat = e_all[:, :S].reshape(B * S, -1)
        e_c = e_flat - e_flat.mean(dim=0)
        std = torch.sqrt(e_c.var(dim=0) + 1e-4)
        loss_var = F.relu(1.0 - std).mean()
        cov = (e_c.T @ e_c) / (e_c.shape[0] - 1)
        D = cov.shape[0]
        loss_cov = (cov.pow(2).sum() - cov.diagonal().pow(2).sum()) / D

        total = (train_cfg.w_action * loss_action
                 + train_cfg.w_safety * (loss_safety + loss_fdanger)
                 + train_cfg.w_jepa * loss_jepa
                 + train_cfg.w_map * loss_map
                 + train_cfg.w_progress * loss_prog
                 + train_cfg.w_reg * (loss_var + loss_cov))

        return total, {
            "act": float(loss_action.detach()),
            "safe": float(loss_safety.detach()),
            "fsafe": float(loss_fdanger.detach()),
            "jepa": float(loss_jepa.detach()),
            "map": float(loss_map.detach()),
            "iou": occ_iou,
            "prog": float(loss_prog.detach()),
            "var": float(loss_var.detach()),
            "cov": float(loss_cov.detach()),
        }


# ==========================================================================
# Frozen backbone wrapper (used by preprocessing and live inference)
# ==========================================================================
class Backbone(nn.Module):
    """Frozen DINOv2 -> (n_tokens, feat_dim) per frame: CLS + pooled patch grid."""

    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    def __init__(self, cfg: ModelConfig, device="cuda"):
        super().__init__()
        self.cfg = cfg
        self.net = torch.hub.load("facebookresearch/dinov2", cfg.backbone)
        self.net.eval()
        for p in self.net.parameters():
            p.requires_grad = False
        self.device = device
        self.to(device)
        mean = torch.tensor(self.IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
        std = torch.tensor(self.IMAGENET_STD, device=device).view(1, 3, 1, 1)
        self.register_buffer("mean", mean, persistent=False)
        self.register_buffer("std", std, persistent=False)

    @torch.no_grad()
    def forward(self, imgs_uint8):
        """imgs_uint8: (B, 3, H, W) RGB uint8 on any device -> (B, n_tokens, feat_dim)."""
        cfg = self.cfg
        x = imgs_uint8.to(self.device, non_blocking=True).float() / 255.0
        if x.shape[-1] != cfg.img_size:
            x = F.interpolate(x, size=(cfg.img_size, cfg.img_size),
                              mode="bilinear", align_corners=False)
        x = (x - self.mean) / self.std
        with torch.autocast("cuda", dtype=torch.float16,
                            enabled=self.device != "cpu"):
            out = self.net.forward_features(x)
        cls = out["x_norm_clstoken"]                          # (B, 384)
        patches = out["x_norm_patchtokens"]                   # (B, G*G, 384)
        B, N, D = patches.shape
        g = int(N ** 0.5)
        grid = patches.view(B, g, g, D).permute(0, 3, 1, 2)   # (B, D, g, g)
        pooled = F.adaptive_avg_pool2d(grid, (cfg.pool_rows, cfg.pool_cols))
        pooled = pooled.flatten(2).permute(0, 2, 1)           # (B, r*c, D)
        return torch.cat([cls[:, None, :], pooled], dim=1).float()

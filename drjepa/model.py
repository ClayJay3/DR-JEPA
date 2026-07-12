"""RoverJEPA: map-space JEPA perception for goal-driven driving.

Design (all sizes from ModelConfig):

  frozen DINOv2 (precomputed)      145 tokens x 384 per frame
        |
  MapDecoder (multi-frame + ego-motion)
        |-- 5-channel metric wedge: occupancy / visibility / elevation /
        |                           sand / tip-hazard (48x48 @ 0.5 m)
        |-- danger logit: "trouble within ~1 s" from the same trunk
        |
  persistent belief map (fusion lives in drjepa.pilot)
        |
  MapCompleter (map-space JEPA)  -> predicted layout of UNSEEN map cells
                                    (I-JEPA recipe: mask what the camera
                                    has not swept, predict it from context)

The temporal/embedding JEPA branch (FrameEncoder, causal transformer, EMA
target encoder, action-conditioned predictor, BC policy head) was REMOVED
after v11: with the map pilot driving, its only inference output was the
danger scalar, and a danger head on the perception trunk matches it
without the extra ~2.5M parameters and per-step latency. The JEPA that
survives is the one operating on the world model itself -- the map.

The DINOv2 backbone stays frozen, which is what preserves sim-to-real
robustness of the visual features.
"""

import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig


class MapDecoder(nn.Module):
    """Multi-frame metric perception: DINOv2 tokens -> terrain wedge.

    Predicts, for the 24m x 24m area in front of the camera (48x48 cells),
    five per-cell quantities:
      occ  : obstacle logit (rocks, trees, bushes)
      conf : visibility/confidence logit (what the camera can actually see)
      elev : terrain elevation relative to the rover (metres, tanh-bounded)
      sand : soft-ground logit
      haz  : steep-ground logit, supervised DIRECTLY on true terrain grade.
             Gradients of the predicted (smoothed) elevation carry no
             usable steepness signal -- measured: true 0.5-grade banks got
             LOWER predicted gradients than mild slopes -- so the decoder
             classifies steepness from visual cues (bank shading, texture)
             instead of us differentiating its regression output.
    The decoder sees the current frame plus cfg.frame_offsets lookbacks
    (channel-stacked per grid position) together with the ego-motion of
    those frames (speed + commanded steer), so it can exploit motion
    parallax -- the strongest monocular depth cue -- instead of texture
    scale alone. Outputs are fused into the persistent world map.
    """

    ELEV_RANGE = 3.5    # metres; elevation output = ELEV_RANGE * tanh(x)

    def __init__(self, cfg: ModelConfig):
        """Per-token projection + transposed-conv pyramid up to wedge size."""
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
        self.net = nn.Sequential(*layers, nn.Conv2d(ch[n_up], 5, 3, 1, 1))
        # danger head on the SAME perception trunk: strided convs keep the
        # spatial layout (a rock dead ahead is not a rock off to the side)
        # before pooling into a scalar "trouble within ~1 s" logit. The
        # trunk already fuses multi-frame parallax + ego-motion, which is
        # everything a 1 s-horizon outcome estimate needs.
        # two stride-2 convs then adaptive-pool to a fixed 3x3, so the head
        # is independent of the token-grid resolution (12x12 -> 3x3 is a
        # no-op pool; 24x24 -> 6x6 -> pooled to 3x3) and old checkpoints
        # still load unchanged
        self.danger_net = nn.Sequential(
            nn.Conv2d(256, 128, 3, 2, 1), nn.GELU(),
            nn.Conv2d(128, 128, 3, 2, 1), nn.GELU(),
            nn.AdaptiveAvgPool2d((3, 3)))
        self.danger_out = nn.Sequential(
            nn.Linear(128 * 3 * 3 + 64, 128), nn.GELU(), nn.Linear(128, 1))

    def forward(self, tokens, motion):
        """tokens (..., F, n_tokens, feat_dim) stacked [current, -2, -4, ...];
        motion (..., F*2) = (speed, steer) per stacked frame.
        Returns (occ, conf, elev, sand, haz, danger): the first five
        (..., C, C) indexed [i = x_right, j = z_forward] (rover frame),
        danger a (...,) scalar logit for "trouble within ~1 s".
        occ/conf/sand/haz/danger are logits; elev is metres."""
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
        d = self.danger_net(x).flatten(1)                  # (N, 128*3*3)
        danger = self.danger_out(torch.cat([d, c], dim=-1))  # (N, 1)
        out = self.net(x)                        # (N, 5, cells, cells)
        # conv output is image-aligned (row ~ image-y ~ distance-from-far,
        # col ~ image-x ~ lateral); re-index to the rover-frame (i, j)
        # convention so convs only learn a local perspective warp, never a
        # global transpose
        out = out.flip(-2).transpose(-1, -2)
        shape = (*lead, self.cells, self.cells)
        occ = out[:, 0].reshape(shape)
        conf = out[:, 1].reshape(shape)
        elev = self.ELEV_RANGE * torch.tanh(out[:, 2]).reshape(shape)
        sand = out[:, 3].reshape(shape)
        haz = out[:, 4].reshape(shape)
        danger = danger.reshape(lead) if lead else danger.reshape(())
        return occ, conf, elev, sand, haz, danger


class MapCompleter(nn.Module):
    """Map-space JEPA: predict the map where the rover has not looked.

    Input is a partial top-down belief map around the rover (channels:
    observed occupancy, hazard, sand, and the observed mask); output is the
    predicted occupancy / terrain-hazard / sand probability for every cell,
    trained with the masked-prediction recipe (I-JEPA on maps): show the
    network the observed part, grade its guess for the hidden part against
    the episode's ground-truth grids.

    At inference the planner reads these predictions for unobserved cells,
    so route choices anticipate what is *probably* around the corner
    (walls tend to continue, open ground tends to stay open) instead of
    treating all unknown space as uniformly mild.
    """

    IN_CH = 4     # occ, hazard, sand, observed-mask
    OUT_CH = 3    # occ, hazard, sand logits

    def __init__(self, cfg: ModelConfig):
        """Small 3-level U-Net over the comp_cells x comp_cells crop."""
        super().__init__()
        c = 32

        def block(ci, co):
            return nn.Sequential(nn.Conv2d(ci, co, 3, 1, 1), nn.GELU(),
                                 nn.Conv2d(co, co, 3, 1, 1), nn.GELU())

        self.enc1 = block(self.IN_CH, c)
        self.enc2 = block(c, 2 * c)
        self.enc3 = block(2 * c, 4 * c)
        self.pool = nn.MaxPool2d(2)
        self.up2 = nn.ConvTranspose2d(4 * c, 2 * c, 2, 2)
        self.dec2 = block(4 * c, 2 * c)
        self.up1 = nn.ConvTranspose2d(2 * c, c, 2, 2)
        self.dec1 = block(2 * c, c)
        self.head = nn.Conv2d(c, self.OUT_CH, 1)

    def forward(self, x):
        """(N, IN_CH, G, G) partial map -> (N, OUT_CH, G, G) logits."""
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        d2 = self.dec2(torch.cat([self.up2(e3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.head(d1)


class RoverJEPA(nn.Module):
    """The full trainable model: metric perception + map-space JEPA.

    Two submodules: the MapDecoder (tokens -> 5-channel wedge + danger
    logit) and the MapCompleter (partial belief map -> predicted hidden
    cells). Everything trains jointly in `compute_losses`; inference-side
    orchestration (map fusion, planning, control) lives in `drjepa.pilot`.
    """

    def __init__(self, cfg: ModelConfig = None):
        """Build the perception decoder and the map completer."""
        super().__init__()
        self.cfg = cfg = cfg or ModelConfig()
        self.map_decoder = MapDecoder(cfg)
        self.map_completer = MapCompleter(cfg)

    # ------------------------------------------------------------------
    def compute_losses(self, batch, train_cfg):
        """All training losses for one batch (dict of tensors).

        batch keys actually consumed (W = window, C = wedge_cells,
        G = comp_cells) -- the dataset may carry extra keys (expert action
        labels etc.) which are simply ignored here:
          tokens    (B, W, n_tokens, feat_dim)  frozen backbone features
          danger    (B, W, 1)   "trouble within ~1 s" in [0, 1]
          occ/vis   (B, W, C, C)  wedge occupancy / visibility targets
          elev      (B, W, C, C)  wedge elevation targets (metres)
          sand      (B, W, C, C)  wedge soft-ground targets in [0, 1]
          motion    (B, W, 2)   (speed, executed steer) per frame
          comp_in   (B, 4, G, G)  partial map (map-completion input)
          comp_tgt  (B, 3, G, G)  full ground-truth map
          comp_mask (B, 1, G, G)  1 = hidden cell (graded), 0 = shown
        """
        cfg = self.cfg
        tokens = batch["tokens"]
        danger = batch["danger"]
        occ_gt, vis_gt = batch["occ"], batch["vis"]
        motion = batch["motion"]
        B, W = tokens.shape[:2]

        # ---------------- metric perception (occupancy wedge) ----------------
        # stack the multi-frame input with strided slices; supervise every
        # second frame past the lookback warm-up
        offs = cfg.frame_offsets
        m_off = max(offs)
        tok_stack = torch.stack(
            [tokens[:, m_off - o:W - o:2] for o in offs], dim=2)
        mot_stack = torch.cat(
            [motion[:, m_off - o:W - o:2] for o in offs], dim=-1)
        occ_logit, conf_logit, elev_pred, sand_logit, haz_logit, \
            d_logit = self.map_decoder(tok_stack, mot_stack)
        occ_gt = occ_gt[:, m_off::2]
        vis_gt = vis_gt[:, m_off::2]
        elev_gt = batch["elev"][:, m_off::2]
        sand_gt = batch["sand"][:, m_off::2]
        # steep-ground target from the GT elevation wedge (exact geometry).
        # The ramp is anchored TIGHT on the tip-over threshold (0.38 ->
        # tip_roll 0.50): this channel must mean "can tip the rover", not
        # "sloped". A wider ramp from 0.30 was tried first and walled off
        # perfectly drivable 0.3-0.4 hillsides once fused (log-odds
        # saturation erases the moderate/fatal distinction downstream, so
        # the distinction must be drawn in the label).
        ggx, ggz = torch.gradient(elev_gt, spacing=cfg.wedge_res,
                                  dim=(-2, -1))
        grade_gt = torch.sqrt(ggx * ggx + ggz * ggz)
        haz_gt = ((grade_gt - 0.38) / 0.12).clamp(0.0, 1.0)
        C = occ_logit.shape[-1]
        # nearer rows matter more for driving; beyond wedge_range_cells the
        # monocular distance ambiguity is too large to supervise usefully
        row_w = torch.linspace(1.6, 0.7, C, device=occ_logit.device)
        row_w[cfg.wedge_range_cells:] = 0.0
        row_w = row_w[None, None, None, :]
        w = vis_gt * row_w
        w_norm = w.numel() / w.sum().clamp(min=1.0)
        loss_occ = F.binary_cross_entropy_with_logits(
            occ_logit, occ_gt, weight=w,
            pos_weight=torch.tensor(train_cfg.occ_pos_weight,
                                    device=occ_logit.device)) * w_norm
        loss_conf = F.binary_cross_entropy_with_logits(conf_logit, vis_gt)
        loss_elev = (F.smooth_l1_loss(elev_pred, elev_gt, reduction="none")
                     * w).sum() / w.sum().clamp(min=1.0)

        # ---- real episodes do not supervise hazard or sand ----
        # Both are ABSENT labels on real data, not measurements:
        #   hazard is derived from the GRADIENT of the elevation wedge, and
        #   differentiating a depth-derived elevation field amplifies depth
        #   noise into phantom cliffs -- measured 14.2% of real cells labelled
        #   tip-hazard (grade p95 = 1.23!) vs ~0.2% in sim. Trained on that,
        #   the head fired on flat ground AND missed real banks: val haz rose
        #   0.102 -> 0.143 while occupancy improved, which vetoed checkpoint
        #   selection and shipped a pre-convergence model (IoU 0.017).
        #   sand is written as all-zeros by the converters because we cannot
        #   measure it -- supervising on that teaches "no sand" on sandy
        #   desert.
        # No signal beats 25% noise; sim teaches both channels, and --augment
        # is what carries them across domains. (occ/vis/elev stay supervised:
        # occupancy from depth is robust -- measured 3.4% on ZED, matching
        # sim -- and elevation is a real measurement, noisy but zero-mean.)
        keep = 1.0 - batch["is_real"].reshape(-1, *([1] * (w.dim() - 1)))
        w_r = w * keep
        denom = w_r.sum().clamp(min=1.0)
        loss_sand = (F.binary_cross_entropy_with_logits(
            sand_logit, sand_gt, reduction="none") * w_r).sum() / denom
        # steep cells are rare (~2-4%); pos_weight keeps recall alive.
        # missing a bank tips the rover -- terminal -- while a false alarm
        # only costs a detour
        loss_haz = (F.binary_cross_entropy_with_logits(
            haz_logit, haz_gt, reduction="none",
            pos_weight=torch.tensor(4.0, device=haz_logit.device))
            * w_r).sum() / denom
        loss_map = loss_occ + 0.25 * loss_conf
        with torch.no_grad():
            R = cfg.wedge_range_cells
            pred = (occ_logit[..., :R] > 0) & (vis_gt[..., :R] > 0.5)
            gt = (occ_gt[..., :R] > 0.5) & (vis_gt[..., :R] > 0.5)
            inter = (pred & gt).sum()
            union = (pred | gt).sum().clamp(min=1)
            occ_iou = float(inter) / float(union)

        # ---------------- map-space JEPA (hidden-map completion) ----------------
        comp_logit = self.map_completer(batch["comp_in"])     # (B, 3, G, G)
        hidden = batch["comp_mask"]                           # 1 = grade here
        pw = torch.tensor([4.0, 4.0, 2.0], device=comp_logit.device)
        loss_comp = 0.0
        for ch in range(3):
            l = F.binary_cross_entropy_with_logits(
                comp_logit[:, ch], batch["comp_tgt"][:, ch],
                weight=hidden[:, 0], pos_weight=pw[ch], reduction="sum")
            loss_comp = loss_comp + l / hidden.sum().clamp(min=1.0)
        loss_comp = loss_comp / 3.0
        with torch.no_grad():
            hp = (comp_logit[:, 0] > 0) & (hidden[:, 0] > 0.5)
            hg = (batch["comp_tgt"][:, 0] > 0.5) & (hidden[:, 0] > 0.5)
            comp_iou = float((hp & hg).sum()) / float((hp | hg).sum().clamp(min=1))

        # ---------------- danger (same trunk as the wedge) ----------------
        # supervised on the decoder's frames; the label is the sim's
        # "trouble within ~1 s" outcome signal
        d_gt = danger[:, m_off::2].reshape(-1, 1)
        loss_danger = F.binary_cross_entropy_with_logits(
            d_logit.reshape(-1, 1), d_gt)

        total = (train_cfg.w_map * loss_map
                 + train_cfg.w_elev * loss_elev
                 + train_cfg.w_sand * loss_sand
                 + train_cfg.w_haz * loss_haz
                 + train_cfg.w_complete * loss_comp
                 + train_cfg.w_safety * loss_danger)

        return total, {
            "safe": float(loss_danger.detach()),
            "map": float(loss_map.detach()),
            "elev": float(loss_elev.detach()),
            "sand": float(loss_sand.detach()),
            "haz": float(loss_haz.detach()),
            "comp": float(loss_comp.detach()),
            "ciou": comp_iou,
            "iou": occ_iou,
        }



# ==========================================================================
# Frozen backbone wrapper (used by preprocessing and live inference)
# ==========================================================================
class Backbone(nn.Module):
    """Frozen DINOv2/DINOv3 -> (n_tokens, feat_dim): CLS + pooled patch grid.

    DINOv2 loads from torch.hub (open weights). DINOv3 loads through
    HuggingFace transformers -- the DINOv3 HF repo ships only the
    transformers format (safetensors), not the original .pth, and its
    weights are license-gated: accept the licence on the model page, run
    `huggingface-cli login` once, and it downloads automatically. The HF
    id (or a local dir) comes from cfg.backbone_weights.
    """

    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    def __init__(self, cfg: ModelConfig, device="cuda"):
        """Load the frozen pretrained backbone."""
        super().__init__()
        self.cfg = cfg
        self._hf = cfg.backbone.startswith("dinov3")
        if self._hf:
            from transformers import AutoModel
            src = (getattr(cfg, "backbone_weights", "") or
                   os.environ.get("DINOV3_WEIGHTS", "") or
                   "facebook/dinov3-vits16-pretrain-lvd1689m")
            self.net = AutoModel.from_pretrained(src)
            # skip CLS + register tokens to get the patch grid
            self.n_prefix = 1 + getattr(self.net.config,
                                        "num_register_tokens", 0)
        else:
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
            if self._hf:                                      # DINOv3 (HF)
                h = self.net(pixel_values=x).last_hidden_state
                cls = h[:, 0]
                patches = h[:, self.n_prefix:]                # drop registers
            else:                                             # DINOv2 (hub)
                out = self.net.forward_features(x)
                cls = out["x_norm_clstoken"]                  # (B, 384)
                patches = out["x_norm_patchtokens"]           # (B, G*G, 384)
        cls = cls.float()
        patches = patches.float()
        B, N, D = patches.shape
        g = int(N ** 0.5)
        grid = patches.view(B, g, g, D).permute(0, 3, 1, 2)   # (B, D, g, g)
        pooled = F.adaptive_avg_pool2d(grid, (cfg.pool_rows, cfg.pool_cols))
        pooled = pooled.flatten(2).permute(0, 2, 1)           # (B, r*c, D)
        return torch.cat([cls[:, None, :], pooled], dim=1).float()

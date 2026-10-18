"""GeoBEV: one camera image -> robot-frame ground grid, for ANY camera.

    image ──► encoder (small pretrained CNN) ──► stride-8 feature map
                                                      │
    camera intrinsics K + pose (R, t) in the robot frame
        └──► every ground cell x every height level is projected into the
             image and the feature map is bilinearly sampled there
             (parameter-free geometry: correct for any lens / mast / tilt)
                                                      │
    per-cell column of samples + ray geometry ──► top-down U-Net ──► heads:
        vis (logit)  occ (logit)  obs_h (m)  elev (m)  grade  sand (logit)

The view transform is the Simple-BEV recipe (Harley et al. 2023): sampling
image features at the projections of 3D points is as good as learned
depth lifting and far cheaper. Because the projection uses the real camera,
the network never has to memorize one camera's row -> range table -- the
failure that broke the previous model on the ZED (a 26 % range error).
"""

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

HEADS = ("vis", "occ", "obs_h", "elev", "grade", "sand")


@dataclass
class ModelConfig:
    """GeoBEV architecture."""
    encoder: str = "efficientnet_b0"
    input_w: int = 512                # images are resized to this before
    input_h: int = 320                # the encoder (intrinsics scaled too)
    feat_ch: int = 64                 # sampled feature width per height
    heights: tuple = (-2.0, -1.0, -0.4, 0.0, 0.3, 0.7, 1.2, 2.0)
    cells: int = 48
    res: float = 0.5
    bev_ch: tuple = (96, 128, 192)    # U-Net widths at 48 / 24 / 12 cells
    pretrained: bool = True
    depth_aux: bool = False           # training-only per-pixel depth head on the
    #                                   encoder (BEVDepth-style supervision);
    #                                   dropped at inference / export
    lift: str = "geometric"           # "geometric" (projection sampling) or
    #                                   "implicit" (the v12 design: pooled
    #                                   feature grid + camera vector + learned
    #                                   up-convolutions) -- kept only so the
    #                                   two can be A/B'd on identical data

    def to_dict(self):
        return asdict(self)

    @staticmethod
    def from_dict(d):
        d = dict(d)
        if "heights" in d:
            d["heights"] = tuple(d["heights"])
        if "bev_ch" in d:
            d["bev_ch"] = tuple(d["bev_ch"])
        return ModelConfig(**{k: v for k, v in d.items() if k in ModelConfig.__dataclass_fields__})


# --------------------------------------------------------------------------
# encoders: image (B, 3, H, W) normalized -> (B, C, H/8, W/8)
# --------------------------------------------------------------------------
class TorchvisionEncoder(nn.Module):
    """Stride-8 + stride-16 features of a torchvision backbone, fused."""

    def __init__(self, name, out_ch, pretrained=True):
        super().__init__()
        import torchvision.models as tvm
        w = "DEFAULT" if pretrained else None
        if name == "efficientnet_b0":
            f = tvm.efficientnet_b0(weights=w).features
            self.s8, self.s16 = f[:4], f[4:6]                 # 40 ch, 112 ch
            c8, c16 = 40, 112
        elif name == "mobilenet_v3_large":
            f = tvm.mobilenet_v3_large(weights=w).features
            self.s8, self.s16 = f[:7], f[7:13]                # 40 ch, 112 ch
            c8, c16 = 40, 112
        elif name == "resnet18":
            m = tvm.resnet18(weights=w)
            self.s8 = nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool, m.layer1, m.layer2)
            self.s16 = m.layer3
            c8, c16 = 128, 256
        elif name == "regnet_y_800mf":
            m = tvm.regnet_y_800mf(weights=w)
            self.s8 = nn.Sequential(m.stem, m.trunk_output.block1, m.trunk_output.block2)
            self.s16 = m.trunk_output.block3
            c8, c16 = 144, 320
        else:
            raise ValueError(f"unknown encoder {name}")
        self.lat8 = nn.Conv2d(c8, out_ch, 1)
        self.lat16 = nn.Conv2d(c16, out_ch, 1)
        self.fuse = nn.Sequential(nn.Conv2d(out_ch, out_ch, 3, 1, 1), nn.GroupNorm(8, out_ch), nn.GELU())

    def forward(self, x):
        a = self.s8(x)
        b = self.s16(a)
        y = self.lat8(a) + F.interpolate(self.lat16(b), size=a.shape[-2:], mode="bilinear",
                                         align_corners=False)
        return self.fuse(y)


class DinoEncoder(nn.Module):
    """Frozen DINOv3 ViT-S/16 patch features (stride 16) -> upsampled to stride 8."""

    def __init__(self, out_ch, weights="facebook/dinov3-vits16-pretrain-lvd1689m"):
        super().__init__()
        from transformers import AutoModel
        self.net = AutoModel.from_pretrained(weights).eval()
        for p in self.net.parameters():
            p.requires_grad = False
        self.n_prefix = 1 + getattr(self.net.config, "num_register_tokens", 0)
        self.proj = nn.Sequential(nn.Conv2d(384, out_ch, 1), nn.GroupNorm(8, out_ch), nn.GELU(),
                                  nn.Conv2d(out_ch, out_ch, 3, 1, 1))

    def train(self, mode=True):
        super().train(mode)
        self.net.eval()
        return self

    def forward(self, x):
        B, _, H, W = x.shape
        with torch.no_grad():
            h = self.net(pixel_values=x).last_hidden_state[:, self.n_prefix:]
        g = h.transpose(1, 2).reshape(B, 384, H // 16, W // 16).float()
        g = F.interpolate(g, scale_factor=2, mode="bilinear", align_corners=False)
        return self.proj(g)


def make_encoder(cfg: ModelConfig):
    if cfg.encoder.startswith("dinov3"):
        return DinoEncoder(cfg.feat_ch)
    return TorchvisionEncoder(cfg.encoder, cfg.feat_ch, cfg.pretrained)


# --------------------------------------------------------------------------
# top-down network
# --------------------------------------------------------------------------
class ResBlock(nn.Module):
    def __init__(self, ci, co, stride=1):
        super().__init__()
        self.a = nn.Conv2d(ci, co, 3, stride, 1)
        self.n1 = nn.GroupNorm(8, co)
        self.b = nn.Conv2d(co, co, 3, 1, 1)
        self.n2 = nn.GroupNorm(8, co)
        self.skip = None if (ci == co and stride == 1) else nn.Conv2d(ci, co, 1, stride)

    def forward(self, x):
        y = F.gelu(self.n1(self.a(x)))
        y = self.n2(self.b(y))
        return F.gelu(y + (x if self.skip is None else self.skip(x)))


class BEVNet(nn.Module):
    """Three-level U-Net over the ground grid."""

    def __init__(self, ci, ch):
        super().__init__()
        c1, c2, c3 = ch
        self.inp = nn.Sequential(nn.Conv2d(ci, c1, 1), nn.GroupNorm(8, c1), nn.GELU())
        self.e1 = ResBlock(c1, c1)
        self.e2 = nn.Sequential(ResBlock(c1, c2, 2), ResBlock(c2, c2))
        self.e3 = nn.Sequential(ResBlock(c2, c3, 2), ResBlock(c3, c3), ResBlock(c3, c3))
        self.d2 = ResBlock(c3 + c2, c2)
        self.d1 = ResBlock(c2 + c1, c1)
        self.out = nn.Conv2d(c1, len(HEADS), 1)

    def forward(self, x):
        a = self.e1(self.inp(x))
        b = self.e2(a)
        c = self.e3(b)
        b = self.d2(torch.cat([F.interpolate(c, size=b.shape[-2:], mode="bilinear", align_corners=False), b], 1))
        a = self.d1(torch.cat([F.interpolate(b, size=a.shape[-2:], mode="bilinear", align_corners=False), a], 1))
        return self.out(a)


# --------------------------------------------------------------------------
class ImplicitLift(nn.Module):
    """The v12 view transform, generalized to any camera for the A/B.

    The stride-8 feature map is pooled to a fixed grid, every grid position
    is concatenated with a camera descriptor (focal lengths, principal
    point, mount height, pitch, roll and f*h -- the v12 camera conditioning
    extended to non-square images), and learned transposed convolutions map
    it to the ground grid. Output rows are re-indexed so image axes map
    locally onto grid axes (the v8 fix).
    """

    N_CAM = 10

    def __init__(self, feat_ch, out_ch, cells, grid=24):
        super().__init__()
        self.grid, self.cells = grid, cells
        self.cam = nn.Sequential(nn.Linear(self.N_CAM, 64), nn.GELU(), nn.Linear(64, 64))
        self.tok = nn.Conv2d(feat_ch + 64, 256, 1)
        n_up = int(math.log2(cells // grid))
        ch = [256, 192, 128, 96]
        layers = [nn.Conv2d(256, 256, 3, 1, 1), nn.GELU()]
        for i in range(n_up):
            layers += [nn.ConvTranspose2d(ch[i], ch[i + 1], 4, 2, 1), nn.GELU(),
                       nn.Conv2d(ch[i + 1], ch[i + 1], 3, 1, 1), nn.GELU()]
        self.net = nn.Sequential(*layers, nn.Conv2d(ch[n_up], out_ch, 1))

    @staticmethod
    def camera_vector(K, R, t, hw):
        H, W = hw
        fwd, right = R[:, :, 2], R[:, :, 0]
        pitch = torch.asin(fwd[:, 1].clamp(-1, 1))
        roll = torch.asin((-right[:, 1]).clamp(-1, 1))
        fxn, fyn = 2 * K[:, 0, 0] / W, 2 * K[:, 1, 1] / H
        h = t[:, 1]
        return torch.stack([fxn, fyn, 2 * K[:, 0, 2] / W - 1, 2 * K[:, 1, 2] / H - 1, h / 1.5,
                            torch.sin(pitch), torch.cos(pitch), torch.sin(roll),
                            fyn * h / 1.5, t[:, 2]], 1)

    def forward(self, feat, K, R, t, hw):
        g = F.adaptive_avg_pool2d(feat, (self.grid, self.grid))
        c = self.cam(self.camera_vector(K, R, t, hw))
        x = self.tok(torch.cat([g, c[:, :, None, None].expand(-1, -1, self.grid, self.grid)], 1))
        out = self.net(x)                                 # image-aligned (row ~ image y)
        return out.flip(-2).transpose(-1, -2)             # -> [i = x, j = z]


class GeoBEV(nn.Module):
    """Image + camera -> ground-grid physical quantities (see module doc)."""

    MEAN = (0.485, 0.456, 0.406)
    STD = (0.229, 0.224, 0.225)

    def __init__(self, cfg: ModelConfig = None):
        super().__init__()
        self.cfg = cfg = cfg or ModelConfig()
        self.encoder = make_encoder(cfg)
        C, res = cfg.cells, cfg.res
        xs = (torch.arange(C) + 0.5) * res - C * res / 2
        zs = (torch.arange(C) + 0.5) * res
        X, Z = torch.meshgrid(xs, zs, indexing="ij")
        hs = torch.tensor(cfg.heights, dtype=torch.float32)
        K = len(hs)
        P = torch.stack([X[None].expand(K, C, C), hs[:, None, None].expand(K, C, C),
                         Z[None].expand(K, C, C)], -1)                # (K, C, C, 3)
        self.register_buffer("points", P.reshape(-1, 3), persistent=False)
        self.register_buffer("mean", torch.tensor(self.MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(self.STD).view(1, 3, 1, 1), persistent=False)
        self.n_geo = 3
        if cfg.depth_aux:
            self.depth_head = nn.Sequential(nn.Conv2d(cfg.feat_ch, cfg.feat_ch, 3, 1, 1), nn.GELU(),
                                            nn.Conv2d(cfg.feat_ch, 1, 1))
        if cfg.lift == "implicit":
            self.implicit = ImplicitLift(cfg.feat_ch, cfg.bev_ch[0], C)
            self.bev = BEVNet(cfg.bev_ch[0], cfg.bev_ch)
        else:
            self.bev = BEVNet(K * (cfg.feat_ch + self.n_geo), cfg.bev_ch)

    def lift(self, feat, K, R, t, img_hw):
        """Sample features at every (cell, height) projection.

        feat (B, Cf, h, w); K (B, 3, 3) at input-image pixels (edge
        convention: pixel u spans [u, u+1)); R (B, 3, 3), t (B, 3) camera ->
        robot frame. Returns (B, Kh*(Cf+3), C, C)."""
        cfg = self.cfg
        B = feat.shape[0]
        H, W = img_hw
        P = self.points[None] - t[:, None, :]                       # (B, N, 3)
        pc = torch.einsum("bnj,bjk->bnk", P, R)                    # R^T (p - t)
        z = pc[..., 2]
        zs = z.clamp(min=1e-3)
        u = K[:, 0, 0, None] * pc[..., 0] / zs + K[:, 0, 2, None]
        v = K[:, 1, 1, None] * pc[..., 1] / zs + K[:, 1, 2, None]
        gx = 2.0 * u / W - 1.0
        gy = 2.0 * v / H - 1.0
        valid = (z > 0.1) & (gx > -1) & (gx < 1) & (gy > -1) & (gy < 1)
        grid = torch.stack([gx, gy], -1).clamp(-2, 2)[:, :, None, :]
        s = F.grid_sample(feat, grid.to(feat.dtype), mode="bilinear", padding_mode="zeros",
                          align_corners=False)[..., 0]               # (B, Cf, N)
        vf = valid.to(s.dtype)
        s = s * vf[:, None]
        geo = torch.stack([vf, torch.log(zs.clamp(max=200.0)) / 3.0 * vf, gy.clamp(-1.5, 1.5) * vf], 1)
        x = torch.cat([s, geo.to(s.dtype)], 1)                      # (B, Cf+3, N)
        Kh, C = len(cfg.heights), cfg.cells
        return x.view(B, -1, Kh, C, C).reshape(B, -1, C, C)

    def forward(self, img, K, R, t, return_feat=False):
        """img (B, 3, H, W) float RGB in [0, 1] at (input_h, input_w)."""
        x = (img - self.mean) / self.std
        f = self.encoder(x)
        if self.cfg.lift == "implicit":
            bev = self.implicit(f, K, R, t, img.shape[-2:])
        else:
            bev = self.lift(f, K, R, t, img.shape[-2:])
        out = self.bev(bev)
        extra = {"depth_log": self.depth_head(f)[:, 0]} if (self.cfg.depth_aux and self.training) else {}
        if return_feat:
            extra["feat"] = f
        return {**extra,
            "vis": out[:, 0], "occ": out[:, 1],
            "obs_h": F.softplus(out[:, 2]),
            "elev": 3.5 * torch.tanh(out[:, 3] / 3.5),
            "grade": F.softplus(out[:, 4]),
            "sand": out[:, 5],
        }


def count_params(m, trainable=True):
    return sum(p.numel() for p in m.parameters() if p.requires_grad or not trainable)

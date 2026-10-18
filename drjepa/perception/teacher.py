"""Depth-model teacher: monocular relative depth + mount-plane alignment.

A heavier, training-free perception path used as (1) a baseline in the
benchmarks and (2) an offline labeller for unlabeled real video. The robot
never needs to run it.

Depth Anything V2 predicts affine-invariant inverse depth. Its metric
variants are unusable at rover height (measured on ZED 2i at 0.4 m:
AbsRel 2-8, the ground placed 4-6x too far), but every robot knows its
camera's mount: the ground the robot stands on is a known plane in the
camera frame. Fitting inverse depth = a * prediction + b on near-ground
pixels (0.6-4 m, iteratively trimmed) recovers metric depth with
AbsRel 0.04 (0.5-3 m) / 0.10 (3-6 m) / 0.24 (6-10 m) on held-out ZED
stereo -- then the same geometric labeller used for stereo turns it into
ground grids.
"""

import numpy as np
import torch

from ..config import GridConfig
from ..labels import depth_targets


class DepthTeacher:
    """Callable with the Perceiver signature (rgb, camera, R, t) -> grids."""

    def __init__(self, size="Base", input_h=518, device="cuda", step=0.25):
        from transformers import AutoModelForDepthEstimation
        self.m = AutoModelForDepthEstimation.from_pretrained(
            f"depth-anything/Depth-Anything-V2-{size}-hf").to(device).eval().half()
        self.dev = device
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
        self.ih, self.grid, self.step = input_h, GridConfig(), step

    @torch.no_grad()
    def disparity(self, rgb):
        H, W = rgb.shape[:2]
        x = torch.tensor(rgb / 255.0, dtype=torch.float32, device=self.dev).permute(2, 0, 1)[None]
        hh, ww = self.ih, int(round(self.ih * W / H / 14)) * 14
        x = torch.nn.functional.interpolate(x, (hh, ww), mode="bicubic", align_corners=False)
        p = self.m(pixel_values=((x - self.mean) / self.std).half()).predicted_depth[:, None].float()
        return torch.nn.functional.interpolate(p, (H, W), mode="bilinear", align_corners=False)[0, 0].cpu().numpy()

    def depth(self, rgb, cam, R, t):
        """Metric depth (H, W) via affine inverse-depth fit to the mount plane."""
        disp = np.maximum(self.disparity(rgb), 0)
        n_c = R.T @ np.array([0.0, 1.0, 0.0])           # ground normal, camera frame
        h = float(t[1])
        nr = -(cam.rays() @ n_c)
        plane_d = np.where(nr > 1e-3, h / np.maximum(nr, 1e-6), 0.0)
        v = np.arange(cam.height)[:, None] * np.ones((1, cam.width))
        near = (plane_d > 0.6) & (plane_d < 4.0) & (v > cam.cy)
        pdisp = 1.0 / np.maximum(plane_d, 1e-6)
        sel = near & (disp > 0)
        dd = disp
        for thr in (0.5, 0.25, 0.12):
            if sel.sum() < 200:
                break
            A = np.stack([disp[sel], np.ones(sel.sum())], 1)
            a, b = np.linalg.lstsq(A, pdisp[sel], rcond=None)[0]
            dd = a * disp + b
            sel = near & (disp > 0) & (np.abs(dd - pdisp) / pdisp < thr)
        return np.where(dd > 1e-3, 1.0 / np.maximum(dd, 1e-6), 0.0)

    def __call__(self, rgb, camera, R=None, t=None, body_pitch=0.0, body_roll=0.0):
        if R is None:
            R, t = camera.pose(body_pitch, body_roll)
        cam = camera if (camera.width, camera.height) == rgb.shape[1::-1] else \
            camera.resized(rgb.shape[1], rgb.shape[0])
        L = depth_targets(self.depth(rgb, cam, R, t), cam, R, t, self.grid, dmax=25.0, step=self.step)
        occ = L["occ"].astype(np.float32)
        return {"occ": occ, "occ_logit": np.where(L["occ"], 2.0, np.where(L["vis"], -6.0, -2.5)),
                "vis": L["vis"].astype(np.float32), "obs_h": L["obs_h"],
                "elev": np.nan_to_num(L["elev"]), "grade": np.nan_to_num(L["grade"]),
                "sand": np.zeros_like(occ), "score": np.where(L["vis"], L["obs_h"], 0.0)}

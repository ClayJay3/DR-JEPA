"""Runtime perception: any RGB image + its Camera -> robot-frame ground grids.

    perceiver = Perceiver("runs/geobev/best.pt")
    grids = perceiver(rgb, camera, body_pitch=imu_pitch, body_roll=imu_roll)
    grids["occ"]    # (C, C) obstacle probability
    grids["obs_h"]  # obstacle height above the terrain (m)
    grids["elev"], grids["grade"], grids["vis"], grids["sand"]

The camera pose comes from the Camera mount composed with the body
attitude (IMU), or can be passed directly as (R, t).
"""

import cv2
import numpy as np
import torch

from .model import GeoBEV, ModelConfig


class Perceiver:
    """Loads a GeoBEV checkpoint and runs it on single frames."""

    def __init__(self, checkpoint, device=None, half=True):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.cfg = ModelConfig.from_dict(ck["model_cfg"])
        self.cfg.pretrained = False
        self.model = GeoBEV(self.cfg)
        self.model.load_state_dict(ck["model"])
        self.model.to(self.device).eval()
        self.half = half and self.device.type == "cuda"

    def prepare(self, rgb, camera, R, t):
        """Resize to the network input and scale the intrinsics to match."""
        c = self.cfg
        H0, W0 = rgb.shape[:2]
        img = cv2.resize(rgb, (c.input_w, c.input_h), interpolation=cv2.INTER_AREA)
        sx, sy = c.input_w / W0, c.input_h / H0
        # the camera may be described at a different resolution than `rgb`
        kx, ky = W0 / camera.width, H0 / camera.height
        K = np.array([[camera.fx * kx * sx, 0, camera.cx * kx * sx],
                      [0, camera.fy * ky * sy, camera.cy * ky * sy], [0, 0, 1]], np.float32)
        return img, K

    @torch.no_grad()
    def __call__(self, rgb, camera, R=None, t=None, body_pitch=0.0, body_roll=0.0):
        """One frame -> dict of (C, C) float32 numpy grids."""
        if R is None:
            R, t = camera.pose(body_pitch, body_roll)
        img, K = self.prepare(rgb, camera, R, t)
        dev = self.device
        x = torch.from_numpy(img).to(dev).permute(2, 0, 1)[None].float() / 255.0
        Kt = torch.from_numpy(K).to(dev)[None]
        Rt = torch.as_tensor(np.asarray(R), dtype=torch.float32, device=dev)[None]
        tt = torch.as_tensor(np.asarray(t), dtype=torch.float32, device=dev)[None]
        with torch.autocast(dev.type, dtype=torch.float16, enabled=self.half):
            out = self.model(x, Kt, Rt, tt)
        # one device -> host copy (each .cpu() is a sync)
        names = list(out)
        stack = torch.stack([out[k][0].float() for k in names]).cpu().numpy()
        o = dict(zip(names, stack))
        return {"occ": 1 / (1 + np.exp(-o["occ"])), "occ_logit": o["occ"],
                "vis": 1 / (1 + np.exp(-o["vis"])), "obs_h": o["obs_h"], "elev": o["elev"],
                "grade": o["grade"], "sand": 1 / (1 + np.exp(-o["sand"]))}

    def benchmark_method(self):
        """Adapter for drjepa.eval.perception.evaluate (uses the episode's pose)."""
        def f(ep, k):
            g = self(ep.rgb(k), ep.camera, R=ep.R[k], t=ep.t[k])
            g["occ_threshold"] = 0.5
            return g
        return f

"""Export a GeoBEV checkpoint to ONNX (one graph, camera as an INPUT).

    python -m drjepa export --checkpoint runs/geobev/best.pt --out runs/geobev/geobev.onnx

Graph I/O (all float32):
    inputs   image (1, 3, H, W) RGB in [0, 1] at the model input size
             K     (1, 3, 3) intrinsics at that size (pixel-edge convention)
             R     (1, 3, 3) camera -> robot rotation (OpenCV camera axes)
             t     (1, 3)    camera position in the robot frame (m)
    outputs  occ_logit, vis_logit, obs_h, elev, grade, sand_logit   (1, C, C)
             indexed [i = x right, j = z forward]

Because K / R / t are graph inputs, one exported file serves every camera:
nothing is re-exported for a new lens or mount. A JSON manifest next to the
.onnx records the input size, grid geometry and output semantics, so a C++ /
Rust runtime needs no Python. `--verify` diffs onnxruntime against PyTorch
and reports CPU latency (1 and 4 threads) -- the low-power reference point.
"""

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn

from .model import GeoBEV, ModelConfig

OUTPUTS = ("occ_logit", "vis_logit", "obs_h", "elev", "grade", "sand_logit")


class ExportWrapper(nn.Module):
    """Raw-tensor outputs in a fixed order (no dict) for ONNX."""

    def __init__(self, model):
        super().__init__()
        self.m = model
        # torch.onnx.export restores the wrapper's training flag afterwards;
        # a default (train-mode) wrapper would flip every BatchNorm in the
        # model to batch statistics after export
        self.eval()

    def forward(self, image, K, R, t):
        o = self.m(image, K, R, t)
        return o["occ"], o["vis"], o["obs_h"], o["elev"], o["grade"], o["sand"]


def load(checkpoint):
    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = ModelConfig.from_dict(ck["model_cfg"])
    cfg.pretrained = False
    m = GeoBEV(cfg)
    m.load_state_dict(ck["model"])
    return m.eval(), cfg, ck


def example_inputs(cfg):
    W, H = cfg.input_w, cfg.input_h
    K = torch.tensor([[0.8 * W, 0, W / 2], [0, 0.8 * W, H / 2], [0, 0, 1.0]])[None]
    R = torch.diag(torch.tensor([1.0, -1.0, 1.0]))[None]
    t = torch.tensor([[0.0, 0.6, 0.0]])
    return torch.rand(1, 3, H, W), K, R, t


def export(checkpoint, out, opset=17):
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    model, cfg, ck = load(checkpoint)
    w = ExportWrapper(model)
    args = example_inputs(cfg)
    torch.onnx.export(w, args, out, input_names=["image", "K", "R", "t"],
                      output_names=list(OUTPUTS), opset_version=opset, dynamo=False)
    manifest = {
        "format": "drjepa-geobev-v1",
        "onnx": os.path.basename(out),
        "input": {"width": cfg.input_w, "height": cfg.input_h, "layout": "NCHW RGB float [0,1]",
                  "K": "intrinsics at the input size, pixel-edge convention (u spans [u, u+1))",
                  "R": "camera->robot rotation; camera axes x right, y down, z forward",
                  "t": "camera position in the robot frame (m): robot frame x right, y up, z forward, "
                       "origin on the ground under the robot"},
        "grid": {"cells": cfg.cells, "res_m": cfg.res,
                 "x_range_m": [-cfg.cells * cfg.res / 2, cfg.cells * cfg.res / 2],
                 "z_range_m": [0.0, cfg.cells * cfg.res], "index": "[i = x, j = z]"},
        "outputs": {"occ_logit": "obstacle (rigid height >= step) logit; trained with pos_weight 3",
                    "vis_logit": "cell observed by the camera, logit",
                    "obs_h": "rigid obstacle height above terrain (m)",
                    "elev": "terrain elevation relative to the ground under the robot (m)",
                    "grade": "terrain slope magnitude (rise/run)",
                    "sand_logit": "soft ground, logit"},
        "model_cfg": cfg.to_dict(),
        "trained_epoch": ck.get("epoch"),
        # DINOv3 has a few tokens with enormous activations; ONNX Runtime's
        # fused kernels (LayerNorm / attention rewrites) lose them -- measured
        # errors of 1-4 logits and even NaN on real frames, while the unfused
        # graph matches PyTorch to 2e-4. CNN encoders are unaffected.
        "ort_graph_optimization": "disable" if cfg.encoder.startswith("dinov3") else "all",
    }
    with open(os.path.splitext(out)[0] + ".json", "w") as f:
        json.dump(manifest, f, indent=1)
    return model, cfg, args


def verify(model, cfg, args, path):
    import onnxruntime as ort
    model.eval()
    level = (ort.GraphOptimizationLevel.ORT_DISABLE_ALL if cfg.encoder.startswith("dinov3")
             else ort.GraphOptimizationLevel.ORT_ENABLE_ALL)
    with torch.no_grad():
        ref = ExportWrapper(model)(*args)
    feeds = {n: a.numpy() for n, a in zip(["image", "K", "R", "t"], args)}
    res = {}
    for threads in (4, 1):
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.graph_optimization_level = level
        s = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
        out = s.run(None, feeds)
        for _ in range(2):
            s.run(None, feeds)
        t0 = time.perf_counter()
        n = 10
        for _ in range(n):
            s.run(None, feeds)
        res[f"cpu_{threads}t_ms"] = 1000 * (time.perf_counter() - t0) / n
    err = max(float(np.abs(o - r.numpy()).max()) for o, r in zip(out, ref))
    res["max_abs_err"] = err
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args()
    out = a.out or os.path.splitext(a.checkpoint)[0] + ".onnx"
    model, cfg, args = export(a.checkpoint, out)
    print(f"wrote {out} ({os.path.getsize(out) / 1e6:.1f} MB) + manifest")
    if a.verify:
        r = verify(model, cfg, args, out)
        print(f"verify: max |onnx - torch| = {r['max_abs_err']:.2e}; "
              f"CPU latency {r['cpu_4t_ms']:.1f} ms (4 threads), {r['cpu_1t_ms']:.1f} ms (1 thread)")
        if r["max_abs_err"] > 1e-2:
            raise SystemExit("verification FAILED")


if __name__ == "__main__":
    main()

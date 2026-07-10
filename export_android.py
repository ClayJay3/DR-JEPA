"""Export a trained checkpoint as a phone-loadable `.drjepa` bundle.

The bundle is a zip holding three ONNX graphs plus a manifest:

    backbone.onnx   image  (1, 3, S, S) float32 RGB in [0, 1]
                    -> tokens (1, n_tokens, feat_dim)
                    (ImageNet normalization and the exact adaptive
                    token pooling are baked into the graph)
    decoder.onnx    tokens (1, F, n_tokens, feat_dim), motion (1, 2F)
                    -> occ / conf / elev / sand / haz (1, C, C), danger (1,)
                    (raw logits + metres, same as MapDecoder.forward)
    completer.onnx  belief (1, 4, G, G)
                    -> pred (1, 3, G, G) PROBABILITIES
                    (per-channel temperature calibration + sigmoid baked)
    manifest.json   geometry, normalization, and every MapPilot fusion /
                    planning constant, so the app needs no hardcoded
                    numbers that could drift from this repo

The frozen DINOv2 backbone is identical across training runs but is still
packed into every bundle so one file on the phone == one comparable run.

    python export_android.py --checkpoint runs/best.pth
    python export_android.py --checkpoint runs/best.pth --verify
"""

import argparse
import json
import math
import os
import tempfile
import zipfile

import numpy as np
import torch
import torch.nn as nn

from drjepa.config import Config
from drjepa.model import RoverJEPA, Backbone
from drjepa.pilot import MapPilot

OPSET = 17


# --------------------------------------------------------------------------
# Export wrappers
# --------------------------------------------------------------------------
def _adaptive_pool_matrix(n_in, n_out):
    """(n_out, n_in) row-stochastic matrix reproducing adaptive_avg_pool1d.

    adaptive_avg_pool2d does not export to ONNX when the output size is not
    a factor of the input, so the pooling is rewritten as two exact matmuls.
    """
    A = torch.zeros(n_out, n_in)
    for i in range(n_out):
        lo = (i * n_in) // n_out
        hi = -(-((i + 1) * n_in) // n_out)          # ceil division
        A[i, lo:hi] = 1.0 / (hi - lo)
    return A


class BackboneExport(nn.Module):
    """DINOv2 + normalization + token pooling as a single static graph."""

    def __init__(self, backbone: Backbone):
        super().__init__()
        cfg = backbone.cfg
        self.net = backbone.net
        self.register_buffer("mean", backbone.mean.float())
        self.register_buffer("std", backbone.std.float())
        g = cfg.img_size // 14                       # DINOv2 patch grid
        self.register_buffer("pool_r", _adaptive_pool_matrix(g, cfg.pool_rows))
        self.register_buffer("pool_c", _adaptive_pool_matrix(g, cfg.pool_cols))
        self.g = g

    def forward(self, image):
        """(1, 3, S, S) RGB float in [0, 1] -> (1, n_tokens, feat_dim)."""
        x = (image - self.mean) / self.std
        out = self.net.forward_features(x)
        cls = out["x_norm_clstoken"]                          # (1, D)
        patches = out["x_norm_patchtokens"]                   # (1, g*g, D)
        B, N, D = patches.shape
        grid = patches.view(B, self.g, self.g, D).permute(0, 3, 1, 2)
        pooled = torch.matmul(torch.matmul(self.pool_r, grid),
                              self.pool_c.transpose(0, 1))    # (1, D, r, c)
        pooled = pooled.flatten(2).permute(0, 2, 1)           # (1, r*c, D)
        return torch.cat([cls[:, None, :], pooled], dim=1)


class CompleterExport(nn.Module):
    """MapCompleter with temperature calibration + sigmoid baked in."""

    def __init__(self, completer, comp_temp):
        super().__init__()
        self.completer = completer
        self.register_buffer("t", comp_temp.view(1, 3, 1, 1).float())

    def forward(self, belief):
        """(1, 4, G, G) -> (1, 3, G, G) calibrated probabilities."""
        return torch.sigmoid(self.completer(belief) / self.t)


def _export(module, args, path, in_names, out_names):
    """torch.onnx.export with the tracer; static shapes, fixed opset."""
    module.eval()
    torch.onnx.export(module, args, path, input_names=in_names,
                      output_names=out_names, opset_version=OPSET,
                      dynamo=False)


# --------------------------------------------------------------------------
# Manifest
# --------------------------------------------------------------------------
def _manifest(cfg: Config, name):
    """Everything the app needs beyond the graphs, in one JSON blob."""
    mc, sc = cfg.model, cfg.sim
    pilot = {k: getattr(MapPilot, k) for k in (
        "DT", "GPS_GAIN", "LODDS_CLAMP", "OCC_THRESH", "PRIOR_LOGIT",
        "HAZ_PRIOR_LOGIT", "SAND_PRIOR", "GUARD_EVIDENCE", "REPLAN_EVERY",
        "N_ARCS", "ARC_T", "ARC_DT", "PAINT_RANGE_CELLS",
        "POS_EVIDENCE_SCALE", "DECAY", "VO_WINDOW", "VO_GAIN", "VO_MARGIN",
        "VO_MIN_L")}
    return {
        "format": "drjepa-android-v1",
        "name": name,
        "backbone": mc.backbone,
        "img_size": mc.img_size,
        "feat_dim": mc.feat_dim,
        "n_tokens": mc.n_tokens,
        "pool_rows": mc.pool_rows,
        "pool_cols": mc.pool_cols,
        "frame_offsets": list(mc.frame_offsets),
        "wedge_cells": mc.wedge_cells,
        "wedge_res": mc.wedge_res,
        "wedge_range_cells": mc.wedge_range_cells,
        "map_cells": mc.map_cells,
        "comp_cells": mc.comp_cells,
        "comp_res": mc.comp_res,
        "speed_norm": mc.speed_norm,
        "sim_fov_deg": sc.fov_deg,       # training-time camera FOV, for
        "sim_cam_height": sc.cam_height,  # reference / AR ground projection
        "rover_radius": sc.rover_radius,
        "pilot": pilot,
        "files": {"backbone": "backbone.onnx", "decoder": "decoder.onnx",
                  "completer": "completer.onnx"},
    }


# --------------------------------------------------------------------------
# Verification
# --------------------------------------------------------------------------
def _verify(bundle, model, backbone, cfg: Config, comp_temp):
    """Run onnxruntime on random inputs and diff against PyTorch."""
    import onnxruntime as ort

    mc = cfg.model
    with tempfile.TemporaryDirectory() as td:
        with zipfile.ZipFile(bundle) as z:
            z.extractall(td)

        def sess(f):
            return ort.InferenceSession(os.path.join(td, f),
                                        providers=["CPUExecutionProvider"])

        rng = np.random.default_rng(0)
        img = rng.random((1, 3, mc.img_size, mc.img_size), np.float32)
        ref = BackboneExport(backbone)(torch.from_numpy(img)).detach().numpy()
        out = sess("backbone.onnx").run(None, {"image": img})[0]
        e_bb = float(np.abs(out - ref).max())

        F = len(mc.frame_offsets)
        toks = rng.standard_normal((1, F, mc.n_tokens, mc.feat_dim)) \
            .astype(np.float32)
        mot = rng.standard_normal((1, 2 * F)).astype(np.float32)
        with torch.no_grad():
            refs = model.map_decoder(torch.from_numpy(toks),
                                     torch.from_numpy(mot))
        outs = sess("decoder.onnx").run(None, {"tokens": toks, "motion": mot})
        e_dec = max(float(np.abs(o - r.numpy()).max())
                    for o, r in zip(outs, refs))

        belief = rng.random((1, 4, mc.comp_cells, mc.comp_cells), np.float32)
        with torch.no_grad():
            refc = CompleterExport(model.map_completer, comp_temp)(
                torch.from_numpy(belief)).numpy()
        outc = sess("completer.onnx").run(None, {"belief": belief})[0]
        e_comp = float(np.abs(outc - refc).max())

    print(f"verify  max|onnx - torch|  backbone {e_bb:.2e}  "
          f"decoder {e_dec:.2e}  completer {e_comp:.2e}")
    ok = e_bb < 1e-3 and e_dec < 1e-3 and e_comp < 1e-4
    if not ok:
        raise SystemExit("verification FAILED: exported graphs diverge")
    print("verify  OK")


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", default=None,
                    help="output bundle path (default: <checkpoint>.drjepa)")
    ap.add_argument("--verify", action="store_true",
                    help="diff onnxruntime against PyTorch after export")
    ap.add_argument("--img-size", type=int, default=None,
                    help="override backbone input resolution (multiple of "
                         "14). The token pooling absorbs the grid change, "
                         "so smaller = faster on-phone at some sharpness "
                         "cost; the trained weights are unchanged.")
    args = ap.parse_args()

    out = args.out or os.path.splitext(args.checkpoint)[0] + ".drjepa"
    name = os.path.splitext(os.path.basename(out))[0]

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = Config.from_dict(ckpt["config"])
    mc = cfg.model
    if args.img_size:
        assert args.img_size % 14 == 0 and args.img_size >= 14 * mc.pool_rows
        mc.img_size = args.img_size
    model = RoverJEPA(mc)
    model.load_state_dict(ckpt["model"])
    model.eval()
    calib = ckpt.get("comp_calib", {"t": [1.0, 1.0, 1.0]})
    comp_temp = torch.tensor(calib["t"], dtype=torch.float32)
    backbone = Backbone(mc, device="cpu")

    F = len(mc.frame_offsets)
    with tempfile.TemporaryDirectory() as td, torch.no_grad():
        print(f"exporting backbone ({mc.backbone}, {mc.img_size}px) ...")
        _export(BackboneExport(backbone),
                (torch.zeros(1, 3, mc.img_size, mc.img_size),),
                os.path.join(td, "backbone.onnx"), ["image"], ["tokens"])

        print("exporting decoder ...")
        _export(model.map_decoder,
                (torch.zeros(1, F, mc.n_tokens, mc.feat_dim),
                 torch.zeros(1, 2 * F)),
                os.path.join(td, "decoder.onnx"), ["tokens", "motion"],
                ["occ", "conf", "elev", "sand", "haz", "danger"])

        print("exporting completer ...")
        _export(CompleterExport(model.map_completer, comp_temp),
                (torch.zeros(1, 4, mc.comp_cells, mc.comp_cells),),
                os.path.join(td, "completer.onnx"), ["belief"], ["pred"])

        with open(os.path.join(td, "manifest.json"), "w") as f:
            json.dump(_manifest(cfg, name), f, indent=2)

        with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
            for f in ("manifest.json", "backbone.onnx", "decoder.onnx",
                      "completer.onnx"):
                z.write(os.path.join(td, f), f)

    mb = os.path.getsize(out) / 1e6
    print(f"wrote {out} ({mb:.1f} MB)")
    if args.verify:
        _verify(out, model, backbone, cfg, comp_temp)


if __name__ == "__main__":
    main()

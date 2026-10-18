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

The frozen DINOv2/DINOv3 backbone is identical across training runs but is
still packed into every bundle so one file on the phone == one comparable
run.

    python export_android.py --checkpoint runs/best.pth
    python export_android.py --checkpoint runs/best.pth --verify

Notes on DINOv3 (Hugging Face) backbones
-----------------------------------------
Unlike DINOv2, DINOv3's `last_hidden_state` prepends a handful of register
tokens after the CLS token and before the patch tokens. We never hardcode
that count -- we derive how many leading tokens to drop from the known
patch-grid size (`self.g * self.g`), so this works whether there are 0
register tokens (DINOv2) or 4 (DINOv3) without needing per-backbone
special-casing.
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
    """DINOv2/DINOv3 + normalization + token pooling as a single static graph."""

    def __init__(self, backbone: Backbone):
        super().__init__()
        cfg = backbone.cfg
        self.net = backbone.net
        self.register_buffer("mean", backbone.mean.float())
        self.register_buffer("std", backbone.std.float())

        # Dynamically determine patch size from the backbone name
        patch_size = 16 if "16" in cfg.backbone else 14
        g = cfg.img_size // patch_size

        self.register_buffer("pool_r", _adaptive_pool_matrix(g, cfg.pool_rows))
        self.register_buffer("pool_c", _adaptive_pool_matrix(g, cfg.pool_cols))
        self.g = g

    def forward(self, image):
        """(1, 3, S, S) RGB float in [0, 1] -> (1, n_tokens, feat_dim)."""
        x = (image - self.mean) / self.std
        out = self.net(x)

        # Handle different backbone API output formats
        if hasattr(out, "last_hidden_state"):
            # Hugging Face transformers output. DINOv3 prepends register
            # tokens after CLS and before the patch tokens; DINOv2-style
            # HF outputs don't. Derive the split point from the known
            # patch-grid size instead of assuming exactly 1 leading token,
            # so this is correct for either.
            hidden_state = out.last_hidden_state
            n_extra = hidden_state.shape[1] - self.g * self.g
            cls = hidden_state[:, 0]
            patches = hidden_state[:, n_extra:]
        elif isinstance(out, dict) and "x_norm_patchtokens" in out:
            # Original Meta DINOv2 dictionary output
            cls = out["x_norm_clstoken"]
            patches = out["x_norm_patchtokens"]
        elif isinstance(out, torch.Tensor):
            # Standard raw tensor (Batch, 1 + Patches, Dim)
            cls = out[:, 0]
            patches = out[:, 1:]
        else:
            # Fallback if it returns a raw tuple
            hidden_state = out[0]
            cls = hidden_state[:, 0]
            patches = hidden_state[:, 1:]

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
# Quantization
# --------------------------------------------------------------------------
def _quantize_backbone(bb_path):
    """Dynamic int8 quantize the backbone's MatMuls, excluding attention
    projections.

    Dynamic quantization computes each activation tensor's int8 scale from
    that tensor's own runtime min/max. ViT backbones (DINOv2/DINOv3
    especially) have a small number of very high-norm "outlier" tokens; a
    single outlier stretches the whole int8 range for that tensor, crushing
    the other ~99% of normal-magnitude values into a handful of levels near
    zero. This shows up worst in the attention QKV/output projections,
    where outlier-token activations propagate directly. We whitelist "/mlp/"
    MatMuls for quantization and leave everything else (attention
    projections, and any unnamed/non-per-layer nodes) fp32 -- MLP is the
    bulk of the FLOPs, so this keeps most of the speedup while avoiding
    both the outlier-token accuracy loss and NaN output previously seen
    when a blacklist let unidentified nodes slip through and get quantized.

    MatMul ONLY (never Conv): quantizing Conv emits ConvInteger, which the
    ONNX Runtime Android build does not implement (bundle fails to load
    on-device); the backbone's lone conv is the patch embed, a negligible
    fraction of the compute.

    IMPORTANT: ORT's own tooling recommends running shape-inference
    preprocessing before quantize_dynamic (the "Please consider to run
    pre-processing" warning). Skipping it can produce broken scale /
    zero-point computation on some nodes -- up to and including NaN
    outputs -- so we always run it first.
    """
    import onnx
    from onnxruntime.quantization import quantize_dynamic, QuantType
    from onnxruntime.quantization.shape_inference import quant_pre_process

    pre_path = bb_path + ".pre.onnx"
    try:
        # skip_symbolic_shape=True: ORT's SymbolicShapeInference has a known
        # bug on nested If subgraphs (asserts in is_sequence/_fuse_tensor_type).
        # The graph has an If node from SDPA's data-dependent is_causal branch
        # (see the TracerWarning at export time). Plain ONNX shape inference
        # is weaker but doesn't touch that code path.
        quant_pre_process(bb_path, pre_path, skip_symbolic_shape=True)
    except Exception as e:
        print(f"  quant_pre_process failed ({e!r}); quantizing without "
              f"preprocessing -- verify carefully, this is the path that "
              f"produced NaN before")
        import shutil
        shutil.copy(bb_path, pre_path)

    model = onnx.load(pre_path)
    matmul_names = [n.name for n in model.graph.node if n.op_type == "MatMul"]
    # Whitelist, not blacklist: only quantize MatMuls explicitly inside an
    # MLP block ("/mlp/" in the node path). Everything else -- attention
    # projections (q/k/v/o_proj, the QK^T and attn@V matmuls) AND any
    # top-level/unnamed MatMuls outside the per-layer stack (seen in
    # practice: two unnamed nodes, likely a pooling head, that are NOT
    # under any layer.N/ path) -- stays fp32. A blacklist of attention
    # keywords is fragile: "proj" also matches mlp/up_proj and
    # mlp/down_proj, and it says nothing about unnamed nodes outside the
    # transformer stack, which is exactly what produced NaN output when
    # they ended up quantized instead of the intended attention layers.
    quantized = [n for n in matmul_names if "/mlp/" in n]
    excluded = [n for n in matmul_names if n not in quantized]

    print(f"  {len(matmul_names)} MatMul nodes after preprocessing:")
    for n in matmul_names:
        tag = "quantize" if n in quantized else "EXCLUDE"
        print(f"    [{tag:8s}] {n}")
    print(f"  quantizing {len(quantized)}/{len(matmul_names)} MatMuls "
          f"(mlp only)")
    if len(quantized) == 0:
        print("  WARNING: no MatMuls matched '/mlp/' -- whitelist doesn't "
              "match this graph's naming, quantization will be a no-op")

    quantize_dynamic(pre_path, bb_path, weight_type=QuantType.QInt8,
                     per_channel=True,
                     op_types_to_quantize=["MatMul"],
                     nodes_to_exclude=excluded)
    os.remove(pre_path)



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
def _verify(bundle, model, backbone, cfg: Config, comp_temp,
            quantized=False):
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
        e_bb_mean = float(np.abs(out - ref).mean())

        F = len(mc.frame_offsets)
        toks = rng.standard_normal((1, F, mc.n_tokens, mc.feat_dim)) \
            .astype(np.float32)
        mot = rng.standard_normal((1, 2 * F)).astype(np.float32)
        cam = rng.standard_normal((1, mc.cam_dim)).astype(np.float32)
        with torch.no_grad():
            refs = model.map_decoder(torch.from_numpy(toks),
                                     torch.from_numpy(mot),
                                     torch.from_numpy(cam))
        outs = sess("decoder.onnx").run(
            None, {"tokens": toks, "motion": mot, "camera": cam})
        e_dec = max(float(np.abs(o - r.numpy()).max())
                    for o, r in zip(outs, refs))

        belief = rng.random((1, 4, mc.comp_cells, mc.comp_cells), np.float32)
        with torch.no_grad():
            refc = CompleterExport(model.map_completer, comp_temp)(
                torch.from_numpy(belief)).numpy()
        outc = sess("completer.onnx").run(None, {"belief": belief})[0]
        e_comp = float(np.abs(outc - refc).max())

    rel = e_bb_mean / max(float(np.abs(ref).mean()), 1e-6)
    print(f"verify  max|onnx - torch|  backbone {e_bb:.2e} "
          f"(mean {e_bb_mean:.2e}, {rel * 100:.1f}% rel)  "
          f"decoder {e_dec:.2e}  completer {e_comp:.2e}")
    # DINOv3/Hugging Face models show higher FP32 max divergence due to a
    # small number of high-norm outlier tokens amplifying SDPA-vs-ONNXRuntime
    # kernel differences, but mean rel error stays low (~1-2%). Dynamic int8
    # quantization is more sensitive to those same outlier tokens (they blow
    # out the per-tensor min/max scale), so its threshold is looser even
    # with attention projections excluded from quantization.
    bb_ok = (rel < 0.40 and e_bb < 3.0) if quantized else (rel < 0.05 and e_bb < 3.0)
    if not (bb_ok and e_dec < 1e-3 and e_comp < 1e-4):
        raise SystemExit("verification FAILED: exported graphs diverge")
    print("verify  OK" + (" (backbone int8: remaining error is quantization "
                          "noise on outlier tokens; A/B it against the fp32 "
                          "bundle in the field)"
                          if quantized else ""))


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
                         "patch size). The token pooling absorbs the grid change, "
                         "so smaller = faster on-phone at some sharpness "
                         "cost; the trained weights are unchanged.")
    ap.add_argument("--quantize", action="store_true",
                    help="dynamic-int8-quantize the backbone MatMuls, "
                         "excluding attention projections "
                         "(~2-2.5x faster on ARMv8.2+/ARMv9 phones; the "
                         "small decoder/completer stay fp32)")
    args = ap.parse_args()

    out = args.out or os.path.splitext(args.checkpoint)[0] + ".drjepa"
    name = os.path.splitext(os.path.basename(out))[0]

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = Config.from_dict(ckpt["config"])
    mc = cfg.model

    # 1. Apply overrides FIRST
    if args.img_size:
        patch_size = 16 if "16" in mc.backbone else 14
        min_size = patch_size * mc.pool_rows
        if args.img_size % patch_size != 0 or args.img_size < min_size:
            raise ValueError(f"--img-size must be a multiple of {patch_size} and at least {min_size} (to feed the decoder grid). Got {args.img_size}.")
        mc.img_size = args.img_size

    # 2. Instantiate models using the updated config
    model = RoverJEPA(mc)
    model.load_state_dict(ckpt["model"])
    model.eval()
    calib = ckpt.get("comp_calib", {"t": [1.0, 1.0, 1.0]})
    comp_temp = torch.tensor(calib["t"], dtype=torch.float32)
    backbone = Backbone(mc, device="cpu")

    F = len(mc.frame_offsets)
    with tempfile.TemporaryDirectory() as td, torch.no_grad():
        print(f"exporting backbone ({mc.backbone}, {mc.img_size}px) ...")
        bb_path = os.path.join(td, "backbone.onnx")
        _export(BackboneExport(backbone),
                (torch.zeros(1, 3, mc.img_size, mc.img_size),),
                bb_path, ["image"], ["tokens"])
        if args.quantize:
            print("quantizing backbone (dynamic int8, per-channel) ...")
            _quantize_backbone(bb_path)

        print("exporting decoder ...")
        # `camera` is a real input now: f_norm / height / pitch of the phone
        # mount (config.camera_features). The app must supply it per frame --
        # a decoder that cannot see its own optics silently reverts to the
        # one camera it was trained on.
        _export(model.map_decoder,
                (torch.zeros(1, F, mc.n_tokens, mc.feat_dim),
                 torch.zeros(1, 2 * F),
                 torch.zeros(1, mc.cam_dim)),
                os.path.join(td, "decoder.onnx"),
                ["tokens", "motion", "camera"],
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
        _verify(out, model, backbone, cfg, comp_temp,
                quantized=args.quantize)


if __name__ == "__main__":
    main()
"""ONNX export round trip (skipped without onnxruntime)."""

import numpy as np
import pytest
import torch

ort = pytest.importorskip("onnxruntime")

from drjepa.perception.export import OUTPUTS, ExportWrapper, example_inputs  # noqa: E402
from drjepa.perception.model import GeoBEV, ModelConfig  # noqa: E402


def test_onnx_matches_torch(tmp_path):
    cfg = ModelConfig(pretrained=False, input_w=256, input_h=160)
    m = GeoBEV(cfg).eval()
    w = ExportWrapper(m)
    args = example_inputs(cfg)
    path = str(tmp_path / "g.onnx")
    torch.onnx.export(w, args, path, input_names=["image", "K", "R", "t"],
                      output_names=list(OUTPUTS), opset_version=17, dynamo=False)
    assert not m.training                       # export must not flip BatchNorm back to train
    with torch.no_grad():
        ref = w(*args)
    s = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    out = s.run(None, {n: a.numpy() for n, a in zip(["image", "K", "R", "t"], args)})
    for o, r in zip(out, ref):
        assert np.abs(o - r.numpy()).max() < 1e-3


def test_onnx_perceiver_matches_torch_perceiver(tmp_path):
    from drjepa.camera import Camera
    from drjepa.perception.export import export
    from drjepa.perception.infer import Perceiver
    from drjepa.perception.onnx_runtime import OnnxPerceiver
    cfg = ModelConfig(pretrained=False, input_w=256, input_h=160)
    ck = tmp_path / "m.pt"
    torch.save({"model": GeoBEV(cfg).state_dict(), "model_cfg": cfg.to_dict(), "epoch": 0}, ck)
    export(str(ck), str(tmp_path / "m.onnx"))
    cam = Camera.from_fov(320, 200, 95, mount_height=0.5, mount_pitch=-5)
    rgb = (np.random.default_rng(0).random((200, 320, 3)) * 255).astype(np.uint8)
    a = Perceiver(str(ck), device="cpu", half=False)(rgb, cam)
    b = OnnxPerceiver(str(tmp_path / "m.onnx"))(rgb, cam)
    for k in ("occ", "vis", "obs_h", "elev", "grade", "sand"):
        assert np.abs(a[k] - b[k]).max() < 1e-3, k

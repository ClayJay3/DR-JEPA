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

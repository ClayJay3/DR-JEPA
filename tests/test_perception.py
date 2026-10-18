"""GeoBEV: geometric lift lands on the right cells; shapes; flip consistency."""

import numpy as np
import pytest
import torch

from drjepa.camera import Camera
from drjepa.perception.model import GeoBEV, ModelConfig


@pytest.fixture(scope="module")
def model():
    return GeoBEV(ModelConfig(pretrained=False)).eval()


def test_forward_shapes(model):
    B, cfg = 2, model.cfg
    img = torch.rand(B, 3, cfg.input_h, cfg.input_w)
    K = torch.tensor([[300.0, 0, 256], [0, 300, 160], [0, 0, 1]]).expand(B, 3, 3)
    R = torch.diag(torch.tensor([1.0, -1, 1])).expand(B, 3, 3)
    t = torch.tensor([0.0, 0.6, 0.0]).expand(B, 3)
    out = model(img, K, R, t)
    for k in ("vis", "occ", "obs_h", "elev", "grade", "sand"):
        assert out[k].shape == (B, cfg.cells, cfg.cells)
    assert (out["obs_h"] >= 0).all() and (out["grade"] >= 0).all()


@pytest.mark.parametrize("hfov,h,pitch,roll", [(90, 0.6, 0, 0), (120, 0.35, -15, 4), (60, 2.0, -25, -5)])
def test_lift_samples_the_projected_pixel(model, hfov, h, pitch, roll):
    """A one-hot feature at the projection of a cell's ground point must be
    picked up by that cell at height 0 -- for any camera rig."""
    cfg = model.cfg
    cam = Camera.from_fov(cfg.input_w, cfg.input_h, hfov, mount_height=h,
                          mount_pitch=pitch, mount_roll=roll)
    R, t = cam.pose()
    ci, cj = 30, 14                                   # x = +3.25 m, z = 7.25 m
    p = np.array([[(ci + 0.5) * cfg.res - cfg.cells * cfg.res / 2, 0.0, (cj + 0.5) * cfg.res]])
    uv, z = cam.project(p, R, t)
    assert z[0] > 0
    fh, fw = cfg.input_h // 8, cfg.input_w // 8
    u8, v8 = int(uv[0, 0] / 8), int(uv[0, 1] / 8)
    if not (0 <= u8 < fw and 0 <= v8 < fh):
        pytest.skip("cell outside this rig's view")
    feat = torch.zeros(1, cfg.feat_ch, fh, fw)
    feat[0, :, v8, u8] = 1.0
    x = model.lift(feat, torch.tensor(cam.K(), dtype=torch.float32)[None],
                   torch.tensor(R, dtype=torch.float32)[None], torch.tensor(t, dtype=torch.float32)[None],
                   (cfg.input_h, cfg.input_w))
    Kh = len(cfg.heights)
    x = x.view(1, cfg.feat_ch + 3, Kh, cfg.cells, cfg.cells)   # (channel, height) layout
    k0 = list(cfg.heights).index(0.0)
    resp = x[0, 0, k0]                                # first feature channel at height 0
    assert resp[ci, cj] > 0.05
    # one stride-8 feature pixel spans several cells along the ray at range,
    # so the peak may sit a few cells away -- but never anywhere else
    am = np.unravel_index(int(resp.argmax()), resp.shape)
    assert abs(am[0] - ci) <= 3 and abs(am[1] - cj) <= 3


def test_implicit_lift_variant_runs():
    m = GeoBEV(ModelConfig(pretrained=False, lift="implicit")).eval()
    cfg = m.cfg
    out = m(torch.rand(1, 3, cfg.input_h, cfg.input_w),
            torch.tensor([[[300.0, 0, 256], [0, 300, 160], [0, 0, 1]]]),
            torch.diag(torch.tensor([1.0, -1, 1]))[None], torch.tensor([[0.0, 0.6, 0.0]]))
    assert out["occ"].shape == (1, cfg.cells, cfg.cells)

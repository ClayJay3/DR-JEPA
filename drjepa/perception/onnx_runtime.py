"""Torch-free perception runtime: run an exported GeoBEV .onnx with ONNX Runtime.

    from drjepa.perception.onnx_runtime import OnnxPerceiver
    perceiver = OnnxPerceiver("runs/geobev/geobev.onnx")            # CPU
    perceiver = OnnxPerceiver("geobev.onnx", providers=["CUDAExecutionProvider"])
    grids = perceiver(rgb, camera, body_pitch=imu_pitch, body_roll=imu_roll)

Same call signature and outputs as drjepa.perception.infer.Perceiver, so it
drops into the Navigator unchanged (Navigator("geobev.onnx") selects it).
A robot install then needs only numpy, OpenCV and onnxruntime.
"""

import json
import os

import cv2
import numpy as np


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class OnnxPerceiver:
    """ONNX Runtime twin of Perceiver (reads the export manifest)."""

    def __init__(self, path, providers=None, threads=None):
        import onnxruntime as ort
        man = os.path.splitext(path)[0] + ".json"
        with open(man) as f:
            self.manifest = json.load(f)
        so = ort.SessionOptions()
        if threads:
            so.intra_op_num_threads = int(threads)
        self.session = ort.InferenceSession(path, so, providers=providers or ["CPUExecutionProvider"])
        self.w = self.manifest["input"]["width"]
        self.h = self.manifest["input"]["height"]
        self.outputs = [o.name for o in self.session.get_outputs()]

    def __call__(self, rgb, camera, R=None, t=None, body_pitch=0.0, body_roll=0.0):
        if R is None:
            R, t = camera.pose(body_pitch, body_roll)
        H0, W0 = rgb.shape[:2]
        img = cv2.resize(rgb, (self.w, self.h), interpolation=cv2.INTER_AREA)
        sx, sy = self.w / W0, self.h / H0
        kx, ky = W0 / camera.width, H0 / camera.height
        K = np.array([[camera.fx * kx * sx, 0, camera.cx * kx * sx],
                      [0, camera.fy * ky * sy, camera.cy * ky * sy], [0, 0, 1]], np.float32)
        feeds = {"image": (img.astype(np.float32) / 255.0).transpose(2, 0, 1)[None],
                 "K": K[None], "R": np.asarray(R, np.float32)[None],
                 "t": np.asarray(t, np.float32).reshape(1, 3)}
        o = dict(zip(self.outputs, (x[0] for x in self.session.run(None, feeds))))
        return {"occ": _sigmoid(o["occ_logit"]), "occ_logit": o["occ_logit"],
                "vis": _sigmoid(o["vis_logit"]), "obs_h": o["obs_h"], "elev": o["elev"],
                "grade": o["grade"], "sand": _sigmoid(o["sand_logit"])}

"""DR-JEPA: camera-only rover navigation for any robot, any camera.

One forward camera (+ IMU attitude and a pose source) in; drive commands
and a persistent metric map of the world out.

    perception  GeoBEV: image + camera geometry -> robot-frame ground grids
                (obstacle probability / height, elevation, slope, sand,
                visibility); the view transform projects the real camera,
                so one model serves every lens and mount
    nav         belief-map fusion, A* planning, arc control -> Navigator
    sim         GPU-rendered domain-randomized worlds + rover physics
    data        real recordings (ZED stereo) -> labelled episodes
    eval        open-loop perception and closed-loop driving benchmarks

Quick start:

    from drjepa import Navigator, Camera, RobotProfile
    nav = Navigator("runs/geobev/best.pt",
                    camera=Camera.from_fov(1280, 720, 101.5, mount_height=0.40),
                    robot=RobotProfile())
    cmd = nav.step(rgb, sensors)       # see drjepa.nav.navigator
"""

from .camera import Camera  # noqa: F401
from .config import GridConfig, RobotProfile, SimConfig  # noqa: F401

__version__ = "13.0.0"


def __getattr__(name):
    # heavy imports (torch) only when used
    if name == "Navigator":
        from .nav.navigator import Navigator
        return Navigator
    if name == "Perceiver":
        from .perception.infer import Perceiver
        return Perceiver
    raise AttributeError(name)

"""Domain-randomized rover simulator: GPU-rendered worlds, physics, sensors.

    world    terrain heightfield + props (+ colliders)
    rover    RoverSim: one episode (physics, sensors, camera rig, rendering)
    render   moderngl renderer (headless EGL)
    style    per-episode appearance + camera-sensor effects
    assets   procedural meshes
    expert   privileged A* + arc-sampling driver
"""

from .rover import RoverSim, random_camera  # noqa: F401

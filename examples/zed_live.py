"""Live navigation with a ZED 2i: camera, IMU and positional tracking from
the ZED SDK, drive commands out.

    python examples/zed_live.py --model models/geobev.onnx --goal-x 0 --goal-z 30 --mount-height 0.40

Uses only the LEFT rectified image (DR-JEPA is monocular; stereo is not
needed at run time). The ZED's gravity-aligned positional tracking (IMU +
visual odometry) supplies both the camera attitude -- passed to the
navigator as the exact camera -> robot pose, the same geometry the real
training labels were built with -- and the robot position, so this runs
without GPS. Goals are metres in the tracking frame: x = right of the
start heading, z = ahead.

Replace `send_drive()` with your robot's drive interface (Autonomy_Software,
ROS, a serial link, ...). Needs the ZED SDK + pyzed (tools/zed has a Docker
image for hosts the SDK does not support).
"""

import argparse
import math
import time

import numpy as np

from drjepa import Camera, Navigator, RobotProfile
from drjepa.data.recordings import robot_frame_pose


def send_drive(throttle, steer):
    """Stub: forward the command to the robot."""
    print(f"throttle {throttle:+.2f}  steer {steer:+.2f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help=".onnx (no torch) or .pt checkpoint")
    ap.add_argument("--goal-x", type=float, required=True)
    ap.add_argument("--goal-z", type=float, required=True)
    ap.add_argument("--mount-height", type=float, default=0.40, help="camera above ground (m)")
    ap.add_argument("--hz", type=float, default=10.0)
    a = ap.parse_args()

    import pyzed.sl as sl
    zed = sl.Camera()
    init = sl.InitParameters()
    init.camera_resolution = sl.RESOLUTION.HD720
    init.coordinate_units = sl.UNIT.METER
    init.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
    # positional tracking needs the SDK's depth; the navigator never uses it
    init.depth_mode = sl.DEPTH_MODE.PERFORMANCE
    if zed.open(init) != sl.ERROR_CODE.SUCCESS:
        raise SystemExit("could not open the ZED")
    tracking = sl.PositionalTrackingParameters()
    tracking.set_gravity_as_origin = True
    zed.enable_positional_tracking(tracking)

    calib = zed.get_camera_information().camera_configuration.calibration_parameters.left_cam
    res = zed.get_camera_information().camera_configuration.resolution
    camera = Camera(res.width, res.height, calib.fx, calib.fy, calib.cx, calib.cy,
                    mount_height=a.mount_height)
    nav = Navigator(a.model, camera=camera, robot=RobotProfile())

    image, pose = sl.Mat(), sl.Pose()
    runtime = sl.RuntimeParameters()
    last = None
    period = 1.0 / a.hz
    while True:
        t0 = time.time()
        if zed.grab(runtime) != sl.ERROR_CODE.SUCCESS:
            continue
        if zed.get_position(pose, sl.REFERENCE_FRAME.WORLD) != sl.POSITIONAL_TRACKING_STATE.OK:
            continue
        zed.retrieve_image(image, sl.VIEW.LEFT)
        rgb = np.ascontiguousarray(image.get_data()[:, :, 2::-1])        # BGRA -> RGB
        p = pose.get_translation().get()
        q = np.array(pose.get_orientation().get())       # x, y, z, w (gravity-aligned)
        R, t = robot_frame_pose(q, a.mount_height)       # exact camera -> robot pose
        x, z = float(p[0]), float(-p[2])                 # planar: -z_gl = forward at start
        fwd_x = 2 * (q[0] * q[2] + q[1] * q[3])          # third column of the rotation:
        fwd_z = 1 - 2 * (q[0] ** 2 + q[1] ** 2)          # the camera looks along -z_gl
        heading = math.degrees(math.atan2(-fwd_x, fwd_z)) % 360
        speed = 0.0 if last is None else float(np.hypot(x - last[0], z - last[1]) / max(t0 - last[2], 1e-3))
        last = (x, z, t0)

        cmd = nav.step(rgb, {"x": x, "z": z, "heading": heading, "speed": speed,
                             "goal_x": a.goal_x, "goal_z": a.goal_z}, R=R, t=t)
        send_drive(cmd.throttle, cmd.steer)
        if np.hypot(a.goal_x - x, a.goal_z - z) < 4.0:
            print("goal reached")
            break
        time.sleep(max(0.0, period - (time.time() - t0)))


if __name__ == "__main__":
    main()

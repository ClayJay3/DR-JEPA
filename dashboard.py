"""Navigation diagnostics dashboard (1920 x 1080 video or live window).

Shows, every control step: the camera with AR overlays, all six perception
heads, ground truth with hit / false-alarm / miss and error maps, the fused
belief map with every candidate arc scored, live accuracy, controller
state, telemetry and per-stage latency.

    # closed loop in the simulator (exact ground truth)
    python dashboard.py --checkpoint runs/geobev/best.pt --frames 600 --out dash.mp4
    python dashboard.py --checkpoint runs/geobev/best.pt --camera zed --scenario camp --show

    # replay a real labelled recording (stereo ground truth, recorded poses;
    # the controller's commands are shown but not executed)
    python dashboard.py --checkpoint runs/geobev/best.pt --replay "recordings/zed/ZED_2i_31237348_(1)"
"""

import argparse
import math

import cv2
import numpy as np

from drjepa.nav.navigator import Navigator
from drjepa.viz.dashboard import Dashboard


def run_sim(a, writer_fn):
    from drjepa.config import GridConfig
    from drjepa.eval.closed_loop import make_camera
    from drjepa.labels import sim_targets
    from drjepa.sim import RoverSim

    sim = RoverSim(seed=a.seed, scenario=a.scenario, camera=make_camera(a.camera))
    nav = Navigator(a.checkpoint, camera=sim.camera)
    dash = Dashboard(nav, "SIM")
    grid = GridConfig()
    goals = 0
    for n in range(a.frames):
        rgb, depth, cls, (R_true, t_true) = sim.render(gt=True)
        gt = sim_targets(sim, depth, cls, R_true, t_true, grid)
        s = sim.sensor_readout()
        cmd = nav.step(rgb, s)
        R, t = sim.camera.pose(s["pitch"], s["roll"])        # what the robot believes
        info = {"scenario": sim.scenario, "goals": goals, "contacts": sim.collision_count,
                "pose_err": float(np.hypot(nav.pose[0] - sim.x, nav.pose[1] - sim.z))}
        if writer_fn(dash.render(rgb, cmd, s, R, t, gt, info)):
            break
        step = sim.step(cmd.throttle, cmd.steer)
        if step["tipped"]:
            print(f"[{n}] tipped over")
            break
        if step["reached"] or step["no_progress"] or step["timeout"]:
            goals += int(step["reached"])
            print(f"[{n}] {'goal reached' if step['reached'] else 'gave up'} ({goals} total)")
            sim.respawn_goal()
            nav.camera = sim.camera
    sim.close()


def run_replay(a, writer_fn):
    from drjepa.data.recordings import Recording, quat_to_mat
    from drjepa.eval.perception import Episode

    ep = Episode(a.replay)
    rec = Recording(a.replay)
    nav = Navigator(a.checkpoint, camera=ep.camera)
    dash = Dashboard(nav, "REAL · " + ep.meta.get("sensor", "replay"))
    ids = ep.frame_ids
    # planar track in the VIO world: x = x_gl, z = -z_gl (forward-positive)
    pos = np.array([[rec.position(int(i))[0], -rec.position(int(i))[2]] for i in ids])
    tt = rec.rows[ids, 1] / 1e9
    along = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(pos, axis=0), axis=1))]
    for k in range(0, min(len(ep), a.frames)):
        Rq = quat_to_mat(*rec.quat(int(ids[k])))
        fwd = -Rq[:, 2]
        heading = math.degrees(math.atan2(fwd[0], -fwd[2])) % 360
        j = int(np.searchsorted(along, along[k] + 30.0))
        goal = pos[min(j, len(pos) - 1)]
        ahead = np.array([math.sin(math.radians(heading)), math.cos(math.radians(heading))])
        # the recorded track can run behind the camera (a rear-facing
        # camera): then aim straight ahead of the camera instead
        if np.linalg.norm(goal - pos[k]) < 5 or np.dot(goal - pos[k], ahead) < 0:
            goal = pos[k] + 30 * ahead
        dt = max(tt[k] - tt[k - 1], 1e-3) if k else 1.0
        speed = float(np.linalg.norm(pos[k] - pos[k - 1]) / dt) if k else 0.0
        s = {"x": pos[k, 0], "z": pos[k, 1], "heading": heading, "speed": speed,
             "goal_x": goal[0], "goal_z": goal[1]}
        R, t = ep.R[k], ep.t[k]
        rgb = ep.rgb(k)
        cmd = nav.step(rgb, s, R=R, t=t)
        gt = {"vis": ep.vis[k], "occ": ep.occ[k], "elev": ep.elev[k], "grade": ep.grade[k]}
        info = {"scenario": "replay", "frame": int(ids[k])}
        if writer_fn(dash.render(rgb, cmd, s, R, t, gt, info)):
            break


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--replay", default=None, help="labelled real recording directory")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--scenario", default=None)
    ap.add_argument("--camera", default="random")
    ap.add_argument("--frames", type=int, default=600)
    ap.add_argument("--out", default="dashboard.mp4")
    ap.add_argument("--fps", type=float, default=None)
    ap.add_argument("--snapshot", default=None, help="also save the last frame as a PNG")
    ap.add_argument("--show", action="store_true")
    a = ap.parse_args()

    state = {"writer": None, "last": None}
    fps = a.fps or (4.0 if a.replay else 10.0)

    def write(frame):
        if state["writer"] is None:
            state["writer"] = cv2.VideoWriter(a.out, cv2.VideoWriter_fourcc(*"mp4v"), fps,
                                              (frame.shape[1], frame.shape[0]))
        bgr = np.ascontiguousarray(frame[:, :, ::-1])
        state["writer"].write(bgr)
        state["last"] = bgr
        if a.show:
            cv2.imshow("DR-JEPA diagnostics", bgr)
            return cv2.waitKey(1) == ord("q")
        return False

    (run_replay if a.replay else run_sim)(a, write)
    if state["writer"] is not None:
        state["writer"].release()
    if a.snapshot and state["last"] is not None:
        cv2.imwrite(a.snapshot, state["last"])
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()

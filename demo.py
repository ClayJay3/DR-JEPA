"""Endless closed-loop demo: the trained navigator drives a simulated rover
forever (a new goal and a fresh obstacle field spawn on every arrival) and
records camera + belief-map HUD video.

    python demo.py --checkpoint runs/geobev/best.pt                     # random camera rig
    python demo.py --checkpoint runs/geobev/best.pt --camera zed --frames 3000 --show
"""

import argparse

import cv2

from drjepa.eval.closed_loop import CAMERAS, make_camera
from drjepa.nav.navigator import Navigator
from drjepa.sim import RoverSim
from drjepa.viz.hud import compose_frame


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--camera", default="random", help="random | " + " | ".join(CAMERAS))
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--frames", type=int, default=1500, help="0 = endless")
    ap.add_argument("--out", default="demo.mp4")
    ap.add_argument("--show", action="store_true")
    a = ap.parse_args()

    sim = RoverSim(seed=a.seed, camera=make_camera(a.camera))
    nav = Navigator(a.checkpoint, camera=sim.camera)
    writer = None
    goals = n = 0
    while not a.frames or n < a.frames:
        rgb = sim.render()
        cmd = nav.step(rgb, sim.sensor_readout())
        info = sim.step(cmd.throttle, cmd.steer)
        frame = compose_frame(rgb, nav, sim, cmd.throttle, cmd.steer)
        cv2.putText(frame, f"goals {goals}", (8, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 1, cv2.LINE_AA)
        if writer is None:
            writer = cv2.VideoWriter(a.out, cv2.VideoWriter_fourcc(*"mp4v"), 10.0,
                                     (frame.shape[1], frame.shape[0]))
        writer.write(frame[:, :, ::-1])
        if a.show:
            cv2.imshow("DR-JEPA", frame[:, :, ::-1])
            if cv2.waitKey(1) == ord("q"):
                break
        if info["tipped"]:
            print(f"[{n}] tipped over after {goals} goals")
            break
        if info["reached"] or info["no_progress"] or info["timeout"]:
            goals += int(info["reached"])
            print(f"[{n}] {'goal reached' if info['reached'] else 'gave up on goal'} "
                  f"({goals} total, {sim.collision_count} contacts)")
            sim.respawn_goal()
        n += 1
    writer.release()
    t = nav.timing_ms()
    print(f"{n} frames, {goals} goals, {sim.collision_count} contacts; per-step ms: "
          + ", ".join(f"{k} {v:.1f}" for k, v in t.items()))


if __name__ == "__main__":
    main()

"""Live closed-loop test: the trained model drives forever in an endless
domain-randomized world. Each time it reaches the goal, a new goal and a
fresh obstacle field are spawned around it.

    python live_inference_test.py --checkpoint runs/best.pth
    python live_inference_test.py --checkpoint runs/best.pth --headless --frames 1200
"""

import argparse

import cv2
import numpy as np

from drjepa.simulator import RoverSim, SimConfig
from drjepa.pilot import Pilot, MapPilot, draw_hud


def main():
    """Run the endless closed-loop demo and record it to a video file."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--pilot", choices=["map", "bc"], default="map")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--output_video", default="live_run.mp4")
    ap.add_argument("--headless", action="store_true", help="no window, just record")
    ap.add_argument("--frames", type=int, default=0, help="stop after N frames (0 = endless)")
    ap.add_argument("--no_shield", action="store_true")
    args = ap.parse_args()

    if args.pilot == "map":
        pilot = MapPilot(args.checkpoint)
    else:
        pilot = Pilot(args.checkpoint, shield=not args.no_shield)
    sim = RoverSim(SimConfig(), seed=args.seed)
    writer = cv2.VideoWriter(args.output_video, cv2.VideoWriter_fourcc(*"mp4v"),
                             10.0, (sim.cfg.img_w, sim.cfg.img_h))
    print(f"Recording -> {args.output_video}  (q to quit)" if not args.headless
          else f"Headless run, recording -> {args.output_video}")

    goals = 0
    collisions_at = sim.collision_count
    n = 0
    while True:
        frame = sim.render()
        if isinstance(pilot, MapPilot):
            out = pilot.step(frame, sim.sensor_readout())
        else:
            dist, rel = sim.goal_vector_measured()
            out = pilot.step(frame, dist, rel, sim.meas["speed"])
        info = sim.step(out["throttle"], out["steer"])

        hud = draw_hud(frame, out["throttle"], out["steer"], out["danger"],
                       out["risk"], sim.goal_dist_true(),
                       extra=f"goals {goals} | contacts {sim.collision_count} | {sim.scenario}")
        if isinstance(pilot, MapPilot):
            mv = pilot.map_view()
            hud[8:8 + mv.shape[0], -mv.shape[1] - 8:-8] = mv
        writer.write(hud)
        if not args.headless:
            cv2.imshow("DR-JEPA live", cv2.resize(hud, (512, 512)))
            if cv2.waitKey(1) == ord("q"):
                break

        if info["tipped"]:
            print(f"[{n:5d}] TIPPED OVER -- run ends here "
                  f"(goals reached: {goals})")
            break
        if info["reached"]:
            goals += 1
            print(f"[{n:5d}] goal #{goals} reached "
                  f"(contacts so far: {sim.collision_count})")
            sim.respawn_goal()
        elif info["timeout"]:
            print(f"[{n:5d}] goal timed out, respawning "
                  f"(dist was {sim.goal_dist_true():.0f} m)")
            sim.respawn_goal()

        n += 1
        if args.frames and n >= args.frames:
            break

    writer.release()
    cv2.destroyAllWindows()
    print(f"Frames: {n}, goals reached: {goals}, "
          f"contact events: {sim.collision_count - collisions_at}")


if __name__ == "__main__":
    main()

# DR-JEPA Android test rig — the phone is the rover

Walk around with your phone and watch the v12 navigator perceive, remember,
and plan against the real world. The app runs the full MapPilot loop
on-device:

- **camera** → frozen DINOv2 → MapDecoder terrain wedge (ONNX Runtime)
- **GPS + compass** → complementary pose filter (same gains, dt-scaled)
- wedges fuse into the **persistent log-odds belief map** (occupancy,
  elevation, sand, tip-hazard; VO scan-matching included)
- **A\*** replans on the believed map; the arc controller says what it
  *would* command (throttle/steer HUD — you are the actuator)
- the **map-space JEPA ghost layer** shows what the completer imagines in
  unseen cells, same violet/grey convention as `fsd_viz`

On screen: live camera with the planned route projected onto the ground
(AR) and the goal flag, the top-down belief map in the top-right corner
(tap it to set a goal), and the `draw_hud` telemetry (danger bar, steering
needle, throttle bar).

## 1 · Export a bundle

```bash
python export_android.py --checkpoint runs/best.pth --verify
# fast phone variant (backbone at 280px instead of 448px):
python export_android.py --checkpoint runs/best.pth --img-size 280 --out runs/best_280.drjepa
```

A `.drjepa` file is a zip of three ONNX graphs (backbone / decoder /
completer, ~90 MB) plus a manifest carrying the geometry and every fusion
and planning constant — the app hardcodes nothing that could drift from
`drjepa/pilot.py`. One file = one run: copy several to the phone and
switch between them with **Load model** to compare checkpoints in the
field.

## 2 · Build the app

Open `android/` in Android Studio and press Run, or from the CLI:

```bash
cd android
gradle wrapper --gradle-version 8.7   # once, if you have no wrapper
./gradlew installDebug
```

Needs Android Studio Koala+ / AGP 8.5, a device with Android 8+ (a recent
SoC recommended — the DINOv2 forward dominates the step time).

## 3 · In the field

1. Copy `.drjepa` bundles anywhere on the phone (Downloads is fine).
2. Launch, grant camera + location, tap **Load model**, pick a bundle
   (remembered across restarts).
3. Wait for a GPS fix, hold the phone upright facing your direction of
   travel (~1.4 m up, back camera forward).
4. Tap the top-down map to set a goal, or **Goal 30m ahead**.
5. Walk. Follow the steering needle / throttle bar if you want to *be*
   the closed loop; the AR path shows the current A\* route, the map
   corner shows what the model believes and imagines.
6. **Reset map** wipes the belief map and pose filter (new run, new
   memory) without reloading the model.

The status line shows bundle name, per-step latency, achieved rate, GPS
speed, heading, GPS accuracy, and goal distance (`NO ROUTE` = A\* found
no path on the believed map).

## Deliberate deviations from the sim pilot

Documented in `Pilot.kt`; the fusion/planning math is otherwise a direct
port with constants read from the bundle manifest.

- **Variable step time.** The sim ticks at 10 Hz; the phone steps as fast
  as the backbone runs (~0.3–1.5 s on CPU). Pose integration, GPS gain,
  and log-odds decay scale by real dt. `frame_offsets` still index
  perception steps, so the multi-frame parallax baseline stretches with
  the actual rate — the motion input tells the decoder the true speed.
- **No stuck detector.** A standing human is not a stuck rover. The
  arc-infeasibility recovery branches remain and surface as reverse
  throttle ("back up") on the HUD.
- **Goal from the UI** (map tap / N m ahead) instead of a goal GPS fix —
  same local metric frame.
- **Camera FOV gap.** The sim trains at 90° square; a phone's center-crop
  is ~50–65°. That compression is part of what you are measuring — a
  wide-angle/0.6x lens gets closer to the training FOV.

## Performance notes

- The step budget is almost entirely the DINOv2 forward. A 280px export
  (~2.5× faster) is the first knob; ONNX Runtime runs multithreaded CPU
  by default.
- GPS speed is the `speed` motion input; at walking pace it is noisy near
  zero — expect the pose filter to breathe until you move steadily.
- Compass heading needs calibration (wave the phone in a figure-eight if
  the map paints smeared walls).

# DR-JEPA Android test rig — the phone is the rover

Walk around with your phone and watch the v12 navigator perceive, remember,
and plan against the real world. The app runs the full MapPilot loop
on-device:

- **ARCore** owns the camera and provides VIO pose (cm-accurate, 30 Hz),
  heading, and speed; the compass is used once at startup to align
  ARCore's arbitrary world yaw to true north, then the offset is frozen
- **camera frames** → frozen DINOv2 → MapDecoder terrain wedge
  (ONNX Runtime, optionally int8-quantized)
- wedges fuse into the **persistent log-odds belief map** (occupancy,
  elevation, sand, tip-hazard)
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
# fast phone variant: 280px backbone + int8 quantization (~2.5x each axis)
python export_android.py --checkpoint runs/best.pth --img-size 280 --quantize \
    --out runs/best_280q.drjepa --verify
```

`--quantize` (dynamic int8, per-channel, **MatMul ops only**) is ~2.5×
faster on ARMv9 phones and 4× smaller, at ~8% relative token error from
DINOv2's activation outliers — perception quality may degrade; export a
fp32 twin and A/B them in the field (that's what bundle switching is
for). Conv ops stay fp32 deliberately: quantizing them emits
`ConvInteger`, which ONNX Runtime's Android build does not implement
(the bundle loads on desktop but fails on-device).

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

Needs Android Studio Koala+ / AGP 8.5 and an ARCore-supported device
(Google Play Services for AR installs on first launch if missing). The
DINOv2 forward dominates the step time — a recent SoC is strongly
recommended.

## 3 · In the field

1. Copy `.drjepa` bundles anywhere on the phone (Downloads is fine).
2. Launch, grant camera + location, tap **Load model**, pick a bundle
   (remembered across restarts).
3. Hold the phone in landscape (~1.4 m up, back camera facing your
   direction of travel) and move it slowly for a few seconds: VIO
   initializes, then the compass+GPS lock ARCore's world to true north
   (the hints on screen walk you through it). The model always sees a
   square center-crop, so orientation changes the display, not the
   model's field of view.
4. Tap the top-down map to set a goal, or **Goal 30m ahead**.
5. Walk. Follow the steering needle / throttle bar if you want to *be*
   the closed loop; the AR path shows the current A\* route, the map
   corner shows what the model believes and imagines.
6. **Reset map** wipes the belief map (new run, new memory) without
   reloading the model or re-aligning.

The status line shows bundle name, per-step latency, achieved rate, VIO
tracking state, speed, heading, and goal distance (`NO ROUTE` = A\*
found no path on the believed map).

## 4 · Collect real training data (Record mode)

The sim-trained decoder misses real-world obstacle classes it never saw
(trees, foliage); **Record** turns the phone into a labeling rig that
fixes exactly that. Tap **Record** and just walk around (no model, GPS,
or north alignment needed — VIO tracking is enough); tap again to stop.
At ~5 Hz the app stores the camera frame plus the physical camera pose +
intrinsics under `Android/data/com.mrdt.drjepa/files/collect/rec_*/`.

**No depth is captured on-device.** ARCore depth (motion-stereo, no ToF
on our phones) is far too noisy on the ground plane at range — it labelled
a flat lawn as ~80% obstacle, because grazing-angle depth error faked
±1 m of height. Depth is instead estimated **offline** by a Depth Anything
model on the recorded RGB, which produces smooth, geometrically consistent
depth. (For rover-range obstacle labels a phone can't beat a real stereo
sensor — the ZED 2i path (`zed2dataset.py`) is the primary source, run
via `./zed_docker.sh --svo ~/svos/*.svo --output data_zed` which carries
the ZED SDK in a container; the phone path is for quick, sensor-free
iteration.)

Point the camera the way the rover would see the world (~1.4 m up,
slightly down) and prefer varied scenes with plenty of open drivable
ground, not just obstacle-dense edges.

Back on the workstation (needs `pip install transformers` for the depth
model; it downloads on first run):

```bash
adb pull /sdcard/Android/data/com.mrdt.drjepa/files/collect
python real2dataset.py --sessions collect/rec_* --output data_real
python real2dataset.py --selftest        # geometry check (no model needed)
python drjepa.py preprocess --data_dir data_sim,data_sim_wall,data_real --output packed
python drjepa.py train --dataset packed  # --real_weight auto is the default
```

`real2dataset.py` runs Depth Anything on each RGB frame and unprojects
that depth through the recorded pose to build the same occupancy /
visibility / elevation wedge labels the simulator emits analytically
(tip-hazard derives from elevation in the trainer; sand has no real label
and stays zero). Monocular depth is only approximately metric — pass
`--depth_scale` to correct a systematic offset — but the obstacle test is
per-cell relative, so smooth depth already fixes the false-obstacle
problem. Episodes mix with simulator data in `preprocess`/`train` with
**zero trainer changes** — real episodes simply contribute nothing to the
map completer, which needs ground-truth grids only the sim has.

**Balancing sim vs real.** Real frames are hugely outnumbered by sim, so
`train` reweights them. `--real_weight auto` (default) lifts real to a
capped share of the sampled signal; pass a float to override, or `0` to
exclude real for an A/B. Episodes are tagged real at pack time by the
absence of ground-truth grids (naming-independent — renamed captures
still classify correctly). The split is stratified so real appears in both
train and one **combined validation set** (sim + real), which the
checkpoint score is computed over — you optimize and validate on both
together. **Never train real-only:** from-scratch on a few obstacle-heavy
episodes collapses to "everything is an obstacle"; always mix sim.

## Deliberate deviations from the sim pilot

Documented in `Pilot.kt`; the fusion/planning math is otherwise a direct
port with constants read from the bundle manifest.

- **VIO pose replaces the complementary filter.** ARCore's pose is
  cm-accurate at walking scale, so the GPS/odometry fusion and the VO
  scan-matching correction are bypassed (`vioMode` in `Pilot.kt`) — the
  sim needed them because its GPS walks and its compass wobbles; the
  phone has something strictly better. Speed also comes from VIO pose
  deltas (GPS speed is useless at walking pace). GPS's only remaining
  jobs: compass declination and the status line.
- **North alignment.** ARCore's world yaw is arbitrary; a ~4 s circular
  mean of (compass − VIO yaw) fixes the offset, then it freezes — a
  frozen small error is a constant map rotation the pilot never notices,
  a drifting one would smear the belief map.
- **Variable step time.** The sim ticks at 10 Hz; the phone steps as fast
  as the backbone runs. Log-odds decay scales by real dt; `frame_offsets`
  index perception steps, and the motion input tells the decoder the
  true speed.
- **No stuck detector.** A standing human is not a stuck rover. The
  arc-infeasibility recovery branches remain and surface as reverse
  throttle ("back up") on the HUD.
- **Goal from the UI** (map tap / N m ahead) instead of a goal GPS fix —
  same local metric frame.
- **Camera FOV gap.** The sim trains at 90° square; a phone's center-crop
  is ~50–65°. That compression is part of what you are measuring — a
  wide-angle/0.6x lens gets closer to the training FOV.

## Performance notes

- The step budget is almost entirely the DINOv2 forward. Two multiplying
  knobs: `--img-size 280` (~2.5×) and `--quantize` (~2.5× on ARMv9,
  e.g. Tensor G3 / Pixel 8 Pro with i8mm). Expect roughly: 448 fp32
  ~0.7 Hz → 280 fp32 ~1.7 Hz → 280 int8 ~4 Hz on a Pixel 8 Pro.
- ORT threads are capped at 4 to stay on the big/mid cores of
  big.LITTLE SoCs; the little cores slow the parallel sections down.
- If int8 perception looks degraded (phantom obstacles, mushy walls),
  fall back to the fp32 bundle — the A/B is one **Load model** away.

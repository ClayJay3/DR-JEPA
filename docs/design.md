# Design notes: why DR-JEPA looks the way it does

Every component in this repository exists because a measurement demanded
it, and several obvious-looking alternatives were tried and measured worse.
This page is the condensed record, so nobody re-learns these the hard way.
Numbers are from the experiment logs of v7 to v13.

## 1. Plan on a map; don't map pixels to actions

v7 drove with a behavior-cloned policy (camera + goal vector -> steering).
Its ceiling was **62.5 % success**: the moment an obstacle leaves the field
of view it stops existing, so the policy circles and grinds against things
it saw two seconds ago. Classifying steering instead of regressing it
(22.5 % -> 47.5 %) and executing chunks with hysteresis (-> 57.5 %) helped,
DAgger hurt (30-45 %: corrective data made the policy timid), and none of it
moved the ceiling.

v8 split the problem: **learn perception, do memory with math, do planning
with search**. The same perception budget went from 62.5 % to 100 % on the
obstacle-only worlds. Everything since is that design.

## 2. The camera must be a model input, and the geometry must be exact

The v12 model never knew which camera it was looking through: every sim
episode used one 1.1 m / 90 deg rig, so "read the image row" was an exact
solution and the decoder memorized one row -> range table. On the ZED
(0.4 m, 101 deg) it read the ground at the wrong range, confidently: a
systematic, range-growing elevation bias reaching -1.16 m at 15 m on a
scene whose true relief was +-0.16 m. Feeding a
camera descriptor and randomizing the rig fixed most of that, but the
network still had to *learn* projective geometry, and only within the
randomized range.

v13 removes the problem instead: the view transform **projects every ground
cell through the real camera** (intrinsics + mount + IMU attitude) and
samples image features there. There is nothing camera-specific to learn, so
any lens, aspect ratio, mast height, pitch or roll works by construction.
Measured head-to-head on identical data and encoder, the geometric lift beat
the v12-style learned (implicit) lift by roughly 2x in obstacle average
precision from the first epoch on (see [benchmarks](benchmarks.md)).

## 3. Predict physical quantities; let the robot decide what is blocking

v12 predicted a "tip hazard" class whose ramp was anchored on one rover's
tip angle, and an occupancy class at one step height. v13 predicts
**obstacle height above the terrain, terrain elevation, slope grade, soft
ground, and visibility** per cell. The robot profile (`RobotProfile`)
turns those into costs: what is lethal for a 0.75 m skid-steer is fine for
a bigger rover. One trained model serves robots with different clearances.

## 4. Fusion math is not a detail

- **Prior correction** (v8): a calibrated network's "no information" logit
  is the base rate (~2.6 % obstacles), not 0. Fusing raw logits counted
  weak positive evidence as free space. Fixing it: 75 % -> 100 % success.
- **Correlated false positives** (v8): a terrain ridge misread once is
  misread every frame, which breaks naive log-odds independence and grew
  phantom walls. Countermeasures, all still in `nav/mapping.py`: positive
  evidence x0.45, decay 0.9985/step, positive cap +3.5, fusion only within
  12 m and above 0.55 predicted visibility.
- **Never derive lethality from fused-elevation gradients** (v10): pose
  drift leaves seams between frames that read as cliffs (cost 30 points of
  success). Slope is predicted per frame and fused as a range-weighted mean.
- **Slope and elevation fuse as weighted means, not log-odds** (v13): far,
  noisy estimates never accumulate; near views overwrite them on approach.
  The v12 equivalent needed a hand-tuned range gate on positive hazard
  evidence because far false positives flooded the map with phantom lethal
  ground.

## 5. Control details that each cost points when removed

- **Plan-consistency prior**: near-tie routes flip sides every replan and
  the alternating waypoint drives the robot *at* the obstacle. A 0.45
  discount within 3 m of the incumbent route halves flips (+1.9 points of
  success); it must be gated to cells >1.6 m from obstacles or it offsets
  the inflation penalty and the planner threads corridors. A stronger
  path-commitment family reached mission parity with -23 % contacts but
  paths a few percent longer; path efficiency won.
- **Creep mode**: in tight gaps the arc check rejects everything while A*
  correctly says the corridor is passable; creeping along the route beats
  thrashing into recovery.
- **Map-checked reverse**: the map remembers what is behind; a blind
  fixed-length reverse backed into obstacles it had already seen.
- **No sand brake**: better sand detection plus a throttle governor
  quadrupled stalls -- in soft ground momentum is what gets you through.

## 6. Sim-to-real: what actually transfers

- **Photorealism alone does not close the gap.** A linear probe on DINOv3
  features separates real ZED frames from sim frames with 100 % accuracy
  for both the old numpy painter and the new GPU renderer (feature-space
  distance 110 -> 93). The new renderer still matters -- shadows, the
  robot's own shadow, cluttered ground, URC-camp objects and exact depth
  buffers -- but real data is the lever.
- **Real labels must be built carefully.** On the long ZED recordings the
  VIO gravity was off by a median 4 deg (p90 31 deg) -- metres of fake
  elevation at range. Each frame is re-levelled on a RANSAC ground plane.
  Cells that see only a wall or vehicle side get elevation *unknown*, never
  the wall height. The mount height is measured from stereo (0.39-0.44 m),
  not trusted from a config (the old labels assumed 0.6 m).
- **Monocular metric depth is unusable at rover height** (Depth Anything V2
  metric, MoGe-2: AbsRel 2-8 raw, the ground placed 4-6x too far). Aligning
  relative depth to the robot's own ground plane makes it excellent near
  (AbsRel 0.04 at 0.5-3 m) -- that is the offline teacher.
- **Never train real-only** (v12): a few cluttered yard recordings labelled
  75 % of visible cells as obstacles and the model learned "everything is
  red". Mix sim and real.

## 7. Evaluation hygiene

- **n = 30 lies.** The same model measured 93.3 % / 3.3 % tips and
  80.0 % / 12.5 % tips on two 30-episode runs. Use >= 120 episodes; contacts
  are heavy-tailed and need even more.
- **Rank detection, not pos-weighted losses.** Pos-weighted BCE rises as a
  head gets better at rare positives; checkpoint selection uses average
  precision.
- **The oracle-perception run** (perfect grids through the same
  map/planner/controller) separates perception failures from navigation
  failures. It is `--policy oracle` in the closed-loop benchmark.

## 8. What was removed, and why

| removed | measured reason |
|---|---|
| temporal/embedding JEPA branch (v12) | its only runtime output, the danger scalar, changed closed-loop results by ~0 when disabled |
| map completer, the "map-space JEPA" (v13) | AUC 0.67-0.71 on hidden cells, but every planner integration measured at or below baseline (5 schemes, 2 generations) |
| danger head (v13) | speed cap measured as a closed-loop no-op |
| Android phone rig (v13) | retired by decision; it mirrored the old pilot line for line |

A JEPA-style objective would earn its keep again in three places, none
buildable with today's data: traversability learned from the robot's own
proprioception (slip, vibration, stalls), forecasting moving obstacles, and
self-supervised adaptation on unlabeled rover video.

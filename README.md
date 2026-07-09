# DR-JEPA v9: Camera + Goal-Vector Rover Navigation

DR-JEPA drives a rover from a single forward camera and a GPS goal vector.
It does not mimic actions: it **understands the world it drives in**. A
JEPA-trained perception network converts every camera frame into metric
occupancy evidence, which is fused into a **persistent world map** — the
rover's ever-growing spatial memory for the whole run — and a planner
navigates on that learned map.

## Architecture

```
                camera frames t, t-2, t-4 (448x448)
                              |
             frozen DINOv2 ViT-S/14  ->  12x12 + CLS tokens per frame
                              |
        +---------------------+----------------------+
        |                                            |
  MapDecoder (CNN, multi-frame                FrameEncoder (MLP)
  + ego-motion conditioning)                  256-d embedding e_t --- EMA
  per-frame occupancy WEDGE                          |                 |
  (24m x 24m ahead, 0.5 m cells)              causal Transformer   target
  + per-cell visibility confidence                   |            ~e_{t+k}
        |                                     belief state s_t        ^
        v                                      |-- DangerHead         |
  VO scan-match alignment, then                |-- PolicyHead (BC)    |
  Bayesian log-odds fusion into a              |-- JEPAPredictor -----+
  WORLD-ANCHORED MAP (256m x 256m,                 (action-conditioned
  grows by re-centering, permanent                  world model)
  for the whole run)
        |
        v
  A* route on the learned costmap  ->  arc-sampling local controller
  (replans continuously)               + zero-lag near-field guard
                                       + gap-speed governor
                                       + reverse/escape recovery
```

* **Temporal memory**: the log-odds map never forgets — obstacles seen once
  stay known after leaving the camera view (that is what eliminates
  circling and re-exploration). Unbounded run length at O(1) cost per step.
* **JEPA**: trains the shared representation (action-conditioned future-
  embedding prediction with an EMA target encoder + VICReg anti-collapse);
  the danger head rides on the belief state as a speed governor. A pure
  behavior-cloned pilot (`--pilot bc`) is kept as a baseline.
* **Multi-frame perception with motion parallax**: the wedge decoder sees
  the current frame plus lookbacks (`frame_offsets`, default t, t-2, t-4)
  together with their ego-motion (speed + commanded steer), so distance
  estimates use parallax rather than texture scale alone.
* **Visual-odometry map alignment**: before each fusion, the new wedge is
  scan-matched against established map structure and the pose filter is
  nudged by the residual — removing relative GPS drift between
  observations (`--no_vo` to disable; in-sim it is roughly neutral because
  the simulated GPS drift is bounded, on real hardware it matters more).
* **Supervision**: the occupancy wedge is trained against exact geometry
  from the simulator with **occlusion-aware raycast visibility masks** —
  the net is never asked to hallucinate what the camera cannot see. This is
  privileged supervision at training time only; at inference the model sees
  pixels and noisy GPS/compass, nothing else.
* **Configurable resolution**: `ModelConfig.img_size` (default 448, any
  multiple of 14) sets the DINOv2 input size; `SimConfig.img_w/img_h` set
  the render size and should stay >= img_size. Lower for speed, raise for
  sharper perception; re-run `preprocess` (and regenerate data if the
  render size changes) after changing them.
* **Perception-driven navigation**: the planner is the same A* + arc
  recipe as the privileged expert that generates the data — but running on
  the *learned* map. Navigation quality therefore tracks perception
  quality, instead of compounding like behavior cloning.
* ~6.4M trainable parameters + frozen DINOv2. **12 ms per control step**
  (83 Hz) at 448 px on an RTX 4080 — 8x real-time at the 10 Hz control
  rate.

## Results (closed-loop)

Pooled over 110 unseen worlds across three seed blocks (v9 = 448 input +
multi-frame perception + VO alignment + gap-speed governor):

| policy                             | success | SPL   | contact events/ep |
|------------------------------------|--------:|------:|------------------:|
| privileged expert (true obstacle map) | ~98%  | 0.94 |              ~0.3 |
| **DR-JEPA v9 (camera + goal only)**   | **100%** | **0.85** |          2.0  |
| v7 behavior-cloning pilot (reference) | 62.5%   | 0.44 |             6.5   |

Held-out block (seeds 7000+, never used for any tuning): v9 100% / SPL
0.889 / 1.87 contacts vs v8's 100% / 0.878 / 2.10. Perception quality:
occupancy AUC 0.930 (v8: 0.914), validation map loss down 31%. Contacts
are now limited by pose error and actuation lag while threading tight
gaps, not by perception — an oracle-perception ablation (ground-truth
wedges through the same fusion/planning stack) scores in the same 1.5-3
contacts band.

Endless mode: consecutive goals for minutes on end with a single
persistent map (`live_inference_test.py`).

## Synthetic data (sim-to-real oriented)

`generate_synth_data.py` renders domain-randomized worlds: heightfield
terrain with camera pitch/roll, randomized sun/sky/palettes/fog, irregular
shaded rock/tree/bush meshes, ground clutter, motion blur / exposure /
vignette / sensor noise, actuation latency, steering lag, wheel slip,
contact physics with tangential sliding. Logs contain only what a real
rover would have: noisy GPS, compass, odometry — plus training-only ground
truth (true pose, occupancy wedges, expert action labels from an A* + arc
planner). `--scenario wall` forces a scenario; `--dagger ckpt` runs a
DAgger round for the BC head.

## Setup

```bash
python3.12 -m venv .venv
.venv/bin/pip install torch torchvision opencv-python pandas numpy tqdm
source .venv/bin/activate        # or use pipenv install
```

## Pipeline

```bash
# 1. generate episodes (video + telemetry CSV + wedge ground truth)
python generate_synth_data.py --episodes 400 --output data_v9
python generate_synth_data.py --episodes 150 --output data_v9_wall --scenario wall --seed 50000

# 2. pack: one frozen-DINOv2 pass per frame, features + wedge targets to memmaps
python drjepa.py preprocess --data_dir data_v9,data_v9_wall --output packed

# 3. train (~20 min)
python drjepa.py train --dataset packed --save_dir runs

# 4. closed-loop evaluation (records show the live map + planned route)
python drjepa.py eval --policy model --pilot map --checkpoint runs/best.pth --episodes 40 --record 4
python drjepa.py eval --policy expert --episodes 40        # privileged baseline
python drjepa.py eval --policy model --pilot bc ...        # BC ablation

# 5. endless live demo (persistent map across goals)
python live_inference_test.py --checkpoint runs/best.pth

# 5b. cinematic FSD-style visualization of the belief world (30 fps video):
#     obstacles, route ribbon, goal beacon, neural wedge, memory map -- all
#     rendered from what the MODEL believes, never from ground truth
python fsd_viz.py --checkpoint runs/best.pth --frames 900 --output_video fsd_demo.mp4

# 6. open-loop HUD over a recorded episode
python drjepa.py viz --video data_v9/<episode>.mp4 --checkpoint runs/best.pth
```

Real rover data drops into the same pipeline: (video, CSV) pairs with the
same telemetry columns train the JEPA/policy/danger heads directly; the
occupancy head additionally needs wedge ground truth (from lidar/stereo or
sim pretraining — the perception net transfers via the frozen DINOv2
features).

## Repository layout

```
drjepa/config.py     every hyperparameter (sim, model, training)
drjepa/simulator.py  domain-randomized world, physics, sensors, wedge GT
drjepa/expert.py     privileged A* + arc-sampling demonstration expert
drjepa/model.py      RoverJEPA: MapDecoder, JEPA world model, heads
drjepa/dataset.py    feature/wedge packing + training dataset
drjepa/pilot.py      MapPilot (map+plan navigator), BC Pilot, HUD
drjepa.py            CLI: preprocess / train / eval / viz
generate_synth_data.py  episode generator (+ --scenario, --dagger)
live_inference_test.py  endless closed-loop demo
fsd_viz.py           cinematic FSD-style belief-world visualization
```

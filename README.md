# DR-JEPA v8: Camera + Goal-Vector Rover Navigation

DR-JEPA drives a rover from a single forward camera and a GPS goal vector.
It does not mimic actions: it **understands the world it drives in**. A
JEPA-trained perception network converts every camera frame into metric
occupancy evidence, which is fused into a **persistent world map** — the
rover's ever-growing spatial memory for the whole run — and a planner
navigates on that learned map.

## Architecture

```
                        camera frame (224x224)
                              |
             frozen DINOv2 ViT-S/14  ->  12x12 + CLS tokens
                              |
        +---------------------+----------------------+
        |                                            |
  MapDecoder (CNN)                            FrameEncoder (MLP)
  per-frame occupancy WEDGE                   256-d embedding e_t --- EMA
  (24m x 24m ahead, 0.5 m cells)                     |                 |
  + per-cell visibility confidence            causal Transformer   target
        |                                            |            ~e_{t+k}
        v                                     belief state s_t        ^
  Bayesian log-odds fusion                     |-- DangerHead         |
  into a WORLD-ANCHORED MAP                    |-- PolicyHead (BC)    |
  (256m x 256m, grows by re-centering,         |-- JEPAPredictor -----+
   permanent for the whole run)                    (action-conditioned
        |                                           world model)
        v
  A* route on the learned costmap  ->  arc-sampling local controller
  (replans continuously)               + zero-lag near-field guard
                                       + reverse/escape recovery
```

* **Temporal memory**: the log-odds map never forgets — obstacles seen once
  stay known after leaving the camera view (that is what eliminates
  circling and re-exploration). Unbounded run length at O(1) cost per step.
* **JEPA**: trains the shared representation (action-conditioned future-
  embedding prediction with an EMA target encoder + VICReg anti-collapse);
  the danger head rides on the belief state as a speed governor. A pure
  behavior-cloned pilot (`--pilot bc`) is kept as a baseline.
* **Supervision**: the occupancy wedge is trained against exact geometry
  from the simulator with **occlusion-aware raycast visibility masks** —
  the net is never asked to hallucinate what the camera cannot see. This is
  privileged supervision at training time only; at inference the model sees
  pixels and noisy GPS/compass, nothing else.
* **Perception-driven navigation**: the planner is the same A* + arc
  recipe as the privileged expert that generates the data — but running on
  the *learned* map. Navigation quality therefore tracks perception
  quality, instead of compounding like behavior cloning.
* ~7M trainable parameters + frozen DINOv2. **10 ms per control step**
  (99 Hz) on an RTX 4080 — 10x real-time at the 10 Hz control rate.

## Results (closed-loop, held-out seed block, 30 unseen worlds)

| policy                             | success | SPL   | contact events/ep |
|------------------------------------|--------:|------:|------------------:|
| privileged expert (true obstacle map) | 96.7% | 0.94 |              0.23 |
| **DR-JEPA v8 (camera + goal only)**   | **100%** | **0.88** |          2.1  |
| v7 behavior-cloning pilot (reference) | 62.5%*  | 0.44* |             6.5*  |

\* measured on the tuning seed block; the BC pilot was the previous
architecture's best result.

Endless mode: 11 consecutive goals in one continuous 5-minute run with a
single persistent map (`live_inference_test.py`). An oracle-perception
ablation (ground-truth wedges through the same fusion/planning stack)
scores 95-100%, confirming perception is the remaining gap; occupancy
AUC is 0.914.

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
python generate_synth_data.py --episodes 400 --output data_v8
python generate_synth_data.py --episodes 150 --output data_v8_wall --scenario wall --seed 50000

# 2. pack: one frozen-DINOv2 pass per frame, features + wedge targets to memmaps
python drjepa.py preprocess --data_dir data_v8,data_v8_wall --output packed

# 3. train (~20 min)
python drjepa.py train --dataset packed --save_dir runs

# 4. closed-loop evaluation (records show the live map + planned route)
python drjepa.py eval --policy model --pilot map --checkpoint runs/best.pth --episodes 40 --record 4
python drjepa.py eval --policy expert --episodes 40        # privileged baseline
python drjepa.py eval --policy model --pilot bc ...        # BC ablation

# 5. endless live demo (persistent map across goals)
python live_inference_test.py --checkpoint runs/best.pth

# 6. open-loop HUD over a recorded episode
python drjepa.py viz --video data_v8/<episode>.mp4 --checkpoint runs/best.pth
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
```

# Benchmarks

Every number here is reproducible with `tools/run_benchmarks.sh` and the
commands quoted per section, and the raw results (per-episode for driving,
per-range-band for perception) are in [results/](results/README.md).
Hardware: RTX 4080 SUPER, Ryzen 9 7950X. The default model is published as
`models/geobev.pt` / `models/geobev.onnx`; results files call it
`runs/geobev/best.pt`.

**Protocol.** Closed-loop results use 144 unseen simulator worlds (seeds
10000-10143, all five scenario types, ~40-150 m goals) with a *different
random camera rig every episode* unless a rig is pinned. An episode ends on
reaching the goal (4 m), tipping over, 60 s without getting closer to the
goal than ever before, or a 300 s backstop. We never report fewer than 120
episodes: on 30-episode runs the same model measured both 93 % and 80 %
success. Real-world results use ZED 2i recordings from a site the networks
never trained on (Hanksville, UT), scored against stereo depth.

**Models.** Every network is GeoBEV (geometric lift + top-down U-Net); they
differ in encoder and training recipe. Two ship:

| id | encoder | training recipe | parameters |
|---|---|---|---:|
| A | EfficientNet-B0 (stride 8) | sim + real (15 %) | 4.5 M |
| B | EfficientNet-B0 | A with the v12-style implicit lift | 6.2 M |
| C | EfficientNet-B0 | A + training-only depth head | 4.5 M |
| A2 | EfficientNet-B0 | C + steep-weighted slope loss | 4.5 M |
| **F -- default, `runs/geobev`** | EfficientNet-B0 | A2 + distilled from D (features + outputs), real share 30 %, unlabeled phone video | **4.5 M** |
| D | DINOv3 ViT-S/16, frozen | C recipe | 25.3 M (21.7 M frozen) |
| **D2 -- accuracy, `runs/geobev_dino`** | DINOv3 ViT-S/16, frozen | D + steep-weighted slope loss | 25.3 M |

---

## 1. Closed-loop driving (simulator)

| policy | success | tipped | contacts / ep | collision-free | SPL |
|---|---:|---:|---:|---:|---:|
| privileged expert (true map) | 97.2 % | 1.4 % | 0.38 | 91.7 % | 0.909 |
| perfect perception through the same map / planner / controller | 94.4 % | 4.2 % | 1.16 | 71.5 % | 0.862 |
| **F: GeoBEV-B0, distilled (default)** | 93.8 % | 6.2 % | 1.42 | 77.1 % | 0.814 |
| A2: GeoBEV-B0, not distilled | 94.4 % | 5.6 % | 0.90 | 81.9 % | 0.828 |
| D2: GeoBEV-DINOv3 (accuracy model) | 93.1 % | 5.6 % | 1.07 | 75.0 % | 0.809 |
| *v12 legacy model, in the v12 simulator* | *79.9 %* | *14.6 %* | *2.31* | *58.3 %* | *0.630* |

SPL = success weighted by straight-line / driven distance. One episode is
0.7 points, so the three GeoBEV rows are statistically tied -- and tied
with *perfect* perception through the same stack: on random rigs, sim
success is limited by navigation and pose, not perception (tips are the
exception: 5.6-6.2 % vs 4.2 %). Contacts are heavy-tailed (two or three
episodes with 15-35 grazes move the mean by 0.5) and limited by the GPS
pose, which wanders 1-2 m. F is the default because it is the best small
model on *real* data (section 2), not because it drives the simulator
best. The v12 row is the previous shipping model on the previous simulator
(same protocol, same seeds); its own expert scored 98.6 % there, so both
worlds are equally solvable.

`python -m drjepa eval-drive --policy model --checkpoint models/geobev.pt --episodes 144` (add `--camera zed` etc. for a pinned rig, `--policy oracle` / `expert` for the references)

### Cross-domain: a world that looks different

Both models driving the *new* simulator (seeds 20000-20143) through square
448 px cameras drawn from the v12 model's own training ranges (height
0.4-2.0 m, HFOV 60-120 deg, pitch -25 to +5 deg), so the only thing new for
the v12 model is appearance:

| in the new simulator | success | tipped | contacts / ep | collision-free | SPL |
|---|---:|---:|---:|---:|---:|
| v12 legacy model | 75.7 % | 11.1 % | **14.25** | 15.3 % | 0.609 |
| **GeoBEV (A)** | **91.7 %** | 7.6 % | **0.71** | **84.7 %** | **0.818** |

The v12 network learned the old renderer's look, not geometry: in a world
that merely looks different it hits obstacles 20x as often (wall worlds:
48.6 % success, 28 contacts per episode). Any real camera looks different
from any simulator, so this is the generalization that matters.

### Pinned camera rigs (model F, same 144 worlds)

| rig | success | tipped | contacts / ep | stalls / ep | SPL |
|---|---:|---:|---:|---:|---:|
| random per episode | 93.8 % | 6.2 % | 1.42 | 0.00 | 0.814 |
| phone: 1.4 m, 66 deg, pitch -8 | 91.0 % | 9.0 % | 0.33 | 0.05 | 0.798 |
| mast: 2.0 m, 70 deg, pitch -15 | 91.0 % | 9.0 % | 0.37 | 0.35 | 0.808 |
| rolled: 0.8 m, 90 deg, roll +6 | 88.9 % | 11.1 % | 0.39 | 0.30 | 0.779 |
| low-wide: 0.30 m, 120 deg, pitch -5 | 87.5 % | 11.8 % | 1.17 | 0.00 | 0.740 |
| ZED 2i: 0.40 m, 101.5 deg, level | 84.7 % | 14.6 % | 0.85 | 0.08 | 0.747 |

Every rig drives (mean 88.6 %, model A: 86.7 %); the losses are almost
entirely tip-overs, and they grow as the camera gets lower: at 0.3-0.4 m
the ground ahead is seen at a grazing angle, which is where slope is
hardest to read. With perfect perception the ZED rig reaches 92.4 % /
5.6 % tips, so the low-camera gap is perception: slope read from 0.4 m.

### 1b. Why the robot tips: slopes believed too gentle

For every tip-over of model A, the belief map at the tip location had been
**observed with high confidence (fusion weight 7-20) but held a slope of
0.12-0.42 where the truth was 0.53-0.73**. Measured over held-out sim
episodes (cells with true grade > 0.45 within 12 m):

| grade head | steep AUC | recall at the lethal threshold (0.46) | bias on steep cells | flat cells flagged |
|---|---:|---:|---:|---:|
| A: L1 regression | 0.90 | 20 % | -0.22 | 0.00 % |
| A2: steep-weighted pinball (q 0.7, w 4) | **0.96** | **60 %** | **-0.08** | 0.31 % |
| F: A2 + distillation (default) | **0.96** | 54 % | -0.10 | 0.14 % |
| D: frozen DINOv3, L1 | 0.90 | 14 % | -0.23 | 0.00 % |
| D2: frozen DINOv3, steep-weighted pinball | 0.95 | 50 % | -0.11 | 0.23 % |

The network *ranks* slopes well but regression shrinks rare steep cells
toward the flat majority. Weighting steep cells and penalizing
under-estimation (it tips you; over-estimation costs a detour) triples
recall at the planner's threshold for a 0.3 % false-alarm rate on flat
ground. Lowering the threshold on the unfixed model instead (0.35: 50 %
recall) flags a quarter of drivable 0.30-0.45 slopes as lethal.

`python -m drjepa eval-perception --checkpoint X --sim_val data/sim --stride 4`

**Every remaining tip is a roll-over.** With the A2 head, all nine tips
had a side-slope of 0.51-0.58 (limit 0.50) while pitch stayed moderate, and
seven of nine happened in plain drive mode. Robots tip sideways long before
they tip forward, so the map now fuses a slope *vector* (magnitude from the
grade head, direction from the gradient of the elevation head -- median 9
deg direction error on steep sim cells) and every candidate arc is checked
against side-slope, downhill and climb limits in the direction it travels:

| model A2, same 144 worlds | success | tipped | stalls / ep | SPL |
|---|---:|---:|---:|---:|
| isotropic grade check (0.45) | 93.1 % | 6.2 % | 0.28 | 0.813 |
| + creep refuses believed-steep arcs (reverses instead) | 91.0 % | 8.3 % | 0.25 | 0.791 |
| **direction-aware check, side slope 0.40** | **94.4 %** | **4.9 %** | 0.25 | **0.829** |
| direction-aware check, side slope 0.35 | 91.7 % | 6.9 % | 0.50 | 0.805 |

The creep rule fixed none of the original tips and added three, all while
reversing onto ground the forward camera had never seen; it was reverted.
The direction-aware check fixed two tips and added none. The seven that
remain are perception: the map believed 0.16-0.49 where the truth was
0.52-0.73.

On the low ZED rig, 6 of 10 analysed tips came while *creeping* (the arc
check had rejected everything and creep followed the route regardless),
one while backing down a believed-steep slope. Creep now takes the
slope-safe arc nearest the route, and the map-checked reverse also stops on
believed-steep ground (model F, same seeds and precision):

| | random rigs | ZED rig |
|---|---:|---:|
| creep follows the route regardless | 91.7 % / 6.9 % tips | 83.3 % / 15.3 % tips |
| **creep on slope-safe arcs + reverse stops on steep ground** | **93.8 % / 6.2 %** | **84.7 % / 14.6 %** |

---

## 2. Real-world perception (held-out site, ZED 2i stereo truth)

Two recordings from one Hanksville drive (front and rear camera, 0.39 m /
0.44 m above the ground, 101 deg HFOV, 1280 x 720). Scored on stereo-visible
cells, by forward range. *bestF1* uses the best threshold for every method
(same procedure for all); elevation error is on cells where ground is
observed.

| method | obstacles AUC (0-6 / 6-12 m) | best F1 (0-6 / 6-12 m) | elevation MAE, front (0-6 / 6-12 / 12-24 m) | elevation MAE, rear (0-6 / 6-12 m) |
|---|---|---|---|---|
| A: GeoBEV-B0 | 0.92 / 0.64 | 0.39 / 0.31 | 0.20 / 0.33 / 0.33 | 0.22 / 0.46 |
| B: implicit lift | 0.82 / 0.78 | 0.32 / 0.32 | 0.12 / 0.41 / 0.60 | 0.17 / 0.50 |
| C: + depth head | 0.91 / 0.65 | 0.41 / 0.27 | 0.34 / 0.45 / 0.47 | 0.14 / 0.32 |
| A2: + slope loss | 0.94 / 0.72 | 0.46 / 0.27 | 0.35 / 0.44 / 0.47 | 0.17 / 0.39 |
| **F: + distillation (default, 4.5 M)** | 0.95 / 0.77 | 0.56 / 0.35 | 0.27 / 0.34 / **0.27** | 0.11 / 0.23 |
| D: frozen DINOv3 | **0.97** / 0.76 | **0.58** / 0.34 | 0.08 / 0.22 / 0.39 | 0.10 / 0.18 |
| **D2: frozen DINOv3 + slope loss (accuracy)** | 0.96 / **0.84** | 0.56 / **0.36** | 0.11 / 0.28 / 0.42 | 0.10 / 0.20 |
| depth teacher (DA-V2-B + mount plane, no training) | 0.70 / 0.45 | 0.33 / 0.18 | **0.03 / 0.11** / 0.25 | **0.03 / 0.17** |

Obstacle columns are from the rear (camp) recording; the front recording is
open desert with 0.5 % obstacle cells, too few to score.

- **Every trained network sees real obstacles better than the depth model**
  (AUC 0.92-0.97 vs 0.70 near).
- **Frozen DINOv3 features are what transfer real terrain.** The B0 models
  trained on sim + two flat real lots read gently rising desert as flat
  (front elevation 0.20-0.35 m near: a range-growing under-estimate of
  about -2.3 cm per metre); the frozen DINOv3 encoder cuts that to
  0.08-0.11 m.
- **Distillation moves the small model most of the way.** F is A2's network
  trained against D's features and outputs (plus a larger real share and
  unlabeled phone video): obstacle F1 0.46 -> 0.56 (the teacher's 0.58),
  rear elevation 0.17 / 0.39 -> 0.11 / 0.23 m (teacher 0.10 / 0.18), front
  elevation 0.35 -> 0.27 m near. The front-terrain gap to DINOv3 is the
  largest one left; more varied real terrain is the direct fix.

`python -m drjepa eval-perception --checkpoint X --episodes "recordings/zed/ZED_2i_31237348_(1)" "recordings/zed/ZED_2i_39163798_(1)"`

---

## 3. View transform: geometric vs learned (same data, same encoder)

| after 10 epochs | sim obstacle AP | sim elevation MAE | real val AP (training site) |
|---|---:|---:|---:|
| geometric lift (projection sampling) | **0.492** | **0.080 m** | 0.455-0.496 |
| implicit lift (v12 design, camera vector + learned up-convolutions) | 0.397 | 0.107 m | 0.440 |

The geometric lift led from the first epoch (AP 0.34 vs 0.11). In closed
loop the two are tied (section 1): with this navigation stack, sim driving
is limited by pose and control, not by the perception gap -- the same
finding as the v9 oracle study.

---

## 4. Monocular depth at rover height (why the teacher is aligned to the ground)

*One-off study (not a maintained benchmark); the teacher itself is
`drjepa/perception/teacher.py` and is scored by `eval-perception --teacher Base`.*

AbsRel vs ZED stereo on 40 frames of a held-out recording (camera 0.39 m).
"Mount-plane affine" fits inverse depth = a * prediction + b on near-ground
pixels using only the known mount height -- deployable on any robot.

| model | raw | mount-plane affine, 0.5-3 / 3-6 / 6-10 / 10-20 m | ms / frame |
|---|---:|---|---:|
| Depth Anything V2 Base (relative) | -- | **0.038 / 0.105 / 0.244 / 0.43** | 18 |
| Depth Anything V2 Small (relative) | -- | 0.042 / 0.125 / 0.277 / 0.59 | 11 |
| MoGe-2 ViT-S (metric) | 2.45 | 0.050 / 0.169 / 0.342 / 0.59 | 48 |
| Depth Anything V2 Metric Outdoor S | 8.19 | 0.042 / 0.134 / 0.321 / 0.95 | 11 |

Metric monocular depth fails at rover height (it places the ground 4-6x too
far, as if the camera were at eye level); relative depth aligned to the
robot's own ground plane is excellent near and degrades with range.

---

## 5. Simulator realism

*One-off study (not a maintained benchmark); the v12 renderer it compares
against is in git history.*

Frechet distance between DINOv3 ViT-S feature distributions of 800 real ZED
frames and 800 simulator frames (FID recipe, lower = closer to real), and
the accuracy of a linear probe separating real from sim.

| | Frechet distance | real-vs-sim probe |
|---|---:|---:|
| real vs real (two halves) | 4.1 | 45 % (chance) |
| **new GPU simulator, ZED-like camera** | **93.9** | 100 % |
| new GPU simulator, random rigs | 95.8 | 100 % |
| v12 numpy simulator | 121.0 | 100 % |

The renderer closes about a fifth of the gap; no procedural renderer fools a
linear probe on DINOv3 features. Real data, not photorealism, is the lever.

---

## 6. Latency

Isolated, one frame at a time, 1280 x 720 input resized to 512 x 320.
Ryzen 9 7950X for CPU numbers.

| stage | GPU (RTX 4080, PyTorch fp32) | CPU (ONNX Runtime) |
|---|---:|---:|
| perception, GeoBEV-B0 (F) -- network only | 3.0 ms | 18.7 ms (4 threads), 57 ms (1 thread) |
| perception, GeoBEV-B0 (F) -- per call, incl. resize + copies | 4.5 ms | 20.6 ms (4 threads), 58 ms (1 thread) |
| perception, GeoBEV-DINOv3 (D2) -- per call | 6.3 ms | 70 ms (16 threads), 93 ms (4), 312 ms (1) |
| belief-map fusion | 1.0 ms | 1.0 ms |
| A* planning to a 60 m goal (every 3rd step, amortized) | 2.0 ms | 2.0 ms |
| arc control | 0.2 ms | 0.2 ms |
| **full `Navigator.step`, median (p95)** | **5.9 ms (11.9 ms)** | **22 ms (29 ms)**, 4 threads |

fp16 autocast is *slower* at batch 1 on this GPU (B0 5.9 ms, DINOv3 7.6 ms
per call), so `Perceiver` defaults to fp32; for embedded fp16 / int8 use
the ONNX file with TensorRT. ONNX vs PyTorch outputs agree to 4e-5 (B0)
and 5e-5 (DINOv3). For the small model on CPU pass `threads=4` to
`OnnxPerceiver`: ONNX Runtime's default (one thread per core) is slower
(26 ms per call, p95 43 ms).

`python -m drjepa bench-latency --checkpoint models/geobev.pt --onnx models/geobev.onnx --recording "recordings/zed/ZED_2i_31237348_(1)"`
(B0 rows; network-only CPU numbers from `export --verify`; the DINOv3 row
was measured the same way on `runs/geobev_dino`).

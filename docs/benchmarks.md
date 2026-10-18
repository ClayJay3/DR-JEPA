# Benchmarks

Every number here is reproducible with `tools/run_benchmarks.sh` and the
commands quoted per section. Hardware: RTX 4080 SUPER, Ryzen 9 7950X.

**Protocol.** Closed-loop results use 144 unseen simulator worlds (seeds
10000-10143, all five scenario types, ~40-150 m goals) with a *different
random camera rig every episode* unless a rig is pinned. An episode ends on
reaching the goal (4 m), tipping over, 60 s without getting closer to the
goal than ever before, or a 300 s backstop. We never report fewer than 120
episodes: on 30-episode runs the same model measured both 93 % and 80 %
success. Real-world results use ZED 2i recordings from a site the networks
never trained on (Hanksville, UT), scored against stereo depth.

---

## 1. Closed-loop driving (simulator)

| policy | success | tipped | contacts / ep | collision-free | SPL |
|---|---:|---:|---:|---:|---:|
| privileged expert (true map) | 97.2 % | 1.4 % | 0.38 | 91.7 % | 0.909 |
| **GeoBEV (geometric lift)** | **93.8 %** | 6.2 % | 0.93 | 77.1 % | 0.824 |
| GeoBEV with the v12-style implicit lift | 94.4 % | 4.2 % | 0.82 | 77.1 % | 0.834 |
| *v12 legacy model, in the v12 simulator* | *79.9 %* | *14.6 %* | *2.31* | *58.3 %* | *0.630* |

SPL = success weighted by straight-line / driven distance. The v12 row is
the previous shipping model on the previous simulator (same protocol, same
seeds); its own expert scored 98.6 % there, so both worlds are equally
solvable.

`python -m drjepa eval-drive --policy model --checkpoint runs/A_geo_b0/best.pt --episodes 144`

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

---

## 2. Real-world perception (held-out site, ZED 2i stereo truth)

Two recordings from one Hanksville drive (front and rear camera, 0.39 m /
0.44 m above the ground, 101 deg HFOV, 1280 x 720). Scored on stereo-visible
cells, by forward range. *bestF1* uses the best threshold for every method
(same procedure for all); elevation error is on cells where ground is
observed.

| method | obstacles AUC (0-6 / 6-12 m) | best F1 (0-6 / 6-12 m) | elevation MAE, front (0-6 / 6-12 / 12-24 m) | elevation MAE, rear (0-6 / 6-12 m) |
|---|---|---|---|---|
| GeoBEV geometric (A) | **0.92** / 0.64 | **0.39** / 0.31 | 0.20 / 0.33 / 0.33 | 0.22 / 0.46 |
| GeoBEV implicit (B) | 0.82 / **0.78** | 0.32 / 0.32 | 0.12 / 0.41 / 0.60 | 0.17 / 0.50 |
| depth teacher (DA-V2-B + mount plane, no training) | 0.70 / 0.45 | 0.33 / 0.18 | **0.03 / 0.11 / 0.25** | **0.04 / 0.17** |

Obstacle columns are from the rear (camp) recording; the front recording is
open desert with 0.5 % obstacle cells, too few to score. **The trained
networks see real obstacles better than the depth model, but their real
terrain elevation is 4-6x worse**: a systematic, range-growing under-
estimate (-0.10 m near, about -2.3 cm per metre of range) on terrain that
rises gently away from the rover. Both real training recordings are flat
dirt lots, so "real ground is flat" is a learnable shortcut; more varied
real terrain is the fix.

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

| stage | RTX 4080 | CPU (ONNX Runtime) |
|---|---:|---:|
| perception (GeoBEV-B0, 512 x 320 input) | see note | 29 ms (4 threads), 68 ms (1 thread) |
| belief-map fusion | 1.1 ms | 1.1 ms |
| A* planning (every 3rd step, amortized) | 2.8 ms | 2.8 ms |
| arc control | 0.3 ms | 0.3 ms |

GPU perception latency measured during evaluation (shared GPU): ~20 ms with
8 concurrent simulator workers; isolated timing is in section 7 once
measured.

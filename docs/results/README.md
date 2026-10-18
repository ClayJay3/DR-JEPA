# Raw benchmark results

Every number in [../benchmarks.md](../benchmarks.md), as written by the
benchmark commands. Closed-loop files hold a `summary` (with the checkpoint,
camera rig and navigation overrides used) and one record per episode, so any
two runs can be compared seed by seed. `runs/geobev/best.pt` in these files
is the published `models/geobev.pt` (model F).

## closed_loop/ -- 144 sim worlds, seeds 10000-10143, final code

| file | benchmarks.md |
|---|---|
| `F_random.json` | section 1, F row; pinned-rig table, random row |
| `F_{phone,mast,rolled,low-wide,zed}.json` | pinned-rig table |
| `A2_random.json`, `D2_random.json` | section 1 |
| `oracle_random.json`, `oracle_zed.json` | perfect perception through the same map / planner / controller |
| `expert_random.json` | privileged expert (true map, its own planner) |

`python -m drjepa eval-drive --policy model --checkpoint models/geobev.pt --episodes 144 [--camera zed]`

## closed_loop/ablations/ -- history and controlled comparisons

Earlier code or models; each summary records what was run.

| file | what |
|---|---|
| `A_geometric_lift.json`, `B_implicit_lift.json` | view-transform comparison (section 3), controller of that time |
| `A_pinned_*.json` | model A on the pinned rigs (before the slope work) |
| `A2_slope_isotropic.json`, `A2_creep_reverses_on_slope.json`, `A2_slope_directional_0.40.json`, `A2_slope_directional_0.35.json` | section 1b, slope-check table |
| `F_creep_ignores_slope_{random,zed}.json` | section 1b, creep table (same seeds and precision as the final runs) |
| `cross_domain_A.json`, `cross_domain_v12_model.json` | cross-domain table (new simulator, square v12-range cameras) |
| `v12_model_in_v12_sim.json`, `v12_expert_in_v12_sim.json` | the v12 rows (previous model in the previous simulator; code in git history) |

## perception/

| file | what |
|---|---|
| `hanksville_*.json` | section 2: held-out real site, per recording and range band |
| `simval_*.json` | section 1b: held-out sim split, incl. steep-slope metrics |

```
python -m drjepa eval-perception --checkpoint models/geobev.pt --stride 2 \
    --episodes "recordings/zed/ZED_2i_31237348_(1)" "recordings/zed/ZED_2i_39163798_(1)"
python -m drjepa eval-perception --checkpoint models/geobev.pt --sim_val data/sim --stride 4
```

## latency/

`geobev.json`: section 6, from
`python -m drjepa bench-latency --checkpoint models/geobev.pt --onnx models/geobev.onnx --recording "recordings/zed/ZED_2i_31237348_(1)"`
on an idle RTX 4080 SUPER + Ryzen 9 7950X.

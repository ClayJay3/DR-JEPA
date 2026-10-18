#!/usr/bin/env bash
# Reproduce every number in docs/benchmarks.md for one checkpoint.
#
#   ./tools/run_benchmarks.sh runs/geobev/best.pt bench/geobev
#
# Closed loop: 144 unseen sim worlds (seeds 10000-10143) with a random camera
# rig per episode, then the same worlds through five pinned rigs; the
# privileged expert and the perfect-perception oracle for reference.
# Open loop: the held-out sim validation split (incl. steep-slope metrics) and
# the held-out Hanksville ZED recordings (stereo ground truth).
# Deployment: ONNX export + verification, then isolated latency (run this
# last, on an otherwise idle machine).
set -euo pipefail
CK=${1:?checkpoint}
OUT=${2:-bench/$(basename "$(dirname "$CK")")}
EPS=${EPISODES:-144}
W=${WORKERS:-8}
mkdir -p "$OUT"

python -m drjepa eval-drive --policy model --checkpoint "$CK" --episodes "$EPS" --workers "$W" \
    --out "$OUT/drive_random.json"
for cam in zed phone low-wide mast rolled; do
    python -m drjepa eval-drive --policy model --checkpoint "$CK" --episodes "$EPS" --workers "$W" \
        --camera "$cam" --out "$OUT/drive_$cam.json"
done
[ -f bench/expert_144.json ] || python -m drjepa eval-drive --policy expert --episodes "$EPS" \
    --workers "$W" --out bench/expert_144.json
# the oracle shares the map / planner / controller, so it is never cached
python -m drjepa eval-drive --policy oracle --episodes "$EPS" --workers "$W" --out "$OUT/oracle_random.json"
python -m drjepa eval-drive --policy oracle --camera zed --episodes "$EPS" --workers "$W" \
    --out "$OUT/oracle_zed.json"

python -m drjepa eval-perception --checkpoint "$CK" --sim_val data/sim --stride 4 --out "$OUT/simval.json"
python -m drjepa eval-perception --checkpoint "$CK" --stride 2 --out "$OUT/hanksville.json" \
    --episodes "recordings/zed/ZED_2i_31237348_(1)" "recordings/zed/ZED_2i_39163798_(1)"

python -m drjepa export --checkpoint "$CK" --out "$OUT/geobev.onnx" --verify | tee "$OUT/export.txt"
python -m drjepa bench-latency --checkpoint "$CK" --onnx "$OUT/geobev.onnx" \
    --recording "recordings/zed/ZED_2i_31237348_(1)" --out "$OUT/latency.json"

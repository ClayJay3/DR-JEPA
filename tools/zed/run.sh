#!/usr/bin/env bash
# Run tools/zed/export_svo.py inside the ZED SDK container. Builds the image on
# first use, then passes every argument straight through:
#
#   ./tools/zed/run.sh --svo zed_data/*.svo2 --output recordings/zed --hz 1
#
# $HOME is bind-mounted at the same path and the working dir is the repo root,
# so host paths (including globs) resolve exactly as they do on the host. The
# ZED neural-depth model is cached in a named volume so it downloads once.
# Needs Docker + the NVIDIA container toolkit.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
IMG=drjepa-zed:5.4

docker image inspect "$IMG" >/dev/null 2>&1 || {
    echo ">> building $IMG (first run only, a few minutes) ..."
    docker build --network=host -f "$HERE/Dockerfile" -t "$IMG" "$HERE"
}

# The container runs as root so the SDK can cache its model in the volume;
# hand the output back to the host user afterwards.
OUT="recordings/zed"
args=("$@")
for ((k = 0; k < ${#args[@]}; k++)); do
    [[ "${args[k]}" == "--output" ]] && OUT="${args[k + 1]:-$OUT}"
done
# create it host-side so every parent directory is user-owned, not root
mkdir -p "$OUT"

exec docker run --rm --gpus all \
    -v "$HOME":"$HOME" \
    -v drjepa-zed-resources:/usr/local/zed/resources \
    -w "$REPO" \
    -e OUT="$OUT" -e UGID="$(id -u):$(id -g)" \
    "$IMG" bash -c \
        'python3 tools/zed/export_svo.py "$@"; rc=$?;
         chown -R "$UGID" "$OUT" 2>/dev/null || true; exit $rc' \
        bash "$@"

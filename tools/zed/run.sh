#!/usr/bin/env bash
# Run zed2dataset.py inside the ZED SDK container -- the Fedora-friendly way
# to process ZED SVO recordings (installing the SDK natively on Fedora is
# painful; the container carries the SDK + pyzed + CUDA). Builds the image
# on first use, then passes every argument straight through to
# zed2dataset.py:
#
#   ./zed_docker.sh --svo ~/svos/*.svo --output data_zed --cam_height 0.6
#   ./zed_docker.sh --selftest
#
# Host paths work verbatim inside the container: $HOME is bind-mounted at
# the same path and the working dir is this repo, so a globbed --svo list
# and the --output dir resolve exactly as they do on the host. The ZED AI
# depth model is cached in a named volume so it downloads only once.
#
# Needs: Docker + the NVIDIA container toolkit (both already present here).
set -euo pipefail

REPO="$(cd "$(dirname "$0")" && pwd)"
IMG=drjepa-zed:5.4

docker image inspect "$IMG" >/dev/null 2>&1 || {
    echo ">> building $IMG (first run only, a few minutes) ..."
    docker build -f "$REPO/Dockerfile.zed" -t "$IMG" "$REPO"
}

# The container runs as root so the SDK can cache its AI model into the
# resources volume; parse --output so the results are handed back to the
# host user afterward instead of staying root-owned.
OUT="data_zed"
args=("$@")
for ((k = 0; k < ${#args[@]}; k++)); do
    [[ "${args[k]}" == "--output" ]] && OUT="${args[k + 1]:-$OUT}"
done

exec docker run --rm --gpus all \
    -v "$HOME":"$HOME" \
    -v drjepa-zed-resources:/usr/local/zed/resources \
    -w "$REPO" \
    -e OUT="$OUT" -e UGID="$(id -u):$(id -g)" \
    "$IMG" bash -c \
        'python3 zed2dataset.py "$@"; rc=$?;
         chown -R "$UGID" "$OUT" 2>/dev/null || true; exit $rc' \
        bash "$@"

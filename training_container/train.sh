#!/usr/bin/env bash
set -euo pipefail

SCRIPTPATH="$(cd "$(dirname "$0")" && pwd -P)"
REPO_ROOT="$(cd "$SCRIPTPATH/.." && pwd -P)"
IMAGE="${IMAGE:-fedscale-umamba-trainer:latest}"
DATA_DIR="${DATA_DIR:?Set DATA_DIR to the host directory containing the dataset}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/training_runs}"

mkdir -p "$OUTPUT_DIR"

docker run --rm \
    --gpus all \
    --ipc=host \
    --shm-size=16g \
    --volume "$DATA_DIR:/data:ro" \
    --volume "$OUTPUT_DIR:/outputs" \
    --env WANDB_API_KEY \
    "$IMAGE" \
    "$@"

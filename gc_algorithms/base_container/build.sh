#!/usr/bin/env bash
set -euo pipefail

SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"
MODEL="${1:-umamba_mtl}"
IMAGE="picai_${MODEL}_inferer:latest"

docker build \
    --platform linux/amd64 \
    --build-arg "MODEL=$MODEL" \
    --tag "$IMAGE" \
    "$SCRIPTPATH"

echo "Built $IMAGE"

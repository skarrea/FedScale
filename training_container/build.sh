#!/usr/bin/env bash
set -euo pipefail

SCRIPTPATH="$(cd "$(dirname "$0")" && pwd -P)"
REPO_ROOT="$(cd "$SCRIPTPATH/.." && pwd -P)"
IMAGE="${IMAGE:-fedscale-umamba-trainer:latest}"

docker build \
    --platform linux/amd64 \
    --file "$SCRIPTPATH/Dockerfile" \
    --tag "$IMAGE" \
    "$REPO_ROOT"

echo "Built $IMAGE"

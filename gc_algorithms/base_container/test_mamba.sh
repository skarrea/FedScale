#!/usr/bin/env bash
set -euo pipefail

IMAGE="${1:-picai_umamba_mtl_inferer:latest}"

docker run --rm \
    --gpus all \
    --entrypoint python \
    "$IMAGE" \
    /opt/ml/model/mamba_smoke_test.py

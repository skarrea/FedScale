#!/usr/bin/env bash
set -euo pipefail

SCRIPTPATH="$(cd "$(dirname "$0")" && pwd -P)"
IMAGE="${IMAGE:-fedscale-umamba-trainer:latest}"
ARCHIVE="${1:-$SCRIPTPATH/fedscale-umamba-trainer.tar.gz}"

"$SCRIPTPATH/build.sh"
docker save "$IMAGE" | gzip > "$ARCHIVE"

echo "Exported $IMAGE to $ARCHIVE"

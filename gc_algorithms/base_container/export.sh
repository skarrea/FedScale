#!/usr/bin/env bash
set -euo pipefail

SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"
MODEL="${1:-umamba_mtl}"
IMAGE="picai_${MODEL}_inferer:latest"
ARCHIVE="$SCRIPTPATH/picai_${MODEL}_inferer.tar.gz"

"$SCRIPTPATH/build.sh" "$MODEL"
docker save "$IMAGE" | gzip -c > "$ARCHIVE"

echo "Exported $ARCHIVE"

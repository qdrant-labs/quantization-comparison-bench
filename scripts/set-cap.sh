#!/usr/bin/env bash
# Change the running Qdrant container's memory cap in place:
#   scripts/set-cap.sh <bytes>
# Uses docker update, so the container, its process, and the telemetry targets
# stay the same. Swap stays disabled (memory-swap equals memory).
set -euo pipefail
bytes="${1:?usage: scripts/set-cap.sh <bytes>}"
: "${QDRANT_CONTAINER_ID:?source scripts/vm-env.sh first}"
docker update --memory "$bytes" --memory-swap "$bytes" "$QDRANT_CONTAINER_ID" >/dev/null
cgroup="${QDRANT_CGROUP_PATH:?}"
for _ in $(seq 1 50); do
  [ "$(cat "$cgroup/memory.max")" = "$bytes" ] && { echo "memory.max=$bytes"; exit 0; }
  sleep 0.2
done
echo "memory.max did not change to $bytes" >&2
exit 1

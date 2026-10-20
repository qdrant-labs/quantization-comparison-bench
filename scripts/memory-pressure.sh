#!/usr/bin/env bash
# Quora/Cohere under a fixed 1 GiB container cap:
#
#   scripts/memory-pressure.sh [results-root]
#
# Sets QDRANT_MEMORY_LIMIT=1g in .env, recreates the container, re-sources the
# telemetry targets, and runs configs/memory-pressure.yaml. The runner itself
# verifies the effective cgroup limit before measuring. Restore the normal cap
# afterwards (QDRANT_MEMORY_LIMIT=28g) and recreate the container again.
set -euo pipefail
root="${1:-results}"
out="$root/quora-cohere/memory-pressure"
if [ -e "$out" ]; then
  echo "$out already exists; move it aside before rerunning" >&2
  exit 1
fi
[ -f .env ] || cp .env.example .env
if grep -q '^QDRANT_MEMORY_LIMIT=' .env; then
  sed -i 's/^QDRANT_MEMORY_LIMIT=.*/QDRANT_MEMORY_LIMIT=1g/' .env
else
  echo 'QDRANT_MEMORY_LIMIT=1g' >> .env
fi
docker compose up -d --force-recreate
for _ in $(seq 1 120); do
  curl --fail --silent "${QDRANT_URL:-http://127.0.0.1:6333}/readyz" >/dev/null && break
  sleep 1
done
# shellcheck source=scripts/vm-env.sh
source scripts/vm-env.sh "$root/environment-memory-pressure"
docker inspect --format '{{.HostConfig.Memory}}' "$QDRANT_CONTAINER_ID" > "$root/environment-memory-pressure/host-config-memory.txt"
uv run qbench readiness --data-root data --datasets quora --models cohere --require-all
uv run qbench run --config configs/memory-pressure.yaml --data data/embeddings/quora/cohere --output "$out"
echo "Memory-pressure run complete under $out. Restore QDRANT_MEMORY_LIMIT=28g in .env and recreate the container."

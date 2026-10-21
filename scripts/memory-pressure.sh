#!/usr/bin/env bash
# Quora/Cohere under a fixed 1 GiB container cap:
#
#   scripts/memory-pressure.sh [results-root] [model] [cap] [config]
#
# Defaults: cohere, 1g, configs/memory-pressure.yaml. For BGE-small use
# `scripts/memory-pressure.sh results bge 512m configs/memory-pressure-bge.yaml`.
# Sets QDRANT_MEMORY_LIMIT in .env, recreates the container, re-sources the
# telemetry targets, and runs the config. The runner itself verifies the
# effective cgroup limit before measuring. Restore the normal cap afterwards
# (QDRANT_MEMORY_LIMIT=28g) and recreate the container again.
set -euo pipefail
root="${1:-results}"
model="${2:-cohere}"
cap="${3:-1g}"
config="${4:-configs/memory-pressure.yaml}"
stage="$(basename "$config" .yaml)"
out="$root/quora-$model/$stage"
if [ -e "$out" ]; then
  echo "$out already exists; move it aside before rerunning" >&2
  exit 1
fi
[ -f .env ] || cp .env.example .env
if grep -q '^QDRANT_MEMORY_LIMIT=' .env; then
  sed -i "s/^QDRANT_MEMORY_LIMIT=.*/QDRANT_MEMORY_LIMIT=$cap/" .env
else
  echo "QDRANT_MEMORY_LIMIT=$cap" >> .env
fi
docker compose up -d --force-recreate
for _ in $(seq 1 120); do
  curl --fail --silent "${QDRANT_URL:-http://127.0.0.1:6333}/readyz" >/dev/null && break
  sleep 1
done
# shellcheck source=scripts/vm-env.sh
source scripts/vm-env.sh "$root/environment-$stage"
docker inspect --format '{{.HostConfig.Memory}}' "$QDRANT_CONTAINER_ID" > "$root/environment-$stage/host-config-memory.txt"
uv run qbench readiness --data-root data --datasets quora --models "$model" --require-all
uv run qbench run --config "$config" --data "data/embeddings/quora/$model" --output "$out"
echo "Memory-pressure run complete under $out. Restore QDRANT_MEMORY_LIMIT=28g in .env and recreate the container."

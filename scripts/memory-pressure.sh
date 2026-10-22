#!/usr/bin/env bash
# Rescoring from disk: Quora under a container cap applied after each index
# is built.
#
#   scripts/memory-pressure.sh [results-root] [model] [config]
#
# Defaults: cohere and configs/search-cap.yaml (1 GiB). For BGE-small use
# `scripts/memory-pressure.sh results bge configs/search-cap-bge.yaml` (512 MiB).
# The container keeps its normal 28 GiB cap for ingestion and indexing; the
# runner lowers the cap in place with scripts/set-cap.sh before the measured
# passes of each method and restores it afterwards. An OOM kill under the cap
# is recorded for that method and the run continues.
set -euo pipefail
root="${1:-results}"
model="${2:-cohere}"
config="${3:-configs/search-cap.yaml}"
stage="$(basename "$config" .yaml)"
out="$root/quora-$model/$stage"
if [ -e "$out" ]; then
  if python3 -c 'import json,sys; sys.exit(0 if json.load(open(sys.argv[1]))["status"] == "complete" else 1)' "$out/manifest.json" 2>/dev/null; then
    echo "$out already complete"; exit 0
  fi
  stamp="$(date -u +%Y%m%dT%H%M%SZ)"
  echo "$out is incomplete; keeping it as $out-failed-$stamp and rerunning"
  mv "$out" "$out-failed-$stamp"
fi
# shellcheck source=scripts/vm-env.sh
source scripts/vm-env.sh "$root/environment-$stage"
docker inspect --format '{{.HostConfig.Memory}}' "$QDRANT_CONTAINER_ID" > "$root/environment-$stage/host-config-memory.txt"
uv run qbench readiness --data-root data --datasets quora --models "$model" --require-all
uv run qbench run --config "$config" --data "data/embeddings/quora/$model" --output "$out"
scripts/restore-cap.sh 0
echo "Capped-search run complete under $out."

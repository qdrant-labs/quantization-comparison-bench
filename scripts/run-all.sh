#!/usr/bin/env bash
# Run the complete protocol on the VM, in order, resuming completed stages:
#
#   tmux new -s run-all
#   scripts/run-all.sh [results-root]
#
# 1. Small workloads: scifact and nfcorpus, both models (needs prepared data).
# 2. Stage fiqa and quora with BGE (no timing runs while embedding).
# 3. Large workloads: fiqa and quora, both models.
# 4. Memory-pressure run (Quora/Cohere, 1 GiB cap), restore the cap, then
#    cold-start runs for every artifact.
#
# Run as the repository owner; measured stages use sudo because the Qdrant
# process runs as root inside the container.
set -euo pipefail
root="${1:-results}"
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$PATH"
export HF_HOME="${HF_HOME:-/mnt/bench/hf-cache}"
export QDRANT_URL="${QDRANT_URL:-http://127.0.0.1:6333}"

log() { echo "$(date -u +%FT%TZ) [run-all] $*"; }

as_root() {
  sudo -E env PATH="$PATH" HF_HOME="$HF_HOME" QDRANT_URL="$QDRANT_URL" \
    QBENCH_ALLOW_CACHE_RESET="${QBENCH_ALLOW_CACHE_RESET:-}" bash -c "$*"
}

complete() {
  [ -f "$1/manifest.json" ] && python3 -c 'import json,sys; sys.exit(0 if json.load(open(sys.argv[1]))["status"] == "complete" else 1)' "$1/manifest.json"
}

wait_ready() {
  for _ in $(seq 1 120); do
    curl --fail --silent "$QDRANT_URL/readyz" >/dev/null && return 0
    sleep 1
  done
  log "Qdrant did not become ready"; return 1
}

workload() {
  log "workload $1/$2"
  as_root "source scripts/vm-env.sh $root/environment && scripts/run-workload.sh $1 $2 $root"
}

set_cap() {
  sed -i "s/^QDRANT_MEMORY_LIMIT=.*/QDRANT_MEMORY_LIMIT=$1/" .env
  docker compose up -d --force-recreate >/dev/null
  wait_ready
  log "container recreated with cap $1"
}

log "step 1: small workloads"
for pair in "scifact cohere" "scifact bge" "nfcorpus cohere" "nfcorpus bge"; do
  # shellcheck disable=SC2086
  workload $pair
done

log "step 2: stage fiqa and quora with BGE"
uv run qbench prepare-all --with-bge
uv run qbench readiness --require-all

log "step 3: large workloads"
for pair in "fiqa cohere" "fiqa bge" "quora cohere" "quora bge"; do
  # shellcheck disable=SC2086
  workload $pair
done

log "step 4a: memory pressure"
if complete "$root/quora-cohere/memory-pressure"; then
  log "memory pressure already complete"
else
  as_root "scripts/memory-pressure.sh $root"
fi
set_cap 28g

log "step 4b: cold start"
export QBENCH_ALLOW_CACHE_RESET=1
for dataset in scifact nfcorpus fiqa quora; do
  for model in cohere bge; do
    out="$root/$dataset-$model/cold-start"
    if complete "$out"; then
      log "cold start $dataset/$model already complete"
      continue
    fi
    [ -d "$out" ] && { log "$out exists but is incomplete; move it aside"; exit 1; }
    as_root "source scripts/vm-env.sh $root/environment && uv run qbench run --config configs/cold-start.yaml --data data/embeddings/$dataset/$model --output $out"
  done
done
sudo chown -R "$(id -u):$(id -g)" "$root"
log "all steps complete under $root"

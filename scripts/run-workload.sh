#!/usr/bin/env bash
# Run one dataset/model artifact through the full protocol on the VM:
#
#   scripts/run-workload.sh <dataset> <model> [results-root]
#
# Stages, in order, each skipped when its output already completed:
#   primary   controlled comparison at fixed settings (configs/benchmark.yaml)
#   tuning    development-split sweep (configs/tuning.yaml)
#   select    freeze the fastest eligible point per method and k
#   selected  held-out test evaluation of the frozen settings
#   datatype  turbo4 stored-datatype comparison (configs/datatype.yaml)
#   datatype-binary  binary retrieval rescored against turbo4 storage
#   ablation  fixed-effort rescoring/oversampling ablation (configs/ablation.yaml)
#   throughput  the frozen points at concurrency 8 (queries per second)
#   primary-full, selected-full  scifact and nfcorpus only: the whole official
#             test split instead of the 150-query subset
#
# Source scripts/vm-env.sh first so telemetry targets are exported.
set -euo pipefail

dataset="${1:?usage: scripts/run-workload.sh <dataset> <model> [results-root]}"
model="${2:?usage: scripts/run-workload.sh <dataset> <model> [results-root]}"
root="${3:-results}"
data="data/embeddings/$dataset/$model"
out="$root/$dataset-$model"

if [ -z "${QDRANT_PID:-}" ] || [ -z "${QDRANT_CGROUP_PATH:-}" ] || [ -z "${QDRANT_STORAGE_PATH:-}" ]; then
  echo "Telemetry targets are not exported; run: source scripts/vm-env.sh" >&2
  exit 1
fi

complete() {
  [ -f "$1/manifest.json" ] && python3 -c 'import json,sys; sys.exit(0 if json.load(open(sys.argv[1]))["status"] == "complete" else 1)' "$1/manifest.json"
}

run_stage() {
  local name="$1" config="$2"
  shift 2
  if complete "$out/$name"; then
    echo "[$dataset/$model] $name already complete; skipping"
    return
  fi
  if [ -d "$out/$name" ]; then
    echo "[$dataset/$model] $name exists but is not complete; move it aside before rerunning" >&2
    exit 1
  fi
  echo "[$dataset/$model] $name: $config $*"
  uv run qbench run --config "$config" --data "$data" --output "$out/$name" "$@"
}

mkdir -p "$out"
uv run qbench readiness --data-root data --datasets "$dataset" --models "$model" --require-all
uv run qbench verify --data "$data"
if [ ! -f "$data/ground_truth.json" ]; then
  uv run qbench ground-truth --data "$data"
fi

run_stage primary configs/benchmark.yaml
run_stage tuning configs/tuning.yaml
if [ ! -f "$out/selected.yaml" ]; then
  uv run qbench select --input "$out/tuning" --output "$out/selected.yaml" --base-config configs/benchmark.yaml
fi
run_stage selected "$out/selected.yaml"
run_stage datatype configs/datatype.yaml
run_stage datatype-binary configs/datatype-binary.yaml
run_stage ablation configs/ablation.yaml
run_stage throughput "$out/selected.yaml" --concurrency 8
case "$dataset" in
  scifact|nfcorpus)
    run_stage primary-full configs/benchmark.yaml --max-queries 0
    run_stage selected-full "$out/selected.yaml" --max-queries 0
    ;;
esac
echo "[$dataset/$model] workload complete under $out"

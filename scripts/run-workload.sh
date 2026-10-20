#!/usr/bin/env bash
set -euo pipefail

if [[ $# != 2 ]]; then
    printf 'Usage: bash scripts/run-workload.sh DATASET MODEL\n' >&2
    exit 1
fi
qbench_dataset="$1"
qbench_model="$2"
case "$qbench_dataset" in fiqa|scifact|nfcorpus|quora) ;; *) exit 1 ;; esac
case "$qbench_model" in cohere|bge) ;; *) exit 1 ;; esac
qbench_data="data/embeddings/$qbench_dataset/$qbench_model"
qbench_run="results/$qbench_dataset-$qbench_model"
qbench_frozen="$qbench_run/selected-test.yaml"
if [[ -e "$qbench_run" ]]; then
    printf 'Preserve the existing results before starting a new run: %s\n' "$qbench_run" >&2
    exit 1
fi
source scripts/vm-env.sh
mkdir -p "$qbench_run/environment"
cp -a results/environment/. "$qbench_run/environment/"
uv run qbench run --config configs/benchmark.yaml --data "$qbench_data" --output "$qbench_run/controlled"
uv run qbench run --config configs/tuning.yaml --data "$qbench_data" --output "$qbench_run/dev"
uv run qbench freeze --input "$qbench_run/dev" --output "$qbench_frozen"
uv run qbench run --config "$qbench_frozen" --data "$qbench_data" --output "$qbench_run/heldout"
uv run qbench run --config configs/datatype.yaml --data "$qbench_data" --output "$qbench_run/datatype"

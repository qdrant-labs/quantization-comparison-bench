# Qdrant quantization comparison

Prepare and run a controlled Qdrant benchmark on a dedicated Linux VM. Compare
scalar, binary, product, and TurboQuant quantization on four English retrieval
datasets, with supplied Cohere Embed English v3 vectors (1,024 dimensions) and
locally generated BGE-small-en-v1.5 vectors (384 dimensions).

This repository contains data preparation, development tuning, held-out comparison,
and a synthetic smoke workflow. It does not contain benchmark findings.
Synthetic results verify the pipeline; they do not establish retrieval quality or
production performance.

## Start on the VM

Requirements: Linux x86-64, Docker Engine with Compose, Git, and
[uv](https://docs.astral.sh/uv/getting-started/installation/). The working machine
assumption is 8 vCPUs, 32 GiB RAM, and 200 GiB SSD; record the actual machine.

```bash
git clone https://github.com/qdrant-labs/quantization-comparison-bench.git
cd quantization-comparison-bench
uv sync --locked
cp .env.example .env
docker compose up -d
curl --fail http://127.0.0.1:6333/readyz
uv run qbench doctor
```

If readiness is not yet available, wait for startup and repeat the readiness
check. The Compose service binds to VM loopback and stores its data under
`data/qdrant/`. See [VM operation](docs/vm.md) for SSH forwarding, memory limits,
version recording, and run isolation.

## Execute the smoke experiment

```bash
uv run qbench fixture --output data/synthetic
uv run qbench verify --data data/synthetic
uv run qbench ground-truth --data data/synthetic
uv run qbench run --config configs/smoke.yaml --data data/synthetic --output results/smoke
uv run qbench summarize --input results/smoke
```

The fixture uses deterministic synthetic vectors and relevance labels, with 256
documents, 32 queries, and 384 dimensions. It needs no model download or paid API.
Use a dedicated Qdrant instance for experiments; the runner creates and removes
its experiment collections. Read `qbench run --help` before using an existing
server.

The [Colab smoke notebook](notebooks/smoke.ipynb) runs the same workflow against
a user-supplied HTTPS Qdrant endpoint. Download a source ZIP from the private
repository while signed into GitHub, then upload it in the notebook; no GitHub
token is stored in Colab. It requires an accessible experimental server; Colab
does not run this Docker Compose service. Colab timings include its network path
and are not the VM benchmark results.

## Prepare data before provisioning

The four corpora are **FiQA, SciFact, NFCorpus, and Quora**, evaluated separately.
The [CohereLabs BEIR dataset](https://huggingface.co/datasets/CohereLabs/beir-embed-english-v3)
already supplies English-v3 vectors and relevance judgments. The `cohere` path
imports those vectors; it requires no API key and makes no paid embedding calls.
The second model, BGE, encodes the same document and query IDs locally.

Stage the full source data and Cohere vectors:

```bash
uv run --extra datasets qbench prepare-all
uv run qbench readiness --data-root data
```

The readiness report lists every raw-data and model artifact and writes
`data/readiness.json`. A missing BGE entry means its vectors still need generating;
downloaded model weights alone do not count as completed embeddings.

Generate BGE as well, either locally before transfer or on the VM before timing
Qdrant:

```bash
uv run --extra datasets --extra bge qbench prepare-all --with-bge --device cpu
uv run qbench readiness --data-root data --require-all
```

BGE also supports `--device mps` or `--device cuda` on compatible preparation
machines. Keep one device and software configuration for each artifact; completed
batch checkpoints support resume with identical settings. BGE generation is
separate from benchmark timing.

Dataset revision `f018922a46a51388348a307ec2fef28019a33026` and BGE revision
`5c38ec7c405ec4b44b94cc5a9bb96e735b38267a` are pinned. Source Parquet sizes and
SHA-256 values are locked in `src/quantization_comparison_bench/dataset_sources.json`.
See [data preparation](docs/data.md) for splits, normalization and the artifact
layout. Keep the large local `data/` directory outside Git; transfer it to the
same location in the VM checkout and rerun `readiness --require-all` there.

## Tune, freeze, then evaluate

Each run handles **one dataset/model artifact**. Repeat this sequence for all
four datasets and both models; the lists printed by `plan` are not an automatic
eight-workload dispatcher.

For FiQA/Cohere:

```bash
uv run qbench plan --config configs/tuning.yaml
uv run qbench verify --data data/embeddings/fiqa/cohere
uv run qbench ground-truth --data data/embeddings/fiqa/cohere
uv run qbench run --config configs/tuning.yaml --data data/embeddings/fiqa/cohere --output results/fiqa-cohere-dev
uv run qbench freeze --input results/fiqa-cohere-dev --output configs/frozen/fiqa-cohere.yaml
uv run qbench run --config configs/frozen/fiqa-cohere.yaml --data data/embeddings/fiqa/cohere --output results/fiqa-cohere-test
uv run qbench summarize --input results/fiqa-cohere-test
uv run qbench run --config configs/datatype.yaml --data data/embeddings/fiqa/cohere --output results/fiqa-cohere-datatype
```

The development sweep combines `hnsw_ef=64/128/256/512`, rescoring off/on, and
oversampling 1/2/4/8. `freeze` selects the fastest qualifying development setting
per method and k: ANN recall at least 0.95 and nDCG@10 no more than 0.01 below
the exact float32 reference. The frozen file retains selection provenance and
uses three builds, five measured passes, and a seeded sample of 150 official
test queries. Raw artifacts retain all official test queries. If no setting
qualifies, extend the development sweep or preregister a revised target; do not
choose settings from test results.

`configs/benchmark.yaml` is a fixed-setting comparison with one representative
configuration per family. It is distinct from the quality-matched frozen
comparison. Use `configs/ablations.yaml` to isolate rescoring/oversampling at fixed
search effort. The `turbo4` datatype experiment is separate. Keep full tables in
the results directory and present only the useful trade-offs in the guide.
Turbo8 requires separately verified released server/client support. See the
[protocol](docs/protocol.md) for measurement and interpretation.
The [review follow-up](docs/review-followup.md) maps the setup and guide changes
to all 22 review threads.

## What still needs a measured run

- Verify all prepared data artifacts; freeze machine details, image digest and settings.
- Generate exact rankings and select settings only on development queries.
- Repeat index builds and report per-dataset quality with uncertainty.
- Collect VM memory, storage, and I/O alongside search results. Heap alone is not
  the resident working set.
- Run the constrained-memory profile and verify actual I/O. Document controlled
  cold-cache experiments separately; a warm run or process restart is not proof
  of a cold OS page cache.

The full design intentionally exceeds the small smoke workflow. No claim that the
full benchmark is complete should be based on a successful smoke run.

## Development

```bash
uv sync --locked
uv run ruff check .
uv run ruff format --check .
uv run ty check src/
```

CI checks the code and offline fixture without downloading datasets or calling
embedding APIs. A separate `live-smoke` job runs the synthetic workflow against
the pinned Qdrant release. Keep raw data, vectors, secrets, and
generated results out of Git; publish reviewed result artifacts with the exact
run manifest when the experiments are complete.


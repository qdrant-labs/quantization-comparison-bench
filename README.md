# Qdrant quantization comparison

Experiment scaffold for a dedicated Linux VM. Compare scalar, binary, product,
and TurboQuant quantization on the same English retrieval datasets, with Cohere
Multilingual v3 (1,024 dimensions) and BGE-small-en-v1.5 (384 dimensions).

This repository contains an executable synthetic smoke workflow and the starting
pipeline for real-data experiments. It does not contain benchmark findings.
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

## Prepare the real experiment

First inspect the planned matrix:

```bash
uv run qbench plan --config configs/benchmark.yaml
```

The four corpora are **FiQA, SciFact, NFCorpus, and Quora**, evaluated separately.
Both models embed the same document/query IDs. The linked
[CohereLabs BEIR dataset](https://huggingface.co/datasets/CohereLabs/beir-embed-english-v3)
contains English-v3 vectors; those precomputed vectors are not the selected
Multilingual-v3 embeddings.

Dataset and BGE model revisions must be immutable Hugging Face commit IDs. The
following examples use verified repository revisions; record these alongside the
resulting embedding hashes. Set `COHERE_API_KEY` securely in the process
environment before the Cohere command.

```bash
uv run --extra datasets qbench prepare --dataset fiqa --revision f018922a46a51388348a307ec2fef28019a33026
uv run qbench embed --dataset fiqa --model cohere
uv run --extra bge qbench embed --dataset fiqa --model bge --revision 5c38ec7c405ec4b44b94cc5a9bb96e735b38267a
```

Do not commit credentials. Cohere embedding makes paid API calls. Review the
corpus, truncation policy, and expected cost before running it. BGE downloads the
model and optional local-model dependencies.
Repeat for `scifact`, `nfcorpus`, and `quora`, then build float32 ground truth and
run each dataset/model artifact independently. Use `qbench --help` and the
subcommand help for paths and available options.

For the FiQA/Cohere artifact, after embedding:

```bash
uv run qbench verify --data data/embeddings/fiqa/cohere
uv run qbench ground-truth --data data/embeddings/fiqa/cohere
uv run qbench run --config configs/benchmark.yaml --data data/embeddings/fiqa/cohere --output results/fiqa-cohere-primary
uv run qbench summarize --input results/fiqa-cohere-primary
uv run qbench run --config configs/datatype.yaml --data data/embeddings/fiqa/cohere --output results/fiqa-cohere-datatype
```

Use `data/embeddings/fiqa/bge` and different output directories for BGE. Each run
handles one artifact; the dataset/model lists in the plan do not automatically
dispatch all eight workloads. The default configurations report test queries.
Copy the configuration and set `search.split: dev` when selecting settings, then
freeze the choices before the test runs.

The [protocol](docs/protocol.md) defines the full research comparison. The core
matrix tests rescoring off/on, oversampling 1/2/4/8, and limits 10/100. The
`turbo4` datatype experiment is separate. Turbo8 is conditional on a verified
release and client API, not a default experiment.

Use `configs/tuning.yaml` for the development split before freezing choices for
`configs/benchmark.yaml`. The runner executes one dataset/model at a time;
repeat the commands for both models and all four datasets. The synthetic smoke
configuration exercises all implemented quantizers and the `turbo4` datatype.

## What still needs a measured run

- Freeze dataset/model revisions, machine details, image digest, and all settings.
- Generate real embeddings and exact rankings; select settings only on development
  queries, then report untouched test queries.
- Repeat index builds and report per-dataset quality with uncertainty.
- Collect VM memory, storage, and I/O alongside search results. Heap alone is not
  the resident working set.
- Run controlled cold-cache and constrained-memory experiments explicitly. The
  default warm workflow does not create either condition.

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

# Qdrant quantization comparison

Reproducible VM benchmark comparing scalar, binary, product, and TurboQuant
quantization in Qdrant v1.19.2 on four English BEIR datasets (FiQA, SciFact,
NFCorpus, Quora) with two embedding models: the supplied Cohere
embed-english-v3.0 vectors (1,024 dimensions) and locally generated
BGE-small-en-v1.5 vectors (384 dimensions). A separate experiment compares
`turbo4` stored vectors with float32 originals plus a TurboQuant search copy.

The repository holds the protocol, the pinned data manifest and preparation
commands, the runner, and the VM scripts. It does not contain benchmark
findings. Large data files stay outside Git; `data/` and `results/` are
ignored.

## 1. Set up the VM

Dedicated Linux x86-64 VM, 8 vCPUs, 32 GiB RAM, 200 GiB SSD, no GPU. Install
Docker Engine with the Compose plugin, Git, and
[uv](https://docs.astral.sh/uv/getting-started/installation/). Then:

```bash
git clone https://github.com/qdrant-labs/quantization-comparison-bench.git
cd quantization-comparison-bench
uv sync --locked --extra bge
cp .env.example .env
docker compose up -d
curl --fail http://127.0.0.1:6333/readyz
uv run qbench doctor
```

Compose binds Qdrant to the VM loopback, caps the container at 28 GiB with
swap disabled, and stores data under `data/qdrant/`. Before measured runs,
resolve the image digest and pin it in `.env` as described in
[VM operation](docs/vm.md).

Verify the pipeline once with the synthetic smoke workflow. It needs no
download or API key and its numbers are not results:

```bash
uv run qbench fixture --output data/synthetic
uv run qbench verify --data data/synthetic
uv run qbench ground-truth --data data/synthetic
uv run qbench run --config configs/smoke.yaml --data data/synthetic --output results/smoke
uv run qbench summarize --input results/smoke
```

## 2. Stage the datasets

One command stages all four corpora, their query splits and relevance
judgments, and the Cohere arrays from the pinned Hugging Face revision. Add
`--with-bge` to generate the BGE vectors on the VM CPU afterwards (Quora takes
the longest; run it before any timing).

```bash
uv run qbench prepare-all --with-bge
uv run qbench readiness --require-all
```

`prepare-all` is idempotent: artifacts whose checksums verify are skipped.
`readiness` checks hashes, dimensions, IDs, splits, labels, and ground-truth
freshness for every dataset/model artifact and exits non-zero if anything is
missing or inconsistent. No Cohere API calls are made at any point; the
dataset revision and BGE revision are pinned in the CLI defaults and recorded
in every manifest. Details: [data](docs/data.md).

## 3. Run a workload

Source the VM environment script once per shell. It records the machine,
kernel, swap, container image digest, repository commit, and lockfile hash
under `results/environment/`, and exports the telemetry targets (process ID,
cgroup path, storage path) the runner samples every 100 ms.

```bash
source scripts/vm-env.sh
scripts/run-workload.sh fiqa cohere
```

`run-workload.sh` takes one dataset/model artifact through the whole protocol
and skips any stage that already completed:

| Stage | Config | What it does |
| --- | --- | --- |
| `primary` | `configs/benchmark.yaml` | Controlled comparison at fixed settings: scalar8, binary1, product8, turbo4 against float32 HNSW, rescoring on, oversampling 2, `hnsw_ef` 256, top-10 and top-100, 150 seeded test queries, 5 warm passes, 3 index builds |
| `tuning` | `configs/tuning.yaml` | Development sweep over `hnsw_ef` 64/128/256/512, oversampling 1/2/4/8, rescoring off/on |
| `select` | `qbench select` | Per method and k, the fastest dev point with ANN Recall@k ≥ 0.95 and nDCG@10 within 0.01 of exact float32; frozen to `selected.yaml` before any test query is read |
| `selected` | `selected.yaml` | Held-out evaluation of the frozen points; misses are reported, never retuned |
| `datatype` | `configs/datatype.yaml` | `turbo4` stored datatype versus float32 plus TurboQuant search copies |
| `ablation` | `configs/ablation.yaml` | Fixed-effort rescoring/oversampling ablation plus product16, turbo2, turbo1 |

Repeat for `scifact`, `nfcorpus`, `quora` and for `bge`. The two extra
experiments are separate because they change the host condition:

```bash
scripts/memory-pressure.sh                 # Quora/Cohere under a fixed 1 GiB container cap
QBENCH_ALLOW_CACHE_RESET=1 uv run qbench run --config configs/cold-start.yaml \
  --data data/embeddings/fiqa/cohere --output results/fiqa-cohere/cold-start
```

The memory-pressure runner refuses to start unless the effective cgroup limit
equals the configured cap. The cold-start config runs `scripts/cache-reset.sh`
before every pass: it stops the container, drops the host page cache, starts
the container, and the pass warms as queries proceed. Both require readable VM
telemetry. See [VM operation](docs/vm.md) and the [protocol](docs/protocol.md).

## Outputs

Every run directory holds `manifest.json` (server version, configuration,
artifact checksums, query IDs, telemetry targets, selection provenance),
per-query `queries.jsonl`, `summary.json`, and per build/method directories
with the collection configuration, storage breakdown, memory snapshots before
and after indexing and search, Prometheus metrics, 100 ms resource samples,
and per-pass I/O and page-fault counters. How to read them:
[results](docs/results.md).

Turbo8 is excluded until a release with verified server and client support
exists; the merged pull requests are not in v1.19.2.

## Development

```bash
uv sync --locked
uv run ruff check .
uv run ruff format --check .
uv run ty check src/
uv run pytest
```

CI runs those checks, the offline fixture, every shipped configuration through
`qbench plan`, and a live synthetic smoke job against the pinned Qdrant
release. The [Colab smoke notebook](notebooks/smoke.ipynb) runs the synthetic
workflow against a user-supplied HTTPS endpoint; its timings are not VM
results. Keep raw data, vectors, secrets, and generated results out of Git.

# Experimental protocol

## Question

At fixed corpus, query IDs, embedding model, graph-construction settings, and
hardware, how do scalar, binary, product, and TurboQuant quantization compare
on retrieval quality, latency, resident memory, and storage? Separately, how
do `turbo4` stored vectors compare with float32 originals plus a TurboQuant
search copy?

This document specifies the experiment. It reports no measurements.

## Data and embeddings

Four English BEIR datasets, each evaluated separately: FiQA, SciFact,
NFCorpus, Quora. Cohere `embed-english-v3.0` vectors (1,024 dimensions) are
imported from the pinned Hugging Face dataset revision; `BAAI/bge-small-en-v1.5`
vectors (384 dimensions) are generated locally for the same document and query
IDs. No embedding API calls are required. Revisions, source file hashes,
normalized float32 arrays, ordered IDs, judgments, split assignments, and
preprocessing details are saved under `data/` in the benchmark checkout and
are listed in every run manifest. See [data](data.md).

Fifty seeded development queries per dataset come from the official dev split
(train for SciFact). Default measurements use 150 seeded queries from the
official test split, held out during tuning; the full test split stays
available for extended runs. Self matches are excluded consistently,
especially for Quora. Original BEIR relevance grades are preserved.

Differences between the two models include the embedding model and vector
distribution; they cannot be attributed to dimension alone. Two models cannot
establish universal rules.

## Software and VM

Qdrant v1.19.2, the latest published release verified on 20 October 2026. The
resolved container digest, repository commit, and Python lockfile hash are
recorded by `scripts/vm-env.sh` before measurements. Turbo8 stays excluded
until released server and client support is verified in a separate protocol.

Dedicated Linux x86-64 VM, 8 vCPUs, 32 GiB RAM, 200 GiB SSD, no GPU. The
client runs on the VM. One active collection, one shard, one replica. CPU,
disk, kernel, thread counts, and swap are recorded. The normal container cap is
28 GiB with swap disabled.

## Comparison and selection

Exact float32 neighbors are computed independently of Qdrant; float32 HNSW is
the approximate baseline. HNSW `m=16` and `ef_construct=100` are fixed, with eight indexing threads for
every reported run (build time and build-time CPU depend on it; search results
do not).

**Primary comparison** (`configs/benchmark.yaml`): scalar8 (int8, quantile
0.99), binary1 (one-bit storage, binary query encoding), product8, and
TurboQuant bits4, with rescoring enabled and oversampling 2 at `hnsw_ef=256`.
Independent top-10 and top-100 requests.

**Development selection** (`configs/tuning.yaml`, then `qbench select`):
sweep `hnsw_ef` 64/128/256/512, oversampling 1/2/4/8, and rescoring off/on on
development queries. For each method and k, select the fastest eligible point
(server p95 by default) with ANN Recall@k at least 0.95 and mean nDCG@10 no
more than 0.01 below exact float32. The chosen points are frozen into a
test-split configuration before any test query is read. A method with no
eligible point keeps its highest-recall point and is reported as a miss;
nothing is retuned on test.

**Ablations**: `configs/ablation.yaml` holds the fixed-effort
rescoring/oversampling ablation together with the extended set from the
modern-quantization-bench reference: binary 1.5-bit and 2-bit encodings, binary
1-bit with an 8-bit scalar query, TurboQuant bits2, bits1_5, bits1, and
product16. `configs/datatype.yaml` compares float32 without
quantization, float32 plus TurboQuant bits4 with and without rescoring, the
`turbo4` datatype without a search copy, and the bits1 pairing. Rescoring
against `turbo4` storage cannot restore float32 precision, and changing the
stored datatype can affect index construction.

Stored vectors are placed in cold memory and quantized copies in pinned
memory; HNSW placement is fixed. All search parameters are set explicitly. The
server's effective collection configuration is logged. Unsupported
configurations fail; they are never silently replaced.

## Measurements

- **ANN quality**: Recall@10 and Recall@100 as overlap with the exact float32
  top-k of the same model.
- **Relevance**: nDCG@10 against BEIR judgments, per dataset; an equally
  weighted macro-average is a secondary summary only.
- **Latency**: server processing time and client wall-clock p50/p95, with
  concurrency one, embedding generation excluded, and no vectors or payloads
  returned.
- **Memory**: allocator heap deltas (`heap_mb` in the reference bench), process
  RSS, cgroup current/peak, anonymous/file-backed breakdown, sampled every 100 ms, with snapshots
  before indexing, after indexing, and after search. Indexing and query
  intervals are separate; overlapping counters are never summed.
- **Storage**: allocated and logical collection bytes by file while the
  collection exists, classified into vectors, quantized data, HNSW, payload,
  and WAL during analysis.
- **I/O**: per-query CPU and vector/payload I/O read counters from the server,
  process read/write bytes, minor/major page faults, and cgroup `io.stat`
  before and after every pass.

Five measured warm passes and three independently built indexes for the
primary comparison. Per-query records are retained; paired uncertainty across
queries and build variation are reported instead of single-run precision.

## Memory pressure and cache state

`configs/memory-pressure.yaml` runs Quora/Cohere with a fixed 1 GiB container
cap, below the roughly 2 GiB float32 corpus. The same cap applies to every
method; OOMs, timeouts, and build failures are recorded as outcomes. The
runner checks the effective cgroup limit and requires VM telemetry before
starting.

Warm and cold-start results are kept separate. `configs/cold-start.yaml`
runs the dedicated-VM cache-reset hook (`scripts/cache-reset.sh`) before each
pass: stop the container, drop the host page cache, start the container. The
pass then warms as queries proceed, so per-query `position` is part of each
record. A process restart alone does not establish cold filesystem caches.
Actual reads and page faults are inspected before any disk effect is
interpreted.

## Execution

```bash
uv run qbench prepare-all --with-bge
uv run qbench readiness --require-all
source scripts/vm-env.sh
scripts/run-workload.sh <dataset> <model>
scripts/memory-pressure.sh
```

Synthetic and small real-data smoke runs validate execution; they are not
benchmark findings. Measured comparisons, the author's analysis, and the
Dylan/Neil publication review follow the VM runs. The article remains 3,000 to
5,000 words with author-owned analysis and runnable snippets.

## Sources

- [Qdrant quantization](https://qdrant.tech/documentation/manage-data/quantization/)
- [Turbo4 stored-vector datatype](https://qdrant.tech/documentation/manage-data/vectors/#turbo4)
- [Collections that outgrow RAM](https://qdrant.tech/documentation/search-tuning/when-your-collection-outgrows-ram/)
- [Qdrant releases](https://github.com/qdrant/qdrant/releases)
- [CohereLabs BEIR embeddings](https://huggingface.co/datasets/CohereLabs/beir-embed-english-v3)
- [BGE-small-en-v1.5 model card](https://huggingface.co/BAAI/bge-small-en-v1.5)
- [BEIR datasets and evaluation](https://github.com/beir-cellar/beir)
- Turbo8 (not in v1.19.2): [qdrant#10874](https://github.com/qdrant/qdrant/pull/10874), [qdrant#10939](https://github.com/qdrant/qdrant/pull/10939)

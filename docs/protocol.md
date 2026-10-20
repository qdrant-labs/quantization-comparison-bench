# Experimental protocol

## Questions

At fixed corpus, queries, embedding model, index construction settings, and
hardware, how do scalar, binary, product, and TurboQuant quantization compare on
search quality, resident memory, storage, and latency? How do oversampling and
rescoring change those trade-offs?

Separately: what changes when `turbo4` replaces stored float32 vectors instead of
adding a quantized copy? Preserve a float32 reference outside the measured
collection for every comparison.

This is the experiment specification. It does not report measurements.

## Data and embeddings

Use the full English FiQA, SciFact, NFCorpus, and Quora corpora separately.
Language is not an experimental axis. Import supplied `embed-english-v3.0`
vectors (1,024 dimensions) from the
[CohereLabs English-v3 dataset](https://huggingface.co/datasets/CohereLabs/beir-embed-english-v3),
then generate `BAAI/bge-small-en-v1.5` vectors (384 dimensions) for the same IDs.
The first path is a precomputed-vector import and requires no paid API.
Differences between the models include their training, preprocessing and vector
distribution; they cannot be attributed to dimension alone.

Freeze the dataset revision, source file checksums, BGE model/tokenizer revision,
title/body concatenation, truncation, output type, preparation device and
normalization. The Cohere vectors retain their publisher's preprocessing;
record their provenance rather than claiming to regenerate them. Save hashes of
vector and ID files, and reuse those exact arrays across quantization methods.
The pinned revisions and artifact layout are in [data preparation](data.md).

For BGE on FiQA, SciFact and NFCorpus, prefix queries with
`Represent this sentence for searching relevant passages: ` and leave document
text unprefixed. Quora is symmetric question retrieval: use no passage instruction
on either BGE side. Record these task choices; never vary them across quantization
configurations.

Use cosine consistently. Qdrant normalizes cosine vectors at ingestion. Explicit
normalization of saved arrays keeps the independent exact ranking consistent;
it is not an additional server speedup.

Use existing BEIR relevance judgments. The loader reserves 50 seeded development
queries per dataset from its dev split, or the train split for SciFact, alongside
the official test queries. Choose settings with `search.split: dev`, then use
`search.split: test` after freezing them. The primary run selects 150 official
held-out test queries per dataset deterministically using seed 42; every method
and model uses the same selected IDs. Report this query subset explicitly.
Prepared artifacts retain all official test queries for an additional full-test
run. A corpus subset remains a smoke test and cannot stand in for the full-corpus
experiment.

Remove self matches consistently from exact, approximate, and relevance rankings,
particularly for Quora. A top-10 request must yield ten eligible candidates.

## Baselines and matrix

For each dataset/model pair, compute exact cosine rankings against the float32
corpus independently of Qdrant's approximate index. Report float32 HNSW as the
unquantized approximate baseline.

| Family | Configurations |
| --- | --- |
| Float32 HNSW | No additional quantization |
| Scalar | int8, quantile 0.99 |
| Binary | 1-bit storage, explicit query encoding |
| Product | x8 primary; x16 as a targeted follow-up |
| TurboQuant | bits4 primary; bits2/bits1 as targeted follow-ups |
| Turbo8 | Only after verifying released server/client support |

Use one representative configuration per family for the initial fixed-setting
comparison in `configs/benchmark.yaml`. Keep HNSW construction fixed at `m=16`
and `ef_construct=100`; graph placement and indexing thresholds are also fixed.
Issue independent `limit=10,100` requests and report expanded candidate budgets.
Oversampling expands work; it is not a free precision increase.

`configs/ablations.yaml` holds search effort fixed while comparing quantization,
rescoring off/on and oversampling 1/2/4/8. `configs/tuning.yaml` instead sweeps
`hnsw_ef=64/128/256/512` jointly with those rescoring/oversampling settings, using
only the 50 development queries. Separate these two purposes when reporting:
the ablation isolates effects, while tuning finds useful operating points.

Keep original float32 vectors in the primary matrix. Put stored vectors in cold
memory and quantized copies in pinned memory; keep HNSW placement fixed. Pin the
server and client versions and log the actual collection configuration returned
by the server. Record indexing thresholds and ensure HNSW is built even for the
small corpora. Do not silently replace an unsupported configuration.

The preregistered `qbench freeze` target is ANN recall at least 0.95 at each
requested k, with nDCG@10 no more than 0.01 below exact float32. It chooses the
qualifying operating point with the lowest mean development client p50 for each
method and k. The output records development provenance and writes a held-out
configuration with three builds, five measured passes and 150 test queries.
Measure p95 and memory as well as the selection metric.

Freeze independently for each dataset/model. If no point qualifies, extend the
development sweep or preregister a revised target before examining test results.
Report held-out failures without retuning. A fixed-setting run demonstrates a
trade-off; it does not establish a speed advantage at matched quality.

## Stored-vector experiment

Compare float32 without quantization; float32 plus TurboQuant bits4 with and
without rescoring; and `turbo4` datatype without an additional quantized copy.
Optionally compare float32 plus bits1 against turbo4 plus bits1 under matching
search settings. Rescoring with stored turbo4 vectors cannot recover float32
precision. Changing the stored datatype can also affect index construction.

## Measurement

- **ANN quality:** Recall@10 and Recall@100 as overlap with exact float32 top-k,
  for each model/dataset and against its own reference.
- **Relevance:** nDCG@10 against BEIR judgments, per dataset and equally weighted
  macro-average. Exact-neighbor overlap and relevance are separate metrics.
- **Memory:** allocator heap where available, process RSS, cgroup current/peak,
  anonymous/file-backed breakdown, and pages loaded for rescoring. Build peaks
  and steady-state search measurements are separate. Do not sum overlapping
  counters.
- **Storage:** allocated collection bytes with vector, quantization, HNSW, payload,
  and WAL breakdown. Do not include the external exact-reference arrays in the
  measured collection footprint.
- **Latency:** server processing and client wall-clock p50/p95, plus I/O and page
  faults where accessible. Exclude embedding generation and omit response vectors
  and payloads.

Use concurrency one for the primary run, a common randomized query order, at
least five warm passes, and at least three independently built indexes. Preserve
per-query samples. Report paired confidence intervals across queries and index
build variation instead of precision implied by a single run.

Warm each workload before timing. For cold measurements, control the relevant
Linux cache state and verify it using I/O counters; a process restart alone does
not clear the OS page cache. Separate independent cold requests from a cold-start
pass that gradually warms. The initial runner records warm samples; it does not
drop caches.

Quora's raw 1,024-dimensional float32 array is approximately 2 GiB. A 32 GiB host
therefore does not automatically exercise disk-bound rescoring. Add a separate
Quora memory-pressure run with one fixed cgroup cap per model, selected in
preflight and held constant across methods. Verify that
`corpus_count × dimensions × 4` exceeds Qdrant's effective memory cap. Keep the
originals on disk and the quantized copy placement fixed; record swap policy,
disk reads and page faults. Record OOMs, timeouts and index failures as outcomes.
The tested cap is not an estimate of every deployment's minimum RAM.

## Reproducibility and interpretation

Freeze source revisions, lockfile, server image digest, run configuration, ID and
vector hashes, insertion order, index build identity, query order/seed, hardware,
kernel, thread counts, cache condition, memory limit, and client transport.
Use one active collection, one shard, and one replica. Archive raw results before
summarizing. Never merge results across server releases without rerunning controls.

Inspect sign balance, coordinate means/variances, outliers, anisotropy, and
neighbor-score margins to explore explanations of quantization error. Two models
cannot establish universal rules or prove a benefit from quantization-aware
training. Avoid converting a descriptive pattern into a causal conclusion.

The later article remains 3,000–5,000 words with author-owned analysis, runnable
snippets, and Dylan/Neil factual review. This repository is implementation work;
it does not supply the article's conclusions.

## Sources

- [Qdrant quantization](https://qdrant.tech/documentation/manage-data/quantization/)
- [Turbo4 stored-vector datatype](https://qdrant.tech/documentation/manage-data/vectors/#turbo4)
- [Collections that outgrow RAM](https://qdrant.tech/documentation/search-tuning/when-your-collection-outgrows-ram/)
- [Qdrant releases](https://github.com/qdrant/qdrant/releases)
- [CohereLabs BEIR embeddings](https://huggingface.co/datasets/CohereLabs/beir-embed-english-v3)
- [BGE-small-en-v1.5 model card](https://huggingface.co/BAAI/bge-small-en-v1.5)
- [BEIR datasets and evaluation](https://github.com/beir-cellar/beir)

Treat the documentation as explanatory context. The pinned server/client source
and recorded effective collection settings determine the implementation used in
each run.


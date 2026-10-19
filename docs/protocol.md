# Experimental protocol

## Questions

At fixed corpus, queries, embedding model, index construction settings, and
hardware, how do scalar, binary, product, and TurboQuant quantization compare on
search quality, resident memory, storage, and latency? How do oversampling and
rescoring change those trade-offs?

Separately: what changes when `turbo4` replaces stored float32 vectors instead of
adding a quantized copy? Preserve a float32 reference outside the measured
collection for every comparison.

This is an experiment specification and scaffold, not a report of measurements.

## Data and embeddings

Use the full English FiQA, SciFact, NFCorpus, and Quora corpora separately.
Language is not an experimental axis. Re-embed identical IDs and text with
`embed-multilingual-v3.0` (1,024 dimensions) and
`BAAI/bge-small-en-v1.5` (384 dimensions). Differences between the two include the
embedding model and vector distribution; they cannot be attributed to dimension
alone.

Freeze title/body concatenation, truncation, model/tokenizer revisions, API model
name/date, output type, and normalization. Save hashes of vector and ID files.
The [CohereLabs English-v3 dataset](https://huggingface.co/datasets/CohereLabs/beir-embed-english-v3)
is a reference for the available corpora, not a source of the chosen model's
precomputed vectors.

For FiQA, SciFact, and NFCorpus, use Cohere `search_document` for documents and
`search_query` for queries. For BGE, prefix queries with
`Represent this sentence for searching relevant passages: ` and leave document
text unprefixed. Quora is symmetric question retrieval: use Cohere `search_query`
on both sides and no passage instruction on either BGE side. Record these task
choices; never vary them across quantization configurations.

Use cosine consistently. Qdrant normalizes cosine vectors at ingestion. Explicit
normalization of saved arrays keeps the independent exact ranking consistent;
it is not an additional server speedup.

Use existing BEIR relevance judgments. The loader reserves 50 seeded development
queries per dataset from its dev split, or the train split for SciFact, alongside
the official test queries. Choose settings with `search.split: dev`, then use
`search.split: test` after freezing them. Report full official test splits. A
pilot subset must be labelled as such.

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
| Product | x8 and x16 compression |
| TurboQuant | bits4 primary; bits2 and bits1 sensitivity |
| Turbo8 | Only after verifying released server/client support |

For quantized methods, cross `rescore=false,true` with
`oversampling=1,2,4,8`, and issue independent `limit=10,100` requests. Set all
parameters explicitly, including `hnsw_ef`. Hold requested HNSW settings fixed and
report expanded candidate budgets. Oversampling expands work; it is not a free
precision increase.

Keep original float32 vectors in the primary matrix. Put stored vectors in cold
memory and quantized copies in pinned memory; keep HNSW placement fixed. Pin the
server and client versions and log the actual collection configuration returned
by the server. Record indexing thresholds and ensure HNSW is built even for the
small corpora. Do not silently replace an unsupported configuration.

Run fixed-setting comparisons first. If selecting configurations at a target
quality, preregister the target and select only on development queries. One
example is Recall@10 >= 0.95 with nDCG@10 no more than 0.01 below exact float32.
Report held-out failures without retuning.

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
preflight and held constant across methods. Record failures and I/O as outcomes.
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
- [Cohere embedding API](https://docs.cohere.com/v2/reference/embed)
- [BGE-small-en-v1.5 model card](https://huggingface.co/BAAI/bge-small-en-v1.5)
- [BEIR datasets and evaluation](https://github.com/beir-cellar/beir)

Treat the documentation as explanatory context. The pinned server/client source
and recorded effective collection settings determine the implementation used in
each run.

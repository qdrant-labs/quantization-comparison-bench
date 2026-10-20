# Reading benchmark results

The scaffold produces measurements, not predetermined conclusions. Keep results for each dataset, embedding model, result limit, cache condition, and index build separate before making comparisons. A 100-result request has a different candidate budget from a 10-result request; do not derive the latter's latency by slicing the former.

## Quality definitions

**ANN recall at k** is the overlap between the returned top-k document IDs and exact float32 top-k IDs, divided by the number of eligible exact neighbors (normally k). This measures recovery of the same model's exact neighbors, not recovery of all relevant documents. Keep exact rankings separate for each dataset and embedding model.

Exact reference search reads normalized float32 arrays in blocks, computes dot products, and retains only the current top-k candidates. It never constructs the full query-by-corpus score matrix. Equal scores are ordered by ascending corpus row, so preserve artifact row order and checksums. Floating-point libraries and hardware can affect very close scores. Exclude a document whose ID equals the query ID in both exact and approximate search; a result limit of ten must mean ten eligible documents.

**nDCG at 10** uses each dataset's original relevance judgments as linear gains: `sum(relevance / log2(rank + 1))`, divided by the ideal ranking's discounted gain. Missing and nonpositive judgments contribute zero. A query without positive judgments has nDCG zero; artifact verification rejects such query sets so missing labels do not silently depress aggregate quality. Do not convert gains to `2**relevance - 1`: BEIR's evaluator calls `pytrec_eval` with `ndcg_cut`, and the underlying implementation uses the relevance values themselves. See the primary [BEIR evaluation source](https://github.com/beir-cellar/beir/blob/main/beir/retrieval/evaluation.py) and [trec_eval nDCG source](https://github.com/usnistgov/trec_eval/blob/master/m_ndcg_cut.c).

The self-match exclusion removes results only; preserve the supplied relevance labels when computing ideal DCG, matching BEIR's evaluation procedure. Publish exact float32 and unquantized HNSW nDCG alongside quantized results. Quantization can change relevance in either direction even when its ANN recall falls.

The fixture verifier requires every selected query to have relevance judgments and every judged document to exist in the selected corpus. A demonstration subset should retain its relevant documents and filter judgments consistently; subset measurements are not full BEIR results.

## Aggregation and uncertainty

The summary helper groups records by `dataset`, `model`, `config`, `split`, `k`, `cache_state`, `hnsw_ef`, `rescore`, `oversampling`, and `build`. Every record must include `query_id`. Populate the grouping fields consistently; absent fields become null. Use `ann_recall`, `ndcg_at_10`, `request_ms` (client wall-clock), and `server_ms` (server-reported duration). A configuration label must uniquely identify quantization and other requested search settings. Separate grouping fields preserve the rescoring and oversampling ablations. Keep each build's latency summary separate and report build variation before pooling quality measurements across builds.

Quality is averaged first over repeats of each query and then over distinct queries, so repeated requests do not increase a query's relevance weight. Latency percentiles use all measured requests within a group. Exclude warm-up requests from measured records. The helper reports p50 and p95; it does not fabricate confidence intervals from repeated requests.

For the final analysis, pair each method's quality observations with its baseline using query ID and build ID. Bootstrap query-level differences using shared sampled query IDs, retaining the pairing across methods. Report variation across independently constructed indexes separately. Choose settings using development queries only (`qbench select` applies the preregistered rule: fastest point with ANN Recall@k ≥ 0.95 and nDCG@10 within 0.01 of exact), then freeze settings before opening test results. The frozen configuration and the chosen points, including misses, are copied into the test run's manifest under `selection`. Any overall relevance mean should give each dataset equal weight, rather than allowing Quora's larger query set to dominate.

## Resource and latency interpretation

Three memory views are recorded on purpose, and they answer different questions:

- **Allocator heap** (`server-memory-*.json`, `footprint.json`): Qdrant's jemalloc gauges from `/metrics`, with deltas from the pre-index snapshot to after indexing and after search. This is the `heap_mb` figure of the modern-quantization-bench reference. It excludes memory-mapped file pages, so pages read back for rescoring are not in it.
- **Process RSS** with its anonymous/file-backed split (`memory-*.json`, `resources.jsonl`): what the process actually holds, including mapped original vectors that were touched.
- **Cgroup current/peak** and `memory.stat`: what the container is charged for, including page cache. This is what a deployment needs in RAM.

Per-query hardware usage comes from the query response when the server runs with `QDRANT__SERVICE__HARDWARE_REPORTING=true` (the Compose file sets it): `cpu`, `vector_io_read`, `payload_io_read`, and `payload_index_io_read` are stored on every record and averaged per group as `mean_*_per_query`. `vector_io_read` is non-zero only when cold vectors are read from disk, which is how rescoring reads show up. The units are Qdrant's own; compare configurations with each other. `passes.jsonl` also holds the cumulative per-collection counters before and after every pass.

Report server processing time and client wall-clock time separately. Record protocol, client location, concurrency, result limit, and whether vectors or payloads were returned. Embedding generation and collection construction are not search latency.

Keep warm, cold-start, and controlled cold-request results separate. Cold-start records carry `cache_state: cold_start` and a per-query `position` within the pass; the first queries after a reset are the cold ones. Restarting a process does not clear the OS page cache. Only label a run controlled cold when the cache-reset procedure and disk/page-fault counters support that claim. Managed environments without cache control can report observed conditions.

Use process RSS and container memory counters as separate measurements; they overlap and must not be summed. Record anonymous and file-backed memory where available, build peaks separately from search peaks, and rescoring's page reads. A before/after snapshot is not a peak measurement; use the container's peak counter or a sampled time series. Report collection disk usage including originals, quantized copies, HNSW, payload, and WAL, and identify the measurement tool and whether it counts allocated or apparent bytes.

Compare configurations at both fixed requested settings and a development-selected quality target. List failures and quality-target misses. Hardware, memory caps, index settings, cache procedures, embedding checksums, and server/client versions belong beside the results so readers can reproduce the trade-offs.

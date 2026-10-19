# Frozen datasets and embeddings

Prepare the four English BEIR datasets separately: `fiqa`, `scifact`, `nfcorpus`,
and `quora`. The source is [CohereLabs/beir-embed-english-v3](https://huggingface.co/datasets/CohereLabs/beir-embed-english-v3),
but its existing `emb` columns are **not used**. The two experiment models re-embed
the same documents and queries: Cohere `embed-multilingual-v3.0` (1,024 dimensions)
and `BAAI/bge-small-en-v1.5` (384 dimensions).

Use an immutable 40-character Hugging Face commit SHA for the dataset source.
Store that SHA in the run configuration; do not use `main`. Model weights for BGE
also require an immutable commit SHA. The hosted Cohere model exposes a model ID
rather than a weights revision, so archive its generated arrays and checksums.

## Preparation

`prepare_dataset(dataset, output, revision, dev_queries=50, seed=42, max_docs=None,
max_test_queries=None)` writes `output/<dataset>/`:

- `corpus.jsonl`: `{"id": "...", "text": "..."}`.
- `queries.jsonl`: `{"id": "...", "text": "...", "split": "dev|test"}`.
- `qrels.jsonl`: `{"query_id": "...", "doc_id": "...", "relevance": 1, "split": "dev|test"}`.
- `manifest.json`: source revision, seed, counts, text policy, and SHA-256 checksums.

The source configurations are `<dataset>-corpus`, `<dataset>-queries`, and
`<dataset>-qrels`. Corpus documents are in `train`; that name does not mean the
documents are used to train either embedding model. Queries and judgments retain
their official source splits. Development queries are sampled deterministically
from `dev`, except SciFact, where they come from `train`. Official test queries are
not used for development. The default keeps every official test query. The code
rejects overlapping query IDs between development and test.

Document serialization is stripped title, a newline, and stripped body, with empty
parts omitted. Query text is unchanged. IDs are strings; rows are sorted by ID.
Original integer relevance grades are preserved, including non-positive judgments.
Do not substitute a new 0–3 annotation scale. For Quora, exclude a query's own
document ID consistently in exact ground truth and Qdrant searches.

`max_test_queries` creates a seeded test subset. `max_docs` retains every judged
document for the selected queries, retains corpus documents with the same IDs as
selected queries, then fills the remaining budget with seeded distractors. It
raises an error if the document budget cannot hold those required documents;
reduce query counts or raise the budget. It never silently drops positive labels.
The manifest labels either option as a smoke subset. Such subsets verify execution
and are not the complete-corpus benchmark. Preparation streams source files but
holds the selected corpus text in memory; source Parquet downloads can still include
the unused embedding columns. A document subset reduces embedding costs, not
necessarily the preparation download.

## Embedding

`await embed_dataset(dataset_dir, output, model, batch_size=64, revision=None)`
reads prepared data and validates its checksums. `model` is `cohere` or `bge`.
Use separate output directories per model. Corpus subsetting belongs in preparation;
the legacy `max_documents` argument explicitly rejects truncation at embedding time.

Cohere requires `COHERE_API_KEY` in the environment. Requests use the v2 embedding
endpoint, floating-point output, and `truncate="END"`; a batch may have at most
96 texts. Ordinary retrieval uses `search_document` for documents and `search_query`
for queries. Quora uses `search_query` for both. These are **paid external API calls**
when the embedding command is run; the synthetic fixture makes no calls. See the
[Cohere API contract](https://docs.cohere.com/v2/reference/embed).

BGE runs locally on CPU in float32 with a 512-token maximum and right truncation.
The passage-retrieval instruction is prepended only to queries:
`Represent this sentence for searching relevant passages: `. Quora uses no instruction
on either side. See the [BGE model card](https://huggingface.co/BAAI/bge-small-en-v1.5).
The two tokenizers can retain different text spans at their context limits. Keep
this policy fixed for all quantization methods within a model.

Both paths write L2-normalized float32 matrices. Qdrant cosine collections also
normalize vectors internally; external normalization keeps independent exact
search consistent. It does not add a Qdrant cosine-search speed improvement.

Completed batches are atomically checkpointed under `checkpoints/`. Identical text,
IDs, model settings, source manifest, and batch boundaries reuse those arrays on a
retry. Keep `batch_size` unchanged when resuming to reuse every batch. A timeout or
process failure between a successful API response and its disk write can cause a
repeat charge; this is not an exactly-once billing guarantee. Retryable HTTP status
codes use a bounded backoff. Network timeouts stop the command for an explicit resume.
Credentials are not recorded in files. Do not run multiple embedding writers against
the same output directory.

The final artifact directory contains:

| File | Content |
| --- | --- |
| `corpus.npy` | Float32 matrix, rows in `corpus_ids.json` order |
| `queries.npy` | Float32 matrix, rows in `query_ids.json` order |
| `corpus_ids.json` / `query_ids.json` | Lists of original IDs |
| `query_splits.json` | Query ID to `dev` or `test` |
| `qrels.json` | Query ID to document ID to original integer grade |
| `embedding-config.json` | Resume identity and embedding parameters |
| `manifest.json` | Provenance, creation time, dimensions, versions, checksums |

The full `corpus.npy` is assembled using a memory-mapped file. Checkpoints intentionally
remain available, so budget additional disk space for a second copy of the arrays.
Archive and reuse generated embeddings for all Qdrant configurations. Do not generate
fresh vectors per quantization method.

## Offline fixture

`create_fixture(output, dimensions=384, documents=256, queries=32, seed=42)` produces
the same final artifact format from seeded random unit vectors. Queries are noisy
copies of selected document vectors with one positive label each. Query IDs differ
from all document IDs. A quarter of queries form the development split and the
remainder form the test split. Its manifest marks it as synthetic. The fixture checks
the runner and metrics; it cannot support claims about real embedding quality or
quantization performance.

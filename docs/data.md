# Frozen datasets and embeddings

Prepare four English BEIR tasks: `fiqa` (financial question answering),
`scifact` (scientific claims), `nfcorpus` (medical/nutrition retrieval), and
`quora` (duplicate questions). The source is
[CohereLabs/beir-embed-english-v3](https://huggingface.co/datasets/CohereLabs/beir-embed-english-v3).
The `cohere` path imports its supplied `emb` columns: Cohere
`embed-english-v3.0`, 1,024 dimensions. It does not call an embedding API.
`BAAI/bge-small-en-v1.5` generates 384-dimensional vectors locally for the same
document and query IDs.

The source revision is `f018922a46a51388348a307ec2fef28019a33026`; BGE weights are
pinned to `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a`.
`src/quantization_comparison_bench/dataset_sources.json` records each source
Parquet path, byte size and SHA-256. The loader checks those files before using
them. A different source revision requires a reviewed source lock update.

```bash
uv run --extra datasets qbench prepare-all
uv run qbench readiness --data-root data
uv run --extra datasets --extra bge qbench prepare-all --with-bge --device cpu
uv run qbench readiness --data-root data --require-all
```

The first command stages all four raw datasets and their Cohere vectors.
`--with-bge` also computes BGE. Readiness validates prepared files and reports
missing artifacts; `--require-all` fails unless all four raw datasets and all
eight embedding artifacts are ready. It writes `data/readiness.json`.
Downloaded model weights without final vector manifests are still incomplete.

Source files are cached under `data/sources/`, prepared text under
`data/raw/<dataset>/`, embeddings under `data/embeddings/<dataset>/<model>/`,
and BGE model files under `data/models/`. These are local files in the checkout,
excluded from Git. Copy the prepared directory to the VM and rerun readiness.

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

At the pinned source revision, full preparation with the default 50 development
queries produces these counts:

| Dataset | Corpus documents | Development queries | Official test queries |
| --- | ---: | ---: | ---: |
| FiQA | 57,638 | 50 | 648 |
| SciFact | 5,183 | 50 | 300 |
| NFCorpus | 3,633 | 50 | 323 |
| Quora | 522,931 | 50 | 10,000 |

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
and are not the complete-corpus benchmark. Preparation reads the locked Parquet
sources and holds the selected corpus text in memory; the files include supplied
embeddings. A document subset reduces BGE work, not necessarily the download.

The primary runner's `search.max_queries: 150` is a seeded selection from the
prepared official test queries; it does not truncate the stored corpus or remove
judgments. Raw artifacts retain all test queries so a full-test run can reuse the
same vectors. Always report the selected query IDs and count.

## Embedding

`await embed_dataset(dataset_dir, output, model, batch_size=64, revision=None, device="cpu")`
reads prepared data and validates its checksums. `model` is `cohere` or `bge`.
Use separate output directories per model. Corpus subsetting belongs in preparation;
the legacy `max_documents` argument explicitly rejects truncation at embedding time.

Cohere imports corpus rows from the source `train` split and query rows from the
selected development-source split plus `test`, matching the prepared IDs. The
vectors retain the publisher's embedding preprocessing, with L2 normalization
applied when saving float32 arrays. Do not claim the benchmark regenerated Cohere
embeddings with a newly chosen tokenizer or truncation policy. No Cohere API key,
embedding request or paid service is involved.

BGE runs locally in float32 with a 512-token maximum and right truncation.
CPU is the default; MPS and CUDA are optional preparation devices.
The passage-retrieval instruction is prepended only to queries:
`Represent this sentence for searching relevant passages: `. Quora uses no instruction
on either side. See the [BGE model card](https://huggingface.co/BAAI/bge-small-en-v1.5).
The two models' text handling can differ. Keep the recorded policy and completed
vectors fixed for every quantization method within a model. Device, library
versions and model revision are recorded in artifact identity. Resuming partial
checkpoints requires matching preparation settings. Completed, verified arrays
are portable: an MPS-prepared artifact can be searched on a CPU VM without
re-embedding. Preserve its original generation metadata.

Both paths write L2-normalized float32 matrices. Qdrant cosine collections also
normalize vectors internally; external normalization keeps independent exact
search consistent. It does not add a Qdrant cosine-search speed improvement.

BGE batches are atomically checkpointed under `checkpoints/`. Identical text, IDs,
model settings, source manifest and batch boundaries reuse those arrays on retry.
Keep `batch_size` unchanged when resuming to reuse every batch. A completed
manifest is reused only after its checksums pass. Do not run multiple writers
against the same output directory.

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


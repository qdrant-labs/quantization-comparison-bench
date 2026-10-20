# Frozen datasets and embeddings

Four English BEIR datasets, prepared and evaluated separately: `fiqa`,
`scifact`, `nfcorpus`, `quora`. Language is not an experimental axis.

| Dataset | Documents | Dev source split | Official test queries |
| --- | --- | --- | --- |
| scifact | 5,183 | train (no dev split) | 300 |
| nfcorpus | 3,633 | dev | 323 |
| fiqa | 57,638 | dev | 648 |
| quora | 522,931 | dev | 10,000 |

Source: [CohereLabs/beir-embed-english-v3](https://huggingface.co/datasets/CohereLabs/beir-embed-english-v3)
at the immutable commit `f018922a46a51388348a307ec2fef28019a33026`. It supplies
text, queries, relevance judgments, and 1,024-dimensional vectors from Cohere
`embed-english-v3.0`. Those vectors **are** the Cohere model in this benchmark;
no embedding API is called. The second model, `BAAI/bge-small-en-v1.5` at
commit `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a`, is run locally on the same
document and query IDs.

## Preparation

```bash
uv run qbench prepare-all             # all four datasets, Cohere arrays
uv run qbench prepare-all --with-bge  # also generate BGE arrays
uv run qbench prepare --dataset fiqa  # one dataset
```

`prepare` downloads the pinned Parquet files with `huggingface_hub`, records
each file's SHA-256, and writes two directories:

- `data/raw/<dataset>/`: `corpus.jsonl` (`{"id", "text"}`), `queries.jsonl`
  (`{"id", "text", "split"}`), `qrels.jsonl` (`{"query_id", "doc_id",
  "relevance", "split"}`), and `manifest.json` with source revision, source file
  hashes, counts, policies, and checksums.
- `data/embeddings/<dataset>/cohere/`: the final artifact format below, built
  from the supplied `emb` columns.

Document text is stripped title, newline, stripped body, with empty parts
omitted. Query text is unchanged. IDs are strings; rows are sorted by ID and
the Cohere arrays follow that order. The corpus is streamed into a temporary
memory-mapped file and permuted into sorted order, so Quora's 2 GiB float32
array does not need to be held in process memory twice.

Fifty development queries per dataset are sampled with the seed from the
official dev split, or from train for SciFact, which has no dev split. The
full official test split is kept so extended runs remain possible; the
default run configurations select 150 seeded test queries at run time and
record the chosen IDs in the run manifest. Development and test query IDs may
not overlap; the code rejects duplicates. Original integer relevance grades are
preserved, including non-positive ones. Do not substitute a new annotation
scale. For Quora, a query's own document ID is excluded consistently in exact
ground truth and in every Qdrant search.

Source vectors are within about 1e-4 of unit length; they are L2-normalized
on import and the largest deviation seen is recorded in the manifest. Qdrant
cosine collections also normalize internally; the explicit step keeps the
independent exact reference consistent.

`--max-docs` and `--max-test-queries` on `prepare` create seeded smoke subsets
that retain every judged document for the selected queries. Their manifests
are labelled as subsets; they verify execution and are not benchmark data.

## BGE embedding

```bash
uv run qbench embed --dataset fiqa --model bge
```

Reads the prepared text, validates its checksums, and runs BGE on CPU in
float32 with a 512-token right-truncated context. The passage instruction
`Represent this sentence for searching relevant passages: ` is prepended to
queries only. Quora uses no instruction on either side, matching the symmetric
`search_query` encoding of the supplied Cohere vectors. Completed batches are
checkpointed under `checkpoints/` and reused on resume with an identical batch
size. Run BGE before any timing: Quora takes the longest and competes with the
server for CPU. `qbench embed --model cohere` is rejected; Cohere vectors come
from `prepare`.

## Artifact format

| File | Content |
| --- | --- |
| `corpus.npy` | Float32 matrix, rows in `corpus_ids.json` order |
| `queries.npy` | Float32 matrix, rows in `query_ids.json` order |
| `corpus_ids.json` / `query_ids.json` | Lists of original IDs |
| `query_splits.json` | Query ID to `dev` or `test` |
| `qrels.json` | Query ID to document ID to original integer grade |
| `embedding-config.json` | Model, dimensions, input types, revisions, library versions |
| `manifest.json` | Provenance, source hashes, counts, checksums |
| `ground_truth.json` | Exact float32 top-100 per query, tied to the artifact checksums |

`qbench readiness --require-all` checks every dataset/model artifact: manifest
checksums, unit-length float32 vectors, expected dimensions (1,024 Cohere, 384
BGE), unique IDs, dev and test splits present, every query with a positive
judgment on an existing document, and whether `ground_truth.json` matches the
current arrays. `qbench verify` additionally checks the exact-search
invariants, and `qbench ground-truth` writes the reference rankings.

## Offline fixture

`qbench fixture` produces the same artifact format from seeded random unit
vectors (256 documents, 32 queries, 384 dimensions, one positive label per
query). It checks the runner and metrics only.

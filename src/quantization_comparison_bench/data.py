from __future__ import annotations

import asyncio
import hashlib
import json
import os
import random
import re
from collections.abc import Awaitable, Callable, Iterable
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Literal, TypedDict, cast

import httpx
import numpy as np
from numpy.typing import NDArray

DATASET_SOURCE = "CohereLabs/beir-embed-english-v3"
BGE_MODEL = "BAAI/bge-small-en-v1.5"
COHERE_MODEL = "embed-multilingual-v3.0"
BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "
SUPPORTED_DATASETS = {"fiqa", "scifact", "nfcorpus", "quora"}
type FloatArray = NDArray[np.float32]
type Split = Literal["dev", "test"]
type Qrels = dict[str, dict[str, int]]


class TextRow(TypedDict):
    id: str
    text: str


class QueryRow(TextRow):
    split: Split


class QrelRow(TypedDict):
    query_id: str
    doc_id: str
    relevance: int
    split: Split


class SourceQuery(TypedDict):
    _id: str
    text: str


class SourceText(SourceQuery):
    title: str


class SourceQrel(TypedDict):
    query_id: str
    corpus_id: str
    score: int


def _write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_jsonl(path: Path, rows: Iterable[object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def _read_jsonl[T](path: Path) -> list[T]:
    with path.open(encoding="utf-8") as file:
        return [cast(T, json.loads(line)) for line in file if line.strip()]


def _checksum(path: Path) -> str:
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def _require_commit(revision: str | None) -> str:
    if revision is None or re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise ValueError(
            "Use an immutable 40-character lowercase Hugging Face commit SHA."
        )
    return revision


def _sample[T](rows: list[T], count: int | None, seed: int) -> list[T]:
    if count is None or count >= len(rows):
        return rows
    selected = sorted(random.Random(seed).sample(range(len(rows)), count))
    return [rows[index] for index in selected]


def _source_rows(
    dataset: str, component: str, split: str, revision: str
) -> Iterable[object]:
    try:
        from datasets import load_dataset  # ty: ignore[unresolved-import]
    except ImportError as error:
        raise RuntimeError(
            "Install dataset support with: uv sync --extra datasets"
        ) from error
    return load_dataset(
        DATASET_SOURCE,
        f"{dataset}-{component}",
        split=split,
        revision=revision,
        streaming=True,
    )


def _prepare_queries(
    dataset: str,
    revision: str,
    dev_queries: int,
    max_test_queries: int | None,
    seed: int,
) -> tuple[list[QueryRow], list[QrelRow]]:
    queries: list[QueryRow] = []
    qrels: list[QrelRow] = []
    for split, source_split, limit in [
        ("dev", "train" if dataset == "scifact" else "dev", dev_queries),
        ("test", "test", max_test_queries),
    ]:
        source = cast(
            Iterable[SourceQuery],
            _source_rows(dataset, "queries", source_split, revision),
        )
        candidates = sorted(
            [
                {"id": str(row["_id"]), "text": row["text"], "split": split}
                for row in source
            ],
            key=lambda row: row["id"],
        )
        selected = cast(list[QueryRow], _sample(candidates, limit, seed))
        selected_ids = {row["id"] for row in selected}
        queries.extend(selected)
        judgments = cast(
            Iterable[SourceQrel], _source_rows(dataset, "qrels", source_split, revision)
        )
        qrels.extend(
            QrelRow(
                query_id=str(row["query_id"]),
                doc_id=str(row["corpus_id"]),
                relevance=int(row["score"]),
                split=cast(Split, split),
            )
            for row in judgments
            if str(row["query_id"]) in selected_ids
        )
    ids = [row["id"] for row in queries]
    if len(set(ids)) != len(ids):
        raise ValueError(
            "Duplicate query IDs or overlap between development and test splits."
        )
    return queries, sorted(qrels, key=lambda row: (row["query_id"], row["doc_id"]))


def _subset_corpus(
    corpus: list[TextRow],
    queries: list[QueryRow],
    qrels: list[QrelRow],
    count: int | None,
    seed: int,
) -> list[TextRow]:
    corpus_ids = {row["id"] for row in corpus}
    required = {row["doc_id"] for row in qrels}
    missing = required - corpus_ids
    if missing:
        raise ValueError(f"The corpus is missing {len(missing)} judged documents.")
    if count is None or count >= len(corpus):
        return corpus
    required.update(row["id"] for row in queries if row["id"] in corpus_ids)
    if len(required) > count:
        raise ValueError(
            f"max_docs={count} cannot retain {len(required)} judged/self-match documents. "
            "Use fewer queries or a larger max_docs."
        )
    candidates = [row for row in corpus if row["id"] not in required]
    sampled = _sample(candidates, count - len(required), seed)
    selected_ids = required | {row["id"] for row in sampled}
    return [row for row in corpus if row["id"] in selected_ids]


def prepare_dataset(
    dataset: str,
    output: Path,
    revision: str,
    dev_queries: int = 50,
    seed: int = 42,
    max_docs: int | None = None,
    max_test_queries: int | None = None,
) -> None:
    """Freeze raw BEIR text, query splits, and original relevance labels."""
    if dataset not in SUPPORTED_DATASETS:
        raise ValueError(f"Choose a dataset from {sorted(SUPPORTED_DATASETS)}.")
    revision = _require_commit(revision)
    if dev_queries < 1 or (max_docs is not None and max_docs < 1):
        raise ValueError("dev_queries and max_docs must be positive.")
    if max_test_queries is not None and max_test_queries < 1:
        raise ValueError("max_test_queries must be positive.")
    destination = output / dataset
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"Use an empty output directory: {destination}")
    queries, qrels = _prepare_queries(
        dataset, revision, dev_queries, max_test_queries, seed
    )
    source = cast(
        Iterable[SourceText], _source_rows(dataset, "corpus", "train", revision)
    )
    corpus = sorted(
        [
            TextRow(
                id=str(row["_id"]),
                text="\n".join(
                    part for part in [row["title"].strip(), row["text"].strip()] if part
                ),
            )
            for row in source
        ],
        key=lambda row: row["id"],
    )
    full_corpus_count = len(corpus)
    corpus = _subset_corpus(corpus, queries, qrels, max_docs, seed)
    if len({row["id"] for row in corpus}) != len(corpus):
        raise ValueError("Duplicate document IDs in the source corpus.")
    if not corpus or not queries or not qrels:
        raise ValueError(
            "Prepared dataset must have documents, queries, and judgments."
        )
    destination.mkdir(parents=True, exist_ok=True)
    _write_jsonl(destination / "corpus.jsonl", corpus)
    _write_jsonl(destination / "queries.jsonl", queries)
    _write_jsonl(destination / "qrels.jsonl", qrels)
    _write_json(
        destination / "manifest.json",
        {
            "schema_version": 1,
            "dataset": dataset,
            "source": DATASET_SOURCE,
            "source_revision": revision,
            "datasets_version": version("datasets"),
            "seed": seed,
            "development_source_split": "train" if dataset == "scifact" else "dev",
            "counts": {
                "documents": len(corpus),
                "full_corpus_documents": full_corpus_count,
                "dev_queries": sum(row["split"] == "dev" for row in queries),
                "test_queries": sum(row["split"] == "test" for row in queries),
                "qrels": len(qrels),
            },
            "smoke_subset": max_docs is not None or max_test_queries is not None,
            "serialization": "stripped title + newline + stripped body; omit empty parts",
            "source_embeddings_used": False,
            "label_policy": "original integer grades; retain all judgments for selected queries",
            "checksums": {
                name: _checksum(destination / name)
                for name in ("corpus.jsonl", "queries.jsonl", "qrels.jsonl")
            },
        },
    )


def _normalize(values: FloatArray) -> FloatArray:
    vectors = np.asarray(values, dtype=np.float32)
    if vectors.ndim != 2 or not np.isfinite(vectors).all():
        raise ValueError("Embeddings must be a finite two-dimensional float matrix.")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("Cannot normalize a zero embedding.")
    return np.asarray(vectors / norms, dtype=np.float32)


def _embedding_versions(model: str) -> dict[str, str]:
    packages = (
        ("numpy", "httpx")
        if model == "cohere"
        else ("numpy", "sentence-transformers", "transformers", "torch")
    )
    try:
        return {name: version(name) for name in packages}
    except PackageNotFoundError as error:
        raise RuntimeError("Install BGE support with: uv sync --extra bge") from error


def _save_array(path: Path, values: FloatArray) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as file:
        np.save(file, values, allow_pickle=False)
    temporary.replace(path)


async def _cohere_batch(
    client: httpx.AsyncClient, texts: list[str], input_type: str
) -> FloatArray:
    for attempt in range(5):
        response = await client.post(
            "https://api.cohere.com/v2/embed",
            json={
                "model": COHERE_MODEL,
                "texts": texts,
                "input_type": input_type,
                "embedding_types": ["float"],
                "truncate": "END",
            },
        )
        if response.status_code in {429, 500, 502, 503, 504} and attempt < 4:
            await asyncio.sleep(min(2 ** (attempt + 1), 30))
            continue
        response.raise_for_status()
        values = _normalize(
            np.asarray(response.json()["embeddings"]["float"], dtype=np.float32)
        )
        if values.shape != (len(texts), 1024):
            raise ValueError(f"Unexpected Cohere embedding shape: {values.shape}")
        return values
    raise RuntimeError("Cohere retry budget exhausted.")


async def _embed_batches(
    rows: list[TextRow],
    output: Path,
    kind: str,
    config: dict[str, object],
    batch_size: int,
    encode: Callable[[list[str]], Awaitable[FloatArray]],
) -> None:
    cache = output / "checkpoints" / kind
    cache.mkdir(parents=True, exist_ok=True)
    batches: list[Path] = []
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        identity = json.dumps(
            {"config": config, "rows": batch}, sort_keys=True, ensure_ascii=False
        )
        key = hashlib.sha256(identity.encode("utf-8")).hexdigest()
        path = cache / f"{start:09d}-{key}.npy"
        if not path.exists():
            values = await encode([row["text"] for row in batch])
            _save_array(path, values)
        values = np.load(path, allow_pickle=False)
        if values.shape != (len(batch), config["dimensions"]):
            raise ValueError(f"Invalid embedding checkpoint shape in {path}")
        batches.append(path)
    temporary = output / f"{kind}.npy.tmp"
    combined = np.lib.format.open_memmap(
        temporary,
        mode="w+",
        dtype=np.float32,
        shape=(len(rows), cast(int, config["dimensions"])),
    )
    offset = 0
    for path in batches:
        values = np.load(path, allow_pickle=False)
        combined[offset : offset + len(values)] = values
        offset += len(values)
    combined.flush()
    del combined
    temporary.replace(output / f"{kind}.npy")


def _write_embedding_metadata(
    output: Path,
    corpus_ids: list[str],
    query_ids: list[str],
    query_splits: dict[str, str],
    qrels: Qrels,
    manifest: dict[str, object],
) -> None:
    _write_json(output / "corpus_ids.json", corpus_ids)
    _write_json(output / "query_ids.json", query_ids)
    _write_json(output / "query_splits.json", query_splits)
    _write_json(output / "qrels.json", qrels)
    names = (
        "corpus.npy",
        "queries.npy",
        "corpus_ids.json",
        "query_ids.json",
        "query_splits.json",
        "qrels.json",
    )
    _write_json(
        output / "manifest.json",
        {
            **manifest,
            "schema_version": 1,
            "created_at_utc": datetime.now(UTC).isoformat(),
            "dtype": "float32",
            "normalization": "L2",
            "counts": {"documents": len(corpus_ids), "queries": len(query_ids)},
            "checksums": {name: _checksum(output / name) for name in names},
        },
    )


async def embed_dataset(
    dataset_dir: Path,
    output: Path,
    model: str,
    batch_size: int = 64,
    revision: str | None = None,
    max_documents: int | None = None,
) -> None:
    """Embed frozen data, reusing completed batches with identical inputs and settings."""
    if model not in {"cohere", "bge"}:
        raise ValueError("model must be 'cohere' or 'bge'.")
    if batch_size < 1 or (model == "cohere" and batch_size > 96):
        raise ValueError("batch_size must be positive; Cohere allows at most 96 texts.")
    if max_documents is not None:
        raise ValueError(
            "Use prepare_dataset(max_docs=...) to preserve query judgments in subsets."
        )
    if model == "bge":
        revision = _require_commit(revision)
    elif revision is not None:
        raise ValueError(
            "Cohere exposes a model ID, not a Hugging Face model revision."
        )
    source_manifest = json.loads(
        (dataset_dir / "manifest.json").read_text(encoding="utf-8")
    )
    for name, expected in source_manifest["checksums"].items():
        if _checksum(dataset_dir / name) != expected:
            raise ValueError(f"Prepared data checksum mismatch: {name}")
    corpus = cast(list[TextRow], _read_jsonl(dataset_dir / "corpus.jsonl"))
    queries = cast(list[QueryRow], _read_jsonl(dataset_dir / "queries.jsonl"))
    qrel_rows = cast(list[QrelRow], _read_jsonl(dataset_dir / "qrels.jsonl"))
    dataset = str(source_manifest["dataset"])
    library_versions = _embedding_versions(model)
    config: dict[str, object] = {
        "dataset": dataset,
        "model": COHERE_MODEL if model == "cohere" else BGE_MODEL,
        "model_revision": revision,
        "dimensions": 1024 if model == "cohere" else 384,
        "corpus_input_type": "search_query"
        if dataset == "quora"
        else "search_document",
        "query_input_type": "search_query",
        "query_instruction": BGE_QUERY_INSTRUCTION
        if model == "bge" and dataset != "quora"
        else "",
        "truncate": "END" if model == "cohere" else "right, max_seq_length=512",
        "normalization": "L2 float32",
        "source_manifest_sha256": _checksum(dataset_dir / "manifest.json"),
        "library_versions": library_versions,
        "device": "hosted" if model == "cohere" else "cpu",
    }
    output.mkdir(parents=True, exist_ok=True)
    config_path = output / "embedding-config.json"
    if config_path.exists():
        previous = json.loads(config_path.read_text(encoding="utf-8"))
        if previous != config:
            raise ValueError(
                "Output already has a different embedding configuration; use a new directory."
            )
    else:
        if any(output.iterdir()):
            raise FileExistsError(
                "Embedding output must be empty or have a matching embedding-config.json."
            )
        _write_json(config_path, config)
    completed_manifest = output / "manifest.json"
    if completed_manifest.exists():
        completed = json.loads(completed_manifest.read_text(encoding="utf-8"))
        if all(
            (output / name).is_file() and _checksum(output / name) == expected
            for name, expected in completed["checksums"].items()
        ):
            return
    if model == "cohere":
        key = os.environ.get("COHERE_API_KEY")
        if not key:
            raise ValueError(
                "Set COHERE_API_KEY before running paid Cohere embedding requests."
            )
        async with httpx.AsyncClient(
            headers={"Authorization": f"Bearer {key}"}, timeout=120
        ) as client:

            async def encode_corpus(texts: list[str]) -> FloatArray:
                return await _cohere_batch(
                    client, texts, str(config["corpus_input_type"])
                )

            async def encode_queries(texts: list[str]) -> FloatArray:
                return await _cohere_batch(client, texts, "search_query")

            await _embed_batches(
                corpus, output, "corpus", config, batch_size, encode_corpus
            )
            await _embed_batches(
                cast(list[TextRow], queries),
                output,
                "queries",
                config,
                batch_size,
                encode_queries,
            )
    else:
        try:
            from sentence_transformers import SentenceTransformer  # ty: ignore[unresolved-import]
        except ImportError as error:
            raise RuntimeError(
                "Install BGE support with: uv sync --extra bge"
            ) from error
        encoder = await asyncio.to_thread(
            SentenceTransformer, BGE_MODEL, revision=revision, device="cpu"
        )
        encoder.float()
        encoder.eval()
        encoder.max_seq_length = 512
        encoder.tokenizer.truncation_side = "right"

        async def encode_corpus(texts: list[str]) -> FloatArray:
            values = await asyncio.to_thread(
                encoder.encode,
                texts,
                batch_size=batch_size,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            return _normalize(np.asarray(values, dtype=np.float32))

        async def encode_queries(texts: list[str]) -> FloatArray:
            return await encode_corpus(
                [str(config["query_instruction"]) + text for text in texts]
            )

        await _embed_batches(
            corpus, output, "corpus", config, batch_size, encode_corpus
        )
        await _embed_batches(
            cast(list[TextRow], queries),
            output,
            "queries",
            config,
            batch_size,
            encode_queries,
        )
    qrels: Qrels = {row["id"]: {} for row in queries}
    for row in qrel_rows:
        qrels[row["query_id"]][row["doc_id"]] = row["relevance"]
    _write_embedding_metadata(
        output,
        [row["id"] for row in corpus],
        [row["id"] for row in queries],
        {row["id"]: row["split"] for row in queries},
        qrels,
        {
            **config,
            "source_manifest": source_manifest,
            "library_versions": library_versions,
            "synthetic": False,
        },
    )


def create_fixture(
    output: Path,
    dimensions: int = 384,
    documents: int = 256,
    queries: int = 32,
    seed: int = 42,
) -> None:
    """Create a reproducible synthetic fixture for checking benchmark plumbing."""
    if dimensions < 1 or documents < 2 or queries < 2:
        raise ValueError("Require dimensions >= 1, documents >= 2, and queries >= 2.")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Use an empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    corpus = _normalize(rng.standard_normal((documents, dimensions)).astype(np.float32))
    target_indices = rng.integers(0, documents, size=queries)
    query_vectors = _normalize(
        corpus[target_indices]
        + rng.standard_normal((queries, dimensions)).astype(np.float32) * 0.01
    )
    corpus_ids = [f"doc-{index:06d}" for index in range(documents)]
    query_ids = [f"query-{index:06d}" for index in range(queries)]
    dev_count = max(1, queries // 4)
    _save_array(output / "corpus.npy", corpus)
    _save_array(output / "queries.npy", query_vectors)
    _write_embedding_metadata(
        output,
        corpus_ids,
        query_ids,
        {
            qid: "dev" if index < dev_count else "test"
            for index, qid in enumerate(query_ids)
        },
        {
            qid: {corpus_ids[int(target)]: 1}
            for qid, target in zip(query_ids, target_indices, strict=True)
        },
        {
            "dataset": "synthetic",
            "model": "synthetic-normal",
            "dimensions": dimensions,
            "seed": seed,
            "synthetic": True,
            "numpy_version": version("numpy"),
        },
    )

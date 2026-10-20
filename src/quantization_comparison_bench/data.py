from __future__ import annotations

import asyncio
import hashlib
import json
import random
import re
from collections.abc import Awaitable, Callable, Iterable
from datetime import UTC, datetime
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version
from importlib.resources import files
from os.path import relpath
from pathlib import Path
from typing import Literal, TypedDict, cast

import numpy as np
from numpy.typing import NDArray

DATASET_SOURCE = "CohereLabs/beir-embed-english-v3"
BGE_MODEL = "BAAI/bge-small-en-v1.5"
COHERE_MODEL = "embed-english-v3.0"
DATASET_REVISION = "f018922a46a51388348a307ec2fef28019a33026"
BGE_REVISION = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
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


class EmbeddingBatch(TypedDict):
    start: int
    indices: list[int]


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
    dataset: str, component: str, split: str, revision: str, cache: Path
) -> Iterable[object]:
    path = _source_file(dataset, component, split, revision, cache)
    try:
        import pyarrow.parquet as parquet
    except ImportError as error:
        raise RuntimeError(
            "Install dataset support with: uv sync --extra datasets"
        ) from error
    source = parquet.ParquetFile(path)
    columns = [name for name in source.schema_arrow.names if name != "emb"]
    for batch in source.iter_batches(batch_size=1024, columns=columns):
        yield from batch.to_pylist()


@lru_cache(maxsize=32)
def _source_file(
    dataset: str, component: str, split: str, revision: str, cache: Path
) -> Path:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as error:
        raise RuntimeError(
            "Install dataset support with: uv sync --extra datasets"
        ) from error
    if revision != DATASET_REVISION:
        raise ValueError(
            f"Source checksums are locked to revision {DATASET_REVISION}; "
            "update dataset_sources.json before using another revision."
        )
    name = f"{dataset}/{component}/{'0000' if component == 'corpus' else split}.parquet"
    locked = json.loads(
        files("quantization_comparison_bench")
        .joinpath("dataset_sources.json")
        .read_text(encoding="utf-8")
    )["files"][name]
    destination = cache / revision / name
    if not destination.is_file():
        destination = Path(
            hf_hub_download(
                DATASET_SOURCE,
                name,
                repo_type="dataset",
                revision=revision,
                local_dir=cache / revision,
            )
        )
    if destination.stat().st_size != locked["bytes"]:
        raise ValueError(
            f"Source size mismatch; remove and download again: {destination}"
        )
    if _checksum(destination) != locked["sha256"]:
        raise ValueError(
            f"Source checksum mismatch; remove and download again: {destination}"
        )
    return destination


def _prepare_queries(
    dataset: str,
    revision: str,
    dev_queries: int,
    max_test_queries: int | None,
    seed: int,
    cache: Path,
) -> tuple[list[QueryRow], list[QrelRow]]:
    queries: list[QueryRow] = []
    qrels: list[QrelRow] = []
    for split, source_split, limit in [
        ("dev", "train" if dataset == "scifact" else "dev", dev_queries),
        ("test", "test", max_test_queries),
    ]:
        source = cast(
            Iterable[SourceQuery],
            _source_rows(dataset, "queries", source_split, revision, cache),
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
            Iterable[SourceQrel],
            _source_rows(dataset, "qrels", source_split, revision, cache),
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
    source_cache: Path | None = None,
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
    cache = source_cache or output.parent / "sources"
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"Use an empty output directory: {destination}")
    queries, qrels = _prepare_queries(
        dataset, revision, dev_queries, max_test_queries, seed, cache
    )
    source = cast(
        Iterable[SourceText], _source_rows(dataset, "corpus", "train", revision, cache)
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
            "pyarrow_version": version("pyarrow"),
            "source_cache": relpath(cache.resolve(), destination.resolve()),
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
            "source_embeddings_available": COHERE_MODEL,
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
        ("numpy", "pyarrow")
        if model == "cohere"
        else ("numpy", "sentence-transformers", "transformers", "torch")
    )
    try:
        return {name: version(name) for name in packages}
    except PackageNotFoundError as error:
        extra = "datasets" if model == "cohere" else "bge"
        raise RuntimeError(
            f"Install embedding support with: uv sync --extra {extra}"
        ) from error


def _save_array(path: Path, values: FloatArray) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as file:
        np.save(file, values, allow_pickle=False)
    temporary.replace(path)


def _import_cohere(
    dataset: str,
    source_revision: str,
    cache: Path,
    output: Path,
    kind: str,
    rows: list[TextRow],
    source_splits: list[str],
) -> None:
    import pyarrow.parquet as parquet

    positions = {row["id"]: index for index, row in enumerate(rows)}
    found: set[str] = set()
    temporary = output / f"{kind}.npy.tmp"
    combined = np.lib.format.open_memmap(
        temporary, mode="w+", dtype=np.float32, shape=(len(rows), 1024)
    )
    for split in source_splits:
        path = _source_file(dataset, kind, split, source_revision, cache)
        source = parquet.ParquetFile(path)
        for batch in source.iter_batches(batch_size=1024, columns=["_id", "emb"]):
            ids = [str(value) for value in batch.column("_id").to_pylist()]
            selected = [index for index, value in enumerate(ids) if value in positions]
            if not selected:
                continue
            embeddings = batch.column("emb").to_pylist()
            vectors = _normalize(
                np.asarray([embeddings[index] for index in selected], dtype=np.float32)
            )
            if vectors.shape[1] != 1024:
                raise ValueError("Expected 1024-dimensional source Cohere embeddings.")
            for source_index, vector in zip(selected, vectors, strict=True):
                row_id = ids[source_index]
                if row_id in found:
                    raise ValueError(f"Duplicate source embedding ID: {row_id}")
                combined[positions[row_id]] = vector
                found.add(row_id)
    if found != set(positions):
        raise ValueError(
            f"Source embeddings missing {len(set(positions) - found)} IDs."
        )
    combined.flush()
    del combined
    temporary.replace(output / f"{kind}.npy")


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

    def checkpoint(start: int, indices: list[int]) -> Path:
        batch = [rows[index] for index in indices]
        identity = json.dumps(
            {"config": config, "rows": batch}, sort_keys=True, ensure_ascii=False
        )
        key = hashlib.sha256(identity.encode("utf-8")).hexdigest()
        return cache / f"{start:09d}-{key}.npy"

    plan_identity = json.dumps(
        {
            "config": config,
            "batch_size": batch_size,
            "ids": [row["id"] for row in rows],
        },
        sort_keys=True,
    )
    plan_key = hashlib.sha256(plan_identity.encode("utf-8")).hexdigest()
    plan_path = cache / f"batch-plan-{plan_key}.json"
    if plan_path.exists():
        plan = cast(
            list[EmbeddingBatch], json.loads(plan_path.read_text(encoding="utf-8"))
        )
        if sorted(index for batch in plan for index in batch["indices"]) != list(
            range(len(rows))
        ):
            raise ValueError(
                f"Checkpoint batch plan has invalid row alignment: {plan_path}"
            )
    else:
        plan: list[EmbeddingBatch] = []
        pending: list[int] = []
        for start in range(0, len(rows), batch_size):
            indices = list(range(start, min(start + batch_size, len(rows))))
            path = checkpoint(start, indices)
            if path.exists():
                plan.append(EmbeddingBatch(start=start, indices=indices))
            else:
                pending.extend(indices)
        pending.sort(key=lambda index: (-len(rows[index]["text"]), index))
        for start in range(0, len(pending), batch_size):
            plan.append(
                EmbeddingBatch(start=start, indices=pending[start : start + batch_size])
            )
        _write_json(plan_path, plan)
    batches = [
        (batch["indices"], checkpoint(batch["start"], batch["indices"]))
        for batch in plan
    ]
    for indices, path in batches:
        if not path.exists():
            values = await encode([rows[index]["text"] for index in indices])
            _save_array(path, values)
        values = np.load(path, allow_pickle=False)
        if values.shape != (len(indices), config["dimensions"]):
            raise ValueError(f"Invalid embedding checkpoint shape in {path}")
        if not np.isfinite(values).all() or not np.allclose(
            np.linalg.norm(values, axis=1), 1, atol=1e-5
        ):
            raise ValueError(f"Invalid embedding checkpoint vectors in {path}")
    temporary = output / f"{kind}.npy.tmp"
    combined = np.lib.format.open_memmap(
        temporary,
        mode="w+",
        dtype=np.float32,
        shape=(len(rows), cast(int, config["dimensions"])),
    )
    for indices, path in batches:
        values = np.load(path, allow_pickle=False)
        combined[indices] = values
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
    device: str = "cpu",
) -> None:
    """Embed frozen data, reusing completed batches with identical inputs and settings."""
    if model not in {"cohere", "bge"}:
        raise ValueError("model must be 'cohere' or 'bge'.")
    if batch_size < 1:
        raise ValueError("batch_size must be positive.")
    if max_documents is not None:
        raise ValueError(
            "Use prepare_dataset(max_docs=...) to preserve query judgments in subsets."
        )
    if model == "bge":
        revision = _require_commit(revision)
    elif revision is not None:
        raise ValueError(
            "Cohere vectors use the prepared dataset's pinned source revision."
        )
    source_manifest = json.loads(
        (dataset_dir / "manifest.json").read_text(encoding="utf-8")
    )
    for name, expected in source_manifest["checksums"].items():
        if _checksum(dataset_dir / name) != expected:
            raise ValueError(f"Prepared data checksum mismatch: {name}")
    dataset = str(source_manifest["dataset"])
    stable_config = {
        "dataset": dataset,
        "model": COHERE_MODEL if model == "cohere" else BGE_MODEL,
        "model_revision": revision,
        "dimensions": 1024 if model == "cohere" else 384,
        "source_manifest_sha256": _checksum(dataset_dir / "manifest.json"),
    }
    completed_manifest = output / "manifest.json"
    if completed_manifest.exists():
        from quantization_comparison_bench.evaluation import (
            load_embeddings,
            validate_embeddings,
        )

        completed = json.loads(completed_manifest.read_text(encoding="utf-8"))
        if any(completed.get(key) != value for key, value in stable_config.items()):
            raise ValueError(
                "Completed embeddings use different source data or model settings."
            )
        required = {
            "corpus.npy",
            "queries.npy",
            "corpus_ids.json",
            "query_ids.json",
            "query_splits.json",
            "qrels.json",
        }
        if set(completed["checksums"]) != required:
            raise ValueError("Completed embedding manifest has an invalid file list.")
        if all(
            (output / name).is_file() and _checksum(output / name) == expected
            for name, expected in completed["checksums"].items()
        ):
            validate_embeddings(load_embeddings(output))
            return
    corpus = cast(list[TextRow], _read_jsonl(dataset_dir / "corpus.jsonl"))
    queries = cast(list[QueryRow], _read_jsonl(dataset_dir / "queries.jsonl"))
    qrel_rows = cast(list[QrelRow], _read_jsonl(dataset_dir / "qrels.jsonl"))
    library_versions = _embedding_versions(model)
    config: dict[str, object] = {
        **stable_config,
        "corpus_input_type": "search_query"
        if dataset == "quora"
        else "search_document",
        "query_input_type": "search_query",
        "query_instruction": BGE_QUERY_INSTRUCTION
        if model == "bge" and dataset != "quora"
        else "",
        "truncate": "source-precomputed (publisher settings)"
        if model == "cohere"
        else "right, max_seq_length=512",
        "normalization": "L2 float32",
        "library_versions": library_versions,
        "device": "precomputed" if model == "cohere" else device,
        "embedding_source": DATASET_SOURCE if model == "cohere" else BGE_MODEL,
        "embedding_source_revision": source_manifest["source_revision"]
        if model == "cohere"
        else revision,
    }
    if model == "cohere":
        config["corpus_input_type"] = (
            "search_query" if dataset == "quora" else "publisher_precomputed"
        )
        config["query_input_type"] = "publisher_precomputed"
        config["publisher_provenance"] = (
            f"https://huggingface.co/datasets/{DATASET_SOURCE}/blob/"
            f"{source_manifest['source_revision']}/README.md"
        )
        config["publisher_preprocessing"] = (
            "Imported by ID. Publisher documents Quora corpus search_query; "
            "other input settings and exact serialization were not reconstructed."
        )
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
    if model == "cohere":
        cache = dataset_dir / source_manifest.get("source_cache", "../../sources")
        await asyncio.to_thread(
            _import_cohere,
            dataset,
            str(source_manifest["source_revision"]),
            cache,
            output,
            "corpus",
            corpus,
            ["train"],
        )
        await asyncio.to_thread(
            _import_cohere,
            dataset,
            str(source_manifest["source_revision"]),
            cache,
            output,
            "queries",
            cast(list[TextRow], queries),
            [str(source_manifest["development_source_split"]), "test"],
        )
    else:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as error:
            raise RuntimeError(
                "Install BGE support with: uv sync --extra bge"
            ) from error
        encoder = await asyncio.to_thread(
            SentenceTransformer,
            BGE_MODEL,
            revision=revision,
            device=device,
            cache_folder=str(dataset_dir.parent.parent / "models"),
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
            "embedding_batch_size": batch_size if model == "bge" else None,
            "batch_order": "descending text length after retained checkpoints"
            if model == "bge"
            else "source order with explicit ID alignment",
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

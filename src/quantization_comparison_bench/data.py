from __future__ import annotations

import asyncio
import hashlib
import json
import random
import re
from collections.abc import Awaitable, Callable, Iterable
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Literal, TypedDict, cast

import numpy as np
from numpy.typing import NDArray

DATASET_SOURCE = "CohereLabs/beir-embed-english-v3"
BGE_MODEL = "BAAI/bge-small-en-v1.5"
# Local sentence-transformers models by label: (Hugging Face ID, dimensions).
LOCAL_MODELS = {
    "bge": ("BAAI/bge-small-en-v1.5", 384),
    "bge_base": ("BAAI/bge-base-en-v1.5", 768),
}
# The source dataset ships vectors from this hosted model; no API calls are made.
COHERE_MODEL = "embed-english-v3.0"
COHERE_DIMENSIONS = 1024
BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "
SUPPORTED_DATASETS = {"fiqa", "scifact", "nfcorpus", "quora"}
DATASETS_IN_ORDER = ("scifact", "nfcorpus", "fiqa", "quora")
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
    emb: list[float]


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


def _source_files(
    dataset: str, component: str, split: str, revision: str
) -> list[Path]:
    """Download the pinned Parquet files for one component and split."""
    try:
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError as error:
        raise RuntimeError(
            "Install dataset support with: uv sync --extra datasets"
        ) from error
    prefix = f"{dataset}/{component}"
    entries = HfApi().list_repo_tree(
        DATASET_SOURCE, path_in_repo=prefix, repo_type="dataset", revision=revision
    )
    names = sorted(
        entry.path
        for entry in entries
        if entry.path.endswith(".parquet")
        and (component == "corpus" or entry.path == f"{prefix}/{split}.parquet")
    )
    if not names:
        raise ValueError(f"No {component}/{split} Parquet files for {dataset}")
    return [
        Path(
            hf_hub_download(
                DATASET_SOURCE, name, repo_type="dataset", revision=revision
            )
        )
        for name in names
    ]


def _parquet_rows(paths: list[Path], columns: list[str]) -> Iterable[dict[str, object]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as error:
        raise RuntimeError(
            "Install dataset support with: uv sync --extra datasets"
        ) from error
    for path in paths:
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(batch_size=2048, columns=columns):
            yield from batch.to_pylist()


def _parquet_row_count(paths: list[Path]) -> int:
    import pyarrow.parquet as pq

    return sum(pq.ParquetFile(path).metadata.num_rows for path in paths)


def _source_rows(
    dataset: str, component: str, split: str, revision: str, columns: list[str]
) -> tuple[Iterable[dict[str, object]], dict[str, str]]:
    paths = _source_files(dataset, component, split, revision)
    hashes = {f"{dataset}/{component}/{path.name}": _checksum(path) for path in paths}
    return _parquet_rows(paths, columns), hashes


def _as_vector(values: object, label: str) -> FloatArray:
    vector = np.asarray(values, dtype=np.float32)
    if vector.shape != (COHERE_DIMENSIONS,) or not np.isfinite(vector).all():
        raise ValueError(f"{label}: expected a finite {COHERE_DIMENSIONS}-d vector")
    return vector


def _prepare_queries(
    dataset: str,
    revision: str,
    dev_queries: int,
    max_test_queries: int | None,
    seed: int,
) -> tuple[list[QueryRow], list[QrelRow], dict[str, FloatArray], dict[str, str]]:
    queries: list[QueryRow] = []
    qrels: list[QrelRow] = []
    vectors: dict[str, FloatArray] = {}
    hashes: dict[str, str] = {}
    plan: list[tuple[Split, str, int | None]] = [
        ("dev", "train" if dataset == "scifact" else "dev", dev_queries),
        ("test", "test", max_test_queries),
    ]
    for split, source_split, limit in plan:
        rows, found = _source_rows(
            dataset, "queries", source_split, revision, ["_id", "text", "emb"]
        )
        hashes.update(found)
        source = cast(Iterable[SourceQuery], rows)
        candidates: list[QueryRow] = []
        embeddings: dict[str, FloatArray] = {}
        for row in source:
            query_id = str(row["_id"])
            candidates.append({"id": query_id, "text": row["text"], "split": split})
            embeddings[query_id] = _as_vector(row["emb"], f"query {query_id}")
        candidates.sort(key=lambda row: row["id"])
        selected = _sample(candidates, limit, seed)
        selected_ids = {row["id"] for row in selected}
        queries.extend(selected)
        vectors.update({qid: embeddings[qid] for qid in selected_ids})
        judgments, found = _source_rows(
            dataset, "qrels", source_split, revision, ["query_id", "corpus_id", "score"]
        )
        hashes.update(found)
        qrels.extend(
            QrelRow(
                query_id=str(row["query_id"]),
                doc_id=str(row["corpus_id"]),
                relevance=int(row["score"]),
                split=split,
            )
            for row in cast(Iterable[SourceQrel], judgments)
            if str(row["query_id"]) in selected_ids
        )
    ids = [row["id"] for row in queries]
    if len(set(ids)) != len(ids):
        raise ValueError(
            "Duplicate query IDs or overlap between development and test splits."
        )
    return (
        queries,
        sorted(qrels, key=lambda row: (row["query_id"], row["doc_id"])),
        vectors,
        hashes,
    )


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


def _import_versions() -> dict[str, str]:
    try:
        return {name: version(name) for name in ("numpy", "huggingface_hub", "pyarrow")}
    except PackageNotFoundError as error:
        raise RuntimeError(
            "Install dataset support with: uv sync --extra datasets"
        ) from error


def cohere_artifact_dir(data_root: Path, dataset: str) -> Path:
    return data_root / "embeddings" / dataset / "cohere"


def prepare_dataset(
    dataset: str,
    output: Path,
    revision: str,
    dev_queries: int = 50,
    seed: int = 42,
    max_docs: int | None = None,
    max_test_queries: int | None = None,
    embeddings_output: Path | None = None,
) -> None:
    """Freeze raw BEIR text, query splits, original labels, and Cohere arrays.

    `output/<dataset>/` receives the text artifacts used for BGE embedding.
    `embeddings_output` (default `output/../embeddings/<dataset>/cohere`) receives
    the supplied Cohere embed-english-v3.0 vectors in the final artifact format.
    """
    if dataset not in SUPPORTED_DATASETS:
        raise ValueError(f"Choose a dataset from {sorted(SUPPORTED_DATASETS)}.")
    revision = _require_commit(revision)
    if dev_queries < 1 or (max_docs is not None and max_docs < 1):
        raise ValueError("dev_queries and max_docs must be positive.")
    if max_test_queries is not None and max_test_queries < 1:
        raise ValueError("max_test_queries must be positive.")
    destination = output / dataset
    vectors_dir = embeddings_output or cohere_artifact_dir(output.parent, dataset)
    for directory in (destination, vectors_dir):
        if directory.exists() and any(directory.iterdir()):
            raise FileExistsError(f"Use an empty output directory: {directory}")
    library_versions = _import_versions()
    queries, qrels, query_vectors, source_hashes = _prepare_queries(
        dataset, revision, dev_queries, max_test_queries, seed
    )
    rows, found = _source_rows(
        dataset, "corpus", "train", revision, ["_id", "title", "text", "emb"]
    )
    source_hashes.update(found)
    corpus_count = _parquet_row_count(
        _source_files(dataset, "corpus", "train", revision)
    )
    vectors_dir.mkdir(parents=True, exist_ok=True)
    scratch = vectors_dir / "corpus-source-order.npy.tmp"
    source_vectors = np.lib.format.open_memmap(
        scratch, mode="w+", dtype=np.float32, shape=(corpus_count, COHERE_DIMENSIONS)
    )
    corpus: list[TextRow] = []
    source_row: dict[str, int] = {}
    try:
        for position, row in enumerate(cast(Iterable[SourceText], rows)):
            if position >= corpus_count:
                raise ValueError("Corpus has more rows than its Parquet metadata.")
            doc_id = str(row["_id"])
            corpus.append(
                TextRow(
                    id=doc_id,
                    text="\n".join(
                        part
                        for part in [row["title"].strip(), row["text"].strip()]
                        if part
                    ),
                )
            )
            source_vectors[position] = _as_vector(row["emb"], f"document {doc_id}")
            source_row[doc_id] = position
        if len(corpus) != corpus_count:
            raise ValueError("Corpus has fewer rows than its Parquet metadata.")
        if len(source_row) != len(corpus):
            raise ValueError("Duplicate document IDs in the source corpus.")
        source_vectors.flush()
        corpus.sort(key=lambda row: row["id"])
        full_corpus_count = len(corpus)
        corpus = _subset_corpus(corpus, queries, qrels, max_docs, seed)
        if not corpus or not queries or not qrels:
            raise ValueError(
                "Prepared dataset must have documents, queries, and judgments."
            )
        destination.mkdir(parents=True, exist_ok=True)
        _write_jsonl(destination / "corpus.jsonl", corpus)
        _write_jsonl(destination / "queries.jsonl", queries)
        _write_jsonl(destination / "qrels.jsonl", qrels)
        raw_manifest = {
            "schema_version": 2,
            "dataset": dataset,
            "source": DATASET_SOURCE,
            "source_revision": revision,
            "source_files_sha256": dict(sorted(source_hashes.items())),
            "library_versions": library_versions,
            "seed": seed,
            "development_source_split": "train" if dataset == "scifact" else "dev",
            "counts": {
                "documents": len(corpus),
                "full_corpus_documents": full_corpus_count,
                "dev_queries": sum(row["split"] == "dev" for row in queries),
                "test_queries": sum(row["split"] == "test" for row in queries),
                "full_test_queries": max_test_queries is None,
                "qrels": len(qrels),
            },
            "smoke_subset": max_docs is not None or max_test_queries is not None,
            "serialization": "stripped title + newline + stripped body; omit empty parts",
            "source_embeddings_used": True,
            "label_policy": "original integer grades; retain all judgments for selected queries",
        }
        raw_manifest["checksums"] = {
            name: _checksum(destination / name)
            for name in ("corpus.jsonl", "queries.jsonl", "qrels.jsonl")
        }
        _write_json(destination / "manifest.json", raw_manifest)
        # Final Cohere artifact: rows follow the sorted, possibly subset, corpus.
        order = np.asarray([source_row[row["id"]] for row in corpus], dtype=np.int64)
        temporary = vectors_dir / "corpus.npy.tmp"
        combined = np.lib.format.open_memmap(
            temporary,
            mode="w+",
            dtype=np.float32,
            shape=(len(order), COHERE_DIMENSIONS),
        )
        norm_deviation = 0.0
        for start in range(0, len(order), 8192):
            block = np.asarray(source_vectors[order[start : start + 8192]])
            norms = np.linalg.norm(block, axis=1)
            norm_deviation = max(norm_deviation, float(np.abs(norms - 1).max()))
            combined[start : start + len(block)] = _normalize(block)
        combined.flush()
        del combined
        temporary.replace(vectors_dir / "corpus.npy")
        query_matrix = np.stack([query_vectors[row["id"]] for row in queries])
        norm_deviation = max(
            norm_deviation,
            float(np.abs(np.linalg.norm(query_matrix, axis=1) - 1).max()),
        )
        _save_array(vectors_dir / "queries.npy", _normalize(query_matrix))
    finally:
        del source_vectors
        scratch.unlink(missing_ok=True)
    qrels_map: Qrels = {row["id"]: {} for row in queries}
    for row in qrels:
        qrels_map[row["query_id"]][row["doc_id"]] = row["relevance"]
    config: dict[str, object] = {
        "dataset": dataset,
        "model": COHERE_MODEL,
        "model_revision": None,
        "dimensions": COHERE_DIMENSIONS,
        "corpus_input_type": "search_query"
        if dataset == "quora"
        else "search_document",
        "query_input_type": "search_query",
        "query_instruction": "",
        "truncate": "as published by the source dataset",
        "normalization": "L2 float32",
        "source_manifest_sha256": _checksum(destination / "manifest.json"),
        "library_versions": library_versions,
        "device": "imported",
        "source_revision": revision,
        "source_files_sha256": dict(sorted(source_hashes.items())),
        "source_norm_max_abs_deviation": norm_deviation,
    }
    _write_json(vectors_dir / "embedding-config.json", config)
    _write_embedding_metadata(
        vectors_dir,
        [row["id"] for row in corpus],
        [row["id"] for row in queries],
        {row["id"]: row["split"] for row in queries},
        qrels_map,
        {**config, "source_manifest": raw_manifest, "synthetic": False},
    )


def artifact_complete(path: Path) -> bool:
    """True when a manifest exists and every checksum it lists still matches."""
    manifest = path / "manifest.json"
    if not manifest.is_file():
        return False
    completed = json.loads(manifest.read_text(encoding="utf-8"))
    return all(
        (path / name).is_file() and _checksum(path / name) == expected
        for name, expected in completed.get("checksums", {}).items()
    )


async def prepare_all(
    data_root: Path,
    revision: str,
    with_bge: bool,
    bge_revision: str | None,
    dev_queries: int = 50,
    seed: int = 42,
    batch_size: int = 64,
    datasets: Iterable[str] = DATASETS_IN_ORDER,
    extra_models: dict[str, str] | None = None,
) -> list[dict[str, object]]:
    """Stage every corpus, its judgments, and Cohere arrays; optionally BGE."""
    report: list[dict[str, object]] = []
    for dataset in datasets:
        raw = data_root / "raw" / dataset
        cohere = cohere_artifact_dir(data_root, dataset)
        if artifact_complete(raw) and artifact_complete(cohere):
            status = "present"
        else:
            prepare_dataset(
                dataset,
                data_root / "raw",
                revision,
                dev_queries,
                seed,
                None,
                None,
                cohere,
            )
            status = "prepared"
        entry: dict[str, object] = {"dataset": dataset, "raw": status, "cohere": status}
        if with_bge:
            bge = data_root / "embeddings" / dataset / "bge"
            if artifact_complete(bge):
                entry["bge"] = "present"
            else:
                await embed_dataset(raw, bge, "bge", batch_size, bge_revision)
                entry["bge"] = "embedded"
        for label, model_revision in (extra_models or {}).items():
            target = data_root / "embeddings" / dataset / label
            if artifact_complete(target):
                entry[label] = "present"
            else:
                await embed_dataset(raw, target, label, batch_size, model_revision)
                entry[label] = "embedded"
        report.append(entry)
    return report


def _normalize(values: FloatArray) -> FloatArray:
    vectors = np.asarray(values, dtype=np.float32)
    if vectors.ndim != 2 or not np.isfinite(vectors).all():
        raise ValueError("Embeddings must be a finite two-dimensional float matrix.")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("Cannot normalize a zero embedding.")
    return np.asarray(vectors / norms, dtype=np.float32)


def _embedding_versions(model: str) -> dict[str, str]:
    packages = ("numpy", "sentence-transformers", "transformers", "torch")
    try:
        return {name: version(name) for name in packages}
    except PackageNotFoundError as error:
        raise RuntimeError("Install BGE support with: uv sync --extra bge") from error


def _save_array(path: Path, values: FloatArray) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as file:
        np.save(file, values, allow_pickle=False)
    temporary.replace(path)


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
    """Embed frozen text with BGE, reusing completed batches with identical inputs."""
    if model == "cohere":
        raise ValueError(
            "Cohere embed-english-v3.0 vectors are imported by `qbench prepare`; "
            "no embedding API calls are made."
        )
    if model not in LOCAL_MODELS:
        raise ValueError(f"model must be one of {sorted(LOCAL_MODELS)}.")
    model_id, dimensions = LOCAL_MODELS[model]
    if batch_size < 1:
        raise ValueError("batch_size must be positive.")
    if max_documents is not None:
        raise ValueError(
            "Use prepare_dataset(max_docs=...) to preserve query judgments in subsets."
        )
    revision = _require_commit(revision)
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
        "model": model_id,
        "model_revision": revision,
        "dimensions": dimensions,
        "corpus_input_type": "search_query"
        if dataset == "quora"
        else "search_document",
        "query_input_type": "search_query",
        "query_instruction": BGE_QUERY_INSTRUCTION if dataset != "quora" else "",
        "truncate": "right, max_seq_length=512",
        "normalization": "L2 float32",
        "source_manifest_sha256": _checksum(dataset_dir / "manifest.json"),
        "library_versions": library_versions,
        "device": "cpu",
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
    if artifact_complete(output):
        return
    try:
        from sentence_transformers import SentenceTransformer  # ty: ignore[unresolved-import]
    except ImportError as error:
        raise RuntimeError("Install BGE support with: uv sync --extra bge") from error
    encoder = await asyncio.to_thread(
        SentenceTransformer, model_id, revision=revision, device="cpu"
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

    await _embed_batches(corpus, output, "corpus", config, batch_size, encode_corpus)
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

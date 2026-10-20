from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class EmbeddingData:
    corpus: NDArray[np.float32]
    queries: NDArray[np.float32]
    corpus_ids: list[str]
    query_ids: list[str]
    query_splits: dict[str, Literal["dev", "test"]]
    qrels: dict[str, dict[str, int]]
    manifest: dict[str, Any]


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


ARTIFACT_FILES = (
    "corpus.npy",
    "queries.npy",
    "corpus_ids.json",
    "query_ids.json",
    "query_splits.json",
    "qrels.json",
)
EXPECTED_DIMENSIONS = {"cohere": 1024, "bge": 384}
MODEL_LABELS = {
    "embed-english-v3.0": "cohere",
    "BAAI/bge-small-en-v1.5": "bge",
    "synthetic-normal": "synthetic",
}


def artifact_signature(path: Path) -> dict[str, str]:
    hashes = {}
    for name in ARTIFACT_FILES:
        with (path / name).open("rb") as stream:
            hashes[name] = hashlib.file_digest(stream, "sha256").hexdigest()
    return hashes


def load_embeddings(path: Path) -> EmbeddingData:
    """Load saved embeddings without copying the corpus into process memory."""
    return EmbeddingData(
        corpus=np.load(path / "corpus.npy", mmap_mode="r", allow_pickle=False),
        queries=np.load(path / "queries.npy", mmap_mode="r", allow_pickle=False),
        corpus_ids=_read_json(path / "corpus_ids.json"),
        query_ids=_read_json(path / "query_ids.json"),
        query_splits=_read_json(path / "query_splits.json"),
        qrels=_read_json(path / "qrels.json"),
        manifest=_read_json(path / "manifest.json"),
    )


def _top_positions(
    scores: NDArray[np.float32], indices: NDArray[np.int64], k: int
) -> NDArray[np.int64]:
    positions = np.flatnonzero(np.isfinite(scores))
    if len(positions) > k:
        values = scores[positions]
        threshold = np.partition(values, len(values) - k)[len(values) - k]
        above = positions[values > threshold]
        tied = positions[values == threshold]
        tied = tied[np.argsort(indices[tied], kind="stable")][: k - len(above)]
        positions = np.concatenate((above, tied))
    return positions[np.lexsort((indices[positions], -scores[positions]))]


def exact_ground_truth(
    data: EmbeddingData, k: int = 100, block_size: int = 4096
) -> dict[str, list[str]]:
    """Rank float32 dot products; break ties by corpus row and exclude self IDs."""
    if k < 1 or block_size < 1:
        raise ValueError("k and block_size must be positive")
    if data.corpus.shape[1] != data.queries.shape[1]:
        raise ValueError("Corpus and query dimensions differ")
    corpus_positions = {doc_id: row for row, doc_id in enumerate(data.corpus_ids)}
    rankings: dict[str, list[str]] = {}
    for query_start in range(0, len(data.query_ids), 256):
        query_ids = data.query_ids[query_start : query_start + 256]
        queries = data.queries[query_start : query_start + 256]
        best_scores = [np.empty(0, dtype=np.float32) for _ in query_ids]
        best_indices = [np.empty(0, dtype=np.int64) for _ in query_ids]
        for start in range(0, len(data.corpus_ids), block_size):
            stop = min(start + block_size, len(data.corpus_ids))
            block_scores = (data.corpus[start:stop] @ queries.T).T
            block_indices = np.arange(start, stop, dtype=np.int64)
            for row, query_id in enumerate(query_ids):
                self_index = corpus_positions.get(query_id)
                if self_index is not None and start <= self_index < stop:
                    block_scores[row, self_index - start] = -np.inf
                scores = np.concatenate((best_scores[row], block_scores[row]))
                indices = np.concatenate((best_indices[row], block_indices))
                selected = _top_positions(scores, indices, k)
                best_scores[row] = scores[selected]
                best_indices[row] = indices[selected]
        for query_id, indices in zip(query_ids, best_indices, strict=True):
            rankings[query_id] = [data.corpus_ids[index] for index in indices]
    return rankings


def quality_metric(
    result_ids: list[str],
    exact_ids: list[str],
    relevance: dict[str, int],
    k: int,
) -> tuple[float, float]:
    """Return ANN recall at k and nDCG at 10 using linear relevance gains."""
    if k < 1:
        raise ValueError("k must be positive")
    if len(result_ids) != len(set(result_ids)):
        raise ValueError("Search results contain duplicate document IDs")
    exact = set(exact_ids[:k])
    if not exact:
        raise ValueError("ANN recall requires at least one exact neighbor")
    recall = len(set(result_ids[:k]) & exact) / len(exact)
    gains = np.asarray([max(relevance.get(doc_id, 0), 0) for doc_id in result_ids[:10]])
    ideal = np.asarray(
        sorted((max(grade, 0) for grade in relevance.values()), reverse=True)[:10]
    )
    dcg = float(np.sum(gains / np.log2(np.arange(len(gains)) + 2)))
    ideal_dcg = float(np.sum(ideal / np.log2(np.arange(len(ideal)) + 2)))
    return recall, dcg / ideal_dcg if ideal_dcg > 0 else 0.0


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize one workload without counting repeated queries as new labels."""
    group_fields = (
        "dataset",
        "model",
        "config",
        "split",
        "k",
        "cache_state",
        "hnsw_ef",
        "rescore",
        "oversampling",
        "build",
    )
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[tuple(record.get(field) for field in group_fields)].append(record)
    summaries: list[dict[str, Any]] = []
    for key, rows in groups.items():
        summary = dict(zip(group_fields, key, strict=True))
        summary["requests"] = len(rows)
        summary["queries"] = len({row["query_id"] for row in rows})
        for metric in (
            "ann_recall",
            "ndcg_at_10",
            "exact_ndcg_at_10",
            "delta_ndcg_at_10",
        ):
            by_query: dict[str, list[float]] = defaultdict(list)
            for row in rows:
                if row.get(metric) is not None:
                    by_query[row["query_id"]].append(float(row[metric]))
            if by_query:
                summary[f"mean_{metric}"] = float(
                    np.mean([np.mean(values) for values in by_query.values()])
                )
        for metric in ("request_ms", "server_ms"):
            values = [float(row[metric]) for row in rows if row.get(metric) is not None]
            if values:
                summary[f"{metric}_p50"] = float(np.percentile(values, 50))
                summary[f"{metric}_p95"] = float(np.percentile(values, 95))
        summaries.append(summary)
    return {"groups": summaries}


def _check_vectors(vectors: NDArray[np.float32], ids: list[str], label: str) -> None:
    if vectors.ndim != 2 or vectors.shape[0] != len(ids) or vectors.shape[1] == 0:
        raise ValueError(f"{label}: expected one nonempty vector per ID")
    if (
        not ids
        or len(ids) != len(set(ids))
        or any(not isinstance(value, str) or not value for value in ids)
    ):
        raise ValueError(f"{label}: IDs must be nonempty, unique strings")
    if vectors.dtype != np.float32:
        raise ValueError(f"{label}: expected float32, received {vectors.dtype}")
    for start in range(0, len(ids), 4096):
        block = vectors[start : start + 4096]
        if not np.isfinite(block).all():
            raise ValueError(f"{label}: vectors contain nonfinite values")
        if not np.allclose(np.linalg.norm(block, axis=1), 1.0, atol=1e-5, rtol=1e-5):
            raise ValueError(f"{label}: vectors must be normalized to unit length")


def validate_embeddings(data: EmbeddingData) -> None:
    """Validate vectors, IDs, query splits, and relevance judgments."""
    _check_vectors(data.corpus, data.corpus_ids, "corpus")
    _check_vectors(data.queries, data.query_ids, "queries")
    if data.corpus.shape[1] != data.queries.shape[1]:
        raise ValueError("Corpus and query dimensions differ")
    corpus_ids = set(data.corpus_ids)
    for query_id in data.query_ids:
        if data.query_splits.get(query_id) not in {"dev", "test"}:
            raise ValueError(f"Missing or invalid split for query {query_id}")
        relevance = data.qrels.get(query_id)
        if not relevance:
            raise ValueError(f"No positive relevance judgments for query {query_id}")
        if any(not isinstance(grade, int) for grade in relevance.values()):
            raise ValueError(f"Relevance judgments must be integers for {query_id}")
        if not any(grade > 0 for grade in relevance.values()):
            raise ValueError(f"No positive relevance judgments for query {query_id}")
        if not set(relevance).issubset(corpus_ids):
            raise ValueError(
                f"Relevance judgments reference missing corpus IDs for {query_id}"
            )


def verify_fixture(data_dir: Path) -> dict[str, object]:
    """Check embedding artifacts and bounded-memory exact-search invariants."""
    data = load_embeddings(data_dir)
    validate_embeddings(data)
    exact = exact_ground_truth(data)
    for query_id, ranked in exact.items():
        if query_id in ranked:
            raise ValueError(f"Exact search returned a self match for {query_id}")
        recall, _ = quality_metric(ranked, ranked, data.qrels[query_id], k=100)
        if recall != 1.0:
            raise ValueError(f"Exact search self-recall was not one for {query_id}")
    return {
        "status": "ok",
        "documents": len(data.corpus_ids),
        "queries": len(data.query_ids),
        "dimensions": data.corpus.shape[1],
        "dtype": str(data.corpus.dtype),
        "exact_self_recall": 1.0,
        "self_matches_excluded": True,
    }


def artifact_status(path: Path, dataset: str, model: str) -> dict[str, Any]:
    """Readiness of one dataset/model artifact: hashes, dimensions, IDs, labels."""
    report: dict[str, Any] = {"dataset": dataset, "model": model, "path": str(path)}
    if not (path / "manifest.json").is_file():
        return {**report, "status": "missing", "problems": ["manifest.json absent"]}
    problems: list[str] = []
    try:
        data = load_embeddings(path)
        if artifact_signature(path) != data.manifest.get("checksums"):
            problems.append("artifact checksums differ from manifest")
        validate_embeddings(data)
        if data.manifest.get("dataset") != dataset:
            problems.append(f"manifest dataset is {data.manifest.get('dataset')!r}")
        if MODEL_LABELS.get(str(data.manifest.get("model"))) != model:
            problems.append(f"manifest model is {data.manifest.get('model')!r}")
        expected = EXPECTED_DIMENSIONS.get(model)
        if expected is not None and data.corpus.shape[1] != expected:
            problems.append(f"dimensions {data.corpus.shape[1]} != {expected}")
        if data.manifest.get("synthetic"):
            problems.append("artifact is synthetic")
        splits = {"dev": 0, "test": 0}
        for query_id in data.query_ids:
            splits[data.query_splits[query_id]] += 1
        if not splits["dev"] or not splits["test"]:
            problems.append("both dev and test queries are required")
        report.update(
            {
                "documents": len(data.corpus_ids),
                "dev_queries": splits["dev"],
                "test_queries": splits["test"],
                "dimensions": int(data.corpus.shape[1]),
                "source_revision": data.manifest.get("source_manifest", {}).get(
                    "source_revision"
                )
                or data.manifest.get("source_revision"),
                "model_revision": data.manifest.get("model_revision"),
            }
        )
        ground_truth = path / "ground_truth.json"
        if not ground_truth.is_file():
            report["ground_truth"] = "missing"
        else:
            cached = json.loads(ground_truth.read_text())
            report["ground_truth"] = (
                "ok" if cached.get("artifacts") == artifact_signature(path) else "stale"
            )
            if report["ground_truth"] == "stale":
                problems.append("ground_truth.json does not match the artifacts")
    except (OSError, ValueError, KeyError, TypeError) as error:
        problems.append(f"{type(error).__name__}: {error}")
    report["status"] = "failed" if problems else "ready"
    report["problems"] = problems
    return report

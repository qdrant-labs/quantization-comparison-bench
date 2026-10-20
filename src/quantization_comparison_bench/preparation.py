from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .data import (
    BGE_MODEL,
    BGE_REVISION,
    COHERE_MODEL,
    DATASET_REVISION,
    DATASET_SOURCE,
    embed_dataset,
    prepare_dataset,
)
from .evaluation import EmbeddingData, load_embeddings, validate_embeddings

DATASETS = ("fiqa", "scifact", "nfcorpus", "quora")
RAW_FILES = {"corpus.jsonl", "queries.jsonl", "qrels.jsonl"}
EMBEDDING_FILES = {
    "corpus.npy",
    "queries.npy",
    "corpus_ids.json",
    "query_ids.json",
    "query_splits.json",
    "qrels.json",
}


def checked_manifest(
    directory: Path, expected_files: set[str] | None = None
) -> dict[str, Any]:
    """Check all manifest files before reusing a prepared artifact."""
    manifest = json.loads((directory / "manifest.json").read_text())
    if not manifest.get("checksums"):
        raise ValueError(f"No artifact checksums in {directory}")
    if expected_files is not None and set(manifest["checksums"]) != expected_files:
        raise ValueError(
            f"Manifest does not checksum every required artifact: {directory}"
        )
    for name, expected in manifest["checksums"].items():
        with (directory / name).open("rb") as file:
            actual = hashlib.file_digest(file, "sha256").hexdigest()
        if actual != expected:
            raise ValueError(f"Checksum mismatch: {directory / name}")
    return manifest


def checked_raw(directory: Path, dataset: str) -> dict[str, Any]:
    manifest = checked_manifest(directory, RAW_FILES)
    if (
        manifest["schema_version"] != 1
        or manifest["dataset"] != dataset
        or manifest["source"] != DATASET_SOURCE
        or manifest["source_revision"] != DATASET_REVISION
        or manifest["smoke_subset"]
        or manifest["seed"] != 42
        or manifest["counts"]["dev_queries"] != 50
        or manifest["counts"]["documents"]
        != manifest["counts"]["full_corpus_documents"]
    ):
        raise ValueError(
            f"Existing artifact uses different staging settings: {directory}"
        )
    return manifest


def check_embedding_source(
    data: EmbeddingData,
    raw: Path,
    dataset: str,
    model: str,
    source: dict[str, Any],
) -> None:
    manifest = data.manifest
    expected_model = COHERE_MODEL if model == "cohere" else BGE_MODEL
    expected_revision = DATASET_REVISION if model == "cohere" else BGE_REVISION
    expected_source = DATASET_SOURCE if model == "cohere" else BGE_MODEL
    if (
        manifest["dataset"] != dataset
        or manifest["model"] != expected_model
        or manifest["dimensions"] != (1024 if model == "cohere" else 384)
        or manifest["embedding_source"] != expected_source
        or manifest["embedding_source_revision"] != expected_revision
        or manifest["model_revision"] != (None if model == "cohere" else BGE_REVISION)
        or manifest["synthetic"]
    ):
        raise ValueError(
            f"Unexpected embedding model/source identity: {dataset}/{model}"
        )
    source_hash = hashlib.sha256((raw / "manifest.json").read_bytes()).hexdigest()
    if (
        manifest["source_manifest_sha256"] != source_hash
        or manifest["source_manifest"] != source
    ):
        raise ValueError(
            f"Embeddings belong to a different raw corpus: {dataset}/{model}"
        )
    expected_queries = (
        source["counts"]["dev_queries"] + source["counts"]["test_queries"]
    )
    if (
        len(data.corpus_ids) != source["counts"]["documents"]
        or len(data.query_ids) != expected_queries
        or sum(value == "dev" for value in data.query_splits.values()) != 50
    ):
        raise ValueError(
            f"Embedding counts do not match raw staging: {dataset}/{model}"
        )


async def prepare_all(
    data_root: Path, with_bge: bool = False, device: str = "cpu"
) -> dict[str, Any]:
    """Stage all full corpora and supplied vectors; optionally compute BGE."""
    for dataset in DATASETS:
        raw = data_root / "raw" / dataset
        if (raw / "manifest.json").exists():
            checked_raw(raw, dataset)
        else:
            prepare_dataset(dataset, data_root / "raw", DATASET_REVISION)
        for model in ("cohere", "bge") if with_bge else ("cohere",):
            await embed_dataset(
                raw,
                data_root / "embeddings" / dataset / model,
                model,
                revision=BGE_REVISION if model == "bge" else None,
                device=device,
            )
    return readiness(data_root)


def readiness(data_root: Path) -> dict[str, Any]:
    """Validate data without computing all exact nearest-neighbor rankings."""
    entries: list[dict[str, Any]] = []
    for dataset in DATASETS:
        raw = data_root / "raw" / dataset
        source: dict[str, Any] | None = None
        if not (raw / "manifest.json").exists():
            entries.append({"dataset": dataset, "artifact": "raw", "status": "missing"})
        else:
            source = checked_raw(raw, dataset)
            entries.append(
                {
                    "dataset": dataset,
                    "artifact": "raw",
                    "status": "ready",
                    "counts": source["counts"],
                    "checksums": source["checksums"],
                }
            )
        data_by_model: dict[str, EmbeddingData] = {}
        for model in ("cohere", "bge"):
            path = data_root / "embeddings" / dataset / model
            entry: dict[str, Any] = {"dataset": dataset, "artifact": model}
            if not (path / "manifest.json").exists():
                entry["status"] = "missing"
            else:
                manifest = checked_manifest(path, EMBEDDING_FILES)
                data = load_embeddings(path)
                validate_embeddings(data)
                if source is not None:
                    check_embedding_source(data, raw, dataset, model, source)
                data_by_model[model] = data
                entry.update(
                    status="ready" if source is not None else "raw_source_missing",
                    model=manifest["model"],
                    dimensions=data.corpus.shape[1],
                    counts=manifest["counts"],
                    checksums=manifest["checksums"],
                )
            entries.append(entry)
        if len(data_by_model) == 2:
            cohere, bge = data_by_model["cohere"], data_by_model["bge"]
            if (
                cohere.corpus_ids != bge.corpus_ids
                or cohere.query_ids != bge.query_ids
                or cohere.query_splits != bge.query_splits
                or cohere.qrels != bge.qrels
            ):
                raise ValueError(
                    f"Models do not use identical IDs/order, splits, and qrels for {dataset}"
                )
    report = {
        "checked_at_utc": datetime.now(UTC).isoformat(),
        "source_revision": DATASET_REVISION,
        "bge_revision": BGE_REVISION,
        "all_data_ready": all(entry["status"] == "ready" for entry in entries),
        "artifacts": entries,
        "scope": "Artifact readiness only; VM performance has not been measured.",
    }
    data_root.mkdir(parents=True, exist_ok=True)
    (data_root / "readiness.json").write_text(json.dumps(report, indent=2) + "\n")
    return report

import asyncio
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from quantization_comparison_bench import data as data_module
from quantization_comparison_bench.evaluation import (
    artifact_status,
    load_embeddings,
    verify_fixture,
)

DIMENSIONS = data_module.COHERE_DIMENSIONS


def _vectors(rng: np.random.Generator, count: int) -> list[list[float]]:
    values = rng.standard_normal((count, DIMENSIONS)).astype(np.float32)
    values /= np.linalg.norm(values, axis=1, keepdims=True)
    # Source vectors are approximately, not exactly, unit length.
    values *= 1.0 + rng.uniform(-1e-4, 1e-4, size=(count, 1)).astype(np.float32)
    return values.tolist()


@pytest.fixture
def fake_source(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    rng = np.random.default_rng(7)
    source = tmp_path / "source"
    source.mkdir()
    # Deliberately unsorted IDs so row order must be rebuilt by sorted ID.
    doc_ids = [str(index) for index in (10, 2, 33, 4, 150, 6, 7, 8, 9, 1, 11, 12)]
    corpus = pa.table(
        {
            "_id": doc_ids,
            "title": ["Title " + doc_id if doc_id != "2" else "" for doc_id in doc_ids],
            "text": ["body " + doc_id for doc_id in doc_ids],
            "emb": _vectors(rng, len(doc_ids)),
        }
    )
    corpus_dir = source / "corpus"
    corpus_dir.mkdir()
    pq.write_table(corpus.slice(0, 5), corpus_dir / "0000.parquet")
    pq.write_table(corpus.slice(5), corpus_dir / "0001.parquet")
    files: dict[str, Path] = {"corpus": corpus_dir}
    for split, query_ids in (("dev", ["q1", "q2", "q3"]), ("test", ["q4", "10"])):
        queries = pa.table(
            {
                "_id": query_ids,
                "text": ["question " + qid for qid in query_ids],
                "emb": _vectors(rng, len(query_ids)),
            }
        )
        qrels = pa.table(
            {
                "query_id": query_ids + [query_ids[0]],
                "corpus_id": ["2", "4", "6", "33", "150", "7"][: len(query_ids)]
                + ["9"],
                "score": [1, 2, 1, 1, 1, 0][: len(query_ids)] + [1],
            }
        )
        (source / "queries").mkdir(exist_ok=True)
        (source / "qrels").mkdir(exist_ok=True)
        pq.write_table(queries, source / "queries" / f"{split}.parquet")
        pq.write_table(qrels, source / "qrels" / f"{split}.parquet")
        files[f"queries/{split}"] = source / "queries" / f"{split}.parquet"
        files[f"qrels/{split}"] = source / "qrels" / f"{split}.parquet"

    def fake_files(dataset: str, component: str, split: str, revision: str):
        assert revision == "f" * 40
        if component == "corpus":
            return sorted(corpus_dir.glob("*.parquet"))
        return [files[f"{component}/{split}"]]

    monkeypatch.setattr(data_module, "_source_files", fake_files)
    monkeypatch.setattr(
        data_module,
        "_import_versions",
        lambda: {
            "numpy": np.__version__,
            "huggingface_hub": "test",
            "pyarrow": pa.__version__,
        },
    )
    return files


def test_prepare_imports_supplied_cohere_vectors(tmp_path: Path, fake_source) -> None:
    raw = tmp_path / "data" / "raw"
    data_module.prepare_dataset("quora", raw, "f" * 40, dev_queries=2, seed=1)
    artifact = tmp_path / "data" / "embeddings" / "quora" / "cohere"
    loaded = load_embeddings(artifact)
    assert loaded.corpus_ids == sorted(loaded.corpus_ids)
    assert loaded.corpus.shape == (12, DIMENSIONS)
    assert loaded.manifest["model"] == "embed-english-v3.0"
    assert loaded.manifest["source_manifest"]["source_embeddings_used"] is True
    assert loaded.manifest["source_norm_max_abs_deviation"] < 1e-3
    assert set(loaded.manifest["source_files_sha256"]) == {
        "quora/corpus/0000.parquet",
        "quora/corpus/0001.parquet",
        "quora/queries/dev.parquet",
        "quora/qrels/dev.parquet",
        "quora/queries/test.parquet",
        "quora/qrels/test.parquet",
    }
    # Vectors follow the sorted corpus and match the source rows after normalization.
    source = pq.read_table(fake_source["corpus"] / "0000.parquet").to_pylist()
    for row in source:
        position = loaded.corpus_ids.index(row["_id"])
        expected = np.asarray(row["emb"], dtype=np.float32)
        expected /= np.linalg.norm(expected)
        np.testing.assert_allclose(loaded.corpus[position], expected, atol=1e-6)
    splits = json.loads((artifact / "query_splits.json").read_text())
    assert sorted(splits.values()) == ["dev", "dev", "test", "test"]
    # Query "10" shares an ID with a document; exact search must exclude it.
    report = verify_fixture(artifact)
    assert report["self_matches_excluded"] is True
    assert loaded.qrels["q1"] == {"2": 1, "9": 1}
    manifest = json.loads((raw / "quora" / "manifest.json").read_text())
    assert manifest["counts"]["full_test_queries"] is True
    assert manifest["smoke_subset"] is False
    assert (raw / "quora" / "corpus.jsonl").read_text().count("\n") == 12
    text = [
        json.loads(line)
        for line in (raw / "quora" / "corpus.jsonl").read_text().splitlines()
    ]
    assert next(row for row in text if row["id"] == "2")["text"] == "body 2"
    assert next(row for row in text if row["id"] == "4")["text"] == "Title 4\nbody 4"
    status = artifact_status(artifact, "quora", "cohere")
    assert status["status"] == "ready" and status["ground_truth"] == "missing"


def test_prepare_all_is_idempotent(tmp_path: Path, fake_source) -> None:
    root = tmp_path / "data"
    first = asyncio.run(
        data_module.prepare_all(root, "f" * 40, False, None, 2, 1, datasets=["fiqa"])
    )
    assert first == [{"dataset": "fiqa", "raw": "prepared", "cohere": "prepared"}]
    second = asyncio.run(
        data_module.prepare_all(root, "f" * 40, False, None, 2, 1, datasets=["fiqa"])
    )
    assert second == [{"dataset": "fiqa", "raw": "present", "cohere": "present"}]


def test_cohere_embedding_is_not_an_api_call(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="imported by `qbench prepare`"):
        asyncio.run(data_module.embed_dataset(tmp_path, tmp_path / "out", "cohere"))


def test_readiness_reports_missing_artifacts(tmp_path: Path) -> None:
    status = artifact_status(tmp_path / "nope", "fiqa", "bge")
    assert status["status"] == "missing"

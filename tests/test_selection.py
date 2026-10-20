import json
from pathlib import Path

import pytest

from quantization_comparison_bench.config import load_config
from quantization_comparison_bench.selection import (
    select_cells,
    select_from_run,
    selected_config,
)


def group(config, k, hnsw_ef, rescore, oversampling, recall, ndcg, p95, build=0):
    return {
        "dataset": "fiqa",
        "model": "embed-english-v3.0",
        "config": config,
        "split": "dev",
        "k": k,
        "cache_state": "warm",
        "hnsw_ef": hnsw_ef,
        "rescore": rescore,
        "oversampling": oversampling,
        "build": build,
        "requests": 50,
        "queries": 50,
        "mean_ann_recall": recall,
        "mean_ndcg_at_10": ndcg,
        "mean_exact_ndcg_at_10": 0.40,
        "server_ms_p50": p95 / 2,
        "server_ms_p95": p95,
        "request_ms_p50": p95 / 2,
        "request_ms_p95": p95 + 1,
    }


def test_fastest_eligible_point_is_chosen_and_misses_are_reported() -> None:
    base = load_config(Path("configs/benchmark.yaml"))
    methods = [m for m in base.methods if m.name in {"float32", "turbo4"}]
    groups = []
    for k in (10, 100):
        groups += [
            group("float32", k, 64, None, None, 0.90, 0.40, 1.0),
            group("float32", k, 128, None, None, 0.96, 0.40, 2.0),
            group("float32", k, 256, None, None, 0.99, 0.40, 3.0),
        ]
    groups += [
        # k=10: the fast point fails recall; the middle one passes; slow passes.
        group("turbo4", 10, 64, True, 2.0, 0.93, 0.40, 1.0),
        group("turbo4", 10, 128, True, 2.0, 0.96, 0.395, 2.0),
        group("turbo4", 10, 256, True, 4.0, 0.98, 0.40, 3.0),
        # k=10 relevance miss despite recall: nDCG more than 0.01 below exact.
        group("turbo4", 10, 128, False, 1.0, 0.97, 0.38, 0.5),
        # k=100: nothing eligible; best recall should be kept and flagged.
        group("turbo4", 100, 64, True, 2.0, 0.80, 0.40, 1.0),
        group("turbo4", 100, 256, True, 8.0, 0.94, 0.40, 5.0),
    ]
    choices = select_cells(groups, methods, [10, 100])
    assert [row["hnsw_ef"] for row in choices["float32"]] == [128, 128]
    assert all(row["target_met"] for row in choices["float32"])
    turbo_10, turbo_100 = choices["turbo4"]
    assert (turbo_10["hnsw_ef"], turbo_10["rescore"], turbo_10["oversampling"]) == (
        128,
        True,
        2.0,
    )
    assert turbo_10["target_met"] and turbo_10["eligible"] == 2
    assert turbo_100["target_met"] is False
    assert turbo_100["hnsw_ef"] == 256 and turbo_100["eligible"] == 0


def test_builds_are_pooled_with_slowest_latency() -> None:
    base = load_config(Path("configs/benchmark.yaml"))
    methods = [m for m in base.methods if m.name == "float32"]
    groups = [
        group("float32", 10, 64, None, None, 0.96, 0.40, 1.0, build=0),
        group("float32", 10, 64, None, None, 0.96, 0.40, 9.0, build=1),
        group("float32", 10, 128, None, None, 0.97, 0.40, 2.0, build=0),
        group("float32", 10, 128, None, None, 0.97, 0.40, 2.5, build=1),
    ]
    choices = select_cells(groups, methods, [10])
    assert choices["float32"][0]["hnsw_ef"] == 128
    assert choices["float32"][0]["builds"] == 2


def test_selection_refuses_test_split_results() -> None:
    base = load_config(Path("configs/benchmark.yaml"))
    bad = group("float32", 10, 64, None, None, 0.99, 0.40, 1.0)
    bad["split"] = "test"
    with pytest.raises(ValueError):
        select_cells([bad], base.methods[:1], [10])


def test_select_from_run_writes_frozen_test_config(tmp_path: Path) -> None:
    base = load_config(Path("configs/benchmark.yaml"))
    tuning = load_config(Path("configs/tuning.yaml"))
    groups = []
    for method in tuning.methods:
        for cell in tuning.grid(method):
            groups.append(
                group(
                    method.name,
                    cell.k,
                    cell.hnsw_ef,
                    cell.rescore,
                    cell.oversampling,
                    0.95 + cell.hnsw_ef / 10000,
                    0.40,
                    float(cell.hnsw_ef) * (cell.oversampling or 1.0),
                )
            )
    run = tmp_path / "tuning"
    run.mkdir()
    (run / "summary.json").write_text(json.dumps({"groups": groups}))
    (run / "manifest.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "run_id": "abc123",
                "git_revision": "deadbeef",
                "model_label": "cohere",
                "configuration": tuning.model_dump(),
                "data_manifest": {"dataset": "fiqa"},
                "artifact_checksums": {"corpus.npy": "x"},
            }
        )
    )
    frozen = select_from_run(run, base, 0.95, 0.01, "server_ms_p95")
    assert frozen.datasets == ["fiqa"] and frozen.models == ["cohere"]
    assert frozen.search.split == "test"
    assert frozen.selection is not None
    for method in frozen.methods:
        assert method.cells is not None and [c.k for c in method.cells] == [10, 100]
        # The cheapest eligible effort is hnsw_ef=64 with oversampling 1.
        assert all(cell.hnsw_ef == 64 for cell in method.cells)
        if method.quantizer != "none":
            assert all(cell.oversampling == 1.0 for cell in method.cells)
    rebuilt = selected_config(
        base,
        json.loads((run / "manifest.json").read_text()),
        frozen.selection["choices"],
        frozen.selection["rule"],
    )
    assert rebuilt.methods == frozen.methods

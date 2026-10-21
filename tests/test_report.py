import json
from pathlib import Path

from quantization_comparison_bench.report import (
    bootstrap,
    build_report,
    macro_average,
    summarize_groups,
)


def record(
    query_id, build, repeat, recall, delta, server_ms, cache="warm", position=0, **extra
):
    return {
        "dataset": "fiqa",
        "model": "embed-english-v3.0",
        "config": extra.get("config", "turbo4"),
        "build": build,
        "repeat": repeat,
        "position": position,
        "split": "test",
        "k": 10,
        "rescore": True,
        "oversampling": 2.0,
        "cache_state": cache,
        "hnsw_ef": 256,
        "query_id": query_id,
        "ann_recall": recall,
        "ndcg_at_10": 0.5 + delta,
        "exact_ndcg_at_10": 0.5,
        "delta_ndcg_at_10": delta,
        "request_ms": server_ms + 1,
        "server_ms": server_ms,
        "cpu": 100,
        "vector_io_read": 10,
        "payload_io_read": None,
    }


def test_quality_is_per_query_first_and_builds_are_separated() -> None:
    rows = [
        # query a: repeated 3 times in build 0 with recall 1.0; once in build 1 at 0.0
        record("a", 0, 0, 1.0, 0.0, 1.0),
        record("a", 0, 1, 1.0, 0.0, 1.0),
        record("a", 0, 2, 1.0, 0.0, 1.0),
        record("a", 1, 0, 0.0, -0.1, 9.0),
        # query b: 0.5 in both builds
        record("b", 0, 0, 0.5, 0.0, 1.0),
        record("b", 1, 0, 0.5, 0.0, 1.0),
    ]
    (group,) = summarize_groups(rows)
    assert group["queries"] == 2 and group["builds"] == 2 and group["requests"] == 6
    # a = mean(1.0, 0.0) = 0.5 across builds; b = 0.5 -> mean 0.5, not request-weighted
    assert group["recall"] == 0.5
    assert group["recall_by_build"] == [0.75, 0.25]
    assert group["delta_ndcg_at_10"] == -0.025
    assert group["server_ms"]["p95"] > group["server_ms"]["p50"]
    assert group["cpu_per_query"] == 100 and group["payload_io_read_per_query"] is None
    assert len(group["recall_ci95"]) == 2


def test_bootstrap_interval_contains_the_mean_and_handles_tiny_inputs() -> None:
    values = [0.9, 1.0, 0.8, 1.0, 0.95, 0.7, 1.0, 0.85]
    low, high = bootstrap(values)
    assert low <= sum(values) / len(values) <= high
    assert bootstrap([1.0]) == []


def test_cold_start_head_and_tail_are_split() -> None:
    rows = [
        record(
            f"q{i}",
            0,
            0,
            1.0,
            0.0,
            50.0 if i < 10 else 1.0,
            cache="cold_start",
            position=i,
        )
        for i in range(30)
    ]
    (group,) = summarize_groups(rows)
    assert group["cold_head_server_ms"]["p50"] == 50.0
    assert group["warm_tail_server_ms"]["p50"] == 1.0
    assert group["cold_head_vector_io_read_per_query"] == 10


def _write_run(root: Path, stage: str, rows, status="complete", selection=None) -> None:
    run = root / stage
    (run / "build-0" / "turbo4").mkdir(parents=True)
    (run / "manifest.json").write_text(
        json.dumps(
            {
                "status": status,
                "run_id": "r",
                "git_revision": "abc",
                "server": {"version": "1.19.2"},
                "cache_state": "warm",
                "query_ids": ["a", "b"],
                "selection": selection,
            }
        )
    )
    (run / "queries.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    method = run / "build-0" / "turbo4"
    (method / "footprint.json").write_text(
        json.dumps(
            {
                "post_index_memory_allocated_bytes_delta_bytes": 2_000_000,
                "post_search_memory_allocated_bytes_delta_bytes": 1_000_000,
            }
        )
    )
    (method / "storage.json").write_text(json.dumps({"allocated_bytes": 5_000_000}))
    for name in ("pre-index", "post-index", "post-search"):
        (method / f"memory-{name}.json").write_text(
            json.dumps(
                {
                    "process_VmRSS_bytes": 100_000_000,
                    "process_RssFile_bytes": 40_000_000,
                    "memory.current": 150_000_000,
                    "memory.peak": 300_000_000,
                    "process_faults": {"minflt": 1, "majflt": 7},
                }
            )
        )


def test_build_report_writes_markdown_and_json(tmp_path: Path) -> None:
    results = tmp_path / "results"
    workload = results / "fiqa-cohere"
    rows = [record("a", 0, 0, 1.0, 0.0, 1.0), record("b", 0, 0, 0.9, -0.01, 2.0)]
    _write_run(workload, "primary", rows)
    _write_run(
        workload,
        "selected",
        rows,
        selection={
            "choices": {
                "turbo4": [
                    {
                        "k": 10,
                        "hnsw_ef": 256,
                        "mean_ann_recall": 0.97,
                        "target_met": True,
                    }
                ]
            }
        },
    )
    _write_run(workload, "memory-pressure", [], status="failed")
    (results / "environment").mkdir()
    (results / "pipeline-test-x").mkdir()
    summary = build_report(results, tmp_path / "report")
    assert summary["workloads"] == {
        "fiqa-cohere": {
            "primary": "complete",
            "selected": "complete",
            "memory-pressure": "failed",
        }
    }
    markdown = (tmp_path / "report" / "report.md").read_text()
    assert "### primary — complete" in markdown
    assert "### memory-pressure — failed" in markdown
    assert "| turbo4 | 10 | ef 256, rescore on, ×2 | 0.970 | yes |" in markdown
    assert (
        "| turbo4 | 1/1 | 1 | 2 | 100 | 40 | 300 | n/a | n/a | n/a | n/a | 0 |"
        in markdown
    )
    data = json.loads((tmp_path / "report" / "report.json").read_text())
    macro = macro_average(data["workloads"])
    assert macro[0]["model"] == "cohere" and macro[0]["datasets"] == ["fiqa"]
    assert abs(macro[0]["recall"] - 0.95) < 1e-9


def test_live_segment_storage_ignores_pre_merge_segments() -> None:
    from quantization_comparison_bench.report import live_segment_storage

    files = {
        "0/segments/merged/vector_storage/matrix.dat": {"allocated_bytes": 2000},
        "0/segments/merged/vector_storage/quantized.data": {"allocated_bytes": 500},
        "0/segments/merged/vector_index/links_compressed.bin": {"allocated_bytes": 40},
        "0/segments/merged/id_tracker.mappings": {"allocated_bytes": 5},
        "0/segments/old-a/vector_storage/matrix.dat": {"allocated_bytes": 1000},
        "0/segments/old-a/vector_storage/quantized.data": {"allocated_bytes": 250},
        "0/segments/old-b/vector_storage/matrix.dat": {"allocated_bytes": 900},
        "0/wal/closed-1": {"allocated_bytes": 32},
    }
    out = live_segment_storage({"files": files})
    assert out["vectors"] == 2000 and out["quantized"] == 500 and out["hnsw"] == 40
    assert out["other"] == 5 and out["wal"] == 32
    assert out["segment_directories"] == 3 and out["live_total"] == 2545


def test_throughput_fields_are_summarized() -> None:
    rows = [
        {**record("a", 0, 0, 1.0, 0.0, 1.0), "concurrency": 8, "pass_qps": 1200.0},
        {**record("b", 0, 0, 1.0, 0.0, 1.0), "concurrency": 8, "pass_qps": 1200.0},
        {**record("a", 0, 1, 1.0, 0.0, 1.0), "concurrency": 8, "pass_qps": 1000.0},
        {**record("b", 0, 1, 1.0, 0.0, 1.0), "concurrency": 8, "pass_qps": 1000.0},
    ]
    (group,) = summarize_groups(rows)
    assert group["concurrency"] == 8 and group["qps"] == 1100.0

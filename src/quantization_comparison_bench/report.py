"""Turn completed run directories into tables with paired uncertainty.

Reads `results/<dataset>-<model>/<stage>/` for the stages written by
scripts/run-workload.sh and scripts/run-all.sh. Quality is averaged per query
over repeats and builds first; confidence intervals come from a seeded
bootstrap over query IDs, so repeated requests never count as new evidence.
Latency percentiles pool every measured request. Nothing is interpolated.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml

STAGES = (
    "primary",
    "primary-full",
    "selected",
    "selected-full",
    "throughput",
    "datatype",
    "datatype-binary",
    "ablation",
    "cold-start",
    "memory-pressure",
    "memory-pressure-bge",
)
SEGMENT = re.compile(r"^0/segments/([^/]+)/")


def live_segment_storage(storage: dict[str, Any]) -> dict[str, Any]:
    """Bytes of the merged segment only, by component, plus the WAL.

    Right after indexing the optimizer may not have deleted the pre-merge
    segments yet, so summing every file double counts the originals and the
    quantized copy. The largest segment is the merged one.
    """
    segments: dict[str, dict[str, int]] = defaultdict(dict)
    wal = 0
    for name, entry in (storage.get("files") or {}).items():
        allocated = int(entry["allocated_bytes"])
        if "/wal/" in name:
            wal += allocated
        elif match := SEGMENT.match(name):
            segments[match.group(1)][name] = allocated
    out = {"vectors": 0, "quantized": 0, "hnsw": 0, "other": 0, "wal": wal}
    if segments:
        live = max(segments.values(), key=lambda files: sum(files.values()))
        for name, allocated in live.items():
            if "quantized" in name:
                out["quantized"] += allocated
            elif "vector_index" in name:
                out["hnsw"] += allocated
            elif "vector_storage" in name:
                out["vectors"] += allocated
            else:
                out["other"] += allocated
    out["segment_directories"] = len(segments)
    out["live_total"] = out["vectors"] + out["quantized"] + out["hnsw"] + out["other"]
    return out


GROUP = ("config", "k", "hnsw_ef", "rescore", "oversampling", "cache_state")
COLD_HEAD = 10  # first requests of a cold-start pass reported separately


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def bootstrap(values: list[float], resamples: int = 2000, seed: int = 0) -> list[float]:
    """95% interval of the mean over query IDs; empty when fewer than two."""
    if len(values) < 2:
        return []
    rng = np.random.default_rng(seed)
    array = np.asarray(values, dtype=np.float64)
    means = array[rng.integers(0, len(array), size=(resamples, len(array)))].mean(
        axis=1
    )
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def _percentiles(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"p50": None, "p95": None}
    return {
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
    }


def summarize_groups(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[tuple(record.get(field) for field in GROUP)].append(record)
    rows = []
    for key, members in groups.items():
        row: dict[str, Any] = dict(zip(GROUP, key, strict=True))
        builds = sorted({int(member["build"]) for member in members})
        row["builds"] = len(builds)
        row["requests"] = len(members)
        # Per query and build first, then per query, then the mean over queries.
        per_query_build: dict[tuple[str, int], dict[str, list[float]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for member in members:
            cell = per_query_build[(member["query_id"], int(member["build"]))]
            cell["recall"].append(float(member["ann_recall"]))
            cell["delta"].append(float(member["delta_ndcg_at_10"]))
            cell["ndcg"].append(float(member["ndcg_at_10"]))
            cell["exact"].append(float(member["exact_ndcg_at_10"]))
        per_query: dict[str, dict[str, list[float]]] = defaultdict(
            lambda: defaultdict(list)
        )
        per_build: dict[int, dict[str, list[float]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for (query_id, build), cell in per_query_build.items():
            for metric, values in cell.items():
                per_query[query_id][metric].append(float(np.mean(values)))
                per_build[build][metric].append(float(np.mean(values)))
        query_means = {
            metric: [float(np.mean(cell[metric])) for cell in per_query.values()]
            for metric in ("recall", "delta", "ndcg", "exact")
        }
        row["queries"] = len(per_query)
        row["recall"] = float(np.mean(query_means["recall"]))
        row["recall_ci95"] = bootstrap(query_means["recall"])
        row["ndcg_at_10"] = float(np.mean(query_means["ndcg"]))
        row["exact_ndcg_at_10"] = float(np.mean(query_means["exact"]))
        row["delta_ndcg_at_10"] = float(np.mean(query_means["delta"]))
        row["delta_ndcg_ci95"] = bootstrap(query_means["delta"], seed=1)
        build_recall = [float(np.mean(per_build[build]["recall"])) for build in builds]
        row["recall_by_build"] = build_recall
        row["recall_build_range"] = (
            [min(build_recall), max(build_recall)] if build_recall else []
        )
        for metric in ("server_ms", "request_ms"):
            values = [float(m[metric]) for m in members if m.get(metric) is not None]
            row[metric] = _percentiles(values)
            row[f"{metric}_p95_by_build"] = [
                _percentiles(
                    [float(m[metric]) for m in members if int(m["build"]) == build]
                )["p95"]
                for build in builds
            ]
        for metric in ("cpu", "vector_io_read", "payload_io_read"):
            values = [float(m[metric]) for m in members if m.get(metric) is not None]
            row[f"{metric}_per_query"] = float(np.mean(values)) if values else None
        row["concurrency"] = max(int(m.get("concurrency") or 1) for m in members)
        qps = {
            (int(m["build"]), int(m["repeat"])): float(m["pass_qps"])
            for m in members
            if m.get("pass_qps") is not None
        }
        row["qps"] = float(np.mean(list(qps.values()))) if qps else None
        if key[GROUP.index("cache_state")] == "cold_start":
            head = [
                float(m["server_ms"])
                for m in members
                if m.get("position", 0) < COLD_HEAD
            ]
            tail = [
                float(m["server_ms"])
                for m in members
                if m.get("position", 0) >= COLD_HEAD
            ]
            row["cold_head_server_ms"] = _percentiles(head)
            row["warm_tail_server_ms"] = _percentiles(tail)
            reads = [
                float(m["vector_io_read"])
                for m in members
                if m.get("position", 0) < COLD_HEAD
                and m.get("vector_io_read") is not None
            ]
            row["cold_head_vector_io_read_per_query"] = (
                float(np.mean(reads)) if reads else None
            )
        rows.append(row)
    rows.sort(
        key=lambda r: (
            r["k"],
            r["config"],
            r["hnsw_ef"],
            str(r["rescore"]),
            r["oversampling"] or 0,
        )
    )
    return rows


def _read_json(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text()) if path.is_file() else None


def resources(run: Path) -> list[dict[str, Any]]:
    """Per-method memory and storage, averaged over builds with the peak kept."""
    by_config: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for build_dir in sorted(run.glob("build-*")):
        for method_dir in sorted(p for p in build_dir.iterdir() if p.is_dir()):
            entry: dict[str, Any] = {"build": int(build_dir.name.split("-")[1])}
            footprint = _read_json(method_dir / "footprint.json") or {}
            entry["heap_post_index_bytes"] = footprint.get(
                "post_index_memory_allocated_bytes_delta_bytes"
            )
            entry["heap_post_search_bytes"] = footprint.get(
                "post_search_memory_allocated_bytes_delta_bytes"
            )
            storage = _read_json(method_dir / "storage.json") or {}
            entry["storage_allocated_bytes"] = storage.get("allocated_bytes")
            live = live_segment_storage(storage) if storage.get("files") else {}
            for key in ("vectors", "quantized", "hnsw", "wal", "live_total"):
                entry[f"storage_{key}_bytes"] = live.get(key)
            entry["segment_directories"] = live.get("segment_directories")
            for stage in ("pre-index", "post-index", "post-search"):
                snapshot = _read_json(method_dir / f"memory-{stage}.json") or {}
                label = stage.replace("-", "_")
                entry[f"rss_{label}_bytes"] = snapshot.get("process_VmRSS_bytes")
                entry[f"rss_file_{label}_bytes"] = snapshot.get("process_RssFile_bytes")
                entry[f"cgroup_current_{label}_bytes"] = snapshot.get("memory.current")
                entry[f"cgroup_peak_{label}_bytes"] = snapshot.get("memory.peak")
                entry[f"majflt_{label}"] = (snapshot.get("process_faults") or {}).get(
                    "majflt"
                )
            samples = read_jsonl(method_dir / "resources.jsonl")
            rss = [
                s["process_VmRSS_bytes"]
                for s in samples
                if s.get("process_VmRSS_bytes")
            ]
            current = [
                s["memory.current"]
                for s in samples
                if isinstance(s.get("memory.current"), int)
            ]
            entry["rss_sampled_max_bytes"] = max(rss) if rss else None
            entry["cgroup_current_sampled_max_bytes"] = (
                max(current) if current else None
            )
            entry["complete"] = (method_dir / "footprint.json").is_file()
            by_config[method_dir.name].append(entry)
    rows = []
    for config, entries in sorted(by_config.items()):
        row: dict[str, Any] = {"config": config, "builds": len(entries)}
        row["complete_builds"] = sum(e["complete"] for e in entries)
        for key in entries[0]:
            if key in {"build", "complete"}:
                continue
            values = [e[key] for e in entries if e.get(key) is not None]
            if not values:
                row[key] = None
            elif "peak" in key or "max" in key:
                row[key] = max(values)
            else:
                row[key] = float(np.mean(values))
        rows.append(row)
    return rows


def run_report(run: Path) -> dict[str, Any]:
    manifest = _read_json(run / "manifest.json") or {}
    records = read_jsonl(run / "queries.jsonl")
    report: dict[str, Any] = {
        "path": str(run),
        "status": manifest.get("status", "missing"),
        "error_type": manifest.get("error_type"),
        "run_id": manifest.get("run_id"),
        "git_revision": manifest.get("git_revision"),
        "server_version": (manifest.get("server") or {}).get("version"),
        "cache_state": manifest.get("cache_state"),
        "queries_declared": len(manifest.get("query_ids") or []),
        "telemetry": manifest.get("telemetry"),
        "groups": summarize_groups(records),
        "resources": resources(run),
        "method_failures": manifest.get("method_failures") or [],
    }
    if manifest.get("selection"):
        report["selection"] = manifest["selection"]
    return report


def workload_reports(results: Path) -> dict[str, dict[str, Any]]:
    reports: dict[str, dict[str, Any]] = {}
    for workload in sorted(p for p in results.iterdir() if p.is_dir()):
        if workload.name.startswith(("environment", "pipeline-test", "smoke")):
            continue
        stages = {
            stage: run_report(workload / stage)
            for stage in STAGES
            if (workload / stage / "manifest.json").is_file()
        }
        if not stages:
            continue
        entry: dict[str, Any] = {"stages": stages}
        selected = workload / "selected.yaml"
        if selected.is_file():
            entry["selected_config"] = yaml.safe_load(selected.read_text())
        reports[workload.name] = entry
    return reports


def macro_average(reports: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Secondary summary: equal-weight mean over datasets of the primary stage."""
    buckets: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for name, entry in reports.items():
        primary = entry["stages"].get("primary")
        if not primary or primary["status"] != "complete":
            continue
        dataset, _, model = name.rpartition("-")
        for group in primary["groups"]:
            buckets[(model, group["config"], group["k"])].append(
                {**group, "dataset": dataset}
            )
    rows = []
    for (model, config, k), groups in sorted(buckets.items()):
        rows.append(
            {
                "model": model,
                "config": config,
                "k": k,
                "datasets": [g["dataset"] for g in groups],
                "recall": float(np.mean([g["recall"] for g in groups])),
                "delta_ndcg_at_10": float(
                    np.mean([g["delta_ndcg_at_10"] for g in groups])
                ),
                "server_p95_ms": float(
                    np.mean(
                        [
                            g["server_ms"]["p95"]
                            for g in groups
                            if g["server_ms"]["p95"] is not None
                        ]
                    )
                ),
            }
        )
    return rows


# ---------------------------------------------------------------- markdown


def _f(value: float | None, digits: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def _ci(interval: list[float], digits: int = 3) -> str:
    return (
        "" if not interval else f" [{interval[0]:.{digits}f}, {interval[1]:.{digits}f}]"
    )


def _mb(value: float | None) -> str:
    return "n/a" if value is None else f"{value / 1e6:,.0f}"


def _point(group: dict[str, Any]) -> str:
    if group["rescore"] is None:
        return f"ef {group['hnsw_ef']}"
    rescore = "on" if group["rescore"] else "off"
    return f"ef {group['hnsw_ef']}, rescore {rescore}, ×{group['oversampling']:g}"


def _quality_table(groups: list[dict[str, Any]], show_point: bool) -> list[str]:
    header = (
        ["Method", "k"]
        + (["Point"] if show_point else [])
        + [
            "ANN recall [95% CI]",
            "Build range",
            "ΔnDCG@10 [95% CI]",
            "Server p50 / p95 ms",
            "Client p95 ms",
            "QPS @ concurrency",
            "CPU / query",
            "Vector reads / query",
        ]
    )
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for g in groups:
        cells = (
            [g["config"], str(g["k"])]
            + ([_point(g)] if show_point else [])
            + [
                _f(g["recall"]) + _ci(g["recall_ci95"]),
                (
                    f"{g['recall_build_range'][0]:.3f}–{g['recall_build_range'][1]:.3f}"
                    if g["recall_build_range"]
                    else "n/a"
                ),
                f"{g['delta_ndcg_at_10']:+.3f}" + _ci(g["delta_ndcg_ci95"]),
                f"{_f(g['server_ms']['p50'], 2)} / {_f(g['server_ms']['p95'], 2)}",
                _f(g["request_ms"]["p95"], 2),
                f"{_f(g.get('qps'), 0)} @ {g.get('concurrency', 1)}",
                _f(g["cpu_per_query"], 0),
                _f(g["vector_io_read_per_query"], 0),
            ]
        )
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def _resource_table(rows: list[dict[str, Any]]) -> list[str]:
    header = [
        "Method",
        "Builds",
        "Heap Δ steady MB",
        "Heap Δ before cleanup MB",
        "RSS after search MB",
        "RSS file-backed MB",
        "cgroup peak MB",
        "Originals MB",
        "Quantized copy MB",
        "HNSW MB",
        "WAL MB",
        "Major faults (search)",
    ]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for r in rows:
        faults = (
            None
            if r.get("majflt_post_search") is None or r.get("majflt_post_index") is None
            else r["majflt_post_search"] - r["majflt_post_index"]
        )
        lines.append(
            "| "
            + " | ".join(
                [
                    r["config"],
                    f"{r['complete_builds']}/{r['builds']}",
                    _mb(r.get("heap_post_search_bytes")),
                    _mb(r.get("heap_post_index_bytes")),
                    _mb(r.get("rss_post_search_bytes")),
                    _mb(r.get("rss_file_post_search_bytes")),
                    _mb(r.get("cgroup_peak_post_search_bytes")),
                    _mb(r.get("storage_vectors_bytes")),
                    _mb(r.get("storage_quantized_bytes")),
                    _mb(r.get("storage_hnsw_bytes")),
                    _mb(r.get("storage_wal_bytes")),
                    _f(faults, 0),
                ]
            )
            + " |"
        )
    return lines


def _selected_table(entry: dict[str, Any]) -> list[str]:
    stage = entry["stages"]["selected"]
    choices = (stage.get("selection") or {}).get("choices") or {}
    header = [
        "Method",
        "k",
        "Frozen point",
        "Dev recall",
        "Target met",
        "Test recall [95% CI]",
        "Test ΔnDCG@10 [95% CI]",
        "Server p95 ms",
    ]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for g in stage["groups"]:
        dev = next(
            (
                c
                for c in choices.get(g["config"], [])
                if c["k"] == g["k"] and c["hnsw_ef"] == g["hnsw_ef"]
            ),
            {},
        )
        lines.append(
            "| "
            + " | ".join(
                [
                    g["config"],
                    str(g["k"]),
                    _point(g),
                    _f(dev.get("mean_ann_recall")),
                    "yes" if dev.get("target_met") else ("no" if dev else "n/a"),
                    _f(g["recall"]) + _ci(g["recall_ci95"]),
                    f"{g['delta_ndcg_at_10']:+.3f}" + _ci(g["delta_ndcg_ci95"]),
                    _f(g["server_ms"]["p95"], 2),
                ]
            )
            + " |"
        )
    return lines


def _cold_table(groups: list[dict[str, Any]]) -> list[str]:
    header = [
        "Method",
        "k",
        f"First {COLD_HEAD} requests p50 / p95 ms",
        "Remaining requests p50 / p95 ms",
        f"Vector reads / query (first {COLD_HEAD})",
        "ANN recall",
    ]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for g in groups:
        head, tail = g.get("cold_head_server_ms", {}), g.get("warm_tail_server_ms", {})
        lines.append(
            "| "
            + " | ".join(
                [
                    g["config"],
                    str(g["k"]),
                    f"{_f(head.get('p50'), 2)} / {_f(head.get('p95'), 2)}",
                    f"{_f(tail.get('p50'), 2)} / {_f(tail.get('p95'), 2)}",
                    _f(g.get("cold_head_vector_io_read_per_query"), 0),
                    _f(g["recall"]),
                ]
            )
            + " |"
        )
    return lines


def render_markdown(
    reports: dict[str, dict[str, Any]], macro: list[dict[str, Any]]
) -> str:
    out = ["# Benchmark report", ""]
    out.append(
        "Quality is the mean over queries of per-query means across repeats and "
        "builds; intervals are a 2,000-sample bootstrap over query IDs. Latency "
        "percentiles pool all measured requests. CPU and vector reads are Qdrant's "
        "own hardware units per query. Memory columns are explained in docs/results.md."
    )
    out.append("")
    for name, entry in reports.items():
        out += [f"## {name}", ""]
        for stage, report in entry["stages"].items():
            status = report["status"]
            if stage == "selected-full" or stage == "primary-full":
                out.append("Full official test split (extended run).")
                out.append("")
            out.append(
                f"### {stage} — {status}"
                + (f" ({report['error_type']})" if report.get("error_type") else "")
                + f", Qdrant {report.get('server_version')}, commit {str(report.get('git_revision'))[:10]}"
            )
            out.append("")
            for failure in report.get("method_failures") or []:
                out.append(
                    f"Failure, build {failure['build']}, {failure['method']}: "
                    f"{failure['error_type']} ({failure['message'][:160]})"
                )
            if report.get("method_failures"):
                out.append("")
            if not report["groups"]:
                out += ["No measured requests.", ""]
                continue
            if stage == "selected":
                out += _selected_table(entry)
            elif stage == "cold-start":
                out += _cold_table(report["groups"])
            else:
                out += _quality_table(report["groups"], show_point=stage == "ablation")
            out.append("")
            if report["resources"]:
                out += _resource_table(report["resources"])
                out.append("")
    if macro:
        out += ["## Macro-average over datasets (secondary summary, primary stage)", ""]
        out += [
            "| Model | Method | k | Datasets | Mean recall | Mean ΔnDCG@10 | Mean server p95 ms |",
            "|---|---|---|---|---|---|---|",
        ]
        for r in macro:
            out.append(
                f"| {r['model']} | {r['config']} | {r['k']} | {', '.join(r['datasets'])} | "
                f"{r['recall']:.3f} | {r['delta_ndcg_at_10']:+.3f} | {r['server_p95_ms']:.2f} |"
            )
        out.append("")
    return "\n".join(out)


def tuning_grid(results: Path, output: Path) -> list[str]:
    """CSV per workload of the dev sweep: recall and p95 by method, hnsw_ef, and oversampling."""
    written = []
    for workload in sorted(p for p in results.iterdir() if p.is_dir()):
        summary = workload / "tuning" / "summary.json"
        if (
            workload.name.startswith(("environment", "pipeline-test", "smoke"))
            or not summary.is_file()
        ):
            continue
        groups = json.loads(summary.read_text())["groups"]
        lines = [
            "method,k,hnsw_ef,rescore,oversampling,recall,ndcg_delta,server_p95_ms,server_p50_ms"
        ]
        for g in sorted(
            groups,
            key=lambda g: (
                g["config"],
                g["k"],
                g["hnsw_ef"],
                str(g["rescore"]),
                g["oversampling"] or 0,
            ),
        ):
            lines.append(
                ",".join(
                    [
                        g["config"],
                        str(g["k"]),
                        str(g["hnsw_ef"]),
                        "" if g["rescore"] is None else str(g["rescore"]).lower(),
                        "" if g["oversampling"] is None else f"{g['oversampling']:g}",
                        f"{g['mean_ann_recall']:.4f}",
                        f"{g['mean_ndcg_at_10'] - g['mean_exact_ndcg_at_10']:+.4f}",
                        f"{g['server_ms_p95']:.3f}",
                        f"{g['server_ms_p50']:.3f}",
                    ]
                )
            )
        path = output / f"tuning-grid-{workload.name}.csv"
        path.write_text("\n".join(lines) + "\n")
        written.append(str(path))
    return written


def build_report(results: Path, output: Path) -> dict[str, Any]:
    reports = workload_reports(results)
    macro = macro_average(reports)
    output.mkdir(parents=True, exist_ok=True)
    (output / "report.json").write_text(
        json.dumps(
            {"workloads": reports, "macro_average": macro}, indent=2, allow_nan=False
        )
    )
    (output / "report.md").write_text(render_markdown(reports, macro))
    return {
        "workloads": {
            name: {stage: r["status"] for stage, r in entry["stages"].items()}
            for name, entry in reports.items()
        },
        "markdown": str(output / "report.md"),
        "json": str(output / "report.json"),
        "tuning_grids": tuning_grid(results, output),
    }

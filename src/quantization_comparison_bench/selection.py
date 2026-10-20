"""Development-split selection rule.

For each method and result limit, choose the fastest search point whose ANN
Recall@k is at least the floor and whose mean nDCG@10 is within the tolerance of
exact float32 relevance. Selection reads development-split summaries only; the
frozen output is then evaluated once on held-out test queries.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .config import Benchmark, Cell, Method

LATENCY_METRICS = ("server_ms_p95", "server_ms_p50", "request_ms_p95", "request_ms_p50")


def _point(group: dict[str, Any]) -> tuple[int, int, bool | None, float | None]:
    return (
        int(group["k"]),
        int(group["hnsw_ef"]),
        group.get("rescore"),
        None if group.get("oversampling") is None else float(group["oversampling"]),
    )


def pool_builds(groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Average quality across builds and keep the slowest build's latency."""
    pooled: dict[tuple[str, tuple[Any, ...]], list[dict[str, Any]]] = defaultdict(list)
    for group in groups:
        pooled[(str(group["config"]), _point(group))].append(group)
    rows = []
    for (config, point), members in pooled.items():
        k, hnsw_ef, rescore, oversampling = point
        row: dict[str, Any] = {
            "config": config,
            "k": k,
            "hnsw_ef": hnsw_ef,
            "rescore": rescore,
            "oversampling": oversampling,
            "builds": len(members),
            "queries": min(int(member["queries"]) for member in members),
        }
        for metric in ("mean_ann_recall", "mean_ndcg_at_10", "mean_exact_ndcg_at_10"):
            row[metric] = float(np.mean([float(member[metric]) for member in members]))
        for metric in LATENCY_METRICS:
            values = [float(member[metric]) for member in members if metric in member]
            if values:
                row[metric] = max(values)
        rows.append(row)
    return rows


def select_cells(
    groups: list[dict[str, Any]],
    methods: list[Method],
    ks: list[int],
    recall_floor: float = 0.95,
    ndcg_tolerance: float = 0.01,
    latency_metric: str = "server_ms_p95",
) -> dict[str, Any]:
    if latency_metric not in LATENCY_METRICS:
        raise ValueError(f"latency_metric must be one of {LATENCY_METRICS}")
    if any(group.get("split") != "dev" for group in groups):
        raise ValueError("Selection must read development-split results only")
    rows = pool_builds(groups)
    by_method: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_method[row["config"]].append(row)
    result: dict[str, Any] = {}
    for method in methods:
        candidates = by_method.get(method.name)
        if not candidates:
            raise ValueError(f"No development results for method {method.name}")
        chosen: list[dict[str, Any]] = []
        for k in ks:
            at_k = [row for row in candidates if row["k"] == k]
            if not at_k:
                raise ValueError(f"No development results for {method.name} at k={k}")
            if any(latency_metric not in row for row in at_k):
                raise ValueError(f"{latency_metric} missing for {method.name} k={k}")
            eligible = [
                row
                for row in at_k
                if row["mean_ann_recall"] >= recall_floor
                and row["mean_ndcg_at_10"]
                >= row["mean_exact_ndcg_at_10"] - ndcg_tolerance
            ]
            if eligible:
                best = min(eligible, key=lambda row: row[latency_metric])
                target_met = True
            else:
                # Report the miss and still evaluate the best-quality point on test.
                best = max(
                    at_k,
                    key=lambda row: (row["mean_ann_recall"], -row[latency_metric]),
                )
                target_met = False
            chosen.append({**best, "target_met": target_met, "eligible": len(eligible)})
        result[method.name] = chosen
    return result


def selected_config(
    base: Benchmark,
    tuning_manifest: dict[str, Any],
    choices: dict[str, Any],
    rule: dict[str, Any],
) -> Benchmark:
    """Freeze the chosen points into a test-split copy of the base configuration."""
    dataset = tuning_manifest["data_manifest"]["dataset"]
    model = tuning_manifest["model_label"]
    methods = []
    for method in base.methods:
        cells = [
            Cell(
                k=row["k"],
                hnsw_ef=row["hnsw_ef"],
                rescore=row["rescore"],
                oversampling=row["oversampling"],
            )
            for row in choices[method.name]
        ]
        methods.append(method.model_copy(update={"cells": cells}))
    search = base.search.model_copy(update={"split": "test"})
    return base.model_copy(
        update={
            "datasets": [dataset],
            "models": [model],
            "methods": methods,
            "search": search,
            "selection": {
                "rule": rule,
                "tuning_run_id": tuning_manifest["run_id"],
                "tuning_git_revision": tuning_manifest.get("git_revision"),
                "tuning_artifact_checksums": tuning_manifest["artifact_checksums"],
                "choices": {
                    name: [
                        {
                            key: row[key]
                            for key in (
                                "k",
                                "hnsw_ef",
                                "rescore",
                                "oversampling",
                                "target_met",
                                "eligible",
                                "mean_ann_recall",
                                "mean_ndcg_at_10",
                                "mean_exact_ndcg_at_10",
                                rule["latency_metric"],
                            )
                        }
                        for row in rows
                    ]
                    for name, rows in choices.items()
                },
            },
        }
    )


def select_from_run(
    tuning_output: Path,
    base: Benchmark,
    recall_floor: float,
    ndcg_tolerance: float,
    latency_metric: str,
) -> Benchmark:
    manifest = json.loads((tuning_output / "manifest.json").read_text())
    if manifest.get("status") != "complete":
        raise ValueError("Select only from a complete tuning run")
    if manifest["configuration"]["search"]["split"] != "dev":
        raise ValueError("Tuning run must use search.split: dev")
    summary = json.loads((tuning_output / "summary.json").read_text())
    rule = {
        "recall_floor": recall_floor,
        "ndcg_tolerance": ndcg_tolerance,
        "latency_metric": latency_metric,
        "selected_on": "dev",
    }
    choices = select_cells(
        summary["groups"],
        base.methods,
        base.search.ks,
        recall_floor,
        ndcg_tolerance,
        latency_metric,
    )
    return selected_config(base, manifest, choices, rule)

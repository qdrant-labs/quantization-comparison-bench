import asyncio
import hashlib
import json
import os
import platform
import random
import subprocess
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import httpx
import numpy as np
import yaml

from .config import Benchmark, Method, collection_config
from .evaluation import (
    EmbeddingData,
    exact_ground_truth,
    load_embeddings,
    quality_metric,
    summarize_records,
    validate_embeddings,
)
from .telemetry import (
    collection_storage,
    host_memory,
    resource_delta,
    sample_resources,
    validate_resources,
)


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def artifact_signature(path: Path) -> dict[str, str]:
    hashes = {}
    for name in (
        "corpus.npy",
        "queries.npy",
        "corpus_ids.json",
        "query_ids.json",
        "query_splits.json",
        "qrels.json",
    ):
        with (path / name).open("rb") as stream:
            hashes[name] = hashlib.file_digest(stream, "sha256").hexdigest()
    return hashes


def save_ground_truth(
    data_path: Path, k: int = 100, selected_query_ids: list[str] | None = None
) -> dict[str, list[str]]:
    data = load_embeddings(data_path)
    validate_embeddings(data)
    signatures = artifact_signature(data_path)
    if signatures != data.manifest["checksums"]:
        raise ValueError(
            "Embedding artifacts do not match their saved manifest checksums"
        )
    result = exact_ground_truth(data, k=k, selected_query_ids=selected_query_ids)
    write_json(
        data_path / "ground_truth.json",
        {
            "k": k,
            "artifacts": signatures,
            "rankings": result,
            "exact_ndcg_at_10": {
                qid: quality_metric(ids, ids, data.qrels[qid], k, query_id=qid)[1]
                for qid, ids in result.items()
            },
        },
    )
    return result


def cached_ground_truth(
    data_path: Path, k: int, selected_query_ids: list[str] | None = None
) -> dict[str, list[str]]:
    path = data_path / "ground_truth.json"
    if not path.exists():
        return save_ground_truth(data_path, k, selected_query_ids)
    cached = json.loads(path.read_text())
    if cached["artifacts"] != artifact_signature(data_path):
        raise ValueError(
            "Ground truth does not match embedding artifacts; regenerate it"
        )
    desired = selected_query_ids or json.loads(
        (data_path / "query_ids.json").read_text()
    )
    if cached["k"] < k:
        return save_ground_truth(data_path, k, desired)
    missing = sorted(set(desired) - set(cached["rankings"]))
    if missing:
        data = load_embeddings(data_path)
        additions = exact_ground_truth(data, k=cached["k"], selected_query_ids=missing)
        cached["rankings"].update(additions)
        cached["exact_ndcg_at_10"].update(
            {
                qid: quality_metric(
                    ids, ids, data.qrels[qid], cached["k"], query_id=qid
                )[1]
                for qid, ids in additions.items()
            }
        )
        write_json(path, cached)
    return cached["rankings"]


def connect() -> httpx.AsyncClient:
    headers = {"api-key": key} if (key := os.environ.get("QDRANT_API_KEY")) else {}
    return httpx.AsyncClient(
        base_url=os.environ.get("QDRANT_URL", "http://127.0.0.1:6333"),
        headers=headers,
        timeout=120,
    )


async def request(
    client: httpx.AsyncClient, verb: str, path: str, body: dict[str, Any] | None = None
) -> dict[str, Any]:
    response = await client.request(verb, path, json=body)
    if response.is_error:
        raise RuntimeError(
            f"Qdrant {verb} {path}: HTTP {response.status_code}: {response.text[:1000]}"
        )
    return response.json()


async def doctor() -> dict[str, Any]:
    async with connect() as client:
        root = await request(client, "GET", "/")
        response = await client.get("/healthz")
        response.raise_for_status()
        return {
            "reachable": True,
            "server": root,
            "host_platform": platform.platform(),
            "cpu_count": os.cpu_count(),
        }


async def create_collection(
    client: httpx.AsyncClient,
    name: str,
    data: EmbeddingData,
    config: Benchmark,
    method: Method,
    output: Path,
) -> None:
    body = collection_config(config, method, data.corpus.shape[1])
    write_json(output / "create-request.json", body)
    for start in range(0, len(data.corpus_ids), config.upload_batch_size):
        end = min(start + config.upload_batch_size, len(data.corpus_ids))
        await request(
            client,
            "PUT",
            f"/collections/{name}/points?wait=true",
            {
                "batch": {
                    "ids": list(range(start, end)),
                    "vectors": data.corpus[start:end].tolist(),
                },
            },
        )
    deadline = time.monotonic() + config.indexing_timeout_seconds
    while time.monotonic() < deadline:
        state = (await request(client, "GET", f"/collections/{name}"))["result"]
        if state["optimizer_status"] != "ok":
            raise RuntimeError(f"Qdrant optimizer failed: {state['optimizer_status']}")
        if (
            state["status"] == "green"
            and state["points_count"] == len(data.corpus_ids)
            and state["indexed_vectors_count"] >= len(data.corpus_ids)
        ):
            write_json(output / "collection.json", state)
            return
        await asyncio.sleep(1)
    raise TimeoutError(f"Collection {name} did not finish HNSW indexing")


async def query(
    client: httpx.AsyncClient,
    collection: str,
    data: EmbeddingData,
    row: int,
    k: int,
    hnsw_ef: int,
    rescore: bool | None,
    oversampling: float | None,
    self_match: int | None,
) -> tuple[list[str], float, float]:
    params: dict[str, Any] = {"hnsw_ef": hnsw_ef, "exact": False}
    if rescore is not None:
        params["quantization"] = {
            "ignore": False,
            "rescore": rescore,
            "oversampling": oversampling,
        }
    body: dict[str, Any] = {
        "query": data.queries[row].tolist(),
        "limit": k,
        "with_payload": False,
        "with_vector": False,
        "params": params,
    }
    if self_match is not None:
        body["filter"] = {"must_not": [{"has_id": [self_match]}]}
    started = time.perf_counter()
    response = await request(
        client, "POST", f"/collections/{collection}/points/query", body
    )
    elapsed_ms = (time.perf_counter() - started) * 1000
    points = response["result"]["points"]
    ids = [data.corpus_ids[int(point["id"])] for point in points]
    if len(ids) != k:
        raise ValueError(f"Expected {k} results, received {len(ids)}")
    return ids, elapsed_ms, float(response["time"]) * 1000


async def measure_cell(
    client: httpx.AsyncClient,
    collection: str,
    data: EmbeddingData,
    ground_truth: dict[str, list[str]],
    indices: list[int],
    config: Benchmark,
    method: Method,
    build: int,
    k: int,
    rescore: bool | None,
    oversampling: float | None,
    hnsw_ef: int,
    output: Path,
    directory: Path,
    phase: dict[str, Any],
) -> None:
    corpus_positions = {doc_id: index for index, doc_id in enumerate(data.corpus_ids)}
    phase.update(
        stage="warmup",
        k=k,
        rescore=rescore,
        oversampling=oversampling,
        hnsw_ef=hnsw_ef,
    )
    warmup_count = (
        max(config.search.warmup_queries, len(indices))
        if config.search.cache_state == "warm"
        else 0
    )
    for step in range(warmup_count):
        row = indices[step % len(indices)]
        await query(
            client,
            collection,
            data,
            row,
            k,
            hnsw_ef,
            rescore,
            oversampling,
            corpus_positions.get(data.query_ids[row]),
        )
    with (output / "queries.jsonl").open("a") as stream:
        for repeat in range(config.search.repeats):
            phase.update(stage="cache_reset", repeat=repeat)
            if config.search.cache_state == "cold_start":
                await reset_cache(config, directory, phase)
            phase["stage"] = "measurement"
            before = host_memory()
            order = indices.copy()
            random.Random(config.seed + repeat).shuffle(order)
            for row in order:
                query_id = data.query_ids[row]
                ids, client_ms, server_ms = await query(
                    client,
                    collection,
                    data,
                    row,
                    k,
                    hnsw_ef,
                    rescore,
                    oversampling,
                    corpus_positions.get(query_id),
                )
                recall, ndcg = quality_metric(
                    ids,
                    ground_truth[query_id],
                    data.qrels[query_id],
                    k,
                    query_id=query_id,
                )
                _, exact_ndcg = quality_metric(
                    ground_truth[query_id],
                    ground_truth[query_id],
                    data.qrels[query_id],
                    k,
                    query_id=query_id,
                )
                record = {
                    "dataset": data.manifest.get("dataset", "unknown"),
                    "model": data.manifest.get("model", "unknown"),
                    "config": method.name,
                    "build": build,
                    "repeat": repeat,
                    "split": config.search.split,
                    "k": k,
                    "rescore": rescore,
                    "oversampling": oversampling,
                    "cache_state": config.search.cache_state,
                    "hnsw_ef": hnsw_ef,
                    "candidate_limit": int(np.ceil(k * (oversampling or 1))),
                    "query_id": query_id,
                    "result_ids": ids,
                    "ann_recall": recall,
                    "ndcg_at_10": ndcg,
                    "exact_ndcg_at_10": exact_ndcg,
                    "delta_ndcg_at_10": ndcg - exact_ndcg,
                    "unix_seconds": time.time(),
                    "request_ms": client_ms,
                    "server_ms": server_ms,
                }
                stream.write(json.dumps(record, allow_nan=False) + "\n")
            stream.flush()
            after = host_memory()
            with (directory / "measurement-intervals.jsonl").open("a") as intervals:
                intervals.write(
                    json.dumps(
                        {
                            **phase,
                            "before": before,
                            "after": after,
                            "delta": resource_delta(before, after),
                        }
                    )
                    + "\n"
                )


async def reset_cache(
    config: Benchmark, directory: Path, phase: dict[str, Any]
) -> None:
    command = config.search.cache_reset_command
    if command is None:
        raise ValueError("A cold-start run requires a cache reset command")
    started = time.time()
    process = await asyncio.create_subprocess_exec(
        *command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    try:
        stdout, stderr = await asyncio.wait_for(
            process.communicate(), timeout=config.search.cache_reset_timeout_seconds
        )
    except TimeoutError:
        process.kill()
        await process.communicate()
        raise
    with (directory / "cache-resets.jsonl").open("a") as stream:
        stream.write(
            json.dumps(
                {
                    **phase,
                    "started_at": started,
                    "finished_at": time.time(),
                    "command": command,
                    "returncode": process.returncode,
                    "stdout": stdout.decode(errors="replace"),
                    "stderr": stderr.decode(errors="replace"),
                }
            )
            + "\n"
        )
    if process.returncode:
        raise RuntimeError(f"Cache reset command exited with {process.returncode}")


async def measure_method(
    client: httpx.AsyncClient,
    data: EmbeddingData,
    ground_truth: dict[str, list[str]],
    indices: list[int],
    config: Benchmark,
    method: Method,
    build: int,
    run_id: str,
    output: Path,
) -> None:
    collection = f"qb_{run_id}_{build}_{method.name}"
    directory = output / f"build-{build}" / method.name
    directory.mkdir(parents=True)
    stop = asyncio.Event()
    phase: dict[str, Any] = {"stage": "build", "method": method.name, "build": build}
    sampler = asyncio.create_task(
        sample_resources(directory / "resources.jsonl", stop, phase)
    )
    created = False
    failure: Exception | None = None
    try:
        # A unique run ID makes cleanup independent of any pre-existing collection.
        await request(
            client,
            "PUT",
            f"/collections/{collection}",
            collection_config(config, method, data.corpus.shape[1]),
        )
        created = True
        await create_collection(client, collection, data, config, method, directory)
        write_json(directory / "storage.json", collection_storage(collection))
        write_json(directory / "memory-post-index.json", host_memory())
        metrics = await client.get("/metrics")
        metrics.raise_for_status()
        (directory / "metrics-before.prom").write_text(metrics.text)
        grid = config.grid(method)
        random.Random(config.seed).shuffle(grid)
        for k, rescore, oversampling, hnsw_ef in grid:
            print(
                f"build={build} method={method.name} k={k} rescore={rescore} "
                f"oversampling={oversampling} hnsw_ef={hnsw_ef}",
                flush=True,
            )
            await measure_cell(
                client,
                collection,
                data,
                ground_truth,
                indices,
                config,
                method,
                build,
                k,
                rescore,
                oversampling,
                hnsw_ef,
                output,
                directory,
                phase,
            )
        metrics = await client.get("/metrics")
        metrics.raise_for_status()
        (directory / "metrics-after.prom").write_text(metrics.text)
    except Exception as error:
        failure = error
        write_json(
            directory / "failure.json",
            {
                "phase": phase.copy(),
                "error_type": type(error).__name__,
                "message": str(error),
                "unix_seconds": time.time(),
            },
        )
        raise
    finally:
        phase["stage"] = "cleanup"
        stop.set()
        cleanup_errors: list[str] = []
        try:
            await sampler
        except Exception as error:
            cleanup_errors.append(f"Resource sampler: {type(error).__name__}: {error}")
        if created:
            try:
                await request(client, "DELETE", f"/collections/{collection}")
            except Exception as error:
                cleanup_errors.append(
                    f"Collection cleanup: {type(error).__name__}: {error}"
                )
        if cleanup_errors:
            write_json(
                directory / "cleanup-errors.json",
                {
                    "collection": collection,
                    "errors": cleanup_errors,
                },
            )
            if failure is None:
                raise RuntimeError("; ".join(cleanup_errors))


def summarize(output: Path) -> dict[str, Any]:
    records = [
        json.loads(line)
        for line in (output / "queries.jsonl").read_text().splitlines()
        if line
    ]
    result = summarize_records(records)
    write_json(output / "summary.json", result)
    return result


def freeze_configuration(
    tuning_output: Path,
    output: Path,
    min_recall: float = 0.95,
    max_ndcg_drop: float = 0.01,
    points_per_method: int = 1,
) -> dict[str, Any]:
    """Select fast development operating points and freeze held-out search settings."""
    if not 0 <= min_recall <= 1 or max_ndcg_drop < 0 or points_per_method not in {1, 2}:
        raise ValueError(
            "Require recall in [0, 1], nonnegative nDCG drop, and 1–2 points"
        )
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite frozen configuration: {output}")
    manifest_path = tuning_output / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    config = Benchmark.model_validate(manifest["configuration"])
    if manifest["status"] != "complete" or config.search.split != "dev":
        raise ValueError("Freeze requires a complete development run")
    if config.search.cache_state != "warm":
        raise ValueError("Select quality/latency operating points from warm dev runs")
    groups = summarize(tuning_output)["groups"]
    selected: dict[str, list[dict[str, Any]]] = {}
    report: dict[str, Any] = {
        "minimum_ann_recall": min_recall,
        "maximum_ndcg_drop_from_exact": max_ndcg_drop,
        "selection_metric": "lowest mean client p50 across development builds",
        "points_per_method_and_k": points_per_method,
        "selections": selected,
    }
    for method in config.methods:
        selected[method.name] = []
        cells: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
        for row in groups:
            if row["config"] == method.name:
                cells[
                    row["k"], row["rescore"], row["oversampling"], row["hnsw_ef"]
                ].append(row)
        for k in config.search.ks:
            candidates: list[tuple[float, tuple[Any, ...], dict[str, float]]] = []
            for key, rows in cells.items():
                if key[0] != k or len(rows) != config.builds:
                    continue
                means = {
                    metric: float(np.mean([row[metric] for row in rows]))
                    for metric in (
                        "mean_ann_recall",
                        "mean_delta_ndcg_at_10",
                        "request_ms_p50",
                    )
                }
                if (
                    means["mean_ann_recall"] >= min_recall
                    and means["mean_delta_ndcg_at_10"] >= -max_ndcg_drop
                ):
                    candidates.append((means["request_ms_p50"], key, means))
            if not candidates:
                raise ValueError(
                    f"No development operating point qualifies for {method.name}, k={k}. "
                    "Extend the development sweep or preregister a revised target; "
                    "do not use held-out results to tune."
                )
            candidates.sort(key=lambda candidate: (candidate[0], candidate[1][3]))
            for _, (_, rescore, oversampling, ef), means in candidates[
                :points_per_method
            ]:
                selected[method.name].append(
                    {
                        "k": k,
                        "rescore": rescore,
                        "oversampling": oversampling,
                        "hnsw_ef": ef,
                        "development_metrics": means,
                    }
                )
    frozen = config.model_dump()
    frozen["builds"] = 3
    frozen["datasets"] = [manifest["data_manifest"]["dataset"]]
    frozen["models"] = [_model_label(manifest["data_manifest"]["model"])]
    frozen["search"].update(
        split="test",
        repeats=5,
        max_queries=150,
        hnsw_efs=None,
    )
    frozen["tuning_provenance"] = {
        **report,
        "run_id": manifest["run_id"],
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "artifact_checksums": manifest["artifact_checksums"],
        "development_query_ids": manifest["query_ids"],
    }
    for method in frozen["methods"]:
        method["search_settings"] = [
            {key: value for key, value in point.items() if key != "development_metrics"}
            for point in selected[method["name"]]
        ]
    Benchmark.model_validate(frozen)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(frozen, sort_keys=False))
    return report


def _model_label(model: str) -> str | None:
    return {
        "embed-english-v3.0": "cohere",
        "BAAI/bge-small-en-v1.5": "bge",
        "synthetic-normal": "synthetic",
    }.get(model)


async def run_benchmark(config: Benchmark, data_path: Path, output: Path) -> None:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(
            f"Use a new output directory; refusing to mix runs in {output}"
        )
    data = load_embeddings(data_path)
    validate_embeddings(data)
    if artifact_signature(data_path) != data.manifest["checksums"]:
        raise ValueError(
            "Embedding artifacts do not match their saved manifest checksums"
        )
    model_label = _model_label(data.manifest["model"])
    if (
        data.manifest["dataset"] not in config.datasets
        or model_label not in config.models
    ):
        raise ValueError(
            "Data artifact is not one of the datasets/models declared in the config"
        )
    initial_resources = validate_resources(
        data.corpus.nbytes, config.memory_pressure, config.require_telemetry
    )
    if config.search.cache_state == "cold_start" and not (
        os.environ.get("QDRANT_PID") and os.environ.get("QDRANT_CGROUP_PATH")
    ):
        raise ValueError(
            "Cold-start measurement requires process and cgroup I/O telemetry"
        )
    maximum_k = max(
        cell[0] for method in config.methods for cell in config.grid(method)
    )
    if maximum_k >= len(data.corpus_ids):
        raise ValueError("Corpus must have more points than the largest k")
    indices = [
        index
        for index, query_id in enumerate(data.query_ids)
        if data.query_splits[query_id] == config.search.split
    ]
    random.Random(config.seed).shuffle(indices)
    if config.search.max_queries:
        indices = indices[: config.search.max_queries]
    if not indices:
        raise ValueError(f"No queries for split {config.search.split}")
    if config.tuning_provenance:
        if config.tuning_provenance["artifact_checksums"] != artifact_signature(
            data_path
        ):
            raise ValueError("Frozen settings belong to different embedding artifacts")
        if config.search.split != "test":
            raise ValueError("Frozen development selections must run on the test split")
        dev_ids = set(config.tuning_provenance["development_query_ids"])
        if any(data.query_ids[index] in dev_ids for index in indices):
            raise ValueError("Held-out query IDs overlap the development selection")
    ground_truth = cached_ground_truth(
        data_path, maximum_k, [data.query_ids[index] for index in indices]
    )
    output.mkdir(parents=True, exist_ok=True)
    run_id = uuid4().hex[:12]
    async with connect() as client:
        root = await request(client, "GET", "/")
        if root["version"] != config.expected_qdrant_version:
            raise ValueError(
                f"Expected Qdrant {config.expected_qdrant_version}, received {root['version']}"
            )
        active = (await request(client, "GET", "/collections"))["result"]["collections"]
        if active:
            raise ValueError(
                "Use a dedicated empty Qdrant server; existing collections consume resources"
            )
        revision = await asyncio.to_thread(
            subprocess.run,
            ["git", "-c", f"safe.directory={Path.cwd()}", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            cwd=Path(__file__).resolve().parents[2],
        )
        manifest = {
            "status": "running",
            "run_id": run_id,
            "started_at": datetime.now(UTC).isoformat(),
            "git_revision": revision.stdout.strip() or None,
            "runner_source_checksums": {
                path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                for path in sorted(Path(__file__).parent.glob("*.py"))
            },
            "server": root,
            "configuration": config.model_dump(),
            "data_manifest": data.manifest,
            "artifact_checksums": artifact_signature(data_path),
            "query_ids": [data.query_ids[index] for index in indices],
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
            "cache_state": config.search.cache_state,
            "query_scope": (
                "complete selected split"
                if len(indices)
                == sum(
                    value == config.search.split for value in data.query_splits.values()
                )
                else "seeded subset of selected split"
            ),
            "original_corpus_bytes": data.corpus.nbytes,
            "initial_resources": initial_resources,
            "source_manifest_sha256": hashlib.sha256(
                (data_path / "manifest.json").read_bytes()
            ).hexdigest(),
            "lockfile_sha256": (
                hashlib.sha256(Path("uv.lock").read_bytes()).hexdigest()
                if Path("uv.lock").is_file()
                else None
            ),
            "cold_start_note": (
                "Cache reset precedes each measured pass; the pass warms with each query. "
                "Inspect cache-resets and measurement-intervals for actual reads/faults."
                if config.search.cache_state == "cold_start"
                else None
            ),
            "concurrency": 1,
            "telemetry": {
                name: bool(os.environ.get(name))
                for name in ("QDRANT_PID", "QDRANT_CGROUP_PATH", "QDRANT_STORAGE_PATH")
            },
        }
        write_json(output / "manifest.json", manifest)
        try:
            for build in range(config.builds):
                methods = config.methods.copy()
                random.Random(config.seed + build).shuffle(methods)
                for method in methods:
                    await measure_method(
                        client,
                        data,
                        ground_truth,
                        indices,
                        config,
                        method,
                        build,
                        run_id,
                        output,
                    )
            summarize(output)
            manifest["status"] = "complete"
        except Exception as error:
            manifest["status"] = "failed"
            manifest["error_type"] = type(error).__name__
            manifest["error_message"] = str(error)
            raise
        finally:
            manifest["finished_at"] = datetime.now(UTC).isoformat()
            write_json(output / "manifest.json", manifest)

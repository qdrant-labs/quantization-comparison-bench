import asyncio
import json
import os
import platform
import random
import shlex
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import httpx

from .config import Benchmark, Cell, Method, collection_config
from .evaluation import (
    MODEL_LABELS,
    EmbeddingData,
    artifact_signature,
    exact_ground_truth,
    load_embeddings,
    quality_metric,
    summarize_records,
    validate_embeddings,
)
from .telemetry import (
    check_telemetry,
    collection_storage,
    host_memory,
    refresh_targets,
    sample_resources,
)


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def save_ground_truth(data_path: Path, k: int = 100) -> dict[str, list[str]]:
    data = load_embeddings(data_path)
    validate_embeddings(data)
    signatures = artifact_signature(data_path)
    if signatures != data.manifest["checksums"]:
        raise ValueError(
            "Embedding artifacts do not match their saved manifest checksums"
        )
    result = exact_ground_truth(data, k=k)
    write_json(
        data_path / "ground_truth.json",
        {
            "k": k,
            "artifacts": signatures,
            "rankings": result,
            "exact_ndcg_at_10": {
                qid: quality_metric(ids, ids, data.qrels[qid], k)[1]
                for qid, ids in result.items()
            },
        },
    )
    return result


def cached_ground_truth(data_path: Path, k: int) -> dict[str, list[str]]:
    path = data_path / "ground_truth.json"
    if not path.exists():
        return save_ground_truth(data_path, k)
    cached = json.loads(path.read_text())
    if cached["artifacts"] != artifact_signature(data_path):
        raise ValueError(
            "Ground truth does not match embedding artifacts; regenerate it"
        )
    if cached["k"] < k:
        return save_ground_truth(data_path, k)
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
            "telemetry": {
                name: bool(os.environ.get(name))
                for name in (
                    "QDRANT_CONTAINER_ID",
                    "QDRANT_PID",
                    "QDRANT_CGROUP_PATH",
                    "QDRANT_STORAGE_PATH",
                )
            },
        }


async def wait_for_collection(
    client: httpx.AsyncClient, name: str, points: int, deadline: float
) -> dict[str, Any]:
    while time.monotonic() < deadline:
        try:
            state = (await request(client, "GET", f"/collections/{name}"))["result"]
        except (httpx.HTTPError, RuntimeError):
            await asyncio.sleep(1)
            continue
        if state["optimizer_status"] != "ok":
            raise RuntimeError(f"Qdrant optimizer failed: {state['optimizer_status']}")
        if (
            state["status"] == "green"
            and state["points_count"] == points
            and state["indexed_vectors_count"] >= points
        ):
            return state
        await asyncio.sleep(1)
    raise TimeoutError(f"Collection {name} did not become ready")


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
    state = await wait_for_collection(
        client,
        name,
        len(data.corpus_ids),
        time.monotonic() + config.indexing_timeout_seconds,
    )
    write_json(output / "collection.json", state)


async def query(
    client: httpx.AsyncClient,
    collection: str,
    data: EmbeddingData,
    row: int,
    cell: Cell,
    self_match: int | None,
) -> tuple[list[str], float, float]:
    params: dict[str, Any] = {"hnsw_ef": cell.hnsw_ef, "exact": False}
    if cell.rescore is not None:
        params["quantization"] = {
            "ignore": False,
            "rescore": cell.rescore,
            "oversampling": cell.oversampling,
        }
    body: dict[str, Any] = {
        "query": data.queries[row].tolist(),
        "limit": cell.k,
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
    if len(ids) != cell.k:
        raise ValueError(f"Expected {cell.k} results, received {len(ids)}")
    return ids, elapsed_ms, float(response["time"]) * 1000


async def reset_cache(
    client: httpx.AsyncClient, collection: str, points: int, config: Benchmark
) -> dict[str, Any]:
    """Run the dedicated-VM cache-reset hook, then wait for the server."""
    command = shlex.split(config.cache_reset_command or "")
    started = time.monotonic()
    completed = await asyncio.to_thread(
        subprocess.run, command, capture_output=True, text=True, check=False
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"cache reset failed ({completed.returncode}): {completed.stderr[-2000:]}"
        )
    targets = await asyncio.to_thread(refresh_targets)
    await wait_for_collection(
        client, collection, points, time.monotonic() + config.indexing_timeout_seconds
    )
    return {
        "command": command,
        "seconds": time.monotonic() - started,
        "stdout": completed.stdout[-2000:],
        "targets": targets,
    }


async def measure_cell(
    client: httpx.AsyncClient,
    collection: str,
    data: EmbeddingData,
    ground_truth: dict[str, list[str]],
    indices: list[int],
    config: Benchmark,
    method: Method,
    build: int,
    cell: Cell,
    output: Path,
    directory: Path,
) -> None:
    corpus_positions = {doc_id: index for index, doc_id in enumerate(data.corpus_ids)}
    if config.cache_state == "warm":
        for step in range(max(config.search.warmup_queries, len(indices))):
            row = indices[step % len(indices)]
            await query(
                client,
                collection,
                data,
                row,
                cell,
                corpus_positions.get(data.query_ids[row]),
            )
    with (output / "queries.jsonl").open("a") as stream:
        for repeat in range(config.search.repeats):
            pass_record: dict[str, Any] = {
                "build": build,
                "config": method.name,
                "cell": cell.model_dump(),
                "pass": repeat,
                "cache_state": config.cache_state,
            }
            if config.cache_state == "cold_start":
                pass_record["reset"] = await reset_cache(
                    client, collection, len(data.corpus_ids), config
                )
            pass_record["before"] = host_memory()
            order = indices.copy()
            random.Random(config.seed + repeat).shuffle(order)
            for position, row in enumerate(order):
                query_id = data.query_ids[row]
                ids, client_ms, server_ms = await query(
                    client,
                    collection,
                    data,
                    row,
                    cell,
                    corpus_positions.get(query_id),
                )
                recall, ndcg = quality_metric(
                    ids, ground_truth[query_id], data.qrels[query_id], cell.k
                )
                _, exact_ndcg = quality_metric(
                    ground_truth[query_id],
                    ground_truth[query_id],
                    data.qrels[query_id],
                    cell.k,
                )
                record = {
                    "dataset": data.manifest.get("dataset", "unknown"),
                    "model": data.manifest.get("model", "unknown"),
                    "config": method.name,
                    "build": build,
                    "repeat": repeat,
                    "position": position,
                    "split": config.search.split,
                    "k": cell.k,
                    "rescore": cell.rescore,
                    "oversampling": cell.oversampling,
                    "cache_state": config.cache_state,
                    "hnsw_ef": cell.hnsw_ef,
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
            pass_record["after"] = host_memory()
            with (directory / "passes.jsonl").open("a") as passes:
                passes.write(json.dumps(pass_record) + "\n")


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
    sampler = asyncio.create_task(sample_resources(directory / "resources.jsonl", stop))
    created = False
    try:
        # A unique run ID makes cleanup independent of any pre-existing collection.
        write_json(directory / "memory-pre-index.json", host_memory())
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
        for cell in grid:
            print(
                f"build={build} method={method.name} k={cell.k} hnsw_ef={cell.hnsw_ef} "
                f"rescore={cell.rescore} oversampling={cell.oversampling}",
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
                cell,
                output,
                directory,
            )
        metrics = await client.get("/metrics")
        metrics.raise_for_status()
        (directory / "metrics-after.prom").write_text(metrics.text)
        write_json(directory / "memory-post-search.json", host_memory())
    finally:
        stop.set()
        try:
            await sampler
        finally:
            if created:
                await request(client, "DELETE", f"/collections/{collection}")


def summarize(output: Path) -> dict[str, Any]:
    records = [
        json.loads(line)
        for line in (output / "queries.jsonl").read_text().splitlines()
        if line
    ]
    result = summarize_records(records)
    write_json(output / "summary.json", result)
    return result


def select_queries(data: EmbeddingData, config: Benchmark) -> list[int]:
    """Seeded query subset from the configured split; stable for one artifact."""
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
    return indices


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
    model_label = MODEL_LABELS.get(data.manifest["model"])
    if (
        data.manifest["dataset"] not in config.datasets
        or model_label not in config.models
    ):
        raise ValueError(
            "Data artifact is not one of the datasets/models declared in the config"
        )
    telemetry: dict[str, Any] = {
        name: bool(os.environ.get(name))
        for name in ("QDRANT_PID", "QDRANT_CGROUP_PATH", "QDRANT_STORAGE_PATH")
    }
    if config.require_telemetry:
        telemetry["checked"] = check_telemetry(config.memory_limit_bytes)
    host_memory()
    if max(config.search.ks) >= len(data.corpus_ids):
        raise ValueError("Corpus must have more points than the largest k")
    indices = select_queries(data, config)
    ground_truth = cached_ground_truth(data_path, max(config.search.ks))
    output.mkdir(parents=True, exist_ok=True)
    run_id = uuid4().hex[:12]
    async with connect() as client:
        root = await request(client, "GET", "/")
        if root["version"] != config.expected_qdrant_version:
            raise ValueError(
                f"Expected Qdrant {config.expected_qdrant_version}, received {root['version']}"
            )
        revision = await asyncio.to_thread(
            subprocess.run,
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
        manifest = {
            "status": "running",
            "run_id": run_id,
            "started_at": datetime.now(UTC).isoformat(),
            "git_revision": revision.stdout.strip() or None,
            "server": root,
            "configuration": config.model_dump(),
            "model_label": model_label,
            "data_manifest": data.manifest,
            "artifact_checksums": artifact_signature(data_path),
            "query_ids": [data.query_ids[index] for index in indices],
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
            "cache_state": config.cache_state,
            "concurrency": 1,
            "telemetry": telemetry,
            "selection": config.selection,
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
            raise
        finally:
            manifest["finished_at"] = datetime.now(UTC).isoformat()
            write_json(output / "manifest.json", manifest)

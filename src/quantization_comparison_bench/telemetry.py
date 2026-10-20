import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any


def host_memory() -> dict[str, Any]:
    values: dict[str, Any] = {"unix_seconds": time.time()}
    if pid := os.environ.get("QDRANT_PID"):
        for line in (Path("/proc") / pid / "status").read_text().splitlines():
            key, _, value = line.partition(":")
            if key in {"VmRSS", "VmHWM", "RssAnon", "RssFile", "RssShmem"}:
                values[f"process_{key}_bytes"] = int(value.split()[0]) * 1024
        values["process_io"] = {
            key.rstrip(":"): int(value)
            for key, value in (
                line.split()
                for line in (Path("/proc") / pid / "io").read_text().splitlines()
            )
        }
        fields = (Path("/proc") / pid / "stat").read_text().rsplit(")", 1)[1].split()
        values["process_minor_faults"] = int(fields[7])
        values["process_major_faults"] = int(fields[9])
    if location := os.environ.get("QDRANT_CGROUP_PATH"):
        path = Path(location)
        for name in ("memory.current", "memory.peak", "memory.max"):
            value = (path / name).read_text().strip()
            values[name] = int(value) if value.isdigit() else value
        for name in ("memory.stat", "memory.events"):
            values[name] = {
                key: int(value)
                for key, value in (
                    line.split() for line in (path / name).read_text().splitlines()
                )
            }
        values["io.stat"] = (path / "io.stat").read_text()
    return values


def collection_storage(collection: str) -> dict[str, Any]:
    if not (root := os.environ.get("QDRANT_STORAGE_PATH")):
        return {
            "available": False,
            "reason": "Set QDRANT_STORAGE_PATH to the server storage directory on the VM",
        }
    path = Path(root) / "collections" / collection
    if not path.is_dir():
        raise FileNotFoundError(f"Server collection storage not found: {path}")
    sizes: dict[str, dict[str, int]] = {}
    for item in path.rglob("*"):
        if item.is_file() and not item.is_symlink():
            stat = item.stat()
            sizes[str(item.relative_to(path))] = {
                "logical_bytes": stat.st_size,
                "allocated_bytes": stat.st_blocks * 512,
            }
    return {
        "available": True,
        "logical_bytes": sum(value["logical_bytes"] for value in sizes.values()),
        "allocated_bytes": sum(value["allocated_bytes"] for value in sizes.values()),
        "files": sizes,
    }


async def sample_resources(
    output: Path, stop: asyncio.Event, phase: dict[str, Any] | None = None
) -> None:
    with output.open("a") as stream:
        while not stop.is_set():
            stream.write(json.dumps({**host_memory(), **(phase or {})}) + "\n")
            stream.flush()
            try:
                await asyncio.wait_for(stop.wait(), timeout=0.1)
            except TimeoutError:
                pass


def validate_resources(
    original_bytes: int, memory_pressure: bool, require_telemetry: bool
) -> dict[str, Any]:
    required = ("QDRANT_PID", "QDRANT_CGROUP_PATH", "QDRANT_STORAGE_PATH")
    if require_telemetry or memory_pressure:
        missing = [name for name in required if not os.environ.get(name)]
        if missing:
            raise ValueError(
                f"Required server telemetry is missing: {', '.join(missing)}"
            )
    snapshot = host_memory()
    if memory_pressure:
        limit = snapshot.get("memory.max")
        if not isinstance(limit, int) or limit >= original_bytes:
            raise ValueError(
                "memory_pressure requires a finite server cgroup memory.max "
                f"below the original float32 corpus ({original_bytes} bytes); got {limit}"
            )
    return snapshot


def resource_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key in ("process_minor_faults", "process_major_faults"):
        if key in before and key in after:
            result[key] = after[key] - before[key]
    for key in ("process_io", "memory.events"):
        if key in before and key in after:
            result[key] = {
                name: value - before[key].get(name, value)
                for name, value in after[key].items()
            }
    return result

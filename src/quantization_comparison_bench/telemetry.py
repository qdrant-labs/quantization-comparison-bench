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
        values["process_io"] = (Path("/proc") / pid / "io").read_text()
    if location := os.environ.get("QDRANT_CGROUP_PATH"):
        path = Path(location)
        for name in ("memory.current", "memory.peak", "memory.max"):
            value = (path / name).read_text().strip()
            values[name] = int(value) if value.isdigit() else value
        values["memory.stat"] = dict(
            line.split() for line in (path / "memory.stat").read_text().splitlines()
        )
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


async def sample_resources(output: Path, stop: asyncio.Event) -> None:
    with output.open("a") as stream:
        while not stop.is_set():
            stream.write(json.dumps(host_memory()) + "\n")
            stream.flush()
            try:
                await asyncio.wait_for(stop.wait(), timeout=0.1)
            except TimeoutError:
                pass

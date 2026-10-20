import asyncio
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

TELEMETRY_VARIABLES = ("QDRANT_PID", "QDRANT_CGROUP_PATH", "QDRANT_STORAGE_PATH")


def _read_counters(text: str) -> dict[str, int]:
    values: dict[str, int] = {}
    for line in text.splitlines():
        key, _, value = line.partition(":")
        value = value.strip()
        if key and value.isdigit():
            values[key] = int(value)
    return values


def host_memory() -> dict[str, Any]:
    """Snapshot process and cgroup counters; report missing sources explicitly."""
    values: dict[str, Any] = {"unix_seconds": time.time()}
    if pid := os.environ.get("QDRANT_PID"):
        try:
            for line in (Path("/proc") / pid / "status").read_text().splitlines():
                key, _, value = line.partition(":")
                if key in {"VmRSS", "VmHWM", "RssAnon", "RssFile", "RssShmem"}:
                    values[f"process_{key}_bytes"] = int(value.split()[0]) * 1024
            values["process_io"] = _read_counters(
                (Path("/proc") / pid / "io").read_text()
            )
            values["process_faults"] = _page_faults(pid)
        except OSError as error:
            values["process_unavailable"] = f"{type(error).__name__}: {error}"
    if location := os.environ.get("QDRANT_CGROUP_PATH"):
        path = Path(location)
        try:
            for name in ("memory.current", "memory.peak", "memory.max"):
                value = (path / name).read_text().strip()
                values[name] = int(value) if value.isdigit() else value
            values["memory.stat"] = {
                key: int(value)
                for key, value in (
                    line.split()
                    for line in (path / "memory.stat").read_text().splitlines()
                )
            }
            values["io.stat"] = (path / "io.stat").read_text()
        except OSError as error:
            values["cgroup_unavailable"] = f"{type(error).__name__}: {error}"
    return values


def _page_faults(pid: str) -> dict[str, int]:
    fields = (Path("/proc") / pid / "stat").read_text().rsplit(")", 1)[1].split()
    # Fields after the command name: state is index 0; minflt is 7, majflt is 9.
    return {"minflt": int(fields[7]), "majflt": int(fields[9])}


def refresh_targets() -> dict[str, str]:
    """Re-resolve PID and cgroup path after the container restarted."""
    container = os.environ.get("QDRANT_CONTAINER_ID")
    if not container:
        raise RuntimeError(
            "Set QDRANT_CONTAINER_ID (source scripts/vm-env.sh) so telemetry can "
            "follow the container across a cache reset"
        )
    pid = subprocess.run(
        ["docker", "inspect", "--format", "{{.State.Pid}}", container],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    if not pid.isdigit() or pid == "0":
        raise RuntimeError(f"Container {container} has no running process")
    cgroup = ""
    for line in (Path("/proc") / pid / "cgroup").read_text().splitlines():
        hierarchy, _, path = line.split(":", 2)
        if hierarchy == "0":
            cgroup = f"/sys/fs/cgroup{path}"
    if not cgroup:
        raise RuntimeError("Could not resolve a cgroup v2 path for the container")
    pid = qdrant_process(Path(cgroup))
    os.environ["QDRANT_PID"] = pid
    os.environ["QDRANT_CGROUP_PATH"] = cgroup
    return {"QDRANT_PID": pid, "QDRANT_CGROUP_PATH": cgroup}


def qdrant_process(cgroup: Path) -> str:
    """The qdrant process inside the container cgroup, not its entrypoint shell."""
    for candidate in (cgroup / "cgroup.procs").read_text().split():
        try:
            if (Path("/proc") / candidate / "comm").read_text().strip() == "qdrant":
                return candidate
        except OSError:
            continue
    raise RuntimeError(f"No qdrant process found in {cgroup}")


def check_telemetry(memory_limit_bytes: int | None) -> dict[str, Any]:
    """Require readable VM telemetry sources and the expected cgroup limit."""
    missing = [name for name in TELEMETRY_VARIABLES if not os.environ.get(name)]
    if missing:
        raise RuntimeError(
            f"VM telemetry required; export {', '.join(missing)} "
            "(source scripts/vm-env.sh on the VM)"
        )
    pid = os.environ["QDRANT_PID"]
    cgroup = Path(os.environ["QDRANT_CGROUP_PATH"])
    storage = Path(os.environ["QDRANT_STORAGE_PATH"])
    problems = []
    for path in (
        Path("/proc") / pid / "status",
        Path("/proc") / pid / "io",
        Path("/proc") / pid / "stat",
        cgroup / "memory.current",
        cgroup / "memory.peak",
        cgroup / "memory.max",
        cgroup / "memory.stat",
        cgroup / "io.stat",
    ):
        try:
            path.read_text()
        except OSError as error:
            problems.append(f"{path}: {type(error).__name__}: {error}")
    if not (storage / "collections").is_dir():
        problems.append(f"{storage}/collections is not a readable directory")
    try:
        comm = (Path("/proc") / pid / "comm").read_text().strip()
        if comm != "qdrant":
            problems.append(
                f"QDRANT_PID={pid} is {comm!r}, not the qdrant process; "
                "re-source scripts/vm-env.sh"
            )
    except OSError as error:
        problems.append(f"/proc/{pid}/comm: {type(error).__name__}: {error}")
    if problems:
        raise RuntimeError(
            "Telemetry sources are not readable by this account:\n  "
            + "\n  ".join(problems)
        )
    limit_text = (cgroup / "memory.max").read_text().strip()
    limit = int(limit_text) if limit_text.isdigit() else None
    if memory_limit_bytes is not None and limit != memory_limit_bytes:
        raise RuntimeError(
            f"Expected cgroup memory.max={memory_limit_bytes}, found {limit_text}; "
            "set QDRANT_MEMORY_LIMIT and recreate the container"
        )
    swap_text = (
        (cgroup / "memory.swap.max").read_text().strip()
        if (cgroup / "memory.swap.max").exists()
        else "unavailable"
    )
    return {
        "pid": pid,
        "cgroup_path": str(cgroup),
        "storage_path": str(storage),
        "memory.max": limit_text,
        "memory.swap.max": swap_text,
        "container_id": os.environ.get("QDRANT_CONTAINER_ID"),
    }


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


def parse_metrics(text: str, collection: str | None = None) -> dict[str, Any]:
    """Pick the allocator memory gauges and one collection's hardware counters."""
    memory: dict[str, int] = {}
    hardware: dict[str, int] = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        name, _, value = line.rpartition(" ")
        if name.startswith("memory_") and value.lstrip("-").isdigit():
            memory[name] = int(value)
        elif collection and name.startswith("collection_hardware_metric_"):
            metric, _, label = name.partition("{")
            if label == f'id="{collection}"}}' and value.isdigit():
                hardware[metric.removeprefix("collection_hardware_metric_")] = int(
                    value
                )
    return {"unix_seconds": time.time(), "memory": memory, "hardware": hardware}

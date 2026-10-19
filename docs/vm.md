# Dedicated VM operation

Use a dedicated Linux x86-64 VM. The provisional size is 8 vCPUs, 32 GiB RAM, and
200 GiB SSD. It is an assumption to validate, not a minimum-memory result.
Record the CPU model, physical/core topology, VM type, disk model or cloud volume
type, kernel, and noisy-neighbor constraints. Keep other workloads off the host.

Install Docker Engine and its Compose plugin using the distribution's documented
procedure, and install [uv](https://docs.astral.sh/uv/getting-started/installation/).
Then follow the README setup. The repository does not provision a VM or change
its firewall.

## Connectivity

Compose publishes REST port 6333 and gRPC port 6334 on `127.0.0.1` only. The local
benchmark uses `QDRANT_URL=http://127.0.0.1:6333`. To access the service from your
laptop, keep an SSH tunnel open:

```bash
ssh -N -L 6333:127.0.0.1:6333 -L 6334:127.0.0.1:6334 USER@VM_HOST
```

Run the measured client on the VM for primary latency results. A laptop tunnel or
Colab connection introduces a different network path; label those measurements.
Do not change the host bindings to `0.0.0.0` just to connect a notebook. For an
external endpoint, configure TLS, API-key authentication, and network access
controls following [Qdrant security guidance](https://qdrant.tech/documentation/operations/security/).
Export its `QDRANT_URL` and `QDRANT_API_KEY` in the client environment. Never put
keys in command arguments, notebook outputs, YAML files, or Git.

Compose reads `.env` for image and memory configuration. It does not automatically
export those variables to the CLI shell. The default URL works without an export;
set client variables explicitly when using a different endpoint.

## Freeze the machine and server

Qdrant starts with release tag `v1.19.2`. Before real measurements, resolve and
record the image digest, then set `QDRANT_IMAGE=qdrant/qdrant@sha256:...` in `.env`.
The lockfile fixes the client dependency set. Do not update versions midway
through a comparison.

```bash
mkdir -p results/environment
uname -a > results/environment/kernel.txt
lscpu > results/environment/cpu.txt
lsblk -o NAME,MODEL,SIZE,ROTA,TYPE,MOUNTPOINTS > results/environment/disks.txt
free -b > results/environment/memory.txt
docker version > results/environment/docker.txt
docker image inspect qdrant/qdrant:v1.19.2 --format '{{json .RepoDigests}}' > results/environment/qdrant-image.json
uv run qbench doctor > results/environment/doctor.txt
git rev-parse HEAD > results/environment/repository-commit.txt
```

Record whether swap is enabled, CPU frequency policy, CPU instruction support,
container CPU restrictions, and any provider-level disk limits. The default
Compose memory cap is 28 GiB, leaving room for the OS and client on a 32 GiB VM;
it is not a memory-pressure experiment by itself.

Build embeddings before timing Qdrant. BGE inference and large array operations
can otherwise compete for CPU and RAM with the server. Keep only one measured
collection active. The runner's UUID-prefixed experiment collections are cleaned
up between configurations; unrelated collections still consume resources.

## Artifacts and resource measurements

Each output directory contains `manifest.json`, per-query `queries.jsonl`, and
`summary.json`. Method/build subdirectories retain collection configuration,
before/after Prometheus metrics, and resource samples. Preserve raw records;
avoid rerunning into a completed output directory.

The CLI's HTTP timing and metrics are not a complete host-memory measurement.
Capture container current/peak and anonymous/file-backed memory counters on the
VM alongside the timed interval. In cgroup v2, these are `memory.current`,
`memory.peak`, and `memory.stat` under the container's cgroup. RSS, heap, and
cgroup memory overlap; do not add them. If a counter is unavailable, report it as
unavailable rather than zero.

The runner can sample the host paths when these variables are exported. On a
Linux host using cgroup v2, obtain the process and cgroup path from Docker:

```bash
export QDRANT_PID="$(docker inspect --format '{{.State.Pid}}' "$(docker compose ps -q qdrant)")"
export QDRANT_CGROUP_PATH="/sys/fs/cgroup$(awk -F: '$1 == "0" {print $3}' "/proc/$QDRANT_PID/cgroup")"
export QDRANT_STORAGE_PATH="$PWD/data/qdrant"
```

Confirm the selected paths are readable by the benchmark account before the
run, including `/proc/$QDRANT_PID/io` and collection files. Docker rootless or
nonstandard cgroup layouts may need different host paths. The script does not
escalate permissions automatically. Re-resolve paths after restarting the
container; process IDs and cgroup paths can change.

Resource samples are recorded every 100 ms across indexing and query measurements. Cgroup
`memory.peak` and process high-water marks are cumulative; they do not become
query-only peaks simply because sampling starts at query time. Use fresh/reset
counters for stage-specific peaks and preserve the stage boundaries.

Use `docker compose ps -q qdrant` to identify the measured container. A snapshot
can be captured with `docker stats --no-stream`; repeated cgroup sampling is
needed for time-aligned peak and cache analysis. Capture build peaks separately
from query-time peaks. `/metrics` is available on the REST endpoint for Qdrant
metrics, subject to the same authentication as the server.

The persistent bind mount is `data/qdrant/`. Measure allocated collection bytes
while the target collection still exists and indexing is finished. The runner
cleans its collections after measurement, so post-run directory size is not the
collection footprint. Account separately for vectors, quantized data, HNSW,
payload, and WAL. Snapshots, logs, and reference `.npy` files are outside that
footprint. `QDRANT_STORAGE_PATH` enables a file-level logical/allocated byte
snapshot before cleanup; classify files into those categories during analysis.

## Warm, cold, and memory-pressure runs

The initial CLI benchmark is a warm-cache workflow. It does not claim controlled
cold results, and it does not drop host caches. For a cold experiment, design a
dedicated-host procedure that resets the intended file cache, restores intended
pinned structures, and checks actual disk reads/page faults. A restarted Qdrant
process can read already-cached filesystem pages.

Before adding a cold-cache procedure, verify it cannot affect other services on
the VM. Keep independently cold requests separate from a cold-start pass that
warms with each query. Record exact operations and resulting I/O counters.

For the separate memory-pressure experiment, select one fixed cap per model in
preflight. Edit `QDRANT_MEMORY_LIMIT` in `.env`, recreate the container, and record
the effective limit:

```bash
docker compose up -d --force-recreate
docker inspect --format '{{.HostConfig.Memory}}' "$(docker compose ps -q qdrant)"
```

Wait for readiness again before running. Keep the cap the same across methods,
and report OOMs, timeouts, and index-build failures. Do not lower the cap for one
method until it looks faster than another.

## Shutdown

```bash
docker compose stop
```

This preserves storage and experiment output. `docker compose down` removes the
container and network while preserving the bind-mounted directory. Deleting
`data/qdrant/` removes the stored collections; use a dedicated experiment server
and preserve required outputs before deliberate cleanup.

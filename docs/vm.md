# Dedicated VM operation

Use a dedicated Linux x86-64 VM. The provisional size is 8 vCPUs, 32 GiB RAM, and
200 GiB SSD. It is an assumption to validate, not a minimum-memory result.
Record the CPU model, physical/core topology, VM type, disk model or cloud volume
type, kernel, and noisy-neighbor constraints. Keep other workloads off the host.

Install Docker Engine and its Compose plugin using the distribution's documented
procedure, and install [uv](https://docs.astral.sh/uv/getting-started/installation/).
Then follow the README setup. The repository does not provision a VM or change
its firewall.

## Bring the prepared data

The measurement VM needs no GPU. Prepare BGE before timing Qdrant. Transfer the
prepared `data/raw`, `data/embeddings`, `data/sources`, and `data/models` directories
to the checkout, preserving their relative paths. Vector arrays are portable
float32 `.npy` files; an MPS-prepared BGE artifact can be measured on Linux CPUs
without re-embedding it. Its manifest retains the original preparation device.

If using the prepared data archive, extract it from the repository root, then
check the entire matrix:

```bash
tar -xf /path/to/quantization-vm-data.tar
uv sync --locked
uv run qbench readiness --require-all
```

If preparing from scratch instead, use `uv sync --locked --extra datasets --extra
bge` followed by `uv run qbench prepare-all --with-bge`. Keep downloads and
embedding inference outside measured search intervals.

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
it is not a memory-pressure experiment by itself. Compose also fixes the CPU
limit at eight and disables container swap by setting memory+swap equal to RAM.

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
source scripts/vm-env.sh
```

Confirm the selected paths are readable by the benchmark account before the
run, including `/proc/$QDRANT_PID/io` and collection files. Docker rootless or
nonstandard cgroup layouts may need different host paths. The script does not
escalate permissions automatically. Re-resolve paths after restarting the
container; process IDs and cgroup paths can change.
If `/proc/$QDRANT_PID/io` or the bind-mounted files are root-owned, run the
benchmark from a root shell with the existing tool PATH, for example
`sudo env "PATH=$PATH" bash scripts/run-workload.sh scifact cohere`.
The script records the environment before running. Keep the same execution user
for subsequent measured runs and result processing.

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

The primary configuration is warm-cache. `configs/cold-start.yaml` explicitly
runs `scripts/cold-cache.sh` before each timed pass, with no warm-up. The pass
gradually warms as requests proceed; it is not a series of independently cold
queries. The hook uses `sync` and the Linux cache-drop interface, so use it only
on this dedicated VM. It requires passwordless execution of the cache-drop
command (or a root benchmark shell) and explicit acknowledgement:

```bash
export QBENCH_DEDICATED_VM=1
source scripts/vm-env.sh
uv run qbench run --config configs/cold-start.yaml --data data/embeddings/quora/cohere --output results/quora-cohere-cold-start
```

Inspect `cache-resets.jsonl`, `measurement-intervals.jsonl`, process reads and
major faults before interpreting cache effects. Successful reset command exit
alone does not prove physical disk reads for every request.

For the separate pressure run, set `QDRANT_MEMORY_LIMIT=1g` in `.env` and recreate
the service. Quora/Cohere originals occupy 2,141,925,376 bytes, so a 1 GiB cgroup
cap is below the original-vector footprint. Keep this cap fixed across methods.

```bash
docker compose up -d --force-recreate
docker inspect --format '{{.HostConfig.Memory}}' "$(docker compose ps -q qdrant)"
source scripts/vm-env.sh
uv run qbench doctor
uv run qbench run --config configs/memory-pressure.yaml --data data/embeddings/quora/cohere --output results/quora-cohere-pressure-1g
```

Wait for readiness after recreation. The runner checks the actual cgroup cap and
requires readable resource telemetry; it refuses to label an unconstrained run
as pressure. A BGE pressure run needs a separately preregistered cap below its
803,222,016-byte originals; 1 GiB would fail this check.

Report OOMs, timeouts and build failures. Failure aborts the current run and saves
its error; it does not imply that the remaining methods ran. Recover the server,
inspect any leftover experiment collection, and explicitly schedule unmeasured
methods at the same cap. Restore `QDRANT_MEMORY_LIMIT=28g`, recreate and re-source
the environment before later normal runs.

## Shutdown

```bash
docker compose stop
```

This preserves storage and experiment output. `docker compose down` removes the
container and network while preserving the bind-mounted directory. Deleting
`data/qdrant/` removes the stored collections; use a dedicated experiment server
and preserve required outputs before deliberate cleanup.


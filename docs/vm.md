# Dedicated VM operation

Use a dedicated Linux x86-64 VM: 8 vCPUs, 32 GiB RAM, 200 GiB SSD. Keep other
workloads off the host. Install Docker Engine with the Compose plugin, Git, and
[uv](https://docs.astral.sh/uv/getting-started/installation/), then follow the
README. The repository does not provision a VM or change its firewall.

Disk budget: the Quora Cohere corpus is about 2 GiB as float32, its BGE corpus
about 0.8 GiB, BGE checkpoints double that, the Hugging Face cache holds the
downloaded Parquet files (about 1.3 GiB), and each Qdrant collection holds
originals plus quantized copies plus HNSW while it exists. 200 GiB is ample.

## Connectivity

Compose publishes REST port 6333 and gRPC port 6334 on `127.0.0.1` only. The
measured client runs on the VM with `QDRANT_URL=http://127.0.0.1:6333`. To
inspect the server from a laptop, keep an SSH tunnel open rather than changing
the bindings:

```bash
ssh -N -L 6333:127.0.0.1:6333 -L 6334:127.0.0.1:6334 USER@VM_HOST
```

Laptop or Colab connections use a different network path; label such
measurements. For an external endpoint, configure TLS, API-key authentication,
and network access controls per
[Qdrant security guidance](https://qdrant.tech/documentation/operations/security/).
Never put keys in command arguments, notebook outputs, YAML files, or Git.

## Freeze the server

Qdrant starts from the release tag `v1.19.2`. Before measured runs, resolve and
pin the image digest in `.env`, then recreate the container:

```bash
docker image inspect qdrant/qdrant:v1.19.2 --format '{{json .RepoDigests}}'
# QDRANT_IMAGE=qdrant/qdrant@sha256:...
docker compose up -d --force-recreate
```

The lockfile fixes the client dependency set. Do not update either midway
through a comparison.

## Export telemetry targets

```bash
source scripts/vm-env.sh            # from the repository root, container running
```

The script resolves the container ID, the `qdrant` process ID inside it (the
image entrypoint is a shell wrapper, which is not the process to measure), its
cgroup v2 path, and
the storage bind mount, exports `QDRANT_CONTAINER_ID`, `QDRANT_PID`,
`QDRANT_CGROUP_PATH`, and `QDRANT_STORAGE_PATH`, and checks that this account
can read `/proc/<pid>/status`, `/proc/<pid>/io`, `/proc/<pid>/stat`, the cgroup
memory and I/O files, and the collections directory. The Qdrant process runs
as root inside the container, so `/proc/<pid>/io` is only readable by root;
run the client as root (`sudo -E`) or grant access before measuring. It also
writes the machine record under `results/environment/`: kernel, CPU model and
flags, core count, governor, disks, free space, memory, swap, container
inspection, image digest, cgroup memory/swap/CPU limits, Qdrant thread count,
repository commit and status, lockfile hash, and `qbench doctor` output.

Re-source the script after recreating the container; process IDs and cgroup
paths change. Configurations with `require_telemetry: true` refuse to start
without readable targets, and `memory_limit_bytes` is compared with the
cgroup's `memory.max`.

## Resource measurements

The Compose service starts Qdrant with `QDRANT__SERVICE__HARDWARE_REPORTING=true`
so every query response carries CPU and vector I/O counters and `/metrics`
exposes per-collection hardware counters. Configurations with
`require_telemetry: true` fail if the counters are absent.

Each method/build directory contains `memory-pre-index.json`,
`memory-post-index.json`, `memory-post-search.json`, `resources.jsonl` sampled
every 100 ms, `storage.json` taken while the collection exists after indexing,
`passes.jsonl` with counters before and after every measured pass, and the
Prometheus `/metrics` text before and after search. Process RSS, heap, and
cgroup memory overlap; do not add them. Cgroup `memory.peak` and process
high-water marks are cumulative, so stage peaks come from the samples between
stage boundaries, not from the final value. Unavailable counters are reported
as unavailable, never as zero.

`storage.json` lists logical and allocated bytes per file under the
collection directory; classify vectors, quantized data, HNSW, payload, and WAL
during analysis. Post-run directory size is not the collection footprint
because the runner deletes its collections.

## Memory-pressure run

```bash
scripts/memory-pressure.sh
```

Sets `QDRANT_MEMORY_LIMIT=1g`, recreates the container, re-sources the
telemetry targets, records the effective host limit, and runs
`configs/search-cap.yaml` on Quora/Cohere. The runner verifies
`memory.max` equals 1 GiB before measuring. Keep the cap identical across
methods and report OOMs, timeouts, and index-build failures. Restore
`QDRANT_MEMORY_LIMIT=28g` and recreate the container afterwards.

## Cold-start run

```bash
export QBENCH_ALLOW_CACHE_RESET=1
uv run qbench run --config configs/cold-start.yaml \
  --data data/embeddings/fiqa/cohere --output results/fiqa-cohere/cold-start
```

Before every measured pass the runner executes `scripts/cache-reset.sh`: stop
the container (releasing its mapped files), `sync`, write `3` to
`/proc/sys/vm/drop_caches`, start the container, wait for readiness, then
re-resolve the process ID through `QDRANT_CONTAINER_ID`. Dropping caches
affects every process on the host, which is why the hook requires the
explicit opt-in variable and a dedicated VM. No warm-up requests are sent; the
pass warms as its queries proceed, and `passes.jsonl` holds the I/O and
page-fault counters that show whether the reset actually produced disk reads.

## Shutdown

```bash
docker compose stop     # keeps storage and results
docker compose down     # removes container and network, keeps data/qdrant/
```

Deleting `data/qdrant/` removes stored collections. Preserve required outputs
before deliberate cleanup.

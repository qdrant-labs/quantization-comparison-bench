#!/usr/bin/env bash
# Source this on the dedicated VM from the repository root after
# `docker compose up -d`:
#
#   source scripts/vm-env.sh [results/environment]
#
# It records the machine and server, resolves the Qdrant container's PID,
# cgroup, and storage paths, and exports the QDRANT_* telemetry variables the
# runner uses. Re-source it after recreating the container.
if [ "${BASH_SOURCE[0]}" = "$0" ]; then
  echo "Source this script instead of executing it: source scripts/vm-env.sh" >&2
  exit 1
fi

qbench_env() {
  local record_dir="${1:-results/environment}"
  if [ ! -f compose.yaml ] || [ ! -f pyproject.toml ]; then
    echo "Run from the repository root" >&2
    return 1
  fi
  if [ "$(uname -s)" != "Linux" ]; then
    echo "Telemetry needs a Linux host with cgroup v2; this is $(uname -s)" >&2
    return 1
  fi
  local container
  container="$(docker compose ps -q qdrant)"
  if [ -z "$container" ]; then
    echo "The qdrant Compose service is not running: docker compose up -d" >&2
    return 1
  fi
  local pid
  pid="$(docker inspect --format '{{.State.Pid}}' "$container")"
  if [ -z "$pid" ] || [ "$pid" = "0" ]; then
    echo "Container $container has no running process" >&2
    return 1
  fi
  local cgroup
  cgroup="/sys/fs/cgroup$(awk -F: '$1 == "0" {print $3}' "/proc/$pid/cgroup")"
  export QDRANT_URL="${QDRANT_URL:-http://127.0.0.1:6333}"
  export QDRANT_CONTAINER_ID="$container"
  export QDRANT_PID="$pid"
  export QDRANT_CGROUP_PATH="$cgroup"
  export QDRANT_STORAGE_PATH="$PWD/data/qdrant"

  local problems=0
  local path
  for path in "/proc/$pid/status" "/proc/$pid/io" "/proc/$pid/stat" \
    "$cgroup/memory.current" "$cgroup/memory.peak" "$cgroup/memory.max" \
    "$cgroup/memory.stat" "$cgroup/io.stat"; do
    if ! cat "$path" >/dev/null 2>&1; then
      echo "Not readable: $path" >&2
      problems=1
    fi
  done
  if [ ! -d "$QDRANT_STORAGE_PATH/collections" ] || ! ls "$QDRANT_STORAGE_PATH/collections" >/dev/null 2>&1; then
    echo "Not readable: $QDRANT_STORAGE_PATH/collections" >&2
    problems=1
  fi
  if [ "$problems" -ne 0 ]; then
    echo "The Qdrant process runs as root inside the container; run the benchmark" >&2
    echo "client as root (sudo -E ...) or grant read access before measuring." >&2
    return 1
  fi

  mkdir -p "$record_dir"
  uname -a > "$record_dir/kernel.txt"
  lscpu > "$record_dir/cpu.txt"
  nproc > "$record_dir/nproc.txt"
  lsblk -o NAME,MODEL,SIZE,ROTA,TYPE,MOUNTPOINTS > "$record_dir/disks.txt"
  df -h "$PWD" > "$record_dir/disk-space.txt"
  free -b > "$record_dir/memory.txt"
  { swapon --show 2>/dev/null || true; cat /proc/swaps; } > "$record_dir/swap.txt"
  cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor > "$record_dir/cpu-governor.txt" 2>/dev/null || echo unavailable > "$record_dir/cpu-governor.txt"
  grep -m1 flags /proc/cpuinfo > "$record_dir/cpu-flags.txt"
  docker version > "$record_dir/docker.txt"
  docker inspect "$container" > "$record_dir/container.json"
  docker inspect --format '{{.Config.Image}} {{.Image}}' "$container" > "$record_dir/container-image.txt"
  docker image inspect "$(docker inspect --format '{{.Config.Image}}' "$container")" --format '{{json .RepoDigests}}' > "$record_dir/qdrant-image.json"
  cat "$cgroup/memory.max" > "$record_dir/cgroup-memory-max.txt"
  cat "$cgroup/memory.swap.max" > "$record_dir/cgroup-memory-swap-max.txt" 2>/dev/null || echo unavailable > "$record_dir/cgroup-memory-swap-max.txt"
  cat "$cgroup/cpu.max" > "$record_dir/cgroup-cpu-max.txt" 2>/dev/null || echo unavailable > "$record_dir/cgroup-cpu-max.txt"
  grep -c '' "/proc/$pid/task/"* 2>/dev/null | wc -l > "$record_dir/qdrant-threads.txt" || true
  ls "/proc/$pid/task" | wc -l > "$record_dir/qdrant-threads.txt"
  git rev-parse HEAD > "$record_dir/repository-commit.txt"
  git status --porcelain > "$record_dir/repository-status.txt"
  sha256sum uv.lock > "$record_dir/uv-lock.sha256"
  cp .env "$record_dir/env.txt" 2>/dev/null || true
  uv run qbench doctor > "$record_dir/doctor.txt"
  {
    echo "QDRANT_URL=$QDRANT_URL"
    echo "QDRANT_CONTAINER_ID=$QDRANT_CONTAINER_ID"
    echo "QDRANT_PID=$QDRANT_PID"
    echo "QDRANT_CGROUP_PATH=$QDRANT_CGROUP_PATH"
    echo "QDRANT_STORAGE_PATH=$QDRANT_STORAGE_PATH"
  } > "$record_dir/telemetry-targets.txt"
  echo "Telemetry targets exported; environment recorded under $record_dir"
}

qbench_env "$@"

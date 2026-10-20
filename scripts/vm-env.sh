#!/usr/bin/env bash
# Source this file from the repository root on the benchmark VM.

if [[ "$(uname -s)" != Linux ]]; then
    printf 'VM telemetry requires Linux.\n' >&2
    return 1 2>/dev/null || exit 1
fi
qbench_container="$(docker compose ps -q qdrant)"
if [[ -z "$qbench_container" ]]; then
    printf 'Start the repository Qdrant service with docker compose up -d.\n' >&2
    return 1 2>/dev/null || exit 1
fi
QDRANT_PID="$(docker inspect --format '{{.State.Pid}}' "$qbench_container")"
qbench_cgroup="$(awk -F: '$1 == "0" {print $3}' "/proc/$QDRANT_PID/cgroup")"
if [[ -z "$qbench_cgroup" || ! -r "/sys/fs/cgroup$qbench_cgroup/memory.current" ]]; then
    printf 'A readable cgroup v2 memory controller is required.\n' >&2
    return 1 2>/dev/null || exit 1
fi
export QDRANT_PID
export QDRANT_CGROUP_PATH="/sys/fs/cgroup$qbench_cgroup"
export QDRANT_STORAGE_PATH="$PWD/data/qdrant"
export QDRANT_URL="http://127.0.0.1:6333"
mkdir -p results/environment
uname -a > results/environment/kernel.txt
lscpu > results/environment/cpu.txt
lsblk -o NAME,MODEL,SIZE,ROTA,TYPE,MOUNTPOINTS > results/environment/disks.txt
free -b > results/environment/memory.txt
docker inspect --format '{{json .HostConfig}}' "$qbench_container" > results/environment/container-limits.json
docker image inspect "$(docker inspect --format '{{.Image}}' "$qbench_container")" \
    --format '{{json .RepoDigests}}' > results/environment/qdrant-image.json
git -c safe.directory="$PWD" rev-parse HEAD > results/environment/repository-commit.txt
printf 'Telemetry configured. Re-source this file after recreating the container.\n'
unset qbench_container qbench_cgroup

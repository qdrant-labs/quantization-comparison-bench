#!/usr/bin/env bash
# Dedicated-VM cache-reset hook used before every cold-start pass.
# Stops the Qdrant container so its mapped files are released, drops the host
# page cache, starts the container again, and waits for readiness. The runner
# re-resolves the process ID afterwards via QDRANT_CONTAINER_ID.
#
# Dropping caches affects every process on the host. Set
# QBENCH_ALLOW_CACHE_RESET=1 only on a VM dedicated to this benchmark.
set -euo pipefail

if [ "${QBENCH_ALLOW_CACHE_RESET:-}" != "1" ]; then
  echo "Refusing to drop host caches: export QBENCH_ALLOW_CACHE_RESET=1 on a dedicated VM" >&2
  exit 1
fi
: "${QDRANT_CONTAINER_ID:?source scripts/vm-env.sh first}"
url="${QDRANT_URL:-http://127.0.0.1:6333}"

docker stop "$QDRANT_CONTAINER_ID" >/dev/null
sync
if [ "$(id -u)" -eq 0 ]; then
  echo 3 > /proc/sys/vm/drop_caches
else
  echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null
fi
docker start "$QDRANT_CONTAINER_ID" >/dev/null
for _ in $(seq 1 120); do
  if curl --fail --silent "$url/readyz" >/dev/null; then
    echo "cache reset complete; container restarted"
    exit 0
  fi
  sleep 1
done
echo "Qdrant did not become ready after the cache reset" >&2
exit 1

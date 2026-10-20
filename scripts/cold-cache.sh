#!/usr/bin/env bash
set -euo pipefail

if [[ "$(uname -s)" != Linux || "${QBENCH_DEDICATED_VM:-}" != 1 ]]; then
    printf 'Use only on the dedicated benchmark VM, with QBENCH_DEDICATED_VM=1.\n' >&2
    exit 1
fi
sync
sudo -n sh -c 'echo 3 > /proc/sys/vm/drop_caches'

#!/usr/bin/env bash
# Restore the normal 28 GiB cap in place after a capped search phase.
#   scripts/restore-cap.sh <ignored>
set -euo pipefail
exec "$(dirname "$0")/set-cap.sh" 30064771072

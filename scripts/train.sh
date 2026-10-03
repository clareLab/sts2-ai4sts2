#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 RAY_USAGE_STATS_ENABLED=0
if (( $# == 0 )); then set -- pilot; fi
./scripts/build.sh
if command -v systemd-run >/dev/null && systemctl --user show-environment >/dev/null 2>&1; then
  exec systemd-run --user --scope --quiet --unit="ai4sts2-pilot-$$" \
    -p CPUQuota=200% -p MemoryHigh=3G -p MemoryMax=4G -p MemorySwapMax=0 -p RuntimeMaxSec=1800 \
    timeout --signal=INT --kill-after=20 1800 ./scripts/python.sh ai4sts2 "$@"
fi
exec timeout --signal=INT --kill-after=20 1800 ./scripts/python.sh ai4sts2 "$@"

#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p artifacts
exec systemd-run --user --scope --quiet -p MemoryHigh=14G -p MemoryMax=16G -p MemorySwapMax=0 \
  flock -n artifacts/train.lock ./scripts/python.sh python -m ai4sts2.train "$@"

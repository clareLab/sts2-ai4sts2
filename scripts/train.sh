#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/common.sh
mkdir -p artifacts
ai4sts2_contained -p MemoryHigh=17G -p MemoryMax=18G \
  flock -n artifacts/train.lock ./scripts/python.sh python -m ai4sts2.train "$@"

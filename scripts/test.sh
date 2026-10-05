#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/common.sh
./scripts/build.sh
ai4sts2_contained ./scripts/python.sh pytest -q "$@"

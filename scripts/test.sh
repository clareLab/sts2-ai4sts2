#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
./scripts/build.sh
./scripts/python.sh pytest -q "$@"

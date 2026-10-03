#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
case "${1:-}" in
  --source) python3 tests/check_package.py --source ;;
  "") ./scripts/build.sh; python3 tests/check_package.py ;;
  *) echo 'Usage: ./scripts/test.sh [--source]' >&2; exit 2 ;;
esac
./scripts/python.sh pytest -q

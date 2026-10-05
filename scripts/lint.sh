#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/common.sh
ai4sts2_dotnet format whitespace mod --folder --verify-no-changes
./scripts/python.sh --only-group dev ruff check
./scripts/python.sh --only-group dev ruff format --check
if command -v shellcheck >/dev/null; then shellcheck -x scripts/*.sh
elif command -v nix >/dev/null; then nix shell nixpkgs#shellcheck -c shellcheck -x scripts/*.sh
else echo 'ShellCheck is required.' >&2; exit 1; fi
git diff --check

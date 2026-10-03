#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/common.sh
ai4sts2_dotnet format whitespace . --folder --include src/*.cs --verify-no-changes
if command -v shellcheck >/dev/null; then shellcheck -x scripts/*.sh
elif command -v nix >/dev/null; then nix shell nixpkgs#shellcheck -c shellcheck -x scripts/*.sh
else echo 'ShellCheck is required.' >&2; exit 1; fi
python3 -m compileall -q tests
git diff --check
git diff --cached --check

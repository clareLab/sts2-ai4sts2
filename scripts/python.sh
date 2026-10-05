#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ -f /etc/NIXOS && "${AI4STS2_SHELL:-}" != 1 ]]; then
  printf -v command '%q ' uv run --frozen "$@"
  exec nix-shell --quiet --run "exec $command"
fi
exec uv run --frozen "$@"

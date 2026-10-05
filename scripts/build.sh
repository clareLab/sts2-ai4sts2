#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/common.sh
ai4sts2_game_paths "${1:-}"
ai4sts2_dotnet build mod/ai4sts2.csproj -c Release --disable-build-servers "-p:Sts2DataDir=$data_dir"

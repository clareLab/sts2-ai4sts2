#!/usr/bin/env bash

export DOTNET_CLI_TELEMETRY_OPTOUT=1 DOTNET_NOLOGO=1

ai4sts2_dotnet() {
  if command -v dotnet >/dev/null; then dotnet "$@"
  elif command -v nix >/dev/null; then nix shell nixpkgs#dotnet-sdk_9 -c dotnet "$@"
  else echo '.NET SDK 9 is required.' >&2; return 1; fi
}

ai4sts2_game_paths() {
  game_dir="${1:-${STS2_DIR:-$HOME/.local/share/Steam/steamapps/common/Slay the Spire 2}}"
  data_dir="${STS2_DATA_DIR:-$game_dir/data_sts2_linuxbsd_x86_64}"
  [[ -f "$data_dir/sts2.dll" ]] || { echo 'sts2.dll not found. Set STS2_DIR or STS2_DATA_DIR.' >&2; return 1; }
}

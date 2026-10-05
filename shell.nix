{ pkgs ? import <nixpkgs> {} }:
pkgs.mkShell {
  packages = [ pkgs.dotnet-sdk_9 pkgs.python312 pkgs.uv pkgs.shellcheck ];
  LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [ pkgs.stdenv.cc.cc.lib pkgs.zlib pkgs.icu pkgs.intel-compute-runtime.drivers pkgs.level-zero ];
  AI4STS2_SHELL = "1";
  UV_PYTHON = pkgs.python312.interpreter;
  UV_PYTHON_DOWNLOADS = "never";
  DOTNET_CLI_TELEMETRY_OPTOUT = "1";
  DOTNET_NOLOGO = "1";
}

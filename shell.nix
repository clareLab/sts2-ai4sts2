{
  pkgs ? import (fetchTarball {
    url = "https://github.com/NixOS/nixpkgs/archive/cf9d2fb3e50fa1cd5114c47505ea9177f7ff5f49.tar.gz";
    sha256 = "sha256-cFTsMQz8Hzn8MT49oaeLSHg62tE86NykugCWkeM2ypk=";
  }) { },
}:
pkgs.mkShell {
  packages = [ pkgs.dotnet-sdk_9 pkgs.python312 pkgs.uv pkgs.shellcheck ];
  LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [ pkgs.stdenv.cc.cc.lib pkgs.zlib pkgs.icu pkgs.intel-compute-runtime.drivers pkgs.level-zero ];
  AI4STS2_SHELL = "1";
  OCL_ICD_VENDORS = "${pkgs.intel-compute-runtime}/etc/OpenCL/vendors";
  UV_PYTHON = pkgs.python312.interpreter;
  UV_PYTHON_DOWNLOADS = "never";
  DOTNET_CLI_TELEMETRY_OPTOUT = "1";
  DOTNET_NOLOGO = "1";
}

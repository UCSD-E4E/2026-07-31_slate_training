{
  description = "slate-training — dive-slate pre-annotation for FishSense assisted labeling";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs =
    { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = import nixpkgs { inherit system; };

        # This repo is pure Python, but `fishsense-core` is a PyO3/maturin
        # crate. When `[tool.uv.sources]` points it at the local checkout (the
        # default, for co-development), uv builds it from source and needs a
        # full Rust + C toolchain. Without it the failure is an opaque
        # `linker 'cc' not found` from deep inside a cargo build.
        #
        # Not needed if you switch that source to the published wheels
        # (GitHub release `fishsense-core-v2.4.1`, note the *hyphen* — the
        # underscore tag carries no assets). Kept here because building from
        # source is what you want while both repos are moving.
        nativeLibs = [
          pkgs.openblas   # ndarray-linalg, located via pkg-config, dlopen'd at run time
          pkgs.openssl    # reqwest in fishsense-core's build.rs
        ];
      in
      {
        devShells.default = pkgs.mkShell {
          nativeBuildInputs = [
            pkgs.pkg-config
            pkgs.cargo
            pkgs.rustc
            pkgs.uv
            pkgs.python313
          ];
          buildInputs = nativeLibs;

          LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath (nativeLibs ++ [
            # CUDA userspace lives outside nixpkgs on NixOS hosts; torch needs
            # libcuda.so from the running driver to see the GPU at all. Harmless
            # when absent — training falls back to CPU, and inference is CPU-only
            # by design (the mask is ~200 ms/frame).
            "/run/opengl-driver"
          ]);
        };

        formatter = pkgs.nixpkgs-fmt;
      }
    );
}

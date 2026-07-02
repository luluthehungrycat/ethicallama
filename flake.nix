{
  description = "ethicallama — ethical local LLM inference";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    rust-overlay.url = "github:oxalica/rust-overlay";
  };

  outputs =
    { self, nixpkgs, flake-utils, rust-overlay }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        overlays = [ (import rust-overlay) ];
        pkgs = import nixpkgs { inherit system overlays; };
        rustToolchain = pkgs.rust-bin.stable.latest.default.override {
          extensions = [ "rust-src" "rust-analyzer" "clippy" ];
        };
      in
      {
        devShells.default = pkgs.mkShell {
          buildInputs = with pkgs; [
            rustToolchain
            python3
            python3Packages.pip
            python3Packages.venvShellHook
            pkg-config
            openssl
            vulkan-loader
            clang
          ];

          shellHook = ''
            echo "🦙 ethllama-core dev shell"
            rustc --version
            cargo --version
          '';
        };
      }
    );
}

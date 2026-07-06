{
  description = "ethicallama — ethical local LLM inference (Python + Rust + llama.cpp)";

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

        # Latest stable Rust with the standard dev tooling.
        rustToolchain = pkgs.rust-bin.stable.latest.default.override {
          extensions = [
            "rust-src"
            "rust-analyzer"
            "clippy"
            "rustfmt"
          ];
        };
      in
      {
        # ---------------------------------------------------------------------
        # Dev shell
        # ---------------------------------------------------------------------
        # `nix develop` drops you into an environment with everything needed to
        # build ethllama-core (PyO3 + llama.cpp) and the Python package.
        # The shellHook prints a quickstart; the recommended workflow is:
        #
        #   uv venv && source .venv/bin/activate
        #   uv pip install maturin '.[all]'
        #   maturin develop --release
        #
        devShells.default = pkgs.mkShell {
          buildInputs = with pkgs; [
            # --- Rust toolchain (cargo, rustc, clippy, rust-analyzer, rustfmt) ---
            rustToolchain

            # --- Python interpreter + build tools ---
            # The project is consumed via a venv (uv is recommended, but
            # `python -m venv` works too). Maturin is also pulled in here so
            # `pip install -e .` works without a separate install step.
            python3
            python3Packages.pip
            # `maturin` is exposed at the top level in modern nixpkgs
            # (it was removed from `python3Packages` in the 24.11 release).
            maturin

            # --- Native build deps for llama.cpp + ethllama-core --------------
            cmake
            ninja
            pkg-config
            openssl
            git

            # --- C/C++ compilers (the cmake and cc crates pick one) -----------
            gcc
            clang

            # --- Optional GPU support (Vulkan). The standalone llama.cpp build
            # enables these via `-DLLAMA_VULKAN=ON`, `-DLLAMA_CUDA=ON`, etc.
            # The bundled `ethllama-core` is CPU-only (see `build.rs`).
            vulkan-headers
            vulkan-loader
          ];

          # Library search paths for pkg-config consumers (Python wheels, etc.)
          env = {
            PKG_CONFIG_PATH = "${pkgs.openssl.dev}/lib/pkgconfig";
          };

          shellHook = ''
            # Auto-initialize the llama.cpp git submodule on first entry.
            # The Rust core's `build.rs` shells out to cmake against this path
            # and will panic if it's missing.
            if [ -f .gitmodules ] && [ ! -d ethllama-core/llama.cpp/common ]; then
              echo "📦 Initializing git submodules (llama.cpp)…"
              git submodule update --init --recursive
              echo ""
            fi

            echo "╭──────────────────────────────────────────────────────╮"
            echo "│  🦙  ethicallama dev shell                            │"
            echo "╰──────────────────────────────────────────────────────╯"
            echo ""
            printf '  %-9s %s\n' \
              "python"  "$(python3 --version 2>&1)" \
              "rustc"   "$(rustc --version 2>&1)" \
              "cargo"   "$(cargo --version 2>&1)" \
              "cmake"   "$(cmake --version 2>&1 | head -n1)" \
              "maturin" "$(maturin --version 2>&1)" \
              "git"     "$(git --version 2>&1)"
            echo ""

            cat <<'EOF'
Quickstart:

  # 1. Create a venv (uv recommended) and install Python deps
  uv venv
  source .venv/bin/activate
  uv pip install maturin '.[all]'

  # 2. Build the Rust extension into the venv
  maturin develop --release

  # 3. Run the test suite
  pytest ethllama/tests/ -v

  # 4. (Optional) Build standalone llama.cpp binaries
  cmake -S ethllama-core/llama.cpp -B llama.cpp-build \
        -DLLAMA_BUILD_TOOLS=ON -DCMAKE_BUILD_TYPE=Release
  cmake --build llama.cpp-build --config Release -j
EOF
          '';
        };

        # ---------------------------------------------------------------------
        # Formatter (`nix fmt`)
        # ---------------------------------------------------------------------
        # `pkgs.nixfmt` is the RFC-style formatter in modern nixpkgs
        # (it was renamed from `pkgs.nixfmt-rfc-style` in 24.11).
        formatter = pkgs.nixfmt;
      }
    );
}

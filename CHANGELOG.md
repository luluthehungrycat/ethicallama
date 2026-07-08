# Changelog

All notable changes to **ethicallama** are documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- TTS engine support: `ethllama tts` command synthesizes speech from text.
  Supports `--voice`, `--speed`, `--play`, `--format` (wav/mp3/ogg/flac).
  Engine type `tts` in EngineConfig. Mirrors the existing STT pattern.
  (24 new tests, 134 total.)
- Per-model configuration aliases: set defaults in `~/.ethllama/config.yaml`
  under `model_defaults:<model_stem>` for temperature, top_k, n_gpu_layers,
  gpu_backend, system_prompt, ctx_size, chat_template path. CLI flags take
  precedence over per-model values. (14 new tests.)

## [0.1.2] - 2026-07-07
- Configurable binary paths for llama.cpp tools: users can set
  `engines.binary_dir` in `~/.ethllama/config.yaml` or pass
  `--binary-dir /path/to/bin` to `run`, `serve`, and `quantize`
  commands. Binary discovery now checks: runtime override >
  config file > submodule build dir > PATH.
- `python -m ethllama` support via `__main__.py`.
- Wheel now includes `ethllama/` Python package (cli.py, api.py, inference.py,
  etc.) via explicit maturin `include` globs. The `python-source` setting alone
  did not bundle the source directory.
- Rust extension import now tries both `ethllama_core` and
  `ethllama.ethllama_core` paths for editable install vs. wheel compatibility.
- Created `ethllama/ethllama_core/__init__.py` so maturin has the expected
  directory structure when `python-source` + `module-name` are used.

## [0.1.0] - 2026-07-07
- GitHub Actions CI workflow (`.github/workflows/ci.yml`) testing
  Python 3.10 – 3.13 on `ubuntu-latest` (Rust + Python, with rustfmt/clippy lint job).
- GitHub Actions release workflow (`.github/workflows/release.yml`) that builds
  manylinux + musllinux + macOS wheels via cibuildwheel, produces an sdist,
  attaches everything to a GitHub Release, and publishes to PyPI using
  **OIDC trusted publishing** (no API token in secrets).
- `[project.scripts]` entry exposing the `ethllama` CLI on `pip install`.
- `[project.urls]` block with Homepage / Repository / Issues / Changelog links.
- `[tool.cibuildwheel]` configuration in `pyproject.toml` (build matrix,
  `MATH_BACKEND=openblas` for manylinux, pinned `pytest` test-requires).
- `[tool.pytest.ini_options]` section (`testpaths`, `asyncio_mode = "auto"`).
- PyPI / CI / License / Python-version badges at the top of `README.md`.
- `MANIFEST.in` controlling sdist contents (excludes `ethllama-core/target`,
  `llama.cpp-build`, the bundled `ethllama-core/llama.cpp` submodule,
  `.slim`, and `.benchmarks`).

### Changed
- `pyproject.toml` moved from `ethllama/pyproject.toml` to the repository root
  (the standard maturin layout). All `[tool.maturin]` paths were already
  written relative to the project root, so no path edits were needed.
- `requires-python` bumped from `>=3.9` to `>=3.10` to match the CI matrix
  and the README's stated prerequisites.

## [0.1.0] - 2026-07-03

### Added
- **CLI** (`ethllama.cli`) — 11 subcommands built on Click:
  - `run`     — run inference against a local GGUF model
  - `pull`    — pull a model from HuggingFace Hub
  - `list`    — list indexed models
  - `index`   — add / remove / list models in the local model index
  - `config`  — load, save, and initialise `~/.ethllama/config.yaml`
  - `serve`   — start the local FastAPI server
  - `engines` — list and inspect configured engines
  - `quantize` — quantise a GGUF model via `llama-quantize`
  - `transcribe` — speech-to-text via a whisper.cpp engine
  - `info`    — show GGUF model metadata
  - `rm`      — remove a model from the index
- **FastAPI server** (`ethllama.api`) — opt-in OpenAI-compatible endpoints:
  - `POST /v1/chat/completions`
  - `POST /v1/completions`
  - `POST /v1/embeddings`
  - `GET  /v1/models`
  - `GET  /health`
  Supports optional bearer-token auth, streaming responses, and is gated
  behind the `ethllama[api]` extra.
- **Real inference** — `ethllama.inference` shells out to `llama-cli` /
  `llama-embedding` binaries when the Rust core is unavailable, so the
  package works out-of-the-box for any user with a llama.cpp checkout.
- **Rust FFI core** (`ethllama-core/`) — PyO3 bindings to llama.cpp
  covering model loading, context creation, tokenisation, and a full
  sampler chain (greedy / temperature / top-p / top-k / dist).
- **Model pulling**:
  - HuggingFace Hub backend (`ethllama[pull]`, `huggingface_hub`)
  - Ollama registry backend (built-in, OCI Distribution Spec v2, no auth
    required for public models on `registry.ollama.ai`; the
    `application/vnd.ollama.image.model` blob is the raw GGUF, no
    conversion needed).
- **Model quantisation** — built-in `ethllama quantize` command calling
  `llama-quantize` (11 quant types, default `q4_k_m`).
- **Embeddings endpoint** — `POST /v1/embeddings` returning deterministic
  768-dim vectors (stub until the Rust core exposes a real embedding API).
- **GPU selection flags** — `--gpu {vulkan,rocm,cuda,cpu}` and
  `--gpu-layers <N>` on the `run` command; auto-detection order
  Vulkan → ROCm → CUDA → CPU.
- **Async API** — FastAPI endpoints are async; the API layer exposes
  `async def` handlers and uses `uvicorn[standard]`.
- **Pluggable engine system** — `ethllama.engines.EngineConfig` loads
  YAML files from `~/.ethllama/engines/` and renders CLI invocations via
  Jinja2 templates; ships with examples for `llama-cpp`, `whisper-cpp`,
  and a generic custom engine.
- **Engine YAML examples** in `docs/examples/` (`llama-cpp.yaml`,
  `whisper-cpp.yaml`, `custom-engine.yaml`).
- **Nix development shell** (`flake.nix`) — reproducible dev environment
  with Rust toolchain, Python 3, maturin, cmake/ninja, openssl, and
  Vulkan headers pre-installed; auto-initialises the `llama.cpp`
  submodule on first entry.
- **Documentation** — `README.md`, `docs/USAGE.md`, `docs/PRIVACY.md`,
  `docs/CREDITS.md`, `AGENTS.md` (project memory for coding agents).
- **Python test suite** — 76 tests across `test_cli.py`, `test_api.py`,
  `test_stt.py`, covering the CLI surface, the FastAPI server, the
  whisper.cpp STT path, the model index, and the engine registry.
- **Setup / benchmark scripts** — `scripts/setup.sh` and
  `scripts/benchmark.sh` for one-command environment bring-up and
  timed inference benchmarking.

### Notes
- This is the first public release of **ethicallama**.
- Inference runs **fully locally**. The only network calls the package
  makes are to HuggingFace / Ollama when you explicitly invoke `pull`.
- Telemetry is **off by default**; the only outbound traffic is model
  pulls, and they are user-initiated.

[Unreleased]: https://github.com/luluthehungrycat/ethicallama/compare/v0.1.2...HEAD
[0.1.2]: https://github.com/luluthehungrycat/ethicallama/releases/tag/v0.1.2
[0.1.0]: https://github.com/luluthehungrycat/ethicallama/releases/tag/v0.1.0

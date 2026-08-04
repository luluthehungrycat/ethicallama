# Changelog

All notable changes to **ethicallama** are documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Made the Rust `ethllama-core` native extension optional. The
  `ethllama-core` crate now exposes a `llama-cpp` Cargo feature (on
  by default) and the build script no longer panics when the
  `llama.cpp` git submodule is missing. Instead, it emits a warning
  and the lib compiles a stub `PyLlamaModel` that loads but raises
  `RuntimeError` on instantiation. The Python side already handles
  this via `has_inference_engine()` and falls back to the subprocess
  inference path (llama-cli, ollama, etc.). This unblocks
  `pip install` and `uv tool install` from sdist on platforms where
  no pre-built wheel is available (e.g. Raspberry Pi / ARM Linux).

## [0.2.0] - 2026-07-22

### Added
- Generated systemd setup: the default user unit runs as the current user with
  a resolved executable and private user configuration; explicit system mode
  generates a unit for the invoking user/group and supplies root-owned config
  through `LoadCredential`.
- Profile propagation for inline templates, stop sequences, and token limits.
- Quote-aware custom-engine argument rendering with documented output policies.

### Fixed
- Streaming output now works without prompt echoing, preserves assistant text
  inside chat delimiters, exposes partial stdout before a newline, and keeps
  `--debug` fully raw.
- Server configuration now consistently uses CLI > config > defaults; API keys
  are read once, compared in constant time, and never persisted by startup.
- Configuration overrides through absolute `ETHLLAMA_CONFIG` fail closed when
  missing, unreadable, or malformed.

## [0.1.7] - 2026-07-11

### Fixed
- Systemd service: `ExecStart` no longer hardcodes `/usr/local/bin/ethllama`.
  Now uses bare `ethllama` with an expanded `Environment=PATH` so the
  service works for both `pip install` and `uv tool install` (binary
  at `~/.local/bin/ethllama`). INSTALL.md covers both install paths.

## [0.1.6] - 2026-07-11

### Added
- Default `ethllama serve` port changed from 8080 to **10434** (homage
  to Ollama's 11434). Override with `--port`.
- Native HTTPS support: `--ssl-keyfile`, `--ssl-certfile`,
  `--ssl-keyfile-password`, `--ssl-ca-certs` flags on `serve`.
- `contrib/systemd/ethllama.service` — turnkey systemd unit (hardened:
  NoNewPrivileges, PrivateTmp, ProtectSystem=strict, no caps).
  Plus `ethllama.env.example` and `INSTALL.md`.
- `contrib/nginx/ethllama.conf` — production reverse proxy with HTTP→HTTPS
  301, Let's Encrypt-ready, large timeouts for big models, SSE streaming.
- Production deployment guide in USAGE.md (systemd, HTTPS, CORS,
  rate limiting).
- `ethllama discover` command: scans PATH for known inference engine
  binaries (ollama, llama-cli, llama-server, whisper-cli, voxtral, etc.)
  and auto-generates engine config YAMLs in `~/.ethllama/engines/`.
  Supports `--overwrite`, `--no-generate` (dry-run), and `--engines-dir`.
  (12 new tests, 153 total.)
- Voxtral (voxtral-mini-realtime-rs) example engine configs:
  `docs/examples/voxtral-stt.yaml` and `docs/examples/voxtral-tts.yaml`.
  Set up voxtral locally and copy the YAML to `~/.ethllama/engines/`.
- llama2.c compatibility: `docs/examples/llama2-c.yaml` engine config,
  `ethllama/llama2c.py` helper with `is_llama2c_model()` and
  `find_tokenizer_for()`. Run via `ethllama run --engine llama2-c`.
- TTL / idle model unloading: `--ttl` / `--idle-timeout` flag on
  `ethllama serve`. Models idle longer than the timeout are unloaded
  from memory. Configured in `config.yaml` via `api.idle_timeout`.
  (3 new tests, 141 total.)

## [0.1.3] - 2026-07-07

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

[Unreleased]: https://github.com/luluthehungrycat/ethicallama/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/luluthehungrycat/ethicallama/releases/tag/v0.2.0
[0.1.9]: https://github.com/luluthehungrycat/ethicallama/releases/tag/v0.1.9
[0.1.2]: https://github.com/luluthehungrycat/ethicallama/releases/tag/v0.1.2
[0.1.0]: https://github.com/luluthehungrycat/ethicallama/releases/tag/v0.1.0

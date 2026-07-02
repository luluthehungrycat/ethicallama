# AGENTS.md — ethicallama Project Memory

> This file captures architecture, patterns, decisions, and constraints discovered during development.
> Update this file whenever you learn something important that future coding agents should know.

---

## Project Overview

**ethicallama** — ethical, local-only inference wrapper for llama.cpp and other engines.

- **Rust core** (`ethllama-core/`): PyO3 bindings to llama.cpp FFI
- **Python CLI/API** (`ethllama/`): Click CLI + optional FastAPI (OpenAI-compatible)
- **Linux-first, no Windows, no telemetry by default**

---

## Directory Layout

```
ethicallama/
├── ethllama-core/              # Rust crate (cdylib, PyO3)
│   ├── Cargo.toml              # Dependencies: pyo3, serde, libloading, anyhow, cmake (build-dep)
│   ├── build.rs                # Compiles llama.cpp via cmake crate (requires submodule)
│   └── src/
│       ├── lib.rs              # PyO3 #[pymodule]: exports PyLlamaModel class
│       ├── llama.rs            # FFI declarations (extern "C") matching real llama.h API
│       └── utils.rs            # GPU detection (nvidia-smi, rocm-smi, vulkaninfo)
│
├── ethllama/                   # Python package
│   ├── pyproject.toml          # maturin build config + Python deps
│   ├── __init__.py             # Re-exports from submodules
│   ├── cli.py                  # Click CLI (run, pull, list, index, config, serve, engines, quantize)
│   ├── api.py                  # FastAPI (OpenAI-compatible, opt-in)
│   ├── config.py               # Config load/save/init (~/.ethllama/config.yaml)
│   ├── engines.py              # EngineConfig with Jinja2 templating for custom engine binaries
│   ├── index.py                # Model index (~/.ethllama/index.json), path resolution
│   ├── pull.py                 # Model pulling from HuggingFace Hub (optional)
│   ├── convert.py              # Opt-in Safetensors → GGUF conversion
│   └── tests/
│       └── test_cli.py         # 14 CLI tests (pytest)
│
├── docs/
│   ├── CREDITS.md              # Dependency credits (llama.cpp, PyO3, FastAPI, whisper.cpp)
│   ├── PRIVACY.md              # No telemetry policy
│   ├── USAGE.md                # Full CLI/API/config documentation
│   └── examples/               # Example engine YAML configs
│       ├── llama-cpp.yaml
│       ├── whisper-cpp.yaml
│       ├── custom-engine.yaml
│       └── README.md
│
├── scripts/
│   ├── setup.sh                # Prerequisites + build + install
│   └── benchmark.sh            # Timed inference benchmarking
│
├── flake.nix                   # Nix development shell
├── README.md                   # Project documentation
└── AGENTS.md                   # This file
```

---

## Architecture & Data Flow

```
User CLI (click)     FastAPI (uvicorn)
       |                    |
       v                    v
   ethllama Python package
       |                    |
       v                    v
   EngineConfig (Jinja2)   OR   PyLlamaModel (Rust/PyO3)
       |                             |
       v                             v
   Custom binary (e.g., llama-cli)   llama.cpp C library
```

### Two Inference Paths

1. **Native Rust path**: `PyLlamaModel` class via PyO3 → calls llama.cpp FFI directly
2. **Engine binary path**: `EngineConfig.render_command()` → Jinja2 template → shell command to external binary

The `run` CLI command tries native first, falls back to engine binary if the model isn't loadable via Rust.

---

## Key Patterns & Conventions

### Rust Core (`ethllama-core/src/`)

- **FFI declarations go in `llama.rs`** — all `extern "C"` blocks are consolidated here, NOT in `lib.rs`
- **`llama.rs`** defines: `LlamaModel` (opaque, `#[repr(C)]` with `_private: [u8; 0]`), `LlamaModelParams` (with `Default` impl), and all unsafe FFI wrappers
- **`lib.rs`** only does: `mod llama; mod utils;`, PyO3 `#[pymodule]`, and `#[pyclass]` wrappers
- **`utils.rs`** is for system introspection: GPU detection, file system queries, etc.
- **Memory safety**: `PyLlamaModel` has a `Drop` impl that calls `llama_free`, and `unsafe impl Send` for PyO3 thread safety
- **Build**: `cmake` crate is a **build-dependency** (in `[build-dependencies]` in Cargo.toml). The `build.rs` uses the `cmake` crate to build the full llama.cpp library tree, and panics with a clear message if `llama.cpp/` submodule is missing — it does NOT auto-clone during build. Additional build-deps: `cc` for compiling any C shim sources if needed.

### Rust Core (`ethllama-core/src/`) — Current State

- **`build.rs`** uses the `cmake` crate to build llama.cpp as a static library (replaced the old `cc`-based approach that only compiled a subset of source files)
- **`llama.rs`** contains complete FFI bindings matching the real `llama.h` API:
  - 4 opaque types (`llama_model`, `llama_context`, `llama_vocab`, `llama_sampler`)
  - 6 struct types with exact `#[repr(C)]` layouts: `llama_model_params`, `llama_context_params` (~35 fields), `llama_batch`, `llama_sampler_chain_params`, `llama_token_data`, `llama_token_data_array`
  - 30+ `extern "C"` function declarations covering model loading, context creation, tokenization, decoding, and the backend sampling API
  - Safe helper functions: `load_model()`, `tokenize()`, `detokenize()`, `infer_tokens()`
- **`lib.rs`** `PyLlamaModel` implements real inference:
  - Constructor: calls `llama_backend_init()`, loads model via `llama_model_load_from_file()`, creates context via `llama_init_from_model()`
  - `infer()` method: tokenizes prompt, runs `llama_decode()` for prefill, builds a sampler chain (greedy or temp+top_p+top_k+dist), loops token-by-token with `llama_sampler_sample()` and `llama_decode()`, detokenizes output
  - `Drop`: frees context, model, and calls `llama_backend_free()`
- **Linking**: 4 static libraries — `libllama.a`, `libggml.a`, `libggml-cpu.a`, `libggml-base.a` — plus pthread, dl, stdc++
- **`llama_backend_init/llama_backend_free`** are called per-instance (not global); fine for typical single-model usage

### Known Rust Core Limitations
- `llama_backend_free()` is called once per `PyLlamaModel::drop()`. If multiple models are created in the same Python process, the backend will be freed after the first model is destroyed. For multi-model workflows, a global reference-counted backend init/free should be implemented.
- The `llama_context_params` struct alignment depends on repr(C) matching the C compiler's layout. Verified on x86_64 Linux with GCC 16.

### Python Package (`ethllama/`)

- **CLI**: click-based, single `@click.group()` entry point in `cli.py`, 8 subcommands (`run`, `pull`, `list`, `index`, `config`, `serve`, `engines`, `quantize`)
- **Config**: YAML at `~/.ethllama/config.yaml`, loaded via `config.load_config()`, defaults in `DEFAULT_CONFIG` dict
- **Engine configs**: YAML at `~/.ethllama/engines/*.yaml`, parsed by `EngineConfig` class with Jinja2 templating
- **Model index**: JSON at `~/.ethllama/index.json`, manages symlinks/hardlinks for storage efficiency
- **API**: FastAPI is **opt-in only** (extra dependency group `ethllama[api]`). Endpoints: `/v1/chat/completions`, `/v1/completions`, `/v1/models`, `/v1/embeddings`
- **Pulling**: HuggingFace Hub (`ethllama[pull]`) and Ollama registry (built-in, `requests`-based, no auth needed for public models). Ollama's `application/vnd.ollama.image.model` blob **IS a complete GGUF file** — no conversion needed.
- **Conversion**: torch+transformers is optional (`ethllama[convert]`)
- **Quantize**: `ethllama quantize <model>` command built-in (uses llama.cpp's `llama-quantize` binary, 11 quantization types supported, default q4_k_m)

### Jinja2 Template Variables (for engine configs)

Available variables in `args_template`:
- `{{ binary }}` — engine binary path
- `{{ model_path }}` — model file path
- `{{ prompt }}` — input prompt
- `{{ temperature }}`, `{{ top_p }}`, `{{ top_k }}` — sampling params
- `{{ threads }}` — CPU threads
- `{{ n_gpu_layers }}` — GPU offload layers
- `{{ gpu_backend }}` — selected GPU backend
- `{{ output }}` — output file path

### Engine Config YAML Schema

```yaml
name: engine-name          # Unique identifier
type: text|stt|tts|image   # Engine type
binary: /path/to/binary    # Executable path
args_template: "..."       # Jinja2 template for CLI args
env:                       # Environment variables (optional)
  KEY: value
pre_check: "..."           # Shell command to verify engine (optional)
supports_streaming: true|false  # Streaming support flag (optional)
model_extensions:          # Supported file extensions (optional)
  - .gguf
```

---

## Constraints & Decisions

### Inviolable Rules

1. **No Windows support** — reject any Windows-specific code
2. **No telemetry by default** — telemetry requires explicit `telemetry.enabled: true` in config. The `config.py` init flow warns users before enabling.
3. **No model duplication** — use symlinks/hardlinks for storage efficiency via model index
4. **GPU backend choice** — user-selected (Vulkan/ROCm/CUDA/CPU), auto-detected as priority fallback
5. **LLM inference is local-only** — no cloud inference calls, no API calls to external model providers

### Build Dependencies

- `llama.cpp` is a **git submodule** at `ethllama-core/llama.cpp/`
- Must run `git submodule update --init --recursive` before building
- Build script (`build.rs`) uses `cmake` crate to build the full llama.cpp library tree
- Use `cargo build --release -p ethllama-core` for the Rust crate
- Use `maturin develop` or `pip install -e .` for the Python package

### Development Setup with `uv`

`uv` is the recommended Python runtime/venv manager for this project. It works seamlessly with PyO3/maturin:

```bash
# Create a virtual environment
uv venv

# Activate it (or use `uv run` for one-off commands)
source .venv/bin/activate

# Install maturin + all deps
uv pip install maturin ".[all]"

# Build the Rust crate and install Python package in one step
maturin develop

# Or equivalently, for production-style wheel:
uv pip install -e ".[all]"
```

**Why uv works with PyO3**: `maturin develop` discovers the active Python interpreter from the venv. `uv`'s venvs are standard virtual environments — maturin sees them the same as any other venv. No special configuration needed.

Key commands during development:
- `uv run ethllama --help` — run CLI without activating venv
- `uv run pytest ethllama/tests/ -v` — run tests
- `uv pip install -e ".[all]"` — reinstall after Python-only changes
- `maturin develop --release` — rebuild Rust core

### Ollama Registry Protocol

When implementing or modifying `pull_from_ollama()`:
- **The model blob IS a GGUF file** — `application/vnd.ollama.image.model` contains the raw GGUF bytes, no conversion needed. Verify with magic bytes `b"GGUF"` (0x46554747 LE) on download.
- **OCI Distribution Spec v2** — uses `/v2/<namespace>/<repo>/manifests/<tag>` for manifest and `/v2/<namespace>/<repo>/blobs/sha256:<digest>` for blobs.
- **No auth needed** for public models on `registry.ollama.ai`.
- **Accept header required** — must set `Accept: application/vnd.docker.distribution.manifest.v2+json` on manifest GET.
- **Hostname detection** in `_parse_model_ref`: dots in the first path component distinguish hostnames (e.g., `registry.ollama.ai/library/foo`) from namespaces (`foo/bar`).
- **Content-Type bug**: manifest returns `Content-Type: text/plain; charset=utf-8` (known Ollama issue), parse body as JSON regardless.
- **307 redirect**: blob GETs redirect to Cloudflare R2 (`*.r2.cloudflarestorage.com`). Standard `requests` follows it automatically.
- **Resumable downloads**: use `.partial` temp files with `Range` headers for resume. Verify SHA-256 after completion.
- Existing reference implementations: `iven86/ollama_gguf_downloader`, `olamide226/ollama-gguf-downloader`, `leeroopedia/workflow-ollama-ollama-model-registry-operations`.

### Quantize Command

The `ethllama quantize <model>` command:
- Accepts `--type` (11 quantization types, default `q4_k_m`) and `--binary` (auto-detected from llama.cpp build dirs or PATH via `shutil.which()`)
- Resolves model from index, auto-generates output path (`<model>-<type>.gguf`)
- Runs `llama-quantize` via subprocess with progress logging

### Embeddings Endpoint

The `/v1/embeddings` endpoint in `api.py`:
- Accepts `EmbeddingRequest` model with `input` (str or list) and `model` fields
- Returns deterministic pseudo-embeddings (fixed 768-dim vector) for now. When the Rust core is wired to the API, replace with real embeddings from llama.cpp.

### Python Dependencies

- **Required**: click, pyyaml, jinja2, tqdm, requests
- **Optional (api)**: fastapi, uvicorn, pydantic
- **Optional (pull)**: huggingface_hub
- **Optional (convert)**: torch, transformers

---

## Testing

- Python tests: `pytest ethllama/tests/ -v` (14 CLI tests, covering basic CLI, pull, quantize, ollama pull)
- Rust tests: None yet (no test harness for FFI crate)
- Test the CLI via `click.testing.CliRunner` (already in test_cli.py)

---

## Project Skills

**Before every non-trivial implementation task, check `.agents/skills/` for a matching skill and load it with the `skill` tool.** Skills contain the exact workflow, code patterns, and reference files for recurring operations — loading them saves time and prevents mistakes.

| Skill | Load when... |
|-------|-------------|
| `rust-ffi-bind` | Adding new llama.cpp C API functions through Rust FFI → PyO3 (three-layer pattern: extern "C" → safe wrapper → pyclass) |
| `ollama-registry-pull` | Working with Ollama's OCI registry (pull, protocol quirks, SHA-256 verification, Content-Type bug, 307 redirects) |

If you encounter a new recurring workflow during development, write it as a skill in `.agents/skills/<name>/SKILL.md` so future agent sessions can reuse the pattern.

**Skills are living documents.** When executing a skill's workflow (or any undocumented workflow that should have had a matching skill), if you discover a new fact, edge case, gotcha, or improvement — update the `SKILL.md` to capture it. Skills degrade fast if they omit critical quirks; the first execution always finds something the skill author missed. If you found yourself doing a structured multi-step workflow that has no skill yet, write one. Future agents (and you) will benefit more from a rough-but-honest skill than from none.

## Cross-Session Context Tips

- The `engines.py` `EngineConfig` class with Jinja2 templating is the **extension mechanism** for custom engines — this is the plugin system
- When adding new subcommands, add them to `cli.py`'s `@click.group()` and re-export from `__init__.py`
- When extending the Rust FFI, always add the `extern "C"` declaration to `llama.rs`, then wrap it in a safe function, then expose via PyO3 in `lib.rs`
- The `__all__` in `__init__.py` must be kept in sync with exports
- `pyproject.toml` uses maturin, and the `[tool.maturin]` section configures how the Rust crate is built
- Always use `options = ["pyo3/extension-module"]` in `[features]` (no "pyo3" prefix for maturin — just the feature name)
- Run `pip install -e ".[all]"` to install all optional dependencies for development

## Future Roadmap (from initial Mistral Vibe discussion)

These features were discussed and are planned for post-MVP phases:

### Modality Expansion (Phase 2+)
- **whisper.cpp** — STT/TTS engine via `EngineConfig` (example YAML already in `docs/examples/whisper-cpp.yaml`)
- **stable-diffusion.cpp** — image generation (add as engine type `image`)
- General pattern: new backends = create `/path/to/engine.cpp` → write engine YAML → done

### Multi-GPU Support
- Not yet implemented in the Rust core or CLI
- Important for larger models on multi-GPU workstations

### Async Support
- Rust core could support async via `tokio` for concurrent inference
- Python side could use `asyncio` for non-blocking API endpoints
- Not yet implemented

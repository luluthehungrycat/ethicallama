# ethicallama

[![CI](https://github.com/luluthehungrycat/ethicallama/actions/workflows/ci.yml/badge.svg)](https://github.com/luluthehungrycat/ethicallama/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/ethicallama)](https://pypi.org/project/ethicallama/)
[![License](https://img.shields.io/github/license/luluthehungrycat/ethicallama)](LICENSE)
[![Python](https://img.shields.io/pypi/pyversions/ethicallama)](https://pypi.org/project/ethicallama/)

A local-first, privacy-respecting LLM inference wrapper for running large language models entirely on your own hardware.

## Features

- **Local-Only Inference**: Everything runs on your machine. No data ever leaves your computer unless you explicitly configure it otherwise.
- **Multi-Engine Support**: Use llama.cpp, whisper.cpp, or any custom inference engine via Jinja2-templated configuration.
- **Multiple GPU Backends**: Choose between Vulkan, ROCm, CUDA, or CPU inference.
- **Built-in HTTP API**: Optional FastAPI-powered REST API for remote inference.
- **Model Indexing**: Automatically discover and manage models across your configured directories.
- **Configurable Telemetry**: Telemetry is DISABLED by default. Opt-in only with explicit confirmation.

## Quick Start

### Prerequisites

- Python 3.10+
- Rust/Cargo (for building the native core)
- Git

### Installation

Pick the install method that matches your workflow. `ethicallama` is
local-only: nothing is sent to the network and no telemetry runs unless
you explicitly enable it.

#### Using pip (recommended, once on PyPI)

The standard, works-everywhere install. Wheels are pre-built for Linux
and macOS on Python 3.10+.

```bash
pip install ethicallama

# With API server support:
pip install "ethicallama[api]"

# With everything (API, HF pulling, Safetensors conversion):
pip install "ethicallama[all]"
```

#### Using uv (10-100× faster than pip)

[uv](https://docs.astral.sh/uv/) is a Rust-based Python package
manager — drop-in faster `pip`. Use it inside an existing virtualenv
or for one-off `pip`-style installs.

```bash
# Inside a uv-managed venv
uv pip install ethicallama
uv pip install "ethicallama[api]"

# Run without permanently installing (one-shot)
uv run --with ethicallama ethllama run llama3.2
```

#### Using pipx (isolated CLI, no venv management)

[pipx](https://pipx.pypa.io/) installs Python CLI tools in their own
isolated virtualenv and exposes the `ethllama` executable on your
`PATH` globally — ideal if you just want the CLI on your machine.

```bash
pipx install ethicallama

# With API server support:
pipx install "ethicallama[api]"

# Upgrade later:
pipx upgrade ethicallama
```

#### From source (latest code, full Rust core)

For the latest unreleased code, custom Rust core builds, or
contributing to the project:

```bash
git clone --recursive https://github.com/luluthehungrycat/ethicallama
cd ethicallama

# Set up a venv (uv or stdlib venv both work)
uv venv && source .venv/bin/activate
# (or:  python3 -m venv .venv && source .venv/bin/activate)

uv pip install maturin ".[all]"
# (or:  pip install maturin ".[all]")

# Build the Rust extension and install the Python package
maturin develop --release
```

After installing, initialize the user config:

```bash
ethllama config --init
```

### Optional extras

| Extra       | Adds                                                        |
|-------------|-------------------------------------------------------------|
| `[api]`     | FastAPI server (`ethllama serve`) + uvicorn + pydantic      |
| `[pull]`    | HuggingFace Hub model pulling (`ethllama pull`)             |
| `[convert]` | Safetensors → GGUF conversion (`ethllama convert`)           |
| `[all]`     | All of the above                                            |

Extras are stacked with commas, e.g. `pip install "ethicallama[api,pull]"`.

### Basic Usage

```bash
# Run a model with default settings
ethllama run ~/models/qwen2.5-7b-q4_k_m.gguf --prompt "What is the capital of France?"

# Use a specific GPU backend
ethllama run ~/models/model.gguf --gpu cuda --gpu-layers 32

# Enable the HTTP API
ethllama serve --host 127.0.0.1 --port 10434

# List all indexed models
ethllama index list

# Index a directory of models
ethllama index add ~/models
```

## CLI Reference

### Global Options

| Option | Description |
|--------|-------------|
| `--config` | Path to config file (default: `~/.ethllama/config.yaml`) |
| `--verbose` | Enable verbose logging |

### Commands

| Command | Description |
|---------|-------------|
| `run` | Run inference with a model |
| `serve` | Start the HTTP API server |
| `config` | Manage configuration |
| `index` | Manage model index |

#### `run`

```bash
ethllama run <model> [options]
```

Options:

| Option | Default | Description |
|--------|---------|-------------|
| `--prompt`, `-p` | `"Hello"` | Input prompt |
| `--temperature`, `-t` | `0.7` | Sampling temperature |
| `--top-p` | `0.95` | Top-p sampling |
| `--top-k` | `40` | Top-k sampling |
| `--threads` | `4` | Number of CPU threads |
| `--gpu` | `cpu` | GPU backend: `vulkan`, `rocm`, `cuda`, `cpu` |
| `--gpu-layers` | `0` | Number of layers offloaded to GPU |
| `--engine` | `llama-cpp` | Engine to use |
| `--output`, `-o` | `None` | Output file path |

Examples:

```bash
# Basic inference
ethllama run model.gguf --prompt "Write a poem about AI"

# With GPU acceleration
ethllama run model.gguf --gpu cuda --gpu-layers 35 --threads 8

# Using a custom engine
ethllama run model.safetensors --engine my-custom-engine

# Save output to file
ethllama run model.gguf --prompt "Translate to French: Hello" --output result.txt
```

#### `serve`

```bash
ethllama serve [options]
```

Options:

| Option | Default | Description |
|--------|---------|-------------|
| `--host` | `127.0.0.1` | Bind address |
| `--port` | `10434` | Port number |
| `--api-key` | `""` | API key for authentication |

Example:

```bash
ethllama serve --host 0.0.0.0 --port 10434 --api-key "YOUR_API_KEY"
```

### Configuration

Configuration is stored in `~/.ethllama/config.yaml`:

```yaml
gpu:
  backend: vulkan
  fallback: true
api:
  enabled: false
  host: 127.0.0.1
  port: 10434
  api_key: ""
telemetry:
  enabled: false
model_dirs:
  - /home/user/models
```

Run the interactive setup:

```bash
ethllama config --init
```

## Architecture

```
┌─────────────────────────────────────────────┐
│              ethllama (Python CLI)           │
│  ┌─────────┐ ┌──────────┐ ┌──────────────┐ │
│  │ Config  │ │  Index   │ │  Engines     │ │
│  │ Manager │ │  Manager │ │  (YAML defs) │ │
│  └────┬────┘ └────┬─────┘ └──────┬───────┘ │
└───────┼───────────┼──────────────┼─────────┘
        │           │              │
┌───────┴───────────┴──────────────┴──────────┐
│           ethllama-core (Rust/PyO3)          │
│  ┌─────────────┐  ┌──────────────────┐      │
│  │ Model Loader │  │  Inference Engine │     │
│  └──────┬──────┘  └────────┬─────────┘      │
└─────────┼──────────────────┼────────────────┘
          │                  │
┌─────────┴──────────────────┴────────────────┐
│  External Backends (llama.cpp, whisper.cpp)  │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐    │
│  │  Vulkan  │ │   ROCm   │ │   CUDA   │    │
│  └──────────┘ └──────────┘ └──────────┘    │
└─────────────────────────────────────────────┘
```

### Components

- **ethllama (Python)**: CLI interface, configuration management, model indexing, and engine orchestration using Jinja2-templated YAML engine definitions.
- **ethllama-core (Rust)**: Native model loading and inference via PyO3 bindings to llama.cpp.
- **Engines (YAML)**: Pluggable engine configurations that define how to invoke external inference binaries (llama.cpp, whisper.cpp, or custom).

## Engine Configuration

Engines are defined as YAML files placed in `~/.ethllama/engines/`. See `docs/examples/` for ready-to-use templates.

Each engine config specifies:
- The binary to invoke
- A Jinja2 args template for building CLI commands
- Environment variables
- A pre-check command for validation
- Streaming support flag
- Supported model file extensions

## Development

### Setup

```bash
# Clone and enter the project
git clone https://github.com/luluthehungrycat/ethicallama.git
cd ethicallama

# Initialize submodules (for llama.cpp dependency)
git submodule update --init --recursive

# Build the Rust core
cargo build --release -p ethllama-core

# Install the Python package in editable mode
pip install -e ".[dev]"
```

### Project Structure

```
ethicallama/
├── ethllama/                 # Python package
│   ├── __init__.py           # Package exports
│   ├── config.py             # Configuration management
│   ├── engines.py            # Engine config loading/running
│   └── index.py              # Model index management
├── ethllama-core/            # Rust core (PyO3 bindings)
│   ├── src/
│   │   ├── lib.rs            # PyO3 module definition
│   │   ├── llama.rs          # llama.cpp FFI bindings
│   │   └── utils.rs          # Utilities
│   ├── build.rs              # Build script (links llama.cpp)
│   └── Cargo.toml
├── docs/                     # Documentation
│   ├── USAGE.md              # Detailed usage guide
│   ├── PRIVACY.md            # Privacy policy
│   └── examples/             # Example engine configs
├── scripts/                  # Helper scripts
│   ├── setup.sh              # Development setup
│   └── benchmark.sh          # Simple benchmark
├── CREDITS.md                # Open-source credits
├── LICENSE                   # MIT License
└── README.md                 # This file
```

### Testing

```bash
# Run Python tests
pytest

# Run Rust tests
cargo test -p ethllama-core
```

> **Using uv?** The entire project works seamlessly with `uv` — see
> [AGENTS.md](AGENTS.md) for the full uv development workflow (venv
> creation, maturin builds, and the recommended test commands).

## Credits

ethicallama is built on the shoulders of several excellent open-source projects. See [CREDITS.md](CREDITS.md) for the full list.

Key dependencies:
- **llama.cpp** - GGUF model loading and inference
- **whisper.cpp** - Speech-to-text (planned)
- **PyO3** - Rust-Python bindings
- **FastAPI** - HTTP API server

## License

MIT License. See [LICENSE](LICENSE) for details.

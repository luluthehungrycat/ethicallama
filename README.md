# ethicallama

<!-- TODO: replace `your-org` with the actual GitHub owner once the repo is created. -->
[![CI](https://github.com/your-org/ethicallama/actions/workflows/ci.yml/badge.svg)](https://github.com/your-org/ethicallama/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/ethicallama)](https://pypi.org/project/ethicallama/)
[![License](https://img.shields.io/github/license/your-org/ethicallama)](LICENSE)
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

```bash
# Clone the repository
git clone https://github.com/luluthehungrycat/ethicallama.git
cd ethicallama

# Run the setup script
bash scripts/setup.sh

# Initialize configuration
ethllama config --init
```

### Basic Usage

```bash
# Run a model with default settings
ethllama run ~/models/qwen2.5-7b-q4_k_m.gguf --prompt "What is the capital of France?"

# Use a specific GPU backend
ethllama run ~/models/model.gguf --gpu cuda --gpu-layers 32

# Enable the HTTP API
ethllama serve --host 127.0.0.1 --port 8080

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
| `--port` | `8080` | Port number |
| `--api-key` | `""` | API key for authentication |

Example:

```bash
ethllama serve --host 0.0.0.0 --port 8080 --api-key "my-secret-key"
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
  port: 8080
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

## Credits

ethicallama is built on the shoulders of several excellent open-source projects. See [CREDITS.md](CREDITS.md) for the full list.

Key dependencies:
- **llama.cpp** - GGUF model loading and inference
- **whisper.cpp** - Speech-to-text (planned)
- **PyO3** - Rust-Python bindings
- **FastAPI** - HTTP API server

## License

MIT License. See [LICENSE](LICENSE) for details.

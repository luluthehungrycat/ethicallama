# Usage Guide

ethicallama is a local-first LLM inference wrapper with support for multiple backends, custom engines, and an optional HTTP API.

## Installation

### From Source

```bash
# Clone the repository
git clone https://github.com/luluthehungrycat/ethicallama.git
cd ethicallama

# Initialize submodules (for llama.cpp dependency)
git submodule update --init --recursive

# Build the Rust core
cargo build --release -p ethllama-core

# Install the Python package
pip install -e ".[all]"

# Initialize configuration
ethllama config --init
```

### Via pip (when available)

```bash
pip install ethicallama
```

## CLI Reference

### Global Options

```
--config PATH     Path to config file (default: ~/.ethllama/config.yaml)
--verbose         Enable verbose logging
--help            Show help message
```

### Commands

#### `run` -- Run inference

```
ethllama run <model> [options]
```

Arguments:

| Argument | Description |
|----------|-------------|
| `model`  | Path to model file or indexed model name |

Options:

| Option | Default | Description |
|--------|---------|-------------|
| `--prompt`, `-p` | `"Hello"` | Input prompt |
| `--temperature`, `-t` | `0.7` | Sampling temperature (0.0 to 2.0) |
| `--top-p` | `0.95` | Nucleus sampling threshold |
| `--top-k` | `40` | Top-k sampling count |
| `--threads` | `4` | Number of CPU threads to use |
| `--gpu` | `cpu` | GPU backend: `vulkan`, `rocm`, `cuda`, `cpu` |
| `--gpu-layers` | `0` | Number of layers to offload to GPU |
| `--engine` | `llama-cpp` | Inference engine to use |
| `--output`, `-o` | `None` | Write output to file |

Examples:

```bash
# Simple prompt
ethllama run ~/models/llama-3.2-3b-q4.gguf --prompt "Explain quantum computing in simple terms"

# High creativity
ethllama run model.gguf -p "Write a haiku" -t 1.2 --top-p 0.9

# GPU acceleration with CUDA
ethllama run model.gguf --gpu cuda --gpu-layers 32 --threads 8

# Using a specific engine
ethllama run model.safetensors --engine my-custom-engine -p "Hello"

# Save output
ethllama run model.gguf -p "Summarize this" -o summary.txt
```

#### `serve` -- Start HTTP API server

```
ethllama serve [options]
```

Options:

| Option | Default | Description |
|--------|---------|-------------|
| `--host` | `127.0.0.1` | Bind address |
| `--port` | `8080` | Port number |
| `--api-key` | `""` | API key for request authentication |

Examples:

```bash
# Local server
ethllama serve

# Network-accessible server with API key
ethllama serve --host 0.0.0.0 --port 8080 --api-key "sk-ethicallama-1234"
```

#### `config` -- Manage configuration

```
ethllama config [options]
```

Options:

| Option | Description |
|--------|-------------|
| `--init` | Run interactive configuration setup |
| `--show` | Display current configuration |
| `--reset` | Reset configuration to defaults |

Examples:

```bash
# Interactive setup
ethllama config --init

# View current config
ethllama config --show
```

#### `index` -- Manage model index

```
ethllama index <command> [options]
```

Commands:

| Command | Description |
|---------|-------------|
| `list`  | List all indexed models |
| `add`   | Add a directory to the model index |
| `scan`  | Scan configured directories for new models |
| `clear` | Clear the model index |

Examples:

```bash
# List indexed models
ethllama index list

# Add models directory
ethllama index add ~/models

# Add HuggingFace cache
ethllama index add ~/.cache/huggingface/hub

# Rescan
ethllama index scan

# Clear index
ethllama index clear
```

## Configuration

### Config File Format

Configuration is stored in `~/.ethllama/config.yaml`. Here is the full reference:

```yaml
# GPU Backend Configuration
gpu:
  backend: vulkan          # Options: vulkan, rocm, cuda, cpu
  fallback: true           # Fall back to CPU if GPU backend fails

# HTTP API Server
api:
  enabled: false           # Enable the API server on startup
  host: 127.0.0.1          # Bind address
  port: 8080               # Port number
  api_key: ""              # API key for auth (empty = no auth)

# Telemetry
telemetry:
  enabled: false           # NO telemetry by default (see PRIVACY.md)

# Model directories (for indexing)
model_dirs:
  - /home/user/models
  - /home/user/.cache/huggingface/hub
```

### Default Values

If no config file exists, ethicallama uses these defaults:

- GPU backend: `vulkan` (with CPU fallback)
- API: disabled
- Telemetry: disabled
- Model dirs: `~/models`, `~/.cache/huggingface/hub` (if they exist)

## GPU Backend Selection

### Vulkan (Default)

Works across AMD, NVIDIA, and Intel GPUs. Good performance with broad compatibility.

```bash
ethllama run model.gguf --gpu vulkan --gpu-layers 99
```

### CUDA (NVIDIA Only)

Best performance on NVIDIA hardware. Requires CUDA toolkit.

```bash
ethllama run model.gguf --gpu cuda --gpu-layers 99 --threads 8
```

### ROCm (AMD Only)

For AMD GPUs on Linux. Requires ROCm stack.

```bash
ethllama run model.gguf --gpu rocm --gpu-layers 99
```

### CPU

No GPU acceleration. Uses only CPU threads.

```bash
ethllama run model.gguf --gpu cpu --threads 16
```

## Model Management

### Discovering Models

Models are automatically discovered from configured directories:

```bash
# Scan configured directories
ethllama index scan

# Add a custom directory
ethllama index add /path/to/models

# List all discovered models
ethllama index list
```

### Supported Formats

| Format | Extension | Engine | Description |
|--------|-----------|--------|-------------|
| GGUF   | `.gguf`   | llama-cpp | Primary format for llama.cpp models |
| GGML   | `.ggml`   | whisper-cpp | Legacy format (whisper models) |
| SafeTensors | `.safetensors` | custom | HuggingFace format |
| Checkpoint | `.ckpt` | custom | PyTorch checkpoints |

### Model Directory Structure

Models are referenced by their on-disk path or by filename if indexed. Organize them however you like:

```
~/models/
  llama-3.2-3b-q4_k_m.gguf
  llama-3.2-1b-q4_0.gguf
  qwen2.5-7b-q4_k_m.gguf
  whisper-base.ggml
```

You can also use models directly from the HuggingFace cache:

```
~/.cache/huggingface/hub/
  models--meta-llama--Llama-3.2-3B/...
```

## API Usage

### Starting the Server

```bash
ethllama serve --host 127.0.0.1 --port 8080 --api-key "sk-ethicallama-1234"
```

### Endpoints

#### POST /v1/chat/completions

OpenAI-compatible chat completions endpoint.

```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-ethicallama-1234" \
  -d '{
    "model": "llama-model",
    "messages": [
      {"role": "user", "content": "What is the meaning of life?"}
    ],
    "temperature": 0.7,
    "max_tokens": 500
  }'
```

#### POST /v1/completions

Simple text completions endpoint.

```bash
curl -X POST http://localhost:8080/v1/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-ethicallama-1234" \
  -d '{
    "model": "llama-model",
    "prompt": "Once upon a time",
    "temperature": 0.8,
    "max_tokens": 200
  }'
```

#### GET /v1/models

List available models.

```bash
curl http://localhost:8080/v1/models \
  -H "Authorization: Bearer sk-ethicallama-1234"
```

#### GET /health

Health check endpoint.

```bash
curl http://localhost:8080/health
```

## Custom Engine Configuration

ethicallama supports pluggable inference engines via YAML configuration files placed in `~/.ethllama/engines/`.

### Engine Configuration Format

```yaml
name: my-engine              # Engine name (used with --engine flag)
type: text                   # Engine type: text, stt (speech-to-text), tts (text-to-speech)
binary: /usr/local/bin/run   # Path to the inference binary
args_template: >             # Jinja2 template for CLI arguments
  {{ binary }} --model {{ model_path }} --prompt "{{ prompt }}"
  {% if temperature %}--temp {{ temperature }}{% endif %}
  -t {{ threads }}
env:                         # Environment variables
  MY_VAR: "value"
pre_check: "{{ binary }} --version"   # Command to validate the binary
supports_streaming: false    # Whether the engine supports streaming output
model_extensions:            # File extensions this engine handles
  - .gguf
  - .bin
```

### Template Variables

The following variables are available in `args_template`:

| Variable | Type | Description |
|----------|------|-------------|
| `binary` | string | Path to the engine binary |
| `model_path` | string | Path to the model file |
| `prompt` | string | Input prompt text |
| `temperature` | float or None | Sampling temperature |
| `top_p` | float or None | Top-p sampling threshold |
| `top_k` | int or None | Top-k sampling count |
| `threads` | int | Number of CPU threads |
| `n_gpu_layers` | int | Number of GPU layers |
| `gpu_backend` | string | GPU backend name |
| `output` | string or None | Output file path |

### Example: Custom Sentiment Engine

```yaml
name: sentiment-engine
type: text
binary: /opt/sentiment/bin/predict
args_template: >
  {{ binary }} --checkpoint {{ model_path }} --text "{{ prompt }}"
  --threads {{ threads }}
  {% if temperature %}--temperature {{ temperature }}{% endif %}
env:
  SENTIMENT_HOME: "/opt/sentiment"
  OMP_NUM_THREADS: "{{ threads }}"
pre_check: "{{ binary }} --help"
supports_streaming: false
model_extensions:
  - .pt
  - .pth
```

See `docs/examples/` for ready-to-use engine configuration files.

## Troubleshooting

### Common Issues

#### "Failed to load model"

**Cause**: The model file is corrupt, incompatible, or the path is incorrect.
**Solution**: Verify the file exists and is a valid GGUF file for llama.cpp. Check that the model is not truncated by verifying its checksum.

#### "Engine validation failed"

**Cause**: The engine binary specified in the YAML config is missing or broken.
**Solution**: Verify the `binary` path in the engine config file. Run the `pre_check` command manually to diagnose the issue.

#### "CUDA out of memory"

**Cause**: The model is too large for your GPU's VRAM.
**Solution**: Reduce `--gpu-layers` to offload less to the GPU, or use CPU inference. Consider a quantized model (Q4_K_M, Q5_K_M, etc.).

#### "Cannot find model"

**Cause**: The model path is incorrect or the model index does not include it.
**Solution**: Use the full path to the model, or run `ethllama index add <directory>` to index it.

#### "Submodule not initialized"

**Cause**: The llama.cpp submodule was not initialized during setup.
**Solution**: Run `git submodule update --init --recursive` in the project root.

### Debugging

Enable verbose output to get more information:

```bash
ethllama --verbose run model.gguf --prompt "Hello"
```

### Log Files

Logs are written to `~/.ethllama/logs/`. Check these for detailed error information.

### Getting Help

- Open an issue on the project repository
- Check the FAQ in the project wiki
- Review the engine configuration examples in `docs/examples/`

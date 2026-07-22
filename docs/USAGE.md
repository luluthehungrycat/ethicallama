# Usage Guide

ethicallama is a local-first LLM inference wrapper with support for multiple backends, custom engines, and an optional HTTP API.

## Installation

Pick the install method that matches your workflow. `ethicallama` is
local-only: nothing is sent to the network and no telemetry runs unless
you explicitly enable it.

### Via pip (when on PyPI)

Once published to [PyPI](https://pypi.org/project/ethicallama/), the
simplest install is:

```bash
pip install ethicallama

# With API server support:
pip install "ethicallama[api]"

# With everything (API, HF pulling, Safetensors conversion):
pip install "ethicallama[all]"
```

Wheels are pre-built for Linux and macOS on Python 3.10+, so no
compiler toolchain is required.

### Via pipx (isolated CLI install)

[pipx](https://pipx.pypa.io/) installs Python CLI tools in isolated
environments, placing the `ethllama` command on your `PATH` without
requiring manual venv management:

```bash
pipx install ethicallama

# With API server support:
pipx install "ethicallama[api]"

# Upgrade later:
pipx upgrade ethicallama
```

Use pipx when you want a global `ethllama` command without polluting
system Python or maintaining a project-local venv.

### Via uv (fast package manager)

[uv](https://docs.astral.sh/uv/) is a Rust-based Python package manager
that installs dependencies 10-100× faster than pip. If your system has
`uv` installed:

```bash
# Inside a uv-managed venv
uv pip install ethicallama

# With all extras:
uv pip install "ethicallama[all]"
```

**Run without installing** (one-shot — great for trying the CLI):

```bash
uv run --with ethicallama ethllama run llama3.2
```

For the full development workflow with uv (building the Rust core,
running tests, etc.), see the "From source" section below and
[AGENTS.md](../AGENTS.md).

### From source

For the latest unreleased code, custom Rust core builds, or
contributing to the project. Requires the Rust toolchain, CMake, and a
C/C++ compiler in addition to Python.

```bash
# Clone the repository
git clone --recursive https://github.com/luluthehungrycat/ethicallama.git
cd ethicallama

# Set up a venv (uv or stdlib venv both work)
uv venv && source .venv/bin/activate
# (or:  python3 -m venv .venv && source .venv/bin/activate)

# Install build tooling + all extras
uv pip install maturin ".[all]"

# Build the Rust extension and install the Python package
maturin develop --release

# Initialize configuration
ethllama config --init
```

If you cloned without `--recursive`, run
`git submodule update --init --recursive` so the `llama.cpp` submodule
is present (the Rust build script needs it).

### Via Nix (reproducible dev shell)

The project ships a `flake.nix` that provides a reproducible dev shell with
the full Rust + Python + llama.cpp build chain pre-installed. If you are on
NixOS or have the Nix package manager installed, this is the recommended
way to develop ethicallama.

```bash
# Enter the dev shell (provides rustc, cargo, maturin, cmake, ninja,
# gcc, clang, vulkan-headers, vulkan-loader, openssl, git, etc.)
nix develop

# The shellHook auto-initializes the llama.cpp git submodule if missing.

# Build the Python extension (recommended uv workflow)
uv venv
source .venv/bin/activate
uv pip install maturin '.[all]'
maturin develop --release

# Run the test suite
pytest ethllama/tests/ -v
```

Run `nix flake show` to list the available outputs and `nix fmt` to
reformat the flake with the bundled `nixfmt-rfc-style`.

### Optional extras

| Extra       | Adds                                                        |
|-------------|-------------------------------------------------------------|
| `[api]`     | FastAPI server (`ethllama serve`) + uvicorn + pydantic      |
| `[pull]`    | HuggingFace Hub model pulling (`ethllama pull`)             |
| `[convert]` | Safetensors → GGUF conversion (`ethllama convert`)          |
| `[all]`     | All of the above                                            |

Extras are stacked with commas, e.g. `pip install "ethicallama[api,pull]"`.
They apply identically to `pip`, `uv pip`, and `pipx install` commands.

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

#### `run --interactive` -- Interactive REPL mode

```
ethllama run <model> -i [options]
```

`ethllama run` enters an interactive chat loop when invoked with `-i`/`--interactive`,
or automatically when no `--prompt` is given and stdin is a TTY.

REPL options (added to `run`):

| Option | Default | Description |
|--------|---------|-------------|
| `--interactive`, `-i` | `False` | Enter interactive REPL mode (overrides `--prompt`) |
| `--prompt-prefix` | `"> "` | REPL prompt prefix shown before each input |
| `--max-history` | `10` | Max conversation turns to keep in history |
| `--system` | `None` | Initial system prompt for the session |

Slash commands available inside the REPL:

| Command | Description |
|---------|-------------|
| `/exit`, `/quit` | Exit the REPL |
| `/clear` | Clear conversation history (keeps system prompt) |
| `/system <text>` | Set or replace the system prompt |
| `/temp <float>` | Change sampling temperature for subsequent turns |
| `/help` | Show available commands |
| `/history` | Show the current conversation history |

Input rules:

- A line ending with `\` continues to the next line (multi-line input).
- A blank line (or EOF / Ctrl+D) submits the accumulated buffer.
- A non-blank, non-continuation line submits immediately as a single-line message.
- `Ctrl+C` cancels the current input or exits the REPL gracefully.

Examples:

```bash
# Start an interactive chat session
ethllama run ~/models/llama-3.2-3b-q4.gguf -i

# REPL with a custom prompt prefix and larger history
ethllama run model.gguf -i --prompt-prefix ">>> " --max-history 20

# Start with an initial system prompt
ethllama run model.gguf -i --system "You are a helpful coding assistant."

# --interactive overrides --prompt (prompt is ignored)
ethllama run model.gguf -i -p "this prompt is ignored"
```

Welcome banner on entry:

```
ethicallama REPL — model: llama-3.2-3b-q4.gguf, type /exit to quit
Type /help for a list of commands.
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

#### `rm` -- Remove a model

```
ethllama rm <model> [options]
```

Removes a model from the index. By default, only the index entry is
removed — the file on disk is left intact. Pass `--purge` to also
delete the file from disk; because that's destructive, `rm` prompts
for confirmation unless `--yes` is given.

Arguments:

| Argument | Description |
|----------|-------------|
| `model`  | Filename (matched against the index), a path, or `~`-expanded path |

Options:

| Option | Description |
|--------|-------------|
| `--purge`, `-p` | Also delete the model file from disk |
| `--yes`, `-y`   | Skip the confirmation prompt when `--purge` is set |

Examples:

```bash
# Just remove from the index, keep the file
ethllama rm qwen3-0.8b-q4km.gguf

# Remove from the index AND delete the file (with confirmation)
ethllama rm /home/user/models/qwen3-0.8b-q4km.gguf --purge

# Skip the confirmation (scriptable)
ethllama rm /home/user/models/qwen3-0.8b-q4km.gguf --purge --yes
```

Resolution order: direct file path → `~`-expanded path → index
lookup via `resolve_model_path()`. If the model cannot be found,
`rm` exits with status 1 and prints a "not found" message.

#### `info` -- Show model metadata

```
ethllama info <model> [options]
```

Displays filesystem metadata (path, size, modification time, whether
the file is in the index) and, for `.gguf` files, the parsed GGUF
header: architecture, context length, embedding length, quantization
label, and parameter count.

Arguments:

| Argument | Description |
|----------|-------------|
| `model`  | Filename (matched against the index) or a direct path to a file |

Options:

| Option | Description |
|--------|-------------|
| `--json`         | Output the metadata as JSON (machine-readable) |
| `--verbose`, `-v` | Show every metadata KV pair in the GGUF header |

Examples:

```bash
# Human-readable summary
ethllama info qwen3-0.8b-q4km.gguf

# Direct path to a model file
ethllama info /home/user/models/qwen3-0.8b-q4km.gguf

# JSON output (pipe to jq, etc.)
ethllama info qwen3-0.8b-q4km.gguf --json

# Full metadata dump (every KV pair)
ethllama info qwen3-0.8b-q4km.gguf --verbose
```

Sample human-readable output:

```
Model: qwen3-0.8b-q4km.gguf
  Path:     /home/user/.ethllama/models/qwen3-0.8b-q4km.gguf
  Size:     524.3 MB
  Modified: 2026-06-15 14:23:01
  Indexed:  yes

  GGUF Metadata:
    Architecture:      qwen3
    Context length:    32768
    Embedding length:  1024
    GGUF version:      3
    Quantization:      Q4_K_M
    Parameters:        0.8B

  (197 tensors, 5 metadata keys)
```

For non-GGUF files, `info` still prints the basic filesystem
metadata and a "Not a GGUF file" warning — useful for inspecting
files you haven't fully identified yet.

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

### llama2.c (educational)

For Andrej Karpathy's [llama2.c](https://github.com/karpathy/llama2.c) models
(tiny stories in raw `.bin` format):

````bash
git clone https://github.com/karpathy/llama2.c.git
cd llama2.c
make run
# Download a model: stories15M.bin + tokenizer.bin
# Configure engine: cp docs/examples/llama2-c.yaml ~/.ethllama/engines/

ethllama run stories15M.bin
````


## Model profiles

A **profile** is a reusable preset of inference parameters (system
prompt, temperature, chat template, stop sequences, …) bound to a
model by name.  Profiles are the ethicallama equivalent of
[Ollama's Modelfile](https://github.com/ollama/ollama/blob/main/docs/modelfile.md),
but they do **not** copy the underlying GGUF file: the model is
referenced by index name or absolute path, and the parameters are
applied at runtime.  Switching between "Python coding" and "creative
writing" personas is just a flag change, no extra gigabytes on disk.

### Where profiles live

Profiles are stored as YAML files in `~/.ethllama/profiles/<name>.yaml`.
The directory is created on demand when you save your first profile.

### Creating a profile

```bash
ethllama profile create chat-python \
  --model Qwen3.5-0.8B-UD-IQ2_XXS \
  --temperature 0.3 \
  --top-p 0.9 \
  --top-k 30 \
  --max-tokens 2048 \
  --n-gpu-layers -1 \
  --ctx-size 4096 \
  --system-prompt "You are an expert Python developer. Use type hints." \
  --description "Default profile for Python coding help"
```

The `--model` value is either an indexed model stem (looked up in
`~/.ethllama/index.json`) or an absolute path to a GGUF file.  All
other parameters are optional.

### YAML format

The same file format is produced by `profile show` and consumed by
`profile run`, so you can also edit it directly with your favorite
editor (or `ethllama profile edit <name>`, which opens it in `$EDITOR`):

```yaml
name: chat-python
description: Default profile for Python coding help
model: Qwen3.5-0.8B-UD-IQ2_XXS   # index stem OR absolute path
parameters:
  temperature: 0.3
  top_p: 0.9
  top_k: 30
  max_tokens: 2048
  n_gpu_layers: -1
  ctx_size: 4096
system_prompt: |
  You are an expert Python developer. Always explain your code.
  Use type hints. Prefer standard library. Keep answers concise.
template: |
  <|im_start|>system
  {{ .System }}<|im_end|>
  <|im_start|>user
  {{ .Prompt }}<|im_end|>
  <|im_start|>assistant
  {{ .Response }}<|im_end|>
stop:
  - "<|im_end|>"
  - "<|im_start|>"
```

Recognised `parameters` keys mirror the `ethllama run` flags:
`temperature`, `top_p`, `top_k`, `max_tokens`, `n_gpu_layers`,
`ctx_size`, `threads`, `gpu_backend`.  Unknown keys are preserved on
round-trip but ignored by the CLI.

### Running with a profile

Two ways:

```bash
# Direct:  `ethllama run <model> --profile <name>`
ethllama run Qwen3.5-0.8B-UD-IQ2_XXS -p "How do I cache a function in Python?" --profile chat-python

# Profile-driven:  `ethllama profile run <name> --prompt ...`
# (the model is taken from the profile itself)
ethllama profile run chat-python -p "How do I cache a function in Python?"
```

The profile's parameters act as **fallbacks**: any explicit CLI flag
you pass on the command line wins.  So `ethllama run <model>
--profile chat-python --temperature 0.7` uses `0.7` for the
temperature but everything else from the profile.

The same `--profile` flag works for `ethllama serve` to pre-apply the
profile's settings to the pre-loaded model:

```bash
ethllama serve --profile chat-python --port 10434
```

### Other profile commands

```bash
ethllama profile list                # list all configured profiles
ethllama profile show <name>         # show the YAML (or --json)
ethllama profile edit <name>         # open in $EDITOR
ethllama profile delete <name>       # remove (prompts for confirmation)
```

Use `ethllama profile list --json` for machine-readable output
suitable for scripting.

### Comparison to Ollama's Modelfile

| Concern              | Ollama Modelfile               | ethicallama profile            |
|----------------------|--------------------------------|--------------------------------|
| Storage              | Separate copy of GGUF + Modelfile | One YAML referencing the model |
| Disk overhead        | 1× GGUF per Modelfile          | None (the GGUF is shared)      |
| Storage format       | Plain-text DSL                 | YAML                           |
| System prompt        | `SYSTEM`                       | `system_prompt`                |
| Chat template        | `TEMPLATE`                     | `template` (Jinja2 inline)     |
| Parameters           | `PARAMETER <key> <value>`      | `parameters:` mapping          |
| Stop sequences       | `PARAMETER stop <seq>`         | `stop:` list                   |
| Edit in $EDITOR      | `ollama edit <model>`          | `ethllama profile edit <name>`  |


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

#### `transcribe` -- Speech-to-text (whisper.cpp)

```
ethllama transcribe <audio_file> [options]
```

Transcribes an audio file (wav, mp3, m4a, ...) using a whisper.cpp engine.
The command auto-detects an installed STT engine from
`~/.ethllama/engines/`; if none is registered, it falls back to looking
for `whisper-cli` (or the legacy `main` binary) on `$PATH`.

Arguments:

| Argument | Description |
|----------|-------------|
| `audio_file` | Path to the audio file to transcribe |

Options:

| Option | Default | Description |
|--------|---------|-------------|
| `--engine`, `-e` | (auto) | Engine name from `~/.ethllama/engines/` (must have `type: stt`) |
| `--model`, `-m` | engine default | Explicit path to the whisper model file (.bin / .ggml) |
| `--output-format`, `-of` | `text` | Output format: `text`, `json`, `srt`, `vtt` |
| `--language`, `-l` | `auto` | Language code (e.g. `en`, `de`, `fr`) or `auto` |
| `--threads`, `-t` | `4` | Number of CPU threads |
| `--output`, `-o` | (stdout) | Save output to FILE instead of printing |

Examples:

```bash
# Plain text transcription
ethllama transcribe recording.wav

# Explicit model + JSON output
ethllama transcribe interview.mp3 -m ~/models/ggml-large.bin -of json

# German, save to subtitles
ethllama transcribe meeting.wav -l de -of srt -o meeting.srt

# Use a specific engine
ethllama transcribe sample.wav --engine whisper-cpp
```

##### Setting up whisper.cpp

1. Build whisper.cpp and place the `whisper-cli` binary somewhere on `$PATH`
   (e.g. `/usr/local/bin/whisper-cli`).
2. Copy the example engine config to `~/.ethllama/engines/`:
   ```bash
   cp docs/examples/whisper-cpp.yaml ~/.ethllama/engines/
   ```
3. Edit `binary` and `default_model` in the YAML to match your install.
4. Verify with `ethllama engines` -- the whisper-cpp entry should show ✓.

See `docs/examples/whisper-cpp.yaml` for a complete engine config.

### Voxtral (real-time TTS/STT)

[voxtral-mini-realtime-rs](https://github.com/TrevorS/voxtral-mini-realtime-rs)
is a Rust real-time speech engine. Single `voxtral` binary with `transcribe` (STT)
and `speak` (TTS) subcommands. Uses GGUF v3 Q4_0 models.

Pull a model and wire it up:
```bash
ethllama pull -s hf TrevorJS/voxtral-mini-realtime-gguf    # STT (~2.5 GB)
ethllama pull -s hf TrevorJS/voxtral-tts-q4-gguf          # TTS (~2.7 GB)

# Engine configs
cp docs/examples/voxtral-stt.yaml ~/.ethllama/engines/
cp docs/examples/voxtral-tts.yaml ~/.ethllama/engines/

ethllama transcribe recording.wav --engine voxtral-stt
ethllama tts "Hello" --engine voxtral-tts --output hello.wav
```

Requires a GPU (Vulkan/Metal/WebGPU). Single `voxtral` binary is shipped from
the upstream GitHub release; pre-compiled binaries for Linux/macOS/Windows
are available in the releases page.

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

## Production Deployment

### Quick start with systemd

The fastest way to run ethicallama as a network service:

1. **Install the package:**
   ```bash
   pip install ethicallama
   ```

2. **Configure the system user:**
   ```bash
   sudo useradd --system --shell /usr/sbin/nologin --home /var/lib/ethllama ethllama
   sudo mkdir -p /var/lib/ethllama /etc/ethllama
   sudo chown ethllama:ethllama /var/lib/ethllama
   ```

3. **Install the systemd service:**
   ```bash
   sudo cp contrib/systemd/ethllama.service /etc/systemd/system/
   sudo systemctl daemon-reload
   sudo systemctl enable --now ethllama
   ```

4. **Test the API:**
   ```bash
   curl http://localhost:10434/health
   ```

5. **Front it with nginx + Let's Encrypt** (optional):
   ```bash
   sudo cp contrib/nginx/ethllama.conf /etc/nginx/sites-available/
   sudo ln -s /etc/nginx/sites-available/ethllama.conf /etc/nginx/sites-enabled/
   sudo certbot --nginx -d ethllama.example.com
   sudo systemctl restart nginx
   ```

### API extras required

The HTTP server requires FastAPI dependencies. Install with:

```bash
# uv tool users
uv tool install --with fastapi --with 'uvicorn[standard]' --with pydantic ethicallama

# pip users
pip install "ethicallama[api]"
```

Without these, `ethllama serve` fails with `No module named 'fastapi'`.

### HTTPS with native uvicorn

ethicallama can serve HTTPS directly without a reverse proxy:

```bash
ethllama serve \
  --host 0.0.0.0 \
  --port 443 \
  --ssl-keyfile /etc/ethllama/server.key \
  --ssl-certfile /etc/ethllama/server.crt
```

Generate a self-signed cert for testing:
```bash
openssl req -x509 -newkey rsa:4096 -nodes -keyout server.key -out server.crt -days 365
```

### Authentication

Set an API key:
```bash
ethllama serve --api-key your-secret-here
```

Clients must send `Authorization: Bearer your-secret-here` on every request.

### Default port

ethicallama listens on port **10434** by default (an homage to Ollama's 11434). Override with `--port`.

### Idle model unloading

Use `--idle-timeout` to auto-unload models after N seconds of inactivity:
```bash
ethllama serve --idle-timeout 600
```

This saves GPU memory when running multiple models.

### Running as a public service

**CORS**: The default config doesn't enable CORS. For browser clients, set `cors_allow_origins` in `~/.ethllama/config.yaml`:
```yaml
api:
  cors_allow_origins:
    - "https://myapp.example.com"
```

**Rate limiting**: Recommended for public deployments. Use nginx's `limit_req` zone:
```nginx
limit_req_zone $binary_remote_addr zone=ethllama:10m rate=10r/s;
location / {
    limit_req zone=ethllama burst=20 nodelay;
    proxy_pass http://ethllama_backend;
}
```

**Metrics**: ethicallama exposes `/health` for liveness. For Prometheus, consider deploying an exporter.

### Getting Help

- Open an issue on the project repository
- Check the FAQ in the project wiki
- Review the engine configuration examples in `docs/examples/`

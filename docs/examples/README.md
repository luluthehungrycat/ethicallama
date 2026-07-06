# Engine Configuration Examples

This directory contains example YAML engine configuration files for ethicallama.

## What Are Engine Configs?

Engine configurations define how ethicallama invokes external inference binaries.
Each config specifies the binary path, CLI argument templates (using Jinja2),
environment variables, validation checks, and supported model file formats.

## How to Use

1. Copy the desired `.yaml` file to `~/.ethllama/engines/`:
   ```bash
   cp docs/examples/llama-cpp.yaml ~/.ethllama/engines/
   ```

2. Edit the config to match your system (update paths, adjust defaults):
   ```bash
   # Check your llama-cli path
   which llama-cli
   # Update binary path if needed
   ```

3. Verify the engine is valid:
   ```bash
   # The pre_check command in the YAML will be run automatically
   # when ethicallama loads the engine
   ethllama run --help
   ```

4. Use the engine with the `--engine` flag:
   ```bash
   ethllama run model.gguf --engine llama-cpp --prompt "Hello"
   ```

## Available Examples

| File | Type | Description |
|------|------|-------------|
| `llama-cpp.yaml` | text | Standard llama.cpp inference engine for GGUF models |
| `whisper-cpp.yaml` | stt | whisper.cpp speech-to-text engine for GGML models |
| `custom-engine.yaml` | text | Template for creating your own custom inference engine |

## Creating Your Own Engine

1. Copy `custom-engine.yaml` to `~/.ethllama/engines/`
2. Rename it (the filename becomes the engine name)
3. Fill in your binary path, args template, and settings
4. Each engine config can use Jinja2 template variables:
   - `{{ binary }}` - Engine binary path
   - `{{ model_path }}` - Model file path
   - `{{ prompt }}` - Input prompt
   - `{{ temperature }}`, `{{ top_p }}`, `{{ top_k }}` - Sampling params
   - `{{ threads }}` - CPU thread count
   - `{{ n_gpu_layers }}` - GPU layer offload count
   - `{{ gpu_backend }}` - GPU backend name
   - `{{ output }}` - Output file path

## Directory Structure

Engine configs should be placed at `~/.ethllama/engines/`:

```
~/.ethllama/
  config.yaml              # Main ethicallama config
  engines/
    llama-cpp.yaml         # llama.cpp engine
    whisper-cpp.yaml       # whisper.cpp engine
    my-custom-engine.yaml  # Your custom engines
```

Each `.yaml` file in this directory is automatically loaded by ethicallama.

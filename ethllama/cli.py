"""Main CLI entrypoint for ethicallama using click."""

import sys
import os
import time
import json
import subprocess
from pathlib import Path
from typing import Optional

import click

from .config import load_config, init_config
from .index import load_index, add_to_index, resolve_model_path
from .engines import load_engines, EngineConfig

CONTEXT_SETTINGS = dict(help_option_names=["-h", "--help"])


@click.group(context_settings=CONTEXT_SETTINGS)
@click.version_option(version="0.1.0", prog_name="ethllama")
def main():
    """ethicallama - ethical, local-only LLM inference wrapper."""
    pass


def _simulate_inference(
    prompt: str,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 40,
    stream: bool = False,
    output: Optional[str] = None,
) -> str:
    """Placeholder inference until Rust core is connected.

    Simulates a model generating a response token by token.
    """
    simulated_response = (
        f"This is a simulated response from ethicallama.\n\n"
        f"Your prompt was: {prompt}\n\n"
        f"The inference engine is not yet connected, so this is a placeholder. "
        f"Once the Rust core (ethllama-core) is built and linked, "
        f"actual llama.cpp inference will run here.\n\n"
        f"Settings: temperature={temperature}, top_p={top_p}, top_k={top_k}"
    )

    if stream:
        # Simulate token-by-token streaming
        words = simulated_response.split()
        for i, word in enumerate(words):
            click.echo(word, nl=False)
            if i < len(words) - 1:
                click.echo(" ", nl=False)
            time.sleep(0.02)  # Simulated delay
        click.echo()
    elif output:
        os.makedirs(os.path.dirname(os.path.abspath(output)) or ".", exist_ok=True)
        with open(output, "w") as f:
            f.write(simulated_response)
        click.echo(f"Output saved to {output}")
    else:
        click.echo(simulated_response)

    return simulated_response


@main.command()
@click.argument("model")
@click.option("--prompt", "-p", default="Hello", show_default=True, help="Input prompt")
@click.option("--temperature", "-t", default=0.7, show_default=True, type=float, help="Sampling temperature")
@click.option("--top-p", default=0.9, show_default=True, type=float, help="Top-p sampling")
@click.option("--top-k", default=40, show_default=True, type=int, help="Top-k sampling")
@click.option("--threads", default=4, show_default=True, type=int, help="Number of CPU threads")
@click.option("--n-gpu-layers", default=0, show_default=True, type=int, help="Number of layers to offload to GPU")
@click.option("--gpu-backend", default="auto", show_default=True, type=str, help="GPU backend (vulkan, rocm, cuda, auto)")
@click.option("--engine", "-e", default=None, type=str, help="Use a custom engine from ~/.ethllama/engines/")
@click.option("--output", "-o", default=None, type=str, help="Save output to file")
@click.option("--stream", "-s", is_flag=True, default=False, help="Stream output token by token")
def run(
    model: str,
    prompt: str,
    temperature: float,
    top_p: float,
    top_k: int,
    threads: int,
    n_gpu_layers: int,
    gpu_backend: str,
    engine: Optional[str],
    output: Optional[str],
    stream: bool,
):
    """Run inference with a model.

    MODEL is a model filename, path, or identifier from the index.
    """
    # 1. Resolve model path from index
    model_path = resolve_model_path(model)
    if not model_path:
        # Try as direct path
        if os.path.exists(model):
            model_path = os.path.abspath(model)
        else:
            click.echo(f"Error: Model '{model}' not found in index and is not a valid path.", err=True)
            click.echo("Use 'ethllama index <directory>' to index models, or provide a full path.", err=True)
            sys.exit(1)

    click.echo(f"Model: {model_path}")

    # 2. Load config for GPU settings
    config = load_config()
    if gpu_backend == "auto":
        gpu_backend = config.get("gpu", {}).get("backend", "cpu")

    click.echo(f"GPU backend: {gpu_backend}")

    # 3. If --engine specified, use EngineConfig.render_command
    if engine:
        engines = load_engines()
        if engine not in engines:
            available = list(engines.keys())
            click.echo(f"Error: Engine '{engine}' not found.", err=True)
            if available:
                click.echo(f"Available engines: {', '.join(available)}", err=True)
            else:
                click.echo("No engines installed. Check ~/.ethllama/engines/", err=True)
            sys.exit(1)

        engine_config = engines[engine]
        click.echo(f"Engine: {engine_config.name} ({engine_config.type})")

        cmd = engine_config.render_command(
            model_path=model_path,
            prompt=prompt,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            threads=threads,
            n_gpu_layers=n_gpu_layers,
            gpu_backend=gpu_backend,
            output=output,
        )

        click.echo(f"Running: {' '.join(cmd)}")
        try:
            if stream and engine_config.supports_streaming:
                proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                for line in iter(proc.stdout.readline, ""):
                    click.echo(line, nl=False)
                proc.wait()
                if proc.returncode != 0:
                    stderr = proc.stderr.read()
                    click.echo(f"Engine error: {stderr}", err=True)
                    sys.exit(1)
            else:
                result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                if result.stdout:
                    click.echo(result.stdout)
                if output:
                    with open(output, "w") as f:
                        f.write(result.stdout)
                    click.echo(f"Output saved to {output}")
        except subprocess.CalledProcessError as e:
            click.echo(f"Engine failed (exit {e.returncode}): {e.stderr}", err=True)
            sys.exit(1)
        return

    # 4. Otherwise use placeholder simulation
    _simulate_inference(
        prompt=prompt,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        stream=stream,
        output=output,
    )


@main.command()
@click.argument("model")
@click.option("--source", "-s", default="hf", show_default=True,
              type=click.Choice(["hf", "ollama"], case_sensitive=False),
              help="Source to pull from")
@click.option("--revision", "-r", default="main", show_default=True,
              help="Revision/branch for Hugging Face models")
def pull(model: str, source: str, revision: str):
    """Download a model from Hugging Face or Ollama registry."""
    from .pull import pull_model

    try:
        result = pull_model(model, source=source, revision=revision)
        click.echo(f"Model downloaded to: {result}")
    except ImportError as e:
        click.echo(f"Error: {e}", err=True)
        click.echo("Install optional dependencies: pip install ethllama[pull]", err=True)
        sys.exit(1)
    except NotImplementedError as e:
        click.echo(f"Not yet supported: {e}", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"Failed to pull model: {e}", err=True)
        sys.exit(1)


@click.option("--dir", "-d", "dir_path", default=None, type=str,
              help="List models in a specific directory instead of the index")
@main.command(name="list")
def list_models(dir_path: Optional[str]):
    """List indexed models or models in a specific directory."""
    if dir_path:
        dir_path = os.path.expanduser(dir_path)
        if not os.path.isdir(dir_path):
            click.echo(f"Error: Directory not found: {dir_path}", err=True)
            sys.exit(1)
        gguf_files = list(Path(dir_path).glob("*.gguf"))
        if not gguf_files:
            click.echo(f"No .gguf files found in {dir_path}")
            return
        click.echo(f"Models in {dir_path}:")
        for f in sorted(gguf_files):
            size_mb = f.stat().st_size / (1024 * 1024)
            click.echo(f"  {f.name} ({size_mb:.1f} MB)")
    else:
        index = load_index()
        if not index:
            click.echo("No models indexed.")
            click.echo("Use 'ethllama index <directory>' to index models.")
            return
        click.echo("Indexed models:")
        total = 0
        for dir_path, models in sorted(index.items()):
            click.echo(f"  {dir_path}/")
            for model in models:
                size_mb = model.get("size", 0) / (1024 * 1024)
                click.echo(f"    {model['filename']} ({size_mb:.1f} MB)")
                total += 1
        click.echo(f"\nTotal: {total} model(s)")


@main.command()
@click.argument("directory", default=".", required=False)
def index(directory: str):
    """Index .gguf models in a directory."""
    directory = os.path.expanduser(directory)
    if not os.path.isdir(directory):
        click.echo(f"Error: Directory not found: {directory}", err=True)
        sys.exit(1)

    gguf_files = list(Path(directory).rglob("*.gguf"))
    if not gguf_files:
        click.echo(f"No .gguf files found in {directory}")
        return

    added = 0
    for f in gguf_files:
        try:
            add_to_index(str(f))
            added += 1
        except Exception as e:
            click.echo(f"Warning: Could not index {f}: {e}", err=True)

    click.echo(f"Indexed {added} model(s) from {directory}")


@main.command()
@click.option("--init", "do_init", is_flag=True, default=False,
              help="Run interactive setup wizard")
def config(do_init: bool):
    """Show current configuration or run setup wizard."""
    if do_init:
        init_config()
        return

    cfg = load_config()
    click.echo("Current configuration:")
    click.echo(json.dumps(cfg, indent=2, default=str))
    click.echo(f"\nConfig file: {Path.home() / '.ethllama' / 'config.yaml'}")


@main.command()
@click.option("--host", default="127.0.0.1", show_default=True, help="Host to bind to")
@click.option("--port", "-p", default=8080, show_default=True, type=int, help="Port to listen on")
@click.option("--api-key", default="", help="API key for authentication")
def serve(host: str, port: int, api_key: str):
    """Start the HTTP API server (FastAPI, opt-in)."""
    try:
        from .api import run_server
    except ImportError as e:
        click.echo(f"Error: {e}", err=True)
        click.echo("Install API dependencies: pip install ethllama[api]", err=True)
        sys.exit(1)

    click.echo(f"Starting ethicallama API on http://{host}:{port}")
    if api_key:
        click.echo("API key authentication enabled")
    click.echo("Press Ctrl+C to stop")
    try:
        run_server(host=host, port=port, api_key=api_key)
    except KeyboardInterrupt:
        click.echo("\nServer stopped.")


@main.command()
def engines():
    """List installed engines from ~/.ethllama/engines/."""
    engines_dir = Path.home() / ".ethllama" / "engines"
    if not engines_dir.exists():
        click.echo(f"No engines directory found at {engines_dir}")
        click.echo("Create engine config files (*.yaml) there to register custom engines.")
        return

    yaml_files = list(engines_dir.glob("*.yaml"))
    if not yaml_files:
        click.echo(f"No engine configs found in {engines_dir}")
        return

    engines_map = load_engines()
    if not engines_map:
        click.echo("Engines found but none passed validation:")
        for yf in yaml_files:
            click.echo(f"  {yf.name} (invalid or missing binary)")
        return

    click.echo("Installed engines:")
    for name, eng in sorted(engines_map.items()):
        status = "✓" if eng.validate() else "✗"
        click.echo(f"  {status} {name} ({eng.type})")
        click.echo(f"      binary: {eng.binary}")
        click.echo(f"      streaming: {eng.supports_streaming}")
        if eng.model_extensions:
            click.echo(f"      extensions: {', '.join(eng.model_extensions)}")


@main.command()
@click.argument("model")
@click.option("--output", "-o", default=None, type=str, help="Output path for quantized model")
@click.option("--type", "-t", "quantize_type", default="q4_k_m", show_default=True,
              type=click.Choice(["q4_0", "q4_1", "q4_k_m", "q4_k_s", "q5_0", "q5_1", "q5_k_m", "q5_k_s", "q6_k", "q8_0", "f16"], case_sensitive=False),
              help="Quantization type")
@click.option("--binary", default=None, type=str, help="Path to llama.cpp quantize binary")
def quantize(model: str, output: Optional[str], quantize_type: str, binary: Optional[str]):
    """Quantize a model to a smaller precision format.

    MODEL is a model name (from the index) or a path to a GGUF file.
    """
    # 1. Resolve model path
    model_path = resolve_model_path(model)
    if not model_path:
        if os.path.exists(model):
            model_path = os.path.abspath(model)
        else:
            click.echo(f"Error: Model '{model}' not found. Use 'ethllama index <dir>' to index models or provide a full path.", err=True)
            sys.exit(1)

    # 2. Resolve quantize binary
    if binary:
        quantize_binary = binary
    else:
        # Auto-detect from common build locations
        script_dir = Path(__file__).resolve().parent
        project_root = script_dir.parent

        candidates = [
            project_root / "ethllama-core" / "llama.cpp" / "build" / "bin" / "quantize",
            project_root / "ethllama-core" / "build" / "llama.cpp-build" / "bin" / "quantize",
        ]
        quantize_binary = None
        for candidate in candidates:
            if candidate.exists():
                quantize_binary = str(candidate)
                break

        if not quantize_binary:
            import shutil
            quantize_binary = shutil.which("quantize")

    # Verify binary exists
    if not quantize_binary or not os.path.exists(quantize_binary):
        click.echo("Error: quantize binary not found. Build llama.cpp first or specify --binary.", err=True)
        sys.exit(1)

    # 3. Auto-generate output path if not provided
    if not output:
        model_path_obj = Path(model_path)
        stem = model_path_obj.stem
        output = str(model_path_obj.parent / f"{stem}-{quantize_type}.gguf")

    # 4. Log and run
    click.echo(f"Quantizing {model_path} -> {output} (type: {quantize_type})")

    try:
        result = subprocess.run(
            [quantize_binary, model_path, output, quantize_type.upper()],
            capture_output=True, text=True, check=True,
        )
        if result.stdout:
            click.echo(result.stdout)
        if result.stderr:
            click.echo(result.stderr, err=True)

        output_size = os.path.getsize(output) / (1024 * 1024)
        click.echo(f"Quantization complete: {output} ({output_size:.1f} MB)")
    except subprocess.CalledProcessError as e:
        click.echo(f"Quantization failed (exit {e.returncode})", err=True)
        if e.stderr:
            click.echo(e.stderr, err=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

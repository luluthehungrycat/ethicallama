"""Main CLI entrypoint for ethicallama using click."""

import sys
import os
import time
import json
import struct
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any

import click

from .config import load_config, init_config
from .index import load_index, add_to_index, resolve_model_path, remove_from_index, find_in_index
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


# ---------------------------------------------------------------------------
# GGUF header parser (stdlib only, no gguf-py dependency required)
# ---------------------------------------------------------------------------

# GGUF value types
_GGUF_TYPE_UINT8 = 0
_GGUF_TYPE_INT8 = 1
_GGUF_TYPE_UINT16 = 2
_GGUF_TYPE_INT16 = 3
_GGUF_TYPE_UINT32 = 4
_GGUF_TYPE_INT32 = 5
_GGUF_TYPE_FLOAT32 = 6
_GGUF_TYPE_BOOL = 7
_GGUF_TYPE_STRING = 8
_GGUF_TYPE_ARRAY = 9
_GGUF_TYPE_UINT64 = 10
_GGUF_TYPE_INT64 = 11
_GGUF_TYPE_FLOAT64 = 12

_GGUF_MAGIC = b"GGUF"


def _read_gguf_metadata(path: str, max_bytes: int = 4096) -> Optional[Dict[str, Any]]:
    """Read GGUF header metadata from the first *max_bytes* of a file.

    Uses ``gguf`` Python package when available for full accuracy;
    otherwise falls back to a minimal stdlib parser that handles the
    common value types.  Returns ``None`` when the file is not a valid
    GGUF file.
    """
    # --- try the full gguf library first ---
    try:
        import gguf  # type: ignore[import-untyped]
        reader = gguf.GGUFReader(str(path), "r")
        metadata: Dict[str, Any] = {}
        for field in reader.fields.values():
            # Skip internal / array fields we don't care about
            if field.name.startswith("tensor"):
                continue
            vals = field.parts[field.data]
            if len(vals) == 1:
                metadata[field.name] = vals[0].tolist() if hasattr(vals[0], "tolist") else vals[0]
            else:
                metadata[field.name] = [v.tolist() if hasattr(v, "tolist") else v for v in vals]
        metadata["__tensor_count"] = reader.tensor_count
        metadata["__metadata_kv_count"] = reader.metadata_kv_count
        return metadata
    except Exception:
        pass

    # --- fallback: stdlib binary parser ---
    try:
        with open(path, "rb") as f:
            header = f.read(max_bytes)
    except OSError:
        return None

    if len(header) < 12 or header[:4] != _GGUF_MAGIC:
        return None

    offset = 0

    def _u32() -> int:
        nonlocal offset
        val = struct.unpack_from("<I", header, offset)[0]
        offset += 4
        return val

    def _u64() -> int:
        nonlocal offset
        val = struct.unpack_from("<Q", header, offset)[0]
        offset += 8
        return val

    def _f32() -> float:
        nonlocal offset
        val = struct.unpack_from("<f", header, offset)[0]
        offset += 4
        return val

    def _read_string() -> Optional[str]:
        nonlocal offset
        if offset + 8 > len(header):
            return None
        slen = _u64()
        if offset + slen > len(header):
            return None
        s = header[offset:offset + slen].decode("utf-8", errors="replace")
        offset += slen
        return s

    def _skip_value(vtype: int) -> bool:
        nonlocal offset
        if vtype in (_GGUF_TYPE_UINT8, _GGUF_TYPE_INT8, _GGUF_TYPE_BOOL):
            offset += 1
        elif vtype in (_GGUF_TYPE_UINT16, _GGUF_TYPE_INT16):
            offset += 2
        elif vtype in (_GGUF_TYPE_UINT32, _GGUF_TYPE_INT32, _GGUF_TYPE_FLOAT32):
            offset += 4
        elif vtype in (_GGUF_TYPE_UINT64, _GGUF_TYPE_INT64, _GGUF_TYPE_FLOAT64):
            offset += 8
        elif vtype == _GGUF_TYPE_STRING:
            if offset + 8 > len(header):
                return False
            slen = struct.unpack_from("<Q", header, offset)[0]
            offset += 8 + slen
        elif vtype == _GGUF_TYPE_ARRAY:
            if offset + 4 > len(header):
                return False
            atype = struct.unpack_from("<I", header, offset)[0]
            offset += 4
            alen = _u64()
            for _ in range(alen):
                if not _skip_value(atype):
                    return False
        else:
            return False
        return True

    def _read_value(vtype: int) -> Any:
        nonlocal offset
        if vtype == _GGUF_TYPE_UINT8:
            val = header[offset]
            offset += 1
            return val
        elif vtype == _GGUF_TYPE_INT8:
            val = struct.unpack_from("<b", header, offset)[0]
            offset += 1
            return val
        elif vtype == _GGUF_TYPE_BOOL:
            val = bool(header[offset])
            offset += 1
            return val
        elif vtype == _GGUF_TYPE_UINT16:
            val = struct.unpack_from("<H", header, offset)[0]
            offset += 2
            return val
        elif vtype == _GGUF_TYPE_INT16:
            val = struct.unpack_from("<h", header, offset)[0]
            offset += 2
            return val
        elif vtype == _GGUF_TYPE_UINT32:
            return _u32()
        elif vtype == _GGUF_TYPE_INT32:
            val = struct.unpack_from("<i", header, offset)[0]
            offset += 4
            return val
        elif vtype == _GGUF_TYPE_FLOAT32:
            return _f32()
        elif vtype == _GGUF_TYPE_UINT64:
            return _u64()
        elif vtype == _GGUF_TYPE_INT64:
            val = struct.unpack_from("<q", header, offset)[0]
            offset += 8
            return val
        elif vtype == _GGUF_TYPE_FLOAT64:
            val = struct.unpack_from("<d", header, offset)[0]
            offset += 8
            return val
        elif vtype == _GGUF_TYPE_STRING:
            return _read_string()
        elif vtype == _GGUF_TYPE_ARRAY:
            if offset + 4 > len(header):
                return None
            atype = struct.unpack_from("<I", header, offset)[0]
            offset += 4
            alen = _u64()
            arr = []
            for _ in range(alen):
                arr.append(_read_value(atype))
            return arr
        return None

    try:
        version = _u32()
        tensor_count = _u64()
        metadata_kv_count = _u64()

        result: Dict[str, Any] = {
            "magic": "GGUF",
            "version": version,
            "__tensor_count": tensor_count,
            "__metadata_kv_count": metadata_kv_count,
        }

        for _ in range(metadata_kv_count):
            key = _read_string()
            if key is None:
                break
            if offset + 4 > len(header):
                break
            vtype = struct.unpack_from("<I", header, offset)[0]
            offset += 4
            val = _read_value(vtype)
            if val is not None:
                result[key] = val

        return result
    except (struct.error, IndexError):
        return None


_REPL_HELP = """REPL commands:
  /exit, /quit          Exit the REPL
  /clear                Clear conversation history (keeps system prompt)
  /system <text>        Set or replace the system prompt
  /temp <float>         Change sampling temperature
  /history              Show conversation history
  /help                 Show this help message

Input:
  Lines ending with \\ continue to the next line
  Empty line (or EOF / Ctrl+D) sends the accumulated message
  Ctrl+C cancels the current input or exits the REPL
"""


def _read_repl_input(prompt_prefix: str) -> Optional[str]:
    r"""Read multi-line REPL input from stdin.

    Behaviour:
    - Lines ending with ``\`` are treated as continuations; the next
      line is appended to the buffer.
    - A blank line submits the accumulated buffer.
    - A non-blank line that does not end with ``\`` is treated as a
      complete single-line message and submitted immediately.
    - ``EOFError`` (Ctrl+D) or ``KeyboardInterrupt`` (Ctrl+C) with a
      non-empty buffer submits the buffer; with an empty buffer the
      caller should treat the result as "exit the REPL".

    Returns the submitted message string, or ``None`` to signal exit.
    """
    lines: list = []
    first = True
    while True:
        try:
            line = input(prompt_prefix if first else "...")
        except (EOFError, KeyboardInterrupt):
            if lines:
                return "\n".join(lines)
            return None

        first = False

        if line.endswith("\\"):
            lines.append(line[:-1])
            continue

        if line == "":
            if lines:
                return "\n".join(lines)
            # Empty input with nothing accumulated: re-prompt.
            first = True
            continue

        # Non-blank, non-continuation line: single-line message, submit now.
        lines.append(line)
        return "\n".join(lines)


def _run_repl_loop(
    model_path: str,
    model_name: str,
    *,
    system_prompt: Optional[str],
    temperature: float,
    top_p: float,
    top_k: int,
    max_tokens: int,
    threads: int,
    n_gpu_layers: int,
    ctx_size: int,
    max_history: int,
    prompt_prefix: str,
) -> None:
    """Drive the interactive REPL session.

    The CLI owns all I/O (stdin/stdout).  :class:`REPLSession` from
    :mod:`.inference` owns the conversation state and streaming.
    """
    from .inference import REPLSession, has_inference_engine

    click.echo(f"ethicallama REPL — model: {model_name}, type /exit to quit")
    if not has_inference_engine():
        click.echo(
            "Warning: no inference engine found — using simulated responses.",
            err=True,
        )
    click.echo("Type /help for a list of commands.")
    click.echo()

    session = REPLSession(
        model_path=model_path,
        initial_system=system_prompt,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_tokens=max_tokens,
        n_gpu_layers=n_gpu_layers,
        n_threads=threads,
        ctx_size=ctx_size,
        max_history=max_history,
    )

    try:
        while True:
            try:
                user_input = _read_repl_input(prompt_prefix)
            except (EOFError, KeyboardInterrupt):
                click.echo()
                click.echo("Goodbye!")
                return

            if user_input is None:
                click.echo("Goodbye!")
                return

            stripped = user_input.strip()
            if not stripped:
                continue

            # Slash commands
            if stripped.startswith("/"):
                parts = stripped[1:].split(maxsplit=1)
                cmd = parts[0].lower()
                arg = parts[1] if len(parts) > 1 else ""

                if cmd in ("exit", "quit"):
                    click.echo("Goodbye!")
                    return
                if cmd == "help":
                    click.echo(_REPL_HELP)
                    continue
                if cmd == "clear":
                    session.clear_history()
                    click.echo("History cleared.")
                    continue
                if cmd == "system":
                    if not arg:
                        click.echo("Usage: /system <text>", err=True)
                        continue
                    session.set_system(arg)
                    preview = arg if len(arg) <= 80 else arg[:77] + "..."
                    click.echo(f"System prompt set: {preview}")
                    continue
                if cmd == "temp":
                    try:
                        new_temp = float(arg)
                    except ValueError:
                        click.echo(f"Invalid temperature: {arg!r}", err=True)
                        continue
                    session.set_temperature(new_temp)
                    click.echo(f"Temperature set to {new_temp}")
                    continue
                if cmd == "history":
                    snap = session.get_history_snapshot()
                    if not snap:
                        click.echo("(no history yet)")
                    else:
                        for entry in snap:
                            role = entry.get("role", "?")
                            content = entry.get("content", "")
                            preview = content if len(content) <= 200 else content[:197] + "..."
                            click.echo(f"[{role}] {preview}")
                    continue

                click.echo(
                    f"Unknown command: /{cmd}. Type /help for available commands.",
                    err=True,
                )
                continue

            # Regular user turn: stream the assistant response
            click.echo()  # blank line before assistant reply
            try:
                for chunk in session.send(stripped):
                    click.echo(chunk, nl=False)
            except KeyboardInterrupt:
                click.echo()
                click.echo("[interrupted]", err=True)
                continue
            except Exception as exc:  # noqa: BLE001 — surface to user
                click.echo()
                click.echo(f"Error: {exc}", err=True)
                continue
            click.echo()
            click.echo()
    except KeyboardInterrupt:
        click.echo()
        click.echo("Goodbye!")


# ---------------------------------------------------------------------------
# CLI commands
# ---------------------------------------------------------------------------

@main.command()
@click.argument("model")
@click.option("--prompt", "-p", default=None, show_default=True, help="Input prompt (omit to enter REPL when stdin is a TTY)")
@click.option("--temperature", "-t", default=0.7, show_default=True, type=float, help="Sampling temperature")
@click.option("--top-p", default=0.9, show_default=True, type=float, help="Top-p sampling")
@click.option("--top-k", default=40, show_default=True, type=int, help="Top-k sampling")
@click.option("--threads", default=4, show_default=True, type=int, help="Number of CPU threads")
@click.option("--n-gpu-layers", default=0, show_default=True, type=int, help="Number of layers to offload to GPU")
@click.option("--gpu-backend", default="auto", show_default=True, type=str, help="GPU backend (vulkan, rocm, cuda, auto)")
@click.option("--engine", "-e", default=None, type=str, help="Use a custom engine from ~/.ethllama/engines/")
@click.option("--output", "-o", default=None, type=str, help="Save output to file")
@click.option("--stream", "-s", is_flag=True, default=False, help="Stream output token by token")
@click.option("--max-tokens", "-n", default=2048, show_default=True, type=int,
              help="Maximum tokens to generate")
@click.option("--interactive", "-i", is_flag=True, default=False,
              help="Enter interactive REPL mode (overrides --prompt)")
@click.option("--prompt-prefix", default="> ", show_default=True,
              help="REPL prompt prefix (only used with --interactive)")
@click.option("--max-history", default=10, show_default=True, type=int,
              help="Max conversation turns to keep in REPL history")
@click.option("--system", "system_prompt", default=None, type=str,
              help="Initial system prompt for REPL mode")
def run(
    model: str,
    prompt: Optional[str],
    temperature: float,
    top_p: float,
    top_k: int,
    threads: int,
    n_gpu_layers: int,
    gpu_backend: str,
    engine: Optional[str],
    output: Optional[str],
    stream: bool,
    max_tokens: int,
    interactive: bool,
    prompt_prefix: str,
    max_history: int,
    system_prompt: Optional[str],
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

    # 3. Determine mode: REPL vs one-shot
    #    -i always wins (explicit user intent)
    #    Otherwise enter REPL automatically when no --prompt was given
    #    AND stdin is a TTY (so piped/captured stdin still errors out).
    enter_repl = interactive or (prompt is None and sys.stdin.isatty())
    if enter_repl:
        if prompt is not None:
            # -i wins: ignore --prompt and enter REPL.
            click.echo(
                "Note: --prompt ignored because --interactive was set.",
                err=True,
            )
        _run_repl_loop(
            model_path=model_path,
            model_name=os.path.basename(model_path),
            system_prompt=system_prompt,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_tokens=max_tokens,
            threads=threads,
            n_gpu_layers=n_gpu_layers,
            ctx_size=0,
            max_history=max_history,
            prompt_prefix=prompt_prefix,
        )
        return

    if prompt is None:
        # No prompt, no TTY, no -i: explicit error.
        click.echo(
            "Error: no prompt provided. Use --prompt/-p, or --interactive/-i "
            "for REPL mode (stdin must be a TTY).",
            err=True,
        )
        sys.exit(1)

    # 4. If --engine specified, use EngineConfig.render_command
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
                assert proc.stdout is not None
                assert proc.stderr is not None
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

    # 5. Otherwise use the built inference engine
    from .inference import run_inference, run_inference_stream, has_inference_engine

    if not has_inference_engine():
        click.echo("Warning: No inference engine found. Using simulated output.", err=True)
        _simulate_inference(
            prompt=prompt,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            stream=stream,
            output=output,
        )
        return

    if stream:
        click.echo(f"Streaming output for {model_path} ...")
        chunks = []
        for chunk in run_inference_stream(
            model_path=model_path,
            prompt=prompt,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_tokens=max_tokens,
            n_gpu_layers=n_gpu_layers,
            n_threads=threads,
        ):
            click.echo(chunk, nl=False)
            if output:
                chunks.append(chunk)
        click.echo()
        if output:
            with open(output, "w") as f:
                f.write("".join(chunks))
            click.echo(f"Output saved to {output}")
    else:
        result = run_inference(
            model_path=model_path,
            prompt=prompt,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_tokens=max_tokens,
            n_gpu_layers=n_gpu_layers,
            n_threads=threads,
        )
        click.echo(result)
        if output:
            with open(output, "w") as f:
                f.write(result)
            click.echo(f"Output saved to {output}")


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
                size_mb = int(model.get("size", 0)) / (1024 * 1024)
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
@click.option("--n-gpu-layers", default=0, show_default=True, type=int, help="Layers to offload to GPU")
@click.option("--gpu-backend", default="auto", show_default=True, type=str, help="GPU backend (vulkan, rocm, cuda, auto)")
@click.option("--threads", default=4, show_default=True, type=int, help="CPU thread count")
def serve(host: str, port: int, api_key: str, n_gpu_layers: int, gpu_backend: str, threads: int):
    """Start the HTTP API server (FastAPI, opt-in)."""
    from .inference import set_gpu_config
    set_gpu_config(n_gpu_layers=n_gpu_layers, gpu_backend=gpu_backend, n_threads=threads)

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


@main.command()
@click.argument("model")
@click.option("--purge", is_flag=True, default=False,
              help="Also delete the model file from disk")
def rm(model: str, purge: bool):
    """Remove a model from the index (and optionally delete the file)."""
    # 1. Resolve model path
    model_path = resolve_model_path(model)
    if not model_path:
        if os.path.exists(model):
            model_path = os.path.abspath(model)
        else:
            click.echo(f"Error: Model '{model}' not found in index.", err=True)
            click.echo("Use 'ethllama list' to see indexed models.", err=True)
            sys.exit(1)

    name = os.path.basename(model_path)

    # 2. Remove from index
    removed = remove_from_index(model_path)
    if removed:
        click.echo(f"Removed {name} from index.")
    else:
        click.echo(f"Warning: {name} was not in the index (removing anyway).")

    # 3. Optionally delete the file from disk
    if purge:
        if os.path.exists(model_path):
            try:
                os.remove(model_path)
                click.echo(f"Deleted {model_path}.")
            except OSError as e:
                click.echo(f"Error: Could not delete {model_path}: {e}", err=True)
                sys.exit(1)
        else:
            click.echo(f"Warning: File {model_path} does not exist on disk.", err=True)


@main.command()
@click.argument("model")
@click.option("--json", "as_json", is_flag=True, default=False, help="Output as JSON")
def info(model: str, as_json: bool):
    """Show metadata for a model from the index and GGUF header."""
    # 1. Resolve model
    entry = find_in_index(model)
    if entry:
        model_path = entry["path"]
    elif os.path.exists(model):
        model_path = os.path.abspath(model)
        entry = {"path": model_path, "filename": os.path.basename(model_path)}
    else:
        click.echo(f"Error: Model '{model}' not found in index and is not a valid path.", err=True)
        click.echo("Use 'ethllama list' to see indexed models.", err=True)
        sys.exit(1)

    # 2. File info from filesystem
    file_info: Dict[str, Any] = {}
    try:
        stat = os.stat(model_path)
        file_info["path"] = model_path
        file_info["filename"] = os.path.basename(model_path)
        file_info["size_bytes"] = stat.st_size
        file_info["size_human"] = _human_size(stat.st_size)
        file_info["modified"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(stat.st_mtime))
    except OSError as e:
        click.echo(f"Error: Could not stat file {model_path}: {e}", err=True)
        sys.exit(1)

    # 3. Read GGUF metadata
    gguf_meta = _read_gguf_metadata(model_path)
    is_gguf = gguf_meta is not None

    if as_json:
        output: Dict[str, Any] = {"file": file_info}
        if is_gguf:
            output["gguf"] = gguf_meta
        else:
            output["gguf"] = None
            output["warning"] = "Not a GGUF file or header too large to parse"
        click.echo(json.dumps(output, indent=2, default=str))
        return

    # 4. Human-readable output
    click.echo(f"Model: {file_info['filename']}")
    click.echo(f"  Path:     {file_info['path']}")
    click.echo(f"  Size:     {file_info['size_human']}")
    click.echo(f"  Modified: {file_info['modified']}")

    if not is_gguf:
        click.echo("\n  Warning: Not a GGUF file or header could not be parsed.")
        return

    click.echo("\n  GGUF Metadata:")
    # Show key fields in a nice order
    priority_keys = [
        "general.architecture", "general.name", "general.file_type",
        "general.context_length", "general.description",
    ]
    # Also include architecture-specific context length keys
    arch_keys = [
        "llama.context_length", "llama.embedding_length",
        "qwen2.context_length", "phi3.context_length",
        "mistral.context_length", "gemma.context_length",
        "command_r.context_length", "deepseek2.context_length",
    ]

    shown: set = set()
    for key in priority_keys + arch_keys:
        if key in gguf_meta:
            click.echo(f"    {key}: {gguf_meta[key]}")
            shown.add(key)

    # Show remaining metadata (skip internal keys)
    remaining = {k: v for k, v in gguf_meta.items()
                 if k not in shown and not k.startswith("__")}
    if remaining:
        for key in sorted(remaining.keys()):
            val = remaining[key]
            # Truncate long strings
            if isinstance(val, str) and len(val) > 120:
                val = val[:117] + "..."
            click.echo(f"    {key}: {val}")

    tensor_count = gguf_meta.get("__tensor_count")
    kv_count = gguf_meta.get("__metadata_kv_count")
    parts = []
    if tensor_count is not None:
        parts.append(f"{tensor_count} tensors")
    if kv_count is not None:
        parts.append(f"{kv_count} metadata keys")
    if parts:
        click.echo(f"\n  ({', '.join(parts)})")


def _human_size(size_bytes: int) -> str:
    """Format a byte count as a human-readable string."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(size_bytes) < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0  # type: ignore[assignment]
    return f"{size_bytes:.1f} PB"


# ---------------------------------------------------------------------------
# Sidecar subcommand groups
# ---------------------------------------------------------------------------
# Management commands (rm, info) live in cli_mgmt.py and STT (transcribe)
# lives in cli_stt.py. They are wired in here so the main `ethllama` group
# has a single entrypoint, while the implementations remain decoupled and
# can be tested in isolation.
from .cli_mgmt import register_commands as _register_mgmt
from .cli_stt import register_commands as _register_stt
_register_mgmt(main)
_register_stt(main)


if __name__ == "__main__":
    main()

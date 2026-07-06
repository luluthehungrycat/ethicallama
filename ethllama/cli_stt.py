"""CLI command for speech-to-text transcription via whisper.cpp.

The :func:`transcribe` click command is registered on the main ``ethllama``
CLI group via :func:`register_commands` (called by the orchestrator from
``cli.py`` -- this file deliberately does not import ``cli.py``).
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any, Optional

import click

from .engines import EngineConfig, get_engine, load_engines
from .stt import (
    SUPPORTED_FORMATS,
    WhisperBinaryNotFound,
    find_whisper_binary,
    transcribe_audio,
)


DEFAULT_ENGINE_NAME = "whisper-cpp"


# ---------------------------------------------------------------------------
# Engine resolution
# ---------------------------------------------------------------------------

def _resolve_engine(engine_name: Optional[str]) -> Optional[EngineConfig]:
    """Resolve an STT engine config by name or auto-detect.

    * If ``engine_name`` is provided, look it up by name and verify its
      ``type`` is ``"stt"``.
    * Otherwise, scan the user's engine directory for the first engine
      with ``type: stt``. If none is found, fall back to the conventional
      name ``"whisper-cpp"`` (which may still be on ``$PATH`` even when
      its YAML config is missing or fails validation).
    """
    if engine_name:
        engine = get_engine(engine_name)
        if engine is None:
            return None
        if engine.type != "stt":
            click.echo(
                f"Error: Engine '{engine_name}' has type '{engine.type}', "
                "expected 'stt'.",
                err=True,
            )
            return None
        return engine

    # Auto-detect: prefer the first engine that declares type: stt.
    engines = load_engines()
    for eng in engines.values():
        if eng.type == "stt":
            return eng

    # Fallback: conventional name (binary may still be on PATH).
    if DEFAULT_ENGINE_NAME in engines:
        return engines[DEFAULT_ENGINE_NAME]
    return None


def _resolve_model(
    model: Optional[str],
    engine_config: Optional[EngineConfig],
) -> Optional[str]:
    """Return the resolved model path, expanding ``~`` if needed."""
    if model:
        return os.path.expanduser(model)
    if engine_config is not None:
        default = getattr(engine_config, "default_model", None)
        if default:
            return os.path.expanduser(default)
    return None


def _print_result(
    result: Any,
    output_format: str,
    output_file: Optional[str],
) -> None:
    """Write the transcription result to stdout or to a file."""
    if output_format == "json" and not isinstance(result, str):
        text = json.dumps(result, indent=2, ensure_ascii=False)
    elif isinstance(result, (dict, list)):
        # Be forgiving: if the binary returned a structured result for
        # a non-JSON format, serialize it.
        text = json.dumps(result, indent=2, ensure_ascii=False)
    else:
        text = str(result)

    if output_file:
        out_dir = os.path.dirname(os.path.abspath(output_file)) or "."
        os.makedirs(out_dir, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(text)
            if not text.endswith("\n"):
                f.write("\n")
        click.echo(f"Transcription saved to {output_file}")
    else:
        click.echo(text)


# ---------------------------------------------------------------------------
# Click command
# ---------------------------------------------------------------------------

@click.command(name="transcribe")
@click.argument(
    "audio_file",
    type=click.Path(exists=True, dir_okay=False, readable=True),
)
@click.option(
    "--engine", "-e",
    default=None,
    help=(
        "Engine name from ~/.ethllama/engines/ (must have type: stt). "
        f"Default: auto-detect, falling back to '{DEFAULT_ENGINE_NAME}'."
    ),
)
@click.option(
    "--model", "-m", "model",
    default=None,
    help=(
        "Explicit path to the whisper model file (.bin/.ggml/.gguf). "
        "Overrides the engine config's default model."
    ),
)
@click.option(
    "--output-format", "-of", "output_format",
    default="text",
    type=click.Choice(list(SUPPORTED_FORMATS), case_sensitive=False),
    help="Output format: text, json, srt, or vtt.",
)
@click.option(
    "--language", "-l",
    default="auto",
    help="Language code (e.g. en, de, fr) or 'auto' for auto-detect.",
)
@click.option(
    "--threads", "-t",
    default=4, type=int, show_default=True,
    help="Number of CPU threads.",
)
@click.option(
    "--output", "-o", "output_file",
    default=None,
    help="Save output to FILE instead of stdout.",
)
def transcribe_cmd(
    audio_file: str,
    engine: Optional[str],
    model: Optional[str],
    output_format: str,
    language: str,
    threads: int,
    output_file: Optional[str],
) -> None:
    """Transcribe an audio file using a whisper.cpp engine.

    AUDIO_FILE is the path to an audio file (wav, mp3, m4a, ...).
    """
    fmt = (output_format or "text").lower()

    # 1. Resolve engine
    engine_config = _resolve_engine(engine)
    if engine is not None and engine_config is None:
        # The user explicitly named an engine that we couldn't load.
        click.echo(
            f"Error: Engine '{engine}' not found. "
            "Check ~/.ethllama/engines/ for installed engines.",
            err=True,
        )
        sys.exit(1)
    if engine_config is not None:
        click.echo(
            f"Engine: {engine_config.name} (type: {engine_config.type})",
            err=True,
        )
    else:
        click.echo("Engine: (auto-detect)", err=True)

    # 2. Resolve model
    resolved_model = _resolve_model(model, engine_config)
    if not resolved_model:
        click.echo(
            "Error: No model specified. Use --model /path/to/whisper-model.bin "
            "or set `default_model` in your whisper-cpp engine config.",
            err=True,
        )
        sys.exit(1)

    # 3. Locate whisper binary
    try:
        binary = find_whisper_binary(engine_config)
    except WhisperBinaryNotFound as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    click.echo(f"Binary: {binary}", err=True)
    click.echo(f"Model:  {resolved_model}", err=True)
    click.echo(f"Audio:  {audio_file}", err=True)

    # 4. Run transcription
    try:
        result = transcribe_audio(
            audio_path=audio_file,
            model_path=resolved_model,
            binary=binary,
            engine_config=engine_config,
            language=language,
            threads=threads,
            output_format=fmt,
        )
    except FileNotFoundError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    except (RuntimeError, ValueError) as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    except Exception as e:  # pragma: no cover - defensive
        click.echo(f"Transcription failed: {e}", err=True)
        sys.exit(1)

    # 5. Output
    _print_result(result, fmt, output_file)


# ---------------------------------------------------------------------------
# Registration helper
# ---------------------------------------------------------------------------

def register_commands(cli: click.Group) -> None:
    """Register the transcribe command on the given click group.

    The orchestrator calls this from ``cli.py`` so we do not import
    ``cli.py`` here (avoids a circular import).
    """
    cli.add_command(transcribe_cmd)


__all__ = ["transcribe_cmd", "register_commands"]


if __name__ == "__main__":
    transcribe_cmd()

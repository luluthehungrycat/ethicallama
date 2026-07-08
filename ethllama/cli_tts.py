"""CLI command for text-to-speech generation.

The :func:`tts` click command is registered on the main ``ethllama`` CLI
group via :func:`register_commands` (called by the orchestrator from
``cli.py`` -- this file deliberately does not import ``cli.py``).
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
from typing import Any, List, Optional

import click

from .engines import EngineConfig, get_engine, load_engines
from .index import resolve_model_path
from .tts import (
    SUPPORTED_FORMATS,
    TTSBinaryNotFound,
    find_tts_binary,
    synthesize_speech,
)


# Conventional engine name used as a last-ditch hint. There is no
# hard-coded engine YAML for this name; the binary may still be on
# ``$PATH`` and resolved by :func:`find_tts_binary`.
DEFAULT_ENGINE_NAME = "llama-tts"


# ---------------------------------------------------------------------------
# Engine / model / output path resolution
# ---------------------------------------------------------------------------

def _resolve_engine(engine_name: Optional[str]) -> Optional[EngineConfig]:
    """Resolve a TTS engine config by name or auto-detect.

    * If ``engine_name`` is provided, look it up by name and verify its
      ``type`` is ``"tts"``.
    * Otherwise, scan the user's engine directory for the first engine
      with ``type: tts``. If none is found, return ``None`` -- the CLI
      will then try to resolve a binary purely via ``$PATH``.
    """
    if engine_name:
        engine = get_engine(engine_name)
        if engine is None:
            return None
        if engine.type != "tts":
            click.echo(
                f"Error: Engine '{engine_name}' has type '{engine.type}', "
                "expected 'tts'.",
                err=True,
            )
            return None
        return engine

    # Auto-detect: prefer the first engine that declares type: tts.
    engines = load_engines()
    for eng in engines.values():
        if eng.type == "tts":
            return eng

    return None


def _resolve_model(
    model: Optional[str],
    engine_config: Optional[EngineConfig],
) -> Optional[str]:
    """Return the resolved model path, or ``None`` for model-less engines.

    Tries, in order:

    1. Explicit ``--model`` argument -- first as a literal path, then
       looked up in the index.
    2. ``engine_config.default_model`` from the engine YAML.
    """
    if model:
        # Already a real path? Use it.
        if os.path.isfile(model):
            return os.path.abspath(model)
        # Otherwise try resolving against the model index.
        resolved = resolve_model_path(model)
        if resolved:
            return os.path.abspath(resolved)
        # Last resort: treat the literal as a path (will fail later
        # with a clear error if missing).
        return os.path.abspath(model)

    if engine_config is not None:
        default = getattr(engine_config, "default_model", None)
        if default:
            return os.path.abspath(os.path.expanduser(default))

    return None


def _default_output_path(text: str, fmt: str) -> str:
    """Generate a default output path from a short hash of the text.

    Using a hash keeps the filename stable and collision-resistant
    without forcing the user to pass ``--output`` for ad-hoc invocations.
    """
    safe_fmt = (fmt or _DEFAULT_FORMAT).lower()
    if safe_fmt not in SUPPORTED_FORMATS:
        safe_fmt = "wav"
    digest = hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[:12]
    return f"tts-{digest}.{safe_fmt}"


def _try_play_audio(path: str) -> bool:
    """Attempt to launch a platform-appropriate audio player.

    Returns ``True`` if a player was launched (we don't wait for it to
    finish -- playback happens in the background). Returns ``False`` if
    no supported player is on ``$PATH``.

    Note: this is best-effort UX sugar. The generated file is always
    written to disk regardless of whether playback succeeds.
    """
    players: List[str] = [
        "paplay",   # PulseAudio (Linux)
        "aplay",    # ALSA (Linux)
        "ffplay",   # ffmpeg (cross-platform)
        "play",     # SoX
        "mpv",      # mpv media player
        "afplay",   # macOS
    ]
    for player in players:
        if shutil.which(player):
            try:
                subprocess.Popen(  # noqa: S603 -- trusted local binary
                    [player, path],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                return True
            except Exception:
                continue
    return False


# ---------------------------------------------------------------------------
# Click command
# ---------------------------------------------------------------------------

# ``_DEFAULT_FORMAT`` is referenced from within the click option defaults
# below; importing it lazily would just complicate the module surface.
from .tts import _DEFAULT_FORMAT  # noqa: E402


@click.command(name="tts")
@click.argument("text")
@click.option(
    "--model", "-m",
    default=None,
    help=(
        "Model path or name from the ethllama index (.gguf/.bin). "
        "Optional for model-less TTS engines."
    ),
)
@click.option(
    "--output", "-o",
    default=None,
    help="Output audio file path. Default: tts-<hash>.<format> in cwd.",
)
@click.option(
    "--format", "-f", "output_format",
    default=_DEFAULT_FORMAT,
    type=click.Choice(list(SUPPORTED_FORMATS), case_sensitive=False),
    show_default=True,
    help="Output audio format.",
)
@click.option(
    "--engine", "-e",
    default=None,
    help=(
        "Engine name from ~/.ethllama/engines/ (must have type: tts). "
        f"Default: auto-detect, falling back to '{DEFAULT_ENGINE_NAME}'."
    ),
)
@click.option(
    "--binary-dir",
    default=None,
    help="Directory containing TTS binaries (overrides engines.binary_dir).",
)
@click.option(
    "--voice",
    default=None,
    help="Voice ID (engine-dependent).",
)
@click.option(
    "--speed",
    default=1.0, type=float, show_default=True,
    help="Speech speed multiplier (1.0 = normal).",
)
@click.option(
    "--play",
    is_flag=True, default=False,
    help="Try to play the generated audio after writing it to disk.",
)
def tts_cmd(
    text: str,
    model: Optional[str],
    output: Optional[str],
    output_format: str,
    engine: Optional[str],
    binary_dir: Optional[str],
    voice: Optional[str],
    speed: float,
    play: bool,
) -> None:
    """Synthesize speech from TEXT using a TTS engine.

    TEXT is the text to speak. Wrap it in quotes for multi-word input.

    Examples:

        ethllama tts "Hello world" --output hello.wav

        ethllama tts "Hello" --model voxtral-tts-mini-q4.gguf \\
            --engine voxtral-tts --voice en-female --speed 1.2
    """
    fmt = (output_format or _DEFAULT_FORMAT).lower()
    if fmt not in SUPPORTED_FORMATS:
        click.echo(
            f"Error: Unsupported format '{output_format}'. "
            f"Expected one of: {', '.join(SUPPORTED_FORMATS)}",
            err=True,
        )
        sys.exit(1)

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

    # 2. Resolve model (may legitimately be None for model-less engines).
    resolved_model = _resolve_model(model, engine_config)
    if resolved_model:
        click.echo(f"Model:  {resolved_model}", err=True)
    else:
        click.echo("Model:  (none -- model-less engine)", err=True)

    # 3. Resolve output path (CLI override wins; otherwise auto-generate).
    output_path = output or _default_output_path(text, fmt)
    click.echo(f"Output: {output_path}", err=True)
    click.echo(f"Format: {fmt}", err=True)

    # 4. Resolve binary (CLI --binary-dir takes precedence; engine config
    #    is also consulted via find_tts_binary).
    binary = find_tts_binary(binary_dir=binary_dir, engine_config=engine_config)
    if binary:
        click.echo(f"Binary: {binary}", err=True)
    else:
        # Don't fail yet -- synthesize_speech will raise a clean error.
        click.echo("Binary: (not found yet -- will report on failure)", err=True)

    # 5. Run synthesis.
    try:
        result = synthesize_speech(
            text=text,
            output_path=output_path,
            model_path=resolved_model,
            binary=binary,
            engine_config=engine_config,
            voice=voice,
            speed=speed,
        )
    except TTSBinaryNotFound as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    except FileNotFoundError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    except (RuntimeError, ValueError) as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    except Exception as e:  # pragma: no cover - defensive
        click.echo(f"Synthesis failed: {e}", err=True)
        sys.exit(1)

    click.echo(f"Audio saved to {result}")

    # 6. Best-effort playback.
    if play:
        if _try_play_audio(result):
            click.echo("Playing audio (in background)")
        else:
            click.echo(
                "Note: No supported audio player found on $PATH. "
                "Install one of: paplay, aplay, ffplay, play (sox), "
                "mpv, or afplay.",
                err=True,
            )


# ---------------------------------------------------------------------------
# Registration helper
# ---------------------------------------------------------------------------

def register_commands(cli: click.Group) -> None:
    """Register the tts command on the given click group.

    The orchestrator calls this from ``cli.py`` so we do not import
    ``cli.py`` here (avoids a circular import).
    """
    cli.add_command(tts_cmd)


__all__ = ["tts_cmd", "register_commands"]


if __name__ == "__main__":
    tts_cmd()

"""Text-to-speech inference via compatible TTS engines.

This module provides a thin wrapper around TTS binaries for generating
audio from text. Supports multiple TTS engine types:

- llama-tts: built-in TTS tool from llama.cpp (``llama-cli --tts``)
- voxtral: voxtral-mini-realtime-rs TTS engine (``voxtral-tts``)
- Custom: any TTS engine via :class:`EngineConfig` with ``type: tts``

Configuration is read from engine YAML files in ``~/.ethllama/engines/``
(see :class:`ethllama.engines.EngineConfig`). The engine must have
``type: tts`` for it to be usable with ``ethllama tts``.

The public API is intentionally small:

* :func:`find_tts_binary` -- locate a TTS binary (returns ``None`` if not found)
* :func:`synthesize_speech` -- high-level: text in, audio file out
* :func:`get_tts_engine` -- find a TTS engine config in ``~/.ethllama/engines/``
"""

from __future__ import annotations

import os
import shutil
import subprocess
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:  # pragma: no cover - only for type checkers
    from .engines import EngineConfig


# Output audio formats supported by ``synthesize_speech``.
SUPPORTED_FORMATS = ("wav", "mp3", "ogg", "flac")

_DEFAULT_FORMAT = "wav"

# Candidate binary names, in priority order. ``llama-tts`` is the canonical
# name when the llama.cpp build exposes TTS as a separate tool; ``tts`` is
# the generic fallback for custom engines.
_TTS_BINARY_CANDIDATES = ("llama-tts", "voxtral-tts", "tts")


class TTSBinaryNotFound(RuntimeError):
    """Raised when no TTS binary can be located on the system."""


# ---------------------------------------------------------------------------
# Binary discovery
# ---------------------------------------------------------------------------

def find_tts_binary(
    name: str = "llama-tts",
    binary_dir: Optional[str] = None,
    engine_config: Optional["EngineConfig"] = None,
) -> Optional[str]:
    """Return an absolute path to a usable TTS binary, or *None*.

    Resolution order:

    1. ``engine_config.binary`` (if it is set, exists, and is executable)
    2. ``<binary_dir>/<name>`` (if ``binary_dir`` is provided and the file exists)
    3. Config file ``engines.binary_dir`` / ``<binary_dir>/<name>``,
       then ``engines.tts_binary`` (if set)
    4. ``shutil.which(name)`` and the other candidate names

    Unlike :func:`ethllama.stt.find_whisper_binary`, this function does
    *not* raise when nothing is found -- a TTS engine is an optional
    capability, and many installations won't have one. Callers that need
    a hard failure should pass the result to :func:`synthesize_speech`
    which raises :class:`TTSBinaryNotFound`.
    """
    # 1. engine_config.binary
    if engine_config is not None:
        binary = getattr(engine_config, "binary", None)
        if binary and os.path.isfile(binary) and os.access(binary, os.X_OK):
            return os.path.abspath(binary)

    # 2. explicit binary_dir override
    if binary_dir:
        candidate = os.path.join(binary_dir, name)
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return os.path.abspath(candidate)

    # 3. config file engines.binary_dir / engines.tts_binary
    try:
        from .config import load_config

        config = load_config()
        engines_cfg = config.get("engines", {}) or {}
        if not binary_dir:
            cfg_binary_dir = engines_cfg.get("binary_dir")
            if cfg_binary_dir:
                candidate = os.path.join(cfg_binary_dir, name)
                if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
                    return os.path.abspath(candidate)
        tts_binary = engines_cfg.get("tts_binary")
        if tts_binary and os.path.isfile(tts_binary) and os.access(tts_binary, os.X_OK):
            return os.path.abspath(tts_binary)
    except Exception:
        # If config loading fails for any reason, fall through to PATH lookup.
        pass

    # 4. PATH lookup -- check the requested name first, then fall back to
    #    the candidate list.
    candidates: List[str] = []
    if name not in candidates:
        candidates.append(name)
    for cand in _TTS_BINARY_CANDIDATES:
        if cand not in candidates:
            candidates.append(cand)
    for cand in candidates:
        found = shutil.which(cand)
        if found:
            return found

    return None


# ---------------------------------------------------------------------------
# Engine resolution
# ---------------------------------------------------------------------------

def get_tts_engine(
    engine_name: Optional[str] = None,
) -> Optional["EngineConfig"]:
    """Return a TTS :class:`EngineConfig` if one can be found.

    * If ``engine_name`` is provided, look it up by name and verify its
      ``type`` is ``"tts"``. Returns ``None`` if the engine is missing
      or has a non-TTS type.
    * Otherwise, scan the user's engine directory for the first engine
      with ``type: tts``. If none is found, return ``None``.
    """
    # Local import to avoid a top-level circular import with engines.py.
    from .engines import get_engine, load_engines

    if engine_name:
        engine = get_engine(engine_name)
        if engine is None:
            return None
        if engine.type != "tts":
            return None
        return engine

    engines = load_engines()
    for eng in engines.values():
        if eng.type == "tts":
            return eng

    return None


# ---------------------------------------------------------------------------
# Command building
# ---------------------------------------------------------------------------

def _render_engine_command(
    engine_config: "EngineConfig",
    text: str,
    output_path: str,
    model_path: Optional[str],
    voice: Optional[str],
    speed: float,
    extra_args: Optional[List[str]] = None,
) -> List[str]:
    """Render the TTS command from ``engine_config.args_template``.

    This is intentionally a small local helper (rather than going through
    :meth:`EngineConfig.render_command`) because the stock text-inference
    template variables do not include ``voice`` / ``speed``. Using our own
    Jinja2 render lets the engine YAML carry TTS-specific context.
    """
    from jinja2 import Template

    template = Template(engine_config.args_template)
    rendered = template.render(
        binary=engine_config.binary,
        model_path=model_path or "",
        prompt=text,
        output=output_path,
        voice=voice or "default",
        speed=speed,
    )
    cmd = [part for part in rendered.split() if part]
    if extra_args:
        cmd.extend(extra_args)
    return cmd


def _build_default_command(
    binary: str,
    text: str,
    output_path: str,
    model_path: Optional[str],
    voice: Optional[str],
    speed: float,
) -> List[str]:
    """Build a default TTS command based on binary-name detection.

    * ``voxtral`` style: ``voxtral-tts --input <text> --output <out> [--model <m>]``
    * default (``llama-tts`` style): ``<bin> -m <model> -p <text> -o <out>``
    """
    binary_name = os.path.basename(binary).lower()

    if "voxtral" in binary_name:
        cmd: List[str] = [binary, "--input", text, "--output", output_path]
        if model_path:
            cmd += ["--model", model_path]
        if voice:
            cmd += ["--voice", voice]
        if speed is not None and float(speed) != 1.0:
            cmd += ["--speed", str(speed)]
        return cmd

    # Default llama-tts style. Model is required for llama-tts but we
    # don't enforce that here -- the binary itself will error out if so.
    cmd = [binary]
    if model_path:
        cmd += ["-m", model_path]
    cmd += ["-p", text, "-o", output_path]
    if voice:
        cmd += ["--voice", voice]
    if speed is not None and float(speed) != 1.0:
        cmd += ["--speed", str(speed)]
    return cmd


# ---------------------------------------------------------------------------
# High-level entry point
# ---------------------------------------------------------------------------

def synthesize_speech(
    text: str,
    output_path: str,
    model_path: Optional[str] = None,
    *,
    binary: Optional[str] = None,
    engine_config: Optional["EngineConfig"] = None,
    voice: Optional[str] = None,
    speed: float = 1.0,
    timeout: int = 600,
    extra_args: Optional[List[str]] = None,
) -> str:
    """Synthesize speech from text via a TTS engine.

    Args:
        text: The text to speak. The CLI quotes the argument for you;
            the value here is the raw string.
        output_path: Path to the output audio file (e.g. ``hello.wav``).
            Parent directories are created automatically.
        model_path: Path to the TTS model file (``.gguf``/``.bin``).
            Optional for model-less engines.
        binary: Explicit path to the TTS binary. If omitted, the binary
            is resolved via :func:`find_tts_binary`.
        engine_config: Optional :class:`EngineConfig`. Used both to
            locate the binary and to render the command from
            ``args_template`` when available.
        voice: Optional voice identifier (engine-dependent).
        speed: Speech speed multiplier. ``1.0`` means normal speed.
            Values other than 1.0 are passed through to engines that
            support them.
        timeout: Subprocess timeout in seconds.
        extra_args: Additional CLI args passed verbatim to the binary.

    Returns:
        The path to the generated audio file (i.e. ``output_path``).

    Raises:
        TTSBinaryNotFound: when no TTS binary can be located.
        FileNotFoundError: when ``model_path`` is given but missing.
        RuntimeError: when the TTS binary exits with a non-zero status.
        ValueError: when ``engine_config.type`` is not ``"tts"``.
    """
    if engine_config is not None and getattr(engine_config, "type", None) not in (None, "tts"):
        raise ValueError(
            f"Engine '{engine_config.name}' has type "
            f"'{engine_config.type}', expected 'tts'."
        )

    if binary is None:
        binary = find_tts_binary(engine_config=engine_config)
    if not binary:
        raise TTSBinaryNotFound(
            "No TTS binary found. Install llama-tts, voxtral-tts, or any "
            "TTS engine on $PATH, or set `binary` in a tts engine config "
            "under ~/.ethllama/engines/."
        )

    if model_path and not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # Build the command: prefer the engine's args_template (when given)
    # over the built-in format-detection heuristic.
    template = getattr(engine_config, "args_template", None) if engine_config else None
    if template:
        cmd = _render_engine_command(
            engine_config,  # type: ignore[arg-type]
            text,
            output_path,
            model_path,
            voice,
            speed,
            extra_args,
        )
    else:
        cmd = _build_default_command(
            binary, text, output_path, model_path, voice, speed
        )
        if extra_args:
            cmd.extend(extra_args)

    # Make sure the output directory exists.
    out_dir = os.path.dirname(os.path.abspath(output_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except FileNotFoundError as e:
        # Raised when ``binary`` cannot be executed at all.
        raise TTSBinaryNotFound(
            f"Could not execute TTS binary '{binary}': {e}"
        ) from e

    if proc.returncode != 0:
        stderr = (proc.stderr or "").strip()
        raise RuntimeError(
            f"TTS binary failed (exit {proc.returncode}): "
            f"{stderr or 'no stderr output'}"
        )

    return output_path


__all__ = [
    "SUPPORTED_FORMATS",
    "TTSBinaryNotFound",
    "find_tts_binary",
    "get_tts_engine",
    "synthesize_speech",
]

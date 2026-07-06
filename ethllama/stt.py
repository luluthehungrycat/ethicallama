"""Speech-to-text inference via whisper.cpp (or compatible) engines.

This module provides a thin wrapper around the ``whisper-cli`` binary
(formerly known as ``main`` in older whisper.cpp builds) for transcribing
audio files into text, JSON, SRT, or VTT output.

Configuration is read from engine YAML files in ``~/.ethllama/engines/``
(see :class:`ethllama.engines.EngineConfig`). The engine must have
``type: stt`` for it to be usable with ``ethllama transcribe``.

The public API is intentionally small:

* :func:`find_whisper_binary` -- locate a whisper.cpp binary
* :func:`parse_whisper_output` -- normalize whisper-cli's stdout
* :func:`transcribe_audio` -- high-level: audio file in, transcription out
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:  # pragma: no cover - only for type checkers
    from .engines import EngineConfig


# Output formats supported by whisper-cli's ``-oX`` short flags
SUPPORTED_FORMATS = ("text", "json", "srt", "vtt")

_WHISPER_FORMAT_FLAGS: Dict[str, str] = {
    "text": "-otxt",
    "json": "-oj",
    "srt":  "-osrt",
    "vtt":  "-ovtt",
}

# Candidate binary names, in priority order. ``main`` is the legacy
# whisper.cpp binary name used before the v1.4 rename to ``whisper-cli``.
_WHISPER_BINARY_CANDIDATES = ("whisper-cli", "whisper", "main")


class WhisperBinaryNotFound(RuntimeError):
    """Raised when no whisper.cpp binary can be located on the system."""


# ---------------------------------------------------------------------------
# Binary discovery
# ---------------------------------------------------------------------------

def find_whisper_binary(engine_config: Optional["EngineConfig"] = None) -> str:
    """Return an absolute path to a usable whisper.cpp binary.

    Resolution order:

    1. ``engine_config.binary`` (if it is set, exists, and is executable)
    2. ``whisper-cli`` on ``$PATH``
    3. ``whisper`` on ``$PATH``
    4. ``main`` on ``$PATH`` (legacy whisper.cpp name)

    Raises:
        WhisperBinaryNotFound: when no candidate binary can be located.
    """
    if engine_config is not None:
        binary = getattr(engine_config, "binary", None)
        if binary and os.path.isfile(binary) and os.access(binary, os.X_OK):
            return binary

    for name in _WHISPER_BINARY_CANDIDATES:
        found = shutil.which(name)
        if found:
            return found

    raise WhisperBinaryNotFound(
        "No whisper.cpp binary found. Install whisper.cpp (provides "
        "`whisper-cli`) or set `binary` in a whisper-cpp engine config "
        "under ~/.ethllama/engines/."
    )


# ---------------------------------------------------------------------------
# Output parsing
# ---------------------------------------------------------------------------

def parse_whisper_output(
    raw: str,
    output_format: str = "text",
) -> Any:
    """Normalize whisper-cli's stdout into the requested output format.

    Args:
        raw: The captured stdout of ``whisper-cli``.
        output_format: One of ``"text"``, ``"json"``, ``"srt"``, ``"vtt"``.

    Returns:
        * For ``"text"``/``"srt"``/``"vtt"``: the cleaned string.
        * For ``"json"``: a parsed ``dict`` (or list of segments) when
          the payload is valid JSON, otherwise an empty dict.

    Raises:
        ValueError: if ``output_format`` is not one of the supported formats.
    """
    fmt = (output_format or "text").lower()
    if fmt not in SUPPORTED_FORMATS:
        raise ValueError(
            f"Unsupported output format: {output_format!r}. "
            f"Expected one of: {', '.join(SUPPORTED_FORMATS)}"
        )

    cleaned = (raw or "").strip()

    if fmt == "json":
        if not cleaned:
            return {}
        # whisper.cpp with -oj emits a JSON object per line. We first
        # try to parse the entire payload; if that fails we merge per
        # line to be robust against multi-line JSON output.
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            merged: Dict[str, Any] = {"segments": []}
            for line in cleaned.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(obj, dict):
                    if "segments" in obj and isinstance(obj["segments"], list):
                        merged["segments"].extend(obj["segments"])
                    else:
                        merged["segments"].append(obj)
                elif isinstance(obj, list):
                    merged["segments"].extend(obj)
            return merged

    return cleaned


# ---------------------------------------------------------------------------
# Command building
# ---------------------------------------------------------------------------

def _build_command(
    binary: str,
    model_path: str,
    audio_path: str,
    *,
    language: str = "auto",
    threads: int = 4,
    output_format: str = "text",
    n_processors: int = 1,
    extra_args: Optional[List[str]] = None,
) -> List[str]:
    """Build the whisper-cli command line.

    The resulting command uses ``--output-file -`` so the transcription
    is written to stdout, which we then capture in the caller.
    """
    fmt = (output_format or "text").lower()
    if fmt not in SUPPORTED_FORMATS:
        raise ValueError(
            f"Unsupported output format: {output_format!r}. "
            f"Expected one of: {', '.join(SUPPORTED_FORMATS)}"
        )
    flag = _WHISPER_FORMAT_FLAGS[fmt]

    cmd: List[str] = [
        binary,
        "-m", model_path,
        "-f", audio_path,
    ]
    if language and language != "auto":
        cmd += ["-l", language]
    cmd += ["-t", str(threads)]
    if n_processors and n_processors > 1:
        cmd += ["-p", str(n_processors)]
    cmd += [flag]
    # Hyphen means "stdout", letting us capture the result cleanly.
    cmd += ["--output-file", "-"]
    if extra_args:
        cmd += list(extra_args)
    return cmd


# ---------------------------------------------------------------------------
# High-level entry point
# ---------------------------------------------------------------------------

def transcribe_audio(
    audio_path: str,
    model_path: str,
    *,
    binary: Optional[str] = None,
    engine_config: Optional["EngineConfig"] = None,
    language: str = "auto",
    threads: int = 4,
    output_format: str = "text",
    n_processors: int = 1,
    timeout: int = 600,
    extra_args: Optional[List[str]] = None,
) -> Any:
    """Transcribe an audio file via whisper.cpp.

    Args:
        audio_path: Path to the audio file (``wav``, ``mp3``, ``m4a``, ...).
        model_path: Path to the whisper model file (``.bin``/``.ggml``).
        binary: Explicit path to the whisper-cli binary. If omitted, the
            binary is resolved via :func:`find_whisper_binary`.
        engine_config: Optional :class:`EngineConfig`. Used both to locate
            the binary and to validate that the engine type is ``"stt"``.
        language: ISO-639-1 code (``"en"``, ``"de"``, ...) or ``"auto"``.
        threads: Number of CPU threads to use.
        output_format: One of ``"text"``, ``"json"``, ``"srt"``, ``"vtt"``.
        n_processors: Number of processors for parallel inference
            (passed as ``-p`` to whisper-cli).
        timeout: Subprocess timeout in seconds.
        extra_args: Additional CLI args passed verbatim to whisper-cli.

    Returns:
        The transcription result. For ``"text"``/``"srt"``/``"vtt"`` this
        is a :class:`str`. For ``"json"`` this is a parsed :class:`dict`.

    Raises:
        WhisperBinaryNotFound: when the binary cannot be located.
        FileNotFoundError: when ``audio_path`` or ``model_path`` is missing.
        RuntimeError: when whisper-cli exits with a non-zero status.
        ValueError: when the engine type is not ``"stt"`` or the output
            format is unsupported.
    """
    if engine_config is not None and getattr(engine_config, "type", None) not in (None, "stt"):
        raise ValueError(
            f"Engine '{engine_config.name}' has type "
            f"'{engine_config.type}', expected 'stt'."
        )

    if binary is None:
        binary = find_whisper_binary(engine_config)

    if not os.path.isfile(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    cmd = _build_command(
        binary,
        model_path,
        audio_path,
        language=language,
        threads=threads,
        output_format=output_format,
        n_processors=n_processors,
        extra_args=extra_args,
    )

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
        raise WhisperBinaryNotFound(
            f"Could not execute whisper binary '{binary}': {e}"
        ) from e

    if proc.returncode != 0:
        stderr = (proc.stderr or "").strip()
        raise RuntimeError(
            f"whisper-cli failed (exit {proc.returncode}): "
            f"{stderr or 'no stderr output'}"
        )

    return parse_whisper_output(proc.stdout, output_format)

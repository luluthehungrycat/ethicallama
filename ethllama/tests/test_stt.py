"""Tests for the ethllama STT helper module (`ethllama.stt`)."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from unittest import mock

import pytest

from ethllama.stt import (
    SUPPORTED_FORMATS,
    WhisperBinaryNotFound,
    _build_command,
    find_whisper_binary,
    parse_whisper_output,
    transcribe_audio,
)


# ---------------------------------------------------------------------------
# Imports & module surface
# ---------------------------------------------------------------------------

def test_stt_imports():
    """The public STT API imports cleanly."""
    from ethllama import stt  # noqa: F401

    assert hasattr(stt, "transcribe_audio")
    assert hasattr(stt, "parse_whisper_output")
    assert hasattr(stt, "find_whisper_binary")
    assert hasattr(stt, "WhisperBinaryNotFound")
    assert hasattr(stt, "SUPPORTED_FORMATS")
    assert SUPPORTED_FORMATS == ("text", "json", "srt", "vtt")


# ---------------------------------------------------------------------------
# parse_whisper_output
# ---------------------------------------------------------------------------

def test_stt_parse_text_output():
    """parse_whisper_output returns cleaned text for the text format."""
    raw = "  Hello, this is a test transcription.\n\n"
    result = parse_whisper_output(raw, output_format="text")
    assert isinstance(result, str)
    assert result == "Hello, this is a test transcription."


def test_stt_parse_text_default_format():
    """Default output_format is 'text'."""
    raw = "transcribed text"
    result = parse_whisper_output(raw)
    assert result == "transcribed text"


def test_stt_parse_srt_output():
    """parse_whisper_output preserves SRT subtitle structure."""
    raw = (
        "1\n"
        "00:00:00,000 --> 00:00:02,500\n"
        "Hello world\n"
        "\n"
        "2\n"
        "00:00:02,500 --> 00:00:05,000\n"
        "Second line\n"
    )
    result = parse_whisper_output(raw, output_format="srt")
    assert isinstance(result, str)
    assert "1" in result
    assert "Hello world" in result
    assert "Second line" in result


def test_stt_parse_vtt_output():
    """parse_whisper_output preserves WebVTT structure."""
    raw = (
        "WEBVTT\n"
        "\n"
        "00:00:00.000 --> 00:00:02.500\n"
        "Hello world\n"
    )
    result = parse_whisper_output(raw, output_format="vtt")
    assert isinstance(result, str)
    assert "WEBVTT" in result
    assert "Hello world" in result


def test_stt_parse_json_output_single_object():
    """parse_whisper_output returns a dict for a single JSON object."""
    payload = {
        "text": "Hello world",
        "segments": [
            {"id": 0, "start": 0.0, "end": 1.0, "text": "Hello world"},
        ],
        "language": "en",
    }
    raw = json.dumps(payload)
    result = parse_whisper_output(raw, output_format="json")
    assert isinstance(result, dict)
    assert result["text"] == "Hello world"
    assert result["language"] == "en"
    assert len(result["segments"]) == 1


def test_stt_parse_json_output_multiple_lines():
    """parse_whisper_output merges NDJSON segment output."""
    line1 = json.dumps({"text": "Hello", "segments": [{"id": 0, "text": "Hello"}]})
    line2 = json.dumps({"text": "world", "segments": [{"id": 1, "text": "world"}]})
    raw = line1 + "\n" + line2
    result = parse_whisper_output(raw, output_format="json")
    assert isinstance(result, dict)
    assert "segments" in result
    assert len(result["segments"]) == 2


def test_stt_parse_json_output_empty():
    """parse_whisper_output returns an empty dict for empty input."""
    assert parse_whisper_output("", output_format="json") == {}
    assert parse_whisper_output("   \n  ", output_format="json") == {}


def test_stt_parse_invalid_format_raises():
    """parse_whisper_output raises ValueError for unsupported formats."""
    with pytest.raises(ValueError):
        parse_whisper_output("hi", output_format="xml")


# ---------------------------------------------------------------------------
# find_whisper_binary
# ---------------------------------------------------------------------------

def test_find_whisper_binary_raises_when_missing(monkeypatch, tmp_path):
    """find_whisper_binary raises WhisperBinaryNotFound when nothing is found."""
    # Ensure shutil.which returns None for all candidates.
    monkeypatch.setattr("shutil.which", lambda name: None)
    with pytest.raises(WhisperBinaryNotFound):
        find_whisper_binary()


def test_find_whisper_binary_uses_engine_config(monkeypatch, tmp_path):
    """find_whisper_binary uses engine_config.binary when it is executable."""

    fake_bin = tmp_path / "whisper-cli"
    fake_bin.write_text("#!/bin/sh\n")
    fake_bin.chmod(0o755)

    class FakeEngine:
        binary = str(fake_bin)

    assert find_whisper_binary(FakeEngine()) == str(fake_bin)


def test_find_whisper_binary_falls_back_to_path(monkeypatch, tmp_path):
    """find_whisper_binary falls back to PATH lookup."""

    class FakeEngine:
        binary = "/nonexistent/path/whisper-cli"

    monkeypatch.setattr(
        "shutil.which",
        lambda name: "/usr/bin/whisper-cli" if name == "whisper-cli" else None,
    )
    assert find_whisper_binary(FakeEngine()) == "/usr/bin/whisper-cli"


# ---------------------------------------------------------------------------
# _build_command
# ---------------------------------------------------------------------------

def test_build_command_includes_required_flags():
    """_build_command produces a sane whisper-cli invocation."""
    cmd = _build_command(
        binary="/usr/bin/whisper-cli",
        model_path="/models/ggml-base.en.bin",
        audio_path="/audio/sample.wav",
        language="en",
        threads=8,
        output_format="text",
    )
    # The binary must be the first argument
    assert cmd[0] == "/usr/bin/whisper-cli"
    # Required flags
    assert "-m" in cmd
    assert cmd[cmd.index("-m") + 1] == "/models/ggml-base.en.bin"
    assert "-f" in cmd
    assert cmd[cmd.index("-f") + 1] == "/audio/sample.wav"
    assert "-l" in cmd
    assert cmd[cmd.index("-l") + 1] == "en"
    assert "-t" in cmd
    assert cmd[cmd.index("-t") + 1] == "8"
    assert "-otxt" in cmd
    # Output to stdout via hyphen
    assert "--output-file" in cmd
    assert cmd[cmd.index("--output-file") + 1] == "-"


def test_build_command_auto_language_omits_lang_flag():
    """When language='auto' the -l flag is omitted."""
    cmd = _build_command(
        binary="/bin",
        model_path="/m",
        audio_path="/a",
        language="auto",
        threads=4,
        output_format="srt",
    )
    assert "-l" not in cmd
    assert "-osrt" in cmd


def test_build_command_invalid_format_raises():
    """_build_command rejects unsupported output formats."""
    with pytest.raises(ValueError):
        _build_command(
            binary="/bin",
            model_path="/m",
            audio_path="/a",
            language="en",
            threads=4,
            output_format="xml",
        )


# ---------------------------------------------------------------------------
# transcribe_audio (subprocess mocked)
# ---------------------------------------------------------------------------

def _make_completed_proc(returncode: int = 0, stdout: str = "", stderr: str = ""):
    proc = mock.Mock(spec=subprocess.CompletedProcess)
    proc.returncode = returncode
    proc.stdout = stdout
    proc.stderr = stderr
    return proc


def test_transcribe_audio_runs_subprocess(monkeypatch, tmp_path):
    """transcribe_audio invokes whisper-cli with the expected args."""
    model = tmp_path / "model.bin"
    model.write_bytes(b"")
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"")

    fake_proc = _make_completed_proc(returncode=0, stdout="hello world\n")

    with mock.patch("ethllama.stt.subprocess.run", return_value=fake_proc) as run_mock:
        result = transcribe_audio(
            audio_path=str(audio),
            model_path=str(model),
            binary="/bin/whisper-cli",
            output_format="text",
        )

    assert result == "hello world"
    run_mock.assert_called_once()
    args, kwargs = run_mock.call_args
    cmd = args[0]
    assert cmd[0] == "/bin/whisper-cli"
    assert "-m" in cmd and str(model) in cmd
    assert "-f" in cmd and str(audio) in cmd
    assert "-otxt" in cmd
    assert kwargs.get("capture_output") is True
    assert kwargs.get("text") is True


def test_transcribe_audio_json(monkeypatch, tmp_path):
    """transcribe_audio parses JSON output into a dict."""
    model = tmp_path / "model.bin"
    model.write_bytes(b"")
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"")

    payload = {"text": "hi", "language": "en"}
    fake_proc = _make_completed_proc(returncode=0, stdout=json.dumps(payload))

    with mock.patch("ethllama.stt.subprocess.run", return_value=fake_proc):
        result = transcribe_audio(
            audio_path=str(audio),
            model_path=str(model),
            binary="/bin/whisper-cli",
            output_format="json",
        )

    assert result == payload


def test_transcribe_audio_missing_audio(tmp_path):
    """transcribe_audio raises FileNotFoundError for missing audio."""
    model = tmp_path / "model.bin"
    model.write_bytes(b"")

    with pytest.raises(FileNotFoundError):
        transcribe_audio(
            audio_path=str(tmp_path / "no.wav"),
            model_path=str(model),
            binary="/bin/whisper-cli",
        )


def test_transcribe_audio_missing_model(tmp_path):
    """transcribe_audio raises FileNotFoundError for missing model."""
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"")

    with pytest.raises(FileNotFoundError):
        transcribe_audio(
            audio_path=str(audio),
            model_path=str(tmp_path / "no.bin"),
            binary="/bin/whisper-cli",
        )


def test_transcribe_audio_nonzero_exit(tmp_path):
    """transcribe_audio raises RuntimeError on non-zero exit."""
    model = tmp_path / "model.bin"
    model.write_bytes(b"")
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"")

    fake_proc = _make_completed_proc(
        returncode=1, stdout="", stderr="some error from whisper"
    )
    with mock.patch("ethllama.stt.subprocess.run", return_value=fake_proc):
        with pytest.raises(RuntimeError) as exc:
            transcribe_audio(
                audio_path=str(audio),
                model_path=str(model),
                binary="/bin/whisper-cli",
            )
    assert "whisper-cli failed" in str(exc.value)
    assert "some error" in str(exc.value)


def test_transcribe_audio_rejects_wrong_engine_type(tmp_path):
    """transcribe_audio rejects an engine whose type is not 'stt'."""
    model = tmp_path / "model.bin"
    model.write_bytes(b"")
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"")

    class TextEngine:
        name = "llama-cpp"
        type = "text"
        binary = "/bin/llama-cli"

    with pytest.raises(ValueError) as exc:
        transcribe_audio(
            audio_path=str(audio),
            model_path=str(model),
            engine_config=TextEngine(),
        )
    assert "expected 'stt'" in str(exc.value)

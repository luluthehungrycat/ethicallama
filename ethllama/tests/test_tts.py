"""Tests for the ethllama TTS helper module (`ethllama.tts` and `ethllama.cli_tts`).

Covers:
* module surface / public API
* `find_tts_binary` resolution (engine_config > binary_dir > config > PATH)
* `get_tts_engine` lookup
* default + template-based command construction
* `synthesize_speech` subprocess invocation
* click command registration + --help output
* the `ethllama tts` subcommand wired onto the main CLI group
"""

from __future__ import annotations

import subprocess
from unittest import mock

import pytest
from click.testing import CliRunner

from ethllama.tts import (
    SUPPORTED_FORMATS,
    TTSBinaryNotFound,
    _build_default_command,
    _render_engine_command,
    find_tts_binary,
    get_tts_engine,
    synthesize_speech,
)
from ethllama.cli_tts import tts_cmd, register_commands as register_tts
from ethllama.cli import main


# ---------------------------------------------------------------------------
# Imports & module surface
# ---------------------------------------------------------------------------

def test_tts_imports():
    """The public TTS API imports cleanly and exposes the documented names."""
    from ethllama import tts  # noqa: F401

    assert hasattr(tts, "find_tts_binary")
    assert hasattr(tts, "synthesize_speech")
    assert hasattr(tts, "get_tts_engine")
    assert hasattr(tts, "TTSBinaryNotFound")
    assert hasattr(tts, "SUPPORTED_FORMATS")
    assert SUPPORTED_FORMATS == ("wav", "mp3", "ogg", "flac")


def test_tts_command_imports():
    """The TTS click command and register helper are importable."""
    assert tts_cmd.name == "tts"
    assert callable(register_tts)


# ---------------------------------------------------------------------------
# find_tts_binary
# ---------------------------------------------------------------------------

def test_find_tts_binary_returns_none_when_missing(monkeypatch):
    """find_tts_binary returns None when no candidate is anywhere."""
    monkeypatch.setattr("shutil.which", lambda name: None)
    # Bypass config-file lookup to keep the test focused on the PATH leg.
    monkeypatch.setattr(
        "ethllama.tts.load_config",
        lambda: {"engines": {}},
        raising=False,
    )
    # Some Python versions expose load_config on the module; cover both.
    import ethllama.config as cfg
    monkeypatch.setattr(cfg, "load_config", lambda: {"engines": {}})

    assert find_tts_binary() is None


def test_find_tts_binary_uses_engine_config(monkeypatch, tmp_path):
    """find_tts_binary prefers engine_config.binary when it is executable."""
    fake_bin = tmp_path / "llama-tts"
    fake_bin.write_text("#!/bin/sh\n")
    fake_bin.chmod(0o755)

    class FakeEngine:
        binary = str(fake_bin)

    assert find_tts_binary(engine_config=FakeEngine()) == str(fake_bin)


def test_find_tts_binary_uses_binary_dir(monkeypatch, tmp_path):
    """find_tts_binary checks <binary_dir>/<name>."""
    fake_bin = tmp_path / "llama-tts"
    fake_bin.write_text("#!/bin/sh\n")
    fake_bin.chmod(0o755)

    # Ensure PATH lookup is bypassed so we exercise the binary_dir branch.
    monkeypatch.setattr("shutil.which", lambda name: None)
    import ethllama.config as cfg
    monkeypatch.setattr(cfg, "load_config", lambda: {"engines": {}})

    found = find_tts_binary(name="llama-tts", binary_dir=str(tmp_path))
    assert found == str(fake_bin)


def test_find_tts_binary_falls_back_to_path(monkeypatch, tmp_path):
    """find_tts_binary falls back to shutil.which when config dirs miss."""

    class FakeEngine:
        binary = "/nonexistent/path/llama-tts"

    monkeypatch.setattr(
        "ethllama.config.load_config",
        lambda: {"engines": {}},
    )
    monkeypatch.setattr(
        "shutil.which",
        lambda name: "/usr/bin/llama-tts" if name == "llama-tts" else None,
    )
    assert find_tts_binary(engine_config=FakeEngine()) == "/usr/bin/llama-tts"


# ---------------------------------------------------------------------------
# get_tts_engine
# ---------------------------------------------------------------------------

def test_get_tts_engine_returns_none_when_nothing(monkeypatch):
    """get_tts_engine returns None when no engines are installed."""
    import ethllama.engines as engines_mod
    monkeypatch.setattr(engines_mod, "load_engines", lambda: {})
    monkeypatch.setattr(engines_mod, "get_engine", lambda _n: None)
    assert get_tts_engine() is None
    assert get_tts_engine("any-name") is None


def test_get_tts_engine_returns_first_tts_type(monkeypatch):
    """get_tts_engine returns the first engine whose type is 'tts'."""

    class TtsEngine:
        name = "voxtral-tts"
        type = "tts"
        binary = "/bin/voxtral-tts"

    class OtherEngine:
        name = "llama-cpp"
        type = "text"
        binary = "/bin/llama-cli"

    import ethllama.engines as engines_mod
    tts_instance = TtsEngine()
    monkeypatch.setattr(
        engines_mod,
        "load_engines",
        lambda: {"voxtral-tts": tts_instance, "llama-cpp": OtherEngine()},
    )
    eng = get_tts_engine()
    # The class is re-defined inside this test function on every test
    # invocation, so we assert by identity against the instance we put
    # into the patched load_engines return value.
    assert eng is tts_instance
    assert isinstance(eng, TtsEngine)


def test_get_tts_engine_rejects_wrong_type(monkeypatch):
    """get_tts_engine returns None when the named engine has a non-tts type."""

    class TextEngine:
        name = "llama-cpp"
        type = "text"
        binary = "/bin/llama-cli"

    import ethllama.engines as engines_mod
    monkeypatch.setattr(engines_mod, "get_engine", lambda _n: TextEngine())
    assert get_tts_engine("llama-cpp") is None


# ---------------------------------------------------------------------------
# _build_default_command
# ---------------------------------------------------------------------------

def test_build_default_command_llama_tts():
    """_build_default_command emits llama-tts style flags for unknown binaries."""
    cmd = _build_default_command(
        binary="/usr/bin/llama-tts",
        text="hello",
        output_path="/tmp/out.wav",
        model_path="/models/tts.gguf",
        voice=None,
        speed=1.0,
    )
    assert cmd[0] == "/usr/bin/llama-tts"
    assert "-m" in cmd
    assert cmd[cmd.index("-m") + 1] == "/models/tts.gguf"
    assert "-p" in cmd
    assert cmd[cmd.index("-p") + 1] == "hello"
    assert "-o" in cmd
    assert cmd[cmd.index("-o") + 1] == "/tmp/out.wav"


def test_build_default_command_voxtral():
    """_build_default_command emits voxtral style flags when 'voxtral' in name."""
    cmd = _build_default_command(
        binary="/opt/voxtral-tts",
        text="hi",
        output_path="/tmp/out.wav",
        model_path="/models/v.gguf",
        voice="en-female",
        speed=1.2,
    )
    assert cmd[0] == "/opt/voxtral-tts"
    assert "--input" in cmd and cmd[cmd.index("--input") + 1] == "hi"
    assert "--output" in cmd and cmd[cmd.index("--output") + 1] == "/tmp/out.wav"
    assert "--model" in cmd and cmd[cmd.index("--model") + 1] == "/models/v.gguf"
    assert "--voice" in cmd and cmd[cmd.index("--voice") + 1] == "en-female"
    assert "--speed" in cmd and cmd[cmd.index("--speed") + 1] == "1.2"


def test_build_default_command_voxtral_model_less():
    """voxtral style omits --model and --voice/--speed when not provided."""
    cmd = _build_default_command(
        binary="/opt/voxtral-tts",
        text="hi",
        output_path="/tmp/out.wav",
        model_path=None,
        voice=None,
        speed=1.0,
    )
    assert "--model" not in cmd
    assert "--voice" not in cmd
    assert "--speed" not in cmd


# ---------------------------------------------------------------------------
# _render_engine_command
# ---------------------------------------------------------------------------

def test_render_engine_command_uses_args_template():
    """_render_engine_command renders the engine's args_template with TTS context."""

    class FakeEngine:
        binary = "/bin/voxtral-tts"
        args_template = (
            "--input {{ prompt }} --output {{ output }} "
            "--voice {{ voice | default('default') }} "
            "--speed {{ speed | default(1.0) }}"
        )

    cmd = _render_engine_command(
        FakeEngine(),  # type: ignore[arg-type]
        text="hello",
        output_path="/tmp/out.wav",
        model_path=None,
        voice="en-male",
        speed=1.5,
    )
    # Whitespace-normalized comparisons are friendlier than exact list equality
    joined = " ".join(cmd)
    assert "--input" in joined and "hello" in joined
    assert "--output" in joined and "/tmp/out.wav" in joined
    assert "--voice" in joined and "en-male" in joined
    assert "--speed" in joined and "1.5" in joined


# ---------------------------------------------------------------------------
# synthesize_speech (subprocess mocked)
# ---------------------------------------------------------------------------

def _make_completed_proc(returncode: int = 0, stdout: str = "", stderr: str = ""):
    proc = mock.Mock(spec=subprocess.CompletedProcess)
    proc.returncode = returncode
    proc.stdout = stdout
    proc.stderr = stderr
    return proc


def test_synthesize_speech_runs_subprocess(monkeypatch, tmp_path):
    """synthesize_speech invokes the TTS binary with the expected args."""
    out = tmp_path / "out.wav"
    fake_proc = _make_completed_proc(returncode=0, stdout="")

    with mock.patch("ethllama.tts.subprocess.run", return_value=fake_proc) as run_mock:
        result = synthesize_speech(
            text="hello world",
            output_path=str(out),
            model_path=None,
            binary="/bin/llama-tts",
        )

    assert result == str(out)
    run_mock.assert_called_once()
    args, kwargs = run_mock.call_args
    cmd = args[0]
    assert cmd[0] == "/bin/llama-tts"
    assert "-p" in cmd and "hello world" in cmd
    assert "-o" in cmd and str(out) in cmd
    assert kwargs.get("capture_output") is True
    assert kwargs.get("text") is True


def test_synthesize_speech_raises_when_no_binary(monkeypatch, tmp_path):
    """synthesize_speech raises TTSBinaryNotFound if no binary is resolvable."""
    monkeypatch.setattr("shutil.which", lambda name: None)
    import ethllama.config as cfg
    monkeypatch.setattr(cfg, "load_config", lambda: {"engines": {}})

    with pytest.raises(TTSBinaryNotFound):
        synthesize_speech(
            text="hi",
            output_path=str(tmp_path / "out.wav"),
            model_path=None,
        )


def test_synthesize_speech_missing_model(tmp_path):
    """synthesize_speech raises FileNotFoundError for missing model."""
    with pytest.raises(FileNotFoundError):
        synthesize_speech(
            text="hi",
            output_path=str(tmp_path / "out.wav"),
            model_path=str(tmp_path / "no-model.gguf"),
            binary="/bin/llama-tts",
        )


def test_synthesize_speech_nonzero_exit(tmp_path):
    """synthesize_speech raises RuntimeError on non-zero exit."""
    fake_proc = _make_completed_proc(returncode=1, stderr="boom")
    with mock.patch("ethllama.tts.subprocess.run", return_value=fake_proc):
        with pytest.raises(RuntimeError) as exc:
            synthesize_speech(
                text="hi",
                output_path=str(tmp_path / "out.wav"),
                model_path=None,
                binary="/bin/llama-tts",
            )
    assert "exit 1" in str(exc.value)
    assert "boom" in str(exc.value)


def test_synthesize_speech_rejects_wrong_engine_type(tmp_path):
    """synthesize_speech rejects an engine whose type is not 'tts'."""

    class TextEngine:
        name = "llama-cpp"
        type = "text"
        binary = "/bin/llama-cli"

    with pytest.raises(ValueError) as exc:
        synthesize_speech(
            text="hi",
            output_path=str(tmp_path / "out.wav"),
            model_path=None,
            engine_config=TextEngine(),
        )
    assert "expected 'tts'" in str(exc.value)


# ---------------------------------------------------------------------------
# Click command registration & help
# ---------------------------------------------------------------------------

def test_register_commands_adds_tts():
    """register_commands attaches the tts command onto the given group."""
    import click

    @click.group()
    def grp():
        pass

    register_tts(grp)
    assert "tts" in grp.commands


def test_tts_direct_help():
    """The standalone tts_cmd shows help when invoked directly."""
    runner = CliRunner()
    result = runner.invoke(tts_cmd, ["--help"])
    assert result.exit_code == 0
    out = result.output.lower()
    assert "synthesize speech" in out
    assert "--model" in out
    assert "--output" in out
    assert "--format" in out
    assert "--engine" in out
    assert "--binary-dir" in out
    assert "--voice" in out
    assert "--speed" in out
    assert "--play" in out


def test_tts_visible_on_main_cli_help():
    """`ethllama --help` lists the new `tts` subcommand."""
    runner = CliRunner()
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    # The help table contains a `tts` row; be lenient about whitespace.
    out = result.output
    assert "\n  tts " in out or " tts " in out


def test_tts_subcommand_help_via_main_cli():
    """`ethllama tts --help` works end-to-end through the main group."""
    runner = CliRunner()
    result = runner.invoke(main, ["tts", "--help"])
    assert result.exit_code == 0
    out = result.output.lower()
    assert "synthesize speech" in out


def test_tts_missing_binary(tmp_path, monkeypatch):
    """`ethllama tts` reports a clear error when no TTS binary is on PATH."""
    monkeypatch.setattr("shutil.which", lambda name: None)
    import ethllama.config as cfg
    monkeypatch.setattr(cfg, "load_config", lambda: {"engines": {}})
    # Make engine resolution return None so we exercise the bare PATH path.
    import ethllama.cli_tts as cli_tts_mod
    monkeypatch.setattr(cli_tts_mod, "load_engines", lambda: {})
    monkeypatch.setattr(cli_tts_mod, "get_engine", lambda _n: None)

    runner = CliRunner()
    result = runner.invoke(main, ["tts", "hello", "--output", str(tmp_path / "out.wav")])
    assert result.exit_code != 0
    out = (result.output or "").lower()
    assert "binary" in out or "tts" in out


def test_tts_named_engine_missing(monkeypatch):
    """`ethllama tts --engine missing` reports the missing engine by name."""
    import ethllama.cli_tts as cli_tts_mod
    monkeypatch.setattr(cli_tts_mod, "get_engine", lambda _n: None)

    runner = CliRunner()
    result = runner.invoke(
        main, ["tts", "hi", "--engine", "nonexistent-engine"]
    )
    assert result.exit_code != 0
    out = (result.output or "").lower()
    assert "nonexistent-engine" in out or "not found" in out

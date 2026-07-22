"""Tests for the ethllama CLI."""
import json
import os
import struct
import tempfile
import textwrap

import pytest
import yaml
from click.testing import CliRunner
from ethllama.cli import main
from ethllama.cli_mgmt import register_commands, _read_gguf_metadata

# Register the management subcommands (rm, info) so tests can
# invoke them through the main click group.
register_commands(main)



def test_cli_help():
    """Test that the main CLI group shows help."""
    runner = CliRunner()
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "ethicallama" in result.output


def test_cli_version():
    """Test that --version works."""
    runner = CliRunner()
    result = runner.invoke(main, ["--version"])
    assert result.exit_code == 0
    assert "0.2.0" in result.output


def test_run_help():
    """Test that 'run --help' shows expected options."""
    runner = CliRunner()
    result = runner.invoke(main, ["run", "--help"])
    assert result.exit_code == 0
    assert "prompt" in result.output.lower()
    assert "temperature" in result.output.lower()
    assert "model" in result.output.lower()


def test_run_missing_model():
    """Test that 'run' with a non-existent model exits with error."""
    runner = CliRunner()
    result = runner.invoke(main, ["run", "nonexistent-model"])
    assert result.exit_code != 0
    assert "not found" in result.output.lower()


def test_list_help():
    """Test that 'list --help' works."""
    runner = CliRunner()
    result = runner.invoke(main, ["list", "--help"])
    assert result.exit_code == 0


def test_index_help():
    """Test that 'index --help' works."""
    runner = CliRunner()
    result = runner.invoke(main, ["index", "--help"])
    assert result.exit_code == 0


def test_pull_help():
    """Test that 'pull --help' shows source option."""
    runner = CliRunner()
    result = runner.invoke(main, ["pull", "--help"])
    assert result.exit_code == 0
    assert "source" in result.output.lower()


def test_config_help():
    """Test that 'config --help' works."""
    runner = CliRunner()
    result = runner.invoke(main, ["config", "--help"])
    assert result.exit_code == 0


def test_serve_help():
    """Test that 'serve --help' shows host, port, and --model options."""
    runner = CliRunner()
    result = runner.invoke(main, ["serve", "--help"])
    assert result.exit_code == 0
    assert "host" in result.output.lower()
    assert "port" in result.output.lower()
    # The new --model / -m option must be documented
    assert "--model" in result.output
    assert "-m" in result.output
    assert "pre-load" in result.output.lower() or "preload" in result.output.lower()


def test_serve_short_model_flag():
    """Test that 'serve -m' is recognized as the model option."""
    runner = CliRunner()
    result = runner.invoke(main, ["serve", "-m", "some-model", "--help"])
    # -m should be accepted as a known option (no "no such option" error)
    assert result.exit_code == 0
    assert "no such option" not in result.output.lower()


def test_serve_model_warning_for_missing_model(tmp_path, monkeypatch):
    """`serve --model <missing>` prints a warning but doesn't crash before uvicorn import."""
    from ethllama import index as index_mod

    # Isolate the index so we know the model is definitely not present
    monkeypatch.setattr(index_mod, "INDEX_FILE", tmp_path / "index.json")

    # Patch the lazy api import so the test doesn't actually start a server.
    # If the warning path runs correctly, we expect run_server to be called
    # with model_path=None; if it doesn't, the function will try to import
    # the real api module and may fail on a headless box without fastapi.
    import ethllama.cli as cli_mod
    captured: dict = {}

    def fake_run_server(*args, **kwargs):
        captured.update(kwargs)
        # Raise to break out of the server loop without actually listening
        raise KeyboardInterrupt()

    monkeypatch.setattr(cli_mod, "resolve_model_path", lambda _m: "")
    monkeypatch.setattr("os.path.exists", lambda _p: False)
    # Patch the lazy import by intercepting the import inside serve()
    import builtins
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        mod = real_import(name, *args, **kwargs)
        if name.endswith("api") or name == "ethllama.api":
            mod.run_server = fake_run_server  # type: ignore[attr-defined]
        return mod

    monkeypatch.setattr(builtins, "__import__", fake_import)

    runner = CliRunner()
    result = runner.invoke(
        main, ["serve", "--model", "definitely-not-a-real-model"]
    )

    # The warning should have been printed
    assert "Warning" in result.output or "not found" in result.output.lower()
    # And run_server was called with model_path=None (graceful fallback)
    assert captured.get("model_path") is None


def test_engines_help():
    """Test that 'engines --help' works."""
    runner = CliRunner()
    result = runner.invoke(main, ["engines", "--help"])
    assert result.exit_code == 0


def test_engines_no_dir():
    """Test that 'engines' handles missing directory gracefully."""
    runner = CliRunner()
    result = runner.invoke(main, ["engines"])
    # Should not crash — show helpful message
    assert result.exit_code == 0


def test_list_no_models():
    """Test that 'list' with empty index shows helpful message."""
    runner = CliRunner()
    result = runner.invoke(main, ["list"])
    # Should not crash — show helpful message
    assert result.exit_code == 0
    assert "No models indexed" in result.output or "Indexed models" in result.output


def test_config_no_init():
    """Test that 'config' without --init shows current config."""
    runner = CliRunner()
    result = runner.invoke(main, ["config"])
    assert result.exit_code == 0
    # Should output configuration
    assert "Current configuration" in result.output or "Config file" in result.output


def test_pull_ollama_no_network():
    """Test that pull_model('ollama') with --source ollama doesn't crash with helpful error."""
    runner = CliRunner()
    result = runner.invoke(main, ["pull", "--source", "ollama", "nonexistent-model-test"])
    # Should fail gracefully (network error or model not found), not crash
    assert result.exit_code != 0


# ---------------------------------------------------------------------------
# New tests for rm, info, and GGUF parser
# ---------------------------------------------------------------------------


def test_rm_help():
    """Test that 'rm --help' shows purge option."""
    runner = CliRunner()
    result = runner.invoke(main, ["rm", "--help"])
    assert result.exit_code == 0
    assert "purge" in result.output.lower()


def test_info_help():
    """Test that 'info --help' shows json option."""
    runner = CliRunner()
    result = runner.invoke(main, ["info", "--help"])
    assert result.exit_code == 0
    assert "json" in result.output.lower()


def test_rm_missing_model():
    """Test that 'rm' with a non-existent model exits with error."""
    runner = CliRunner()
    result = runner.invoke(main, ["rm", "nonexistent"])
    assert result.exit_code != 0
    assert "not found" in result.output.lower()


def test_info_missing_model():
    """Test that 'info' with a non-existent model exits with error."""
    runner = CliRunner()
    result = runner.invoke(main, ["info", "nonexistent"])
    assert result.exit_code != 0
    assert "not found" in result.output.lower()


def test_rm_purge_flag():
    """Test that 'rm --help' output includes the purge flag description."""
    runner = CliRunner()
    result = runner.invoke(main, ["rm", "--help"])
    assert result.exit_code == 0
    assert "delete" in result.output.lower() or "purge" in result.output.lower()


def test_info_json_output():
    """Test that 'info' with --json on a non-existent model exits with error (not a JSON crash)."""
    runner = CliRunner()
    result = runner.invoke(main, ["info", "--json", "nonexistent"])
    assert result.exit_code != 0


def test_gguf_parser_magic_only():
    """Test that the GGUF parser returns None for a file with wrong magic bytes."""
    with tempfile.NamedTemporaryFile(suffix=".gguf", delete=False) as f:
        f.write(b"NOTG" + b"\x00" * 20)
        f.flush()
        path = f.name
    try:
        result = _read_gguf_metadata(path)
        assert result is None
    finally:
        os.unlink(path)


def test_gguf_parser_empty_file():
    """Test that the GGUF parser returns None for an empty file."""
    with tempfile.NamedTemporaryFile(suffix=".gguf", delete=False) as f:
        f.write(b"")
        f.flush()
        path = f.name
    try:
        result = _read_gguf_metadata(path)
        assert result is None
    finally:
        os.unlink(path)


def test_gguf_parser_valid_header_no_kv():
    """Test that the GGUF parser handles a valid header with zero metadata KV pairs."""
    with tempfile.NamedTemporaryFile(suffix=".gguf", delete=False) as f:
        # Magic + version(uint32) + tensor_count(uint64) + kv_count(uint64)
        f.write(b"GGUF")
        f.write(struct.pack("<I", 3))   # version
        f.write(struct.pack("<Q", 10))  # tensor_count
        f.write(struct.pack("<Q", 0))   # metadata_kv_count = 0
        f.flush()
        path = f.name
    try:
        result = _read_gguf_metadata(path)
        assert result is not None
        assert result["magic"] == "GGUF"
        assert result["version"] == 3
        assert result["__tensor_count"] == 10
        assert result["__metadata_kv_count"] == 0
    finally:
        os.unlink(path)


def test_gguf_parser_with_string_kv():
    """Test that the GGUF parser reads a string metadata KV correctly."""
    key = "general.architecture"
    value = "llama"
    with tempfile.NamedTemporaryFile(suffix=".gguf", delete=False) as f:
        f.write(b"GGUF")
        f.write(struct.pack("<I", 3))   # version
        f.write(struct.pack("<Q", 0))   # tensor_count
        f.write(struct.pack("<Q", 1))   # metadata_kv_count = 1
        # key: string (type 8 = string)
        f.write(struct.pack("<Q", len(key)))
        f.write(key.encode("utf-8"))
        # value type = 8 (string)
        f.write(struct.pack("<I", 8))
        # value string
        f.write(struct.pack("<Q", len(value)))
        f.write(value.encode("utf-8"))
        f.flush()
        path = f.name
    try:
        result = _read_gguf_metadata(path)
        assert result is not None
        assert result["general.architecture"] == "llama"
    finally:
        os.unlink(path)


def test_gguf_parser_with_uint32_kv():
    """Test that the GGUF parser reads a uint32 metadata KV correctly."""
    key = "general.context_length"
    value = 4096
    with tempfile.NamedTemporaryFile(suffix=".gguf", delete=False) as f:
        f.write(b"GGUF")
        f.write(struct.pack("<I", 3))   # version
        f.write(struct.pack("<Q", 0))   # tensor_count
        f.write(struct.pack("<Q", 1))   # metadata_kv_count
        # key: string (type 8)
        f.write(struct.pack("<Q", len(key)))
        f.write(key.encode("utf-8"))
        # value type = 4 (uint32)
        f.write(struct.pack("<I", 4))
        f.write(struct.pack("<I", value))
        f.flush()
        path = f.name
    try:
        result = _read_gguf_metadata(path)
        assert result is not None
        assert result["general.context_length"] == 4096
    finally:
        os.unlink(path)


def test_rm_direct_path_not_in_index():
    """Test that 'rm' on a direct path that doesn't exist gives proper error."""
    runner = CliRunner()
    result = runner.invoke(main, ["rm", "/nonexistent/path/model.gguf"])
    assert result.exit_code != 0
    assert "not found" in result.output.lower()


def test_info_direct_path_non_gguf():
    """Test that 'info' on a non-GGUF file shows warning but still displays file info."""
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False, mode="wb") as f:
        f.write(b"some random data that is not a GGUF file")
        f.flush()
        path = f.name
    try:
        runner = CliRunner()
        result = runner.invoke(main, ["info", path])
        assert result.exit_code == 0
        assert "not a gguf" in result.output.lower() or "warning" in result.output.lower()
    finally:
        os.unlink(path)


def test_info_json_direct_path():
    """Test that 'info --json' on a valid GGUF produces JSON output."""
    key = "general.architecture"
    value = "qwen2"
    with tempfile.NamedTemporaryFile(suffix=".gguf", delete=False) as f:
        f.write(b"GGUF")
        f.write(struct.pack("<I", 3))
        f.write(struct.pack("<Q", 0))
        f.write(struct.pack("<Q", 1))
        f.write(struct.pack("<Q", len(key)))
        f.write(key.encode("utf-8"))
        f.write(struct.pack("<I", 8))
        f.write(struct.pack("<Q", len(value)))
        f.write(value.encode("utf-8"))
        f.flush()
        path = f.name
    try:
        runner = CliRunner()
        result = runner.invoke(main, ["info", "--json", path])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["gguf"]["general.architecture"] == "qwen2"
    finally:
        os.unlink(path)


# ---------------------------------------------------------------------------
# REPL mode tests for `ethllama run --interactive`
# ---------------------------------------------------------------------------


def _make_fake_gguf() -> str:
    """Create a minimal valid-looking GGUF file for tests that need a
    resolvable model path but never actually load it."""
    with tempfile.NamedTemporaryFile(suffix=".gguf", delete=False) as f:
        f.write(b"GGUF")
        f.write(struct.pack("<I", 3))   # version
        f.write(struct.pack("<Q", 0))   # tensor_count
        f.write(struct.pack("<Q", 0))   # metadata_kv_count
        path = f.name
    return path


def test_run_interactive_help():
    """`ethllama run --help` documents the -i / --interactive flag."""
    runner = CliRunner()
    result = runner.invoke(main, ["run", "--help"])
    assert result.exit_code == 0
    assert "-i" in result.output
    assert "--interactive" in result.output
    # The help text should explain what it does.
    assert "REPL" in result.output or "interactive" in result.output.lower()


def test_run_interactive_eof():
    """Feeding empty stdin to `ethllama run -i` exits gracefully (exit 0)."""
    model_path = _make_fake_gguf()
    try:
        runner = CliRunner()
        result = runner.invoke(main, ["run", model_path, "-i"], input="")
        assert result.exit_code == 0, (
            f"Expected clean exit on EOF, got {result.exit_code}: {result.output!r}"
        )
        # Should show the welcome banner
        assert "REPL" in result.output
    finally:
        os.unlink(model_path)


def test_run_interactive_exit_command():
    """`/exit` (or `/quit`) cleanly exits the REPL."""
    model_path = _make_fake_gguf()
    try:
        runner = CliRunner()
        result = runner.invoke(
            main, ["run", model_path, "-i"], input="/exit\n"
        )
        assert result.exit_code == 0, (
            f"Expected clean exit on /exit, got {result.exit_code}: {result.output!r}"
        )
        # Should mention goodbye
        assert "Goodbye" in result.output
    finally:
        os.unlink(model_path)


def test_run_interactive_question(monkeypatch):
    """Feeding a question in REPL mode calls run_inference_stream and
    streams the response back to stdout."""
    model_path = _make_fake_gguf()
    captured: list = []

    def fake_stream(*args, **kwargs):
        captured.append((args, kwargs))
        yield "Hello"
        yield " "
        yield "world"

    # Patch the module-level names so the lazy import in cli.run() picks
    # them up, and REPLSession.send() (which also references the module
    # attributes) uses them too.
    monkeypatch.setattr("ethllama.inference.has_inference_engine", lambda: True)
    monkeypatch.setattr("ethllama.inference.run_inference_stream", fake_stream)

    try:
        runner = CliRunner()
        result = runner.invoke(
            main, ["run", model_path, "-i"], input="What is 2+2?\n"
        )
        assert result.exit_code == 0, (
            f"Expected clean exit, got {result.exit_code}: {result.output!r}"
        )
        # The mock should have been invoked exactly once (one user turn)
        assert len(captured) == 1, (
            f"Expected 1 stream call, got {len(captured)}: {captured}"
        )
        # The streamed chunks should appear in the output
        assert "Hello" in result.output
        assert "world" in result.output
    finally:
        os.unlink(model_path)


# ---------------------------------------------------------------------------
# Spec-required tests for rm, info (added by cli_mgmt implementation)
# ---------------------------------------------------------------------------


def test_rm_from_index(tmp_path, monkeypatch):
    """`ethllama rm` removes an indexed model from the index."""
    from ethllama import index as index_mod

    # Redirect the index file to a tmp location
    monkeypatch.setattr(index_mod, "INDEX_FILE", tmp_path / "index.json")

    # Create a fake model file and index it
    fake_model = tmp_path / "fake-model.gguf"
    fake_model.write_bytes(b"GGUF\x03\x00\x00\x00" + b"\x00" * 32)
    index_mod.add_to_index(str(fake_model))
    assert str(fake_model) in str(index_mod.load_index())

    runner = CliRunner()
    result = runner.invoke(main, ["rm", "fake-model.gguf"])
    assert result.exit_code == 0
    assert "Removed" in result.output

    # Verify the index no longer contains the model
    assert str(fake_model) not in str(index_mod.load_index())


def test_rm_purge_deletes_file(tmp_path, monkeypatch):
    """`ethllama rm --purge --yes` removes the file from disk."""
    from ethllama import index as index_mod

    monkeypatch.setattr(index_mod, "INDEX_FILE", tmp_path / "index.json")

    # Create a real (tiny) file on disk and index it
    fake_model = tmp_path / "purge-me.gguf"
    fake_model.write_bytes(b"GGUF\x03\x00\x00\x00" + b"\x00" * 32)
    index_mod.add_to_index(str(fake_model))
    assert fake_model.exists()

    runner = CliRunner()
    result = runner.invoke(main, ["rm", "--purge", "--yes", "purge-me.gguf"])
    assert result.exit_code == 0, result.output
    assert "Deleted" in result.output

    # File should be gone, and the index entry should be gone
    assert not fake_model.exists()
    assert str(fake_model) not in str(index_mod.load_index())


def test_rm_purge_confirms_without_yes(tmp_path, monkeypatch):
    """`ethllama rm --purge` without --yes on a TTY aborts the deletion."""
    from ethllama import index as index_mod

    monkeypatch.setattr(index_mod, "INDEX_FILE", tmp_path / "index.json")

    fake_model = tmp_path / "do-not-delete.gguf"
    fake_model.write_bytes(b"GGUF\x03\x00\x00\x00" + b"\x00" * 32)
    index_mod.add_to_index(str(fake_model))

    runner = CliRunner()
    # No --yes and no stdin input -> click.confirm should abort
    result = runner.invoke(main, ["rm", "--purge", "do-not-delete.gguf"], input="n\n")
    # Aborted -> click exits with 1
    assert result.exit_code != 0
    # File should still exist (the abort prevented the deletion)
    assert fake_model.exists()


def test_info_real_gguf(tmp_path):
    """`ethllama info` on a real (synthetic) GGUF returns the parsed fields."""
    # Build a small synthetic GGUF with all the fields info cares about
    key_arch = "general.architecture"
    val_arch = "qwen3"
    key_ctx = "qwen3.context_length"
    val_ctx = 32768
    key_emb = "qwen3.embedding_length"
    val_emb = 1024
    key_ftype = "general.file_type"
    val_ftype = 20  # Q4_K_M per the GGUF enum
    key_n = "general.parameter_count"
    val_n = 800_000_000  # 0.8B

    data = bytearray()
    data += b"GGUF"
    data += struct.pack("<I", 3)        # version
    data += struct.pack("<Q", 197)      # tensor_count
    data += struct.pack("<Q", 5)        # metadata_kv_count

    def write_str(s: str) -> None:
        data.extend(struct.pack("<Q", len(s)))
        data.extend(s.encode("utf-8"))

    def write_kv(k: str, vtype: int, payload: bytes) -> None:
        write_str(k)
        data.extend(struct.pack("<I", vtype))
        data.extend(payload)

    # general.architecture (string, type 8)
    payload = struct.pack("<Q", len(val_arch)) + val_arch.encode("utf-8")
    write_kv(key_arch, 8, payload)
    # qwen3.context_length (uint32, type 4)
    write_kv(key_ctx, 4, struct.pack("<I", val_ctx))
    # qwen3.embedding_length (uint32, type 4)
    write_kv(key_emb, 4, struct.pack("<I", val_emb))
    # general.file_type (uint32, type 4)
    write_kv(key_ftype, 4, struct.pack("<I", val_ftype))
    # general.parameter_count (uint64, type 10) — 1.5B for a clean B format
    val_n = 1_500_000_000
    write_kv(key_n, 10, struct.pack("<Q", val_n))

    gguf_path = tmp_path / "synthetic-qwen3.gguf"
    gguf_path.write_bytes(bytes(data))

    runner = CliRunner()
    result = runner.invoke(main, ["info", str(gguf_path)])
    assert result.exit_code == 0, result.output

    out = result.output
    assert "Model: synthetic-qwen3.gguf" in out
    assert "Architecture:      qwen3" in out
    assert "Context length:    32768" in out
    assert "Embedding length:  1024" in out
    assert "GGUF version:      3" in out
    assert "Quantization:      Q4_K_M" in out
    assert "Parameters:        1.5B" in out


def test_info_verbose_shows_all_kv(tmp_path):
    """`ethllama info --verbose` lists every metadata KV pair."""
    data = bytearray()
    data += b"GGUF"
    data += struct.pack("<I", 3)        # version
    data += struct.pack("<Q", 0)        # tensor_count
    data += struct.pack("<Q", 2)        # metadata_kv_count

    def write_str(s: str) -> None:
        data.extend(struct.pack("<Q", len(s)))
        data.extend(s.encode("utf-8"))

    # general.name (string)
    write_str("general.name")
    data.extend(struct.pack("<I", 8))  # value type 8 = string
    name_value = "my-fancy-model"
    data.extend(struct.pack("<Q", len(name_value)))
    data.extend(name_value.encode("utf-8"))

    # general.block_count (uint32)
    write_str("general.block_count")
    data.extend(struct.pack("<I", 4))
    data.extend(struct.pack("<I", 32))

    gguf_path = tmp_path / "verbose-test.gguf"
    gguf_path.write_bytes(bytes(data))

    runner = CliRunner()
    result = runner.invoke(main, ["info", "--verbose", str(gguf_path)])
    assert result.exit_code == 0, result.output
    out = result.output
    assert "All metadata KV pairs" in out
    assert "general.name: my-fancy-model" in out
    assert "general.block_count: 32" in out


def test_list_all_indexed_helper(tmp_path, monkeypatch):
    """The list_all_indexed() helper returns a flat list of indexed models."""
    from ethllama import index as index_mod
    from ethllama.cli_mgmt import list_all_indexed

    monkeypatch.setattr(index_mod, "INDEX_FILE", tmp_path / "index.json")

    # Empty case
    assert list_all_indexed() == []

    # Index two models
    m1 = tmp_path / "a.gguf"
    m1.write_bytes(b"x")
    m2 = tmp_path / "b.gguf"
    m2.write_bytes(b"x")
    index_mod.add_to_index(str(m1))
    index_mod.add_to_index(str(m2))

    flat = list_all_indexed()
    assert len(flat) == 2
    filenames = {e["filename"] for e in flat}
    assert filenames == {"a.gguf", "b.gguf"}
    # Each entry should have a dir_path attribute
    for entry in flat:
        assert "dir_path" in entry
        assert entry["dir_path"] == str(tmp_path)

# ---------------------------------------------------------------------------
# New tests for the `transcribe` (STT) subcommand
# ---------------------------------------------------------------------------

from ethllama.cli_stt import transcribe_cmd, register_commands as register_stt
from ethllama.stt import WhisperBinaryNotFound

# Wire the STT subcommand onto the main group so that
# `ethllama transcribe ...` resolves just like the other subcommands.
register_stt(main)


def test_transcribe_help():
    """`transcribe --help` shows the correct option set."""
    runner = CliRunner()
    result = runner.invoke(main, ["transcribe", "--help"])
    assert result.exit_code == 0
    out = result.output.lower()
    # AUDIO_FILE argument is mentioned
    assert "audio_file" in out or "audio file" in out
    # All required options
    assert "--engine" in out
    assert "--model" in out
    assert "--output-format" in out
    assert "--language" in out
    assert "--threads" in out
    assert "--output" in out


def test_transcribe_direct_help():
    """The standalone transcribe_cmd shows help when invoked directly."""
    runner = CliRunner()
    result = runner.invoke(transcribe_cmd, ["--help"])
    assert result.exit_code == 0
    assert "transcribe" in result.output.lower()


def test_transcribe_missing_audio():
    """`transcribe` with a non-existent audio file exits with a click error."""
    runner = CliRunner()
    result = runner.invoke(
        main, ["transcribe", "/nonexistent/path/to/audio.wav", "--model", "/tmp/m.bin"]
    )
    # click.Path(exists=True) catches this and exits non-zero
    assert result.exit_code != 0


def test_transcribe_missing_engine(tmp_path, monkeypatch):
    """`transcribe` shows a helpful error when no whisper engine is found."""

    # Create a real audio file so we pass the click.Path(exists=True) check
    audio = tmp_path / "sample.wav"
    audio.write_bytes(b"")
    model = tmp_path / "model.bin"
    model.write_bytes(b"")

    # Make engine discovery fail
    import ethllama.cli_stt as cli_stt_mod

    def _raise(_engine_config=None):
        raise WhisperBinaryNotFound(
            "No whisper.cpp binary found. Install whisper.cpp (provides "
            "`whisper-cli`) or set `binary` in a whisper-cpp engine config "
            "under ~/.ethllama/engines/."
        )

    monkeypatch.setattr(cli_stt_mod, "find_whisper_binary", _raise)
    # Make engine resolution return None (no engine configs installed)
    monkeypatch.setattr(cli_stt_mod, "get_engine", lambda _n: None)
    monkeypatch.setattr(cli_stt_mod, "load_engines", lambda: {})

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "transcribe", str(audio),
            "--model", str(model),
        ],
    )
    assert result.exit_code != 0
    # The error message must mention something useful
    out = (result.output or "").lower()
    assert "whisper" in out or "binary" in out or "engine" in out


def test_transcribe_missing_engine_named(tmp_path, monkeypatch):
    """`transcribe --engine missing` reports the missing engine by name."""

    audio = tmp_path / "sample.wav"
    audio.write_bytes(b"")

    # Make get_engine return None for the named engine
    import ethllama.cli_stt as cli_stt_mod

    monkeypatch.setattr(cli_stt_mod, "get_engine", lambda _n: None)

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "transcribe", str(audio),
            "--engine", "nonexistent-engine",
        ],
    )
    assert result.exit_code != 0
    out = (result.output or "").lower()
    assert "nonexistent-engine" in out or "not found" in out


# ---------------------------------------------------------------------------
# Chat template extraction (GGUF metadata) and format_chat_messages
# ---------------------------------------------------------------------------

from ethllama.inference import read_chat_template, format_chat_messages


def _build_minimal_gguf(
    kv_pairs: list,
    *,
    version: int = 3,
    tensor_count: int = 0,
) -> bytes:
    """Build a synthetic GGUF v3 byte string with the given KV pairs.

    Each KV pair is a ``(key: str, value_type: int, payload: bytes)``
    tuple where ``payload`` is the already-encoded value body
    (e.g. ``struct.pack("<Q", n) + string.encode("utf-8")`` for a
    string).  The GGUF magic, version, tensor count, and KV count are
    written automatically.
    """
    data = bytearray()
    data += b"GGUF"
    data += struct.pack("<I", version)
    data += struct.pack("<Q", tensor_count)
    data += struct.pack("<Q", len(kv_pairs))
    for key, vtype, payload in kv_pairs:
        data += struct.pack("<Q", len(key))
        data += key.encode("utf-8")
        data += struct.pack("<I", vtype)
        data += payload
    return bytes(data)


def _gguf_string_payload(value: str) -> bytes:
    """Encode a Python string as a GGUF string value body."""
    encoded = value.encode("utf-8")
    return struct.pack("<Q", len(encoded)) + encoded


def test_read_chat_template_basic():
    """read_chat_template extracts the tokenizer.chat_template KV."""
    expected = (
        "<|user|>\n{{ .Prompt }}<|end|>\n<|assistant|>\n"
    )
    payload = _gguf_string_payload(expected)
    blob = _build_minimal_gguf(
        [("tokenizer.chat_template", 8, payload)]
    )
    with tempfile.NamedTemporaryFile(suffix=".gguf", delete=False) as f:
        f.write(blob)
        path = f.name
    try:
        assert read_chat_template(path) == expected
    finally:
        os.unlink(path)


def test_read_chat_template_empty_file():
    """read_chat_template returns None for an empty file."""
    with tempfile.NamedTemporaryFile(suffix=".gguf", delete=False) as f:
        f.write(b"")
        path = f.name
    try:
        assert read_chat_template(path) is None
    finally:
        os.unlink(path)


def test_read_chat_template_wrong_magic():
    """read_chat_template returns None when magic bytes are not GGUF."""
    with tempfile.NamedTemporaryFile(suffix=".gguf", delete=False) as f:
        f.write(b"NOTG" + b"\x00" * 24)
        path = f.name
    try:
        assert read_chat_template(path) is None
    finally:
        os.unlink(path)


def test_read_chat_template_unsupported_version():
    """read_chat_template returns None for GGUF versions other than 3."""
    blob = _build_minimal_gguf(
        [("tokenizer.chat_template", 8, _gguf_string_payload("x"))],
        version=2,
    )
    with tempfile.NamedTemporaryFile(suffix=".gguf", delete=False) as f:
        f.write(blob)
        path = f.name
    try:
        assert read_chat_template(path) is None
    finally:
        os.unlink(path)


def test_read_chat_template_key_not_present():
    """read_chat_template returns None when key is missing from metadata."""
    payload = _gguf_string_payload("llama")
    blob = _build_minimal_gguf(
        [("general.architecture", 8, payload)]
    )
    with tempfile.NamedTemporaryFile(suffix=".gguf", delete=False) as f:
        f.write(blob)
        path = f.name
    try:
        assert read_chat_template(path) is None
    finally:
        os.unlink(path)


def test_read_chat_template_later_in_metadata():
    """read_chat_template finds the key even when not the first KV pair."""
    expected = "[INST] {{ .Prompt }} [/INST]"
    blob = _build_minimal_gguf([
        ("general.architecture", 8, _gguf_string_payload("llama")),
        ("general.name", 8, _gguf_string_payload("test-model")),
        ("tokenizer.chat_template", 8, _gguf_string_payload(expected)),
    ])
    with tempfile.NamedTemporaryFile(suffix=".gguf", delete=False) as f:
        f.write(blob)
        path = f.name
    try:
        assert read_chat_template(path) == expected
    finally:
        os.unlink(path)


def test_read_chat_template_skips_non_string_kvs():
    """read_chat_template skips uint/array KVs that precede the template."""
    expected = "<|start_header_id|>{{ .Role }}<|end_header_id|>\n"
    blob = _build_minimal_gguf([
        ("general.architecture", 8, _gguf_string_payload("llama")),
        ("general.context_length", 4, struct.pack("<I", 4096)),
        ("general.file_type", 4, struct.pack("<I", 1)),
        ("tokenizer.chat_template", 8, _gguf_string_payload(expected)),
    ])
    with tempfile.NamedTemporaryFile(suffix=".gguf", delete=False) as f:
        f.write(blob)
        path = f.name
    try:
        assert read_chat_template(path) == expected
    finally:
        os.unlink(path)


def test_format_chat_messages_uses_template_when_provided():
    """format_chat_messages uses the GGUF chat template when available."""
    template = (
        "<|user|>\n{{ .Prompt }}<|end|>\n<|assistant|>\n"
    )
    blob = _build_minimal_gguf(
        [("tokenizer.chat_template", 8, _gguf_string_payload(template))]
    )
    with tempfile.NamedTemporaryFile(suffix=".gguf", delete=False) as f:
        f.write(blob)
        path = f.name
    try:
        out = format_chat_messages(
            [{"role": "user", "content": "Hello"}],
            model_path=path,
        )
        assert out == "<|user|>\nHello<|end|>\n<|assistant|>\n"
    finally:
        os.unlink(path)


def test_format_chat_messages_falls_back_when_no_template():
    """format_chat_messages falls back to the hardcoded format."""
    blob = _build_minimal_gguf([])  # no KV pairs
    with tempfile.NamedTemporaryFile(suffix=".gguf", delete=False) as f:
        f.write(blob)
        path = f.name
    try:
        out = format_chat_messages(
            [{"role": "user", "content": "Hi"}],
            model_path=path,
        )
        assert "<|im_start|>" in out
        assert "Hi" in out
    finally:
        os.unlink(path)


def test_format_chat_messages_legacy_call_no_model_path():
    """format_chat_messages with no model_path uses the hardcoded format."""
    out = format_chat_messages(
        [
            {"role": "system", "content": "You are terse."},
            {"role": "user", "content": "Hi"},
        ]
    )
    assert "<|im_start|>system" in out
    assert "You are terse." in out
    assert "<|im_start|>user" in out
    assert "Hi" in out
    # The legacy function should still end with the assistant prompt
    assert out.rstrip().endswith("<|im_start|>assistant")


def test_format_chat_messages_template_with_system_and_response():
    """The template substitution handles System and Response placeholders."""
    template = (
        "<|system|>{{ .System }}<|/system|>"
        "<|user|>{{ .Prompt }}<|/user|>"
        "<|assistant|>{{ .Response }}"
    )
    blob = _build_minimal_gguf(
        [("tokenizer.chat_template", 8, _gguf_string_payload(template))]
    )
    with tempfile.NamedTemporaryFile(suffix=".gguf", delete=False) as f:
        f.write(blob)
        path = f.name
    try:
        out = format_chat_messages(
            [
                {"role": "system", "content": "Be kind."},
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello!"},
            ],
            model_path=path,
        )
        assert "Be kind." in out
        assert "Hi" in out
        assert "Hello!" in out
    finally:
        os.unlink(path)



# ---------------------------------------------------------------------------
# Per-model config (model_defaults in config.yaml)
# ---------------------------------------------------------------------------


def _isolated_config(monkeypatch, cfg: dict):
    """Patch ethllama.cli.load_config to return a custom config.

    Also patches ethllama.config.load_config (used by other modules
    that may lazily load) so the test sees a single consistent view.
    """
    import ethllama.cli as cli_mod
    import ethllama.config as config_mod
    monkeypatch.setattr(cli_mod, "load_config", lambda: cfg)
    monkeypatch.setattr(config_mod, "load_config", lambda: cfg)


def _isolate_index(monkeypatch, tmp_path):
    """Redirect the model index file to a tmp path so resolve_model_path
    can never accidentally find a real indexed model."""
    from ethllama import index as index_mod
    monkeypatch.setattr(index_mod, "INDEX_FILE", tmp_path / "index.json")


def test_load_model_defaults_returns_empty_for_unknown_stem():
    """_load_model_defaults returns {} when the model stem is not configured."""
    import ethllama.cli as cli_mod
    cfg = {"model_defaults": {"known-model": {"temperature": 0.1}}}
    assert cli_mod._load_model_defaults(cfg, "unknown-model") == {}


def test_load_model_defaults_returns_entry_for_known_stem():
    """_load_model_defaults returns the configured dict for a known stem."""
    import ethllama.cli as cli_mod
    cfg = {"model_defaults": {"phi-4": {"temperature": 0.42, "top_k": 12}}}
    out = cli_mod._load_model_defaults(cfg, "phi-4")
    assert out == {"temperature": 0.42, "top_k": 12}


def test_load_model_defaults_handles_missing_section():
    """_load_model_defaults returns {} when model_defaults is absent."""
    import ethllama.cli as cli_mod
    assert cli_mod._load_model_defaults({}, "anything") == {}


def test_load_model_defaults_handles_non_dict_section():
    """_load_model_defaults returns {} when model_defaults is not a dict."""
    import ethllama.cli as cli_mod
    assert cli_mod._load_model_defaults({"model_defaults": []}, "x") == {}


def test_load_model_defaults_handles_non_dict_entry():
    """_load_model_defaults returns {} when the entry is not a dict."""
    import ethllama.cli as cli_mod
    cfg = {"model_defaults": {"phi-4": "this is not a dict"}}
    assert cli_mod._load_model_defaults(cfg, "phi-4") == {}


def test_per_model_config_applied_in_run(tmp_path, monkeypatch):
    """`ethllama run` picks up model_defaults for the resolved model stem.

    Verifies the verbose log line and that the effective inference
    call receives the per-model temperature (not the CLI default).
    """
    from ethllama import index as index_mod
    _isolate_index(monkeypatch, tmp_path)

    # Build a model with a known stem
    model_name = "phi-4-mini-test"
    model_path = tmp_path / f"{model_name}.gguf"
    model_path.write_bytes(b"GGUF" + b"\x00" * 32)

    # Config with per-model defaults
    cfg = {
        "gpu": {"backend": "cpu", "fallback": True},
        "model_defaults": {
            model_name: {
                "temperature": 0.25,
                "top_k": 12,
                "top_p": 0.7,
                "n_gpu_layers": 99,
                "system_prompt": "You are concise.",
            }
        },
    }
    _isolated_config(monkeypatch, cfg)

    # Patch inference so we capture the call (and don't need a real engine)
    captured: dict = {}
    monkeypatch.setattr(
        "ethllama.inference.has_inference_engine", lambda: True
    )

    def fake_run_inference(**kwargs):
        captured.update(kwargs)
        return "ok"

    monkeypatch.setattr("ethllama.inference.run_inference", fake_run_inference)

    runner = CliRunner()
    result = runner.invoke(
        main, ["run", str(model_path), "-p", "Hello"]
    )

    # The CLI should have applied the per-model config
    assert result.exit_code == 0, result.output
    assert captured, f"run_inference was not called. output={result.output!r}"
    assert captured["temperature"] == 0.25
    assert captured["top_k"] == 12
    assert captured["top_p"] == 0.7
    assert captured["n_gpu_layers"] == 99
    # The verbose log line should be on stderr
    assert "per-model config" in result.output
    # And the system prompt should have been prepended to the user prompt
    assert "You are concise." in captured["prompt"]
    assert "Hello" in captured["prompt"]


def test_cli_flag_overrides_per_model_default(tmp_path, monkeypatch):
    """Explicit CLI flags (e.g. --temperature) take precedence over
    the value configured in model_defaults."""
    from ethllama import index as index_mod
    _isolate_index(monkeypatch, tmp_path)

    model_name = "phi-4-cli-override"
    model_path = tmp_path / f"{model_name}.gguf"
    model_path.write_bytes(b"GGUF" + b"\x00" * 32)

    cfg = {
        "gpu": {"backend": "cpu", "fallback": True},
        "model_defaults": {
            model_name: {
                "temperature": 0.25,
                "top_k": 12,
                "n_gpu_layers": 99,
            }
        },
    }
    _isolated_config(monkeypatch, cfg)

    captured: dict = {}
    monkeypatch.setattr(
        "ethllama.inference.has_inference_engine", lambda: True
    )
    monkeypatch.setattr(
        "ethllama.inference.run_inference",
        lambda **kw: (captured.update(kw) or "ok"),
    )

    runner = CliRunner()
    # User explicitly passes --temperature 0.9; top_k and n_gpu_layers
    # are left at CLI defaults so the per-model values should win for those.
    result = runner.invoke(
        main,
        ["run", str(model_path), "-p", "Hi", "--temperature", "0.9"],
    )
    assert result.exit_code == 0, result.output
    assert captured["temperature"] == 0.9, (
        "Explicit --temperature must override per-model default; "
        f"got {captured.get('temperature')!r}"
    )
    # top_k and n_gpu_layers were not given on CLI, so per-model wins
    assert captured["top_k"] == 12
    assert captured["n_gpu_layers"] == 99


def test_unknown_model_stem_gets_no_defaults(tmp_path, monkeypatch):
    """When the model stem is not in model_defaults, the CLI flags
    are used as-is and the verbose per-model log is NOT printed."""
    from ethllama import index as index_mod
    _isolate_index(monkeypatch, tmp_path)

    model_path = tmp_path / "completely-unknown-model.gguf"
    model_path.write_bytes(b"GGUF" + b"\x00" * 32)

    cfg = {
        "gpu": {"backend": "cpu", "fallback": True},
        "model_defaults": {
            "some-other-model": {"temperature": 0.1},
        },
    }
    _isolated_config(monkeypatch, cfg)

    captured: dict = {}
    monkeypatch.setattr(
        "ethllama.inference.has_inference_engine", lambda: True
    )
    monkeypatch.setattr(
        "ethllama.inference.run_inference",
        lambda **kw: (captured.update(kw) or "ok"),
    )

    runner = CliRunner()
    result = runner.invoke(
        main, ["run", str(model_path), "-p", "Hi"]
    )
    assert result.exit_code == 0, result.output
    # CLI defaults should be in effect (no per-model override)
    assert captured["temperature"] == 0.7
    assert captured["top_k"] == 40
    # The per-model verbose log should NOT appear
    assert "per-model config" not in result.output


def test_per_model_chat_template_file(tmp_path, monkeypatch):
    """When model_defaults.chat_template points to a file, the
    inference call uses that template's rendered output."""
    from ethllama import index as index_mod
    _isolate_index(monkeypatch, tmp_path)

    model_name = "phi-4-with-template"
    model_path = tmp_path / f"{model_name}.gguf"
    model_path.write_bytes(b"GGUF" + b"\x00" * 32)

    template_path = tmp_path / "custom.jinja"
    template_path.write_text(
        "SYS:{{ .System }}|USR:{{ .Prompt }}|REPLY:",
        encoding="utf-8",
    )

    cfg = {
        "gpu": {"backend": "cpu", "fallback": True},
        "model_defaults": {
            model_name: {
                "chat_template": str(template_path),
                "system_prompt": "I am terse.",
            }
        },
    }
    _isolated_config(monkeypatch, cfg)

    captured: dict = {}
    monkeypatch.setattr(
        "ethllama.inference.has_inference_engine", lambda: True
    )
    monkeypatch.setattr(
        "ethllama.inference.run_inference",
        lambda **kw: (captured.update(kw) or "ok"),
    )

    runner = CliRunner()
    result = runner.invoke(
        main, ["run", str(model_path), "-p", "Hello there"]
    )
    assert result.exit_code == 0, result.output
    # The template substitutes .System and .Prompt verbatim
    assert captured["prompt"].startswith("SYS:I am terse.|USR:Hello there|REPLY:")


def test_per_model_config_does_not_apply_when_stem_differs(tmp_path, monkeypatch):
    """Per-model defaults are only used when the model stem matches
    exactly.  Similar-but-different stems must NOT inherit settings."""
    from ethllama import index as index_mod
    _isolate_index(monkeypatch, tmp_path)

    # Create a model whose stem does NOT match the configured key
    model_path = tmp_path / "qwen2.5-7B-Instruct-Q4_K_M.gguf"
    model_path.write_bytes(b"GGUF" + b"\x00" * 32)

    cfg = {
        "gpu": {"backend": "cpu", "fallback": True},
        "model_defaults": {
            "qwen2.5-7B-Instruct-Q5_K_M": {  # note: Q5, not Q4
                "temperature": 0.11,
            }
        },
    }
    _isolated_config(monkeypatch, cfg)

    captured: dict = {}
    monkeypatch.setattr(
        "ethllama.inference.has_inference_engine", lambda: True
    )
    monkeypatch.setattr(
        "ethllama.inference.run_inference",
        lambda **kw: (captured.update(kw) or "ok"),
    )

    runner = CliRunner()
    result = runner.invoke(
        main, ["run", str(model_path), "-p", "Hi"]
    )
    assert result.exit_code == 0, result.output
    # No per-model default for this exact stem -> CLI default wins
    assert captured["temperature"] == 0.7
    assert "per-model config" not in result.output


def test_format_chat_messages_with_template_path_file(tmp_path):
    """format_chat_messages reads the template from chat_template_path
    when given a path to a real file."""
    template_path = tmp_path / "tpl.jinja"
    template_path.write_text("[INST] {{ .Prompt }} [/INST]", encoding="utf-8")
    out = format_chat_messages(
        [{"role": "user", "content": "Hello"}],
        chat_template_path=str(template_path),
    )
    assert out == "[INST] Hello [/INST]"


def test_format_chat_messages_template_path_falls_back_on_missing_file(tmp_path):
    """A missing chat_template_path file should not crash; the function
    falls through to the GGUF template or the hardcoded format."""
    missing = tmp_path / "does-not-exist.jinja"
    # No model_path, no working template -> hardcoded fallback
    out = format_chat_messages(
        [{"role": "user", "content": "Hello"}],
        chat_template_path=str(missing),
    )
    assert "<|im_start|>" in out
    assert "Hello" in out


def test_format_chat_messages_template_path_takes_precedence_over_gguf(tmp_path):
    """The explicit chat_template_path wins over the GGUF-baked template."""
    # Build a GGUF with one template
    gguf_template = "GGUF:{{ .Prompt }}"
    blob = _build_minimal_gguf(
        [("tokenizer.chat_template", 8, _gguf_string_payload(gguf_template))]
    )
    gguf_path = tmp_path / "m.gguf"
    gguf_path.write_bytes(blob)

    # And a file-based template that should win
    file_template = tmp_path / "explicit.jinja"
    file_template.write_text("FILE:{{ .Prompt }}", encoding="utf-8")

    out = format_chat_messages(
        [{"role": "user", "content": "Hi"}],
        model_path=str(gguf_path),
        chat_template_path=str(file_template),
    )
    assert out == "FILE:Hi"
    assert "GGUF:" not in out


def test_serve_applies_per_model_config(tmp_path, monkeypatch):
    """`ethllama serve --model <m>` with a per-model config in
    config.yaml propagates n_gpu_layers / n_threads / gpu_backend to
    set_gpu_config()."""
    from ethllama import index as index_mod
    import ethllama.inference as inf_mod
    _isolate_index(monkeypatch, tmp_path)

    model_name = "phi-4-serve-test"
    model_path = tmp_path / f"{model_name}.gguf"
    model_path.write_bytes(b"GGUF" + b"\x00" * 32)

    cfg = {
        "gpu": {"backend": "vulkan", "fallback": True},
        "model_defaults": {
            model_name: {
                "n_gpu_layers": 42,
                "threads": 7,
                "gpu_backend": "cuda",
                "ctx_size": 2048,
            }
        },
    }
    _isolated_config(monkeypatch, cfg)

    captured_gpu: dict = {}
    monkeypatch.setattr(
        inf_mod, "set_gpu_config", lambda **kw: captured_gpu.update(kw)
    )

    captured_serve: dict = {}

    def fake_run_server(*args, **kwargs):
        captured_serve.update(kwargs)
        raise KeyboardInterrupt()

    import builtins
    real_import = builtins.__import__

    def fake_import(name, *a, **kw):
        mod = real_import(name, *a, **kw)
        if name.endswith("api") or name == "ethllama.api":
            mod.run_server = fake_run_server  # type: ignore[attr-defined]
        return mod

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.setattr("ethllama.cli.resolve_model_path", lambda _m: str(model_path))

    runner = CliRunner()
    result = runner.invoke(
        main, ["serve", "--model", str(model_path), "--host", "127.0.0.1", "--port", "1"]
    )

    # set_gpu_config was called with the per-model values
    assert captured_gpu.get("n_gpu_layers") == 42
    assert captured_gpu.get("n_threads") == 7
    assert captured_gpu.get("gpu_backend") == "cuda"
    assert captured_gpu.get("ctx_size") == 2048
    # And the model is forwarded to run_server
    assert captured_serve.get("model_path") == str(model_path)

# ---------------------------------------------------------------------------
# SSL / HTTPS support for `ethllama serve`
# ---------------------------------------------------------------------------


def test_serve_help_shows_ssl_options():
    """`ethllama serve --help` documents the --ssl-* options."""
    runner = CliRunner()
    result = runner.invoke(main, ["serve", "--help"])
    assert result.exit_code == 0
    out = result.output
    assert "--ssl-keyfile" in out
    assert "--ssl-certfile" in out
    assert "--ssl-keyfile-password" in out
    assert "--ssl-ca-certs" in out
    # The help text should explain that the options enable HTTPS
    assert "HTTPS" in out


def test_serve_default_port_is_10434():
    """`ethllama serve` defaults to port 10434 (Ollama homage)."""
    runner = CliRunner()
    result = runner.invoke(main, ["serve", "--help"])
    assert result.exit_code == 0
    # show_default=True prints the default in the help text
    assert "10434" in result.output
    # The old default (8080) must not appear as a default value
    assert "8080" not in result.output


def test_serve_passes_ssl_to_run_server(monkeypatch, tmp_path):
    """`ethllama serve --ssl-*` passes the SSL args through to run_server()."""
    import builtins
    import ethllama.cli as cli_mod

    # Create dummy PEM files on disk so click.Path(exists=True) accepts them
    keyfile = tmp_path / "server.key"
    keyfile.write_text("dummy key content")
    certfile = tmp_path / "server.crt"
    certfile.write_text("dummy cert content")
    ca_certs = tmp_path / "ca.pem"
    ca_certs.write_text("dummy ca content")

    captured: dict = {}

    def fake_run_server(*args, **kwargs):
        captured.update(kwargs)
        # Break out of the server loop without actually listening
        raise KeyboardInterrupt()

    # Patch the lazy import so the test doesn't need fastapi installed
    real_import = builtins.__import__

    def fake_import(name, *a, **kw):
        mod = real_import(name, *a, **kw)
        if name.endswith("api") or name == "ethllama.api":
            mod.run_server = fake_run_server  # type: ignore[attr-defined]
        return mod

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.setattr(cli_mod, "resolve_model_path", lambda _m: "")
    monkeypatch.setattr(cli_mod, "load_config", lambda: {})

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "serve",
            "--ssl-keyfile", str(keyfile),
            "--ssl-certfile", str(certfile),
            "--ssl-keyfile-password", "supersecret",
            "--ssl-ca-certs", str(ca_certs),
        ],
    )

    # run_server was called with the SSL args forwarded
    assert captured.get("ssl_keyfile") == str(keyfile)
    assert captured.get("ssl_certfile") == str(certfile)
    assert captured.get("ssl_keyfile_password") == "supersecret"
    assert captured.get("ssl_ca_certs") == str(ca_certs)
    # And the port should default to 10434
    assert captured.get("port") == 10434


def test_serve_ssl_only_keyfile_warns_but_does_not_pass_partial_ssl(monkeypatch, tmp_path):
    """Providing only --ssl-keyfile (without --ssl-certfile) still calls
    run_server with both values (the API layer logs a warning and falls
    back to HTTP)."""
    import builtins
    import ethllama.cli as cli_mod

    keyfile = tmp_path / "server.key"
    keyfile.write_text("dummy key content")

    captured: dict = {}

    def fake_run_server(*args, **kwargs):
        captured.update(kwargs)
        raise KeyboardInterrupt()

    real_import = builtins.__import__

    def fake_import(name, *a, **kw):
        mod = real_import(name, *a, **kw)
        if name.endswith("api") or name == "ethllama.api":
            mod.run_server = fake_run_server  # type: ignore[attr-defined]
        return mod

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.setattr(cli_mod, "resolve_model_path", lambda _m: "")
    monkeypatch.setattr(cli_mod, "load_config", lambda: {})

    runner = CliRunner()
    result = runner.invoke(main, ["serve", "--ssl-keyfile", str(keyfile)])

    # SSL args are passed through to run_server (warning is the API's job)
    assert captured.get("ssl_keyfile") == str(keyfile)
    assert captured.get("ssl_certfile") is None


# ---------------------------------------------------------------------------
# Engine discovery: `ethllama discover`
# ---------------------------------------------------------------------------


def test_discover_help():
    """`ethllama discover --help` shows expected options and the help text."""
    runner = CliRunner()
    result = runner.invoke(main, ["discover", "--help"])
    assert result.exit_code == 0
    out = result.output
    assert "PATH" in out
    assert "--overwrite" in out
    assert "--no-generate" in out
    # The help text references the catalogue
    assert "ollama" in out or "llama-cli" in out


def test_discover_appears_in_main_help():
    """The discover command is listed in the main help output."""
    runner = CliRunner()
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "discover" in result.output


def test_discover_engines_no_args_no_engines(monkeypatch, tmp_path):
    """`ethllama discover` with nothing on PATH prints a friendly message."""
    import ethllama.cli as cli_mod

    # Force discover_engines to return empty (no engines on test PATH)
    monkeypatch.setattr(
        cli_mod, "discover_engines", lambda binary_name=None: {}
    )

    runner = CliRunner()
    result = runner.invoke(main, ["discover"])
    assert result.exit_code == 0
    out = result.output
    assert "No inference engines found" in out
    # The message lists known engine names so the user knows what was looked for
    assert "ollama" in out
    assert "llama-cli" in out


def test_discover_engines_finds_multiple(monkeypatch, tmp_path):
    """`ethllama discover` reports each found engine and writes a YAML."""
    import ethllama.cli as cli_mod

    fake_found = {
        "ollama": "/usr/bin/ollama",
        "llama-cli": "/usr/local/bin/llama-cli",
    }
    generated: list = []

    def fake_discover(binary_name=None):
        return dict(fake_found)

    def fake_generate(name, path, engines_dir=None, overwrite=False):
        target = tmp_path / f"{name}.yaml"
        target.write_text(f"name: {name}\nbinary: {path}\n", encoding="utf-8")
        generated.append((name, path, str(target)))
        return target

    monkeypatch.setattr(cli_mod, "discover_engines", fake_discover)
    monkeypatch.setattr(cli_mod, "generate_engine_config", fake_generate)

    runner = CliRunner()
    result = runner.invoke(main, ["discover"])
    assert result.exit_code == 0, result.output
    out = result.output
    assert "Found 2 inference engine" in out
    assert "ollama" in out
    assert "/usr/bin/ollama" in out
    assert "llama-cli" in out
    # Both YAMLs should have been generated
    assert len(generated) == 2
    assert (tmp_path / "ollama.yaml").exists()
    assert (tmp_path / "llama-cli.yaml").exists()


def test_discover_engines_specific_binary_found(monkeypatch, tmp_path):
    """`ethllama discover <name>` for a found binary reports success."""
    import ethllama.cli as cli_mod

    def fake_discover(binary_name=None):
        assert binary_name == "ollama"
        return {"ollama": "/opt/ollama/bin/ollama"}

    generated: list = []
    def fake_generate(name, path, engines_dir=None, overwrite=False):
        target = tmp_path / f"{name}.yaml"
        target.write_text("name: ollama\n", encoding="utf-8")
        generated.append(target)
        return target

    monkeypatch.setattr(cli_mod, "discover_engines", fake_discover)
    monkeypatch.setattr(cli_mod, "generate_engine_config", fake_generate)

    runner = CliRunner()
    result = runner.invoke(main, ["discover", "ollama"])
    assert result.exit_code == 0, result.output
    out = result.output
    assert "Searching for 'ollama'" in out
    assert "/opt/ollama/bin/ollama" in out
    # Should have generated exactly one YAML
    assert len(generated) == 1
    assert generated[0].name == "ollama.yaml"


def test_discover_engines_specific_binary_not_found(monkeypatch):
    """`ethllama discover <missing>` exits non-zero with a clear message."""
    import ethllama.cli as cli_mod

    monkeypatch.setattr(
        cli_mod, "discover_engines", lambda binary_name=None: {}
    )

    runner = CliRunner()
    result = runner.invoke(main, ["discover", "definitely-not-installed"])
    assert result.exit_code != 0
    out = result.output
    assert "Searching for" in out
    assert "definitely-not-installed" in out
    assert "not found" in out.lower()


def test_discover_no_generate_does_not_write(monkeypatch, tmp_path):
    """`--no-generate` only reports; generate_engine_config is not called."""
    import ethllama.cli as cli_mod

    def fake_discover(binary_name=None):
        return {"llama-cli": "/usr/bin/llama-cli"}

    called = {"n": 0}
    def fake_generate(*args, **kwargs):
        called["n"] += 1
        return tmp_path / "x.yaml"

    monkeypatch.setattr(cli_mod, "discover_engines", fake_discover)
    monkeypatch.setattr(cli_mod, "generate_engine_config", fake_generate)

    runner = CliRunner()
    result = runner.invoke(main, ["discover", "--no-generate"])
    assert result.exit_code == 0
    # No generation should have happened
    assert called["n"] == 0
    # But the find should still be reported
    assert "llama-cli" in result.output
    assert "/usr/bin/llama-cli" in result.output


def test_discover_engines_skips_existing(monkeypatch, tmp_path):
    """When a YAML already exists and --overwrite is absent, the
    output marks the entry as 'skipped' (the file is NOT overwritten)."""
    import ethllama.cli as cli_mod

    # Pre-create a YAML at the target path
    engines_dir = tmp_path
    existing = engines_dir / "ollama.yaml"
    existing.write_text("name: ollama\nbinary: /old/path\n", encoding="utf-8")
    old_content = existing.read_text()

    def fake_discover(binary_name=None):
        return {"ollama": "/new/path/ollama"}

    # generate_engine_config returns None when it refuses to overwrite
    def fake_generate(name, path, engines_dir=None, overwrite=False):
        return None  # signal "skipped"

    monkeypatch.setattr(cli_mod, "discover_engines", fake_discover)
    monkeypatch.setattr(cli_mod, "generate_engine_config", fake_generate)

    runner = CliRunner()
    result = runner.invoke(
        main, ["discover", "--engines-dir", str(engines_dir)]
    )
    assert result.exit_code == 0
    out = result.output
    assert "ollama" in out
    # The message announces that the config already exists (skipped)
    assert "config exists" in out or "exists" in out.lower()
    # And the existing file was not touched
    assert existing.read_text() == old_content


def test_generate_engine_config_writes_yaml(tmp_path):
    """generate_engine_config writes a YAML file with the right shape."""
    from ethllama.engines import generate_engine_config

    target = generate_engine_config(
        "llama-cli", "/usr/bin/llama-cli", engines_dir=tmp_path,
    )
    assert target is not None
    assert target.exists()
    assert target.name == "llama-cli.yaml"

    import yaml
    with open(target) as f:
        data = yaml.safe_load(f)
    assert data["name"] == "llama-cli"
    assert data["type"] == "text"
    assert data["binary"] == "/usr/bin/llama-cli"
    # The template should include the model path and prompt variables
    assert "{{ model_path }}" in data["args_template"]
    assert "{{ prompt }}" in data["args_template"]


def test_generate_engine_config_unknown_binary_uses_defaults(tmp_path):
    """A binary not in KNOWN_ENGINES still gets a minimal config written."""
    from ethllama.engines import generate_engine_config

    target = generate_engine_config(
        "my-custom-engine", "/opt/custom/my-engine", engines_dir=tmp_path,
    )
    assert target is not None
    assert target.exists()

    import yaml
    with open(target) as f:
        data = yaml.safe_load(f)
    assert data["name"] == "my-custom-engine"
    assert data["binary"] == "/opt/custom/my-engine"
    # Falls back to type=text and an empty args_template
    assert data["type"] == "text"
    assert data["args_template"] == ""


def test_generate_engine_config_respects_overwrite(tmp_path):
    """When overwrite=False (default) and the file exists, return None."""
    from ethllama.engines import generate_engine_config

    target = tmp_path / "ollama.yaml"
    target.write_text("name: ollama\nbinary: /old\n", encoding="utf-8")
    old = target.read_text()

    # Without --overwrite, generator returns None
    out = generate_engine_config(
        "ollama", "/new/path/ollama", engines_dir=tmp_path, overwrite=False,
    )
    assert out is None
    assert target.read_text() == old  # untouched

    # With --overwrite, generator returns the path and rewrites the file
    out = generate_engine_config(
        "ollama", "/new/path/ollama", engines_dir=tmp_path, overwrite=True,
    )
    assert out is not None
    import yaml
    with open(out) as f:
        data = yaml.safe_load(f)
    assert data["binary"] == "/new/path/ollama"


def test_known_engines_has_minimum_set():
    """KNOWN_ENGINES catalogues the most important engines a user might have."""
    from ethllama.engines import KNOWN_ENGINES

    required = {"ollama", "llama-cli", "llama-server", "whisper-cli"}
    missing = required - set(KNOWN_ENGINES.keys())
    assert not missing, f"KNOWN_ENGINES missing: {missing}"
    # Every entry should have a type, a template, and a description
    for name, spec in KNOWN_ENGINES.items():
        assert "type" in spec, f"{name} missing type"
        assert "template" in spec, f"{name} missing template"
        assert "description" in spec, f"{name} missing description"
        assert spec["type"] in ("text", "stt", "tts", "image"), (
            f"{name} has unexpected type: {spec['type']}"
        )




# ---------------------------------------------------------------------------
# Tests for the llama.cpp noise filter (`_strip_llama_cpp_noise`) and
# the `--debug` flag exposed by `ethllama run`.
# ---------------------------------------------------------------------------

from ethllama.inference import (
    _strip_llama_cpp_noise,
    _strip_cli_output,
    _clean_chat_tokens,
)


def test_strip_llama_cpp_noise_filters_banner():
    """The llama.cpp startup banner / log lines are stripped from output.

    The banner block typically contains lines with markers like
    ``llama_model_loader``, ``llm_load_tensors``,
    ``llama_print_system_info``, ``model type``, ``model size``,
    ``general.architecture``, ``print_info:`` etc.  None of these
    should appear in the cleaned output; the response text should
    survive verbatim.
    """
    raw = (
        "llama_model_loader: loaded meta data with 28 key-value pairs and 291 tensors\n"
        "llm_load_tensors: offloading 0 repeating layers to GPU\n"
        "llm_load_tensors:        CPU buffer size =  2808.00 MiB\n"
        "llama_print_system_info: CPU info: 8 cores, 16 threads\n"
        "print_info:       n_ctx = 4096\n"
        "print_info:       n_batch = 512\n"
        "model type     = 7B\n"
        "model size     = 3.8 GiB (3.84 BPW)\n"
        "general.architecture = llama\n"
        "system_info: AVX = 1 | AVX_VNNI = 0\n"
        "sampling: temp = 0.700\n"
        "generate: n_ctx = 4096, n_batch = 512, n_predict = 2048\n"
        "\n"
        "Hello! I am a friendly assistant. How can I help you today?\n"
    )
    cleaned = _strip_llama_cpp_noise(raw)
    # All the banner / sampling / generate markers must be gone.
    for marker in (
        "llama_model_loader",
        "llm_load_tensors",
        "llama_print_system_info",
        "print_info:",
        "model type",
        "model size",
        "general.architecture",
        "system_info:",
        "sampling:",
        "generate:",
    ):
        assert marker not in cleaned, (
            f"Banner marker {marker!r} leaked through: {cleaned!r}"
        )
    # The actual response text must be preserved.
    assert "Hello! I am a friendly assistant. How can I help you today?" in cleaned


def test_strip_llama_cpp_noise_filters_exiting():
    """`Exiting...` / `cleaning up` / `exit code:` lines are filtered out."""
    raw = (
        "llama_model_loader: loaded\n"
        "\n"
        "The response to your question.\n"
        "\n"
        "main: exiting...\n"
        "cleaning up\n"
        "exit code: 0\n"
    )
    cleaned = _strip_llama_cpp_noise(raw)
    assert "The response to your question." in cleaned
    assert "exiting" not in cleaned.lower()
    assert "cleaning up" not in cleaned.lower()
    assert "exit code" not in cleaned.lower()


def test_strip_llama_cpp_noise_strips_chat_tokens():
    """Chat template tokens like `<|im_start|>` and `[Start thinking]` are stripped.

    The realistic case: the prompt is wrapped in chat-template tokens
    (which the model echoes back to stdout because the binary doesn't
    strip them), and the actual model response comes after the closing
    ``<|im_end|>``.  After filtering, the response should be clean and
    the chat-template tokens gone.
    """
    raw = (
        "<|im_start|>user\n"
        "What is 2+2?<|im_end|>\n"
        "<|im_start|>assistant\n"
        "<|im_end|>\n"
        "The answer is 4.\n"
    )
    cleaned = _clean_chat_tokens(raw)
    # No chat-template control tokens should survive.
    assert "<|im_start|>" not in cleaned
    assert "<|im_end|>" not in cleaned
    # The actual response text must still be there.
    assert "The answer is 4." in cleaned
    # And the user prompt itself should be gone (it was inside a chat block).
    assert "What is 2+2?" not in cleaned


def test_strip_llama_cpp_noise_strips_bracket_thinking_tokens():
    """`[Start thinking]` and `[End thinking]` style markers are stripped."""
    raw = (
        "[Start thinking]\n"
        "The answer is 4.\n"
        "[End thinking]\n"
        "Final answer: 4.\n"
    )
    cleaned = _clean_chat_tokens(raw)
    assert "[Start thinking]" not in cleaned
    assert "[End thinking]" not in cleaned
    # Surrounding text survives
    assert "The answer is 4." in cleaned
    assert "Final answer: 4." in cleaned


def test_strip_llama_cpp_noise_strips_im_start_end_blocks():
    """`<|im_start|>...<|im_end|>` blocks are stripped entirely (including
    the multi-line turn content)."""
    raw = (
        "Pre-text.\n"
        "<|im_start|>user\n"
        "hidden user turn\n"
        "<|im_end|>\n"
        "Post-text.\n"
    )
    cleaned = _clean_chat_tokens(raw)
    assert "Pre-text." in cleaned
    assert "Post-text." in cleaned
    # The hidden turn must be gone (both its content and the delimiters)
    assert "hidden user turn" not in cleaned
    assert "<|im_start|>" not in cleaned
    assert "<|im_end|>" not in cleaned


def test_strip_llama_cpp_noise_debug_keeps_everything():
    """`debug=True` returns the input unchanged (no filtering at all)."""
    raw = (
        "llama_model_loader: blah\n"
        "Exiting...\n"
        "<|im_start|>garbage<|im_end|>\n"
    )
    out = _strip_llama_cpp_noise(raw, debug=True)
    assert out == raw


def test_strip_cli_output_handles_nested_prompt_echo():
    """`_strip_cli_output` recognises the `> > {prompt}` nested form."""
    raw = (
        "llama_model_loader: blah\n"
        "> > Hello world\n"
        "I am doing well.\n"
    )
    out = _strip_cli_output(raw, "Hello world")
    assert "I am doing well." in out
    assert "> >" not in out
    assert "Hello world" not in out


def test_strip_cli_output_handles_chat_prefix_echo():
    """`_strip_cli_output` recognises the `[user]: {prompt}` form."""
    raw = (
        "llama_model_loader: blah\n"
        "[user]: Hi there\n"
        "Greetings, friend!\n"
    )
    out = _strip_cli_output(raw, "Hi there")
    assert "Greetings, friend!" in out
    assert "[user]:" not in out


def test_strip_cli_output_debug_returns_raw():
    """`_strip_cli_output(..., debug=True)` returns the raw stdout."""
    raw = (
        "llama_model_loader: blah\n"
        "> Hello\n"
        "Hi!\n"
    )
    out = _strip_cli_output(raw, "Hello", debug=True)
    # debug mode: nothing filtered, returned as-is (stripped only).
    assert "llama_model_loader: blah" in out
    assert "> Hello" in out
    assert "Hi!" in out


def test_run_help_shows_debug_flag():
    """`ethllama run --help` documents the new --debug flag."""
    runner = CliRunner()
    result = runner.invoke(main, ["run", "--help"])
    assert result.exit_code == 0
    # The flag itself must be listed in the help output.
    assert "--debug" in result.output
    # And the help text should explain what it does.
    out_lower = result.output.lower()
    assert "raw" in out_lower or "debug" in out_lower


def test_run_passes_debug_to_inference(tmp_path, monkeypatch):
    """`ethllama run --debug` forwards ``debug=True`` to run_inference."""

    # Isolate the model index so the fake model can't be picked up from
    # the user's real ~/.ethllama/index.json.
    from ethllama import index as index_mod
    monkeypatch.setattr(index_mod, "INDEX_FILE", tmp_path / "index.json")

    model_path = tmp_path / "debug-test-model.gguf"
    model_path.write_bytes(b"GGUF" + b"\x00" * 32)

    captured: dict = {}
    monkeypatch.setattr("ethllama.inference.has_inference_engine", lambda: True)

    def fake_run_inference(**kwargs):
        captured.update(kwargs)
        return "ok"

    monkeypatch.setattr("ethllama.inference.run_inference", fake_run_inference)

    runner = CliRunner()
    result = runner.invoke(
        main, ["run", str(model_path), "-p", "Hello", "--debug"]
    )
    assert result.exit_code == 0, result.output
    assert captured.get("debug") is True, (
        f"Expected debug=True to be forwarded to run_inference; "
        f"got {captured.get('debug')!r}"
    )

    # And without --debug, the default is False.
    captured.clear()
    result2 = runner.invoke(
        main, ["run", str(model_path), "-p", "Hello"]
    )
    assert result2.exit_code == 0, result2.output
    assert captured.get("debug") is False, (
        f"Expected debug=False by default; got {captured.get('debug')!r}"
    )


def test_run_stream_passes_debug_to_stream(tmp_path, monkeypatch):
    """`ethllama run --debug --stream` forwards debug=True to run_inference_stream."""

    from ethllama import index as index_mod
    monkeypatch.setattr(index_mod, "INDEX_FILE", tmp_path / "index.json")

    model_path = tmp_path / "stream-debug.gguf"
    model_path.write_bytes(b"GGUF" + b"\x00" * 32)

    captured: dict = {}
    monkeypatch.setattr("ethllama.inference.has_inference_engine", lambda: True)

    def fake_stream(**kwargs):
        captured.update(kwargs)
        yield "chunk"

    monkeypatch.setattr("ethllama.inference.run_inference_stream", fake_stream)

    runner = CliRunner()
    result = runner.invoke(
        main, ["run", str(model_path), "-p", "Hi", "--stream", "--debug"]
    )
    assert result.exit_code == 0, result.output
    assert captured.get("debug") is True
    # And the streamed chunk should still appear in the output.
    assert "chunk" in result.output


def test_repl_session_stores_debug_flag():
    """`REPLSession(..., debug=True)` retains the flag and forwards it
    to ``run_inference_stream`` when ``send()`` is called."""
    from ethllama.inference import REPLSession

    captured: dict = []

    def fake_stream(**kwargs):
        captured.append(kwargs)
        # yield nothing so the REPL doesn't have anything to display
        if False:
            yield ""

    class FakeEngine:
        @staticmethod
        def __call__():
            return True

    # Replace has_inference_engine + run_inference_stream with our stubs.
    import ethllama.inference as inf_mod
    monkey = __import__("pytest").MonkeyPatch()
    try:
        monkey.setattr(inf_mod, "has_inference_engine", lambda: True)
        monkey.setattr(inf_mod, "run_inference_stream", fake_stream)

        session = REPLSession("/tmp/fake.gguf", debug=True)
        assert session.debug is True

        # Drive one user turn and ensure debug was forwarded.
        for _ in session.send("Hello"):
            pass
        assert captured, "run_inference_stream was not called"
        assert captured[-1].get("debug") is True

        # Now without --debug.
        captured.clear()
        session2 = REPLSession("/tmp/fake.gguf")
        assert session2.debug is False
        for _ in session2.send("Hello"):
            pass
        assert captured[-1].get("debug") is False
    finally:
        monkey.undo()


# ---------------------------------------------------------------------------
# `ethllama setup` — guided setup / onboarding wizard
# ---------------------------------------------------------------------------


def test_setup_help_shows_options():
    """`ethllama setup --help` lists all the wizard options."""
    runner = CliRunner()
    result = runner.invoke(main, ["setup", "--help"])
    assert result.exit_code == 0
    out = result.output
    # All required options are documented.
    assert "--service-mode" in out
    assert "--binary-dir" in out
    assert "--port" in out
    assert "--api-key" in out
    assert "--no-install" in out
    assert "--yes" in out or "-y" in out
    # The help text describes the wizard steps.
    assert "interactive" in out.lower() or "wizard" in out.lower()


def test_setup_appears_in_main_help():
    """`ethllama setup` is listed in the main group help output."""
    runner = CliRunner()
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "setup" in result.output


def test_setup_no_install_skips_service(monkeypatch, tmp_path):
    """`ethllama setup --no-install --yes` only writes config; no systemctl calls."""
    import ethllama.cli as cli_mod
    import ethllama.config as config_mod

    # Isolate the config file so we don't touch the user's real one.
    monkeypatch.setattr(config_mod, "CONFIG_FILE", tmp_path / "config.yaml")
    monkeypatch.setattr(cli_mod, "save_config", config_mod.save_config)

    # Record every subprocess call; the only one allowed during
    # `--no-install --yes` is the `sudo -n true` probe used by _can_sudo.
    calls: list = []

    class _Result:
        returncode = 1
        stdout = ""
        stderr = b""

    def fake_run(cmd, *args, **kwargs):
        calls.append(list(cmd))
        # Pretend sudo isn't available; forces the wizard down the
        # no-sudo path without spawning a real subprocess.
        return _Result()

    monkeypatch.setattr(cli_mod.subprocess, "run", fake_run)

    runner = CliRunner()
    result = runner.invoke(main, ["setup", "--no-install", "--yes"])
    assert result.exit_code == 0, result.output
    # The wizard wrote the config file.
    assert (tmp_path / "config.yaml").exists()
    # And it includes the default port.
    import yaml
    with open(tmp_path / "config.yaml") as f:
        cfg = yaml.safe_load(f)
    assert cfg["api"]["port"] == 10434
    # Only the `sudo -n true` probe from _can_sudo was allowed; nothing
    # that actually touches systemd (systemctl, sudo cp, etc.) ran.
    bad = [c for c in calls if not (len(c) >= 1 and c[0] == "sudo" and len(c) >= 2 and c[1] == "-n")]
    assert bad == [], f"Unexpected subprocess calls: {bad}"


def test_setup_no_install_saves_binary_dir(monkeypatch, tmp_path):
    """`ethllama setup --no-install --binary-dir <path> --yes` records the dir."""
    import ethllama.cli as cli_mod
    import ethllama.config as config_mod

    monkeypatch.setattr(config_mod, "CONFIG_FILE", tmp_path / "config.yaml")
    monkeypatch.setattr(cli_mod, "save_config", config_mod.save_config)

    class _Result:
        returncode = 1
        stdout = ""
        stderr = b""

    calls: list = []
    def fake_run(cmd, *args, **kwargs):
        calls.append(list(cmd))
        return _Result()

    monkeypatch.setattr(cli_mod.subprocess, "run", fake_run)

    runner = CliRunner()
    result = runner.invoke(
        main,
        ["setup", "--no-install", "--yes", "--binary-dir", "/opt/llama.cpp/build/bin"],
    )
    assert result.exit_code == 0, result.output

    import yaml
    with open(tmp_path / "config.yaml") as f:
        cfg = yaml.safe_load(f)
    assert cfg["engines"]["binary_dir"] == "/opt/llama.cpp/build/bin"
    assert cfg["api"]["port"] == 10434
    # No systemctl / sudo-cp / sudo systemctl invocations happened.
    bad = [c for c in calls if not (len(c) >= 1 and c[0] == "sudo" and len(c) >= 2 and c[1] == "-n")]
    assert bad == [], f"Unexpected subprocess calls: {bad}"


def test_setup_preserves_existing_config(monkeypatch, tmp_path):
    """Re-running `setup --no-install --yes` does not clobber unrelated keys."""
    import ethllama.cli as cli_mod
    import ethllama.config as config_mod

    monkeypatch.setattr(config_mod, "CONFIG_FILE", tmp_path / "config.yaml")
    monkeypatch.setattr(cli_mod, "save_config", config_mod.save_config)

    # Seed an existing config with a custom key.
    import yaml
    (tmp_path / "config.yaml").write_text(yaml.safe_dump({
        "gpu": {"backend": "cuda", "fallback": False},
        "custom_key": "do-not-touch",
        "model_defaults": {"phi-4": {"temperature": 0.3}},
    }))

    class _Result:
        returncode = 1
        stdout = ""
        stderr = b""

    calls: list = []
    def fake_run(cmd, *args, **kwargs):
        calls.append(list(cmd))
        return _Result()

    monkeypatch.setattr(cli_mod.subprocess, "run", fake_run)

    runner = CliRunner()
    result = runner.invoke(main, ["setup", "--no-install", "--yes"])
    assert result.exit_code == 0, result.output

    with open(tmp_path / "config.yaml") as f:
        cfg = yaml.safe_load(f)
    # Pre-existing keys are preserved.
    assert cfg["custom_key"] == "do-not-touch"
    assert cfg["model_defaults"] == {"phi-4": {"temperature": 0.3}}
    assert cfg["gpu"]["backend"] == "cuda"
    # The wizard wrote the new keys.
    assert cfg["api"]["port"] == 10434
    # No real service-install calls happened.
    bad = [c for c in calls if not (len(c) >= 1 and c[0] == "sudo" and len(c) >= 2 and c[1] == "-n")]
    assert bad == [], f"Unexpected subprocess calls: {bad}"


def test_can_sudo_detects_availability(monkeypatch):
    """`_can_sudo` returns True when `sudo -n true` exits 0, False otherwise."""
    import ethllama.cli as cli_mod

    class _Result:
        def __init__(self, rc): self.returncode = rc

    # No sudo binary: returns False.
    monkeypatch.setattr(cli_mod.shutil, "which", lambda _x: None)
    assert cli_mod._can_sudo() is False

    # sudo present and returns 0: returns True.
    monkeypatch.setattr(cli_mod.shutil, "which", lambda _x: "/usr/bin/sudo")
    monkeypatch.setattr(
        cli_mod.subprocess, "run",
        lambda *a, **kw: _Result(0),
    )
    assert cli_mod._can_sudo() is True

    # sudo present but returns non-zero: returns False.
    monkeypatch.setattr(
        cli_mod.subprocess, "run",
        lambda *a, **kw: _Result(1),
    )
    assert cli_mod._can_sudo() is False

    # sudo present but raises FileNotFoundError: returns False.
    def _raise_fnf(*a, **kw):
        raise FileNotFoundError("no sudo")

    monkeypatch.setattr(cli_mod.subprocess, "run", _raise_fnf)
    assert cli_mod._can_sudo() is False


def test_quick_discover_finds_engines_on_path(monkeypatch, tmp_path):
    """`_quick_discover` returns only the engines currently on PATH."""
    import ethllama.cli as cli_mod
    from ethllama.engines import KNOWN_ENGINES

    # Build a fake bin dir with a couple of fake engine binaries.
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    for name in ("llama-cli", "ollama"):
        (fake_bin / name).write_text("#!/bin/sh\nexit 0")
    (fake_bin / "llama-cli").chmod(0o755)
    (fake_bin / "ollama").chmod(0o755)

    # Make shutil.which(name) pretend these two exist, others don't.
    def fake_which(name):
        if name in ("llama-cli", "ollama"):
            return str(fake_bin / name)
        return None

    monkeypatch.setattr(cli_mod.shutil, "which", fake_which)
    found = cli_mod._quick_discover()
    assert "llama-cli" in found
    assert "ollama" in found
    # It must be a strict subset of KNOWN_ENGINES (only those keys are scanned).
    assert set(found.keys()) <= set(KNOWN_ENGINES.keys())


def test_setup_user_mode_calls_systemctl_user(monkeypatch, tmp_path):
    """Generated user service uses only systemctl --user and no sudo."""
    import ethllama.cli as cli_mod
    calls: list = []

    class _Result:
        returncode = 0
        stdout = "Linger=no"
        stderr = ""

    def fake_run(cmd, *args, **kwargs):
        calls.append(list(cmd))
        return _Result()

    monkeypatch.setattr(cli_mod.subprocess, "run", fake_run)
    monkeypatch.setattr(cli_mod.Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.setattr(cli_mod.os, "environ", {"USER": "tester"})
    ok = cli_mod._install_user_service("/opt/bin/ethllama", tmp_path / "config.yaml")
    assert ok is True
    assert all("sudo" not in command for command in calls)
    assert ["systemctl", "--user", "daemon-reload"] in calls
    unit = (tmp_path / ".config/systemd/user/ethllama.service").read_text()
    assert "User=" not in unit
    assert "/opt/bin/ethllama" in unit


def test_setup_service_mode_skip_skips_install(monkeypatch, tmp_path):
    """`--service-mode skip` writes the config but never touches systemd."""
    import ethllama.cli as cli_mod
    import ethllama.config as config_mod

    monkeypatch.setattr(config_mod, "CONFIG_FILE", tmp_path / "config.yaml")
    monkeypatch.setattr(cli_mod, "save_config", config_mod.save_config)

    class _Result:
        returncode = 1
        stdout = ""
        stderr = b""

    calls: list = []
    def fake_run(cmd, *args, **kwargs):
        calls.append(list(cmd))
        return _Result()

    monkeypatch.setattr(cli_mod.subprocess, "run", fake_run)

    runner = CliRunner()
    result = runner.invoke(
        main, ["setup", "--service-mode", "skip", "--yes"]
    )
    assert result.exit_code == 0, result.output
    # The wizard explicitly noted the skip in its output.
    assert "skipped" in result.output.lower() or "skip" in result.output.lower()
    # No real service-install calls happened.
    bad = [c for c in calls if not (len(c) >= 1 and c[0] == "sudo" and len(c) >= 2 and c[1] == "-n")]
    assert bad == [], f"Unexpected subprocess calls: {bad}"


def test_setup_no_install_does_not_require_executable(monkeypatch, tmp_path):
    """Configuration-only setup does not require an executable or sudo."""
    import ethllama.cli as cli_mod
    import ethllama.config as config_mod
    monkeypatch.setattr(config_mod, "CONFIG_FILE", tmp_path / "config.yaml")
    monkeypatch.setattr(cli_mod, "save_config", config_mod.save_config)
    monkeypatch.setattr(cli_mod.shutil, "which", lambda *_a, **_kw: None)
    result = CliRunner().invoke(main, ["setup", "--no-install", "--yes"])
    assert result.exit_code == 0, result.output
    assert (tmp_path / "config.yaml").exists()


# ---------------------------------------------------------------------------
# `ethllama profile` — model profiles subcommand group
# ---------------------------------------------------------------------------

from ethllama import profiles as profiles_mod
from ethllama.cli_profile import register_commands as register_profile

# Wire the profile subcommand onto the main group so the tests can
# invoke it via the standard `ethllama profile ...` syntax.
register_profile(main)


@pytest.fixture
def tmp_profiles_dir(tmp_path, monkeypatch):
    """Redirect ``profiles.PROFILES_DIR`` to a fresh tmp directory."""
    target = tmp_path / "profiles"
    target.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(profiles_mod, "PROFILES_DIR", target)
    return target


def test_profile_help_lists_subcommands():
    """`ethllama profile --help` lists all six subcommands."""
    runner = CliRunner()
    result = runner.invoke(main, ["profile", "--help"])
    assert result.exit_code == 0, result.output
    out = result.output
    for cmd in ("create", "list", "show", "edit", "delete", "run"):
        assert cmd in out, f"Missing subcommand: {cmd}"


def test_profile_list_empty(tmp_profiles_dir):
    """`ethllama profile list` on an empty dir prints a helpful message."""
    runner = CliRunner()
    result = runner.invoke(main, ["profile", "list"])
    assert result.exit_code == 0, result.output
    assert "No profiles" in result.output or "profile create" in result.output


def test_profile_list_shows_existing(tmp_profiles_dir):
    """`ethllama profile list` displays the configured profiles."""
    from ethllama.profiles import Profile
    Profile(name="alpha", model="/a.gguf", description="first").save(
        profiles_dir=tmp_profiles_dir
    )
    Profile(name="beta", model="/b.gguf").save(profiles_dir=tmp_profiles_dir)

    runner = CliRunner()
    result = runner.invoke(main, ["profile", "list"])
    assert result.exit_code == 0, result.output
    assert "alpha" in result.output
    assert "beta" in result.output
    assert "first" in result.output  # description
    # Sorted alphabetically
    assert result.output.index("alpha") < result.output.index("beta")


def test_profile_list_json_format(tmp_profiles_dir):
    """`ethllama profile list --json` outputs a JSON envelope."""
    from ethllama.profiles import Profile
    Profile(name="only", model="/m.gguf").save(profiles_dir=tmp_profiles_dir)

    runner = CliRunner()
    result = runner.invoke(main, ["profile", "list", "--json"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data == {"profiles": ["only"]}


def test_profile_create_writes_yaml(tmp_profiles_dir):
    """`ethllama profile create` writes a profile YAML to disk."""
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "profile", "create", "chat-py",
            "--model", "/path/to/model.gguf",
            "--temperature", "0.3",
            "--top-p", "0.9",
            "--top-k", "30",
            "--max-tokens", "2048",
            "--system-prompt", "You are a Python expert.",
            "--description", "Coding profile",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "saved to" in result.output.lower()

    # The file should be there with the right content
    yaml_path = tmp_profiles_dir / "chat-py.yaml"
    assert yaml_path.exists()
    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    assert data["name"] == "chat-py"
    assert data["model"] == "/path/to/model.gguf"
    assert data["description"] == "Coding profile"
    assert data["parameters"]["temperature"] == 0.3
    assert data["parameters"]["top_p"] == 0.9
    assert data["parameters"]["top_k"] == 30
    assert data["parameters"]["max_tokens"] == 2048
    assert data["system_prompt"] == "You are a Python expert."


def test_profile_create_rejects_existing_without_overwrite(tmp_profiles_dir):
    """`profile create` refuses to overwrite without --overwrite."""
    from ethllama.profiles import Profile
    Profile(name="dup", model="/m.gguf").save(profiles_dir=tmp_profiles_dir)

    runner = CliRunner()
    result = runner.invoke(
        main,
        ["profile", "create", "dup", "--model", "/other.gguf"],
    )
    assert result.exit_code != 0
    assert "already exists" in result.output.lower()


def test_profile_create_overwrite_replaces(tmp_profiles_dir):
    """`profile create --overwrite` replaces the existing YAML."""
    from ethllama.profiles import Profile
    Profile(name="dup", model="/old.gguf").save(profiles_dir=tmp_profiles_dir)

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "profile", "create", "dup",
            "--model", "/new.gguf",
            "--overwrite",
        ],
    )
    assert result.exit_code == 0, result.output

    loaded = Profile.from_yaml(tmp_profiles_dir / "dup.yaml")
    assert loaded.model == "/new.gguf"


def test_profile_create_from_yaml(tmp_profiles_dir, tmp_path):
    """`profile create --from-yaml` reads an existing YAML file."""
    source = tmp_path / "source.yaml"
    source.write_text(
        textwrap.dedent("""\
            name: original
            model: /source-model.gguf
            parameters:
              temperature: 0.11
            description: from source
        """),
        encoding="utf-8",
    )
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "profile", "create", "renamed",
            "--from-yaml", str(source),
            "--model", "/override-model.gguf",
        ],
    )
    assert result.exit_code == 0, result.output

    loaded = profiles_mod.load_profile("renamed", profiles_dir=tmp_profiles_dir)
    assert loaded.name == "renamed"  # re-stamped
    assert loaded.model == "/override-model.gguf"  # CLI override
    assert loaded.parameters["temperature"] == 0.11
    assert loaded.description == "from source"


def test_profile_show_displays_yaml(tmp_profiles_dir):
    """`ethllama profile show` displays the profile details."""
    from ethllama.profiles import Profile
    Profile(
        name="chat-py",
        model="/m.gguf",
        description="Demo",
        parameters={"temperature": 0.3, "top_k": 20},
        system_prompt="You are terse.",
        template="{{ .Prompt }}",
        stop=["</s>"],
    ).save(profiles_dir=tmp_profiles_dir)

    runner = CliRunner()
    result = runner.invoke(main, ["profile", "show", "chat-py"])
    assert result.exit_code == 0, result.output
    out = result.output
    assert "chat-py" in out
    assert "/m.gguf" in out
    assert "Demo" in out
    assert "temperature: 0.3" in out
    assert "top_k: 20" in out
    assert "You are terse." in out
    assert "{{ .Prompt }}" in out
    assert "</s>" in out


def test_profile_show_missing_exits_non_zero(tmp_profiles_dir):
    """`ethllama profile show <missing>` exits with a clear error."""
    runner = CliRunner()
    result = runner.invoke(main, ["profile", "show", "ghost"])
    assert result.exit_code != 0
    assert "not found" in result.output.lower()


def test_profile_show_json(tmp_profiles_dir):
    """`ethllama profile show --json` outputs the full profile as JSON."""
    from ethllama.profiles import Profile
    Profile(
        name="json-demo",
        model="/m.gguf",
        parameters={"temperature": 0.4},
        stop=["<e>"],
    ).save(profiles_dir=tmp_profiles_dir)

    runner = CliRunner()
    result = runner.invoke(main, ["profile", "show", "json-demo", "--json"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["name"] == "json-demo"
    assert data["model"] == "/m.gguf"
    assert data["parameters"] == {"temperature": 0.4}
    assert data["stop"] == ["<e>"]


def test_profile_delete_removes_file(tmp_profiles_dir):
    """`ethllama profile delete --yes` removes the YAML."""
    from ethllama.profiles import Profile
    Profile(name="byebye", model="/m.gguf").save(profiles_dir=tmp_profiles_dir)
    assert (tmp_profiles_dir / "byebye.yaml").exists()

    runner = CliRunner()
    result = runner.invoke(
        main, ["profile", "delete", "byebye", "--yes"]
    )
    assert result.exit_code == 0, result.output
    assert "Deleted" in result.output
    assert not (tmp_profiles_dir / "byebye.yaml").exists()


def test_profile_delete_missing_exits_non_zero(tmp_profiles_dir):
    """`ethllama profile delete <missing>` exits non-zero with a message."""
    runner = CliRunner()
    result = runner.invoke(
        main, ["profile", "delete", "ghost", "--yes"]
    )
    assert result.exit_code != 0
    assert "not found" in result.output.lower()


def test_profile_edit_missing_exits_non_zero(tmp_profiles_dir, monkeypatch):
    """`ethllama profile edit <missing>` exits non-zero."""
    runner = CliRunner()
    result = runner.invoke(main, ["profile", "edit", "ghost"])
    assert result.exit_code != 0
    assert "not found" in result.output.lower()


def test_profile_edit_opens_editor(tmp_profiles_dir, monkeypatch):
    """`ethllama profile edit` execs into $EDITOR (captured via monkeypatch)."""
    from ethllama.profiles import Profile
    Profile(name="edit-me", model="/m.gguf").save(profiles_dir=tmp_profiles_dir)

    captured: dict = {}

    def fake_execvp(file, args):
        captured["file"] = file
        captured["args"] = args
        # Don't actually exec; raise SystemExit so the test continues.
        raise SystemExit(0)

    monkeypatch.setattr("ethllama.cli_profile.os.execvp", fake_execvp)

    runner = CliRunner()
    # Click's CliRunner will surface SystemExit, but we don't care
    # about the exit code; we care that execvp was called.
    runner.invoke(main, ["profile", "edit", "edit-me"], catch_exceptions=False)

    assert captured.get("file") is not None
    assert captured["args"][-1].endswith("edit-me.yaml")


def test_run_with_profile_applies_parameters(
    tmp_profiles_dir, tmp_path, monkeypatch
):
    """`ethllama run --profile <name>` applies the profile's parameters."""
    from ethllama.profiles import Profile
    from ethllama import index as index_mod

    # Set up a fake model + index
    model_path = tmp_path / "model.gguf"
    model_path.write_bytes(b"GGUF" + b"\x00" * 32)
    monkeypatch.setattr(index_mod, "INDEX_FILE", tmp_path / "index.json")
    index_mod.add_to_index(str(model_path))

    # Save a profile
    Profile(
        name="chat-py",
        model="model.gguf",
        parameters={"temperature": 0.3, "top_k": 20, "n_gpu_layers": 7},
        system_prompt="You are helpful.",
    ).save(profiles_dir=tmp_profiles_dir)

    # Capture run_inference
    captured: dict = {}
    import ethllama.inference as inf_mod
    monkeypatch.setattr(inf_mod, "has_inference_engine", lambda: True)
    monkeypatch.setattr(
        inf_mod, "run_inference",
        lambda **kw: (captured.update(kw) or "ok"),
    )

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "run", str(model_path),
            "-p", "Hello",
            "--profile", "chat-py",
        ],
    )
    assert result.exit_code == 0, result.output
    # Profile parameters should be applied
    assert captured.get("temperature") == 0.3
    assert captured.get("top_k") == 20
    assert captured.get("n_gpu_layers") == 7
    # The system prompt should have been prepended
    assert "You are helpful." in captured.get("prompt", "")
    # The CLI should mention which profile was used
    assert "Profile: chat-py" in result.output


def test_run_with_profile_explicit_flag_overrides(
    tmp_profiles_dir, tmp_path, monkeypatch
):
    """Explicit CLI flags win over profile parameters."""
    from ethllama.profiles import Profile
    from ethllama import index as index_mod

    model_path = tmp_path / "model.gguf"
    model_path.write_bytes(b"GGUF" + b"\x00" * 32)
    monkeypatch.setattr(index_mod, "INDEX_FILE", tmp_path / "index.json")
    index_mod.add_to_index(str(model_path))

    Profile(
        name="p",
        model="model.gguf",
        parameters={"temperature": 0.3, "top_k": 20},
    ).save(profiles_dir=tmp_profiles_dir)

    captured: dict = {}
    import ethllama.inference as inf_mod
    monkeypatch.setattr(inf_mod, "has_inference_engine", lambda: True)
    monkeypatch.setattr(
        inf_mod, "run_inference",
        lambda **kw: (captured.update(kw) or "ok"),
    )

    runner = CliRunner()
    # User passes --temperature 0.9 explicitly; top_k stays at default
    # so the profile value (20) should be used.
    result = runner.invoke(
        main,
        [
            "run", str(model_path),
            "-p", "Hi",
            "--profile", "p",
            "--temperature", "0.9",
        ],
    )
    assert result.exit_code == 0, result.output
    assert captured.get("temperature") == 0.9  # explicit flag wins
    assert captured.get("top_k") == 20  # profile fills in


def test_run_with_profile_missing_exits_non_zero(
    tmp_profiles_dir, tmp_path, monkeypatch
):
    """`--profile <missing>` exits non-zero with a clear message."""
    from ethllama import index as index_mod
    model_path = tmp_path / "model.gguf"
    model_path.write_bytes(b"GGUF" + b"\x00" * 32)
    monkeypatch.setattr(index_mod, "INDEX_FILE", tmp_path / "index.json")
    index_mod.add_to_index(str(model_path))

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "run", str(model_path),
            "-p", "Hello",
            "--profile", "does-not-exist",
        ],
    )
    assert result.exit_code != 0
    assert "not found" in result.output.lower()


def test_run_help_shows_profile_option():
    """`ethllama run --help` documents the --profile option."""
    runner = CliRunner()
    result = runner.invoke(main, ["run", "--help"])
    assert result.exit_code == 0
    out = result.output
    assert "--profile" in out
    assert "-P" in out


def test_serve_help_shows_profile_option():
    """`ethllama serve --help` documents the --profile option."""
    runner = CliRunner()
    result = runner.invoke(main, ["serve", "--help"])
    assert result.exit_code == 0
    out = result.output
    assert "--profile" in out
    assert "-P" in out


def test_serve_with_profile_applies_parameters(
    tmp_profiles_dir, tmp_path, monkeypatch
):
    """`ethllama serve --profile <name>` applies the profile to GPU config."""
    import builtins
    from ethllama import index as index_mod
    from ethllama.profiles import Profile
    import ethllama.inference as inf_mod
    import ethllama.cli as cli_mod

    # Fake model
    model_path = tmp_path / "m.gguf"
    model_path.write_bytes(b"GGUF" + b"\x00" * 32)
    monkeypatch.setattr(index_mod, "INDEX_FILE", tmp_path / "index.json")
    index_mod.add_to_index(str(model_path))

    Profile(
        name="serve-prof",
        model="m.gguf",
        parameters={"n_gpu_layers": 42, "threads": 7},
    ).save(profiles_dir=tmp_profiles_dir)

    captured_gpu: dict = {}
    monkeypatch.setattr(
        inf_mod, "set_gpu_config", lambda **kw: captured_gpu.update(kw)
    )

    captured_serve: dict = {}

    def fake_run_server(*args, **kwargs):
        captured_serve.update(kwargs)
        raise KeyboardInterrupt()

    real_import = builtins.__import__

    def fake_import(name, *a, **kw):
        mod = real_import(name, *a, **kw)
        if name.endswith("api") or name == "ethllama.api":
            mod.run_server = fake_run_server
        return mod

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.setattr(cli_mod, "resolve_model_path", lambda _m: str(model_path))

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "serve", "--profile", "serve-prof",
            "--host", "127.0.0.1", "--port", "1",
        ],
    )

    # Profile's parameters should have been applied via set_gpu_config
    assert captured_gpu.get("n_gpu_layers") == 42
    assert captured_gpu.get("n_threads") == 7
    # The profile's model was picked up because --model was not given
    assert captured_serve.get("model_path") == str(model_path)


def test_serve_with_profile_explicit_model_wins(tmp_profiles_dir, tmp_path, monkeypatch):
    """Explicit --model wins over the profile's model."""
    import builtins
    from ethllama import index as index_mod
    from ethllama.profiles import Profile
    import ethllama.inference as inf_mod
    import ethllama.cli as cli_mod

    model_path = tmp_path / "m.gguf"
    model_path.write_bytes(b"GGUF" + b"\x00" * 32)
    monkeypatch.setattr(index_mod, "INDEX_FILE", tmp_path / "index.json")
    index_mod.add_to_index(str(model_path))

    Profile(
        name="p",
        model="ignored-model.gguf",
        parameters={"n_gpu_layers": 5},
    ).save(profiles_dir=tmp_profiles_dir)

    captured_gpu: dict = {}
    monkeypatch.setattr(
        inf_mod, "set_gpu_config", lambda **kw: captured_gpu.update(kw)
    )

    captured_serve: dict = {}

    def fake_run_server(*args, **kwargs):
        captured_serve.update(kwargs)
        raise KeyboardInterrupt()

    real_import = builtins.__import__

    def fake_import(name, *a, **kw):
        mod = real_import(name, *a, **kw)
        if name.endswith("api") or name == "ethllama.api":
            mod.run_server = fake_run_server
        return mod

    monkeypatch.setattr(builtins, "__import__", fake_import)

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "serve", "--profile", "p", "--model", str(model_path),
            "--host", "127.0.0.1", "--port", "1",
        ],
    )

    # The explicit --model takes precedence
    assert captured_serve.get("model_path") == str(model_path)
    # But the profile's parameters still apply
    assert captured_gpu.get("n_gpu_layers") == 5


def test_profile_in_main_help():
    """The `profile` command group appears in the main help output."""
    runner = CliRunner()
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "profile" in result.output


def test_profile_apply_to_kwargs_preserves_explicit_values():
    """`apply_profile_to_kwargs` does not override explicit non-None values."""
    from ethllama.cli_profile import apply_profile_to_kwargs
    from ethllama.profiles import Profile

    p = Profile(
        name="x",
        model="/m.gguf",
        parameters={"temperature": 0.3, "top_k": 20},
    )
    out = apply_profile_to_kwargs(p, {"temperature": 0.99, "top_p": 0.5})
    # Explicit non-None wins
    assert out["temperature"] == 0.99
    # None values get filled in
    assert out["top_k"] == 20
    # top_p was not in the profile, so it stays as the caller set it
    assert out["top_p"] == 0.5


def test_profile_apply_to_kwargs_ignores_none_values_in_profile():
    """Profile parameters that are explicitly None are skipped."""
    from ethllama.cli_profile import apply_profile_to_kwargs
    from ethllama.profiles import Profile

    p = Profile(
        name="x",
        model="/m.gguf",
        parameters={"temperature": None, "top_k": 20},
    )
    out = apply_profile_to_kwargs(p, {"temperature": 0.5})
    # Profile value is None, so it doesn't override
    assert out["temperature"] == 0.5
    assert out["top_k"] == 20

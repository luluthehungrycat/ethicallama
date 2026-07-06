"""Tests for the ethllama CLI."""
import json
import os
import struct
import tempfile

import pytest
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
    assert "0.1.0" in result.output


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
    """Test that 'serve --help' shows host and port options."""
    runner = CliRunner()
    result = runner.invoke(main, ["serve", "--help"])
    assert result.exit_code == 0
    assert "host" in result.output.lower()
    assert "port" in result.output.lower()


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

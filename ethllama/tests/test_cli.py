"""Tests for the ethllama CLI."""
import pytest
from click.testing import CliRunner
from ethllama.cli import main


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

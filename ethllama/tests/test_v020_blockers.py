"""Focused regressions for the v0.2.0 release blockers."""
from __future__ import annotations

import io
from pathlib import Path

import pytest
from click.testing import CliRunner

from ethllama import api, config, inference
from ethllama.cli import _render_system_unit, _render_user_unit, main
from ethllama.engines import EngineConfig
from ethllama.profiles import Profile


class _FakeProcess:
    def __init__(self, stdout: str, returncode: int = 0):
        self.stdout = io.StringIO(stdout)
        self.stderr = io.StringIO("")
        self.returncode = returncode

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def wait(self):
        return self.returncode


def test_stream_yields_without_prompt_echo_and_preserves_assistant(monkeypatch):
    monkeypatch.setattr(inference, "require_binary", lambda *_: "/bin/llama-cli")
    monkeypatch.setattr(
        inference.subprocess,
        "Popen",
        lambda *_a, **_kw: _FakeProcess(
            "llama_model_loader: loading\n<|im_start|>assistant\nreply kept<|im_end|>\nExiting...\n"
        ),
    )
    assert "".join(inference.run_inference_stream("m.gguf", "prompt")) == "reply kept\n"


def test_stream_debug_is_unfiltered_raw_stdout(monkeypatch):
    raw = "llama_model_loader: loading\n> prompt\n<|im_start|>assistant\nreply\nExiting...\n"
    monkeypatch.setattr(inference, "require_binary", lambda *_: "/bin/llama-cli")
    monkeypatch.setattr(inference.subprocess, "Popen", lambda *_a, **_kw: _FakeProcess(raw))
    assert "".join(inference.run_inference_stream("m.gguf", "prompt", debug=True)) == raw


def test_inline_template_and_file_template_are_both_supported(tmp_path):
    messages = [{"role": "user", "content": "Hello"}]
    assert inference.format_chat_messages(messages, chat_template_path="[{{ .Prompt }}]") == "[Hello]"
    template_file = tmp_path / "template.jinja"
    template_file.write_text("FILE: {{ .Prompt }}")
    assert inference.format_chat_messages(messages, chat_template_path=str(template_file)) == "FILE: Hello"


def test_engine_template_is_quote_aware_and_exposes_generation_variables(tmp_path):
    cfg = tmp_path / "engine.yaml"
    cfg.write_text("""name: test\ntype: text\nbinary: /bin/echo\noutput_policy: llama.cpp\nargs_template: '{{ binary }} --prompt "{{ prompt }}" -n {{ max_tokens }} --stop "{{ stop | join(\",\") }}"'\n""")
    engine = EngineConfig(cfg)
    assert engine.render_command("m.gguf", "two words", max_tokens=12, stop=["END", "STOP"]) == ["/bin/echo", "--prompt", "two words", "-n", "12", "--stop", "END,STOP"]


def test_explicit_default_cli_value_beats_profile_and_model_defaults(tmp_path, monkeypatch):
    import ethllama.cli as cli
    model = tmp_path / "model.gguf"
    model.write_bytes(b"GGUF")
    profile = Profile(name="p", model=str(model), parameters={"max_tokens": 99, "temperature": 0.2}, stop=["PROFILE_STOP"])
    captured = {}
    monkeypatch.setattr(cli, "load_profile", lambda _name: profile)
    monkeypatch.setattr(cli, "load_config", lambda: {"gpu": {}, "model_defaults": {model.stem: {"max_tokens": 77, "temperature": 0.3, "stop": ["MODEL_STOP"]}}})
    monkeypatch.setattr(inference, "has_inference_engine", lambda: True)
    monkeypatch.setattr(inference, "run_inference", lambda **kwargs: captured.update(kwargs) or "ok")
    result = CliRunner().invoke(main, ["run", str(model), "-p", "hi", "--profile", "p", "--max-tokens", "2048"])
    assert result.exit_code == 0, result.output
    assert captured["max_tokens"] == 2048
    assert captured["temperature"] == 0.2
    assert captured["stop"] == ["PROFILE_STOP"]


def test_ethllama_config_override_fails_closed(tmp_path, monkeypatch):
    override = tmp_path / "credential.yaml"
    monkeypatch.setenv("ETHLLAMA_CONFIG", str(override))
    with pytest.raises(FileNotFoundError):
        config.load_config()
    override.write_text("api: [not-a-mapping")
    with pytest.raises(ValueError):
        config.load_config()
    override.write_text("api:\n  port: 12000\n")
    assert config.load_config()["api"]["port"] == 12000
    monkeypatch.setenv("ETHLLAMA_CONFIG", "relative.yaml")
    with pytest.raises(ValueError):
        config.get_config_path()


def test_generated_units_keep_credentials_out_of_public_unit():
    secret = "do-not-leak"
    user = _render_user_unit("/opt/ethllama/bin/ethllama", Path("/home/alice/.ethllama/config.yaml"))
    system = _render_system_unit("/opt/ethllama/bin/ethllama", "alice", "alice", Path("/home/alice"))
    assert "User=" not in user
    assert "LoadCredential=ethllama-config:/etc/ethllama/config.yaml" in system
    assert "Environment=ETHLLAMA_CONFIG=%d/ethllama-config" in system
    assert secret not in user + system
    assert "moritz" not in user + system
    assert "10434" not in user + system


def test_serve_uses_config_then_explicit_cli_values(monkeypatch):
    import ethllama.cli as cli
    captured = {}
    monkeypatch.setattr(cli, "load_config", lambda: {"api": {"host": "127.0.0.2", "port": 12001, "api_key": "stored"}, "gpu": {}})
    monkeypatch.setattr(api, "run_server", lambda **kwargs: captured.update(kwargs))
    result = CliRunner().invoke(main, ["serve"])
    assert result.exit_code == 0, result.output
    assert captured["host"] == "127.0.0.2" and captured["port"] == 12001 and captured["api_key"] == "stored"
    captured.clear()
    result = CliRunner().invoke(main, ["serve", "--host", "127.0.0.1", "--port", "10434", "--no-api-key"])
    assert result.exit_code == 0, result.output
    assert captured["host"] == "127.0.0.1" and captured["port"] == 10434 and captured["api_key"] == ""



def test_setup_yes_preserves_existing_key_and_secures_new_config(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "CONFIG_DIR", tmp_path / ".ethllama")
    monkeypatch.setattr(config, "CONFIG_FILE", tmp_path / ".ethllama" / "config.yaml")
    existing = {"api": {"host": "127.0.0.1", "port": 12002, "api_key": "keep-me"}, "telemetry": {"enabled": False}}
    config.save_config(existing)
    result = CliRunner().invoke(main, ["setup", "--no-install", "--yes"])
    assert result.exit_code == 0, result.output
    assert config.load_config()["api"]["api_key"] == "keep-me"
    fresh = tmp_path / "fresh"
    monkeypatch.setattr(config, "CONFIG_FILE", fresh / "config.yaml")
    result = CliRunner().invoke(main, ["setup", "--no-install", "--yes"])
    assert result.exit_code == 0, result.output
    assert (fresh / "config.yaml").stat().st_mode & 0o777 == 0o600
    assert fresh.stat().st_mode & 0o777 == 0o700


def test_system_setup_rejects_root_before_privileged_actions(monkeypatch):
    import ethllama.cli as cli
    monkeypatch.setattr(cli.os, "geteuid", lambda: 0)
    assert cli._install_system_service("/opt/ethllama", {}) is False



class _TrackedProcess(_FakeProcess):
    def __init__(self, stdout: str):
        super().__init__(stdout)
        self.wait_calls = 0

    def wait(self):
        self.wait_calls += 1
        return self.returncode


def test_native_stream_surfaces_partial_stdout_before_newline_or_exit(monkeypatch):
    proc = _TrackedProcess("partial response")
    monkeypatch.setattr(inference, "require_binary", lambda *_: "/bin/llama-cli")
    monkeypatch.setattr(inference.subprocess, "Popen", lambda *_a, **_kw: proc)
    stream = inference.run_inference_stream("m.gguf", "request")
    first_chunk = next(stream)
    assert first_chunk and "partial response".startswith(first_chunk)
    assert proc.wait_calls == 0
    assert first_chunk + "".join(stream) == "partial response"
    assert proc.wait_calls == 1


def test_custom_stream_surfaces_partial_stdout_before_process_wait(monkeypatch, tmp_path):
    import ethllama.cli as cli

    class Engine:
        name = "partial-engine"
        type = "text"
        supports_streaming = True
        output_policy = "raw"

        @staticmethod
        def render_command(**_kwargs):
            return ["partial-engine"]

    model = tmp_path / "model.gguf"
    model.write_bytes(b"GGUF")
    proc = _TrackedProcess("partial response")
    observed = []
    monkeypatch.setattr(cli, "load_engines", lambda: {"partial": Engine()})
    monkeypatch.setattr(cli.subprocess, "Popen", lambda *_a, **_kw: proc)
    monkeypatch.setattr(cli.click, "echo", lambda message="", **_kwargs: observed.append((str(message), proc.wait_calls)))
    result = CliRunner().invoke(main, ["run", str(model), "-p", "request", "--engine", "partial", "--stream"])
    assert result.exit_code == 0, result.exception
    assert ("p", 0) in observed
    assert proc.wait_calls == 1


def test_profile_run_reaches_inference_callback_without_missing_profile_argument(monkeypatch, tmp_path):
    from ethllama import profiles as profiles_module

    model = tmp_path / "model.gguf"
    model.write_bytes(b"GGUF")
    monkeypatch.setattr(profiles_module, "PROFILES_DIR", tmp_path / "profiles")
    Profile(name="delegate", model=str(model), parameters={"max_tokens": 7}).save()
    captured = {}
    monkeypatch.setattr(inference, "has_inference_engine", lambda: True)
    monkeypatch.setattr(inference, "run_inference", lambda **kwargs: captured.update(kwargs) or "ok")
    result = CliRunner().invoke(main, ["profile", "run", "delegate", "--prompt", "hello"])
    assert result.exit_code == 0, result.output
    assert captured["model_path"] == str(model)
    assert captured["max_tokens"] == 7

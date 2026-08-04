"""Configuration loader for ethicallama.

The configuration lives in ``~/.ethllama/config.yaml`` and is loaded
on-demand via :func:`load_config`.  The :data:`DEFAULT_CONFIG` dict
below documents the full schema.

Per-model defaults
------------------

Per-model defaults can be set under the ``model_defaults`` key.  Each
key is a model filename stem (e.g. ``Qwen3.5-0.8B-UD-IQ2_XXS``) and
its value is a dict of settings that act as fallbacks for that
specific model.  Explicit CLI flags always take precedence; the
per-model values only fill in settings the user did not pass.

Example ``~/.ethllama/config.yaml``::

    # Per-model defaults. Key is the model filename stem (without
    # extension). Values override CLI defaults but can be overridden
    # by explicit CLI flags.
    #
    # model_defaults:
    #   Qwen3.5-0.8B-UD-IQ2_XXS:
    #     temperature: 0.3
    #     top_k: 20
    #     n_gpu_layers: -1
    #     system_prompt: "You are a helpful assistant."
    #     ctx_size: 4096
    #   Phi-4-mini-instruct-Q5_K_M:
    #     temperature: 0.7
    #     top_k: 40
    #     n_gpu_layers: 20
    #     ctx_size: 8192
    #     gpu_backend: vulkan
    #     chat_template: /path/to/custom/template.jinja
"""

import copy
import os
from pathlib import Path
from typing import Any, Dict

import yaml

CONFIG_DIR = Path.home() / ".ethllama"
CONFIG_FILE = CONFIG_DIR / "config.yaml"

DEFAULT_CONFIG = {
    "gpu": {"backend": "vulkan", "fallback": True},
    "api": {
        "enabled": False,
        "host": "127.0.0.1",
        "port": 10434,
        "api_key": "",
        "idle_timeout": 0,
    },
    "telemetry": {"enabled": False},
    "model_dirs": [],
    "engines": {
        "binary_dir": None,
        "llama_cli": None,
        "llama_embedding": None,
        "llama_quantize": None,
    },
    "model_defaults": {},
}


def get_config_path() -> Path:
    """Return the active configuration path.

    ``ETHLLAMA_CONFIG`` must be an absolute path.  This is intentionally
    resolved at call time so a systemd credential path can be supplied after
    import, while ordinary users retain ``~/.ethllama/config.yaml``.
    """
    override = os.environ.get("ETHLLAMA_CONFIG")
    if override is None:
        return CONFIG_FILE
    path = Path(override)
    if not path.is_absolute():
        raise ValueError("ETHLLAMA_CONFIG must be an absolute path")
    return path


def load_config() -> Dict[str, Any]:
    """Load config, failing closed for an explicit environment override."""
    override_set = "ETHLLAMA_CONFIG" in os.environ
    path = get_config_path()
    if not path.exists():
        if override_set:
            raise FileNotFoundError(f"ETHLLAMA_CONFIG does not exist: {path}")
        return copy.deepcopy(DEFAULT_CONFIG)
    try:
        with path.open("r", encoding="utf-8") as config_file:
            data = yaml.safe_load(config_file)
    except OSError as exc:
        if override_set:
            raise RuntimeError(f"Cannot read ETHLLAMA_CONFIG: {path}") from exc
        raise
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML configuration: {path}") from exc
    if data is None:
        return copy.deepcopy(DEFAULT_CONFIG)
    if not isinstance(data, dict):
        raise ValueError(f"Configuration must be a mapping: {path}")
    return data


def save_config(config: Dict[str, Any]) -> None:
    """Persist configuration to the active config path."""
    path = get_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as config_file:
        yaml.safe_dump(config, config_file, sort_keys=False)

def init_config() -> None:
    """Run onboarding to create config.yaml."""
    import click
    config = DEFAULT_CONFIG.copy()

    click.echo("Welcome to ethicallama! Let's set up your environment.")
    config["gpu"]["backend"] = click.prompt(
        "Choose GPU backend",
        default="vulkan",
        type=click.Choice(["vulkan", "rocm", "cuda", "cpu"]),
    )
    config["api"]["enabled"] = click.confirm("Enable HTTP API?", default=False)
    if config["api"]["enabled"]:
        config["api"]["api_key"] = click.prompt("Set API key (leave empty for none)", default="")
    config["telemetry"]["enabled"] = click.confirm(
        "Enable anonymous telemetry? (WARNING: This shares usage statistics)",
        default=False,
    )

    # Add default model dirs
    default_dirs = [
        Path.home() / "models",
        Path.home() / ".cache" / "huggingface" / "hub",
    ]
    for dir_path in default_dirs:
        if dir_path.exists():
            config["model_dirs"].append(str(dir_path))

    save_config(config)
    click.echo("Configuration saved to ~/.ethllama/config.yaml")

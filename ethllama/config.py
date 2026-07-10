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

import os
from pathlib import Path
from typing import Dict, Any
import yaml

CONFIG_DIR = Path.home() / ".ethllama"
CONFIG_FILE = CONFIG_DIR / "config.yaml"

DEFAULT_CONFIG = {
    "gpu": {
        "backend": "vulkan",
        "fallback": True,
    },
    "api": {
        "enabled": False,
        "host": "127.0.0.1",
        "port": 8080,
        "api_key": "",
        # TTL (idle model unloading) for ``ethllama serve``.  When > 0 the
        # server auto-unloads a pre-loaded model after this many seconds of
        # inactivity, freeing GPU/RAM.  0 (default) disables the feature and
        # the model stays loaded for the lifetime of the server.  Mirrors
        # the ``--idle-timeout``/``--ttl`` CLI flag on ``ethllama serve``.
        "idle_timeout": 0,
    },
    "telemetry": {
        "enabled": False,
    },
    "model_dirs": [],
    "engines": {
        "binary_dir": None,
        "llama_cli": None,
        "llama_embedding": None,
        "llama_quantize": None,
    },
    # Per-model defaults. Optional dict mapping model filename stems
    # (e.g. "Phi-4-mini-instruct-Q5_K_M") to a dict of inference
    # settings that act as fallbacks for that model. See module
    # docstring for the supported keys and behaviour.
    "model_defaults": {},
}

def load_config() -> Dict[str, Any]:
    if not CONFIG_FILE.exists():
        return DEFAULT_CONFIG.copy()
    with open(CONFIG_FILE, "r") as f:
        return yaml.safe_load(f) or DEFAULT_CONFIG.copy()

def save_config(config: Dict[str, Any]) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_FILE, "w") as f:
        yaml.dump(config, f)

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

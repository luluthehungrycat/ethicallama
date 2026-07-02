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
    },
    "telemetry": {
        "enabled": False,
    },
    "model_dirs": [],
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

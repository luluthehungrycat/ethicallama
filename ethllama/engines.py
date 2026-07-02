import os
import yaml
import subprocess
from pathlib import Path
from typing import Dict, List, Optional
from jinja2 import Template

ENGINES_DIR = Path.home() / ".ethllama" / "engines"

class EngineConfig:
    def __init__(self, config_path: Path):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        self.name = self.config["name"]
        self.type = self.config["type"]
        self.binary = self.config["binary"]
        self.args_template = self.config["args_template"]
        self.env = self.config.get("env", {})
        self.pre_check = self.config.get("pre_check", "")
        self.supports_streaming = self.config.get("supports_streaming", False)
        self.model_extensions = self.config.get("model_extensions", [])

    def validate(self) -> bool:
        """Check if the engine binary exists and pre_check passes."""
        if not os.path.exists(self.binary):
            return False
        if self.pre_check:
            try:
                subprocess.run(
                    Template(self.pre_check).render(binary=self.binary),
                    shell=True,
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
            except subprocess.CalledProcessError:
                return False
        return True

    def render_command(
        self,
        model_path: str,
        prompt: str,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        threads: int = 4,
        n_gpu_layers: int = 0,
        gpu_backend: str = "cpu",
        output: Optional[str] = None,
    ) -> List[str]:
        """Render the engine's CLI command from ethllama's args."""
        template = Template(self.args_template)
        command = template.render(
            binary=self.binary,
            model_path=model_path,
            prompt=prompt,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            threads=threads,
            n_gpu_layers=n_gpu_layers,
            gpu_backend=gpu_backend,
            output=output,
        )
        return command.split()

def load_engines() -> Dict[str, EngineConfig]:
    """Load all engine configs from ~/.ethllama/engines/."""
    engines = {}
    if not ENGINES_DIR.exists():
        return engines
    for config_path in ENGINES_DIR.glob("*.yaml"):
        try:
            config = EngineConfig(config_path)
            if config.validate():
                engines[config.name] = config
        except Exception as e:
            print(f"WARNING: Failed to load engine {config_path}: {e}")
    return engines

def get_engine(engine_name: str) -> Optional[EngineConfig]:
    """Get an engine config by name."""
    engines = load_engines()
    return engines.get(engine_name)

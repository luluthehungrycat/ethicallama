import os
import yaml
import subprocess
import shlex
from pathlib import Path
from typing import Dict, List, Optional
from jinja2 import StrictUndefined, Template

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
        self.default_model = self.config.get("default_model")
        # Arbitrary engines are raw by default. Set output_policy: llama.cpp for
        # compatible tools whose stdout follows llama.cpp banner/echo conventions.
        self.output_policy = self.config.get("output_policy", "raw")
        if self.output_policy not in {"raw", "llama.cpp"}:
            raise ValueError("output_policy must be raw or llama.cpp")

    def validate(self) -> bool:
        """Check if the engine binary exists and pre_check passes."""
        if not os.path.exists(self.binary):
            return False
        if self.pre_check:
            try:
                subprocess.run(
                    shlex.split(Template(self.pre_check, undefined=StrictUndefined).render(binary=self.binary)),
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
        max_tokens: int = 2048,
        stop: Optional[List[str]] = None,
    ) -> List[str]:
        """Render an argv list without invoking a shell.

        Templates receive ``max_tokens`` and ``stop`` (also available as
        ``stop_sequences``).  Use shell-style quoting in YAML for values that
        may contain whitespace; :func:`shlex.split` preserves those arguments
        while subprocess receives an argv list, never a shell command.
        """
        template = Template(self.args_template, undefined=StrictUndefined)
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
            max_tokens=max_tokens,
            stop=list(stop or []),
            stop_sequences=list(stop or []),
        )
        return shlex.split(command)

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


# ---------------------------------------------------------------------------
# Engine discovery: scan PATH for known inference binaries and generate
# YAML configs in ~/.ethllama/engines/ for any matches.
# ---------------------------------------------------------------------------

# Catalogue of well-known inference engine binaries.  Each entry describes
# how to render a CLI invocation from ethllama's arguments.  The keys are
# the bare binary names we search for with shutil.which().  ``template``
# is a Jinja2 template that mirrors the same variables exposed to user
# engine configs in render_command() — model_path, prompt, temperature,
# top_p, top_k, threads, n_gpu_layers, gpu_backend, output.
KNOWN_ENGINES: Dict[str, Dict[str, str]] = {
    "ollama": {
        "type": "text",
        "template": "serve --model {{ model_path }}",
        "description": "Ollama (local LLM server)",
    },
    "llama-cli": {
        "type": "text",
        "template": (
            "-m {{ model_path }} -p \"{{ prompt }}\" "
            "-n {{ max_tokens }} --temp {{ temperature }} "
            "--top-k {{ top_k }} --top-p {{ top_p }}"
        ),
        "description": "llama.cpp CLI",
    },
    "llama-server": {
        "type": "text",
        "template": (
            "-m {{ model_path }} --port {{ port | default(8080) }}"
        ),
        "description": "llama.cpp server",
    },
    "llama-embedding": {
        "type": "text",
        "template": (
            "-m {{ model_path }} -p \"{{ prompt }}\" "
            "--embd-output-format json"
        ),
        "description": "llama.cpp embeddings",
    },
    "llama-quantize": {
        "type": "text",
        "template": (
            "{{ model_path }} {{ output }} "
            "{{ quantize_type | default('q4_k_m') | upper }}"
        ),
        "description": "llama.cpp quantize tool",
    },
    "whisper-cli": {
        "type": "stt",
        "template": "-m {{ model_path }} -f {{ audio_file }} -otxt",
        "description": "whisper.cpp CLI (speech-to-text)",
    },
    "whisper-server": {
        "type": "stt",
        "template": (
            "--model {{ model_path }} --port {{ port | default(8081) }}"
        ),
        "description": "whisper.cpp server",
    },
    "voxtral": {
        "type": "tts",
        "template": (
            "speak --input \"{{ prompt }}\" --output {{ output }} "
            "--model {{ model_path }}"
        ),
        "description": "Voxtral real-time TTS/STT",
    },
}


def discover_engines(binary_name: Optional[str] = None) -> Dict[str, str]:
    """Scan PATH for known inference engine binaries.

    Args:
        binary_name: If provided, only search for this specific binary
            (whether or not it appears in ``KNOWN_ENGINES`` — users may
            want to test for a custom binary they have installed).  If
            ``None``, search for every key in :data:`KNOWN_ENGINES`.

    Returns:
        Dict mapping found binary name -> absolute path on disk.
    """
    import shutil

    found: Dict[str, str] = {}
    if binary_name is not None:
        candidates = [binary_name]
    else:
        candidates = list(KNOWN_ENGINES.keys())

    for name in candidates:
        path = shutil.which(name)
        if path:
            found[name] = path

    return found


def generate_engine_config(
    name: str,
    binary_path: str,
    engines_dir: Optional[Path] = None,
    overwrite: bool = False,
) -> Optional[Path]:
    """Auto-generate a minimal engine YAML file for *name*.

    The generated config uses the template shipped in
    :data:`KNOWN_ENGINES` for the engine type; if the binary is not in
    the catalogue (e.g. user passed a name not in
    ``discover_engines``' default scan) we still write a minimal
    config so the user has something to edit.

    Args:
        name: The engine name (also used as the YAML filename stem).
        binary_path: Absolute path to the discovered binary.
        engines_dir: Directory to write the YAML into.  Defaults to
            ``~/.ethllama/engines/``.
        overwrite: If True, overwrite an existing file.  If False and
            the file already exists, this function returns ``None``
            and writes nothing.

    Returns:
        The path of the generated YAML file, or ``None`` when
        ``overwrite`` is False and the file already exists.
    """
    if engines_dir is None:
        engines_dir = ENGINES_DIR
    engines_dir = Path(engines_dir)
    engines_dir.mkdir(parents=True, exist_ok=True)

    target = engines_dir / f"{name}.yaml"
    if target.exists() and not overwrite:
        return None

    spec = KNOWN_ENGINES.get(name, {})
    engine_type = spec.get("type", "text")
    template = spec.get("template", "")

    config = {
        "name": name,
        "type": engine_type,
        "binary": binary_path,
        "args_template": template,
    }

    description = spec.get("description")
    if description:
        config["description"] = description

    with open(target, "w") as f:
        yaml.safe_dump(config, f, default_flow_style=False, sort_keys=False)

    return target

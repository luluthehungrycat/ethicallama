"""Model profiles: parameter presets that don't require model duplication.

A profile is a YAML file under ``~/.ethllama/profiles/<name>.yaml`` that
references an existing model (by index stem or absolute path) and
specifies inference parameters (temperature, top_p, max_tokens, …) plus
optional system prompt, chat template, and stop sequences.  Unlike
Ollama's Modelfile system, profiles do **not** copy the underlying
GGUF file — the model is referenced by name, and the parameters are
applied at runtime via the engine's CLI flags or the inference API.

The format is intentionally simple: a flat YAML dict that maps to the
:func:`Profile` dataclass below.  See the project README and
``docs/USAGE.md`` for usage examples.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml

# Default location of the profiles directory.  Tests can monkey-patch
# this module attribute (e.g. ``profiles.PROFILES_DIR = tmp_path / ...``)
# to redirect profile reads/writes; helpers below resolve it at call
# time so the patch is honoured.
PROFILES_DIR: Path = Path.home() / ".ethllama" / "profiles"


def _resolve_dir(profiles_dir: Optional[Union[Path, str]]) -> Path:
    """Return the effective profiles directory.

    Falls back to the module-level :data:`PROFILES_DIR` when the caller
    passes ``None``.  Doing the lookup here (not as a default argument)
    means tests can monkey-patch the module attribute and have it take
    effect on subsequent calls.
    """
    if profiles_dir is None:
        return PROFILES_DIR
    return Path(profiles_dir)


@dataclass
class Profile:
    """A named set of inference parameters bound to a model.

    Attributes:
        name: Profile identifier (also the YAML filename stem).
        model: Model identifier — either an index stem (e.g.
            ``"Qwen3.5-0.8B-UD-IQ2_XXS"``) or an absolute path to a
            GGUF file.
        description: Free-form description (shown by ``profile show``).
        parameters: Inference parameters.  Recognised keys:
            ``temperature``, ``top_p``, ``top_k``, ``max_tokens``,
            ``n_gpu_layers``, ``ctx_size``, ``threads``.  Unknown keys
            are preserved on round-trip but ignored by the CLI.
        system_prompt: Optional system message prepended to the user
            prompt.  When empty, no system message is added.
        template: Inline Jinja2 chat template.  When empty, the GGUF
            model's own template is used (if any), otherwise the
            built-in ``<|im_start|>`` fallback.
        stop: Optional list of stop sequences forwarded to the
            inference engine.
        metadata: Free-form dict for user annotations; preserved on
            round-trip.
    """

    name: str
    model: str
    description: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)
    system_prompt: str = ""
    template: str = ""
    stop: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: Path) -> "Profile":
        """Load a profile from a YAML file.

        ``name`` defaults to the file stem when the YAML does not
        include it explicitly.
        """
        path = Path(path)
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            raise ValueError(
                f"Profile YAML must be a mapping, got {type(data).__name__}: {path}"
            )
        if "model" not in data:
            raise ValueError(f"Profile YAML is missing required 'model' key: {path}")
        return cls(
            name=data.get("name", path.stem),
            model=data["model"],
            description=data.get("description", "") or "",
            parameters=data.get("parameters", {}) or {},
            system_prompt=data.get("system_prompt", "") or "",
            template=data.get("template", "") or "",
            stop=list(data.get("stop", []) or []),
            metadata=data.get("metadata", {}) or {},
        )

    def to_yaml(self) -> str:
        """Serialize the profile to a YAML string."""
        return yaml.dump(
            asdict(self),
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
        )

    def save(self, profiles_dir: Optional[Path] = None) -> Path:
        """Write the profile to ``<profiles_dir>/<name>.yaml``.

        Creates the directory if it does not exist.  Returns the path
        that was written.
        """
        profiles_dir = _resolve_dir(profiles_dir)
        profiles_dir.mkdir(parents=True, exist_ok=True)
        path = profiles_dir / f"{self.name}.yaml"
        path.write_text(self.to_yaml(), encoding="utf-8")
        return path

    # ------------------------------------------------------------------
    # Convenience helpers used by the CLI when applying a profile.
    # ------------------------------------------------------------------

    def get_param(self, key: str, default: Any = None) -> Any:
        """Return ``parameters[key]`` if present, else ``default``."""
        if not isinstance(self.parameters, dict):
            return default
        return self.parameters.get(key, default)


def list_profiles(profiles_dir: Optional[Path] = None) -> List[str]:
    """Return sorted list of profile names (without the ``.yaml`` extension).

    Returns an empty list if the profiles directory does not exist.
    """
    profiles_dir = _resolve_dir(profiles_dir)
    if not profiles_dir.exists():
        return []
    return sorted(p.stem for p in profiles_dir.glob("*.yaml"))


def load_profile(name: str, profiles_dir: Optional[Path] = None) -> Profile:
    """Load a profile by name.

    Raises:
        FileNotFoundError: if no YAML file exists for ``name``.
        ValueError: if the YAML is malformed (missing ``model`` or
            not a mapping).
    """
    profiles_dir = _resolve_dir(profiles_dir)
    path = profiles_dir / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Profile not found: {name}")
    return Profile.from_yaml(path)


def profile_exists(name: str, profiles_dir: Optional[Path] = None) -> bool:
    """Return True if a profile YAML exists for ``name``."""
    profiles_dir = _resolve_dir(profiles_dir)
    return (profiles_dir / f"{name}.yaml").exists()


def delete_profile(name: str, profiles_dir: Optional[Path] = None) -> bool:
    """Delete a profile YAML.  Returns True if a file was removed."""
    profiles_dir = _resolve_dir(profiles_dir)
    path = profiles_dir / f"{name}.yaml"
    if not path.exists():
        return False
    path.unlink()
    return True

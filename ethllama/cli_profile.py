"""CLI subcommand group for managing and running model profiles.

Profiles are stored as YAML files under ``~/.ethllama/profiles/<name>.yaml``
and reference an existing model by name (or absolute path).  This
mirrors the *intent* of Ollama's Modelfile system without duplicating
the underlying GGUF file: the parameters are applied at inference time
via the engine's CLI flags or the in-process inference API.

Commands:

    ethllama profile list                 # List all profiles
    ethllama profile show <name>          # Show profile details
    ethllama profile create <name>        # Create (or overwrite) a profile
    ethllama profile edit <name>          # Edit in $EDITOR
    ethllama profile delete <name>        # Delete a profile
    ethllama profile run <name>           # Run inference with profile defaults

The group is registered onto the main ``ethllama`` CLI by
:func:`register_commands`, called from ``cli.py``.
"""

from __future__ import annotations

import copy
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import click

from .index import resolve_model_path
from . import profiles as _profiles_mod
from .profiles import (
    Profile,
    delete_profile,
    list_profiles,
    load_profile,
    profile_exists,
)


__all__ = [
    "profile_group",
    "register_commands",
    "apply_profile_to_kwargs",
]


# ---------------------------------------------------------------------------
# Profile application helpers
# ---------------------------------------------------------------------------

# Mapping from Profile.parameter key → CLI / inference kwarg name.
# Keep this as the single source of truth so ``profile run``,
# ``run --profile``, and ``serve --profile`` stay in sync.
_PROFILE_PARAM_MAP: Dict[str, str] = {
    "temperature": "temperature",
    "top_p": "top_p",
    "top_k": "top_k",
    "max_tokens": "max_tokens",
    "n_gpu_layers": "n_gpu_layers",
    "ctx_size": "ctx_size",
    "threads": "threads",
    "gpu_backend": "gpu_backend",
}


def apply_profile_to_kwargs(profile: Profile, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Return a copy of *kwargs* with profile parameters applied.

    For each parameter the profile defines, if the corresponding kwarg
    is missing or ``None``, the profile value is used.  This mirrors
    the precedence rule used by the ``run`` command: explicit CLI
    flags win, profile values fill in the gaps.
    """
    out: Dict[str, Any] = dict(kwargs)
    for src_key, dst_key in _PROFILE_PARAM_MAP.items():
        if src_key not in profile.parameters:
            continue
        value = profile.parameters[src_key]
        if value is None:
            continue
        if out.get(dst_key) is None:
            out[dst_key] = value
    return out


def _resolve_profile_model(profile: Profile) -> Optional[str]:
    """Resolve the profile's ``model`` field to an on-disk path.

    Resolution order:

    1. ``resolve_model_path(<model>)`` — index lookup.
    2. Direct filesystem check (absolute or relative path).

    Returns the path or ``None`` when the model cannot be located.
    """
    candidate = profile.model
    resolved = resolve_model_path(candidate)
    if resolved:
        return resolved
    if candidate and os.path.exists(candidate):
        return os.path.abspath(candidate)
    return None


def _inject_profile_into_config(
    profile: Profile, model_path: str
) -> Tuple[Any, Any]:
    """Monkey-patch ``load_config`` to apply *profile* via ``model_defaults``.

    Returns a ``(cli_restore, config_restore)`` tuple of the original
    ``load_config`` callables.  The caller MUST call both restorers in
    a ``finally`` block.
    """
    from . import cli as cli_mod
    from . import config as config_mod

    base_cfg = config_mod.load_config()
    if not isinstance(base_cfg, dict):
        base_cfg = {}
    patched_cfg = copy.deepcopy(base_cfg)
    model_defaults = patched_cfg.setdefault("model_defaults", {})
    if not isinstance(model_defaults, dict):
        model_defaults = {}
        patched_cfg["model_defaults"] = model_defaults
    stem = Path(model_path).stem
    entry = model_defaults.setdefault(stem, {})
    if not isinstance(entry, dict):
        entry = {}
        model_defaults[stem] = entry
    if profile.template:
        entry["chat_template"] = profile.template
    if profile.system_prompt:
        entry["system_prompt"] = profile.system_prompt

    cli_orig = cli_mod.load_config
    config_orig = config_mod.load_config
    cli_mod.load_config = lambda: patched_cfg  # type: ignore[assignment]
    config_mod.load_config = lambda: patched_cfg  # type: ignore[assignment]
    return cli_orig, config_orig


# ---------------------------------------------------------------------------
# Profile group definition
# ---------------------------------------------------------------------------


@click.group(name="profile")
def profile_group():
    """Manage and run model profiles (replaces Ollama's Modelfile)."""
    pass


# ---------------------------------------------------------------------------
# `ethllama profile list`
# ---------------------------------------------------------------------------


@profile_group.command(name="list")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def profile_list(as_json: bool) -> None:
    """List all configured profiles."""
    names = list_profiles()
    if as_json:
        click.echo(json.dumps({"profiles": names}, indent=2))
        return

    if not names:
        click.echo("No profiles configured.")
        click.echo(
            "Create one with: ethllama profile create <name> --model <model>"
        )
        return

    click.echo(f"Configured profiles ({len(names)}):")
    for name in names:
        try:
            prof = load_profile(name)
        except Exception as exc:  # noqa: BLE001 — best-effort display
            click.echo(f"  {name}  (failed to load: {exc})")
            continue
        model_display = prof.model
        desc = f" — {prof.description}" if prof.description else ""
        click.echo(f"  {name}  [model: {model_display}]{desc}")


# ---------------------------------------------------------------------------
# `ethllama profile show`
# ---------------------------------------------------------------------------


@profile_group.command(name="show")
@click.argument("name")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def profile_show(name: str, as_json: bool) -> None:
    """Show details of a profile."""
    try:
        prof = load_profile(name)
    except FileNotFoundError:
        click.echo(f"Error: profile '{name}' not found.", err=True)
        click.echo(
            "Use 'ethllama profile list' to see available profiles, or "
            f"'ethllama profile create {name} --model <model>' to create it.",
            err=True,
        )
        sys.exit(1)
    except ValueError as exc:
        click.echo(f"Error loading profile '{name}': {exc}", err=True)
        sys.exit(1)

    if as_json:
        click.echo(json.dumps(asdict(prof), indent=2, default=str))
        return

    click.echo(f"Profile: {prof.name}")
    click.echo(f"  Model:        {prof.model}")
    if prof.description:
        click.echo(f"  Description:  {prof.description}")
    if prof.parameters:
        click.echo("  Parameters:")
        for k, v in prof.parameters.items():
            click.echo(f"    {k}: {v}")
    if prof.system_prompt:
        click.echo("  System prompt:")
        for line in prof.system_prompt.splitlines():
            click.echo(f"    {line}")
    if prof.template:
        click.echo("  Template (inline):")
        for line in prof.template.splitlines():
            click.echo(f"    {line}")
    if prof.stop:
        click.echo(f"  Stop sequences: {prof.stop}")
    if prof.metadata:
        click.echo(f"  Metadata: {prof.metadata}")


# ---------------------------------------------------------------------------
# `ethllama profile create`
# ---------------------------------------------------------------------------


@profile_group.command(name="create")
@click.argument("name")
@click.option("--model", required=True, help="Model path or name from index")
@click.option("--system-prompt", default="", help="System prompt")
@click.option("--description", default="", help="Profile description")
@click.option("--temperature", type=float, default=None)
@click.option("--top-p", type=float, default=None)
@click.option("--top-k", type=int, default=None)
@click.option("--max-tokens", type=int, default=None)
@click.option("--n-gpu-layers", type=int, default=None)
@click.option("--ctx-size", type=int, default=None)
@click.option("--threads", type=int, default=None)
@click.option(
    "--gpu-backend", default=None,
    help="GPU backend (vulkan, rocm, cuda, cpu)",
)
@click.option("--stop", "stop", multiple=True, help="Stop sequences (can be repeated)")
@click.option("--template", default=None, help="Inline Jinja template")
@click.option(
    "--from-yaml",
    "from_yaml",
    type=click.Path(exists=True, dir_okay=False, readable=True),
    help="Create from an existing YAML file (other flags ignored)",
)
@click.option(
    "--overwrite", is_flag=True, default=False,
    help="Overwrite an existing profile",
)
def profile_create(
    name: str,
    model: str,
    system_prompt: str,
    description: str,
    temperature: Optional[float],
    top_p: Optional[float],
    top_k: Optional[int],
    max_tokens: Optional[int],
    n_gpu_layers: Optional[int],
    ctx_size: Optional[int],
    threads: Optional[int],
    gpu_backend: Optional[str],
    stop: tuple,
    template: Optional[str],
    from_yaml: Optional[str],
    overwrite: bool,
) -> None:
    """Create a new profile."""
    if from_yaml is not None:
        try:
            prof = Profile.from_yaml(Path(from_yaml))
        except (ValueError, OSError) as exc:
            click.echo(f"Error reading {from_yaml}: {exc}", err=True)
            sys.exit(1)
        # Re-stamp the profile name with the positional argument so the
        # user can rename on import.
        prof.name = name
        prof.model = model or prof.model
    else:
        params: Dict[str, Any] = {}
        for key, value in (
            ("temperature", temperature),
            ("top_p", top_p),
            ("top_k", top_k),
            ("max_tokens", max_tokens),
            ("n_gpu_layers", n_gpu_layers),
            ("ctx_size", ctx_size),
            ("threads", threads),
            ("gpu_backend", gpu_backend),
        ):
            if value is not None:
                params[key] = value
        prof = Profile(
            name=name,
            model=model,
            description=description,
            parameters=params,
            system_prompt=system_prompt,
            template=template or "",
            stop=list(stop),
        )

    if profile_exists(name) and not overwrite:
        click.echo(
            f"Error: profile '{name}' already exists. Use --overwrite to replace it.",
            err=True,
        )
        sys.exit(1)

    target = prof.save()
    click.echo(f"Profile '{name}' saved to {target}")


# ---------------------------------------------------------------------------
# `ethllama profile edit`
# ---------------------------------------------------------------------------


@profile_group.command(name="edit")
@click.argument("name")
def profile_edit(name: str) -> None:
    """Edit a profile in your $EDITOR."""
    if not profile_exists(name):
        click.echo(f"Error: profile '{name}' not found.", err=True)
        sys.exit(1)
    editor = os.environ.get("EDITOR") or os.environ.get("VISUAL") or "vi"
    path = _profiles_mod.PROFILES_DIR / f"{name}.yaml"
    click.echo(f"Opening {path} with {editor}...")
    try:
        os.execvp(editor, [editor, str(path)])
    except FileNotFoundError:
        click.echo(
            f"Error: editor '{editor}' not found. Set $EDITOR or edit "
            f"{path} manually.",
            err=True,
        )
        sys.exit(1)


# ---------------------------------------------------------------------------
# `ethllama profile delete`
# ---------------------------------------------------------------------------


@profile_group.command(name="delete")
@click.argument("name")
@click.confirmation_option(prompt="Are you sure?")
def profile_delete(name: str) -> None:
    """Delete a profile."""
    if not delete_profile(name):
        click.echo(f"Error: profile '{name}' not found.", err=True)
        sys.exit(1)
    click.echo(f"Deleted profile '{name}'.")


# ---------------------------------------------------------------------------
# `ethllama profile run`
# ---------------------------------------------------------------------------


def _profile_run_impl(
    name: str,
    prompt: str,
    max_tokens: Optional[int],
    temperature: Optional[float],
    top_p: Optional[float],
    top_k: Optional[int],
    n_gpu_layers: Optional[int],
    threads: Optional[int],
    output: Optional[str],
    stream: bool,
) -> None:
    """Run ``ethllama run`` with the profile's settings applied."""
    try:
        prof = load_profile(name)
    except FileNotFoundError:
        click.echo(f"Error: profile '{name}' not found.", err=True)
        sys.exit(1)
    except ValueError as exc:
        click.echo(f"Error loading profile '{name}': {exc}", err=True)
        sys.exit(1)

    model_path = _resolve_profile_model(prof)
    if not model_path:
        click.echo(
            f"Error: profile '{name}' references model '{prof.model}' which "
            f"is not in the index and is not a valid path.",
            err=True,
        )
        click.echo(
            "Use 'ethllama index <directory>' to index the model, or "
            "edit the profile to use an absolute path.",
            err=True,
        )
        sys.exit(1)

    # Build the effective kwargs, starting with profile defaults and
    # then layering the explicit CLI overrides on top.
    effective: Dict[str, Any] = apply_profile_to_kwargs(prof, {})
    if temperature is not None:
        effective["temperature"] = temperature
    if top_p is not None:
        effective["top_p"] = top_p
    if top_k is not None:
        effective["top_k"] = top_k
    if max_tokens is not None:
        effective["max_tokens"] = max_tokens
    if n_gpu_layers is not None:
        effective["n_gpu_layers"] = n_gpu_layers
    if threads is not None:
        effective["threads"] = threads

    # Re-enter ``ethllama run`` programmatically with the resolved args.
    from .cli import run as run_cmd

    ctx_args: Dict[str, Any] = {
        "model": model_path,
        "prompt": prompt,
        "temperature": effective.get("temperature", 0.7),
        "top_p": effective.get("top_p", 0.9),
        "top_k": effective.get("top_k", 40),
        "threads": effective.get("threads", 4),
        "n_gpu_layers": effective.get("n_gpu_layers", 0),
        "gpu_backend": effective.get("gpu_backend", "auto"),
        "engine": None,
        "output": output,
        "stream": stream,
        "max_tokens": effective.get("max_tokens", 2048),
        "interactive": False,
        "prompt_prefix": "> ",
        "max_history": 10,
        "system_prompt": prof.system_prompt or None,
        "binary_dir": None,
        "debug": False,
    }

    needs_inject = bool(prof.template or prof.system_prompt)
    if needs_inject:
        cli_orig, config_orig = _inject_profile_into_config(prof, model_path)
        try:
            run_cmd.callback(**ctx_args)  # type: ignore[attr-defined]
        finally:
            from . import cli as cli_mod
            from . import config as config_mod
            cli_mod.load_config = cli_orig  # type: ignore[assignment]
            config_mod.load_config = config_orig  # type: ignore[assignment]
    else:
        run_cmd.callback(**ctx_args)  # type: ignore[attr-defined]


@profile_group.command(name="run")
@click.argument("name")
@click.option("--prompt", "-p", required=True, help="Input prompt")
@click.option(
    "--max-tokens", type=int, default=None,
    help="Override profile's max_tokens",
)
@click.option(
    "--temperature", "-t", type=float, default=None,
    help="Override profile's temperature",
)
@click.option("--top-p", type=float, default=None, help="Override profile's top_p")
@click.option("--top-k", type=int, default=None, help="Override profile's top_k")
@click.option(
    "--n-gpu-layers", type=int, default=None,
    help="Override profile's n_gpu_layers",
)
@click.option("--threads", type=int, default=None, help="Override profile's threads")
@click.option("--output", "-o", default=None, help="Save output to file")
@click.option("--stream", "-s", is_flag=True, default=False, help="Stream output")
def profile_run(
    name: str,
    prompt: str,
    max_tokens: Optional[int],
    temperature: Optional[float],
    top_p: Optional[float],
    top_k: Optional[int],
    n_gpu_layers: Optional[int],
    threads: Optional[int],
    output: Optional[str],
    stream: bool,
) -> None:
    """Run inference using a profile's settings."""
    _profile_run_impl(
        name=name,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        n_gpu_layers=n_gpu_layers,
        threads=threads,
        output=output,
        stream=stream,
    )


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def register_commands(cli: click.Group) -> None:
    """Attach the ``profile`` group to *cli*."""
    cli.add_command(profile_group)

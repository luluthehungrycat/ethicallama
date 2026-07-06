"""Management CLI subcommands (``rm``, ``info``) for ethicallama.

This module is the extension point for non-inference management
commands. The orchestrator wires these onto the main click group via
:func:`register_commands`. Keeping the commands isolated here lets us
grow the management surface area without touching the inference path.

The GGUF header parser and the human-readable size helper are also
defined here so that ``cli_mgmt`` is self-contained and can be
registered independently of :mod:`ethllama.cli`.
"""
from __future__ import annotations

import os
import sys
import json
import time
import struct
from pathlib import Path
from typing import Any, Dict, List, Optional

import click

from .index import (
    load_index,
    save_index,
    add_to_index,
    remove_from_index,
    resolve_model_path,
    find_in_index,
    list_all_indexed,
)


__all__ = [
    "rm_cmd",
    "info_cmd",
    "register_commands",
    "list_all_indexed",
    "_read_gguf_metadata",
    "_human_size",
]


# ---------------------------------------------------------------------------
# GGUF header parser (stdlib only — no external dependency)
# ---------------------------------------------------------------------------
#
# Layout (https://github.com/ggml-org/ggml/blob/master/docs/gguf.md):
#
#   bytes 0..3    : magic "GGUF" (4 bytes)
#   bytes 4..7    : version    (uint32)
#   bytes 8..15   : tensor_count   (uint64)
#   bytes 16..23  : metadata_kv_count (uint64)
#   then ``metadata_kv_count`` records of { key, value }
#     key   : uint64 length + UTF-8 bytes
#     value : uint32 type tag + value (size depends on type)
#
# Value types we handle: uint8/16/32/64, int8/16/32/64, float32/64,
# bool, string, and array-of-T. Anything else is skipped.
# ---------------------------------------------------------------------------

_GGUF_TYPE_UINT8 = 0
_GGUF_TYPE_INT8 = 1
_GGUF_TYPE_UINT16 = 2
_GGUF_TYPE_INT16 = 3
_GGUF_TYPE_UINT32 = 4
_GGUF_TYPE_INT32 = 5
_GGUF_TYPE_FLOAT32 = 6
_GGUF_TYPE_BOOL = 7
_GGUF_TYPE_STRING = 8
_GGUF_TYPE_ARRAY = 9
_GGUF_TYPE_UINT64 = 10
_GGUF_TYPE_INT64 = 11
_GGUF_TYPE_FLOAT64 = 12

_GGUF_MAGIC = b"GGUF"


def _human_size(size_bytes: int) -> str:
    """Format a byte count as a human-readable string (e.g. ``524.3 MB``)."""
    size = float(size_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(size) < 1024.0:
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} PB"


def _read_gguf_metadata(path: str, max_bytes: int = 4 * 1024 * 1024) -> Optional[Dict[str, Any]]:
    """Read GGUF header metadata from the first *max_bytes* of *path*.

    Returns a ``dict`` of metadata key -> value, plus three synthetic
    fields:

    * ``magic`` — always ``"GGUF"`` for valid files
    * ``version`` — the GGUF version (int, typically 3)
    * ``__tensor_count`` — the number of tensors declared
    * ``__metadata_kv_count`` — the number of metadata KV pairs declared

    Returns ``None`` when the file is missing, unreadable, or doesn't
    start with the GGUF magic bytes.

    Tries the ``gguf`` Python package first for full accuracy; falls
    back to a stdlib parser if the package isn't installed or can't
    handle the file (e.g. for synthetic test fixtures).
    """
    # --- try the full gguf library first ---
    try:
        import gguf  # type: ignore[import-untyped]
        reader = gguf.GGUFReader(str(path), "r")
        metadata: Dict[str, Any] = {}
        for field in reader.fields.values():
            if field.name.startswith("tensor"):
                continue
            try:
                vals = field.parts[field.data]
            except Exception:
                continue
            if len(vals) == 1:
                v = vals[0]
                metadata[field.name] = v.tolist() if hasattr(v, "tolist") else v
            else:
                metadata[field.name] = [
                    (v.tolist() if hasattr(v, "tolist") else v) for v in vals
                ]
        metadata["__tensor_count"] = reader.tensor_count
        metadata["__metadata_kv_count"] = reader.metadata_kv_count
        return metadata
    except Exception:
        pass

    # --- fallback: stdlib binary parser ---
    try:
        with open(path, "rb") as f:
            header = f.read(max_bytes)
    except OSError:
        return None

    if len(header) < 24 or header[:4] != _GGUF_MAGIC:
        return None

    offset = 4  # skip the 4-byte magic

    def _u32() -> int:
        nonlocal offset
        if offset + 4 > len(header):
            raise struct.error("truncated uint32")
        val = struct.unpack_from("<I", header, offset)[0]
        offset += 4
        return val

    def _u64() -> int:
        nonlocal offset
        if offset + 8 > len(header):
            raise struct.error("truncated uint64")
        val = struct.unpack_from("<Q", header, offset)[0]
        offset += 8
        return val

    def _read_string() -> Optional[str]:
        nonlocal offset
        if offset + 8 > len(header):
            return None
        slen = struct.unpack_from("<Q", header, offset)[0]
        offset += 8
        if offset + slen > len(header):
            return None
        s = header[offset:offset + slen].decode("utf-8", errors="replace")
        offset += slen
        return s

    def _read_value(vtype: int) -> Any:
        nonlocal offset
        if vtype == _GGUF_TYPE_UINT8:
            if offset + 1 > len(header):
                return None
            val = header[offset]; offset += 1
            return val
        if vtype == _GGUF_TYPE_INT8:
            if offset + 1 > len(header):
                return None
            val = struct.unpack_from("<b", header, offset)[0]; offset += 1
            return val
        if vtype == _GGUF_TYPE_BOOL:
            if offset + 1 > len(header):
                return None
            val = bool(header[offset]); offset += 1
            return val
        if vtype == _GGUF_TYPE_UINT16:
            if offset + 2 > len(header):
                return None
            val = struct.unpack_from("<H", header, offset)[0]; offset += 2
            return val
        if vtype == _GGUF_TYPE_INT16:
            if offset + 2 > len(header):
                return None
            val = struct.unpack_from("<h", header, offset)[0]; offset += 2
            return val
        if vtype == _GGUF_TYPE_UINT32:
            return _u32()
        if vtype == _GGUF_TYPE_INT32:
            if offset + 4 > len(header):
                return None
            val = struct.unpack_from("<i", header, offset)[0]; offset += 4
            return val
        if vtype == _GGUF_TYPE_FLOAT32:
            if offset + 4 > len(header):
                return None
            val = struct.unpack_from("<f", header, offset)[0]; offset += 4
            return val
        if vtype == _GGUF_TYPE_UINT64:
            return _u64()
        if vtype == _GGUF_TYPE_INT64:
            if offset + 8 > len(header):
                return None
            val = struct.unpack_from("<q", header, offset)[0]; offset += 8
            return val
        if vtype == _GGUF_TYPE_FLOAT64:
            if offset + 8 > len(header):
                return None
            val = struct.unpack_from("<d", header, offset)[0]; offset += 8
            return val
        if vtype == _GGUF_TYPE_STRING:
            return _read_string()
        if vtype == _GGUF_TYPE_ARRAY:
            if offset + 4 + 8 > len(header):
                return None
            atype = _u32()
            alen = _u64()
            arr: List[Any] = []
            for _ in range(alen):
                v = _read_value(atype)
                if v is None:
                    return arr
                arr.append(v)
            return arr
        return None

    def _skip_value(vtype: int) -> bool:
        """Best-effort skip without decoding. Returns False on truncation."""
        nonlocal offset
        if vtype in (_GGUF_TYPE_UINT8, _GGUF_TYPE_INT8, _GGUF_TYPE_BOOL):
            offset += 1
        elif vtype in (_GGUF_TYPE_UINT16, _GGUF_TYPE_INT16):
            offset += 2
        elif vtype in (_GGUF_TYPE_UINT32, _GGUF_TYPE_INT32, _GGUF_TYPE_FLOAT32):
            offset += 4
        elif vtype in (_GGUF_TYPE_UINT64, _GGUF_TYPE_INT64, _GGUF_TYPE_FLOAT64):
            offset += 8
        elif vtype == _GGUF_TYPE_STRING:
            if offset + 8 > len(header):
                return False
            slen = struct.unpack_from("<Q", header, offset)[0]
            offset += 8 + slen
        elif vtype == _GGUF_TYPE_ARRAY:
            if offset + 4 + 8 > len(header):
                return False
            atype = struct.unpack_from("<I", header, offset)[0]
            offset += 4
            alen = struct.unpack_from("<Q", header, offset)[0]
            offset += 8
            for _ in range(alen):
                if not _skip_value(atype):
                    return False
        else:
            return False
        return True

    try:
        version = _u32()
        tensor_count = _u64()
        metadata_kv_count = _u64()

        result: Dict[str, Any] = {
            "magic": "GGUF",
            "version": version,
            "__tensor_count": tensor_count,
            "__metadata_kv_count": metadata_kv_count,
        }

        for _ in range(metadata_kv_count):
            key = _read_string()
            if key is None:
                break
            if offset + 4 > len(header):
                break
            vtype = _u32()
            val = _read_value(vtype)
            if val is None:
                # Try to skip; if we can't, give up on the rest.
                if not _skip_value(vtype):
                    break
                continue
            result[key] = val

        return result
    except (struct.error, IndexError):
        return result  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def _format_quantization(meta: Dict[str, Any]) -> Optional[str]:
    """Extract a short quantization label like ``Q4_K_M`` from GGUF metadata."""
    if not meta:
        return None
    label = meta.get("general.file_type_label")
    if isinstance(label, str) and label:
        return label
    ftype = meta.get("general.file_type")
    mapping = {
        0: "F32", 1: "F16",
        2: "Q4_0", 3: "Q4_1",
        6: "Q5_0", 7: "Q5_1",
        8: "Q8_0",
        9: "Q2_K", 10: "Q3_K", 11: "Q4_K", 12: "Q5_K",
        13: "Q6_K", 14: "Q8_K",
        15: "Q2_K_S",
        16: "Q3_K_S", 17: "Q3_K_M", 18: "Q3_K_L",
        19: "Q4_K_S", 20: "Q4_K_M",
        21: "Q5_K_S", 22: "Q5_K_M",
    }
    try:
        return mapping.get(int(ftype))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _format_parameters(meta: Dict[str, Any]) -> Optional[str]:
    """Derive a short parameter count string (e.g. ``0.8B``) if available."""
    if not meta:
        return None
    for key in ("general.parameter_count", "llama.parameter_count", "general.size_label"):
        if key in meta:
            val = meta[key]
            if isinstance(val, (int, float)):
                n = float(val)
                if n >= 1e9:
                    return f"{n / 1e9:.1f}B"
                if n >= 1e6:
                    return f"{n / 1e6:.1f}M"
                if n >= 1e3:
                    return f"{n / 1e3:.1f}K"
                return str(int(n))
            if isinstance(val, str) and val:
                return val
    return None


def _resolve_model_or_die(model: str) -> str:
    """Resolve ``model`` to an absolute path. Exits with an error if missing.

    Order of resolution:
      1. Direct file path (``os.path.isfile``).
      2. ``~``-expanded path that exists on disk.
      3. Index lookup via :func:`resolve_model_path`.
    """
    if os.path.isfile(model):
        return os.path.abspath(model)

    expanded = os.path.expanduser(model)
    if os.path.isfile(expanded):
        return os.path.abspath(expanded)

    from_index = resolve_model_path(model)
    if from_index:
        return from_index

    click.echo(
        f"Error: Model '{model}' not found in index and is not a valid path.",
        err=True,
    )
    click.echo("Use 'ethllama list' to see indexed models.", err=True)
    sys.exit(1)


# ---------------------------------------------------------------------------
# rm — remove a model from the index (and optionally delete the file)
# ---------------------------------------------------------------------------

@click.command(name="rm")
@click.argument("model")
@click.option(
    "--purge", "-p",
    is_flag=True, default=False,
    help="Also delete the model file from disk (destructive).",
)
@click.option(
    "--yes", "-y",
    is_flag=True, default=False,
    help="Skip confirmation prompt when --purge is set.",
)
def rm_cmd(model: str, purge: bool, yes: bool) -> None:
    """Remove a model from the index (and optionally delete the file).

    \b
    By default, only the index entry is removed — the file on disk is
    left intact. Pass ``--purge`` to also delete the file. Because
    ``--purge`` is destructive, the command prompts for confirmation
    unless ``--yes`` is given.

    \b
    Examples:
      ethllama rm qwen3-0.8b-q4km.gguf
      ethllama rm /path/to/model.gguf --purge --yes
    """
    model_path = _resolve_model_or_die(model)
    name = os.path.basename(model_path)

    # 1. Remove from index
    removed = remove_from_index(model_path)
    if removed:
        click.echo(f"Removed {name} from index.")
    else:
        click.echo(f"Warning: {name} was not in the index (removing anyway).")

    # 2. Optionally delete the file from disk
    if purge:
        if not os.path.exists(model_path):
            click.echo(
                f"Warning: File {model_path} does not exist on disk.",
                err=True,
            )
            return

        if not yes:
            click.confirm(
                f"About to permanently delete {model_path}. Continue?",
                abort=True,
            )

        try:
            os.remove(model_path)
            click.echo(f"Deleted {model_path}.")
        except OSError as e:
            click.echo(
                f"Error: Could not delete {model_path}: {e}",
                err=True,
            )
            sys.exit(1)


# ---------------------------------------------------------------------------
# info — show model metadata
# ---------------------------------------------------------------------------

@click.command(name="info")
@click.argument("model")
@click.option(
    "--verbose", "-v",
    is_flag=True, default=False,
    help="Show every GGUF metadata KV pair (truncated).",
)
@click.option(
    "--json", "as_json",
    is_flag=True, default=False,
    help="Output as JSON instead of human-readable text.",
)
def info_cmd(model: str, verbose: bool, as_json: bool) -> None:
    """Show metadata for a model (size, type, GGUF header, etc.).

    \b
    MODEL can be a filename, an indexed identifier, or a direct path
    to a ``.gguf`` file. For non-GGUF files, basic filesystem metadata
    is still shown and a warning is emitted.

    \b
    Examples:
      ethllama info qwen3-0.8b-q4km.gguf
      ethllama info /path/to/model.gguf --json
      ethllama info model.gguf --verbose
    """
    # 1. Resolve the model
    entry = find_in_index(model)
    if entry:
        model_path = entry["path"]
    elif os.path.isfile(model) or os.path.isfile(os.path.expanduser(model)):
        model_path = os.path.abspath(os.path.expanduser(model))
        entry = {"path": model_path, "filename": os.path.basename(model_path)}
    else:
        click.echo(
            f"Error: Model '{model}' not found in index and is not a valid path.",
            err=True,
        )
        click.echo("Use 'ethllama list' to see indexed models.", err=True)
        sys.exit(1)

    # 2. File info
    try:
        stat = os.stat(model_path)
    except OSError as e:
        click.echo(f"Error: Could not stat file {model_path}: {e}", err=True)
        sys.exit(1)

    file_info: Dict[str, Any] = {
        "path": model_path,
        "filename": os.path.basename(model_path),
        "size_bytes": stat.st_size,
        "size_human": _human_size(stat.st_size),
        "modified": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(stat.st_mtime)),
        "indexed": entry.get("dir_path") is not None,
    }

    # 3. GGUF metadata
    gguf_meta: Optional[Dict[str, Any]] = _read_gguf_metadata(model_path)
    is_gguf = gguf_meta is not None

    # 4a. JSON output
    if as_json:
        output: Dict[str, Any] = {"file": file_info}
        if is_gguf:
            output["gguf"] = gguf_meta
        else:
            output["gguf"] = None
            output["warning"] = "Not a GGUF file or header could not be parsed"
        click.echo(json.dumps(output, indent=2, default=str))
        return

    # 4b. Human-readable output
    click.echo(f"Model: {file_info['filename']}")
    click.echo(f"  Path:     {file_info['path']}")
    click.echo(f"  Size:     {file_info['size_human']}")
    click.echo(f"  Modified: {file_info['modified']}")
    click.echo(f"  Indexed:  {'yes' if file_info['indexed'] else 'no'}")

    if not is_gguf:
        click.echo("")
        click.echo("  Warning: Not a GGUF file or header could not be parsed.")
        return

    if gguf_meta is None:
        return

    click.echo("")
    click.echo("  GGUF Metadata:")

    # Highlight a few well-known fields first.
    arch = gguf_meta.get("general.architecture")
    if arch:
        click.echo(f"    Architecture:      {arch}")
        ctx = gguf_meta.get(f"{arch}.context_length")
        if ctx is not None:
            click.echo(f"    Context length:    {ctx}")
        emb = gguf_meta.get(f"{arch}.embedding_length")
        if emb is not None:
            click.echo(f"    Embedding length:  {emb}")

    gguf_version = gguf_meta.get("version")
    if gguf_version is not None:
        click.echo(f"    GGUF version:      {gguf_version}")

    quant = _format_quantization(gguf_meta)
    if quant:
        click.echo(f"    Quantization:      {quant}")

    params = _format_parameters(gguf_meta)
    if params:
        click.echo(f"    Parameters:        {params}")

    if verbose:
        click.echo("")
        click.echo("  All metadata KV pairs:")
        for key in sorted(gguf_meta.keys()):
            if key == "version":
                continue
            val = gguf_meta[key]
            if isinstance(val, str) and len(val) > 200:
                val = val[:197] + "..."
            click.echo(f"    {key}: {val}")
    else:
        priority = {"version", "general.architecture"}
        if arch:
            priority.add(f"{arch}.context_length")
            priority.add(f"{arch}.embedding_length")
        priority.add("general.file_type")
        priority.add("general.file_type_label")

        remaining = {
            k: v for k, v in gguf_meta.items()
            if k not in priority and not k.startswith("__")
        }
        if remaining:
            click.echo("")
            click.echo("  Other metadata:")
            for key in sorted(remaining.keys()):
                val = remaining[key]
                if isinstance(val, str) and len(val) > 120:
                    val = val[:117] + "..."
                click.echo(f"    {key}: {val}")

    tensor_count = gguf_meta.get("__tensor_count")
    kv_count = gguf_meta.get("__metadata_kv_count")
    parts: List[str] = []
    if tensor_count is not None:
        parts.append(f"{tensor_count} tensors")
    if kv_count is not None:
        parts.append(f"{kv_count} metadata keys")
    if parts:
        click.echo("")
        click.echo(f"  ({', '.join(parts)})")


# ---------------------------------------------------------------------------
# Registration helper
# ---------------------------------------------------------------------------

def register_commands(cli: click.Group) -> None:
    """Attach the management commands to *cli*.

    The orchestrator calls this once after the main click group has
    been created. We register by name to make the call idempotent —
    re-registering simply overwrites the command on the group, which
    is the desired behavior during reloads.
    """
    cli.add_command(rm_cmd)
    cli.add_command(info_cmd)

"""Model pulling from Hugging Face Hub and Ollama registry."""

import os
import sys
import hashlib
import re
import click
from pathlib import Path
from typing import Optional

from .config import load_config
from .index import add_to_index

DEFAULT_MODEL_DIR = Path.home() / ".ethllama" / "models"


def pull_from_hf(model_id: str, revision: str = "main") -> str:
    """Download a GGUF model from Hugging Face Hub.

    Args:
        model_id: Model identifier in the format ``org/model`` or ``org/model:filename``.
        revision: Git revision (branch, tag, or commit hash) to download.

    Returns:
        Path to the downloaded model file.

    Raises:
        ImportError: If ``huggingface_hub`` is not installed.
        ValueError: If no ``.gguf`` files are found in the repository.
    """
    try:
        from huggingface_hub import hf_hub_download, HfApi
    except ImportError:
        raise ImportError(
            "huggingface_hub is required for pulling models from Hugging Face. "
            "Install it with: pip install huggingface_hub"
        )

    config = load_config()
    model_dirs = config.get("model_dirs", [str(DEFAULT_MODEL_DIR)])
    model_dir = Path(model_dirs[0])
    model_dir.mkdir(parents=True, exist_ok=True)

    # Parse model_id: "org/model" or "org/model:filename"
    if ":" in model_id:
        repo_id, filename = model_id.split(":", 1)
        click.echo(f"Downloading {repo_id}/{filename} (revision: {revision})...")
        local_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            revision=revision,
            local_dir=model_dir / repo_id.replace("/", "--"),
            local_dir_use_symlinks=False,
            resume=True,
        )
        add_to_index(local_path)
        return local_path
    else:
        repo_id = model_id
        api = HfApi()
        click.echo(f"Listing files in {repo_id}...")
        all_files = api.list_repo_files(repo_id)
        gguf_files = [f for f in all_files if f.endswith(".gguf")]
        if not gguf_files:
            raise ValueError(
                f"No .gguf files found in {repo_id}. "
                f"Specify a filename with 'org/model:filename.gguf'."
            )

        first_path = None
        for file in gguf_files:
            click.echo(f"Downloading {repo_id}/{file} (revision: {revision})...")
            local_path = hf_hub_download(
                repo_id=repo_id,
                filename=file,
                revision=revision,
                local_dir=model_dir / repo_id.replace("/", "--"),
                local_dir_use_symlinks=False,
                resume=True,
            )
            add_to_index(local_path)
            if first_path is None:
                first_path = local_path

        return first_path


def pull_from_ollama(model_name: str) -> str:
    """Pull a model from the Ollama registry via OCI Distribution Spec v2.

    Args:
        model_name: The Ollama model name (e.g. ``llama3.2``, ``mistral:7b``,
            ``foo/bar:tag``, or a full URL with host prefix).

    Returns:
        Path to the downloaded GGUF model file.

    Raises:
        ValueError: If the manifest has no model layer, or SHA-256 mismatch.
        requests.RequestException: On network errors.
    """
    import requests

    scheme, host, namespace, repo, tag = _parse_model_ref(model_name)

    if host is None:
        host = "registry.ollama.ai"
    if scheme is None:
        scheme = "https"

    base_url = f"{scheme}://{host}"
    manifest_url = f"{base_url}/v2/{namespace}/{repo}/manifests/{tag}"

    click.echo(f"Fetching manifest for {namespace}/{repo}:{tag}...")

    # Fetch manifest (OCI Distribution Spec v2)
    headers = {
        "Accept": "application/vnd.docker.distribution.manifest.v2+json",
    }
    resp = requests.get(manifest_url, headers=headers, timeout=30)
    resp.raise_for_status()

    # Known quirk: Content-Type may be text/plain but body is always JSON
    manifest = resp.json()

    layers = manifest.get("layers", [])
    model_layer = None
    for layer in layers:
        if layer.get("mediaType") == "application/vnd.ollama.image.model":
            model_layer = layer
            break

    if model_layer is None:
        raise ValueError(
            f"No model layer found in manifest for {namespace}/{repo}:{tag}. "
            f"Available layers: {[l.get('mediaType') for l in layers]}"
        )

    digest = model_layer["digest"]
    size = model_layer.get("size", 0)

    # Parse digest (sha256:<64-hex>)
    m = re.match(r"^sha256:([a-f0-9]{64})$", digest)
    if not m:
        raise ValueError(f"Invalid digest format: {digest}")
    expected_hex = m.group(1)

    blob_url = f"{base_url}/v2/{namespace}/{repo}/blobs/{digest}"

    # Prepare output paths
    model_dir = _safe_dirname(namespace, repo, tag)
    model_dir.mkdir(parents=True, exist_ok=True)

    output_filename = f"{repo}-{tag}.gguf"
    output_path = model_dir / output_filename
    partial_path = model_dir / f"{output_filename}.partial"

    # Check if already fully downloaded and valid
    if output_path.exists():
        click.echo(f"Verifying existing file: {output_path}")
        sha256 = hashlib.sha256()
        with open(output_path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                sha256.update(chunk)
        if sha256.hexdigest() == expected_hex:
            click.echo(f"Model already exists and is valid: {output_path}")
            add_to_index(str(output_path))
            return str(output_path)
        else:
            click.echo("Existing file hash mismatch, re-downloading...")
            output_path.unlink(missing_ok=True)

    # Resume from partial download
    resume_bytes = 0
    if partial_path.exists():
        resume_bytes = partial_path.stat().st_size
        click.echo(f"Resuming download from {resume_bytes} bytes...")

    # Download with optional Range header for resume
    dl_headers = {}
    if resume_bytes > 0:
        dl_headers["Range"] = f"bytes={resume_bytes}-"

    click.echo(f"Downloading model...")
    resp = requests.get(blob_url, headers=dl_headers, stream=True, timeout=30)
    resp.raise_for_status()

    # Handle range request response
    if resume_bytes > 0:
        if resp.status_code == 206:
            # Partial content -- resume download
            mode = "ab"
            sha256 = hashlib.sha256()
            with open(partial_path, "rb") as f:
                for chunk in iter(lambda: f.read(65536), b""):
                    sha256.update(chunk)
        else:
            # Range not supported -- start fresh
            click.echo("Resume not supported, starting fresh download...")
            mode = "wb"
            resume_bytes = 0
            sha256 = hashlib.sha256()
    else:
        mode = "wb"
        sha256 = hashlib.sha256()

    with open(partial_path, mode) as f:
        with click.progressbar(
            length=size - resume_bytes,
            label=f"  {output_filename}",
            show_eta=True,
            show_pos=True,
        ) as bar:
            for chunk in resp.iter_content(chunk_size=65536):
                if chunk:
                    f.write(chunk)
                    sha256.update(chunk)
                    bar.update(len(chunk))

    # Verify SHA-256
    actual_hex = sha256.hexdigest()
    if actual_hex != expected_hex:
        partial_path.unlink(missing_ok=True)
        raise ValueError(
            f"SHA-256 mismatch: expected {expected_hex}, got {actual_hex}"
        )

    # Rename partial to final
    partial_path.rename(output_path)

    click.echo(f"Model saved to: {output_path}")
    add_to_index(str(output_path))
    return str(output_path)


def _parse_model_ref(name: str) -> tuple:
    """Parse a model reference into (scheme, host, namespace, repo, tag).

    Examples:
        'llama3.2:3b' -> (None, None, 'library', 'llama3.2', '3b')
        'foo/bar:tag' -> (None, None, 'foo', 'bar', 'tag')
        'llama3.2' -> (None, None, 'library', 'llama3.2', 'latest')
        'https://registry.ollama.ai/library/llama3.2:3b' -> ('https', 'registry.ollama.ai', 'library', 'llama3.2', '3b')
    """
    scheme = None
    host = None
    rest = name

    # Strip optional scheme://host prefix
    if "://" in name:
        scheme, rest = name.split("://", 1)
        if "/" in rest:
            host, rest = rest.split("/", 1)
        else:
            host = rest
            rest = ""
    else:
        # Heuristic: if first component before '/' looks like a hostname
        # (contains a dot and no colons before the tag), treat it as host.
        first_slash = rest.find("/")
        if first_slash > 0:
            candidate = rest[:first_slash]
            if "." in candidate and ":" not in candidate:
                host = candidate
                rest = rest[first_slash + 1:]

    # Parse tag (default: latest)
    tag = "latest"
    if ":" in rest:
        rest, tag = rest.rsplit(":", 1)

    # Parse namespace/repo
    if "/" in rest:
        namespace, repo = rest.split("/", 1)
    else:
        namespace = "library"
        repo = rest

    return scheme, host, namespace, repo, tag


def _safe_dirname(namespace: str, repo: str, tag: str) -> Path:
    """Create a filesystem-safe directory path for a model."""
    safe = f"{namespace}__{repo}"
    return DEFAULT_MODEL_DIR / safe / tag


def pull_model(model_id: str, source: str = "hf", revision: str = "main") -> str:
    """Pull a model from any supported source and add it to the index.

    Args:
        model_id: Model identifier (depends on source).
        source: Source registry (``hf`` for Hugging Face, ``ollama`` for Ollama).
        revision: Revision/branch for Hugging Face models.

    Returns:
        Path to the downloaded model file.

    Raises:
        ValueError: If the source is unknown.
        ImportError: If required dependencies are missing.
        NotImplementedError: If the source is not yet implemented.
    """
    if source in ("hf", "huggingface"):
        return pull_from_hf(model_id, revision)
    elif source == "ollama":
        return pull_from_ollama(model_id)
    else:
        raise ValueError(
            f"Unknown source: '{source}'. Supported sources: hf, ollama"
        )

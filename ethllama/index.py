import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Any

INDEX_FILE = Path.home() / ".ethllama" / "index.json"

def load_index() -> Dict[str, List[Dict[str, str]]]:
    if not INDEX_FILE.exists():
        return {}
    with open(INDEX_FILE, "r") as f:
        return json.load(f)

def save_index(index: Dict[str, List[Dict[str, str]]]) -> None:
    INDEX_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(INDEX_FILE, "w") as f:
        json.dump(index, f, indent=2)

def add_to_index(model_path: str) -> None:
    index = load_index()
    model_dir = os.path.dirname(model_path)
    if model_dir not in index:
        index[model_dir] = []
    index[model_dir].append({
        "filename": os.path.basename(model_path),
        "path": model_path,
        "size": os.path.getsize(model_path),
        "modified": os.path.getmtime(model_path),
    })
    save_index(index)

def resolve_model_path(model_identifier: str) -> str:
    """Resolve a model identifier (e.g., 'qwen3.6:27b') to a path."""
    index = load_index()
    for dir_path, models in index.items():
        for model in models:
            if model["filename"] == model_identifier or model["path"].endswith(model_identifier):
                return model["path"]
    return ""

def remove_from_index(model_path: str) -> bool:
    """Remove a model from the index by its path. Returns True if removed, False if not found."""
    index = load_index()
    removed = False
    for dir_path in list(index.keys()):
        original_len = len(index[dir_path])
        index[dir_path] = [
            entry for entry in index[dir_path]
            if entry["path"] != model_path and entry["filename"] != os.path.basename(model_path)
        ]
        if len(index[dir_path]) < original_len:
            removed = True
        # Clean up empty directories
        if not index[dir_path]:
            del index[dir_path]
    if removed:
        save_index(index)
    return removed

def find_in_index(model_identifier: str) -> Optional[Dict[str, Any]]:
    """Find a model entry in the index. Returns the entry dict with dir_path added, or None."""
    index = load_index()
    for dir_path, models in index.items():
        for model in models:
            if (model["filename"] == model_identifier
                    or model["path"] == model_identifier
                    or model["path"].endswith(model_identifier)):
                entry = dict(model)
                entry["dir_path"] = dir_path
                return entry
    return None


def list_all_indexed() -> List[Dict[str, Any]]:
    """Return a flat list of all indexed models with their full metadata.

    Each entry contains the original index fields (``filename``, ``path``,
    ``size``, ``modified``) plus a synthetic ``dir_path`` field pointing to
    the parent directory in the index. The list is sorted by
    ``dir_path`` then ``filename`` for stable output.
    """
    index = load_index()
    flat: List[Dict[str, Any]] = []
    for dir_path, models in index.items():
        for model in models:
            entry: Dict[str, Any] = dict(model)
            entry["dir_path"] = dir_path
            flat.append(entry)
    flat.sort(key=lambda e: (e.get("dir_path", ""), e.get("filename", "")))
    return flat

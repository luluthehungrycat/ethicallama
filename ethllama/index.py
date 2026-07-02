import os
import json
from pathlib import Path
from typing import Dict, List

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

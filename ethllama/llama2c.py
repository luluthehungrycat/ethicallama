"""Minimal llama2.c (.bin) model helpers.

The llama2.c format is a raw PyTorch checkpoint exported by
`export.py` in the upstream repo. It is NOT GGUF — do not pass
these files to the standard inference path.
"""

from pathlib import Path
from typing import Optional


SUPPORTED_LLAMA2C_FORMATS = (".bin",)


def is_llama2c_model(path: str) -> bool:
    """Heuristic check for a llama2.c model file."""
    if not path.lower().endswith(SUPPORTED_LLAMA2C_FORMATS):
        return False
    # llama2.c files start with a fixed-size header (config dict)
    # but we keep this simple — just rely on extension.
    return True


def find_tokenizer_for(model_path: str) -> Optional[str]:
    """Look for a matching tokenizer.bin next to the model file."""
    parent = Path(model_path).parent
    candidates = [
        parent / "tokenizer.bin",
        parent / "tokenizer.model",  # sentencepiece fallback
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    return None

import os
import tempfile
from pathlib import Path
import pytest

from ethllama.llama2c import is_llama2c_model, find_tokenizer_for


def test_is_llama2c_model_recognises_bin_extension():
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        path = f.name
    try:
        assert is_llama2c_model(path) is True
    finally:
        os.unlink(path)


def test_is_llama2c_model_rejects_gguf(tmp_path):
    gguf = tmp_path / "model.gguf"
    gguf.touch()
    assert is_llama2c_model(str(gguf)) is False


def test_find_tokenizer_for_returns_tokenizer_bin(tmp_path):
    model = tmp_path / "model.bin"
    model.touch()
    tok = tmp_path / "tokenizer.bin"
    tok.touch()
    assert find_tokenizer_for(str(model)) == str(tok)


def test_find_tokenizer_for_returns_none_when_missing(tmp_path):
    model = tmp_path / "model.bin"
    model.touch()
    assert find_tokenizer_for(str(model)) is None

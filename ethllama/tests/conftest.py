"""Shared pytest fixtures for the FastAPI integration tests.

The fixtures here isolate the tests from the user's real ``~/.ethllama/`` directory
and from any real inference engine. The real ``run_inference`` / ``get_embeddings``
functions would shell out to ``llama-cli`` / ``llama-embedding``; we replace them
with deterministic stubs so the FastAPI routes can be exercised in-process via
``TestClient``.
"""

from __future__ import annotations

import os
import struct
from pathlib import Path
from typing import Any, Dict, Iterator, List, Tuple

import pytest
from fastapi.testclient import TestClient

from ethllama import api, config, index, inference


# ---------------------------------------------------------------------------
# Filesystem isolation
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_ethllama_home(tmp_path, monkeypatch) -> Path:
    """Redirect ``~/.ethllama/`` to a fresh tmp directory.

    The user's real ``index.json`` and ``config.yaml`` are never touched. The
    fixture also patches the module-level ``INDEX_FILE`` / ``CONFIG_FILE`` paths
    (they were bound at import time using ``Path.home()``).
    """
    ethllama_dir = tmp_path / ".ethllama"
    ethllama_dir.mkdir(parents=True, exist_ok=True)

    # Make any code calling Path.home() see the tmp location.
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    # Re-bind the module-level paths so load_index / load_config / save_config
    # all read & write inside the tmp dir.
    monkeypatch.setattr(index, "INDEX_FILE", ethllama_dir / "index.json")
    monkeypatch.setattr(config, "CONFIG_DIR", ethllama_dir)
    monkeypatch.setattr(config, "CONFIG_FILE", ethllama_dir / "config.yaml")

    return ethllama_dir


# ---------------------------------------------------------------------------
# Mocked inference
# ---------------------------------------------------------------------------

# A short, deterministic, multi-word string so streaming tests can verify
# that the FastAPI layer splits the response into chunks.
MOCK_INFERENCE_TEXT = "Hello world this is mocked"


@pytest.fixture
def mock_inference(monkeypatch) -> Dict[str, Any]:
    """Patch ``run_inference`` and ``run_inference_stream`` to deterministic stubs.

    Returns a dict with the canned text (useful for assertions) and the fake
    stream function (useful for verifying call signatures).
    """
    def fake_run_inference(*args, **kwargs) -> str:
        return MOCK_INFERENCE_TEXT

    def fake_run_inference_stream(*args, **kwargs) -> Iterator[str]:
        for word in MOCK_INFERENCE_TEXT.split():
            yield word + " "

    # The api module imported these names at import time, so the call sites
    # look up ``api.run_inference`` (not ``inference.run_inference``). Patch
    # both locations for safety.
    monkeypatch.setattr(inference, "run_inference", fake_run_inference)
    monkeypatch.setattr(api, "run_inference", fake_run_inference)
    monkeypatch.setattr(inference, "run_inference_stream", fake_run_inference_stream)
    # The api module does not import run_inference_stream, but patch defensively
    # in case a future change wires it up. raising=False makes this a no-op today.
    monkeypatch.setattr(api, "run_inference_stream", fake_run_inference_stream, raising=False)

    return {
        "text": MOCK_INFERENCE_TEXT,
        "stream": fake_run_inference_stream,
        "run_inference": fake_run_inference,
    }


@pytest.fixture
def mock_embeddings(monkeypatch) -> Any:
    """Patch ``get_embeddings`` to return a deterministic 768-dim vector per input."""
    embedding_dim = 768

    def fake_get_embeddings(model_path: str, texts: List[str], **kwargs) -> List[List[float]]:
        # Deterministic: vector i has all values equal to 0.01 * (i + 1).
        return [[0.01 * (i + 1)] * embedding_dim for i, _ in enumerate(texts)]

    monkeypatch.setattr(inference, "get_embeddings", fake_get_embeddings)
    monkeypatch.setattr(api, "get_embeddings", fake_get_embeddings)

    return fake_get_embeddings


# ---------------------------------------------------------------------------
# Indexed model
# ---------------------------------------------------------------------------

@pytest.fixture
def indexed_model(tmp_ethllama_home) -> Tuple[str, str]:
    """Create a fake GGUF file in the tmp dir and add it to the index.

    Returns ``(filename, absolute_path)``.
    """
    model_dir = tmp_ethllama_home / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / "fake-model.gguf"
    # Write a minimal GGUF-shaped blob (magic + version + zero counts + padding).
    with open(model_path, "wb") as f:
        f.write(b"GGUF")                              # magic
        f.write(struct.pack("<I", 3))                 # version
        f.write(struct.pack("<Q", 0))                 # tensor_count
        f.write(struct.pack("<Q", 0))                 # metadata_kv_count
        f.write(b"\x00" * 1024)                       # padding

    index.add_to_index(str(model_path))

    return model_path.name, str(model_path)


# ---------------------------------------------------------------------------
# TestClient fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def test_client(tmp_ethllama_home, mock_inference, mock_embeddings) -> TestClient:
    """A FastAPI ``TestClient`` with mocked inference and no API key auth.

    The underlying ``index.json`` is empty unless the test also requests
    ``indexed_model`` — keep the two fixtures independent so ``GET /v1/models``
    can be tested in both empty and populated states.
    """
    return TestClient(api.app)


@pytest.fixture
def auth_test_client(
    tmp_ethllama_home, mock_inference, mock_embeddings
) -> Tuple[TestClient, str]:
    """A FastAPI ``TestClient`` with API key auth enabled.

    Returns ``(client, api_key)``. Requests must include
    ``Authorization: Bearer <api_key>`` to reach the route handlers.
    """
    api_key = "test-secret-key-12345"

    cfg = config.DEFAULT_CONFIG.copy()
    cfg["api"] = {**config.DEFAULT_CONFIG["api"], "api_key": api_key}
    config.save_config(cfg)

    return TestClient(api.app), api_key

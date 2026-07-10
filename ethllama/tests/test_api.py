"""Integration tests for the FastAPI HTTP server (``ethllama.api``).

These tests use ``fastapi.testclient.TestClient`` (an in-process, httpx-based
client) to exercise every public route. Inference and embeddings are mocked
deterministically so no real ``llama-cli`` / ``llama-embedding`` binary is
required, and a tmp config + index directory is used so the user's real
``~/.ethllama/`` is never touched.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from ethllama import api


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

def test_health_endpoint(test_client):
    """``GET /health`` returns 200 with ``{"status": "ok"}``."""
    response = test_client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

def test_models_endpoint(test_client, indexed_model):
    """``GET /v1/models`` returns the single indexed model in OpenAI shape."""
    filename, _path = indexed_model

    response = test_client.get("/v1/models")

    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "list"
    assert len(body["data"]) == 1

    model = body["data"][0]
    assert model["id"] == filename
    assert model["object"] == "model"
    assert model["owned_by"] == "user"


def test_models_endpoint_empty(test_client):
    """``GET /v1/models`` with an empty index returns an empty list."""
    response = test_client.get("/v1/models")
    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "list"
    assert body["data"] == []


# ---------------------------------------------------------------------------
# Chat completions
# ---------------------------------------------------------------------------

def test_chat_completion_basic(test_client, indexed_model):
    """Single-turn chat returns the OpenAI shape: id, object, model, choices, usage."""
    filename, _ = indexed_model

    response = test_client.post(
        "/v1/chat/completions",
        json={
            "model": filename,
            "messages": [{"role": "user", "content": "Hello there"}],
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["model"] == filename
    assert body["object"] == "chat.completion"
    assert body["id"].startswith("cmpl-")
    assert isinstance(body["created"], int)

    assert len(body["choices"]) == 1
    choice = body["choices"][0]
    assert choice["index"] == 0
    assert choice["finish_reason"] == "stop"
    assert choice["message"]["role"] == "assistant"
    assert choice["message"]["content"]  # non-empty (from the mock)

    # usage stats
    usage = body["usage"]
    assert "prompt_tokens" in usage
    assert "completion_tokens" in usage
    assert "total_tokens" in usage
    assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]


def test_chat_completion_streaming(test_client, indexed_model):
    """``stream: true`` returns Server-Sent Events with delta chunks + [DONE]."""
    filename, _ = indexed_model

    with test_client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": filename,
            "messages": [{"role": "user", "content": "Stream please"}],
            "stream": True,
        },
    ) as response:
        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]
        lines = list(response.iter_lines())

    # Pull out the SSE data lines (skip blank separators).
    data_lines = [line for line in lines if line.startswith("data: ")]
    done_lines = [line for line in data_lines if line == "data: [DONE]"]
    chunk_lines = [line for line in data_lines if line != "data: [DONE]"]

    # One terminal [DONE] marker + at least one content chunk.
    assert len(done_lines) == 1
    assert len(chunk_lines) >= 1

    # Every non-DONE chunk must be a chat.completion.chunk with a delta.
    for raw in chunk_lines:
        chunk = json.loads(raw[len("data: "):])
        assert chunk["object"] == "chat.completion.chunk"
        assert chunk["model"] == filename
        assert "choices" in chunk
        assert "delta" in chunk["choices"][0]

    # The last content chunk must signal finish_reason="stop".
    final = json.loads(chunk_lines[-1][len("data: "):])
    assert final["choices"][0]["finish_reason"] == "stop"


def test_chat_completion_multi_turn(test_client, indexed_model, monkeypatch):
    """Multi-turn chat passes every message through ``format_chat_messages``."""
    filename, _ = indexed_model

    # Spy on the api module's local binding of format_chat_messages so we
    # can inspect exactly which messages reached the prompt formatter.
    captured: list[dict] = []
    original = api.format_chat_messages

    def spy(messages):
        # Copy to avoid downstream mutation of the recorded list.
        captured.extend(dict(m) for m in messages)
        return original(messages)

    monkeypatch.setattr(api, "format_chat_messages", spy)

    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello!"},
        {"role": "user", "content": "How are you?"},
    ]
    response = test_client.post(
        "/v1/chat/completions",
        json={"model": filename, "messages": messages},
    )
    assert response.status_code == 200

    # The spy must have been called exactly once with all 4 messages, in order.
    assert len(captured) == 4
    assert [m["role"] for m in captured] == ["system", "user", "assistant", "user"]
    assert captured[0]["content"] == "You are helpful."
    assert captured[3]["content"] == "How are you?"


# ---------------------------------------------------------------------------
# Text completions
# ---------------------------------------------------------------------------

def test_completion_basic(test_client, indexed_model):
    """``POST /v1/completions`` returns a text completion in OpenAI shape."""
    filename, _ = indexed_model

    response = test_client.post(
        "/v1/completions",
        json={"model": filename, "prompt": "Once upon a time"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["model"] == filename
    assert body["object"] == "text_completion"
    assert len(body["choices"]) == 1
    choice = body["choices"][0]
    assert choice["index"] == 0
    assert choice["finish_reason"] == "stop"
    assert choice["text"]  # non-empty (from the mock)

    assert "usage" in body
    assert "prompt_tokens" in body["usage"]
    assert "completion_tokens" in body["usage"]


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------

def test_embeddings_single(test_client, indexed_model):
    """Single-string input returns exactly one 768-dim embedding vector."""
    filename, _ = indexed_model

    response = test_client.post(
        "/v1/embeddings",
        json={"model": filename, "input": "Hello world"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["model"] == filename
    assert body["object"] == "list"
    assert len(body["data"]) == 1
    item = body["data"][0]
    assert item["object"] == "embedding"
    assert item["index"] == 0
    assert len(item["embedding"]) == 768
    assert "usage" in body


def test_embeddings_batch(test_client, indexed_model):
    """List input returns N vectors, one per input, with sequential indices."""
    filename, _ = indexed_model
    inputs = ["first text", "second text", "third text"]

    response = test_client.post(
        "/v1/embeddings",
        json={"model": filename, "input": inputs},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["model"] == filename
    assert len(body["data"]) == len(inputs)
    for idx, item in enumerate(body["data"]):
        assert item["index"] == idx
        assert len(item["embedding"]) == 768


# ---------------------------------------------------------------------------
# API key auth
# ---------------------------------------------------------------------------

def test_api_key_auth_required(auth_test_client, indexed_model):
    """With ``api_key`` set, requests without ``Authorization`` get 401."""
    client, _key = auth_test_client
    filename, _ = indexed_model

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": filename,
            "messages": [{"role": "user", "content": "hi"}],
        },
    )

    assert response.status_code == 401
    body = response.json()
    assert "Invalid API key" in body["detail"]
    # Standard Bearer-challenge header per RFC 6750.
    assert response.headers.get("www-authenticate", "").lower() == "bearer"


def test_api_key_auth_valid(auth_test_client, indexed_model):
    """With ``api_key`` set, requests with the right Bearer token succeed."""
    client, key = auth_test_client
    filename, _ = indexed_model

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": filename,
            "messages": [{"role": "user", "content": "hi"}],
        },
        headers={"Authorization": f"Bearer {key}"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["model"] == filename
    assert body["choices"][0]["message"]["role"] == "assistant"


# ---------------------------------------------------------------------------
# Invalid request
# ---------------------------------------------------------------------------

def test_invalid_json_request(test_client):
    """Malformed JSON body returns 422 (Pydantic / FastAPI validation)."""
    response = test_client.post(
        "/v1/chat/completions",
        content=b"{not valid json",
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 422



# ---------------------------------------------------------------------------
# Preloaded model (serve --model)
# ---------------------------------------------------------------------------


def test_run_server_accepts_model_path_kwarg():
    """``run_server`` declares ``model_path`` as a keyword argument."""
    import inspect
    sig = inspect.signature(api.run_server)
    assert "model_path" in sig.parameters
    # Default value should be None so existing call sites keep working
    assert sig.parameters["model_path"].default is None


def test_preloaded_model_appears_in_models_list(tmp_path, tmp_ethllama_home, monkeypatch):
    """``GET /v1/models`` includes a preloaded model even when it isn't indexed."""
    # Create a real file on disk so stat() works
    preloaded_path = tmp_path / "preloaded.gguf"
    preloaded_path.write_bytes(b"GGUF" + b"\x00" * 64)

    # No model is in the index, but one is preloaded via app.state
    monkeypatch.setattr(api.app.state, "preloaded_model", str(preloaded_path), raising=False)

    client = TestClient(api.app)
    response = client.get("/v1/models")
    assert response.status_code == 200
    body = response.json()
    ids = [m["id"] for m in body["data"]]
    assert "preloaded" in ids
    preloaded_entry = next(m for m in body["data"] if m["id"] == "preloaded")
    assert preloaded_entry["object"] == "model"
    assert preloaded_entry["root"] == str(preloaded_path)
    # It should be inserted at the front (most prominent)
    assert body["data"][0]["id"] == "preloaded"


def test_preloaded_model_not_duplicated_when_already_indexed(indexed_model, tmp_ethllama_home, monkeypatch):
    """If the preloaded model is also in the index, ``/v1/models`` does not duplicate it."""
    filename, path = indexed_model
    monkeypatch.setattr(api.app.state, "preloaded_model", path, raising=False)

    client = TestClient(api.app)
    response = client.get("/v1/models")
    assert response.status_code == 200
    body = response.json()
    ids = [m["id"] for m in body["data"]]
    assert ids.count(filename) == 1


def test_chat_completion_uses_preloaded_model_by_basename(tmp_path, tmp_ethllama_home,
                                                          mock_inference, mock_embeddings,
                                                          monkeypatch):
    """A chat request referencing the preloaded model by stem resolves to the preloaded path."""
    preloaded_path = tmp_path / "my-preloaded-model.gguf"
    preloaded_path.write_bytes(b"GGUF" + b"\x00" * 64)
    monkeypatch.setattr(api.app.state, "preloaded_model", str(preloaded_path), raising=False)

    client = TestClient(api.app)
    # Reference the preloaded model by its stem
    response = client.post(
        "/v1/chat/completions",
        json={"model": "my-preloaded-model", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["model"] == "my-preloaded-model"


def test_chat_completion_uses_preloaded_model_by_full_path(tmp_path, tmp_ethllama_home,
                                                           mock_inference, mock_embeddings,
                                                           monkeypatch):
    """A chat request referencing the preloaded model by absolute path works."""
    preloaded_path = tmp_path / "another-preloaded.gguf"
    preloaded_path.write_bytes(b"GGUF" + b"\x00" * 64)
    monkeypatch.setattr(api.app.state, "preloaded_model", str(preloaded_path), raising=False)

    client = TestClient(api.app)
    response = client.post(
        "/v1/chat/completions",
        json={"model": str(preloaded_path), "messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 200


def test_chat_completion_unknown_model_returns_404(test_client, indexed_model):
    """Without a preloaded model, an unknown model name still returns 404."""
    response = test_client.post(
        "/v1/chat/completions",
        json={"model": "definitely-not-indexed", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


def test_no_preloaded_model_means_state_attr_is_none(test_client, monkeypatch):
    """Without ``serve --model``, ``app.state.preloaded_model`` is unset/None."""
    # Reset state so this test is order-independent
    monkeypatch.setattr(api.app.state, "preloaded_model", None, raising=False)
    preloaded = getattr(api.app.state, "preloaded_model", None)
    assert preloaded is None



# ---------------------------------------------------------------------------
# TTL (idle model unloading)
# ---------------------------------------------------------------------------

def test_run_server_accepts_idle_timeout_kwarg():
    """``run_server`` declares ``idle_timeout`` as a keyword argument with default 0.

    A default of 0 means the TTL feature is opt-in and existing call
    sites that do not pass ``idle_timeout`` keep working unchanged.
    """
    import inspect
    sig = inspect.signature(api.run_server)
    assert "idle_timeout" in sig.parameters
    assert sig.parameters["idle_timeout"].default == 0


async def test_idle_unloader_task_created_when_idle_timeout_positive(monkeypatch):
    """Startup hook spawns a background task when ``idle_timeout > 0``."""
    # Ensure clean state — the conftest does not reset these per-test.
    monkeypatch.setattr(api.app.state, "idle_timeout", 60, raising=False)
    monkeypatch.setattr(api.app.state, "preloaded_model", "/tmp/fake.gguf", raising=False)
    monkeypatch.setattr(api.app.state, "last_request_time", None, raising=False)
    monkeypatch.setattr(api.app.state, "_idle_unloader_task", None, raising=False)

    # Invoke the startup hook directly (no need to spin up a real server).
    await api._start_idle_unloader()

    # A task must have been created and stored on app.state.
    task = getattr(api.app.state, "_idle_unloader_task", None)
    assert task is not None
    # It must be a real asyncio task (not just any object).
    import asyncio as _asyncio
    assert isinstance(task, _asyncio.Task)

    # Clean up so we do not leak a 30-second sleep into other tests.
    task.cancel()
    try:
        await task
    except _asyncio.CancelledError:
        pass


def test_last_request_time_updated_on_chat_completion(
    test_client, indexed_model, monkeypatch
):
    """``POST /v1/chat/completions`` bumps ``app.state.last_request_time``."""
    import time as _time
    filename, _ = indexed_model

    # Start from a known sentinel well in the past.
    monkeypatch.setattr(api.app.state, "last_request_time", None, raising=False)

    before = _time.monotonic()
    response = test_client.post(
        "/v1/chat/completions",
        json={
            "model": filename,
            "messages": [{"role": "user", "content": "ping"}],
        },
    )
    after = _time.monotonic()

    assert response.status_code == 200

    # The handler must have touched last_request_time, and the new
    # value must fall inside the wall-clock window of the request.
    last = getattr(api.app.state, "last_request_time", None)
    assert last is not None, "handler did not update last_request_time"
    assert before - 1.0 <= last <= after + 1.0, (
        f"last_request_time={last} outside [{before}, {after}]"
    )


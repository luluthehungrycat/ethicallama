"""OpenAI-compatible HTTP API (opt-in) using FastAPI."""

import os
import time
import json
import asyncio
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from .config import load_config
from .index import load_index, resolve_model_path
from .inference import run_inference, get_embeddings, get_gpu_config, format_chat_messages

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    top_k: Optional[int] = 40
    max_tokens: Optional[int] = 2048
    stream: Optional[bool] = False
    stop: Optional[List[str]] = None


class CompletionRequest(BaseModel):
    model: str
    prompt: str
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    top_k: Optional[int] = 40
    max_tokens: Optional[int] = 2048
    stream: Optional[bool] = False
    stop: Optional[List[str]] = None


class EmbeddingRequest(BaseModel):
    model: str
    input: str | List[str]
    encoding_format: Optional[str] = "float"


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="ethicallama API",
    version="0.1.0",
    description="OpenAI-compatible local LLM inference API",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Auth dependency
# ---------------------------------------------------------------------------

def verify_api_key(request: Request) -> bool:
    """Dependency to verify API key if configured."""
    config = load_config()
    api_key = config.get("api", {}).get("api_key", "")
    if api_key:
        auth = request.headers.get("Authorization", "")
        if not auth.startswith("Bearer ") or auth[7:] != api_key:
            raise HTTPException(
                status_code=401,
                detail="Invalid API key",
                headers={"WWW-Authenticate": "Bearer"},
            )
    return True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _simulate_chat_response(messages: List[ChatMessage], temperature: float = 0.7) -> str:
    """Placeholder chat response until Rust core is connected."""
    last_msg = messages[-1].content if messages else ""
    return (
        f"This is a simulated response from ethicallama.\n\n"
        f"Your message was: {last_msg}\n\n"
        f"The native inference engine is not yet connected. "
        f"Once ethllama-core is built, real llama.cpp inference will run here."
    )


def _simulate_embedding(text: str) -> List[float]:
    """Simulate embedding generation — placeholder until Rust core connects."""
    # Generate a deterministic pseudo-embedding based on text hash
    # Real implementation will use llama_get_embeddings() from the Rust core
    h = hash(text)
    import random
    rng = random.Random(h)
    return [rng.gauss(0, 0.1) for _ in range(384)]  # 384-dim embedding


def _simulate_completion(prompt: str, temperature: float = 0.7) -> str:
    """Placeholder text completion until Rust core is connected."""
    return (
        f"Simulated completion for: {prompt[:80]}{'…' if len(prompt) > 80 else ''}\n\n"
        f"This is a placeholder response. Connect the Rust core for real inference."
    )


def _build_model_list() -> List[Dict[str, Any]]:
    """Build the OpenAI-compatible model list from the local index."""
    index = load_index()
    models = []
    for dir_path, model_list in index.items():
        for m in model_list:
            models.append({
                "id": m["filename"],
                "object": "model",
                "created": int(m.get("modified", 0)),
                "owned_by": "user",
                "permissions": [],
                "root": m.get("path", ""),
                "parent": None,
            })
    return models


def _build_usage(prompt_tokens: int = 0, completion_tokens: int = 0) -> Dict[str, int]:
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }


def _model_path_and_gpu(model_name: str):
    """Resolve model path and return (path, gpu_config).

    If a model was pre-loaded via ``serve --model``, requests that reference
    that model by filename, stem, or absolute path will resolve to the
    preloaded path even if the model is not present in the local index.
    """
    from .inference import get_gpu_config

    preloaded = getattr(app.state, "preloaded_model", None)
    if preloaded and model_name:
        preloaded_path = Path(preloaded)
        preloaded_name = preloaded_path.name
        preloaded_stem = preloaded_path.stem
        # Match by basename, stem, or absolute path
        if model_name == preloaded_name or model_name == preloaded_stem:
            return preloaded, get_gpu_config()
        try:
            if os.path.abspath(model_name) == preloaded:
                return preloaded, get_gpu_config()
        except OSError:
            pass

    path = resolve_model_path(model_name)
    if not path:
        # Fallback: try as a direct filesystem path
        if model_name and os.path.exists(model_name):
            return os.path.abspath(model_name), get_gpu_config()
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model_name}' not found in index",
        )
    return path, get_gpu_config()


# ---------------------------------------------------------------------------
# Idle unloader (TTL)
# ---------------------------------------------------------------------------

async def _idle_unloader(app, idle_timeout: int):
    """Background task that unloads the model after ``idle_timeout`` seconds of inactivity.

    Runs forever (until the event loop is torn down at server shutdown). Every
    30 seconds it inspects ``app.state.last_request_time``; if the elapsed
    idle period has exceeded ``idle_timeout`` and a pre-loaded model is
    still resident, the model reference is cleared so the next request falls
    through to the normal model resolution path (and the model is reloaded
    lazily).

    Setting ``idle_timeout <= 0`` disables the task entirely.
    """
    if idle_timeout <= 0:
        return  # disabled

    while True:
        await asyncio.sleep(30)  # check every 30 seconds
        last_use = getattr(app.state, 'last_request_time', None)
        if last_use is None:
            continue

        elapsed = time.monotonic() - last_use
        if elapsed >= idle_timeout and app.state.preloaded_model is not None:
            logger.info(
                "Model unloaded after %.0fs of inactivity (timeout=%ds)",
                elapsed, idle_timeout,
            )
            app.state.preloaded_model = None
            app.state.last_request_time = None


@app.on_event('startup')
async def _start_idle_unloader():
    """Start the background idle-unloader task when ``idle_timeout`` is set.

    The task is stored on ``app.state._idle_unloader_task`` so tests and
    shutdown handlers can reference / cancel it.  When ``idle_timeout <= 0``
    no task is created and the TTL feature is effectively disabled.
    """
    idle = getattr(app.state, 'idle_timeout', 0)
    if idle > 0:
        logger.info("Idle timeout set to %ds", idle)
        app.state._idle_unloader_task = asyncio.create_task(
            _idle_unloader(app, idle)
        )


# ---------------------------------------------------------------------------
# Streaming helpers
# ---------------------------------------------------------------------------

async def _stream_chat_completion(request: ChatCompletionRequest):
    """Stream chat completion response as Server-Sent Events."""
    from .inference import format_chat_messages
    # Touch the idle-clock so a long-running stream is not unloaded
    # mid-flight by the background TTL unloader.
    app.state.last_request_time = time.monotonic()
    model_path, gpu = _model_path_and_gpu(request.model)
    response_text = await asyncio.to_thread(
        run_inference,
        model_path=model_path,
        prompt=format_chat_messages([m.model_dump() for m in request.messages]),
        temperature=request.temperature or 0.7,
        top_p=request.top_p or 0.9,
        top_k=request.top_k or 40,
        max_tokens=request.max_tokens or 2048,
        n_gpu_layers=gpu["n_gpu_layers"],
        n_threads=gpu["n_threads"],
        stop=request.stop,
    )
    words = response_text.split()
    for i, word in enumerate(words):
        chunk = {
            "id": f"cmpl-{int(time.time())}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": request.model,
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": word + (" " if i < len(words) - 1 else "")},
                    "finish_reason": None,
                }
            ],
        }
        yield f"data: {json.dumps(chunk)}\n\n"
        await asyncio.sleep(0.02)
    final_chunk = {
        "id": f"cmpl-{int(time.time())}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": request.model,
        "choices": [
            {
                "index": 0,
                "delta": {},
                "finish_reason": "stop",
            }
        ],
    }
    yield f"data: {json.dumps(final_chunk)}\n\n"
    yield "data: [DONE]\n\n"


async def _stream_completion(request: CompletionRequest):
    """Stream text completion as Server-Sent Events."""
    # Touch the idle-clock so a long-running stream is not unloaded
    # mid-flight by the background TTL unloader.
    app.state.last_request_time = time.monotonic()
    model_path, gpu = _model_path_and_gpu(request.model)
    response_text = await asyncio.to_thread(
        run_inference,
        model_path=model_path,
        prompt=request.prompt,
        temperature=request.temperature or 0.7,
        top_p=request.top_p or 0.9,
        top_k=request.top_k or 40,
        max_tokens=request.max_tokens or 2048,
        n_gpu_layers=gpu["n_gpu_layers"],
        n_threads=gpu["n_threads"],
        stop=request.stop,
    )
    words = response_text.split()
    for i, word in enumerate(words):
        chunk = {
            "id": f"cmpl-{int(time.time())}",
            "object": "text_completion.chunk",
            "created": int(time.time()),
            "model": request.model,
            "choices": [
                {
                    "index": 0,
                    "text": word + (" " if i < len(words) - 1 else ""),
                    "finish_reason": None,
                }
            ],
        }
        yield f"data: {json.dumps(chunk)}\n\n"
        await asyncio.sleep(0.02)
    final_chunk = {
        "id": f"cmpl-{int(time.time())}",
        "object": "text_completion.chunk",
        "created": int(time.time()),
        "model": request.model,
        "choices": [
            {
                "index": 0,
                "text": "",
                "finish_reason": "stop",
            }
        ],
    }
    yield f"data: {json.dumps(final_chunk)}\n\n"
    yield "data: [DONE]\n\n"


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/v1/models")
async def list_models(auth: bool = Depends(verify_api_key)):
    """List available models from the local index (OpenAI-compatible).

    A model that was pre-loaded via ``serve --model`` is included in the
    response even if it is not present in the local index, so clients can
    discover the active default model.
    """
    models = _build_model_list()

    # Add the preloaded model if it isn't already in the list
    preloaded = getattr(app.state, "preloaded_model", None)
    if preloaded:
        preloaded_path = Path(preloaded)
        preloaded_id = preloaded_path.stem
        if not any(m.get("id") == preloaded_id for m in models):
            try:
                stat = preloaded_path.stat()
                created = int(stat.st_mtime)
            except OSError:
                created = 0
            models.insert(0, {
                "id": preloaded_id,
                "object": "model",
                "created": created,
                "owned_by": "user",
                "permissions": [],
                "root": str(preloaded_path),
                "parent": None,
            })

    return {
        "object": "list",
        "data": models,
    }


@app.post("/v1/embeddings")
async def embeddings(
    request: EmbeddingRequest,
    auth: bool = Depends(verify_api_key),
):
    """OpenAI-compatible embeddings endpoint."""
    app.state.last_request_time = time.monotonic()
    model_path, gpu = _model_path_and_gpu(request.model)
    inputs = request.input if isinstance(request.input, list) else [request.input]
    # Run in executor
    embeddings = await asyncio.to_thread(
        get_embeddings,
        model_path=model_path,
        texts=inputs,
        n_gpu_layers=gpu["n_gpu_layers"],
        n_threads=gpu["n_threads"],
    )
    total_tokens = sum(max(len(t) // 4, 1) for t in inputs)
    data = [
        {"object": "embedding", "index": idx, "embedding": emb}
        for idx, emb in enumerate(embeddings)
    ]
    return {
        "object": "list",
        "data": data,
        "model": request.model,
        "usage": {"prompt_tokens": total_tokens, "total_tokens": total_tokens},
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok", "version": "0.1.0"}


@app.post("/v1/chat/completions")
async def chat_completions(
    request: ChatCompletionRequest,
    auth: bool = Depends(verify_api_key),
):
    """OpenAI-compatible chat completions endpoint."""
    app.state.last_request_time = time.monotonic()
    if request.stream:
        return StreamingResponse(
            _stream_chat_completion(request),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    model_path, gpu = _model_path_and_gpu(request.model)
    response_text = await asyncio.to_thread(
        run_inference,
        model_path=model_path,
        prompt=format_chat_messages([m.model_dump() for m in request.messages]),
        temperature=request.temperature or 0.7,
        top_p=request.top_p or 0.9,
        top_k=request.top_k or 40,
        max_tokens=request.max_tokens or 2048,
        n_gpu_layers=gpu["n_gpu_layers"],
        n_threads=gpu["n_threads"],
        stop=request.stop,
    )
    return {
        "id": f"cmpl-{int(time.time())}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": request.model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response_text,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": _build_usage(
            prompt_tokens=sum(len(m.content) for m in request.messages) // 4,
            completion_tokens=len(response_text) // 4,
        ),
    }


@app.post("/v1/completions")
async def completions(
    request: CompletionRequest,
    auth: bool = Depends(verify_api_key),
):
    """OpenAI-compatible legacy completions endpoint."""
    app.state.last_request_time = time.monotonic()
    if request.stream:
        return StreamingResponse(
            _stream_completion(request),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    model_path, gpu = _model_path_and_gpu(request.model)
    response_text = await asyncio.to_thread(
        run_inference,
        model_path=model_path,
        prompt=request.prompt,
        temperature=request.temperature or 0.7,
        top_p=request.top_p or 0.9,
        top_k=request.top_k or 40,
        max_tokens=request.max_tokens or 2048,
        n_gpu_layers=gpu["n_gpu_layers"],
        n_threads=gpu["n_threads"],
        stop=request.stop,
    )
    return {
        "id": f"cmpl-{int(time.time())}",
        "object": "text_completion",
        "created": int(time.time()),
        "model": request.model,
        "choices": [
            {
                "index": 0,
                "text": response_text,
                "finish_reason": "stop",
            }
        ],
        "usage": _build_usage(
            prompt_tokens=len(request.prompt) // 4,
            completion_tokens=len(response_text) // 4,
        ),
    }


# ---------------------------------------------------------------------------
# Factory / runner
# ---------------------------------------------------------------------------

def create_app() -> FastAPI:
    """Factory function for creating the FastAPI app (useful for testing)."""
    return app


def run_server(host: str = "127.0.0.1", port: int = 8080, api_key: str = "",
               model_path: Optional[str] = None, idle_timeout: int = 0):
    """Run the API server with uvicorn.

    Parameters
    ----------
    host, port, api_key
        Standard uvicorn / API auth configuration.
    model_path
        Absolute path to a GGUF model that should be advertised as
        pre-loaded by the server. Stored on ``app.state.preloaded_model``
        and surfaced in ``GET /v1/models`` and used as a fallback by
        :func:`_model_path_and_gpu`.
    idle_timeout
        If > 0, the background :func:`_idle_unloader` task auto-unloads
        the pre-loaded model after this many seconds of inactivity.  If
        0 (the default) the TTL feature is disabled and the model
        stays resident for the lifetime of the server.
    """
    import uvicorn

    if api_key:
        from .config import save_config
        config = load_config()
        config.setdefault("api", {})["api_key"] = api_key
        save_config(config)

    # Stash the preloaded model path on the FastAPI app state so route
    # handlers can read it from any thread.  Also initialise the idle
    # clock and the idle_timeout setting; the latter is read by the
    # startup hook to decide whether to spawn the background unloader.
    app.state.preloaded_model = model_path if model_path else None
    app.state.last_request_time = None
    app.state.idle_timeout = int(idle_timeout or 0)

    if app.state.idle_timeout > 0:
        logger.info("Idle timeout set to %ds", app.state.idle_timeout)

    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info",
    )

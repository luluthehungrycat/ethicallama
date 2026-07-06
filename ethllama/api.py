"""OpenAI-compatible HTTP API (opt-in) using FastAPI."""

import time
import json
import asyncio
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from .config import load_config
from .index import load_index, resolve_model_path
from .inference import run_inference, get_embeddings, get_gpu_config, format_chat_messages

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
    """Resolve model path and return (path, gpu_config)."""
    from .inference import get_gpu_config
    path = resolve_model_path(model_name)
    if not path:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found in index")
    return path, get_gpu_config()


# ---------------------------------------------------------------------------
# Streaming helpers
# ---------------------------------------------------------------------------

async def _stream_chat_completion(request: ChatCompletionRequest):
    """Stream chat completion response as Server-Sent Events."""
    from .inference import format_chat_messages
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
    """List available models from the local index (OpenAI-compatible)."""
    models = _build_model_list()
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


def run_server(host: str = "127.0.0.1", port: int = 8080, api_key: str = ""):
    """Run the API server with uvicorn."""
    import uvicorn

    if api_key:
        from .config import save_config
        config = load_config()
        config.setdefault("api", {})["api_key"] = api_key
        save_config(config)

    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info",
    )

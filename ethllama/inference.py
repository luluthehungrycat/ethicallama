"""Inference engine abstraction for ethicallama.

Provides binary discovery and subprocess-based inference via built
llama.cpp tools (llama-cli, llama-embedding). The Rust/PyO3 core is
available as a fast-path (try_import_py_model) when the upstream
tokenizer bug is resolved for the given model.
"""

import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Union

# Where our built binaries live (relative to this file's location)
BUILD_BIN_DIR = (Path(__file__).resolve().parent.parent / "llama.cpp-build" / "bin")

# ---------------------------------------------------------------------------
# Runtime GPU configuration (shared between serve command and API routes)
# ---------------------------------------------------------------------------

_GPU_CONFIG: Dict[str, Any] = {
    "n_gpu_layers": 0,
    "gpu_backend": "auto",
    "n_threads": -1,
    "ctx_size": 0,
}


def set_gpu_config(
    n_gpu_layers: int = 0,
    gpu_backend: str = "auto",
    n_threads: int = -1,
    ctx_size: int = 0,
) -> None:
    """Set the runtime GPU configuration used by serve-mode API routes."""
    _GPU_CONFIG["n_gpu_layers"] = n_gpu_layers
    _GPU_CONFIG["gpu_backend"] = gpu_backend
    _GPU_CONFIG["n_threads"] = n_threads
    _GPU_CONFIG["ctx_size"] = ctx_size


def get_gpu_config() -> Dict[str, Any]:
    """Return a copy of the current GPU configuration dict."""
    return dict(_GPU_CONFIG)


# ---------------------------------------------------------------------------
# Binary discovery
# ---------------------------------------------------------------------------

def find_binary(name: str = "llama-cli") -> Optional[str]:
    """Locate a llama.cpp tool binary.

    Checks, in order:
    1. Built from submodule at ``llama.cpp-build/bin/<name>``
    2. System PATH via ``shutil.which``

    Returns an absolute path, or *None* if not found.
    """
    built = BUILD_BIN_DIR / name
    if built.exists():
        return str(built.resolve())
    return shutil.which(name)


def require_binary(name: str = "llama-cli") -> str:
    """Like :func:`find_binary` but raises :class:`RuntimeError` on failure."""
    path = find_binary(name)
    if path is None:
        raise RuntimeError(
            f"{name} not found. Build the submodule: "
            f"cmake --build llama.cpp-build --target {name}"
        )
    return path


# ---------------------------------------------------------------------------
# Engine capability detection
# ---------------------------------------------------------------------------

def has_inference_engine() -> bool:
    """Return *True* if a real inference engine is available."""
    # Check for llama-cli binary
    if find_binary("llama-cli"):
        return True
    # Could also check for maturin-built Rust core
    return _try_import_py_model() is not None


def _try_import_py_model() -> Any:
    """Try to import the Rust PyLlamaModel core (returns None if unavailable)."""
    try:
        from ethllama_core import PyLlamaModel  # type: ignore[import-untyped]
        return PyLlamaModel
    except (ImportError, AttributeError):
        return None


# ---------------------------------------------------------------------------
# Common argument builders
# ---------------------------------------------------------------------------

def _build_base_args(
    model_path: str,
    n_gpu_layers: int = 0,
    n_threads: int = -1,
    ctx_size: int = 0,
) -> List[str]:
    """Build common CLI arguments shared by llama.cpp tools."""
    args: List[str] = [
        "-m", model_path,
    ]
    if n_gpu_layers:
        args.extend(["-ngl", str(n_gpu_layers)])
    if n_threads > 0:
        args.extend(["-t", str(n_threads)])
    if ctx_size > 0:
        args.extend(["-c", str(ctx_size)])
    return args


# ---------------------------------------------------------------------------
# Text generation via llama-cli
# ---------------------------------------------------------------------------

def _build_cli_args(
    binary: str,
    model_path: str,
    prompt: str,
    temperature: float = 0.0,
    top_p: float = 0.9,
    top_k: int = 40,
    max_tokens: int = 2048,
    n_gpu_layers: int = 0,
    n_threads: int = -1,
    ctx_size: int = 0,
    stop: Optional[List[str]] = None,
) -> List[str]:
    """Build llama-cli command-line arguments for text generation."""
    args: List[str] = [
        binary,
        "--single-turn",
        "--no-display-prompt",
        "--no-show-timings",
        * _build_base_args(model_path, n_gpu_layers, n_threads, ctx_size),
        "-p", prompt,
        "-n", str(max_tokens),
        "--temp", str(temperature),
        "--top-k", str(top_k),
        "--top-p", str(top_p),
    ]
    if stop:
        for s in stop:
            args.extend(["-r", s])
    return args


def _strip_cli_output(raw: str, prompt: str) -> str:
    """Strip banner, prompt echo and trailing messages from llama-cli stdout.

    ``llama-cli`` outputs a banner, echoes the ``> <prompt>`` line, and
    appends an ``Exiting...`` message.  This helper extracts just the
    generated text.
    """
    # Find the prompt echo line: "> <prompt_text>\n"
    prompt_marker = f"> {prompt}"
    idx = raw.find(prompt_marker)
    if idx >= 0:
        raw = raw[idx + len(prompt_marker):]

    # Strip trailing "Exiting..." and any trailing whitespace
    if "Exiting..." in raw:
        raw = raw[: raw.index("Exiting...")]
    return raw.strip()


def _strip_spinner(text: str) -> str:
    """Remove loading-spinner and other terminal control artifacts."""
    # Strip backspace and other non-printing control characters (keep \n, \t, \r)
    result = []
    for ch in text:
        if ch == '\n' or ch == '\t' or ch == '\r' or (ch >= ' ' and ch <= '~') or ord(ch) > 127:
            result.append(ch)
    text = ''.join(result)
    # Remove leading spinner characters (orphaned | - \ / from loading indicator)
    text = re.sub(r'^[|/\\\-]+\s*', '', text)
    return text


def run_inference(
    model_path: str,
    prompt: str,
    temperature: float = 0.0,
    top_p: float = 0.9,
    top_k: int = 40,
    max_tokens: int = 2048,
    n_gpu_layers: int = 0,
    n_threads: int = -1,
    ctx_size: int = 0,
    stop: Optional[List[str]] = None,
) -> str:
    """Run text generation with ``llama-cli`` and return the output.

    Args:
        model_path: Path to the GGUF model file.
        prompt: Input text to feed the model.
        temperature: Sampling temperature (0 = greedy, 1 = creative).
        top_p: Nucleus sampling threshold.
        top_k: Top-k sampling limit.
        max_tokens: Maximum tokens to generate.
        n_gpu_layers: Layers to offload to GPU (0 = CPU only).
        n_threads: CPU thread count (-1 = auto).
        ctx_size: Context window size in tokens (0 = model default).
        stop: Optional list of stop strings.

    Returns:
        The generated text (response portion only).

    Raises:
        RuntimeError: If the binary is not found or the subprocess fails.
        subprocess.TimeoutExpired: If generation takes longer than 5 minutes.
    """
    binary = require_binary("llama-cli")

    args = _build_cli_args(
        binary, model_path, prompt,
        temperature=temperature, top_p=top_p, top_k=top_k,
        max_tokens=max_tokens,
        n_gpu_layers=n_gpu_layers, n_threads=n_threads, ctx_size=ctx_size,
        stop=stop,
    )

    result = subprocess.run(
        args, capture_output=True, text=True, timeout=300,
    )
    if result.returncode != 0:
        _raise_cli_error("llama-cli", result)

    return _strip_spinner(_strip_cli_output(result.stdout, prompt))


def run_inference_stream(
    model_path: str,
    prompt: str,
    temperature: float = 0.0,
    top_p: float = 0.9,
    top_k: int = 40,
    max_tokens: int = 2048,
    n_gpu_layers: int = 0,
    n_threads: int = -1,
    ctx_size: int = 0,
    stop: Optional[List[str]] = None,
) -> Iterator[str]:
    """Run text generation with streaming output.

    Yields chunks of generated text as they arrive from ``llama-cli``.
    Accepts the same arguments as :func:`run_inference`.
    """
    binary = require_binary("llama-cli")

    args = _build_cli_args(
        binary, model_path, prompt,
        temperature=temperature, top_p=top_p, top_k=top_k,
        max_tokens=max_tokens,
        n_gpu_layers=n_gpu_layers, n_threads=n_threads, ctx_size=ctx_size,
        stop=stop,
    )

    with subprocess.Popen(
        args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    ) as proc:
        assert proc.stdout is not None
        assert proc.stderr is not None

        # Skip lines until we find the prompt echo
        prompt_marker = f"> {prompt}"
        for line in iter(proc.stdout.readline, ""):
            if prompt_marker in line:
                break

        # Now yield the actual response lines, stopping at "Exiting..."
        for line in iter(proc.stdout.readline, ""):
            if "Exiting..." in line:
                break
            if line:
                yield _strip_spinner(line)

        proc.wait()
        if proc.returncode != 0:
            stderr = proc.stderr.read()
            raise RuntimeError(
                f"llama-cli stream failed (exit {proc.returncode}): "
                f"{stderr.strip()}"
            )


# ---------------------------------------------------------------------------
# Embeddings via llama-embedding
# ---------------------------------------------------------------------------

def _parse_embedding_output(raw: str) -> List[List[float]]:
    """Parse the raw stdout from ``llama-embedding`` into embedding vectors.

    Handles ``--embd-output-format json`` (OpenAI-style) and line-delimited
    arrays.
    """
    raw = raw.strip()
    if not raw:
        raise RuntimeError("Empty response from llama-embedding")

    # Try full JSON parse first (OpenAI-style or array)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        pass
    else:
        if isinstance(data, dict) and "data" in data:
            return [item["embedding"] for item in data["data"]]
        if isinstance(data, list) and data and isinstance(data[0], list):
            return data
        if isinstance(data, list) and data and isinstance(data[0], float):
            return [data]  # single vector
        raise RuntimeError(f"Unexpected embedding JSON format: {type(data)}")

    # Fallback: line-delimited JSON arrays
    embeddings: List[List[float]] = []
    for line in raw.splitlines():
        line = line.strip()
        if line:
            try:
                vec = json.loads(line)
                if isinstance(vec, list):
                    embeddings.append(vec)
            except json.JSONDecodeError:
                pass
    if embeddings:
        return embeddings

    raise RuntimeError(
        f"Cannot parse embedding output (first 200 chars): {raw[:200]}"
    )


def get_embeddings(
    model_path: str,
    texts: List[str],
    n_gpu_layers: int = 0,
    n_threads: int = -1,
    pooling: str = "mean",
    embd_normalize: int = 2,
) -> List[List[float]]:
    """Compute text embeddings with ``llama-embedding``.

    Args:
        model_path: Path to the GGUF model file.
        texts: One or more input strings to embed.
        n_gpu_layers: Layers to offload to GPU.
        n_threads: CPU thread count (-1 = auto).
        pooling: Pooling strategy (``none``, ``mean``, ``cls``, ``last``,
            ``rank``).
        embd_normalize: Normalization mode (-1 = none, 2 = euclidean, etc.).

    Returns:
        List of embedding vectors (one per input text).

    Raises:
        RuntimeError: If the binary is unavailable or the subprocess fails.
    """
    binary = require_binary("llama-embedding")

    args: List[str] = [
        binary,
        * _build_base_args(model_path, n_gpu_layers, n_threads),
        "--pooling", pooling,
        "--embd-output-format", "json",
        "--embd-normalize", str(embd_normalize),
    ]
    # For multiple texts, join with newline separator (natural document boundary)
    args.extend(["-p", "\n".join(texts)])

    result = subprocess.run(
        args, capture_output=True, text=True, timeout=300,
    )
    if result.returncode != 0:
        _raise_cli_error("llama-embedding", result)

    return _parse_embedding_output(result.stdout)


# ---------------------------------------------------------------------------
# Rust core fast-path (available when upstream tokenizer bug is resolved)
# ---------------------------------------------------------------------------

class RustInferenceEngine:
    """Thin wrapper that tries to use the fast Rust/PyO3 core.

    Falls back to the subprocess-based engine automatically.
    """

    def __init__(self) -> None:
        self._py_class = _try_import_py_model()

    def available(self) -> bool:
        return self._py_class is not None

    def infer(
        self,
        model_path: str,
        prompt: str,
        temperature: float = 0.0,
        n_gpu_layers: int = 0,
        n_ctx: int = 4096,
        n_threads: int = 4,
    ) -> str:
        if not self._py_class:
            raise RuntimeError("Rust core not available (not built or import failed)")

        model = self._py_class(
            path=model_path,
            n_gpu_layers=n_gpu_layers,
            n_ctx=n_ctx,
            n_threads=n_threads,
        )
        try:
            return model.infer(prompt, temperature=temperature)
        finally:
            # PyLlamaModel.Drop handles cleanup
            pass


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def format_chat_messages(messages: list) -> str:
    """Format a list of {role, content} dicts into a prompt string.

    Uses a simple role-header format compatible with most instruct-tuned
    models.  For production use, replace with the model's
    ``tokenizer.chat_template`` from GGUF metadata.
    """
    parts: list[str] = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "system":
            parts.append(f"<|im_start|>system\n{content}<|im_end|>")
        elif role == "user":
            parts.append(f"<|im_start|>user\n{content}<|im_end|>")
        elif role == "assistant":
            parts.append(f"<|im_start|>assistant\n{content}<|im_end|>")
        else:
            parts.append(content)
    parts.append("<|im_start|>assistant\n")
    return "\n".join(parts)


def _raise_cli_error(name: str, result: subprocess.CompletedProcess) -> None:
    """Raise a descriptive :class:`RuntimeError` for a failed subprocess."""
    stderr = result.stderr.strip()
    msg = f"{name} failed (exit {result.returncode})"
    if stderr:
        msg += f": {stderr}"
    raise RuntimeError(msg)


# ---------------------------------------------------------------------------
# REPL session — multi-turn conversation state manager
# ---------------------------------------------------------------------------

def _simulated_stream(prompt: str) -> Iterator[str]:
    """Fallback streaming iterator when no real inference engine is available.

    Used by :class:`REPLSession` so the REPL remains usable in
    development environments where llama-cli is not yet built.
    """
    snippet = prompt.strip().splitlines()[-1] if prompt.strip() else ""
    if len(snippet) > 120:
        snippet = snippet[:117] + "..."
    response = (
        "[Simulated response — no inference engine found]\n\n"
        f"Last input: {snippet}"
    )
    for word in response.split():
        yield word + " "


class REPLSession:
    """Stateful multi-turn conversation with a local model.

    The session owns the conversation history and the per-turn inference
    parameters.  The CLI (or any driver) feeds user messages via
    :meth:`send` and receives streamed response chunks.  The session
    appends each completed turn to its history and truncates to the
    configured window.

    Example::

        session = REPLSession("model.gguf", initial_system="You are helpful.")
        for chunk in session.send("Hello!"):
            print(chunk, end="")

    The class deliberately avoids any I/O (no stdin/stdout access) so it
    remains trivially testable.
    """

    DEFAULT_MAX_HISTORY = 10

    def __init__(
        self,
        model_path: str,
        *,
        initial_system: Optional[str] = None,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 40,
        max_tokens: int = 2048,
        n_gpu_layers: int = 0,
        n_threads: int = -1,
        ctx_size: int = 0,
        max_history: int = DEFAULT_MAX_HISTORY,
    ) -> None:
        self.model_path = model_path
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.max_tokens = max_tokens
        self.n_gpu_layers = n_gpu_layers
        self.n_threads = n_threads
        self.ctx_size = ctx_size
        self.max_history = max(0, int(max_history))
        self.history: List[Dict[str, str]] = []
        if initial_system:
            self.history.append({"role": "system", "content": initial_system})

    # -- history management --------------------------------------------------

    def clear_history(self) -> None:
        """Drop all user/assistant turns but preserve the system prompt."""
        if self.history and self.history[0].get("role") == "system":
            system = self.history[0]
            self.history = [system]
        else:
            self.history = []

    def set_system(self, text: str) -> None:
        """Replace (or insert) the system prompt at the head of history."""
        if self.history and self.history[0].get("role") == "system":
            self.history[0]["content"] = text
        else:
            self.history.insert(0, {"role": "system", "content": text})

    def set_temperature(self, value: float) -> None:
        """Update the sampling temperature for subsequent turns."""
        self.temperature = float(value)

    def get_history_snapshot(self) -> List[Dict[str, str]]:
        """Return a copy of the current history list."""
        return [dict(entry) for entry in self.history]

    # -- core send ------------------------------------------------------------

    def send(self, user_message: str) -> Iterator[str]:
        """Send a user message and stream the assistant response.

        Yields text chunks as they arrive.  After the stream completes
        the (user, assistant) turn pair is appended to the history and
        the history is truncated to the configured window.
        """
        if not user_message:
            return

        # Build the chat message list: history + new user turn
        messages: List[Dict[str, str]] = list(self.history)
        messages.append({"role": "user", "content": user_message})
        prompt = format_chat_messages(messages)

        # Choose stream source: real engine or simulation
        if has_inference_engine():
            stream_iter: Iterator[str] = run_inference_stream(
                model_path=self.model_path,
                prompt=prompt,
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
                max_tokens=self.max_tokens,
                n_gpu_layers=self.n_gpu_layers,
                n_threads=self.n_threads,
                ctx_size=self.ctx_size,
            )
        else:
            stream_iter = _simulated_stream(prompt)

        # Accumulate the full response and stream chunks to caller
        full_response_parts: List[str] = []
        for chunk in stream_iter:
            full_response_parts.append(chunk)
            yield chunk

        full_response = "".join(full_response_parts)

        # Commit the turn to history and truncate
        self.history.append({"role": "user", "content": user_message})
        self.history.append({"role": "assistant", "content": full_response})
        self._truncate_history()

    # -- internal helpers ----------------------------------------------------

    def _truncate_history(self) -> None:
        """Keep at most ``max_history`` user/assistant turn pairs.

        The system prompt (if any) at index 0 is always preserved.
        """
        if self.max_history <= 0:
            # History disabled: drop everything (keep nothing for privacy)
            self.history = []
            return

        if not self.history:
            return

        has_system = self.history[0].get("role") == "system"
        if has_system:
            system_entry = self.history[0]
            turns = self.history[1:]
        else:
            system_entry = None
            turns = self.history

        max_turn_entries = self.max_history * 2  # user + assistant per turn
        if len(turns) > max_turn_entries:
            turns = turns[-max_turn_entries:]

        self.history = ([system_entry] if system_entry else []) + turns

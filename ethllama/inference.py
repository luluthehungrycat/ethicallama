"""Inference engine abstraction for ethicallama.

Provides binary discovery and subprocess-based inference via built
llama.cpp tools (llama-cli, llama-embedding). The Rust/PyO3 core is
available as a fast-path (try_import_py_model) when the upstream
tokenizer bug is resolved for the given model.
"""

import json
import os
import re
import struct
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

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
# Runtime binary configuration (paths to llama.cpp tool binaries)
# ---------------------------------------------------------------------------

_BINARY_CONFIG: Dict[str, Any] = {
    "binary_dir": None,       # directory containing llama-cli, llama-embedding, etc.
    "llama_cli": None,        # explicit path to llama-cli binary
    "llama_embedding": None,  # explicit path to llama-embedding
    "llama_quantize": None,   # explicit path to llama-quantize
}


def set_binary_config(
    binary_dir: Optional[str] = None,
    llama_cli: Optional[str] = None,
    llama_embedding: Optional[str] = None,
    llama_quantize: Optional[str] = None,
) -> None:
    """Set runtime binary overrides."""
    if binary_dir is not None:
        _BINARY_CONFIG["binary_dir"] = binary_dir
    if llama_cli is not None:
        _BINARY_CONFIG["llama_cli"] = llama_cli
    if llama_embedding is not None:
        _BINARY_CONFIG["llama_embedding"] = llama_embedding
    if llama_quantize is not None:
        _BINARY_CONFIG["llama_quantize"] = llama_quantize


def get_binary_config() -> Dict[str, Any]:
    """Return a copy of the current binary configuration."""
    return dict(_BINARY_CONFIG)


# ---------------------------------------------------------------------------
# Binary discovery
# ---------------------------------------------------------------------------

def find_binary(name: str = "llama-cli") -> Optional[str]:
    """Locate a llama.cpp tool binary.

    Checks, in order:
    1. Runtime override via ``set_binary_config()`` (for the specific binary name)
    2. ``binary_dir`` from runtime overrides (``<binary_dir>/<name>``)
    3. Config file ``engines.binary_dir`` (via load_config)
    4. Built from submodule at ``llama.cpp-build/bin/<name>``
    5. System PATH via ``shutil.which``

    Returns an absolute path, or *None* if not found.
    """
    # Map generic name to config key
    name_to_key = {
        "llama-cli": "llama_cli",
        "llama-embedding": "llama_embedding",
        "llama-quantize": "llama_quantize",
    }
    key = name_to_key.get(name)

    # 1. Runtime override for this specific binary
    if key and _BINARY_CONFIG.get(key):
        path = _BINARY_CONFIG[key]
        if path and os.path.exists(path):
            return os.path.abspath(path)

    # 2. Runtime binary_dir override
    binary_dir = _BINARY_CONFIG.get("binary_dir")
    if binary_dir:
        candidate = Path(binary_dir) / name
        if candidate.exists():
            return str(candidate.resolve())

    # 3. Config file binary_dir (fallback — load_config is lazy)
    from .config import load_config
    config = load_config()
    engines_cfg = config.get("engines", {})
    if not binary_dir:
        # Only check config if runtime didn't set it
        cfg_binary_dir = engines_cfg.get("binary_dir")
        if cfg_binary_dir:
            candidate = Path(cfg_binary_dir) / name
            if candidate.exists():
                return str(candidate.resolve())
    # Per-binary paths in config
    if key and engines_cfg.get(key):
        path = engines_cfg[key]
        if path and os.path.exists(path):
            return os.path.abspath(path)

    # 4. Built from submodule
    built = BUILD_BIN_DIR / name
    if built.exists():
        return str(built.resolve())

    # 5. System PATH
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
    """Try to import the Rust PyLlamaModel core (returns None if unavailable).

    The Rust extension may be installed as ``ethllama.ethllama_core``
    (wheel / ``python-source``) or as a top-level ``ethllama_core``
    (``maturin develop`` / editable install).
    """
    for import_path in ("ethllama_core", "ethllama.ethllama_core"):
        try:
            mod = __import__(import_path, fromlist=["PyLlamaModel"])
            return getattr(mod, "PyLlamaModel", None)
        except (ImportError, AttributeError):
            continue
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


# Lines in llama.cpp's startup banner that should be filtered out.  These
# appear on stdout between the binary's launch and the actual response.
_BANNER_MARKERS = (
    "llama_model_loader",
    "llm_load_tensors",
    "llama_model_load",
    "llama_print_system_info",
    "print_info",
    "system_info",
    "system_build",
    "llm_load_print_meta",
    "load_tensors",
    "model type",
    "model size",
    "model layers",
    "model params",
    "general.architecture",
    "general.name",
    "print_info:",
)


def _is_banner_line(line: str) -> bool:
    """Return True if a line looks like a llama.cpp startup banner line."""
    return any(marker in line for marker in _BANNER_MARKERS)


# Lines that should be dropped when they appear in the leading banner
# block (after the strict banner lines but before the response).  These
# are session metadata such as sampling / generation parameters that
# aren't part of the model output.
_BANNER_DROP_PREFIXES = (
    "sampling:",
    "generate:",
    "main:",
)


def _is_banner_drop_line(line: str) -> bool:
    """Return True if a line is a banner trailing metadata line."""
    stripped = line.lstrip()
    return any(stripped.startswith(p) for p in _BANNER_DROP_PREFIXES)


def _is_exit_marker(line: str) -> bool:
    """Return True if a line is a llama.cpp end-of-session marker."""
    lower = line.lower()
    return any(
        marker in lower
        for marker in ("exiting...", "cleaning up", "exit code:")
    )


# Patterns llama.cpp uses to echo the prompt back to the user.  Different
# llama.cpp versions / chat-template configurations emit slightly
# different prefixes, so we try the common ones in order.
_PROMPT_ECHO_PREFIXES = ("> ", ">> ", ">>> ", " [user]: ", "[user]: ")


def _find_prompt_echo(raw: str, prompt: str) -> int:
    """Find the start of the prompt echo in ``raw``, or -1 if not found.

    Handles ``> {prompt}``, ``> > {prompt}``, ``[user]: {prompt}`` and
    the bare ``{prompt}`` form.
    """
    if not prompt:
        return -1
    # Build a list of patterns in priority order; shorter/looser first so
    # we don't miss when a longer marker isn't present.
    patterns = []
    for prefix in _PROMPT_ECHO_PREFIXES:
        patterns.append(f"{prefix}{prompt}")
    patterns.append(prompt)
    best = -1
    for pattern in patterns:
        idx = raw.find(pattern)
        if idx >= 0 and (best < 0 or idx < best):
            best = idx
    return best


def _clean_chat_tokens(line: str) -> str:
    """Remove chat control delimiters while preserving assistant content.

    Models sometimes wrap a response in ``<|im_start|>assistant`` and
    ``<|im_end|>``.  Removing a whole paired block loses the generated reply,
    so only delimiter tokens and role labels are removed here.
    """
    # Echoed user/system turns are not generated content. Remove those whole
    # turns first, but keep an assistant turn's payload.
    line = re.sub(r"<\|im_start\|>\s*(?:user|system)\s*\n.*?<\|im_end\|>", "", line, flags=re.DOTALL)
    line = re.sub(r"<\|im_start\|>\s*assistant\s*", "", line)
    line = re.sub(r"<\|im_end\|>", "", line)
    line = re.sub(r"<\|[a-z_]+\|>", "", line)
    line = re.sub(r"\[/?[A-Z][a-z]+(?: [a-z]+)*\]", "", line)
    return line


def _strip_llama_cpp_noise(text: str, debug: bool = False) -> str:
    """Strip llama.cpp's UI noise from stdout.

    Filters out:
    - ASCII logo / banner at the start (lines containing markers like
      ``llama_model_loader``, ``llm_load_tensors``, ``llama_print_system_info``)
    - Spinner characters and progress indicators
    - ``Exiting...`` / ``cleaning up`` / ``exit code:`` end-of-session
      messages
    - Chat template tokens (``<|im_start|>``, ``[Start thinking]``, etc.)

    Args:
        text: Raw stdout from ``llama-cli``.
        debug: When True, skip all filtering and return ``text`` unchanged.
            Useful for diagnosing model loading / banner / echo issues.

    Returns:
        Cleaned text with the model's response only.
    """
    if debug:
        return text

    lines = text.split("\n")
    filtered = []
    in_banner = True

    for line in lines:
        # --- phase 1: skip the leading banner block -------------------
        if in_banner:
            if _is_banner_line(line):
                continue
            if _is_banner_drop_line(line):
                # Sampling / generate / main: lines belong to the banner
                # block but are not strictly model output.  Drop them.
                continue
            if not line.strip():
                # Tolerate a few blank lines around the banner boundary.
                continue
            # First non-banner, non-empty line: end the banner block and
            # fall through to process this line as response content.
            in_banner = False

        # --- phase 2: skip end-of-session markers ----------------------
        if _is_exit_marker(line):
            continue

        # --- phase 3: collapse runs of blank lines --------------------
        if not line.strip():
            if filtered and filtered[-1].strip():
                filtered.append(line)
            continue

        # --- phase 4: strip chat template tokens ----------------------
        line = _clean_chat_tokens(line)
        if not line.strip():
            continue

        filtered.append(line)

    # Strip any trailing blank lines.
    while filtered and not filtered[-1].strip():
        filtered.pop()

    return "\n".join(filtered)


def _strip_cli_output(raw: str, prompt: str, debug: bool = False) -> str:
    """Strip banner, prompt echo and trailing messages from llama-cli stdout.

    ``llama-cli`` outputs a banner, echoes the ``> <prompt>`` line
    (sometimes as ``> > <prompt>`` or ``[user]: <prompt>`` depending on
    version and chat template), and appends an ``Exiting...`` message.
    This helper extracts just the generated text.

    Args:
        raw: Raw stdout captured from ``llama-cli``.
        prompt: The original prompt that was sent to the model.
        debug: When True, skip all filtering (banner / echo / tokens).

    Returns:
        The model's response text.
    """
    if debug:
        return raw.strip()

    # 1. Drop everything up to and including the prompt echo.
    echo_idx = _find_prompt_echo(raw, prompt)
    if echo_idx >= 0:
        # Find end of the echo line so we don't drag a trailing newline
        # off the start of the response.
        nl_idx = raw.find("\n", echo_idx)
        if nl_idx >= 0:
            raw = raw[nl_idx + 1 :]
        else:
            raw = ""

    # 2. Run the rest through the generic noise filter to remove chat
    #    template tokens and any trailing banner-style lines that may
    #    have slipped past the prompt-echo match.
    raw = _strip_llama_cpp_noise(raw, debug=debug)

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
    debug: bool = False,
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
        debug: When True, skip output filtering and return the raw
            ``llama-cli`` stdout (banner, prompt echo, ``Exiting...``
            and all).  Useful for diagnosing model loading issues.

    Returns:
        The generated text (response portion only, unless ``debug`` is True).

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

    cleaned = _strip_cli_output(result.stdout, prompt, debug=debug)
    return cleaned if debug else _strip_spinner(cleaned)


def _strip_prompt_echo_from_line(line: str, prompt: str) -> str:
    """Remove a prompt echo from one stdout line without dropping a reply."""
    echo_idx = _find_prompt_echo(line, prompt)
    if echo_idx < 0:
        return line
    return line[:echo_idx] + line[echo_idx + len(prompt):]


def _iter_stdout_chunks(stream: Any) -> Iterator[str]:
    """Yield stdout as soon as one text character is available."""
    while True:
        chunk = stream.read(1)
        if not chunk:
            return
        yield chunk


def _needs_complete_stream_line(fragment: str, prompt: str) -> bool:
    """Return whether a prefix must stay buffered for clean-mode filtering."""
    stripped = fragment.lstrip()
    if not stripped:
        return True
    candidates = (
        *_BANNER_MARKERS,
        *_BANNER_DROP_PREFIXES,
        "exiting...",
        "cleaning up",
        "exit code:",
        *_PROMPT_ECHO_PREFIXES,
        "<|im_start|>",
        "<|im_end|>",
        "[Start thinking]",
        "[/End]",
        prompt,
    )
    lowered = stripped.lower()
    return any(
        candidate
        and (
            candidate.startswith(stripped)
            or stripped.startswith(candidate)
            or candidate.lower().startswith(lowered)
            or lowered.startswith(candidate.lower())
        )
        for candidate in candidates
    )


def _clean_llama_cpp_stream(stdout: Any, prompt: str) -> Iterator[str]:
    """Clean llama.cpp stdout without withholding generated partial content."""
    in_banner = True
    buffer = ""
    passthrough = False
    for chunk in _iter_stdout_chunks(stdout):
        if passthrough:
            # Keep only a possible control-token prefix buffered. Generated
            # text continues immediately, including before its newline.
            if chunk in "<[":
                buffer = chunk
                passthrough = False
                continue
            yield chunk
            if chunk == "\n":
                passthrough = False
            continue

        buffer += chunk
        if chunk != "\n":
            if not _needs_complete_stream_line(buffer, prompt):
                if in_banner:
                    in_banner = False
                cleaned = _strip_spinner(
                    _clean_chat_tokens(_strip_prompt_echo_from_line(buffer, prompt))
                )
                if cleaned:
                    yield cleaned
                    buffer = ""
                    passthrough = True
            continue

        line = buffer
        buffer = ""
        if in_banner:
            if _is_banner_line(line) or _is_banner_drop_line(line) or not line.strip():
                continue
            in_banner = False
        if _is_exit_marker(line):
            continue
        cleaned = _strip_spinner(
            _clean_chat_tokens(_strip_prompt_echo_from_line(line, prompt))
        )
        if cleaned:
            yield cleaned

    if buffer and not in_banner and not _is_exit_marker(buffer):
        cleaned = _strip_spinner(_clean_chat_tokens(_strip_prompt_echo_from_line(buffer, prompt)))
        if cleaned:
            yield cleaned


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
    debug: bool = False,
) -> Iterator[str]:
    """Run generation while exposing partial stdout before a newline.

    Normal mode filters llama.cpp UI/prompt/control tokens. Debug mode is a raw
    stream and yields every stdout character unchanged.
    """
    binary = require_binary("llama-cli")
    args = _build_cli_args(
        binary, model_path, prompt, temperature=temperature, top_p=top_p,
        top_k=top_k, max_tokens=max_tokens, n_gpu_layers=n_gpu_layers,
        n_threads=n_threads, ctx_size=ctx_size, stop=stop,
    )
    with subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) as proc:
        assert proc.stdout is not None
        assert proc.stderr is not None
        chunks = _iter_stdout_chunks(proc.stdout) if debug else _clean_llama_cpp_stream(proc.stdout, prompt)
        yield from chunks
        proc.wait()
        if proc.returncode != 0:
            stderr = proc.stderr.read()
            raise RuntimeError(f"llama-cli stream failed (exit {proc.returncode}): {stderr.strip()}")


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

# ---------------------------------------------------------------------------
# Chat template extraction (from GGUF v3 metadata)
# ---------------------------------------------------------------------------

_CHAT_TEMPLATE_KEY = "tokenizer.chat_template"
_GGUF_MAGIC = b"GGUF"
_GGUF_SUPPORTED_VERSION = 3

# Safety caps so a malformed GGUF file cannot trick the parser into
# reading or allocating gigabytes of data.
_MAX_HEADER_BYTES = 4 * 1024 * 1024        # 4 MiB of metadata to scan
_MAX_KEY_BYTES = 4096                       # GGUF keys are short identifiers
_MAX_STRING_VALUE_BYTES = 16 * 1024 * 1024  # 16 MiB; any chat template is <1 MiB
_MAX_KV_COUNT = 100_000                     # reasonable upper bound for sanity

# GGUF value types (matches llama.cpp gguf.h)
_GGUF_TYPE_STRING = 8
_GGUF_TYPE_ARRAY = 9

# Match a simple Mustache / Go-template variable reference like
# ``{{ .Prompt }}`` or ``{{- .Prompt -}}`` and capture the bare
# identifier.  Nested paths (``.Foo.Bar``) and control structures
# (``range``, ``if``, ``end``) are intentionally not handled — chat
# templates that need them will simply have those tokens left in the
# output, which is a graceful degradation rather than a crash.
_CHAT_VAR_RE = re.compile(r"\{\{\s*-?\s*\.(\w+)\s*-?\s*\}\}")


def read_chat_template(model_path: str) -> Optional[str]:
    """Extract the ``tokenizer.chat_template`` string from a GGUF v3 file.

    The chat template (used by llama.cpp, transformers, etc. to render
    multi-turn conversations) is stored as a metadata KV pair under the
    key ``tokenizer.chat_template``.  This function parses only the
    GGUF header and stops as soon as the key is found, so it works on
    multi-GB model files without loading them into memory.

    The parser handles GGUF v3 only — older versions use a different
    on-disk layout and will be rejected.  Any parse error, missing
    key, or absent value returns :data:`None` rather than raising.

    Args:
        model_path: Path to a GGUF model file.

    Returns:
        The chat template string, or :data:`None` if the key is
        missing, the file is malformed, or a different GGUF version
        is used.
    """
    try:
        with open(model_path, "rb") as f:
            # Header: magic(4) + version(4) + tensor_count(8) + kv_count(8) = 24
            head = f.read(24)
            if len(head) < 24:
                return None
            if head[:4] != _GGUF_MAGIC:
                return None
            version, _tensor_count, kv_count = struct.unpack(
                "<IQQ", head[4:24]
            )
            if version != _GGUF_SUPPORTED_VERSION:
                return None
            if kv_count > _MAX_KV_COUNT:
                return None

            for _ in range(kv_count):
                if f.tell() > _MAX_HEADER_BYTES:
                    return None

                # --- key (length-prefixed UTF-8 string) -----------------
                key_len_bytes = f.read(8)
                if len(key_len_bytes) < 8:
                    return None
                key_len = struct.unpack("<Q", key_len_bytes)[0]
                if key_len > _MAX_KEY_BYTES:
                    # Pathological key — skip it and its value.
                    f.seek(key_len, 1)
                    type_bytes = f.read(4)
                    if len(type_bytes) < 4:
                        return None
                    value_type = struct.unpack("<I", type_bytes)[0]
                    _skip_gguf_value(f, value_type)
                    continue
                key_bytes = f.read(key_len)
                if len(key_bytes) < key_len:
                    return None
                key = key_bytes.decode("utf-8", errors="replace")

                # --- value type -----------------------------------------
                type_bytes = f.read(4)
                if len(type_bytes) < 4:
                    return None
                value_type = struct.unpack("<I", type_bytes)[0]

                if key == _CHAT_TEMPLATE_KEY and value_type == _GGUF_TYPE_STRING:
                    str_len_bytes = f.read(8)
                    if len(str_len_bytes) < 8:
                        return None
                    str_len = struct.unpack("<Q", str_len_bytes)[0]
                    if str_len > _MAX_STRING_VALUE_BYTES:
                        return None
                    val_bytes = f.read(str_len)
                    if len(val_bytes) < str_len:
                        return None
                    return val_bytes.decode("utf-8", errors="replace")

                _skip_gguf_value(f, value_type)
    except (OSError, struct.error, UnicodeDecodeError, ValueError):
        return None
    return None


# Fixed size (in bytes) for each GGUF primitive value type, used when
# skipping an array of primitives.  Variable-sized types (string, array)
# are not in this table and will cause a parse error if encountered.
_GGUF_PRIMITIVE_SIZES = {
    0: 1, 1: 1,                       # uint8, int8
    2: 2, 3: 2,                       # uint16, int16
    4: 4, 5: 4, 6: 4,                 # uint32, int32, float32
    7: 1,                             # bool
    10: 8, 11: 8, 12: 8,              # uint64, int64, float64
}


def _skip_gguf_value(f, value_type: int) -> None:
    """Skip over a GGUF metadata value of ``value_type``.

    Uses :func:`file.seek` so the underlying bytes are not read into
    memory.  Raises :class:`struct.error` on a truncated value so the
    outer ``read_chat_template`` can convert it to :data:`None`.
    """
    if value_type in (0, 1, 7):  # uint8, int8, bool
        f.seek(1, 1)
    elif value_type in (2, 3):  # uint16, int16
        f.seek(2, 1)
    elif value_type in (4, 5, 6):  # uint32, int32, float32
        f.seek(4, 1)
    elif value_type == _GGUF_TYPE_STRING:  # string
        length_bytes = f.read(8)
        if len(length_bytes) < 8:
            raise struct.error("truncated string length")
        length = struct.unpack("<Q", length_bytes)[0]
        f.seek(length, 1)
    elif value_type == _GGUF_TYPE_ARRAY:  # array
        header = f.read(12)
        if len(header) < 12:
            raise struct.error("truncated array header")
        element_type, count = struct.unpack("<IQ", header)
        elem_size = _GGUF_PRIMITIVE_SIZES.get(element_type)
        if elem_size is None:
            # Variable-size elements (e.g. strings inside an array)
            # are not worth handling here — bail and let the outer
            # try/except turn it into a graceful None.
            raise struct.error(
                f"unsupported array element type {element_type}"
            )
        f.seek(count * elem_size, 1)
    elif value_type in (10, 11, 12):  # uint64, int64, float64
        f.seek(8, 1)
    else:
        raise struct.error(f"unknown GGUF value type {value_type}")


def _render_chat_template(template: str, messages: list) -> str:
    """Render a simple Go-template / Mustache chat template.

    Supports ``{{ .Prompt }}``, ``{{ .System }}``, ``{{ .Response }}``,
    ``{{ .Content }}`` (alias of ``.Prompt``), and ``{{ .Role }}``.
    Any other ``{{ .VarName }}`` placeholder — including unknown
    identifiers like ``.Tools`` or ``.Stop`` — is replaced with an
    empty string.  Nested paths (``.Foo.Bar``) and control structures
    (``{{ range ... }}``, ``{{ if ... }}``) are left untouched in the
    output as a graceful degradation; the spec only requires basic
    Mustache substitution.

    If the template references ``{{ .Role }}`` it is applied per
    message in order (so each turn becomes one block of the template).
    Otherwise it is applied once to the full conversation using the
    first system message, the last user message, and the last
    assistant message.
    """
    uses_role = ("{{ .Role }}" in template) or ("{{.Role}}" in template)

    if uses_role:
        out_parts: List[str] = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            out_parts.append(
                _CHAT_VAR_RE.sub(
                    lambda m, r=role, c=content: {
                        "Role": r,
                        "Prompt": c,
                        "Content": c,
                        "System": "",
                        "Response": "",
                    }.get(m.group(1), ""),
                    template,
                )
            )
        return "".join(out_parts)

    system_text = ""
    for m in messages:
        if m.get("role") == "system":
            system_text = m.get("content", "")
            break
    user_msgs = [m for m in messages if m.get("role") == "user"]
    assistant_msgs = [m for m in messages if m.get("role") == "assistant"]
    prompt_text = user_msgs[-1].get("content", "") if user_msgs else ""
    response_text = (
        assistant_msgs[-1].get("content", "") if assistant_msgs else ""
    )

    return _CHAT_VAR_RE.sub(
        lambda m: {
            "Prompt": prompt_text,
            "Content": prompt_text,
            "System": system_text,
            "Response": response_text,
        }.get(m.group(1), ""),
        template,
    )


def format_chat_messages(
    messages: list,
    model_path: Optional[str] = None,
    chat_template_path: Optional[str] = None,
) -> str:
    """Format a list of ``{role, content}`` dicts into a prompt string.

    Resolution order for the chat template:

    1. If ``chat_template_path`` is provided and points to a readable
       file, the file content is loaded as a Jinja-style template and
       used.  This is the highest-priority override and lets users
       ship a hand-written ``template.jinja`` per model.
    2. Else if ``model_path`` is given and the GGUF file contains a
       ``tokenizer.chat_template`` metadata entry, that template is
       used (with ``{{ .Prompt }}`` / ``{{ .System }}`` / ``{{
       .Response }}`` placeholders substituted in).
    3. Otherwise this falls back to a hardcoded ``<|im_start|>``
       role-header format compatible with most instruct-tuned models.

    A missing or unreadable ``chat_template_path`` file is not an
    error; the function simply falls through to the next option.
    """
    # 1. Explicit file override or inline template.  A readable path retains
    # legacy file behavior; any other non-empty value is treated literally as
    # an inline profile/template string rather than a filesystem path.
    if chat_template_path is not None:
        template_path = Path(chat_template_path)
        try:
            if template_path.is_file():
                _explicit_template = template_path.read_text(encoding="utf-8")
            elif "{{" in chat_template_path or "<|" in chat_template_path or "\n" in chat_template_path:
                _explicit_template = chat_template_path
            else:
                _explicit_template = ""
            if _explicit_template.strip():
                return _render_chat_template(_explicit_template, messages)
        except (OSError, UnicodeDecodeError):
            pass

    # 2. Template baked into the GGUF model file
    if model_path is not None:
        template = read_chat_template(model_path)
        if template:
            return _render_chat_template(template, messages)

    parts: List[str] = []
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
        stop: Optional[List[str]] = None,
        debug: bool = False,
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
        self.stop = list(stop or [])
        self.debug = debug
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
                stop=self.stop,
                debug=self.debug,
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

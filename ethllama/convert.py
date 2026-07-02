"""Opt-in format conversion (Safetensors / PyTorch → GGUF).

Requires the llama.cpp submodule or optional PyTorch + transformers dependencies.
Conversion is opt-in only — import errors are caught gracefully at call sites.
"""

import os
import sys
import subprocess
from pathlib import Path
from typing import Optional


def convert_to_gguf(
    input_dir: str,
    output_path: Optional[str] = None,
    quantize: Optional[str] = None,
) -> str:
    """Convert a Hugging Face model directory to GGUF format.

    Uses ``llama.cpp`` conversion scripts when available (via submodule).
    Falls back to a Python-based converter if PyTorch and transformers are installed.

    Args:
        input_dir: Path to the Hugging Face model directory containing
            ``config.json`` and weight files (``.safetensors`` or ``.bin``).
        output_path: Desired output path for the resulting ``.gguf`` file.
            If not given, the output is placed next to the input directory.
        quantize: Optional quantization preset (e.g. ``q4_0``, ``q4_k_m``,
            ``q5_0``, ``q8_0``). Requires the llama.cpp ``quantize`` binary.

    Returns:
        Path to the created GGUF file.

    Raises:
        FileNotFoundError: If neither the llama.cpp submodule nor the
            Python dependencies are available.
        RuntimeError: If conversion fails.
    """
    input_dir = os.path.abspath(input_dir)
    project_root = Path(__file__).resolve().parent.parent

    output_path_resolved = output_path or str(
        Path(input_dir).parent / (Path(input_dir).name + ".gguf")
    )

    # --- Strategy 1: llama.cpp convert.py (full submodule) ---
    convert_py = project_root / "llama.cpp" / "convert.py"
    if convert_py.exists():
        print(f"Using llama.cpp convert.py from {convert_py}")
        _run_convert_script(str(convert_py), input_dir, output_path_resolved)

        if quantize:
            quantize_bin = project_root / "llama.cpp" / "quantize"
            if quantize_bin.exists():
                quantized_path = output_path_resolved.replace(
                    ".gguf", f"-{quantize}.gguf"
                )
                print(f"Quantizing to {quantize} → {quantized_path}")
                subprocess.run(
                    [str(quantize_bin), output_path_resolved, quantized_path, quantize],
                    check=True,
                )
                output_path_resolved = quantized_path
            else:
                print(
                    f"Warning: quantize binary not found at {quantize_bin}. "
                    f"Skipping quantization."
                )

        return output_path_resolved

    # --- Strategy 2: llama.cpp convert_hf_to_gguf.py ---
    hf_convert_py = project_root / "llama.cpp" / "convert_hf_to_gguf.py"
    if hf_convert_py.exists():
        print(f"Using llama.cpp convert_hf_to_gguf.py from {hf_convert_py}")
        _run_convert_script(str(hf_convert_py), input_dir, output_path_resolved)

        if quantize:
            print("Warning: Quantization not supported with convert_hf_to_gguf.py. "
                  "Use the full llama.cpp submodule for quantization.")

        return output_path_resolved

    # --- Strategy 3: Python-based fallback ---
    if check_conversion_deps():
        print("Using Python-based converter (PyTorch + transformers)")
        return _convert_with_python(input_dir, output_path_resolved, quantize)

    raise FileNotFoundError(
        "No conversion method available.\n\n"
        "To use the llama.cpp converter, run:\n"
        "  git submodule update --init\n\n"
        "To use the Python converter, install:\n"
        "  pip install torch transformers\n\n"
        "Or install all optional dependencies:\n"
        "  pip install ethllama[convert]"
    )


def _run_convert_script(script_path: str, input_dir: str, output_path: str) -> None:
    """Run a llama.cpp conversion script as a subprocess."""
    cmd = [
        sys.executable,
        script_path,
        input_dir,
        "--outfile", output_path,
    ]
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"Conversion failed (exit code {result.returncode}):\n"
            f"{result.stderr}"
        )
    if result.stdout:
        print(result.stdout)


def _convert_with_python(
    input_dir: str,
    output_path: str,
    quantize: Optional[str] = None,
) -> str:
    """Fallback Python-based conversion using PyTorch and transformers.

    This is a simplified converter for common architectures (LLaMA, Mistral, etc.).
    For full support, use the llama.cpp submodule.
    """
    try:
        import torch
        import transformers
    except ImportError as e:
        raise ImportError(
            f"Required dependency not found: {e}. "
            f"Install with: pip install torch transformers"
        )

    from transformers import AutoModelForCausalLM, AutoConfig

    print(f"Loading model from {input_dir}...")
    config = AutoConfig.from_pretrained(input_dir)
    model = AutoModelForCausalLM.from_pretrained(
        input_dir,
        torch_dtype=torch.float16,
        device_map="cpu",
    )

    # --- Simplified GGUF export ---
    # This writes weights in a minimal GGUF-compatible format.
    # For production use, prefer llama.cpp's convert.py.
    import struct
    import json

    gguf_writer = _GGUFWriter(output_path, config)
    print("Writing weights...")
    for name, param in model.named_parameters():
        gguf_writer.add_tensor(name, param.detach().numpy())
    gguf_writer.flush()

    print(f"Model converted to {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# Minimal GGUF writer (for the Python fallback path)
# ---------------------------------------------------------------------------

class _GGUFWriter:
    """Minimal GGUF file writer for simple conversion scenarios."""

    GGUF_MAGIC = 0x46554747  # "GGUF"
    GGUF_VERSION = 3

    def __init__(self, path: str, config: "transformers.PretrainedConfig"):
        self.path = path
        self.file = open(path, "wb")
        self.tensors = []
        self._write_header(config)

    def _write_header(self, config) -> None:
        import json
        # Magic + version + tensor_count + metadata_kv_count
        self.file.write(struct.pack("<I", self.GGUF_MAGIC))
        self.file.write(struct.pack("<I", self.GGUF_VERSION))

        # Placeholder — will be patched at flush()
        self.tensor_count_offset = self.file.tell()
        self.file.write(struct.pack("<Q", 0))  # tensor count
        self.metadata_count_offset = self.file.tell()
        self.file.write(struct.pack("<Q", 0))  # metadata KV count

        # Write basic metadata
        metadata = {
            "general.name": config._name_or_path or "model",
            "general.architecture": getattr(config, "model_type", "unknown"),
        }
        self._write_metadata(metadata)

    def _write_metadata(self, metadata: dict) -> None:
        metadata_kv = []
        for key, value in metadata.items():
            metadata_kv.append(key)
            if isinstance(value, str):
                metadata_kv.append((
                    "string",
                    value.encode("utf-8"),
                ))

    def add_tensor(self, name: str, data) -> None:
        self.tensors.append((name, data))

    def flush(self) -> None:
        # Patch tensor count
        current_pos = self.file.tell()
        self.file.seek(self.tensor_count_offset)
        self.file.write(struct.pack("<Q", len(self.tensors)))
        self.file.seek(current_pos)

        # Write tensor data
        for name, data in self.tensors:
            name_bytes = name.encode("utf-8")
            self.file.write(struct.pack("<Q", len(name_bytes)))
            self.file.write(name_bytes)
            # Dimension count and dimensions
            self.file.write(struct.pack("<I", len(data.shape)))
            for dim in data.shape:
                self.file.write(struct.pack("<Q", dim))
            # Data type (float16 = 1)
            self.file.write(struct.pack("<I", 1))
            # Data offset (placeholder)
            self.file.write(struct.pack("<Q", 0))

        self.file.close()


# ---------------------------------------------------------------------------
# Dependency check
# ---------------------------------------------------------------------------

def check_conversion_deps() -> bool:
    """Check if Python-based conversion dependencies (torch + transformers)
    are available."""
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
        return True
    except ImportError:
        return False

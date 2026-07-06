# ollama-registry-pull — Ollama OCI Registry download pattern

Use when implementing or debugging model pulls from `registry.ollama.ai`.
Documents the protocol quirks, gotchas, and reference implementations discovered through reverse-engineering.

## Trigger phrases

- "implement pull from Ollama"
- "download from registry.ollama.ai"
- "update the Ollama pull logic"
- "Ollama registry protocol"

---

## Key Facts

### The model blob IS a GGUF file

The single most important finding: `application/vnd.ollama.image.model` contains the raw GGUF bytes. No conversion, no assembly, no unpacking needed. Verify with magic bytes `b"GGUF"` (0x46554747 LE) on download.

### Endpoints (OCI Distribution Spec v2)

| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/v2/<ns>/<repo>/manifests/<tag>` | Fetch JSON manifest |
| `GET` | `/v2/<ns>/<repo>/blobs/sha256:<digest>` | Download a blob |

### Model ref parsing

```python
# Dots in first path component = hostname (e.g. registry.ollama.ai/library/llama3)
# No dots = namespace (e.g. library/llama3 or just llama3 → library/llama3)
# Default tag: latest
# Default host: registry.ollama.ai
```

Implementation in `ethllama/pull.py:_parse_model_ref()`.

---

## Protocol Quirks

| Quirk | Impact | Workaround |
|-------|--------|------------|
| Manifest returns `Content-Type: text/plain` | Parser sees wrong type | Parse body as JSON anyway |
| Manifest by digest (`manifests/sha256:...`) returns 500 | Digest-based addressing broken | Always use tags |
| Blob GETs 307-redirect to Cloudflare R2 | Standard `requests` follows it | `allow_redirects=True` (default) |
| No auth needed for public models | Works anonymously | Skip auth entirely |
| `Accept: application/vnd.docker.distribution.manifest.v2+json` required | Without it → MANIFEST_INVALID | Always set this header |

## Download pattern

```python
import hashlib
import requests

def pull_ollama_model(manifest_url: str, blob_url: str, dest: Path,
                      expected_sha: str, expected_size: int) -> Path:
    """Download with SHA-256 verification and .partial resume."""
    temp = dest.with_suffix(".partial")
    pos = temp.stat().st_size if temp.exists() else 0
    headers = {"Range": f"bytes={pos}-"} if pos else {}

    sha = hashlib.sha256()
    if pos:
        with open(temp, "rb") as f:
            while buf := f.read(1 << 20):
                sha.update(buf)

    with requests.get(blob_url, headers=headers, stream=True,
                      allow_redirects=True, timeout=300) as r:
        r.raise_for_status()
        with open(temp, "ab" if pos else "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                sha.update(chunk)

    if sha.hexdigest() != expected_sha:
        temp.unlink()
        raise ValueError(f"SHA-256 mismatch: got {sha.hexdigest()}, expected {expected_sha}")

    temp.rename(dest)
    return dest
```

## Manifest format

```json
{
  "schemaVersion": 2,
  "mediaType": "application/vnd.docker.distribution.manifest.v2+json",
  "config": { "mediaType": "application/vnd.docker.container.image.v1+json", ... },
  "layers": [
    { "mediaType": "application/vnd.ollama.image.model", "digest": "sha256:...", "size": 2019393189 },
    { "mediaType": "application/vnd.ollama.image.license", "digest": "sha256:...", "size": 8433 },
    { "mediaType": "application/vnd.ollama.image.template", "digest": "sha256:...", "size": 136 },
    { "mediaType": "application/vnd.ollama.image.params", "digest": "sha256:...", "size": 84 }
  ]
}
```

## Reference implementations

- `iven86/ollama_gguf_downloader` — simple CLI, MIT
- `olamide226/ollama-gguf-downloader` — CLI with progress
- `leeroopedia/workflow-ollama-ollama-model-registry-operations` — most complete (push/pull, Ed25519 auth, parallel chunked)
- `mudler/LocalAI/pkg/oci/ollama.go` — clean Go reference
- `ethllama/pull.py:pull_from_ollama()` — this project's implementation (~150 lines)

## Known limitations

- Sharded GGUFs (`-00001-of-00003.gguf`) are NOT in the public registry — Ollama rejects them at upload
- Registry is officially **unofficial** — Ollama team won't guarantee stability
- New per-tensor layer schema exists but is not deployed to the public registry yet

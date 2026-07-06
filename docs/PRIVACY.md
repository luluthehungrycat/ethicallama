# Privacy Policy

ethicallama is designed from the ground up to be a **local-first, privacy-respecting** LLM inference tool. This document explains what data is collected, what is never collected, and how data is handled.

## Core Principle

**ethicallama runs entirely on your local machine.** Inference happens on your hardware. Your prompts, model inputs, and generated outputs never leave your computer unless you explicitly configure a network service that transmits them.

## Data That Is NEVER Collected

The following data is **never collected** under any default configuration:

- **Prompts and Inputs**: Text you send to models for inference.
- **Generated Outputs**: Text produced by models during inference.
- **Model Files**: The paths, contents, or metadata of your model files are never transmitted.
- **System Information**: Hardware specs, OS version, or environment details are not collected.
- **Usage Statistics**: No tracking of how, when, or how often you use the software.
- **Personal Identifiers**: No user IDs, machine IDs, or any form of fingerprinting.

## Telemetry Policy

### Default State: DISABLED

Telemetry is **disabled by default**. No data is sent anywhere unless you explicitly opt in.

### Opt-In Telemetry

If you choose to enable telemetry (via `ethllama config --init` or by editing `~/.ethllama/config.yaml`), the following **may** be collected:

- Anonymous usage counters (e.g., number of inference requests)
- GPU backend selection (e.g., "vulkan", "cuda")
- Model format identifiers (e.g., ".gguf")
- Error types (e.g., "out of memory", "model load failure")

The configuration system explicitly asks for confirmation before enabling telemetry:

```
Enable anonymous telemetry? (WARNING: This shares usage statistics) [y/N]:
```

You can view and change your telemetry setting at any time in `~/.ethllama/config.yaml`:

```yaml
telemetry:
  enabled: false   # Set to true to enable
```

### What Opt-In Telemetry NEVER Includes

Even with telemetry enabled, the following are **never** collected:

- Your prompts or model outputs
- Model file paths or contents
- Your IP address (beyond what is necessary for the telemetry endpoint)
- Any personal information

## Data Handling

### Configuration Files

Your configuration is stored locally at `~/.ethllama/config.yaml`. This file contains:
- Your GPU backend preference
- API server settings (if enabled)
- Telemetry opt-in status
- Model directory paths

This file is never shared or transmitted.

### Model Index

The model index is stored at `~/.ethllama/index.json`. It contains:
- File names and paths of discovered models
- File sizes and modification timestamps

This index is used solely to enable model lookup by name. It is never transmitted.

### HTTP API

When you run the HTTP API server (`ethllama serve`), prompts and outputs flow over your local network. This data is **never persisted or logged to external services**. By default, the server binds to `127.0.0.1` (localhost only). You are responsible for securing the API if you expose it to a network.

## Your Rights

- **Right to know**: This document explains all data practices.
- **Right to opt out**: Telemetry is opt-in and can be disabled at any time.
- **Right to delete**: Delete `~/.ethllama/` to remove all local configuration and indexes.
- **Right to audit**: All source code is open (MIT License) and available for inspection.

## Changes to This Policy

If this privacy policy changes, the version number in the repository will be updated along with a changelog entry. Significant changes will be highlighted in release notes.

## Contact

For questions about privacy, please open an issue on the project repository.

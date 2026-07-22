# Systemd installation

Use the packaged setup command; service units are generated from Python and
are not copied from this repository.

## User service (default)

```bash
ethllama setup
```

This writes a user unit using the resolved `ethllama` executable and
`~/.ethllama/config.yaml`. It configures loopback host/port, creates new config
paths with private permissions, never invokes `sudo`, and only reports linger
status (it never enables linger).

## System service (explicit)

```bash
ethllama setup --service-mode system
```

Run this as the non-root account that should run the service. It requires
passwordless `sudo -n`; setup refuses root invocation and never falls back to
user mode. The generated PID-1 service runs as that invoking user and receives
root-owned `/etc/ethllama/config.yaml` through systemd `LoadCredential`, so API
credentials are not present in the public unit file. No privileged installation
is attempted unless this explicit mode is selected.

Use `ethllama setup --no-install` to update only the user configuration. Use
`--no-api-key` only when you intentionally want to disable existing API-key
authentication; `--yes` preserves existing authentication.

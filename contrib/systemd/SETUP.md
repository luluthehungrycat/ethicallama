# Setup guide for ethicallama

This guide walks you through the recommended way to install and configure
ethicallama for day-to-day use.  If you just want a single command that does
all of this for you, run `ethllama setup` — it covers the same ground as this
document in an interactive wizard.

## Before you start

You need:

- A Linux system with `systemd` (most modern distros)
- Python 3.10+ on `PATH`
- An installation of `ethllama` (e.g. `uv tool install ethicallama`)

The `ethllama setup` wizard will check for you and tell you what is missing
before making any changes.

## Step 1 — Pick an installation mode

`ethllama setup` can install a systemd service in two modes.  It chooses one
automatically based on whether your user has `sudo` access:

### System mode (recommended for servers, requires `sudo`)

- Service file lives at `/etc/systemd/system/ethllama.service`
- Runs the configured user (defaults to the user that ran the installer)
- Starts on boot, before any user logs in
- Best for headless boxes, NAS, homelab, etc.

### User mode (no `sudo` required)

- Service file lives at `~/.config/systemd/user/ethllama.service`
- Always runs as the user that ran the installer
- Starts when that user logs in (or on boot if `loginctl enable-linger $USER`
  is set)
- Best for laptops and workstations

If you don't have `sudo` access, `ethllama setup` will fall back to user
mode automatically.

## Step 2 — Run the wizard

```bash
ethllama setup
```

The wizard will:

1. Check that `ethllama` is on your `PATH`.
2. Look for known inference engines (e.g. `llama-cli`, `ollama`) and ask you
   to confirm the binary directory to record in `config.yaml`.
3. Ask whether to enable API key authentication and, if so, optionally
   generate one for you.
4. Save the configuration to `~/.ethllama/config.yaml`.
5. Install the systemd service file in the chosen location.
6. Run `systemctl daemon-reload` and `systemctl enable --now ethllama`.

You can re-run `ethllama setup` at any time to change the configuration; it
will not overwrite unrelated entries in your `config.yaml`.

## Step 3 — Useful flags

```bash
# Skip interactive prompts and accept all defaults
ethllama setup --yes

# Only update config.yaml, don't touch systemd
ethllama setup --no-install

# Force user mode (no sudo required)
ethllama setup --service-mode user

# Force system mode (will prompt for sudo if needed)
ethllama setup --service-mode system

# Pick a different API port (default 10434)
ethllama setup --port 8080

# Use a specific binary directory
ethllama setup --binary-dir /opt/llama.cpp/build/bin
```

## Step 4 — Verify

After the wizard finishes, check that the service is running:

```bash
# System mode
sudo systemctl status ethllama
sudo journalctl -u ethllama -f

# User mode
systemctl --user status ethllama
journalctl --user -u ethllama -f
```

Then try the API:

```bash
curl http://localhost:10434/health
```

## Manual install (without the wizard)

If you'd rather install the service file by hand, see `INSTALL.md` for the
manual install guide.  The two service files in this directory are:

- `ethllama.service` — system-mode unit (Pattern A: per-user, `uv tool`
  install).  Edit `User=` and the home-directory paths before installing.
- `ethllama-user.service` — user-mode unit.  Uses `%u` and `%h` specifiers
  so the same file works for every user; no editing required.

## Enabling linger for user services

User-mode services stop running when you log out.  If you want the service
to stay up across logouts (e.g. on a headless box), enable linger:

```bash
loginctl enable-linger $USER
```

The wizard prints a reminder if linger is not yet enabled.

## Troubleshooting

### `ethllama: command not found` during setup

The wizard cannot find the `ethllama` binary.  Make sure you installed it:

```bash
uv tool install ethicallama
# or
pipx install ethicallama
# or
pip install --user ethicallama
```

…and that the install location is on your `PATH` (e.g. `~/.local/bin`).

### `sudo: a password is required`

You have `sudo` configured but it requires a password for non-interactive
use.  Either run the wizard with `--service-mode user` (no sudo needed) or
configure passwordless sudo for the relevant commands.

### Service installs but fails to start

Check the journal:

```bash
sudo journalctl -u ethllama -n 50           # system mode
journalctl --user -u ethllama -n 50         # user mode
```

Common causes:

- **Missing API extras.** The service needs `fastapi`, `uvicorn`, and
  `pydantic`.  Reinstall with `uv tool install --with fastapi --with
  'uvicorn[standard]' --with pydantic ethicallama`.
- **`ProtectHome=true` blocking access to `~/.local`.**  This is already
  configured as `read-only` in the bundled service files; if you edited it,
  revert to `read-only` (or use the per-user install pattern).

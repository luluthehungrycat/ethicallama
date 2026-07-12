# Systemd installation

This guide covers running ethicallama as a systemd service. There are two
common installation paths depending on how you installed the package.

## Prerequisites

- A Linux system with systemd (most modern distros)
- An `ethllama` configuration in `~/.ethllama/config.yaml`
  (run `ethllama config --init` first)
- sudo access

## Pick the right user pattern

There are two common patterns for running ethicallama as a service. Choose the one that matches your install method.

### Pattern A: Per-user install (recommended for `uv tool install`)

This is the simplest setup if you used `uv tool install ethicallama`. The service runs as **your own user account**, so it can read your `~/.local/bin/ethllama` and the uv-managed venv at `~/.local/share/uv/tools/ethicallama/`.

```ini
[Service]
User=moritz           # ← your username
Environment="PATH=/usr/local/bin:/usr/bin:/bin:/home/moritz/.local/bin"
ProtectHome=read-only
ReadWritePaths=/home/moritz/.local/share/uv
```

**Note:** The v0.1.7+ service file uses `User=moritz` and `ProtectHome=read-only` by default. Edit to match your actual username.

**Don't** use the dedicated `ethllama` user — that user can't access your home dir.

### Pattern B: Dedicated system user (for system-wide `pip install`)

For system-wide installs (e.g., `sudo pip install ethicallama`), use a dedicated `ethllama` user with stronger isolation. You must also install the API dependencies for the system user:

```bash
sudo useradd --system --shell /usr/sbin/nologin --home /var/lib/ethllama ethllama
sudo mkdir -p /etc/ethllama /var/lib/ethllama
sudo chown ethllama:ethllama /var/lib/ethllama

# Run as ethllama user
sudo -u ethllama pip install fastapi 'uvicorn[standard]' pydantic
```

Then use:
```ini
[Service]
User=ethllama
ProtectHome=true  # stronger isolation, can't read /home/$USER
```

## Install steps

### 1. Install ethicallama

**Pattern A (per-user, recommended):**
```bash
# Install ethicallama with API support
uv tool install --with fastapi --with 'uvicorn[standard]' --with pydantic ethicallama
```

**Pattern B (system-wide):**
```bash
sudo pip install "ethicallama[all]"
# or
sudo pip install ethicallama "ethicallama[api]" -- target /usr/local
```

### 2. Install the service file

```bash
cd /path/to/ethicallama  # this repo
sudo cp contrib/systemd/ethllama.service /etc/systemd/system/
```

### 3. Edit the service file

If using **Pattern A**, change `User=moritz` to your actual username. Also update `Environment=PATH` and `ReadWritePaths=` to match.

If using **Pattern B**, change `User=moritz` to `User=ethllama` and uncomment the `Group=ethllama` line.

### 4. Reload and start

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now ethllama
```

### 5. Verify

```bash
sudo systemctl status ethllama
sudo journalctl -u ethllama -f
curl http://localhost:10434/health
```

## Troubleshooting

### `Failed at step NAMESPACE spawning /usr/local/bin/ethllama: No such file or directory`

The `ethllama` binary isn't at `/usr/local/bin/ethllama`. Either:
- (Pattern A users) You installed via `uv tool install`. Make sure the service runs as **your user** and `Environment=PATH` includes `~/.local/bin`. Don't use the v0.1.6-era `User=ethllama` line.
- (Pattern B users) The binary should be at `/usr/local/bin/ethllama`. Reinstall with `sudo pip install ethicallama` if missing.

### `No module named 'fastapi'`

You installed ethicallama without the API extras:
```bash
uv tool uninstall ethicallama
uv tool install --with fastapi --with 'uvicorn[standard]' --with pydantic ethicallama
```

Or install system-wide:
```bash
sudo pip install "ethicallama[api]"
```

### Service runs but `curl` returns "Connection refused"

The service is binding to 127.0.0.1. For network access, edit the service:
```
ExecStart=ethllama serve --host 0.0.0.0 --port 10434
```

### Permission denied reading `~/.local/`

The service user can't traverse your home. Switch to **Pattern A** (run as your own user) instead of `User=ethllama`.

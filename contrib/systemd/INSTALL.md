# Systemd installation

This guide covers running ethicallama as a systemd service. There are two
common installation paths depending on how you installed the package.

## Prerequisites

- A Linux system with systemd (most modern distros)
- An `ethllama` configuration in `~/.ethllama/config.yaml`
  (run `ethllama config --init` first)
- sudo access

## Choose your install method

### Option A: `pip install` (system-wide)

The package is installed to `/usr/local/bin/ethllama` and the system
Python environment. The systemd service can use a dedicated system user.

```bash
sudo pip install ethicallama
# or with all extras:
sudo pip install "ethicallama[all]"
```

### Option B: `uv tool install` (per-user, recommended for development)

The package is installed to `~/.local/bin/ethllama` in a uv-managed
virtual environment. This is recommended for development since you can
have multiple versions side-by-side.

```bash
uv tool install ethicallama
# or with extras:
uv tool install 'ethicallama[all]'
```

This is the install method used by uv. The binary lives at
`~/.local/bin/ethllama` (not `/usr/local/bin/ethllama`).

## Install steps

### 1. Set up the service user (Option A only)

If you used `pip install` and want a dedicated service user:

```bash
sudo useradd --system --shell /usr/sbin/nologin --home /var/lib/ethllama ethllama
sudo mkdir -p /etc/ethllama /var/lib/ethllama
sudo chown ethllama:ethllama /var/lib/ethllama
```

If you used `uv tool install`, you can run as your own user. Skip this
step and edit the `User=` line in the service file to match.

### 2. Install the service file

```bash
sudo cp contrib/systemd/ethllama.service /etc/systemd/system/
```

If running as a non-root user (uv tool case), edit the service file and
change `User=ethllama` to your username (e.g. `User=moritz`).

### 3. (Optional) Configure environment

```bash
sudo cp contrib/systemd/ethllama.env.example /etc/ethllama/ethllama.env
sudo nano /etc/ethllama/ethllama.env
```

Uncomment and set:
- `ETHLLAMA_API_KEY=your-secret-here` — for API key auth
- `ETHLLAMA_SSL_KEYFILE=/etc/ethllama/server.key` — for HTTPS
- `ETHLLAMA_SSL_CERTFILE=/etc/ethllama/server.crt`

### 4. Reload systemd and start

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

The `ethllama` binary isn't at `/usr/local/bin/ethllama`. This means you
installed via `uv tool install` (binary at `~/.local/bin/ethllama`).

**Fix**: Either:
1. Edit the service file and set `User=` to your username (the user that
   installed ethicallama), and add `~/.local/bin` to `Environment=PATH`:
   ```
   [Service]
   User=moritz
   Environment="PATH=/home/moritz/.local/bin:/usr/local/bin:/usr/bin:/bin"
   ```
2. Or create a symlink:
   ```bash
   sudo ln -s /home/$USER/.local/bin/ethllama /usr/local/bin/ethllama
   ```

### Service runs but `curl` returns "Connection refused"

The service is binding to 127.0.0.1 by default. For network access, edit
the service file:
```
ExecStart=ethllama serve --host 0.0.0.0 --port 10434
```

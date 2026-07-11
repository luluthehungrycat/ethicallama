# Systemd installation

1. Create the user:
   ```bash
   sudo useradd --system --shell /usr/sbin/nologin --home /var/lib/ethllama ethllama
   ```

2. Create directories:
   ```bash
   sudo mkdir -p /etc/ethllama /var/lib/ethllama
   sudo chown ethllama:ethllama /var/lib/ethllama
   ```

3. Install the service file:
   ```bash
   sudo cp contrib/systemd/ethllama.service /etc/systemd/system/
   ```

4. (Optional) Configure environment:
   ```bash
   sudo cp contrib/systemd/ethllama.env.example /etc/ethllama/ethllama.env
   sudo nano /etc/ethllama/ethllama.env
   ```

5. Reload and enable:
   ```bash
   sudo systemctl daemon-reload
   sudo systemctl enable --now ethllama
   ```

6. Check status:
   ```bash
   sudo systemctl status ethllama
   sudo journalctl -u ethllama -f
   ```

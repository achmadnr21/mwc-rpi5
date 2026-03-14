#!/bin/bash

# Exit on error
set -e

echo "=== MarkasWalet Stream Installer ==="
echo

# install python3-opencv
echo "[1/6] Installing python3-opencv..."
sudo apt install python3-opencv -y
sudo pip install opencv-contrib-python-headless --break-system-packages
sleep 2

# Menyalin direktori ke /usr/local/bin/
echo "[2/6] Copying markaswalet-stream to /usr/local/bin/..."
sudo cp -rf markaswalet-stream /usr/local/bin/
sleep 1

# Menyalin file service ke /etc/systemd/system/
echo "[3/6] Copying service file to /etc/systemd/system/..."
sudo cp -f markaswalet-stream.service /etc/systemd/system/
sleep 1

# Reload systemd daemon
echo "[4/6] Reloading systemd daemon..."
sudo systemctl daemon-reload
sleep 1

# Enable service
echo "[5/6] Enabling markaswalet-stream service..."
sudo systemctl enable markaswalet-stream.service
sleep 1

# Start service
echo "[6/6] Starting markaswalet-stream service..."
sudo systemctl start markaswalet-stream.service
sleep 2

echo
echo "✓ Installation complete!"
echo
echo "Useful commands:"
echo "  - Check status:  sudo systemctl status markaswalet-stream"
echo "  - View logs:     sudo journalctl -u markaswalet-stream -f"
echo "  - Stop service:  sudo systemctl stop markaswalet-stream"
echo "  - Restart:       sudo systemctl restart markaswalet-stream"
echo

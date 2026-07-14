#!/bin/bash
# Keep codex-shim alive - restart on crash
cd /opt/codes/codex-shim
export PYTHONPATH="/opt/codes/codex-shim:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

while true; do
    echo "[shim-guard] Starting codex-shim at $(date)" >> /tmp/shim.log
    python3 -m codex_shim.server --host 127.0.0.1 --port 8765 >> /tmp/shim.log 2>&1
    EXIT_CODE=$?
    echo "[shim-guard] codex-shim exited with code $EXIT_CODE at $(date)" >> /tmp/shim.log
    sleep 2
done

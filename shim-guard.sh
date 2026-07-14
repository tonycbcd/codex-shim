#!/bin/bash
# Keep codex-shim alive - restart on crash
cd /opt/codes/codex-shim
export PYTHONPATH="/opt/codes/codex-shim:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
# Reasoning effort: low / medium / high (low = fastest, high = best quality)
export CODEX_SHIM_REASONING_EFFORT=medium

while true; do
    # Kill any lingering process on port 8765 before starting
    PID=$(lsof -ti :8765 2>/dev/null || fuser 8765/tcp 2>/dev/null | awk '{print $1}')
    if [ -n "$PID" ]; then
        kill -9 $PID 2>/dev/null
        sleep 1
    fi

    echo "[shim-guard] Starting codex-shim at $(date)" >> /tmp/shim.log
    python3 -m codex_shim.server --host 127.0.0.1 --port 8765 >> /tmp/shim.log 2>&1
    EXIT_CODE=$?
    echo "[shim-guard] codex-shim exited with code $EXIT_CODE at $(date)" >> /tmp/shim.log
    sleep 3
done

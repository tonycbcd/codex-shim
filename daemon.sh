#!/bin/bash
# Codex Shim Daemon - auto-restart on crash

cd /opt/codes/codex-shim
PORT=8765

cleanup() {
    echo "[daemon] Stopping..."
    pkill -f "codex_shim.server.*--port $PORT" 2>/dev/null
    exit 0
}

trap cleanup SIGTERM SIGINT

while true; do
    # Check if already running
    if lsof -i :$PORT >/dev/null 2>&1; then
        echo "[daemon] Port $PORT already in use, killing existing process..."
        pkill -f "codex_shim.server.*--port $PORT" 2>/dev/null
        sleep 2
    fi
    
    echo "[daemon] Starting codex-shim at $(date)"
    .venv/bin/python -m codex_shim.server --host 127.0.0.1 --port $PORT
    EXIT_CODE=$?
    echo "[daemon] codex-shim exited with code $EXIT_CODE at $(date)"
    
    if [ $EXIT_CODE -eq 0 ]; then
        echo "[daemon] Clean exit, stopping daemon"
        break
    fi
    
    echo "[daemon] Restarting in 3 seconds..."
    sleep 3
done

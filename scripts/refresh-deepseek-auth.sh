#!/usr/bin/env bash
# Refresh DeepSeek Web auth from Firefox profile
# Run via cron: 0 */12 * * * /opt/codes/codex-shim/scripts/refresh-deepseek-auth.sh
set -euo pipefail

PROFILE="/config/.config/mozilla/firefox/kwcqje0m.default-release"
DATA_DIR="/opt/codes/codex-shim/.deepseek-web-data"
WORK_DIR="/tmp/deepseek-auth-refresh-$$"

log() { echo "[$(date -Iseconds)] $*"; }

cleanup() { rm -rf "$WORK_DIR" 2>/dev/null || true; }
trap cleanup EXIT

mkdir -p "$WORK_DIR"

# Copy databases to avoid locking issues with live Firefox
cp "$PROFILE/cookies.sqlite" "$WORK_DIR/cookies.sqlite"
cp "$PROFILE/storage/default/https+++chat.deepseek.com/ls/data.sqlite" "$WORK_DIR/localstorage.sqlite"

python3 - "$WORK_DIR" "$DATA_DIR" <<'PY'
import sqlite3, json, pathlib, datetime, sys

work = pathlib.Path(sys.argv[1])
data_dir = pathlib.Path(sys.argv[2])

# Read cookies
con = sqlite3.connect(work / 'cookies.sqlite')
rows = con.execute("""
    SELECT name, value, host, path, expiry, isHttpOnly, isSecure, sameSite
    FROM moz_cookies WHERE host LIKE '%deepseek.com'
""").fetchall()

# Read userToken from localStorage
ls = sqlite3.connect(work / 'localstorage.sqlite')
blob = ls.execute("SELECT value FROM data WHERE key='userToken'").fetchone()
if not blob:
    print("ERROR: userToken not found in localStorage", file=sys.stderr)
    sys.exit(1)

token = json.loads(bytes(blob[0]).decode('utf-8'))['value']

# Build cookies list with proper expiry format
cookies = []
for name, value, host, path, expiry, http, secure, same in rows:
    exp = float(expiry) if expiry else -1
    # Firefox stores expiry in seconds, but some may be in milliseconds
    if exp > 1e12:
        exp = exp / 1000
    cookies.append({
        'name': name,
        'value': value,
        'domain': host,
        'path': path or '/',
        'expires': exp if exp > 0 else -1,
        'httpOnly': bool(http),
        'secure': bool(secure),
        'sameSite': {0: 'None', 1: 'Lax', 2: 'Strict'}.get(same, 'Lax'),
    })

auth = {
    'token': token,
    'cookie': '; '.join(f"{c['name']}={c['value']}" for c in cookies),
    'cookies': cookies,
    'dumped_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
}

out = data_dir / 'auth.json'
out.write_text(json.dumps(auth, separators=(',', ':')))
out.chmod(0o600)

print(f"Refreshed: token={len(token)}B, cookies={len(cookies)}")
PY

log "Auth refreshed, restarting deepseek-web-api adapter..."
s6-svc -r /run/service/svc-deepseek-web-api 2>/dev/null || true
sleep 2

# Verify
if curl -fsS --max-time 10 http://127.0.0.1:8766/health >/dev/null 2>&1; then
    log "DeepSeek Web adapter healthy"
else
    log "WARNING: DeepSeek Web adapter health check failed"
fi

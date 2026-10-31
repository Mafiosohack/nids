#!/usr/bin/env bash
# ─────────────────────────────
#  NIDS — START
#  sudo ./start.sh
# ─────────────────────────────

if [ "$EUID" -ne 0 ]; then
  echo "[!] Run with sudo: sudo ./start.sh"
  exit 1
fi

cd "$(dirname "$0")"
mkdir -p logs   # gitignored, so absent on a fresh clone

echo "[*] Killing any previous NIDS processes..."
pkill -f "python.*main.py"    2>/dev/null || true
pkill -f "python.*live_ids"   2>/dev/null || true
sleep 1

echo "[*] Starting API server (main.py)..."
# Capture starts server-side: this script has no login, so the old
# `curl -X POST /control/start` got a 401 and capture never began.
# The attempt (and any failure reason) is in the audit log as actor "system".
NIDS_AUTOSTART_CAPTURE=1 venv/bin/python main.py > logs/main.log 2>&1 &
echo "    PID: $!"

# TLS by default: verify against our own certificate (created on first start),
# not the system CA store and not -k.
CERT=tls/nids-cert.pem
SCHEME=https
[ "${NIDS_INSECURE_HTTP:-}" = "1" ] && SCHEME=http
URL="$SCHEME://127.0.0.1:${NIDS_PORT:-8000}"

echo "[*] Waiting for API..."
for i in $(seq 1 30); do
  if [ "$SCHEME" = https ]; then
    [ -f "$CERT" ] && curl -sf --cacert "$CERT" "$URL/health" > /dev/null 2>&1 && break
  else
    curl -sf "$URL/health" > /dev/null 2>&1 && break
  fi
  sleep 0.5
done
grep -E "Dashboard|SHA-256|autostart|Capture" logs/main.log | sed 's/^/    /'

echo "[*] Starting ML sensor (live_ids_v2.py)..."
sudo venv/bin/python live_ids_v2.py > logs/live_ids.log 2>&1 &
echo "    PID: $!"

echo ""
echo "✓ NIDS is running"
echo "  Dashboard → $URL"
echo "  Logs      → logs/main.log  |  logs/live_ids.log"
echo "  To stop   → sudo ./stop.sh"
echo ""

# Open browser
sudo -u "$SUDO_USER" xdg-open "$URL" 2>/dev/null &

# Tail logs
tail -f logs/main.log logs/live_ids.log

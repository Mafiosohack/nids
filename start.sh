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

echo "[*] Killing any previous NIDS processes..."
pkill -f "python.*main.py"    2>/dev/null || true
pkill -f "python.*live_ids"   2>/dev/null || true
sleep 1

echo "[*] Starting API server (main.py)..."
sudo venv/bin/python main.py > logs/main.log 2>&1 &
echo "    PID: $!"

echo "[*] Waiting for API..."
for i in $(seq 1 20); do
  curl -sf http://127.0.0.1:8000/health > /dev/null 2>&1 && break
  sleep 0.5
done

echo "[*] Starting sniffer..."
curl -sf -X POST http://127.0.0.1:8000/control/start > /dev/null
echo "    Done."

echo "[*] Starting ML sensor (live_ids_v2.py)..."
sudo venv/bin/python live_ids_v2.py > logs/live_ids.log 2>&1 &
echo "    PID: $!"

echo ""
echo "✓ NIDS is running"
echo "  Dashboard → http://127.0.0.1:8000"
echo "  Logs      → logs/main.log  |  logs/live_ids.log"
echo "  To stop   → sudo ./stop.sh"
echo ""

# Open browser
sudo -u "$SUDO_USER" xdg-open http://127.0.0.1:8000 2>/dev/null &

# Tail logs
mkdir -p logs
tail -f logs/main.log logs/live_ids.log

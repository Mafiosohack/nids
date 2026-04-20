"""
REAL-TIME NETWORK INTRUSION DETECTION SYSTEM (NIDS)
main.py — Rule-Based Engine + FastAPI Server

Responsibilities:
  - Signature detection (NULL/XMAS/FIN/SYN scans)
  - Behavioral detection (port scan, SYN flood, brute force, DDoS, UDP amp)
  - Correlation engine (attack chain detection)
  - REST API (alerts, status, control)
  - Receives ML alerts from live_ids_v2.py via POST /alert

ML detection is intentionally NOT done here.
live_ids_v2.py handles flow-based ML and forwards alerts to /alert.
"""

import threading
import time
from collections import defaultdict, deque
from datetime import datetime
from typing import Deque, Dict, List, Optional

from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

try:
    from scapy.all import ICMP, IP, TCP, UDP, sniff
    SCAPY_AVAILABLE = True
except ImportError:
    SCAPY_AVAILABLE = False
    print("[WARN] Scapy not available. Packet capture disabled.")

# ─────────────────────────────────────────────
#  CONFIG  (tune these to your environment)
# ─────────────────────────────────────────────
NETWORK_INTERFACE = "ens37"

PORT_SCAN_THRESHOLD     = 5      # distinct ports within window → alert
PORT_SCAN_WINDOW        = 5      # seconds

STEALTH_SCAN_THRESHOLD  = 15     # distinct ports within window → alert
STEALTH_SCAN_WINDOW     = 60     # seconds

DDOS_PACKET_THRESHOLD   = 200    # packets within window → alert
DDOS_WINDOW             = 3      # seconds

SYN_FLOOD_THRESHOLD     = 100    # SYN packets without ACKs
SYN_FLOOD_WINDOW        = 5      # seconds
SYN_ACK_RATIO_LIMIT     = 5      # SYN:ACK ratio that indicates flood

BRUTE_FORCE_THRESHOLD   = 8      # attempts within window → alert
BRUTE_FORCE_WINDOW      = 60     # seconds
BRUTE_FORCE_PORTS       = {22, 21, 23, 25, 110, 143, 3306, 3389, 5432}

UDP_AMP_PORTS           = {53, 123, 1900, 11211, 19, 17}
UDP_AMP_THRESHOLD       = 50     # requests within window → alert
UDP_AMP_WINDOW          = 10     # seconds

MAX_ALERTS_STORED       = 500
CORRELATION_EVENT_LIMIT = 15

# ─────────────────────────────────────────────
#  GLOBAL STATE
# ─────────────────────────────────────────────
alerts_lock  = threading.Lock()
alerts: Deque[Dict] = deque(maxlen=MAX_ALERTS_STORED)

# Per-source trackers — all use defaultdict so keys are created on first access
scan_tracker       = defaultdict(lambda: {"ports": set(), "first_seen": None})
stealth_tracker    = defaultdict(lambda: {"ports": set(), "timestamps": []})
ddos_tracker       = defaultdict(lambda: {"timestamps": []})
syn_tracker        = defaultdict(lambda: {"syn": 0, "ack": 0})
bruteforce_tracker = defaultdict(lambda: {"attempts": 0, "timestamps": []})
udp_amp_tracker    = defaultdict(lambda: {"timestamps": []})

correlation_tracker = defaultdict(lambda: {"events": []})
correlation_lock    = threading.Lock()

system_stats = {
    "packets_captured": 0,
    "alerts_generated": 0,
    "start_time": None,
    "running": False,
}

sniffer_thread: Optional[threading.Thread] = None
sniffer_stop_event = threading.Event()

# ─────────────────────────────────────────────
#  ALERT ENGINE
# ─────────────────────────────────────────────
def _new_alert_id() -> int:
    """Thread-safe ID increment — caller must NOT hold alerts_lock."""
    with alerts_lock:
        system_stats["alerts_generated"] += 1
        return system_stats["alerts_generated"]


def generate_alert(
    alert_type: str,
    severity: str,
    src: str,
    dst: Optional[str],
    protocol: str,
    message: str,
    details: Optional[Dict] = None,
    _from_correlation: bool = False,
) -> Dict:
    alert_id = _new_alert_id()
    alert = {
        "id":        alert_id,
        "type":      alert_type,
        "severity":  severity,
        "src":       src,
        "dst":       dst,
        "protocol":  protocol,
        "message":   message,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "details":   details or {},
    }

    with alerts_lock:
        alerts.append(alert)

    print(f"[{severity.upper():8s}] {alert_type} → {message}")

    if not _from_correlation and src:
        with correlation_lock:
            correlation_tracker[src]["events"].append(alert_type)
        correlate_events(src)

    return alert

# ─────────────────────────────────────────────
#  CORRELATION ENGINE
# ─────────────────────────────────────────────
# Map of (frozenset of event types) → (alert type, message template)
CORRELATION_RULES: List[tuple] = [
    (
        {"Port Scan", "Brute Force Attempt"},
        "Attack Chain",
        "critical",
        "Port Scan followed by Brute Force",
    ),
    (
        {"Port Scan", "SYN Flood"},
        "Coordinated Attack",
        "critical",
        "Port Scan combined with SYN Flood",
    ),
    (
        {"DDoS", "SYN Flood"},
        "Volumetric Attack",
        "critical",
        "DDoS combined with SYN Flood",
    ),
    (
        {"ML Anomaly", "Port Scan"},
        "Recon + Anomaly",
        "critical",
        "ML anomaly combined with Port Scan",
    ),
]

def correlate_events(src: str):
    with correlation_lock:
        events = list(correlation_tracker[src]["events"])

    unique = set(events)

    # Check specific rules first
    for required_events, alert_type, severity, description in CORRELATION_RULES:
        if required_events.issubset(unique):
            generate_alert(
                alert_type, severity, src, None, "Multiple",
                f"{alert_type} from {src}: {description}",
                {"events": list(unique)},
                _from_correlation=True,
            )
            with correlation_lock:
                correlation_tracker[src]["events"].clear()
            return

    # Generic multi-vector rule
    if len(unique) >= 3:
        generate_alert(
            "Multi-Vector Attack", "critical", src, None, "Multiple",
            f"Multi-vector attack from {src}: {', '.join(unique)}",
            {"events": list(unique)},
            _from_correlation=True,
        )
        with correlation_lock:
            correlation_tracker[src]["events"].clear()
        return

    # Flush oversized event list to prevent memory growth
    if len(events) > CORRELATION_EVENT_LIMIT:
        with correlation_lock:
            correlation_tracker[src]["events"].clear()

# ─────────────────────────────────────────────
#  SIGNATURE DETECTION
# ─────────────────────────────────────────────
# Maps exact flag values → scan name
TCP_SIGNATURES = {
    0x00: "NULL Scan",
    0x01: "FIN Scan",
    0x03: "FIN+SYN Scan",
    0x02: "SYN Scan",
}

def check_tcp_signature(flags: int) -> Optional[str]:
    """Return a scan name for known malicious TCP flag combinations."""
    if flags in TCP_SIGNATURES:
        return TCP_SIGNATURES[flags]
    if flags & 0x29 == 0x29:   # FIN + PSH + URG
        return "XMAS Scan"
    return None

# ─────────────────────────────────────────────
#  DETECTION MODULES
# ─────────────────────────────────────────────
def detect_port_scan(src: str, dst: str, port: int):
    t = scan_tracker[src]
    now = time.time()

    if t["first_seen"] is None:
        t["first_seen"] = now

    # Reset window if expired
    if now - t["first_seen"] > PORT_SCAN_WINDOW:
        t["ports"].clear()
        t["first_seen"] = now
        return

    t["ports"].add(port)

    if len(t["ports"]) > PORT_SCAN_THRESHOLD:
        generate_alert(
            "Port Scan", "high", src, dst, "TCP",
            f"Port scan: {src} → {dst}, {len(t['ports'])} ports in {PORT_SCAN_WINDOW}s",
            {"ports_scanned": len(t["ports"])},
        )
        t["ports"].clear()
        t["first_seen"] = now


def detect_stealth_scan(src: str, port: int):
    """Detects slow/low-and-slow scans that evade the fast port-scan check."""
    t = stealth_tracker[src]
    now = time.time()

    t["ports"].add(port)
    t["timestamps"] = [ts for ts in t["timestamps"] if now - ts < STEALTH_SCAN_WINDOW]
    t["timestamps"].append(now)

    if len(t["ports"]) >= STEALTH_SCAN_THRESHOLD:
        generate_alert(
            "Stealth Scan", "medium", src, None, "TCP",
            f"Stealth scan: {src} probed {len(t['ports'])} ports over {STEALTH_SCAN_WINDOW}s",
            {"ports_scanned": len(t["ports"])},
        )
        t["ports"].clear()
        t["timestamps"].clear()


def detect_ddos(src: str, dst: str):
    t = ddos_tracker[src]
    now = time.time()

    t["timestamps"] = [ts for ts in t["timestamps"] if now - ts < DDOS_WINDOW]
    t["timestamps"].append(now)

    if len(t["timestamps"]) > DDOS_PACKET_THRESHOLD:
        generate_alert(
            "DDoS", "critical", src, dst, "Multiple",
            f"DDoS: {src} → {dst}, {len(t['timestamps'])} pkts in {DDOS_WINDOW}s",
            {"packet_count": len(t["timestamps"])},
        )
        t["timestamps"].clear()


def detect_syn_flood(src: str, dst: str):
    t = syn_tracker[src]
    t["syn"] += 1

    if t["syn"] > SYN_FLOOD_THRESHOLD:
        ratio = t["syn"] / max(t["ack"], 1)
        if ratio > SYN_ACK_RATIO_LIMIT:
            generate_alert(
                "SYN Flood", "critical", src, dst, "TCP",
                f"SYN flood: {src} → {dst}, {t['syn']} SYNs, ratio={ratio:.1f}",
                {"syn_count": t["syn"], "ack_count": t["ack"], "ratio": round(ratio, 2)},
            )
            t["syn"] = 0
            t["ack"] = 0


def detect_brute_force(src: str, dst: str, port: int):
    if port not in BRUTE_FORCE_PORTS:
        return

    t = bruteforce_tracker[src]
    now = time.time()

    t["timestamps"] = [ts for ts in t["timestamps"] if now - ts < BRUTE_FORCE_WINDOW]
    t["timestamps"].append(now)
    t["attempts"] = len(t["timestamps"])   # attempts = events in window

    if t["attempts"] >= BRUTE_FORCE_THRESHOLD:
        generate_alert(
            "Brute Force Attempt", "high", src, dst, "TCP",
            f"Brute force: {src} → {dst}:{port}, {t['attempts']} attempts in {BRUTE_FORCE_WINDOW}s",
            {"port": port, "attempt_count": t["attempts"]},
        )
        t["attempts"] = 0
        t["timestamps"].clear()


def detect_udp_amplification(src: str, dst: str, dst_port: int, pkt_len: int):
    if dst_port not in UDP_AMP_PORTS:
        return

    t = udp_amp_tracker[src]
    now = time.time()

    t["timestamps"] = [ts for ts in t["timestamps"] if now - ts < UDP_AMP_WINDOW]
    t["timestamps"].append(now)

    if len(t["timestamps"]) > UDP_AMP_THRESHOLD:
        generate_alert(
            "UDP Amplification", "high", src, dst, "UDP",
            f"UDP amp: {src} → {dst}:{dst_port}, {len(t['timestamps'])} reqs in {UDP_AMP_WINDOW}s",
            {"dst_port": dst_port, "request_count": len(t["timestamps"]), "pkt_size": pkt_len},
        )
        t["timestamps"].clear()

# ─────────────────────────────────────────────
#  PACKET PROCESSOR
# ─────────────────────────────────────────────
def process_packet(packet):
    system_stats["packets_captured"] += 1

    if not packet.haslayer(IP):
        return

    src = packet[IP].src
    dst = packet[IP].dst

    # DDoS check runs on every IP packet
    detect_ddos(src, dst)

    if packet.haslayer(TCP):
        tcp   = packet[TCP]
        flags = int(tcp.flags)
        dport = tcp.dport

        # Signature detection
        sig = check_tcp_signature(flags)
        if sig:
            generate_alert(
                "TCP Signature", "low", src, dst, "TCP",
                f"TCP signature '{sig}' from {src} → {dst}:{dport}",
                {"signature": sig, "flags": flags, "dst_port": dport},
            )

        if flags & 0x02:  # SYN
            detect_port_scan(src, dst, dport)
            detect_stealth_scan(src, dport)
            detect_syn_flood(src, dst)

        if flags & 0x10:  # ACK
            syn_tracker[src]["ack"] += 1

        detect_brute_force(src, dst, dport)

    elif packet.haslayer(UDP):
        udp = packet[UDP]
        detect_udp_amplification(src, dst, udp.dport, len(packet))

    # ICMP: currently just counted via DDoS tracker above
    # Extend here if you want ICMP-specific detection (ping flood, smurf, etc.)

# ─────────────────────────────────────────────
#  SNIFFER THREAD
# ─────────────────────────────────────────────
def _sniffer_worker():
    print(f"[SNIFFER] Capturing on interface: {NETWORK_INTERFACE}")
    sniff(
        iface=NETWORK_INTERFACE,
        prn=process_packet,
        store=False,
        stop_filter=lambda _: sniffer_stop_event.is_set(),
    )
    print("[SNIFFER] Stopped.")

# ─────────────────────────────────────────────
#  FASTAPI APPLICATION
# ─────────────────────────────────────────────
app = FastAPI(
    title="NIDS API",
    description="Real-Time Network Intrusion Detection System — Rule-Based Engine",
    version="3.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ExternalAlert(BaseModel):
    src: str
    dst: Optional[str] = None
    proto: Optional[str] = "unknown"
    message: Optional[str] = "Suspicious activity"
    severity: Optional[str] = "high"
    alert_type: Optional[str] = "ML Anomaly"
    details: Optional[Dict] = {}


# ── Endpoints ────────────────────────────────

@app.get("/health", tags=["System"])
def health():
    return {
        "status":       "ok",
        "scapy":        SCAPY_AVAILABLE,
        "timestamp":    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


@app.get("/status", tags=["System"])
def status():
    uptime = None
    if system_stats["start_time"]:
        uptime = round(time.time() - system_stats["start_time"], 1)
    return {
        "running":          system_stats["running"],
        "packets_captured": system_stats["packets_captured"],
        "alerts_generated": system_stats["alerts_generated"],
        "alerts_stored":    len(alerts),
        "uptime_seconds":   uptime,
        "interface":        NETWORK_INTERFACE,
    }


@app.get("/alerts", tags=["Alerts"])
def get_alerts(
    limit:    int = Query(50,   ge=1, le=MAX_ALERTS_STORED),
    severity: Optional[str] = Query(None, description="Filter by severity: low/medium/high/critical"),
    alert_type: Optional[str] = Query(None, description="Filter by alert type"),
):
    with alerts_lock:
        result = list(alerts)

    if severity:
        result = [a for a in result if a["severity"].lower() == severity.lower()]
    if alert_type:
        result = [a for a in result if a["type"].lower() == alert_type.lower()]

    return result[-limit:]


@app.delete("/alerts", tags=["Alerts"])
def clear_alerts():
    """Clear all stored alerts. Called by the dashboard 'Clear Alerts' button."""
    with alerts_lock:
        count = len(alerts)
        alerts.clear()
        system_stats["alerts_generated"] = 0
    with correlation_lock:
        correlation_tracker.clear()
    print(f"[NIDS] Cleared {count} alerts.")
    return {"status": "ok", "cleared": count}


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def serve_dashboard():
    """
    Serve the NIDS dashboard HTML directly from the API server.
    This avoids browser CORS blocks that happen when opening the file via file://.
    Place nids_dashboard.html next to main.py, then open http://127.0.0.1:8000
    """
    dashboard = Path("nids_dashboard.html")
    if not dashboard.exists():
        return HTMLResponse(
            content="""
            <html><body style='font-family:monospace;background:#0c1118;color:#64748b;padding:40px'>
            <h2 style='color:#ef4444'>Dashboard Not Found</h2>
            <p>Place <code>nids_dashboard.html</code> in the same directory as <code>main.py</code></p>
            <p>Then refresh this page.</p>
            </body></html>
            """,
            status_code=404,
        )
    return HTMLResponse(content=dashboard.read_text(encoding="utf-8"))


@app.post("/alert", tags=["Alerts"])
def receive_external_alert(payload: ExternalAlert):
    """
    Accepts alerts from external sensors (e.g. live_ids_v2.py ML sensor).
    Feeds them into the same alert store and correlation engine.
    """
    alert = generate_alert(
        alert_type=payload.alert_type,
        severity=payload.severity,
        src=payload.src,
        dst=payload.dst,
        protocol=payload.proto,
        message=payload.message,
        details=payload.details,
    )
    return {"status": "ok", "alert_id": alert["id"]}


@app.post("/control/start", tags=["Control"])
def control_start():
    global sniffer_thread

    if system_stats["running"]:
        raise HTTPException(400, "Sniffer already running.")
    if not SCAPY_AVAILABLE:
        raise HTTPException(503, "Scapy not installed. Cannot capture packets.")

    sniffer_stop_event.clear()
    sniffer_thread = threading.Thread(target=_sniffer_worker, daemon=True)
    sniffer_thread.start()
    system_stats["running"]    = True
    system_stats["start_time"] = time.time()

    return {"status": "started", "interface": NETWORK_INTERFACE}


@app.post("/control/stop", tags=["Control"])
def control_stop():
    if not system_stats["running"]:
        raise HTTPException(400, "Sniffer is not running.")

    sniffer_stop_event.set()
    system_stats["running"] = False

    return {"status": "stopped"}


# ─────────────────────────────────────────────
#  STARTUP
# ─────────────────────────────────────────────
@app.on_event("startup")
def on_startup():
    print("[NIDS] main.py ready. POST /control/start to begin capture.")
    print("[NIDS] ML alerts expected from live_ids_v2.py via POST /alert")


# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)

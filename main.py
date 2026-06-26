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

import hashlib
import hmac
import json
import secrets
import sqlite3
import threading
import time
from collections import defaultdict, deque
from contextlib import contextmanager
from datetime import datetime, timedelta
from typing import Deque, Dict, List, Optional

from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
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
#  AUTH CONFIG
# ─────────────────────────────────────────────
# Sensor pre-shared key — must match SENSOR_API_KEY in live_ids_v2.py
SENSOR_API_KEY = "sensor-key-change-me-in-production"

# Session token expiry (hours)
TOKEN_EXPIRY_HOURS = 8

# SQLite database path
DB_PATH = Path("nids_users.db")

# ─────────────────────────────────────────────
#  ROLES AND PERMISSIONS
# ─────────────────────────────────────────────
# Three roles — each is a frozenset of allowed actions.
ROLES = {
    "admin": frozenset([
        "view_alerts",
        "export_alerts",
        "delete_alerts",
        "control_sniffer",
        "manage_users",
    ]),
    "analyst": frozenset([
        "view_alerts",
        "export_alerts",
    ]),
    "viewer": frozenset([
        "view_alerts",
    ]),
}

# ─────────────────────────────────────────────
#  DATABASE  (SQLite — persistent across restarts)
# ─────────────────────────────────────────────
db_lock = threading.Lock()

@contextmanager
def get_db():
    """Thread-safe SQLite connection context manager."""
    with db_lock:
        conn = sqlite3.connect(str(DB_PATH))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()


def init_db():
    """
    Create tables and seed a default admin account if the DB is new.
    Safe to call on every startup — uses CREATE IF NOT EXISTS.
    """
    with get_db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                username   TEXT UNIQUE NOT NULL,
                password   TEXT NOT NULL,
                role       TEXT NOT NULL DEFAULT 'viewer',
                created_at TEXT NOT NULL,
                last_login TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                token      TEXT PRIMARY KEY,
                username   TEXT NOT NULL,
                role       TEXT NOT NULL,
                expires_at TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_sessions_token
            ON sessions(token)
        """)

        # Seed default admin only if no users exist yet
        count = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
        if count == 0:
            conn.execute(
                "INSERT INTO users (username, password, role, created_at) VALUES (?,?,?,?)",
                (
                    "admin",
                    _hash_password("nids@admin123"),
                    "admin",
                    datetime.now().isoformat(),
                ),
            )
            print("[AUTH] Default admin created. Username: admin | Password: nids@admin123")
            print("[AUTH] Change the password immediately via POST /auth/change-password")


def _hash_password(password: str) -> str:
    """SHA-256 hash of password. Simple but sufficient for a PoC."""
    return hashlib.sha256(password.encode()).hexdigest()


def db_get_user(username: str) -> Optional[sqlite3.Row]:
    with get_db() as conn:
        return conn.execute(
            "SELECT * FROM users WHERE username = ?", (username,)
        ).fetchone()


def db_create_session(username: str, role: str) -> str:
    token = secrets.token_hex(32)
    expires = (datetime.now() + timedelta(hours=TOKEN_EXPIRY_HOURS)).isoformat()
    with get_db() as conn:
        conn.execute(
            "INSERT INTO sessions (token, username, role, expires_at, created_at) VALUES (?,?,?,?,?)",
            (token, username, role, expires, datetime.now().isoformat()),
        )
        # Update last_login
        conn.execute(
            "UPDATE users SET last_login = ? WHERE username = ?",
            (datetime.now().isoformat(), username),
        )
    return token


def db_get_session(token: str) -> Optional[sqlite3.Row]:
    with get_db() as conn:
        return conn.execute(
            "SELECT * FROM sessions WHERE token = ?", (token,)
        ).fetchone()


def db_revoke_session(token: str):
    with get_db() as conn:
        conn.execute("DELETE FROM sessions WHERE token = ?", (token,))


def db_purge_expired_sessions():
    with get_db() as conn:
        conn.execute(
            "DELETE FROM sessions WHERE expires_at < ?",
            (datetime.now().isoformat(),),
        )


def db_list_users() -> list:
    with get_db() as conn:
        rows = conn.execute(
            "SELECT id, username, role, created_at, last_login FROM users ORDER BY id"
        ).fetchall()
    return [dict(r) for r in rows]

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
#  AUTH HELPERS
# ─────────────────────────────────────────────
bearer_scheme = HTTPBearer(auto_error=False)


def _get_validated_session(token: str) -> Optional[dict]:
    """Return session dict if token exists and is not expired, else None."""
    db_purge_expired_sessions()
    row = db_get_session(token)
    if row is None:
        return None
    if datetime.fromisoformat(row["expires_at"]) < datetime.now():
        db_revoke_session(token)
        return None
    return dict(row)


def require_auth(
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
) -> dict:
    """
    Validates bearer token. Returns session dict with keys:
      username, role, expires_at
    Raises 401 if missing / invalid / expired.
    """
    if credentials is None:
        raise HTTPException(
            status_code=401,
            detail="Authentication required.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    session = _get_validated_session(credentials.credentials)
    if session is None:
        raise HTTPException(
            status_code=401,
            detail="Invalid or expired token. Please log in again.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return session


def require_permission(permission: str):
    """
    Returns a FastAPI dependency that checks the session has a specific permission.
    Usage:  user = Depends(require_permission("delete_alerts"))
    """
    def _check(session: dict = Depends(require_auth)) -> dict:
        role = session.get("role", "viewer")
        if permission not in ROLES.get(role, frozenset()):
            raise HTTPException(
                status_code=403,
                detail=f"Role '{role}' does not have permission: {permission}",
            )
        return session
    return _check


def require_sensor(request: Request):
    """
    POST /alert only — accepts the sensor pre-shared key.
    Intentionally does NOT accept user tokens to keep sensor
    access completely separate from human access.
    """
    sensor_key = request.headers.get("X-Sensor-Key", "")
    if hmac.compare_digest(sensor_key, SENSOR_API_KEY):
        return "sensor"
    raise HTTPException(
        status_code=401,
        detail="Valid X-Sensor-Key header required.",
    )


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


class LoginRequest(BaseModel):
    username: str
    password: str

class ChangePasswordRequest(BaseModel):
    current_password: str
    new_password: str

class CreateUserRequest(BaseModel):
    username: str
    password: str
    role: str = "viewer"

class UpdateRoleRequest(BaseModel):
    role: str

class ExternalAlert(BaseModel):
    src: str
    dst: Optional[str] = None
    proto: Optional[str] = "unknown"
    message: Optional[str] = "Suspicious activity"
    severity: Optional[str] = "high"
    alert_type: Optional[str] = "ML Anomaly"
    details: Optional[Dict] = {}


# ── Auth Endpoints ────────────────────────────

@app.post("/auth/login", tags=["Auth"])
def login(payload: LoginRequest):
    """Exchange username + password for a bearer token."""
    user = db_get_user(payload.username)
    if user is None or not hmac.compare_digest(
        user["password"], _hash_password(payload.password)
    ):
        raise HTTPException(status_code=401, detail="Invalid username or password.")

    token = db_create_session(payload.username, user["role"])
    print(f"[AUTH] Login: {payload.username} (role={user['role']})")
    return {
        "token":      token,
        "username":   payload.username,
        "role":       user["role"],
        "expires_in": TOKEN_EXPIRY_HOURS * 3600,
    }


@app.post("/auth/logout", tags=["Auth"])
def logout(
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
    session: dict = Depends(require_auth),
):
    db_revoke_session(credentials.credentials)
    print(f"[AUTH] Logout: {session['username']}")
    return {"status": "logged out"}


@app.post("/auth/change-password", tags=["Auth"])
def change_password(
    payload: ChangePasswordRequest,
    session: dict = Depends(require_auth),
):
    """Any logged-in user can change their own password."""
    user = db_get_user(session["username"])
    if not hmac.compare_digest(user["password"], _hash_password(payload.current_password)):
        raise HTTPException(status_code=401, detail="Current password is incorrect.")
    if len(payload.new_password) < 8:
        raise HTTPException(status_code=400, detail="New password must be at least 8 characters.")
    with get_db() as conn:
        conn.execute(
            "UPDATE users SET password = ? WHERE username = ?",
            (_hash_password(payload.new_password), session["username"]),
        )
    print(f"[AUTH] Password changed: {session['username']}")
    return {"status": "ok"}


# ── User Management (admin only) ──────────────

@app.get("/users", tags=["User Management"])
def list_users(session: dict = Depends(require_permission("manage_users"))):
    """List all users. Admin only."""
    return db_list_users()


@app.post("/users", tags=["User Management"])
def create_user(
    payload: CreateUserRequest,
    session: dict = Depends(require_permission("manage_users")),
):
    """Create a new user. Admin only."""
    if payload.role not in ROLES:
        raise HTTPException(400, f"Invalid role. Must be one of: {list(ROLES.keys())}")
    if len(payload.password) < 8:
        raise HTTPException(400, "Password must be at least 8 characters.")
    if db_get_user(payload.username):
        raise HTTPException(409, f"Username '{payload.username}' already exists.")

    with get_db() as conn:
        conn.execute(
            "INSERT INTO users (username, password, role, created_at) VALUES (?,?,?,?)",
            (payload.username, _hash_password(payload.password),
             payload.role, datetime.now().isoformat()),
        )
    print(f"[AUTH] User created: {payload.username} (role={payload.role}) by {session['username']}")
    return {"status": "ok", "username": payload.username, "role": payload.role}


@app.delete("/users/{username}", tags=["User Management"])
def delete_user(
    username: str,
    session: dict = Depends(require_permission("manage_users")),
):
    """Delete a user. Admin only. Cannot delete yourself."""
    if username == session["username"]:
        raise HTTPException(400, "Cannot delete your own account.")
    if not db_get_user(username):
        raise HTTPException(404, f"User '{username}' not found.")
    with get_db() as conn:
        conn.execute("DELETE FROM users WHERE username = ?", (username,))
        conn.execute("DELETE FROM sessions WHERE username = ?", (username,))
    print(f"[AUTH] User deleted: {username} by {session['username']}")
    return {"status": "ok"}


@app.patch("/users/{username}/role", tags=["User Management"])
def update_role(
    username: str,
    payload: UpdateRoleRequest,
    session: dict = Depends(require_permission("manage_users")),
):
    """Change a user's role. Admin only."""
    if payload.role not in ROLES:
        raise HTTPException(400, f"Invalid role. Must be one of: {list(ROLES.keys())}")
    if not db_get_user(username):
        raise HTTPException(404, f"User '{username}' not found.")
    with get_db() as conn:
        conn.execute(
            "UPDATE users SET role = ? WHERE username = ?",
            (payload.role, username),
        )
        # Revoke existing sessions — force re-login with new role
        conn.execute("DELETE FROM sessions WHERE username = ?", (username,))
    print(f"[AUTH] Role updated: {username} → {payload.role} by {session['username']}")
    return {"status": "ok", "username": username, "role": payload.role}


# ── System Endpoints ──────────────────────────

@app.get("/health", tags=["System"])
def health():
    # Public — no auth required
    return {
        "status":    "ok",
        "scapy":     SCAPY_AVAILABLE,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


@app.get("/status", tags=["System"])
def status(session: dict = Depends(require_auth)):
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
        "user":             session["username"],
        "role":             session["role"],
    }


# ── Model Endpoint ────────────────────────────

@app.get("/model/info", tags=["System"])
def model_info(session: dict = Depends(require_auth)):
    """Serve the ML model's honestly-evaluated metrics for the dashboard.

    Reads the metadata written by train_random_forest_v2.py so the UI reflects
    real KDDTest+ performance instead of hardcoded placeholder numbers.
    """
    meta_path = Path("models/rf_live_features.json")
    if not meta_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Model metadata not found. Run: python train_random_forest_v2.py",
        )
    try:
        with open(meta_path) as f:
            meta = json.load(f)
    except (ValueError, OSError) as e:
        raise HTTPException(status_code=500, detail=f"Could not read metadata: {e}")

    return {
        "models":             meta.get("models", []),
        "feature_importance": meta.get("feature_importance", []),
        "decision_threshold": meta.get("decision_threshold"),
        "trained_on":         meta.get("trained_on"),
        "evaluated_on":       meta.get("evaluated_on"),
        "test_samples":       meta.get("test_samples"),
    }


# ── Alert Endpoints ───────────────────────────

@app.get("/alerts", tags=["Alerts"])
def get_alerts(
    limit:      int = Query(50, ge=1, le=MAX_ALERTS_STORED),
    severity:   Optional[str] = Query(None),
    alert_type: Optional[str] = Query(None),
    session: dict = Depends(require_permission("view_alerts")),
):
    with alerts_lock:
        result = list(alerts)

    if severity:
        result = [a for a in result if a["severity"].lower() == severity.lower()]
    if alert_type:
        result = [a for a in result if a["type"].lower() == alert_type.lower()]

    return result[-limit:]


@app.delete("/alerts", tags=["Alerts"])
def clear_alerts(session: dict = Depends(require_permission("delete_alerts"))):
    """Admin only."""
    with alerts_lock:
        count = len(alerts)
        alerts.clear()
        system_stats["alerts_generated"] = 0
    with correlation_lock:
        correlation_tracker.clear()
    print(f"[NIDS] {session['username']} cleared {count} alerts.")
    return {"status": "ok", "cleared": count}


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def serve_dashboard():
    # Public — HTML must load before login can happen
    dashboard = Path("nids_dashboard.html")
    if not dashboard.exists():
        return HTMLResponse(
            content="""
            <html><body style='font-family:monospace;background:#0c1118;color:#64748b;padding:40px'>
            <h2 style='color:#ef4444'>Dashboard Not Found</h2>
            <p>Place <code>nids_dashboard.html</code> next to <code>main.py</code></p>
            </body></html>
            """,
            status_code=404,
        )
    return HTMLResponse(content=dashboard.read_text(encoding="utf-8"))


@app.post("/alert", tags=["Alerts"])
def receive_external_alert(
    payload: ExternalAlert,
    sender: str = Depends(require_sensor),
):
    """Sensor-only endpoint. Requires X-Sensor-Key header."""
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


# ── Control Endpoints (admin only) ────────────

@app.post("/control/start", tags=["Control"])
def control_start(session: dict = Depends(require_permission("control_sniffer"))):
    global sniffer_thread

    if system_stats["running"]:
        raise HTTPException(400, "Sniffer already running.")
    if not SCAPY_AVAILABLE:
        raise HTTPException(503, "Scapy not installed.")

    sniffer_stop_event.clear()
    sniffer_thread = threading.Thread(target=_sniffer_worker, daemon=True)
    sniffer_thread.start()
    system_stats["running"]    = True
    system_stats["start_time"] = time.time()
    print(f"[NIDS] Sniffer started by {session['username']}")

    return {"status": "started", "interface": NETWORK_INTERFACE}


@app.post("/control/stop", tags=["Control"])
def control_stop(session: dict = Depends(require_permission("control_sniffer"))):
    if not system_stats["running"]:
        raise HTTPException(400, "Sniffer is not running.")

    sniffer_stop_event.set()
    system_stats["running"] = False
    print(f"[NIDS] Sniffer stopped by {session['username']}")

    return {"status": "stopped"}


# ─────────────────────────────────────────────
#  STARTUP
# ─────────────────────────────────────────────
@app.on_event("startup")
def on_startup():
    init_db()
    print("[NIDS] main.py ready. POST /control/start to begin capture.")
    print("[NIDS] ML alerts expected from live_ids_v2.py via POST /alert")


# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)

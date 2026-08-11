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
import math
import os
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
    from scapy.all import ARP, DNS, ICMP, IP, TCP, UDP, sniff
    SCAPY_AVAILABLE = True
except ImportError:
    SCAPY_AVAILABLE = False
    ARP = DNS = ICMP = IP = TCP = UDP = None  # referenced defensively below
    print("[WARN] Scapy not available. Packet capture disabled.")

# ─────────────────────────────────────────────
#  CONFIG  (tune these to your environment)
# ─────────────────────────────────────────────
# Capture interface. Override per-host with NIDS_IFACE — the lab default (ens37)
# only exists on the Ubuntu sensor VM, so a hardcoded value silently captures
# nothing anywhere else. Empty/"auto" lets scapy pick its default interface.
NETWORK_INTERFACE = os.environ.get("NIDS_IFACE", "ens37").strip()

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
#  POST-COMPROMISE DETECTION  (the "attacker is inside" layer)
# ─────────────────────────────────────────────
# What counts as "our" network. Anything else is external/Internet.
INTERNAL_PREFIXES = ("192.168.", "10.", "172.16.", "172.17.", "172.18.",
                     "172.19.", "172.20.", "172.21.", "172.22.", "172.23.",
                     "172.24.", "172.25.", "172.26.", "172.27.", "172.28.",
                     "172.29.", "172.30.", "172.31.")

# Breach / foothold: a source that brute-forced an auth port and then
# successfully established a session on that same host:port = they got in.
BREACH_MEMORY_SEC       = 180    # how long a brute-forced target stays "hot"

# Lateral movement: one internal host reaching many internal hosts on admin ports.
LATERAL_PORTS           = {22, 23, 135, 139, 445, 1433, 3306, 3389, 5432,
                           5985, 5986, 5900}
LATERAL_THRESHOLD       = 6      # distinct internal targets → alert
LATERAL_WINDOW          = 120    # seconds

# Data exfiltration (network): sustained outbound bytes from an internal host
# to an external IP. This is OUR packet-based detector (separate from the
# cloud VPC-flow-log one in cloud_log_sensor.py).
EXFIL_BYTES_THRESHOLD   = 50_000_000   # 50 MB internal→external in window
EXFIL_WINDOW            = 300          # seconds

# C2 beaconing: repeated, regularly-timed outbound connections to one external IP.
BEACON_MIN_HITS         = 6      # connections needed to judge periodicity
BEACON_WINDOW           = 900    # seconds of history kept per (src,dst)
BEACON_MAX_CV           = 0.25   # coeff. of variation of intervals below this = regular
BEACON_MIN_INTERVAL     = 2.0    # ignore bursts faster than this (not a beacon)

# Reverse shell: an internal host initiating an OUTBOUND session on a port that
# is not normal client traffic — the hallmark of a callback to the attacker.
COMMON_OUTBOUND_PORTS   = {80, 443, 53, 123, 22, 25, 587, 465, 993, 995,
                           110, 143, 8080, 8443, 3128}

# Data staging (collection): one internal host pulling large data FROM many
# internal hosts before exfiltration.
STAGING_SRC_THRESHOLD   = 5              # distinct internal sources feeding it
STAGING_BYTES_THRESHOLD = 20_000_000     # total bytes aggregated in window
STAGING_WINDOW          = 300            # seconds

# Handshake table housekeeping (pending SYNs waiting for SYN-ACK).
HANDSHAKE_TTL_SEC       = 30
HANDSHAKE_MAX_ENTRIES   = 20000

# ─────────────────────────────────────────────
#  PROTOCOL / TRUST-BOUNDARY ATTACKS
#  Pentesters abuse a protocol's implicit trust rather than force the door.
# ─────────────────────────────────────────────
# Cloud instance metadata endpoints (IMDS). A workload tricked via SSRF into
# hitting these can read the instance's IAM role credentials — the classic
# cloud trust-boundary attack. Link-local, so packet capture sees it even
# though AWS VPC Flow Logs do NOT (that's why this lives in the packet sensor).
METADATA_IPS            = {"169.254.169.254", "fd00:ec2::254",
                          "100.100.100.200"}   # AWS/GCP/Azure, Alibaba
IMDS_BURST_THRESHOLD    = 6      # rapid metadata hits = credential harvesting
IMDS_WINDOW             = 20     # seconds

# DNS tunneling: DNS is almost always allowed egress, so it becomes a covert
# channel for data exfil / C2. We flag sustained long / high-entropy queries.
DNS_WINDOW              = 60     # seconds
DNS_TUNNEL_THRESHOLD    = 15     # suspicious queries to one parent domain → alert
DNS_LONG_LABEL          = 45     # a single DNS label longer than this is odd
DNS_QNAME_LEN           = 60     # total query-name length considered long
DNS_ENTROPY_MIN         = 3.6    # Shannon bits/char of the subdomain (random-looking)
DNS_SUSPICIOUS_QTYPES   = {10, 16, 251, 252}   # NULL, TXT, IXFR, AXFR

# ARP spoofing / MITM: layer-2 has no authentication. One IP suddenly claimed
# by a different MAC = poisoning / man-in-the-middle positioning.
ARP_COOLDOWN            = 60     # seconds between repeat alerts per IP

# ─────────────────────────────────────────────
#  ADAPTIVE BASELINING + ASSET INVENTORY
#  Closes the evasions a red-teamer uses against static per-source thresholds.
# ─────────────────────────────────────────────
# Optional asset inventory (known IP↔MAC + authorised hosts). When present it
# gives ARP detection a TRUST ANCHOR (so we know which MAC is the impostor,
# not just that one changed) and enables rogue-device detection.
ASSETS_PATH             = Path(os.environ.get("NIDS_ASSETS", "assets.json"))

# Distributed scan: many sources each probing one destination (each staying
# under the per-source port-scan threshold) — a botnet / coordinated sweep.
DISTRIBUTED_WINDOW      = 20     # seconds
DISTRIBUTED_SRC_MIN     = 4      # distinct scanning sources → alert
DISTRIBUTED_PORTS_MIN   = 3      # ports a source must hit to count as "scanning"

# Slow / low-and-slow scan: distinct ports probed by one source over a LONG
# horizon, defeating the short stealth-scan window.
SLOWSCAN_WINDOW         = 1800   # 30 minutes
SLOWSCAN_THRESHOLD      = 12     # distinct ports over the long horizon → alert

# Adaptive baseline: per-source connection-rate EWMA. Flags a source whose
# activity spikes far above ITS OWN learned normal — no fixed threshold.
BASELINE_BUCKET         = 10     # seconds per rate sample
BASELINE_ALPHA          = 0.3    # EWMA smoothing (higher = adapts faster)
BASELINE_MIN_SAMPLES    = 6      # learn this many buckets before flagging
BASELINE_K              = 4.0    # sigmas above the mean to consider anomalous
BASELINE_MIN_COUNT      = 20     # ignore trivially small buckets (noise floor)

# ─────────────────────────────────────────────
#  AUTH CONFIG
# ─────────────────────────────────────────────
_DEFAULT_SENSOR_KEY = "sensor-key-change-me-in-production"
_DEFAULT_ADMIN_PW   = "nids@admin123"

# Sensor pre-shared key — override in production via env NIDS_SENSOR_KEY
# (must match SENSOR_API_KEY in the sensors). Falls back to the shared default.
SENSOR_API_KEY = os.environ.get("NIDS_SENSOR_KEY", _DEFAULT_SENSOR_KEY)

# Initial admin password used ONLY to seed a brand-new DB. Override via env
# NIDS_ADMIN_PASSWORD so a fresh deployment never ships a publicly-known password.
ADMIN_SEED_PASSWORD = os.environ.get("NIDS_ADMIN_PASSWORD", _DEFAULT_ADMIN_PW)

# Password hashing (PBKDF2-HMAC-SHA256). Stdlib only; salted + iterated.
PBKDF2_ITERATIONS = 200_000

# CORS: restrict to the dashboard's own origin(s). Override via env
# NIDS_ALLOWED_ORIGINS (comma-separated) for remote dashboards.
ALLOWED_ORIGINS = os.environ.get(
    "NIDS_ALLOWED_ORIGINS",
    "http://127.0.0.1:8000,http://localhost:8000",
).split(",")

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
        # Persistent alert store — survives restarts (was in-memory only).
        conn.execute("""
            CREATE TABLE IF NOT EXISTS alerts (
                id         INTEGER PRIMARY KEY,
                type       TEXT NOT NULL,
                severity   TEXT NOT NULL,
                src        TEXT,
                dst        TEXT,
                protocol   TEXT,
                message    TEXT,
                timestamp  TEXT NOT NULL,
                details    TEXT,
                created_at REAL NOT NULL
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_alerts_created ON alerts(created_at)")

        # Seed admin only if no users exist yet.
        count = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
        if count == 0:
            conn.execute(
                "INSERT INTO users (username, password, role, created_at) VALUES (?,?,?,?)",
                ("admin", _hash_password(ADMIN_SEED_PASSWORD), "admin",
                 datetime.now().isoformat()),
            )
            if ADMIN_SEED_PASSWORD == _DEFAULT_ADMIN_PW:
                print("[AUTH][WARN] Seeded admin with the PUBLIC default password "
                      f"'{_DEFAULT_ADMIN_PW}'. Change it now (POST /auth/change-password) "
                      "or set NIDS_ADMIN_PASSWORD before first run.")
            else:
                print("[AUTH] Seeded admin from NIDS_ADMIN_PASSWORD.")


def _hash_password(password: str) -> str:
    """Salted PBKDF2-HMAC-SHA256. Format: pbkdf2$sha256$<iters>$<salt>$<hash>."""
    salt = secrets.token_bytes(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, PBKDF2_ITERATIONS)
    return f"pbkdf2$sha256${PBKDF2_ITERATIONS}${salt.hex()}${dk.hex()}"


def _verify_password(password: str, stored: str) -> bool:
    """Constant-time verify. Supports current PBKDF2 hashes AND legacy unsalted
    SHA-256 hashes (64 hex chars) so pre-existing DBs keep working until the
    user's next login transparently upgrades them (see login())."""
    if stored.startswith("pbkdf2$"):
        try:
            _, _algo, iters, salt_hex, hash_hex = stored.split("$")
            dk = hashlib.pbkdf2_hmac(
                "sha256", password.encode(), bytes.fromhex(salt_hex), int(iters))
            return hmac.compare_digest(dk.hex(), hash_hex)
        except (ValueError, TypeError):
            return False
    # Legacy: unsalted SHA-256 hex digest.
    return hmac.compare_digest(hashlib.sha256(password.encode()).hexdigest(), stored)


def _is_legacy_hash(stored: str) -> bool:
    return not stored.startswith("pbkdf2$")


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


# ── Alert persistence (durable across restarts) ──
ALERT_RETENTION = 50_000   # keep at most this many alerts on disk


def db_insert_alert(alert: Dict):
    """Persist one alert. Details are stored as JSON text."""
    with get_db() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO alerts "
            "(id, type, severity, src, dst, protocol, message, timestamp, details, created_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?)",
            (alert["id"], alert["type"], alert["severity"], alert.get("src"),
             alert.get("dst"), alert.get("protocol"), alert.get("message"),
             alert["timestamp"], json.dumps(alert.get("details", {})), time.time()),
        )


def db_load_recent_alerts(limit: int) -> List[Dict]:
    """Load the most recent alerts (oldest→newest) to warm the in-memory cache."""
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM alerts ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    out = []
    for r in reversed(rows):
        d = dict(r)
        try:
            d["details"] = json.loads(d.get("details") or "{}")
        except (ValueError, TypeError):
            d["details"] = {}
        d.pop("created_at", None)
        out.append(d)
    return out


def db_max_alert_id() -> int:
    with get_db() as conn:
        row = conn.execute("SELECT MAX(id) FROM alerts").fetchone()
    return int(row[0]) if row and row[0] is not None else 0


def db_alert_count() -> int:
    with get_db() as conn:
        return int(conn.execute("SELECT COUNT(*) FROM alerts").fetchone()[0])


def db_prune_alerts():
    """Enforce ALERT_RETENTION by dropping the oldest rows beyond the cap."""
    with get_db() as conn:
        conn.execute(
            "DELETE FROM alerts WHERE id <= "
            "(SELECT MAX(id) FROM alerts) - ?", (ALERT_RETENTION,),
        )


def db_clear_alerts():
    with get_db() as conn:
        conn.execute("DELETE FROM alerts")


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

# ── Post-compromise trackers ──────────────────
# src → {(dst, port): last_bruteforce_ts}  — targets this src recently brute-forced
breach_hot_targets  = defaultdict(dict)
# Pending TCP handshakes: (initiator, target, target_port) → syn_ts
handshake_tracker: Dict[tuple, float] = {}
# internal src → {"dests": set(), "timestamps": []}
lateral_tracker     = defaultdict(lambda: {"dests": set(), "timestamps": []})
# internal src → {"bytes": int, "timestamps": [], "top_dst": str}
exfil_tracker       = defaultdict(lambda: {"bytes": 0, "timestamps": [], "top": None})
# src → dst(external) → [connection timestamps]
beacon_tracker      = defaultdict(lambda: defaultdict(list))
# internal dst → {"srcs": {src: bytes}, "timestamps": []}
staging_tracker     = defaultdict(lambda: {"srcs": defaultdict(int), "timestamps": []})
# de-dupe: don't re-fire the same post-compromise verdict every packet
_recent_pc_alerts: Dict[str, float] = {}
pc_lock             = threading.Lock()

# ── Protocol / trust-boundary trackers ────────
metadata_tracker    = defaultdict(list)                    # src → [ts] of IMDS hits
dns_tracker         = defaultdict(lambda: {"timestamps": []})  # (src,parent) → ts list
arp_ip_mac: Dict[str, str] = {}                            # ip → last-seen MAC

# ── Asset inventory + adaptive-baseline trackers ──
TRUSTED_ARP: Dict[str, str] = {}      # ip → known-good MAC (from assets.json)
KNOWN_HOSTS: set = set()              # authorised host IPs
_rogue_seen: set = set()             # rogue IPs already alerted (de-dupe)
# dst → {"srcs": {src: set(ports)}, "start": ts}  — distributed-scan aggregation
distributed_tracker = defaultdict(lambda: {"srcs": defaultdict(set), "start": 0.0})
# src → {port: last_ts}  — long-horizon slow-scan accumulator
slowscan_tracker    = defaultdict(dict)
# src → EWMA baseline state for connection rate
baseline_tracker    = defaultdict(lambda: {"ewma": None, "ewvar": 0.0,
                                           "bucket": None, "count": 0, "samples": 0})


def load_asset_inventory():
    """Load assets.json if present: trusted IP↔MAC + authorised hosts.

    Optional — without it, ARP detection still works (learn-on-first-sight) but
    cannot tell the real host from the impostor. With it, attribution is exact.
    """
    TRUSTED_ARP.clear()
    KNOWN_HOSTS.clear()
    if not ASSETS_PATH.exists():
        print(f"[ASSETS] No inventory at {ASSETS_PATH} - ARP uses learn-on-first-sight "
              "(no trust anchor). Create one to enable verified ARP + rogue-host detection.")
        return
    try:
        data = json.loads(ASSETS_PATH.read_text(encoding="utf-8"))
    except (ValueError, OSError) as e:
        print(f"[ASSETS][WARN] Could not read {ASSETS_PATH}: {e}")
        return
    gw = data.get("gateway") or {}
    if gw.get("ip") and gw.get("mac"):
        TRUSTED_ARP[gw["ip"]] = gw["mac"].lower()
        KNOWN_HOSTS.add(gw["ip"])
    for ip, mac in (data.get("trusted_arp") or {}).items():
        TRUSTED_ARP[ip] = str(mac).lower()
        KNOWN_HOSTS.add(ip)
    for ip in (data.get("known_hosts") or []):
        KNOWN_HOSTS.add(ip)
    print(f"[ASSETS] Loaded inventory: {len(TRUSTED_ARP)} trusted ARP entries, "
          f"{len(KNOWN_HOSTS)} known hosts.")

# ── Kill-chain / intrusion state (per host = suspected attacker or victim) ──
# host → {"stages": {stage: {"first": ts, "last": ts, "count": n, "detail": str}},
#         "first_seen": ts, "last_activity": ts}
intrusion_tracker: Dict[str, dict] = {}
intrusion_lock    = threading.Lock()

system_stats = {
    "packets_captured": 0,
    "alerts_generated": 0,
    "start_time": None,
    "running": False,
    # Last capture failure, surfaced through /status so a dead sniffer is
    # visible on the dashboard instead of looking like an idle network.
    "sniffer_error": None,
}

sniffer_thread: Optional[threading.Thread] = None
sniffer_stop_event = threading.Event()

# ─────────────────────────────────────────────
#  MITRE ATT&CK MAPPING + KILL-CHAIN MODEL
# ─────────────────────────────────────────────
# Each detectable alert type → (kill-chain stage, MITRE tactic, MITRE technique id).
# Stages are ordered; a host that advances through them is an active intrusion.
KILL_CHAIN_ORDER = [
    "reconnaissance",
    "credential_access",
    "initial_access",
    "privilege_escalation",   # populated by the Phase-2 host log sensor
    "persistence",            # populated by the Phase-2 host log sensor
    "command_and_control",
    "lateral_movement",
    "collection",
    "exfiltration",
]

STAGE_LABELS = {
    "reconnaissance":       "Reconnaissance",
    "credential_access":    "Credential Access",
    "initial_access":       "Initial Access (BREACH)",
    "privilege_escalation": "Privilege Escalation (ROOT)",
    "persistence":          "Persistence",
    "command_and_control":  "Command & Control",
    "lateral_movement":     "Lateral Movement",
    "collection":           "Collection / Staging",
    "exfiltration":         "Exfiltration",
}

# alert_type → (stage, tactic name, technique id)
ATTACK_MAP = {
    "Port Scan":            ("reconnaissance",      "Discovery",           "T1046"),
    "Stealth Scan":         ("reconnaissance",      "Discovery",           "T1046"),
    "Slow Scan":            ("reconnaissance",      "Discovery",           "T1046"),
    "Distributed Scan":     ("reconnaissance",      "Discovery",           "T1046"),
    "TCP Signature":        ("reconnaissance",      "Discovery",           "T1046"),
    "Brute Force Attempt":  ("credential_access",   "Credential Access",   "T1110"),
    "Breach":               ("initial_access",      "Initial Access",      "T1078"),
    "Privilege Escalation": ("privilege_escalation","Privilege Escalation","T1548"),
    "C2 Beaconing":         ("command_and_control", "Command and Control", "T1071"),
    "Reverse Shell":        ("command_and_control", "Command and Control", "T1059"),
    "Lateral Movement":     ("lateral_movement",    "Lateral Movement",    "T1021"),
    "Data Staging":         ("collection",          "Collection",          "T1074"),
    "Data Exfiltration":    ("exfiltration",        "Exfiltration",        "T1041"),
    # Cloud log sensor (cloud_log_sensor.py) alert types → same unified kill chain.
    "Cloud Port Scan":      ("reconnaissance",      "Discovery",           "T1046"),
    "Cloud Recon":          ("reconnaissance",      "Discovery",           "T1046"),
    "Cloud Brute Force":    ("credential_access",   "Credential Access",   "T1110"),
    "Impossible Travel":    ("initial_access",      "Initial Access",      "T1078"),
    # Host auth-log sensor (host_log_sensor.py) — the HONEST source for
    # confirmed logins, root access and persistence (not visible on the wire).
    "Host Brute Force":     ("credential_access",   "Credential Access",   "T1110"),
    "SSH Login":            ("initial_access",      "Valid Accounts",      "T1078"),
    "Root Access":          ("privilege_escalation","Privilege Escalation","T1548"),
    "Persistence":          ("persistence",         "Persistence",         "T1136"),
    "Sensitive File Access":("collection",          "Collection",          "T1005"),
    # Protocol / trust-boundary attacks.
    "Cloud Metadata Access":("credential_access",   "Credential Access",   "T1552.005"),
    "DNS Tunneling":        ("exfiltration",        "Exfiltration",        "T1048"),
    # (ARP Spoofing is intentionally NOT mapped: at L2 the attacker forges the
    #  source IP, so per-attacker kill-chain attribution would be unreliable.
    #  It still fires as a standalone high-severity alert.)
}


def _mitre_for(alert_type: str):
    """Return (stage, tactic, technique) for an alert type, or (None, None, None)."""
    return ATTACK_MAP.get(alert_type, (None, None, None))


def update_intrusion_state(alert_type: str, host: Optional[str], detail: str):
    """Record that `host` reached the kill-chain stage implied by `alert_type`.

    `host` is the attacker/compromised endpoint we attribute the activity to
    (the alert source). This is what powers the per-host Intrusion Status panel.
    """
    stage, _, _ = _mitre_for(alert_type)
    if not stage or not host:
        return
    now = time.time()
    announce = False
    stages_snapshot: List[str] = []
    with intrusion_lock:
        rec = intrusion_tracker.get(host)
        if rec is None:
            rec = {"stages": {}, "first_seen": now, "last_activity": now,
                   "announced": False}
            intrusion_tracker[host] = rec
        rec["last_activity"] = now
        st = rec["stages"].get(stage)
        if st is None:
            rec["stages"][stage] = {"first": now, "last": now, "count": 1, "detail": detail}
        else:
            st["last"] = now
            st["count"] += 1
            st["detail"] = detail
        # Decide whether this host just became a full active intrusion.
        if not rec["announced"] and intrusion_status(rec) == "ACTIVE INTRUSION":
            rec["announced"] = True
            announce = True
            stages_snapshot = [s for s in KILL_CHAIN_ORDER if s in rec["stages"]]

    # Fire the meta-alert OUTSIDE the lock (generate_alert re-enters this fn).
    if announce:
        chain = " → ".join(STAGE_LABELS[s] for s in stages_snapshot)
        generate_alert(
            "Active Intrusion", "critical", host, None, "Multiple",
            f"ACTIVE INTRUSION: {host} has progressed through the kill chain "
            f"[{chain}] — attacker is confirmed inside and operating",
            {"kill_chain": stages_snapshot, "stage_count": len(stages_snapshot)},
            _from_correlation=True,
        )


def intrusion_status(rec: dict) -> str:
    """Classify a host's kill-chain progress into a headline status."""
    stages = set(rec["stages"].keys())
    post_access = {"privilege_escalation", "persistence", "command_and_control",
                   "lateral_movement", "collection", "exfiltration"}
    if "initial_access" in stages and (stages & post_access):
        return "ACTIVE INTRUSION"
    if "initial_access" in stages:
        return "BREACHED"
    if "credential_access" in stages:
        return "UNDER ATTACK"
    if stages:
        return "PROBING"
    return "QUIET"


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
    details = dict(details or {})

    # Tag every alert with its kill-chain stage + MITRE ATT&CK tactic/technique.
    stage, tactic, technique = _mitre_for(alert_type)
    if stage:
        details.setdefault("kill_chain_stage", stage)
        details.setdefault("mitre_tactic", tactic)
        details.setdefault("mitre_technique", technique)

    alert = {
        "id":        alert_id,
        "type":      alert_type,
        "severity":  severity,
        "src":       src,
        "dst":       dst,
        "protocol":  protocol,
        "message":   message,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "details":   details,
    }

    with alerts_lock:
        alerts.append(alert)

    # Durable store — survives restarts. Best-effort: a DB hiccup must never
    # drop the live alert or crash the capture thread.
    try:
        db_insert_alert(alert)
        if alert_id % 1000 == 0:
            db_prune_alerts()
    except Exception as e:
        print(f"[DB][WARN] Could not persist alert {alert_id}: {e}")

    # ASCII arrow only: a non-UTF-8 stdout (Windows cp1252, Linux LANG=C under
    # systemd/nohup) would raise UnicodeEncodeError here and 500 the whole alert.
    print(f"[{severity.upper():8s}] {alert_type} -> {message}")

    # Advance the per-host kill-chain state (drives the Intrusion Status panel).
    update_intrusion_state(alert_type, src, message)

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
        # Mark this target as "hot": if a session later establishes here, it's a breach.
        note_bruteforce_target(src, dst, port)
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
#  POST-COMPROMISE DETECTION HELPERS
# ─────────────────────────────────────────────
def is_internal(ip: str) -> bool:
    """True if an IP belongs to our network (RFC1918 / configured prefixes)."""
    return ip.startswith(INTERNAL_PREFIXES)


def _prune(timestamps: list, window: float, now: float) -> list:
    return [t for t in timestamps if now - t < window]


def _coeff_of_variation(intervals: List[float]) -> float:
    """Std-dev / mean of inter-arrival gaps. Low = very regular = beacon-like."""
    if len(intervals) < 2:
        return 999.0
    mean = sum(intervals) / len(intervals)
    if mean <= 0:
        return 999.0
    var = sum((x - mean) ** 2 for x in intervals) / len(intervals)
    return (var ** 0.5) / mean


def _pc_should_fire(key: str, cooldown: float = 60.0) -> bool:
    """Rate-limit a post-compromise verdict so it fires once per cooldown."""
    now = time.time()
    with pc_lock:
        last = _recent_pc_alerts.get(key, 0.0)
        if now - last < cooldown:
            return False
        _recent_pc_alerts[key] = now
        return True


# A session with a just-brute-forced service that moves clearly more data than a
# failed-auth teardown = the login worked. Tuned above a couple of SSH banners.
BREACH_SESSION_BYTES = 12000


def note_bruteforce_target(src: str, dst: str, port: int):
    """Remember that `src` brute-forced `dst:port`. We then watch traffic with
    that target: a sustained, data-heavy session there implies the login worked.

    (TCP completes even for FAILED logins — auth is above TCP — so mere
    connection establishment is not proof. Byte volume of the session is the
    honest packet-level discriminator between a failed attempt and a breach.)"""
    breach_hot_targets[src][(dst, port)] = {"ts": time.time(), "bytes": 0, "fired": False}


# ─────────────────────────────────────────────
#  POST-COMPROMISE DETECTORS
# ─────────────────────────────────────────────
def track_hot_target_traffic(src: str, dst: str, sport: int, dport: int, plen: int):
    """Accumulate bytes exchanged with a brute-forced ("hot") target. When a
    single session there grows past BREACH_SESSION_BYTES, treat it as a breach.

    Matches traffic in either direction of the session with the hot target."""
    now = time.time()
    initiator = server = None
    port = 0
    rec = None
    # client → server:  src=attacker, dst=target, dport=service port
    r = breach_hot_targets.get(src, {}).get((dst, dport))
    if r is not None:
        initiator, server, port, rec = src, dst, dport, r
    else:
        # server → client:  src=target(service), dst=attacker, sport=service port
        r = breach_hot_targets.get(dst, {}).get((src, sport))
        if r is not None:
            initiator, server, port, rec = dst, src, sport, r
    if rec is None:
        return
    if now - rec["ts"] > BREACH_MEMORY_SEC:
        breach_hot_targets[initiator].pop((server, port), None)
        return
    rec["bytes"] += plen
    if rec["bytes"] >= BREACH_SESSION_BYTES and not rec["fired"] and \
            _pc_should_fire(f"breach:{initiator}:{server}:{port}", cooldown=180):
        rec["fired"] = True
        kb = round(rec["bytes"] / 1024, 1)
        generate_alert(
            "Breach", "critical", initiator, server, "TCP",
            f"SUSPECTED COMPROMISE: {initiator} opened a sustained session on "
            f"{server}:{port} ({kb} KB) right after brute-forcing it — "
            f"attacker appears to be INSIDE",
            {"target_port": port, "session_bytes": rec["bytes"],
             "evidence": "data-heavy session immediately after brute-force storm",
             "confidence": "medium (network-inferred; confirm via host auth logs — Phase 2)"},
        )


def detect_lateral_movement_net(initiator: str, target: str, tport: int):
    """Internal host establishing sessions to many internal hosts on admin ports."""
    if tport not in LATERAL_PORTS:
        return
    if not (is_internal(initiator) and is_internal(target)):
        return
    now = time.time()
    t = lateral_tracker[initiator]
    t["timestamps"] = _prune(t["timestamps"], LATERAL_WINDOW, now)
    t["timestamps"].append(now)
    t["dests"].add(target)
    if len(t["dests"]) >= LATERAL_THRESHOLD and \
            _pc_should_fire(f"lateral:{initiator}", cooldown=90):
        generate_alert(
            "Lateral Movement", "critical", initiator, None, "TCP",
            f"Lateral movement: internal host {initiator} opened sessions to "
            f"{len(t['dests'])} internal hosts on admin ports in {LATERAL_WINDOW}s",
            {"internal_targets": len(t["dests"]), "window_seconds": LATERAL_WINDOW,
             "targets": sorted(t["dests"])[:12]},
        )
        lateral_tracker[initiator] = {"dests": set(), "timestamps": []}


def detect_reverse_shell(initiator: str, target: str, tport: int):
    """Internal host initiating an OUTBOUND session to the Internet on a port
    that is not normal client traffic — a callback to the attacker."""
    if not (is_internal(initiator) and not is_internal(target)):
        return
    if tport in COMMON_OUTBOUND_PORTS:
        return
    if _pc_should_fire(f"revshell:{initiator}:{target}:{tport}", cooldown=120):
        generate_alert(
            "Reverse Shell", "high", initiator, target, "TCP",
            f"Reverse-shell indicator: internal host {initiator} called OUT to "
            f"{target}:{tport} (non-standard port) — possible attacker callback",
            {"target_port": tport, "direction": "internal→external",
             "note": "confirm against known-good egress"},
        )


def detect_c2_beacon(initiator: str, target: str):
    """Repeated, regularly-timed outbound connections to one external IP."""
    if not (is_internal(initiator) and not is_internal(target)):
        return
    now = time.time()
    hist = beacon_tracker[initiator][target]
    hist[:] = _prune(hist, BEACON_WINDOW, now)
    hist.append(now)
    if len(hist) < BEACON_MIN_HITS:
        return
    intervals = [hist[i] - hist[i - 1] for i in range(1, len(hist))]
    if any(iv < BEACON_MIN_INTERVAL for iv in intervals):
        return  # too bursty to be a heartbeat
    cv = _coeff_of_variation(intervals)
    if cv <= BEACON_MAX_CV and _pc_should_fire(f"beacon:{initiator}:{target}", cooldown=300):
        avg = sum(intervals) / len(intervals)
        generate_alert(
            "C2 Beaconing", "high", initiator, target, "TCP",
            f"C2 beaconing: {initiator} → {target} every ~{avg:.0f}s "
            f"({len(hist)} connections, jitter CV={cv:.2f}) — command-and-control pattern",
            {"beacon_interval_sec": round(avg, 1), "connections": len(hist),
             "regularity_cv": round(cv, 3)},
        )
        beacon_tracker[initiator][target] = []


def detect_data_exfiltration_net(src: str, dst: str, plen: int):
    """Sustained outbound bytes from an internal host to an external IP."""
    if not (is_internal(src) and not is_internal(dst)):
        return
    now = time.time()
    t = exfil_tracker[src]
    t["timestamps"] = _prune(t["timestamps"], EXFIL_WINDOW, now)
    if not t["timestamps"]:
        t["bytes"] = 0
    t["timestamps"].append(now)
    t["bytes"] += plen
    t["top"] = dst
    if t["bytes"] > EXFIL_BYTES_THRESHOLD and \
            _pc_should_fire(f"exfil:{src}", cooldown=120):
        mb = round(t["bytes"] / 1_000_000, 1)
        generate_alert(
            "Data Exfiltration", "critical", src, dst, "Multiple",
            f"Data exfiltration: internal host {src} sent {mb} MB to external "
            f"IP {dst} within {EXFIL_WINDOW}s",
            {"bytes_transferred": t["bytes"], "megabytes": mb, "destination": dst,
             "window_seconds": EXFIL_WINDOW},
        )
        exfil_tracker[src] = {"bytes": 0, "timestamps": [], "top": None}


def detect_data_staging(src: str, dst: str, plen: int):
    """One internal host aggregating large data FROM many internal hosts."""
    if not (is_internal(src) and is_internal(dst)):
        return
    now = time.time()
    t = staging_tracker[dst]
    t["timestamps"] = _prune(t["timestamps"], STAGING_WINDOW, now)
    if not t["timestamps"]:
        t["srcs"] = defaultdict(int)
    t["timestamps"].append(now)
    t["srcs"][src] += plen
    total = sum(t["srcs"].values())
    if len(t["srcs"]) >= STAGING_SRC_THRESHOLD and total >= STAGING_BYTES_THRESHOLD \
            and _pc_should_fire(f"staging:{dst}", cooldown=180):
        mb = round(total / 1_000_000, 1)
        generate_alert(
            "Data Staging", "high", dst, None, "TCP",
            f"Data staging: internal host {dst} aggregated {mb} MB from "
            f"{len(t['srcs'])} internal hosts in {STAGING_WINDOW}s — pre-exfil collection",
            {"source_hosts": len(t["srcs"]), "megabytes": mb,
             "window_seconds": STAGING_WINDOW},
        )
        staging_tracker[dst] = {"srcs": defaultdict(int), "timestamps": []}


def on_connection_established(initiator: str, target: str, tport: int):
    """Called when a TCP handshake completes (SYN → SYN-ACK seen).

    Establishment (not just a SYN probe) is what separates a real session
    from reconnaissance — so the highest-signal post-compromise checks run here.
    (Breach itself is byte-driven, handled in track_hot_target_traffic.)
    """
    detect_lateral_movement_net(initiator, target, tport)
    detect_reverse_shell(initiator, target, tport)
    detect_c2_beacon(initiator, target)


def _prune_handshakes(now: float):
    """Drop stale pending SYNs so the handshake table can't grow unbounded."""
    if len(handshake_tracker) < HANDSHAKE_MAX_ENTRIES:
        return
    stale = [k for k, ts in handshake_tracker.items() if now - ts > HANDSHAKE_TTL_SEC]
    for k in stale:
        handshake_tracker.pop(k, None)


# ─────────────────────────────────────────────
#  PROTOCOL / TRUST-BOUNDARY DETECTORS
# ─────────────────────────────────────────────
def _shannon(s: str) -> float:
    """Shannon entropy (bits/char). High = random-looking = encoded payload."""
    if not s:
        return 0.0
    freq = defaultdict(int)
    for ch in s:
        freq[ch] += 1
    n = len(s)
    return -sum((c / n) * math.log2(c / n) for c in freq.values())


def _dns_parent(qname: str) -> str:
    """Registered-ish parent domain: the last two labels of a query name."""
    labels = qname.rstrip(".").split(".")
    return ".".join(labels[-2:]) if len(labels) >= 2 else qname.rstrip(".")


def _is_tunnel_qname(qname: str, qtype: int) -> bool:
    """Heuristic: does this DNS query look like tunneled data rather than a
    normal hostname lookup?"""
    q = qname.rstrip(".")
    if not q:
        return False
    if qtype in DNS_SUSPICIOUS_QTYPES:            # TXT/NULL/zone-transfer
        return True
    labels = q.split(".")
    if any(len(lbl) > DNS_LONG_LABEL for lbl in labels):   # one giant label
        return True
    # Long overall name whose subdomain looks high-entropy (base32/hex payload).
    if len(q) > DNS_QNAME_LEN:
        sub = "".join(labels[:-2]) if len(labels) > 2 else labels[0]
        if _shannon(sub) >= DNS_ENTROPY_MIN:
            return True
    return False


def detect_metadata_access(src: str, dst: str):
    """Burst of connections to a cloud metadata (IMDS) endpoint — the SSRF
    credential-theft trust-boundary attack. A single hit is normal SDK traffic;
    a rapid burst is enumeration/harvesting."""
    if dst not in METADATA_IPS:
        return
    now = time.time()
    t = metadata_tracker[src]
    t[:] = _prune(t, IMDS_WINDOW, now)
    t.append(now)
    if len(t) >= IMDS_BURST_THRESHOLD and _pc_should_fire(f"imds:{src}", 90):
        generate_alert(
            "Cloud Metadata Access", "high", src, dst, "TCP",
            f"Cloud metadata harvesting: {src} hit IMDS {dst} {len(t)} times in "
            f"{IMDS_WINDOW}s — possible SSRF stealing instance IAM credentials",
            {"metadata_ip": dst, "hits": len(t), "window_seconds": IMDS_WINDOW,
             "note": "single accesses are normal SDK traffic; bursts are not"},
        )
        metadata_tracker[src] = []


def detect_dns_tunneling(src: str, qname: str, qtype: int):
    """Sustained long / high-entropy DNS queries to one parent domain = data
    smuggled over DNS (covert exfil / C2)."""
    if not _is_tunnel_qname(qname, qtype):
        return
    parent = _dns_parent(qname)
    now = time.time()
    key = (src, parent)
    t = dns_tracker[key]
    t["timestamps"] = _prune(t["timestamps"], DNS_WINDOW, now)
    t["timestamps"].append(now)
    if len(t["timestamps"]) >= DNS_TUNNEL_THRESHOLD and \
            _pc_should_fire(f"dnstun:{src}:{parent}", 120):
        generate_alert(
            "DNS Tunneling", "high", src, None, "UDP",
            f"DNS tunneling: {src} sent {len(t['timestamps'])} long/high-entropy "
            f"queries to *.{parent} in {DNS_WINDOW}s — data exfil over DNS",
            {"parent_domain": parent, "suspicious_queries": len(t["timestamps"]),
             "window_seconds": DNS_WINDOW, "sample_qname": qname.rstrip(".")[:80]},
        )
        dns_tracker[key] = {"timestamps": []}


def detect_arp_spoofing(ip: str, mac: str):
    """One IP claimed by an unexpected MAC = ARP poisoning / MITM.

    With an asset inventory we know the RIGHT MAC, so we can name the impostor
    even if the sensor started mid-attack (fixes the first-seen-wins inversion).
    Without one, we fall back to learn-on-first-sight and flag it as unverified.
    """
    mac = (mac or "").lower()
    if not ip or not mac:
        return

    # Rogue device: an internal IP not in the authorised inventory.
    if KNOWN_HOSTS and is_internal(ip) and ip not in KNOWN_HOSTS and ip not in _rogue_seen:
        _rogue_seen.add(ip)
        generate_alert(
            "Rogue Host", "medium", ip, None, "ARP",
            f"Rogue device: {ip} ({mac}) announced on the LAN but is not in the "
            f"asset inventory — unauthorised host",
            {"ip": ip, "mac": mac},
        )

    trusted = TRUSTED_ARP.get(ip)
    if trusted:
        # We have ground truth. Any other MAC for this IP is the attacker.
        if mac != trusted and _pc_should_fire(f"arp:{ip}", ARP_COOLDOWN):
            generate_alert(
                "ARP Spoofing", "critical", ip, None, "ARP",
                f"ARP spoofing CONFIRMED: {ip} should be at {trusted} (asset "
                f"inventory) but ARP now claims {mac} — {mac} is the impostor / MITM",
                {"ip": ip, "legitimate_mac": trusted, "impostor_mac": mac,
                 "verified": True},
            )
        return

    # No trust anchor — learn first mapping, flag later conflicts as unverified.
    prev = arp_ip_mac.get(ip)
    if prev is None:
        arp_ip_mac[ip] = mac
        return
    if prev != mac:
        if _pc_should_fire(f"arp:{ip}", ARP_COOLDOWN):
            generate_alert(
                "ARP Spoofing", "high", ip, None, "ARP",
                f"ARP anomaly: IP {ip} MAC changed {prev} → {mac} — possible MITM "
                f"(no asset baseline to confirm which is legitimate)",
                {"ip": ip, "old_mac": prev, "new_mac": mac, "verified": False,
                 "note": "add this host to assets.json for verified attribution"},
            )
        arp_ip_mac[ip] = mac


def detect_distributed_scan(src: str, dst: str, port: int):
    """Many sources each probing one destination = coordinated / botnet scan
    that slips under the per-source port-scan threshold."""
    now = time.time()
    t = distributed_tracker[dst]
    if now - t["start"] > DISTRIBUTED_WINDOW:
        t["srcs"] = defaultdict(set)
        t["start"] = now
    t["srcs"][src].add(port)
    scanning = [s for s, ports in t["srcs"].items() if len(ports) >= DISTRIBUTED_PORTS_MIN]
    if len(scanning) >= DISTRIBUTED_SRC_MIN and \
            _pc_should_fire(f"distscan:{dst}", 60):
        generate_alert(
            "Distributed Scan", "high", None, dst, "TCP",
            f"Distributed scan: {len(scanning)} sources coordinated a port scan "
            f"against {dst} within {DISTRIBUTED_WINDOW}s (each below the single-host "
            f"threshold)",
            {"target": dst, "source_count": len(scanning),
             "sources": sorted(scanning)[:12], "window_seconds": DISTRIBUTED_WINDOW},
        )
        distributed_tracker[dst] = {"srcs": defaultdict(set), "start": now}


def detect_slow_scan(src: str, port: int):
    """Distinct ports probed by one source over a LONG horizon — defeats the
    short stealth-scan window (low-and-slow reconnaissance)."""
    now = time.time()
    ports = slowscan_tracker[src]
    ports[port] = now
    # Decay: forget ports older than the long horizon.
    for p in [p for p, ts in ports.items() if now - ts > SLOWSCAN_WINDOW]:
        del ports[p]
    if len(ports) >= SLOWSCAN_THRESHOLD and _pc_should_fire(f"slowscan:{src}", 300):
        generate_alert(
            "Slow Scan", "medium", src, None, "TCP",
            f"Low-and-slow scan: {src} probed {len(ports)} distinct ports over "
            f"up to {SLOWSCAN_WINDOW // 60} min — evading short-window detection",
            {"distinct_ports": len(ports), "horizon_seconds": SLOWSCAN_WINDOW},
        )
        slowscan_tracker[src] = {}


def update_baseline(src: str):
    """Per-source connection-rate EWMA. Flags a source whose activity spikes far
    above its OWN learned normal — an adaptive alternative to fixed thresholds."""
    now = time.time()
    b = baseline_tracker[src]
    bucket = int(now // BASELINE_BUCKET)
    if b["bucket"] is None:
        b["bucket"] = bucket
    if bucket == b["bucket"]:
        b["count"] += 1
        return
    # Bucket rolled over — evaluate the completed bucket against the baseline.
    completed = b["count"]
    ewma, ewvar, samples = b["ewma"], b["ewvar"], b["samples"]
    if ewma is not None and samples >= BASELINE_MIN_SAMPLES and \
            completed >= BASELINE_MIN_COUNT:
        std = ewvar ** 0.5
        if completed > ewma + BASELINE_K * std and _pc_should_fire(f"baseline:{src}", 120):
            generate_alert(
                "Traffic Anomaly", "medium", src, None, "TCP",
                f"Adaptive baseline: {src} opened {completed} connections in "
                f"{BASELINE_BUCKET}s vs a learned norm of ~{ewma:.1f} "
                f"(> {BASELINE_K:.0f}σ) — unusual burst for this host",
                {"burst": completed, "baseline_mean": round(ewma, 1),
                 "baseline_std": round(std, 1), "sigma": BASELINE_K},
            )
    # Update EWMA/EWVar with the completed bucket, then start a fresh bucket.
    if ewma is None:
        b["ewma"] = float(completed)
    else:
        diff = completed - ewma
        b["ewma"] = ewma + BASELINE_ALPHA * diff
        b["ewvar"] = (1 - BASELINE_ALPHA) * (ewvar + BASELINE_ALPHA * diff * diff)
    b["samples"] = samples + 1
    b["bucket"] = bucket
    b["count"] = 1


# ─────────────────────────────────────────────
#  PACKET PROCESSOR
# ─────────────────────────────────────────────
def process_packet(packet):
    system_stats["packets_captured"] += 1

    # ARP has no IP layer — handle it before the IP guard (L2 trust boundary).
    if ARP is not None and packet.haslayer(ARP):
        arp = packet[ARP]
        # op 1=who-has (request), 2=is-at (reply); both announce psrc→hwsrc.
        detect_arp_spoofing(getattr(arp, "psrc", ""), getattr(arp, "hwsrc", ""))
        return

    if not packet.haslayer(IP):
        return

    src  = packet[IP].src
    dst  = packet[IP].dst
    plen = len(packet)

    # DDoS check runs on every IP packet
    detect_ddos(src, dst)

    # Cloud metadata (IMDS) trust-boundary abuse — any dst, TCP or UDP.
    detect_metadata_access(src, dst)

    # Byte-volume post-compromise detectors (protocol-agnostic)
    detect_data_exfiltration_net(src, dst, plen)   # internal → external, mass upload
    detect_data_staging(src, dst, plen)            # internal → internal aggregation

    if packet.haslayer(TCP):
        tcp   = packet[TCP]
        flags = int(tcp.flags)
        dport = tcp.dport
        sport = tcp.sport
        syn   = bool(flags & 0x02)
        ack   = bool(flags & 0x10)

        # Signature detection
        sig = check_tcp_signature(flags)
        if sig:
            generate_alert(
                "TCP Signature", "low", src, dst, "TCP",
                f"TCP signature '{sig}' from {src} → {dst}:{dport}",
                {"signature": sig, "flags": flags, "dst_port": dport},
            )

        if syn and not ack:          # pure SYN — connection initiation / probe
            detect_port_scan(src, dst, dport)
            detect_stealth_scan(src, dport)
            detect_syn_flood(src, dst)
            detect_distributed_scan(src, dst, dport)   # coordinated / botnet scan
            detect_slow_scan(src, dport)               # low-and-slow recon
            update_baseline(src)                        # adaptive per-host baseline
            now = time.time()
            handshake_tracker[(src, dst, dport)] = now
            _prune_handshakes(now)
        elif syn and ack:            # SYN-ACK — server accepted; handshake completes
            # The originating SYN was (client=dst, server=src, service port=sport).
            if handshake_tracker.pop((dst, src, sport), None) is not None:
                on_connection_established(initiator=dst, target=src, tport=sport)

        if ack:
            syn_tracker[src]["ack"] += 1

        detect_brute_force(src, dst, dport)
        # Watch data volume with any brute-forced ("hot") target → breach signal.
        track_hot_target_traffic(src, dst, sport, dport, plen)

    elif packet.haslayer(UDP):
        udp = packet[UDP]
        detect_udp_amplification(src, dst, udp.dport, len(packet))

        # DNS tunneling — inspect outbound queries (port 53) for covert exfil.
        if DNS is not None and packet.haslayer(DNS):
            dns = packet[DNS]
            if int(getattr(dns, "qr", 0)) == 0 and int(getattr(dns, "qdcount", 0)) > 0:
                try:
                    qd = dns.qd
                    qname = qd.qname.decode("utf-8", "ignore") if qd else ""
                    qtype = int(qd.qtype) if qd else 0
                    if qname:
                        detect_dns_tunneling(src, qname, qtype)
                except Exception:
                    pass

    # ICMP: currently just counted via DDoS tracker above
    # Extend here if you want ICMP-specific detection (ping flood, smurf, etc.)

# ─────────────────────────────────────────────
#  SNIFFER THREAD
# ─────────────────────────────────────────────
def list_interfaces() -> List[str]:
    """Names of capture interfaces scapy can see on this host (empty if no scapy)."""
    if not SCAPY_AVAILABLE:
        return []
    try:
        from scapy.arch import get_if_list
        return sorted(get_if_list())
    except Exception:
        return []


def resolve_interface() -> Optional[str]:
    """NETWORK_INTERFACE, or None to let scapy choose its default interface."""
    if not NETWORK_INTERFACE or NETWORK_INTERFACE.lower() == "auto":
        return None
    return NETWORK_INTERFACE


def _sniffer_worker():
    """Capture loop.

    Runs in short timeout slices instead of one blocking sniff() call so that
    /control/stop takes effect on an idle network too — stop_filter is only
    evaluated when a packet arrives, so on a quiet link a stop request would
    otherwise hang until the next packet.

    Any capture error (missing interface, no CAP_NET_RAW / no Npcap) is recorded
    in system_stats and flips `running` back to False. Previously the thread died
    on the exception while `running` stayed True, so the dashboard showed a live
    sensor capturing zero packets forever.
    """
    iface = resolve_interface()
    shown = iface or "(scapy default)"
    print(f"[SNIFFER] Capturing on interface: {shown}")
    system_stats["sniffer_error"] = None
    try:
        while not sniffer_stop_event.is_set():
            sniff(
                iface=iface,
                prn=process_packet,
                store=False,
                timeout=1,
                stop_filter=lambda _: sniffer_stop_event.is_set(),
            )
        print("[SNIFFER] Stopped.")
    except Exception as e:
        msg = f"{type(e).__name__}: {e}"
        system_stats["sniffer_error"] = msg
        print(f"[SNIFFER][ERROR] Capture failed on '{shown}' -> {msg}")
        available = list_interfaces()
        if available:
            print(f"[SNIFFER][ERROR] Interfaces available here: {', '.join(available)}")
        print("[SNIFFER][ERROR] Set NIDS_IFACE to one of the above "
              "(or 'auto'), and run with root/Administrator privileges.")
    finally:
        # Whatever ended the loop, the sensor is no longer capturing. Keep the
        # reported state honest so the dashboard can't show a phantom "running".
        system_stats["running"] = False

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
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
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
    if user is None or not _verify_password(payload.password, user["password"]):
        raise HTTPException(status_code=401, detail="Invalid username or password.")

    # Transparently upgrade legacy unsalted-SHA256 hashes to PBKDF2 on login.
    if _is_legacy_hash(user["password"]):
        with get_db() as conn:
            conn.execute("UPDATE users SET password = ? WHERE username = ?",
                         (_hash_password(payload.password), payload.username))
        print(f"[AUTH] Upgraded password hash for {payload.username} to PBKDF2.")

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
    if not _verify_password(payload.current_password, user["password"]):
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
    print(f"[AUTH] Role updated: {username} -> {payload.role} by {session['username']}")
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
        "interface":        NETWORK_INTERFACE or "auto",
        "sniffer_error":    system_stats["sniffer_error"],
        "available_interfaces": list_interfaces(),
        "user":             session["username"],
        "role":             session["role"],
    }


# ── Intrusion / Kill-Chain Endpoint ───────────
# Order used to rank hosts by how deep into the kill chain they are.
_STATUS_RANK = {
    "ACTIVE INTRUSION": 0,
    "BREACHED":         1,
    "UNDER ATTACK":     2,
    "PROBING":          3,
    "QUIET":            4,
}


@app.get("/intrusions", tags=["Alerts"])
def get_intrusions(session: dict = Depends(require_permission("view_alerts"))):
    """Per-host kill-chain progression — powers the Intrusion Status panel.

    Answers the operator's real question: is this host just poking at us, or is
    it INSIDE, moving laterally, and stealing data?
    """
    now = time.time()
    out = []
    with intrusion_lock:
        for host, rec in intrusion_tracker.items():
            status = intrusion_status(rec)
            reached = []
            for stage in KILL_CHAIN_ORDER:
                if stage in rec["stages"]:
                    st = rec["stages"][stage]
                    reached.append({
                        "stage":   stage,
                        "label":   STAGE_LABELS[stage],
                        "count":   st["count"],
                        "detail":  st["detail"],
                        "age_sec": round(now - st["last"], 1),
                    })
            out.append({
                "host":            host,
                "status":          status,
                "stages_reached":  reached,
                "stage_count":     len(reached),
                "furthest_stage":  reached[-1]["label"] if reached else None,
                "first_seen_sec":  round(now - rec["first_seen"], 1),
                "last_activity_sec": round(now - rec["last_activity"], 1),
            })
    out.sort(key=lambda h: (_STATUS_RANK.get(h["status"], 9), -h["stage_count"]))
    # Full stage catalogue so the UI can render the whole chain, filled or not.
    return {
        "chain": [{"stage": s, "label": STAGE_LABELS[s]} for s in KILL_CHAIN_ORDER],
        "hosts": out,
    }


# ── Model Endpoint ────────────────────────────

def _read_json(path: Path) -> Optional[dict]:
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, ValueError, OSError):
        return None


def _model_row(name: str, meta: dict, active: bool) -> dict:
    """Flatten a CIC-IDS trainer's metadata into a comparison-table row."""
    m = meta.get("metrics", {})
    return {
        "name":              name,
        "active":            active,
        "accuracy":          m.get("accuracy"),
        "macro_f1":          m.get("macro_f1"),
        "balanced_accuracy": m.get("balanced_accuracy"),
        "detect":            m.get("binary_recall"),   # any-attack recall
        "fpr":               m.get("binary_fpr"),
        "n_features":        meta.get("n_features") or len(
                                 meta.get("numeric_features", [])) + len(
                                 meta.get("categorical_features", [])),
    }


@app.get("/model/info", tags=["System"])
def model_info(session: dict = Depends(require_auth)):
    """Serve the ML models' honestly-evaluated CIC-IDS2017 metrics for the UI.

    Reports both retrained models: Model B (live schema, 13 packet-derivable
    features) which the live sensor actually serves, and Model A (full 78 flow
    features) as the stronger reference. Metadata is written by
    train_cicids_live.py / train_cicids_flow.py.
    """
    live = _read_json(Path("models/cicids_live_meta.json"))   # Model B (served)
    flow = _read_json(Path("models/cicids_flow_meta.json"))   # Model A (reference)

    if live is None and flow is None:
        raise HTTPException(
            status_code=404,
            detail="Model metadata not found. Run: python train_cicids_live.py "
                   "and python train_cicids_flow.py",
        )

    models = []
    if live is not None:
        models.append(_model_row(
            "HistGradientBoosting · live schema (13 feat)", live, active=True))
    if flow is not None:
        models.append(_model_row(
            "HistGradientBoosting · full flow (78 feat)", flow, active=False))

    active_meta = live or flow
    return {
        "models":             models,
        "feature_importance": active_meta.get("feature_importance", []),
        "classes":            active_meta.get("classes", []),
        "trained_on":         "CIC-IDS2017",
        "evaluated_on":       "CIC-IDS2017 hold-out (25%)",
        "test_samples":       active_meta.get("test_samples"),
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
    """Admin only. Clears the in-memory cache AND the durable store."""
    with alerts_lock:
        count = len(alerts)
        alerts.clear()
        system_stats["alerts_generated"] = 0
    db_clear_alerts()
    with correlation_lock:
        correlation_tracker.clear()
    with intrusion_lock:
        intrusion_tracker.clear()
    with pc_lock:
        _recent_pc_alerts.clear()
    print(f"[NIDS] {session['username']} cleared {count} alerts (memory + disk).")
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

    # Validate the interface up front. Reporting "started" for an interface that
    # does not exist is what made a dead sensor look like a quiet network.
    iface = resolve_interface()
    available = list_interfaces()
    if iface and available and iface not in available:
        raise HTTPException(
            400,
            f"Capture interface '{iface}' not found on this host. "
            f"Available: {', '.join(available)}. "
            f"Set NIDS_IFACE to one of these (or 'auto') and restart.",
        )

    sniffer_stop_event.clear()
    system_stats["sniffer_error"] = None
    system_stats["running"]    = True
    system_stats["start_time"] = time.time()
    sniffer_thread = threading.Thread(target=_sniffer_worker, daemon=True)
    sniffer_thread.start()

    # Give the capture loop a moment to fail loudly (permissions, missing Npcap)
    # so the caller learns about it now rather than seeing a phantom "running".
    time.sleep(0.5)
    if system_stats["sniffer_error"] or not sniffer_thread.is_alive():
        err = system_stats["sniffer_error"] or "capture thread exited immediately"
        system_stats["running"] = False
        raise HTTPException(500, f"Sniffer failed to start: {err}")

    print(f"[NIDS] Sniffer started by {session['username']}")

    return {"status": "started", "interface": iface or "(scapy default)"}


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
    load_asset_inventory()

    # Restore alerts from the durable store so a restart keeps history + the
    # alert-id counter (avoids id collisions with persisted rows).
    try:
        recent = db_load_recent_alerts(MAX_ALERTS_STORED)
        with alerts_lock:
            alerts.clear()
            alerts.extend(recent)
            system_stats["alerts_generated"] = db_max_alert_id()
        total = db_alert_count()
        if total:
            print(f"[NIDS] Restored {len(recent)} recent alerts into cache "
                  f"({total} total on disk).")
    except Exception as e:
        print(f"[DB][WARN] Could not restore alerts: {e}")

    if SENSOR_API_KEY == _DEFAULT_SENSOR_KEY:
        print("[NIDS][WARN] Using the PUBLIC default sensor key. Set NIDS_SENSOR_KEY "
              "(and match it in the sensors) before any real deployment.")

    print("[NIDS] main.py ready. POST /control/start to begin capture.")
    print("[NIDS] ML alerts expected from live_ids_v2.py via POST /alert")


# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)

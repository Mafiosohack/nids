"""
host_log_sensor.py — Host / Endpoint Auth-Log IDS Sensor  (Phase 2)

Reads Linux authentication logs (/var/log/auth.log style) and detects the
post-compromise activity that NEITHER packet capture NOR VPC flow logs can
see, because it happens ON the host, above the network layer:

  - Confirmed login        : sshd "Accepted password/publickey"   (Initial Access)
  - Host brute force        : bursts of sshd "Failed password"      (Credential Access)
  - Privilege escalation    : sudo→root, su→root, root SSH login    (Root Access)
  - Sensitive file access   : sudo reading /etc/shadow, ssh keys…   (Collection)
  - Persistence             : useradd, adding a user to sudo/wheel  (Persistence)

This is the HONEST answer to "is he inside?" and "does he have ROOT?".
The network sensors can only INFER a breach from traffic volume; auth logs
CONFIRM it (a real successful login) and are the only truthful source for
privilege escalation and persistence.

Runs alongside main.py and posts to the same /alert endpoint used by
live_ids_v2.py and cloud_log_sensor.py, so everything lands in one unified
kill chain on the dashboard.

    python host_log_sensor.py

Drop auth.log-format files (*.log / *.txt) into ./host_logs/ — or generate a
demo attack with:  python generate_host_logs.py
"""

import re
import time
import threading
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import requests

# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────
NIDS_ALERT_URL   = "http://127.0.0.1:8000/alert"
SENSOR_API_KEY   = "sensor-key-change-me-in-production"   # must match main.py
ALERT_TIMEOUT    = 3

# Directory to watch for auth-log files
LOG_WATCH_DIR    = Path("host_logs")

# How often to scan for new/updated log files (seconds)
POLL_INTERVAL    = 5

# ── Detection thresholds ──────────────────────
FAILED_LOGIN_THRESHOLD = 5      # failed sshd logins from one IP → brute force
FAILED_LOGIN_WINDOW    = 120    # seconds

# A successful login is treated as a CONFIRMED breach if this many failures
# from the same source preceded it (brute force that finally worked).
BREACH_AFTER_FAILURES  = 3

# Don't re-fire the same host verdict more often than this (seconds).
ROOT_COOLDOWN          = 60
PERSIST_COOLDOWN       = 30

# How long a successful login "owns" later host events (sudo/useradd have no IP,
# so we attribute them to whoever most recently logged into that host).
SESSION_TTL            = 3600   # seconds

# Files/paths whose access via sudo is worth flagging as data collection.
SENSITIVE_PATTERNS = ("/etc/shadow", "/etc/passwd", "/etc/sudoers", "/etc/gshadow",
                      "id_rsa", "id_ed25519", "/.ssh/", "authorized_keys",
                      "/etc/ssl", ".aws/credentials", ".kube/config")

# ─────────────────────────────────────────────
#  AUTH-LOG LINE PATTERNS  (syslog / auth.log format)
# ─────────────────────────────────────────────
RE_SYSLOG   = re.compile(
    r"^(?P<mon>\w{3})\s+(?P<day>\d+)\s+(?P<time>\d{2}:\d{2}:\d{2})\s+"
    r"(?P<host>\S+)\s+(?P<msg>.*)$"
)
RE_FAILED   = re.compile(
    r"sshd\[\d+\]:\s+Failed password for (?:invalid user )?(?P<user>\S+)\s+"
    r"from (?P<ip>\d{1,3}(?:\.\d{1,3}){3})"
)
RE_ACCEPT   = re.compile(
    r"sshd\[\d+\]:\s+Accepted (?:password|publickey) for (?P<user>\S+)\s+"
    r"from (?P<ip>\d{1,3}(?:\.\d{1,3}){3})"
)
RE_SUDO     = re.compile(r"sudo:\s+(?P<user>\S+)\s*:.*COMMAND=(?P<cmd>.+)$")
RE_SUDO_TGT = re.compile(r"USER=(?P<target>\S+)")
RE_SU_ROOT  = re.compile(r"\bsu(?:\[\d+\])?:.*(?:to root|session opened for user root)")
RE_USERADD  = re.compile(r"useradd\[\d+\]:\s+new user:\s+name=(?P<name>[^,]+)(?:.*UID=(?P<uid>\d+))?")
RE_GROUPADD = re.compile(
    r"(?:usermod|gpasswd)\[\d+\]:.*?(?:add '(?P<name>[^']+)' to group|"
    r"user (?P<name2>\S+) added by).*?'?(?P<group>sudo|wheel|admin|root)'?"
)

MONTHS = {m: i for i, m in enumerate(
    ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"], 1)}

# ─────────────────────────────────────────────
#  STATE
# ─────────────────────────────────────────────
# ip → [failure timestamps]  (sliding window per source)
failed_tracker: Dict[str, list] = defaultdict(list)
# host → {"ip": attacker_ip, "user": str, "ts": float}  — most recent login owner
active_sessions: Dict[str, dict] = {}
# de-dupe verdicts: key → last_fired_ts
_recent: Dict[str, float] = {}

processed_lines: Dict[str, int] = {}
state_lock = threading.Lock()

# ─────────────────────────────────────────────
#  ALERT
# ─────────────────────────────────────────────
def send_alert(alert_type: str, severity: str, src: str, dst: Optional[str],
               message: str, details: Optional[dict] = None):
    payload = {
        "src":        src,
        "dst":        dst,
        "proto":      "host",
        "alert_type": alert_type,
        "severity":   severity,
        "message":    message,
        "details":    details or {},
    }
    print(f"[HOST-ALERT] [{severity.upper()}] {alert_type} -> {message}")
    try:
        resp = requests.post(NIDS_ALERT_URL, json=payload, timeout=ALERT_TIMEOUT,
                             headers={"X-Sensor-Key": SENSOR_API_KEY})
        if resp.status_code == 200:
            print(f"  OK  Accepted by NIDS (id={resp.json().get('alert_id')})")
        else:
            print(f"  !!  NIDS returned HTTP {resp.status_code}")
    except requests.exceptions.ConnectionError:
        print(f"  !!  NIDS unreachable at {NIDS_ALERT_URL}. Is main.py running?")
    except Exception as e:
        print(f"  !!  Alert error: {e}")

# ─────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────
def parse_ts(mon: str, day: str, tm: str) -> float:
    """Best-effort syslog timestamp → epoch seconds (assumes current year)."""
    try:
        now = datetime.now()
        dt = datetime(now.year, MONTHS.get(mon, now.month), int(day),
                      *[int(x) for x in tm.split(":")])
        return dt.timestamp()
    except Exception:
        return time.time()


def _fire_once(key: str, cooldown: float, ts: float) -> bool:
    last = _recent.get(key, 0.0)
    if ts - last < cooldown:
        return False
    _recent[key] = ts
    return True


def prune(timestamps: list, window: float, now: float) -> list:
    return [t for t in timestamps if now - t < window]


def session_owner(host: str, ts: float) -> Optional[str]:
    """The attacker IP that most recently logged into `host` (if still valid).

    Host events like sudo/useradd carry no IP, so we attribute them to the
    active login session — this keeps the kill chain unified under one attacker.
    """
    s = active_sessions.get(host)
    if s and ts - s["ts"] <= SESSION_TTL:
        return s["ip"]
    return None

# ─────────────────────────────────────────────
#  DETECTORS
# ─────────────────────────────────────────────
def on_failed_login(host: str, user: str, ip: str, ts: float):
    t = failed_tracker[ip]
    t[:] = prune(t, FAILED_LOGIN_WINDOW, ts)
    t.append(ts)
    if len(t) >= FAILED_LOGIN_THRESHOLD and _fire_once(f"brute:{ip}", 60, ts):
        send_alert(
            "Host Brute Force", "high", ip, host,
            f"Host brute force: {len(t)} failed SSH logins from {ip} on host "
            f"'{host}' within {FAILED_LOGIN_WINDOW}s (auth.log)",
            {"failed_attempts": len(t), "targeted_host": host, "last_user": user,
             "detection_source": "host_auth_log"},
        )


def on_accepted_login(host: str, user: str, ip: str, ts: float):
    # Was this preceded by a brute-force burst from the same IP? → confirmed breach.
    recent_fails = len(prune(failed_tracker.get(ip, []), FAILED_LOGIN_WINDOW, ts))
    confirmed = recent_fails >= BREACH_AFTER_FAILURES

    with state_lock:
        active_sessions[host] = {"ip": ip, "user": user, "ts": ts}

    if confirmed:
        msg = (f"CONFIRMED BREACH: successful SSH login as '{user}' from {ip} on "
               f"host '{host}' after {recent_fails} failed attempts (auth.log)")
        sev = "critical"
    else:
        msg = (f"SSH login as '{user}' from {ip} on host '{host}' (auth.log)")
        sev = "high" if user == "root" else "medium"

    send_alert(
        "SSH Login", sev, ip, host, msg,
        {"user": user, "targeted_host": host, "preceding_failures": recent_fails,
         "confirmed_breach": confirmed, "confidence": "high (host auth log)",
         "detection_source": "host_auth_log"},
    )
    failed_tracker[ip] = []

    # Logging in directly AS root is itself root access.
    if user == "root":
        _flag_root(host, ip, f"direct root SSH login from {ip}", "root_login", ts)


def _flag_root(host: str, actor: str, how: str, key_suffix: str, ts: float):
    if not _fire_once(f"root:{host}:{key_suffix}", ROOT_COOLDOWN, ts):
        return
    send_alert(
        "Root Access", "critical", actor, host,
        f"ROOT ACCESS on host '{host}': {how} — attacker has full control (auth.log)",
        {"targeted_host": host, "method": how, "confidence": "high (host auth log)",
         "detection_source": "host_auth_log"},
    )


def on_sudo(host: str, user: str, cmd: str, target: Optional[str], ts: float):
    actor = session_owner(host, ts) or host
    is_root = (target is None) or (target == "root")

    # Sensitive file access via sudo → data collection.
    low = cmd.lower()
    for pat in SENSITIVE_PATTERNS:
        if pat.lower() in low:
            if _fire_once(f"sens:{host}:{pat}", 30, ts):
                send_alert(
                    "Sensitive File Access", "high", actor, host,
                    f"Sensitive access on '{host}': {user} used sudo to touch "
                    f"{pat} (COMMAND={cmd.strip()[:80]}) (auth.log)",
                    {"user": user, "command": cmd.strip()[:200], "matched": pat,
                     "targeted_host": host, "detection_source": "host_auth_log"},
                )
            break

    if is_root:
        interactive = any(s in low for s in ("bash", "/sh", " sh", "zsh", "-i"))
        how = (f"{user} ran an interactive root shell via sudo"
               if interactive else
               f"{user} executed a command as root via sudo (COMMAND={cmd.strip()[:60]})")
        _flag_root(host, actor, how, "sudo", ts)


def on_su_root(host: str, ts: float):
    actor = session_owner(host, ts) or host
    _flag_root(host, actor, "su to root", "su", ts)


def on_new_account(host: str, name: str, uid: Optional[str], ts: float):
    actor = session_owner(host, ts) or host
    if not _fire_once(f"persist:acct:{host}:{name}", PERSIST_COOLDOWN, ts):
        return
    root_uid = uid == "0"
    sev = "critical" if root_uid else "high"
    extra = " with UID 0 (root-equivalent!)" if root_uid else ""
    send_alert(
        "Persistence", sev, actor, host,
        f"Persistence on '{host}': new account '{name}' created{extra} — "
        f"attacker establishing a foothold (auth.log)",
        {"account": name, "uid": uid, "root_equivalent": root_uid,
         "targeted_host": host, "technique": "new account",
         "detection_source": "host_auth_log"},
    )


def on_group_add(host: str, name: str, group: str, ts: float):
    actor = session_owner(host, ts) or host
    if not _fire_once(f"persist:grp:{host}:{name}:{group}", PERSIST_COOLDOWN, ts):
        return
    send_alert(
        "Persistence", "critical", actor, host,
        f"Persistence on '{host}': user '{name}' added to privileged group "
        f"'{group}' — persistent admin access (auth.log)",
        {"account": name, "group": group, "targeted_host": host,
         "technique": "privileged group membership",
         "detection_source": "host_auth_log"},
    )

# ─────────────────────────────────────────────
#  LINE PROCESSOR
# ─────────────────────────────────────────────
def process_line(line: str):
    m = RE_SYSLOG.match(line.strip())
    if not m:
        return
    host = m.group("host")
    msg  = m.group("msg")
    ts   = parse_ts(m.group("mon"), m.group("day"), m.group("time"))

    fm = RE_FAILED.search(msg)
    if fm:
        on_failed_login(host, fm.group("user"), fm.group("ip"), ts)
        return

    am = RE_ACCEPT.search(msg)
    if am:
        on_accepted_login(host, am.group("user"), am.group("ip"), ts)
        return

    sm = RE_SUDO.search(msg)
    if sm:
        tgt = RE_SUDO_TGT.search(msg)
        on_sudo(host, sm.group("user"), sm.group("cmd"),
                tgt.group("target") if tgt else None, ts)
        return

    if RE_SU_ROOT.search(msg):
        on_su_root(host, ts)
        return

    ua = RE_USERADD.search(msg)
    if ua:
        on_new_account(host, ua.group("name").strip(), ua.groupdict().get("uid"), ts)
        return

    ga = RE_GROUPADD.search(msg)
    if ga:
        name = ga.group("name") or ga.group("name2") or "?"
        on_group_add(host, name, ga.group("group"), ts)
        return

# ─────────────────────────────────────────────
#  FILE WATCHER  (same incremental approach as cloud_log_sensor.py)
# ─────────────────────────────────────────────
def process_log_file(filepath: Path):
    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            all_lines = f.readlines()
    except Exception as e:
        print(f"[WATCHER] Cannot read {filepath}: {e}")
        return

    key = str(filepath)
    last = processed_lines.get(key, 0)
    new_lines = all_lines[last:]
    if not new_lines:
        return

    n = 0
    for line in new_lines:
        process_line(line)
        n += 1
    processed_lines[key] = len(all_lines)
    if n:
        print(f"[WATCHER] {filepath.name} — processed {n} new log lines")


def watch_loop():
    print(f"[WATCHER] Watching directory: {LOG_WATCH_DIR.resolve()}")
    print(f"[WATCHER] Poll interval: {POLL_INTERVAL}s")
    while True:
        if LOG_WATCH_DIR.exists():
            for f in sorted(list(LOG_WATCH_DIR.glob("*.log")) +
                            list(LOG_WATCH_DIR.glob("*.txt"))):
                process_log_file(f)
        else:
            print(f"[WATCHER] Directory {LOG_WATCH_DIR} not found — will retry")
        time.sleep(POLL_INTERVAL)

# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    LOG_WATCH_DIR.mkdir(exist_ok=True)

    print("=" * 60)
    print("NIDS HOST AUTH-LOG SENSOR  (Phase 2)")
    print("=" * 60)
    print(f"  Alert endpoint : {NIDS_ALERT_URL}")
    print(f"  Log directory  : {LOG_WATCH_DIR.resolve()}")
    print(f"  Poll interval  : {POLL_INTERVAL}s")
    print()
    print("  Detections enabled (host telemetry — not visible on the wire):")
    print("    Host Brute Force        (sshd Failed password bursts)")
    print("    SSH Login / Breach      (sshd Accepted password)")
    print("    Root Access             (sudo->root, su, root login)")
    print("    Sensitive File Access   (sudo reading /etc/shadow, keys…)")
    print("    Persistence             (useradd, add to sudo/wheel)")
    print()
    print("  Drop auth.log-format files into:")
    print(f"    {LOG_WATCH_DIR.resolve()}")
    print("  or run:  python generate_host_logs.py")
    print("=" * 60)
    print()

    try:
        r = requests.get("http://127.0.0.1:8000/health", timeout=3)
        print("[+] main.py is reachable. Starting watch loop..."
              if r.status_code == 200 else
              f"[!] main.py returned HTTP {r.status_code} — alerts may fail")
    except Exception:
        print("[!] Cannot reach main.py at http://127.0.0.1:8000")
        print("    Start main.py first, or alerts will not be stored.")
    print()

    watch_loop()

"""
cloud_log_sensor.py — Log-Native Cloud IDS Sensor

Reads AWS VPC Flow Log files and detects attack patterns that
packet-based systems like Suricata cannot see.

This sensor requires NO network privileges, NO traffic mirroring,
NO cloud account. It reads standard VPC Flow Log format files
and posts alerts to main.py via the same /alert endpoint
used by live_ids_v2.py.

Run alongside main.py:
    python3 cloud_log_sensor.py

Detection capabilities (impossible with packet capture alone):
  - Impossible Travel     : same source IP seen in two distant regions rapidly
  - Port Scan (log-based) : one source hitting many destination ports
  - Data Exfiltration     : abnormally large bytes transferred to external IPs
  - API Abuse / Recon     : one source hammering many distinct destinations
  - Brute Force (log)     : many REJECT flows to auth ports from one source
  - UDP Amplification     : high-volume UDP to amplification ports
  - Lateral Movement      : internal IP contacting many other internal IPs
"""

import os
import re
import time
import threading
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import requests

# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────
NIDS_ALERT_URL   = "http://127.0.0.1:8000/alert"
SENSOR_API_KEY   = "sensor-key-change-me-in-production"  # must match main.py
ALERT_TIMEOUT    = 3

# Directory to watch for VPC Flow Log files
LOG_WATCH_DIR    = Path("cloud_logs")

# How often to scan for new/updated log files (seconds)
POLL_INTERVAL    = 5

# Internal subnet prefix — IPs matching this are "internal"
INTERNAL_PREFIX  = "192.168."

# ── Detection Thresholds ──────────────────────
PORT_SCAN_THRESHOLD     = 10    # distinct dst ports from one src in window
PORT_SCAN_WINDOW        = 60    # seconds

EXFIL_BYTES_THRESHOLD   = 50_000_000   # 50 MB from one src to external IP
EXFIL_WINDOW            = 300          # seconds

RECON_DEST_THRESHOLD    = 20    # distinct destination IPs from one src
RECON_WINDOW            = 60    # seconds

BRUTE_THRESHOLD         = 10    # REJECT flows to auth ports from one src
BRUTE_WINDOW            = 60    # seconds

LATERAL_THRESHOLD       = 8     # distinct internal destinations from one internal src
LATERAL_WINDOW          = 120   # seconds

UDP_AMP_THRESHOLD       = 30    # UDP flows to amplification ports from one src
UDP_AMP_WINDOW          = 30    # seconds

IMPOSSIBLE_TRAVEL_GAP   = 600   # seconds — same src in two "regions" within this window

# Ports considered authentication / brute-force targets
BRUTE_PORTS = {22, 23, 21, 3389, 3306, 5432, 1433, 25, 110, 143}

# UDP amplification ports
UDP_AMP_PORTS = {53, 123, 1900, 11211, 19, 17, 520, 5353}

# ─────────────────────────────────────────────
#  VPC FLOW LOG FIELD INDICES
#  AWS VPC Flow Log v2 default format:
#  version account-id interface-id srcaddr dstaddr
#  srcport dstport protocol packets bytes start end action log-status
# ─────────────────────────────────────────────
IDX_VERSION   = 0
IDX_ACCOUNT   = 1
IDX_IFACE     = 2
IDX_SRCADDR   = 3
IDX_DSTADDR   = 4
IDX_SRCPORT   = 5
IDX_DSTPORT   = 6
IDX_PROTO     = 7   # 6=TCP, 17=UDP, 1=ICMP
IDX_PACKETS   = 8
IDX_BYTES     = 9
IDX_START     = 10
IDX_END       = 11
IDX_ACTION    = 12  # ACCEPT or REJECT
IDX_STATUS    = 13  # OK, NODATA, SKIPDATA

PROTO_MAP = {"6": "tcp", "17": "udp", "1": "icmp"}

# ─────────────────────────────────────────────
#  STATE TRACKERS  (per-source sliding windows)
# ─────────────────────────────────────────────
# Each tracker: {src_ip: {"events": [...], "first_seen": float, ...}}
port_scan_tracker   = defaultdict(lambda: {"ports": set(), "timestamps": []})
exfil_tracker       = defaultdict(lambda: {"bytes": 0, "timestamps": []})
recon_tracker       = defaultdict(lambda: {"dests": set(), "timestamps": []})
brute_tracker       = defaultdict(lambda: {"timestamps": []})
lateral_tracker     = defaultdict(lambda: {"dests": set(), "timestamps": []})
udp_amp_tracker     = defaultdict(lambda: {"timestamps": []})

# Impossible travel: {src_ip: [(timestamp, region_hint), ...]}
travel_tracker      = defaultdict(list)

# Track which log lines we have already processed
# {filepath: last_line_count}
processed_lines: Dict[str, int] = {}
tracker_lock = threading.Lock()

# ─────────────────────────────────────────────
#  ALERT
# ─────────────────────────────────────────────
def send_alert(
    alert_type: str,
    severity:   str,
    src:        str,
    dst:        Optional[str],
    proto:      str,
    message:    str,
    details:    Optional[dict] = None,
):
    payload = {
        "src":        src,
        "dst":        dst,
        "proto":      proto,
        "alert_type": alert_type,
        "severity":   severity,
        "message":    message,
        "details":    details or {},
    }
    print(f"[CLOUD-ALERT] [{severity.upper()}] {alert_type} → {message}")
    try:
        resp = requests.post(
            NIDS_ALERT_URL,
            json=payload,
            timeout=ALERT_TIMEOUT,
            headers={"X-Sensor-Key": SENSOR_API_KEY},
        )
        if resp.status_code == 200:
            print(f"  ✓ Accepted by NIDS (id={resp.json().get('alert_id')})")
        else:
            print(f"  ✗ NIDS returned HTTP {resp.status_code}")
    except requests.exceptions.ConnectionError:
        print(f"  ✗ NIDS unreachable at {NIDS_ALERT_URL}. Is main.py running?")
    except Exception as e:
        print(f"  ✗ Alert error: {e}")

# ─────────────────────────────────────────────
#  HELPER
# ─────────────────────────────────────────────
def is_internal(ip: str) -> bool:
    return ip.startswith(INTERNAL_PREFIX) or ip.startswith("10.") or ip.startswith("172.")

def prune(timestamps: list, window: float, now: float) -> list:
    return [t for t in timestamps if now - t < window]

def region_hint(ip: str) -> str:
    """
    Simulate region detection from IP.
    In a real deployment this would use a GeoIP database.
    For the demo we derive a fake region from the third octet.
    """
    try:
        third = int(ip.split(".")[2])
        if third < 85:
            return "us-east-1"
        elif third < 170:
            return "eu-west-1"
        else:
            return "ap-southeast-1"
    except Exception:
        return "unknown"

# ─────────────────────────────────────────────
#  DETECTION FUNCTIONS
# ─────────────────────────────────────────────
def detect_port_scan(src: str, dst: str, dport: int, ts: float):
    t = port_scan_tracker[src]
    t["timestamps"] = prune(t["timestamps"], PORT_SCAN_WINDOW, ts)
    t["timestamps"].append(ts)
    t["ports"].add(dport)

    if len(t["ports"]) > PORT_SCAN_THRESHOLD:
        send_alert(
            "Cloud Port Scan", "high", src, dst, "tcp",
            f"Log-based port scan: {src} contacted {len(t['ports'])} distinct ports "
            f"within {PORT_SCAN_WINDOW}s (VPC Flow Logs)",
            {"ports_scanned": len(t["ports"]), "window_seconds": PORT_SCAN_WINDOW,
             "detection_source": "vpc_flow_logs"},
        )
        port_scan_tracker[src] = {"ports": set(), "timestamps": []}


def detect_exfiltration(src: str, dst: str, proto: str, bytes_sent: int, ts: float):
    if is_internal(dst):
        return  # internal transfer — not exfiltration
    t = exfil_tracker[src]
    t["timestamps"] = prune(t["timestamps"], EXFIL_WINDOW, ts)
    t["timestamps"].append(ts)
    t["bytes"] += bytes_sent

    if t["bytes"] > EXFIL_BYTES_THRESHOLD:
        mb = round(t["bytes"] / 1_000_000, 2)
        send_alert(
            "Data Exfiltration", "critical", src, dst, proto,
            f"Possible exfiltration: {src} transferred {mb} MB to external "
            f"IP {dst} within {EXFIL_WINDOW}s (VPC Flow Logs)",
            {"bytes_transferred": t["bytes"], "destination": dst,
             "window_seconds": EXFIL_WINDOW, "detection_source": "vpc_flow_logs"},
        )
        exfil_tracker[src] = {"bytes": 0, "timestamps": []}


def detect_recon(src: str, dst: str, ts: float):
    t = recon_tracker[src]
    t["timestamps"] = prune(t["timestamps"], RECON_WINDOW, ts)
    t["timestamps"].append(ts)
    t["dests"].add(dst)

    if len(t["dests"]) > RECON_DEST_THRESHOLD:
        send_alert(
            "Cloud Recon", "high", src, None, "multiple",
            f"Reconnaissance detected: {src} contacted {len(t['dests'])} distinct "
            f"destinations within {RECON_WINDOW}s (VPC Flow Logs)",
            {"distinct_destinations": len(t["dests"]), "window_seconds": RECON_WINDOW,
             "detection_source": "vpc_flow_logs"},
        )
        recon_tracker[src] = {"dests": set(), "timestamps": []}


def detect_brute_force_log(src: str, dst: str, dport: int, ts: float):
    if dport not in BRUTE_PORTS:
        return
    t = brute_tracker[src]
    t["timestamps"] = prune(t["timestamps"], BRUTE_WINDOW, ts)
    t["timestamps"].append(ts)

    if len(t["timestamps"]) >= BRUTE_THRESHOLD:
        send_alert(
            "Cloud Brute Force", "high", src, dst, "tcp",
            f"Brute force in flow logs: {src} → {dst}:{dport}, "
            f"{len(t['timestamps'])} REJECT flows in {BRUTE_WINDOW}s (VPC Flow Logs)",
            {"port": dport, "reject_count": len(t["timestamps"]),
             "window_seconds": BRUTE_WINDOW, "detection_source": "vpc_flow_logs"},
        )
        brute_tracker[src] = {"timestamps": []}


def detect_lateral_movement(src: str, dst: str, ts: float):
    if not is_internal(src) or not is_internal(dst):
        return
    t = lateral_tracker[src]
    t["timestamps"] = prune(t["timestamps"], LATERAL_WINDOW, ts)
    t["timestamps"].append(ts)
    t["dests"].add(dst)

    if len(t["dests"]) >= LATERAL_THRESHOLD:
        send_alert(
            "Lateral Movement", "critical", src, None, "tcp",
            f"Lateral movement detected: internal host {src} contacted "
            f"{len(t['dests'])} distinct internal destinations within "
            f"{LATERAL_WINDOW}s (VPC Flow Logs)",
            {"internal_targets": len(t["dests"]), "window_seconds": LATERAL_WINDOW,
             "detection_source": "vpc_flow_logs"},
        )
        lateral_tracker[src] = {"dests": set(), "timestamps": []}


def detect_udp_amplification_log(src: str, dst: str, dport: int, ts: float):
    if dport not in UDP_AMP_PORTS:
        return
    t = udp_amp_tracker[src]
    t["timestamps"] = prune(t["timestamps"], UDP_AMP_WINDOW, ts)
    t["timestamps"].append(ts)

    if len(t["timestamps"]) >= UDP_AMP_THRESHOLD:
        send_alert(
            "Cloud UDP Amplification", "high", src, dst, "udp",
            f"UDP amplification in flow logs: {src} → {dst}:{dport}, "
            f"{len(t['timestamps'])} flows in {UDP_AMP_WINDOW}s (VPC Flow Logs)",
            {"dst_port": dport, "flow_count": len(t["timestamps"]),
             "window_seconds": UDP_AMP_WINDOW, "detection_source": "vpc_flow_logs"},
        )
        udp_amp_tracker[src] = {"timestamps": []}


def detect_impossible_travel(src: str, ts: float):
    region = region_hint(src)
    events = travel_tracker[src]

    # Prune old events outside the impossible travel window
    events = [(t, r) for t, r in events if ts - t < IMPOSSIBLE_TRAVEL_GAP]
    events.append((ts, region))
    travel_tracker[src] = events

    # Check if we see two different regions within the window
    regions_seen = set(r for _, r in events)
    if len(regions_seen) >= 2:
        send_alert(
            "Impossible Travel", "critical", src, None, "multiple",
            f"Impossible travel: source {src} appeared in regions "
            f"{', '.join(regions_seen)} within {IMPOSSIBLE_TRAVEL_GAP}s "
            f"(VPC Flow Logs)",
            {"regions": list(regions_seen), "window_seconds": IMPOSSIBLE_TRAVEL_GAP,
             "detection_source": "vpc_flow_logs"},
        )
        travel_tracker[src] = []  # reset after alert

# ─────────────────────────────────────────────
#  FLOW LOG PARSER
# ─────────────────────────────────────────────
def parse_flow_line(line: str) -> Optional[dict]:
    """
    Parse a single VPC Flow Log line into a dict.
    Returns None if the line is a header, comment, or malformed.
    """
    line = line.strip()
    if not line or line.startswith("version") or line.startswith("#"):
        return None

    fields = line.split()
    if len(fields) < 14:
        return None

    try:
        return {
            "src":     fields[IDX_SRCADDR],
            "dst":     fields[IDX_DSTADDR],
            "sport":   int(fields[IDX_SRCPORT]),
            "dport":   int(fields[IDX_DSTPORT]),
            "proto":   PROTO_MAP.get(fields[IDX_PROTO], "other"),
            "packets": int(fields[IDX_PACKETS]),
            "bytes":   int(fields[IDX_BYTES]),
            "start":   float(fields[IDX_START]),
            "end":     float(fields[IDX_END]),
            "action":  fields[IDX_ACTION],   # ACCEPT or REJECT
            "status":  fields[IDX_STATUS],
        }
    except (ValueError, IndexError):
        return None


def process_flow(flow: dict):
    """Run all detections on a single parsed flow record."""
    src    = flow["src"]
    dst    = flow["dst"]
    dport  = flow["dport"]
    proto  = flow["proto"]
    action = flow["action"]
    ts     = flow["start"]
    sent   = flow["bytes"]

    # Skip flows with no data
    if flow["status"] == "NODATA":
        return

    # Impossible travel (runs on all flows from external IPs)
    if not is_internal(src):
        detect_impossible_travel(src, ts)

    # Port scan (ACCEPT flows — attacker is reaching open ports)
    if proto == "tcp" and action == "ACCEPT":
        detect_port_scan(src, dst, dport, ts)

    # Data exfiltration (large outbound transfers to external)
    if sent > 0:
        detect_exfiltration(src, dst, proto, sent, ts)

    # Reconnaissance (many distinct destinations)
    detect_recon(src, dst, ts)

    # Brute force (REJECT flows to auth ports)
    if proto == "tcp" and action == "REJECT":
        detect_brute_force_log(src, dst, dport, ts)

    # Lateral movement (internal → internal)
    if proto == "tcp" and action == "ACCEPT":
        detect_lateral_movement(src, dst, ts)

    # UDP amplification
    if proto == "udp":
        detect_udp_amplification_log(src, dst, dport, ts)

# ─────────────────────────────────────────────
#  FILE WATCHER
# ─────────────────────────────────────────────
def process_log_file(filepath: Path):
    """
    Read new lines from a log file since last check.
    Tracks how many lines were processed per file.
    """
    global processed_lines

    try:
        with open(filepath, "r") as f:
            all_lines = f.readlines()
    except Exception as e:
        print(f"[WATCHER] Cannot read {filepath}: {e}")
        return

    key        = str(filepath)
    last_count = processed_lines.get(key, 0)
    new_lines  = all_lines[last_count:]

    if not new_lines:
        return

    new_flows = 0
    for line in new_lines:
        flow = parse_flow_line(line)
        if flow:
            process_flow(flow)
            new_flows += 1

    processed_lines[key] = len(all_lines)

    if new_flows > 0:
        print(f"[WATCHER] {filepath.name} — processed {new_flows} new flow records")


def watch_loop():
    """
    Main polling loop. Scans LOG_WATCH_DIR for .log and .txt files
    every POLL_INTERVAL seconds and processes any new lines.
    """
    print(f"[WATCHER] Watching directory: {LOG_WATCH_DIR.resolve()}")
    print(f"[WATCHER] Poll interval: {POLL_INTERVAL}s")

    while True:
        if LOG_WATCH_DIR.exists():
            log_files = list(LOG_WATCH_DIR.glob("*.log")) + \
                        list(LOG_WATCH_DIR.glob("*.txt"))
            for f in sorted(log_files):
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
    print("NIDS CLOUD LOG SENSOR")
    print("=" * 60)
    print(f"  Alert endpoint : {NIDS_ALERT_URL}")
    print(f"  Log directory  : {LOG_WATCH_DIR.resolve()}")
    print(f"  Poll interval  : {POLL_INTERVAL}s")
    print()
    print("  Detections enabled:")
    print("    Cloud Port Scan       (VPC Flow Logs)")
    print("    Data Exfiltration     (VPC Flow Logs)")
    print("    Cloud Recon           (VPC Flow Logs)")
    print("    Cloud Brute Force     (VPC Flow Logs — REJECT flows)")
    print("    Lateral Movement      (VPC Flow Logs — internal→internal)")
    print("    Cloud UDP Amplif.     (VPC Flow Logs)")
    print("    Impossible Travel     (VPC Flow Logs — multi-region)")
    print()
    print("  Drop .log or .txt files (VPC Flow Log format) into:")
    print(f"  {LOG_WATCH_DIR.resolve()}")
    print("=" * 60)
    print()

    # Verify main.py is reachable before starting
    try:
        r = requests.get("http://127.0.0.1:8000/health", timeout=3)
        if r.status_code == 200:
            print("[+] main.py is reachable. Starting watch loop...")
        else:
            print(f"[!] main.py returned HTTP {r.status_code} — alerts may fail")
    except Exception:
        print("[!] Cannot reach main.py at http://127.0.0.1:8000")
        print("    Start main.py first, or alerts will not be stored.")
    print()

    watch_loop()

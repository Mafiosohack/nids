"""
NIDS ML SENSOR — live_ids_v2.py
Flow-based anomaly detection using a trained Random Forest model.

Responsibilities:
  - Capture packets on the network interface
  - Build per-flow feature vectors (NSL-KDD style)
  - Run ML inference on mature flows
  - POST alerts to main.py via /alert endpoint

Run this ALONGSIDE main.py (separate process/terminal).
main.py handles rule-based detection; this handles ML detection.

Feature vector layout (matches NSL-KDD training schema):
  Numeric  [38]:  duration, src_bytes, dst_bytes, land, wrong_fragment,
                  urgent, hot, count, serror_rate, rerror_rate, ...
  Categorical [3]: protocol_type, service, flag
  → encoder.transform([[proto, service, flag]]) appended to numeric
"""

import time
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import requests

try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    raise SystemExit("[FATAL] joblib not installed. Run: pip install joblib")

try:
    from scapy.all import IP, TCP, UDP, sniff
    SCAPY_AVAILABLE = True
except ImportError:
    raise SystemExit("[FATAL] scapy not installed. Run: pip install scapy")

# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────
NETWORK_INTERFACE   = "ens37"
MODEL_PATH          = "models/random_forest_v2.pkl"
ENCODER_PATH        = "models/rf_encoder_v2.pkl"
NIDS_ALERT_URL      = "http://127.0.0.1:8000/alert"
ALERT_TIMEOUT_SEC   = 2         # HTTP POST timeout

FLOW_MIN_PACKETS    = 5         # minimum packets before running inference
FLOW_MAX_AGE_SEC    = 120       # expire flows older than this (prevents memory leak)
FLOW_CLEANUP_INTERVAL = 30      # how often the cleanup thread runs

# Labels that the model considers "normal" — covers int and string variants
NORMAL_LABELS = {"normal", "benign", "0", 0, "legitimate"}

# ─────────────────────────────────────────────
#  PORT → SERVICE MAPPING  (NSL-KDD style)
# ─────────────────────────────────────────────
# Maps destination port → service string used during training.
# Extend this if your model was trained with additional service labels.
PORT_SERVICE_MAP: Dict[int, str] = {
    20:   "ftp_data",
    21:   "ftp",
    22:   "ssh",
    23:   "telnet",
    25:   "smtp",
    53:   "domain",
    67:   "domain_u",    # DHCP uses UDP; treat as domain_u
    68:   "domain_u",
    80:   "http",
    110:  "pop_3",
    111:  "sunrpc",
    113:  "auth",
    119:  "nntp",
    123:  "ntp_u",
    143:  "imap4",
    179:  "bgp",
    194:  "IRC",
    389:  "ldap",
    443:  "http_443",
    445:  "netbios_ssn",
    465:  "smtp",
    514:  "shell",
    515:  "printer",
    587:  "smtp",
    993:  "imap4",
    995:  "pop_3",
    1433: "mssql",
    1521: "sql_net",
    1900: "other",
    3306: "sql_net",
    3389: "other",
    5432: "sql_net",
    5900: "other",
    6667: "IRC",
    8080: "http_8001",
    8443: "http_443",
    9200: "other",
    11211:"other",
}

def port_to_service(port: int, proto: str) -> str:
    """Map a destination port to an NSL-KDD service label."""
    if proto == "icmp":
        return "eco_i"
    return PORT_SERVICE_MAP.get(port, "other")

# ─────────────────────────────────────────────
#  TCP FLAGS → KDD FLAG  (connection state)
# ─────────────────────────────────────────────
# KDD flag represents the *final* state of a TCP connection.
# We derive it from what flags were observed in the flow.

def derive_kdd_flag(
    seen_syn: bool,
    seen_syn_ack: bool,
    seen_fin: bool,
    seen_rst_src: bool,
    seen_rst_dst: bool,
) -> str:
    """
    Map observed TCP flags to an NSL-KDD connection flag.

    SF  : Normal establish + termination (SYN, SYN-ACK, FIN both sides)
    S0  : SYN only — no response at all
    S1  : SYN + SYN-ACK, but no FIN/RST yet (still open)
    REJ : Connection rejected (RST received without prior SYN-ACK)
    RSTO: Connection established, then RST from originator
    RSTR: Connection established, then RST from responder
    OTH : None of the above
    """
    if seen_fin and seen_syn_ack:
        return "SF"
    if seen_rst_src and seen_syn_ack:
        return "RSTO"
    if seen_rst_dst and seen_syn_ack:
        return "RSTR"
    if seen_rst_dst and not seen_syn_ack:
        return "REJ"
    if seen_syn and seen_syn_ack:
        return "S1"
    if seen_syn and not seen_syn_ack:
        return "S0"
    return "OTH"

# ─────────────────────────────────────────────
#  FLOW RECORD
# ─────────────────────────────────────────────
@dataclass
class Flow:
    src: str
    dst: str
    proto: str
    dst_port: int
    start_time: float = field(default_factory=time.time)

    # Byte counters
    src_bytes: int = 0
    dst_bytes: int = 0

    # Packet count
    count: int = 0

    # TCP state flags (tracked across packets)
    seen_syn:     bool = False
    seen_syn_ack: bool = False
    seen_fin:     bool = False
    seen_rst_src: bool = False
    seen_rst_dst: bool = False

    # Error counters (for rate features)
    syn_errors: int = 0   # S0-type: SYN with no response
    rej_errors: int = 0   # REJ-type: RST without prior connection

    # Set to True once inference has run on this flow
    inferenced: bool = False

    @property
    def duration(self) -> float:
        return time.time() - self.start_time

    @property
    def land(self) -> int:
        """1 if source and destination IP are the same (loopback attack indicator)."""
        return int(self.src == self.dst)

    @property
    def kdd_flag(self) -> str:
        return derive_kdd_flag(
            self.seen_syn, self.seen_syn_ack, self.seen_fin,
            self.seen_rst_src, self.seen_rst_dst,
        )

    @property
    def service(self) -> str:
        return port_to_service(self.dst_port, self.proto)

    def update_tcp_flags(self, flags: int, direction: str):
        """Update connection state from a packet's TCP flags."""
        syn = bool(flags & 0x02)
        ack = bool(flags & 0x10)
        fin = bool(flags & 0x01)
        rst = bool(flags & 0x04)

        if syn and not ack:
            self.seen_syn = True
        if syn and ack:
            self.seen_syn_ack = True
        if fin:
            self.seen_fin = True
        if rst:
            if direction == "src":
                self.seen_rst_src = True
            else:
                self.seen_rst_dst = True
                if not self.seen_syn_ack:
                    self.rej_errors += 1

    def build_feature_vector(self) -> np.ndarray:
        """
        Build a 38-element numeric feature array (NSL-KDD layout).

        Features we CAN compute from live packets are populated.
        Features requiring application-layer inspection (hot, logged_in,
        num_failed_logins, etc.) are zeroed — they weren't observable
        from raw packets anyway.
        """
        total = max(self.count, 1)
        numeric = np.zeros(38, dtype=np.float64)

        # ── Well-supported features ──────────────────
        numeric[0]  = self.duration
        numeric[1]  = self.src_bytes
        numeric[2]  = self.dst_bytes
        numeric[3]  = self.land
        numeric[4]  = 0              # wrong_fragment (needs IP frag tracking)
        numeric[5]  = 0              # urgent (TCP URG — extend if needed)

        # ── Application-layer features (zeroed) ──────
        # numeric[6]  = hot
        # numeric[7]  = num_failed_logins
        # numeric[8]  = logged_in
        # numeric[9]  = num_compromised
        # ... (10-18 are application layer, left as 0)

        # ── Traffic / connection count features ──────
        numeric[19] = self.count
        numeric[20] = self.count     # srv_count approximation

        # Error rates
        numeric[21] = self.syn_errors / total   # serror_rate
        numeric[22] = self.syn_errors / total   # srv_serror_rate (approx)
        numeric[23] = self.rej_errors / total   # rerror_rate
        numeric[24] = self.rej_errors / total   # srv_rerror_rate (approx)

        # Service rate features (set to 1.0 since we're per-flow here)
        numeric[25] = 1.0   # same_srv_rate
        numeric[26] = 0.0   # diff_srv_rate

        # ── dst_host_* features (28-37) ──────────────
        # These require per-destination historical tracking.
        # Set conservatively — zeros won't trigger false positives.
        # If your model was trained to rely on these heavily, implement
        # a destination-host sliding-window counter here.

        return numeric.reshape(1, -1)

# ─────────────────────────────────────────────
#  FLOW TABLE
# ─────────────────────────────────────────────
FlowKey = Tuple[str, str, str]   # (src_ip, dst_ip, proto)

flow_table: Dict[FlowKey, Flow] = {}
flow_lock = threading.Lock()


def get_or_create_flow(src: str, dst: str, proto: str, dst_port: int) -> Flow:
    key = (src, dst, proto)
    with flow_lock:
        if key not in flow_table:
            flow_table[key] = Flow(src=src, dst=dst, proto=proto, dst_port=dst_port)
        return flow_table[key]


def cleanup_expired_flows():
    """Background thread: remove flows older than FLOW_MAX_AGE_SEC."""
    while True:
        time.sleep(FLOW_CLEANUP_INTERVAL)
        cutoff = time.time() - FLOW_MAX_AGE_SEC
        with flow_lock:
            expired = [k for k, f in flow_table.items() if f.start_time < cutoff]
            for k in expired:
                del flow_table[k]
        if expired:
            print(f"[CLEANUP] Removed {len(expired)} expired flows. Active: {len(flow_table)}")

# ─────────────────────────────────────────────
#  MODEL
# ─────────────────────────────────────────────
model   = None
encoder = None

def load_model():
    global model, encoder
    print("[ML] Loading model and encoder...")
    try:
        model   = joblib.load(MODEL_PATH)
        encoder = joblib.load(ENCODER_PATH)
        print("[ML] Model and encoder loaded successfully.")
    except FileNotFoundError as e:
        raise SystemExit(f"[FATAL] Could not load model/encoder: {e}")
    except Exception as e:
        raise SystemExit(f"[FATAL] Model load error: {e}")


def run_inference(flow: Flow) -> Optional[str]:
    """
    Build features, run model prediction, return label string.
    Returns None on error.
    """
    try:
        numeric    = flow.build_feature_vector()    # shape (1, 38)
        cat_data   = np.array([[flow.proto, flow.service, flow.kdd_flag]])
        cat_encoded = encoder.transform(cat_data)    # shape (1, N)
        features   = np.hstack((numeric, cat_encoded))

        raw_pred   = model.predict(features)[0]

        # Normalise label to lowercase string for consistent comparison
        label = str(raw_pred).strip().lower()
        print(f"[ML] {flow.src} → {flow.dst} | proto={flow.proto} "
              f"svc={flow.service} flag={flow.kdd_flag} | pred='{label}'")
        return label

    except Exception as e:
        print(f"[ML] Inference error for flow {flow.src}→{flow.dst}: {e}")
        return None

# ─────────────────────────────────────────────
#  ALERT
# ─────────────────────────────────────────────
def send_alert(flow: Flow, label: str):
    payload = {
        "src":        flow.src,
        "dst":        flow.dst,
        "proto":      flow.proto,
        "alert_type": "ML Anomaly",
        "severity":   "high",
        "message":    (
            f"ML model flagged flow as '{label}': "
            f"{flow.src} → {flow.dst} ({flow.proto}/{flow.service}), "
            f"{flow.count} pkts, {flow.src_bytes} bytes"
        ),
        "details": {
            "predicted_label": label,
            "service":         flow.service,
            "kdd_flag":        flow.kdd_flag,
            "duration":        round(flow.duration, 3),
            "src_bytes":       flow.src_bytes,
            "dst_bytes":       flow.dst_bytes,
            "packet_count":    flow.count,
        },
    }
    print(f"[ALERT] Sending to NIDS: {payload['message']}")
    try:
        resp = requests.post(NIDS_ALERT_URL, json=payload, timeout=ALERT_TIMEOUT_SEC)
        if resp.status_code == 200:
            print(f"[ALERT] ✓ Accepted by NIDS (id={resp.json().get('alert_id')})")
        else:
            print(f"[ALERT] ✗ NIDS returned HTTP {resp.status_code}: {resp.text}")
    except requests.exceptions.ConnectionError:
        print(f"[ALERT] ✗ NIDS unreachable at {NIDS_ALERT_URL}. Is main.py running?")
    except requests.exceptions.Timeout:
        print(f"[ALERT] ✗ NIDS POST timed out after {ALERT_TIMEOUT_SEC}s")

# ─────────────────────────────────────────────
#  PACKET PROCESSOR
# ─────────────────────────────────────────────
def process_packet(pkt):
    if IP not in pkt:
        return

    src   = pkt[IP].src
    dst   = pkt[IP].dst
    proto = "tcp" if TCP in pkt else "udp" if UDP in pkt else "icmp"

    # Determine destination port for service mapping
    dst_port = 0
    if TCP in pkt:
        dst_port = pkt[TCP].dport
    elif UDP in pkt:
        dst_port = pkt[UDP].dport

    flow = get_or_create_flow(src, dst, proto, dst_port)

    # ── Update flow metrics ───────────────────
    flow.src_bytes += len(pkt)
    flow.count     += 1

    if TCP in pkt:
        flags = int(pkt[TCP].flags)
        flow.update_tcp_flags(flags, direction="src")

    # ── Run inference once flow is mature ─────
    if flow.count >= FLOW_MIN_PACKETS and not flow.inferenced:
        flow.inferenced = True   # mark before inference to avoid race on next packet
        label = run_inference(flow)

        if label is not None and label not in NORMAL_LABELS:
            send_alert(flow, label)

# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    load_model()

    # Start background flow cleanup thread
    cleanup_thread = threading.Thread(target=cleanup_expired_flows, daemon=True)
    cleanup_thread.start()
    print(f"[+] Flow cleanup thread started (interval={FLOW_CLEANUP_INTERVAL}s)")

    print(f"[+] Monitoring interface: {NETWORK_INTERFACE}")
    print(f"[+] Inference triggers at {FLOW_MIN_PACKETS} packets per flow")
    print(f"[+] Alerts → {NIDS_ALERT_URL}")
    print("─" * 60)

    sniff(iface=NETWORK_INTERFACE, prn=process_packet, store=False)

"""
NIDS ML SENSOR — live_ids_v2.py
Flow-based anomaly detection using the live-aligned Random Forest model.

Responsibilities:
  - Capture packets on the network interface
  - Build per-flow feature vectors using ONLY packet-derivable features
    (feature_engineering.live_features.LIVE_FEATURES) so the live vector
    matches exactly what the model was trained on (see train_random_forest_v2.py)
  - Run ML inference on mature flows
  - POST alerts to main.py via /alert endpoint

Run this ALONGSIDE main.py (separate process/terminal).
main.py handles rule-based detection; this handles ML detection.

Feature alignment:
  The model (models/rf_live.pkl) is trained on the live-computable subset of
  NSL-KDD. The 10 numeric features in LIVE_NUMERIC_FEATURES are computed per
  flow below; the 3 categorical features (protocol_type, service, flag) are
  one-hot encoded by models/rf_live_encoder.pkl. Train and serve use the same
  ordered feature list, so there is no distribution mismatch.
"""

import json
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import requests

# Make project packages importable when run directly from the repo root.
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from feature_engineering.live_features import (
    LIVE_NUMERIC_FEATURES,
    LIVE_CATEGORICAL_FEATURES,
    LIVE_MODEL_PATH,
    LIVE_ENCODER_PATH,
    LIVE_FEATURE_META_PATH,
)

try:
    import joblib
except ImportError:
    raise SystemExit("[FATAL] joblib not installed. Run: pip install joblib")

# Scapy is only needed for live capture. Keep the import graceful so this module
# can be imported for offline testing / on hosts without a capture stack.
try:
    from scapy.all import IP, TCP, UDP, sniff
    SCAPY_AVAILABLE = True
except ImportError:
    SCAPY_AVAILABLE = False
    IP = TCP = UDP = None  # type: ignore

# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────
NETWORK_INTERFACE   = "ens37"
MODEL_PATH          = str(ROOT / LIVE_MODEL_PATH)
ENCODER_PATH        = str(ROOT / LIVE_ENCODER_PATH)
META_PATH           = str(ROOT / LIVE_FEATURE_META_PATH)
NIDS_ALERT_URL      = "http://127.0.0.1:8000/alert"
ALERT_TIMEOUT_SEC   = 2         # HTTP POST timeout

# Must match SENSOR_API_KEY in main.py
SENSOR_API_KEY      = "sensor-key-change-me-in-production"

FLOW_MIN_PACKETS    = 5         # minimum packets before running inference
FLOW_MAX_AGE_SEC    = 120       # expire flows older than this (prevents memory leak)
FLOW_CLEANUP_INTERVAL = 30      # how often the cleanup thread runs

HIGH_CONFIDENCE     = 0.80      # attack probability above this → severity "high"

# Labels the model considers "normal" (the live-aligned model emits "normal"/"attack").
NORMAL_LABELS = {"normal", "benign", "0", 0, "legitimate"}

# ── Runtime mode (set by main() based on CLI args) ──
# Live capture infers inline as flows mature and POSTs alerts to main.py.
# PCAP replay evaluates each complete flow once at the end and, by default,
# prints a report instead of POSTing (so it runs standalone without main.py).
MIN_PACKETS       = FLOW_MIN_PACKETS
INLINE_INFERENCE  = True
SEND_ALERTS       = True


# ─────────────────────────────────────────────
#  PORT → SERVICE MAPPING  (NSL-KDD style)
# ─────────────────────────────────────────────
PORT_SERVICE_MAP: Dict[int, str] = {
    20:   "ftp_data",
    21:   "ftp",
    22:   "ssh",
    23:   "telnet",
    25:   "smtp",
    53:   "domain",
    67:   "domain_u",
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
def derive_kdd_flag(
    seen_syn: bool,
    seen_syn_ack: bool,
    seen_fin: bool,
    seen_rst_src: bool,
    seen_rst_dst: bool,
) -> str:
    """Map observed TCP flags to an NSL-KDD connection flag.

    SF  : Normal establish + termination (SYN, SYN-ACK, FIN)
    S0  : SYN only — no response at all
    S1  : SYN + SYN-ACK, but no FIN/RST yet (still open)
    REJ : Connection rejected (RST from responder, no prior SYN-ACK)
    RSTO: Established, then RST from originator
    RSTR: Established, then RST from responder
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
#  FLOW RECORD  (bidirectional)
# ─────────────────────────────────────────────
@dataclass
class Flow:
    """A bidirectional flow between two endpoints.

    Endpoints are stored canonically (ip_a <= ip_b) so traffic in both
    directions maps to a single Flow. `initiator` is the IP we treat as the
    connection source (first packet seen, ideally the SYN sender), which lets
    us attribute src_bytes vs dst_bytes and originator vs responder RSTs.
    """
    ip_a: str
    ip_b: str
    proto: str
    dst_port: int
    initiator: str
    start_time: float = field(default_factory=time.time)

    src_bytes: int = 0       # initiator → responder
    dst_bytes: int = 0       # responder → initiator
    count: int = 0           # packets observed in the flow

    first_ts: Optional[float] = None   # first packet timestamp (drives duration)
    last_ts:  Optional[float] = None   # most recent packet timestamp

    seen_syn:     bool = False   # initiator SYN (no ACK)
    seen_syn_ack: bool = False   # responder SYN-ACK
    seen_fin:     bool = False
    seen_rst_src: bool = False   # RST from initiator
    seen_rst_dst: bool = False   # RST from responder

    inferenced: bool = False

    @property
    def src(self) -> str:
        return self.initiator

    @property
    def dst(self) -> str:
        return self.ip_b if self.initiator == self.ip_a else self.ip_a

    @property
    def duration(self) -> float:
        """Flow duration from packet timestamps (works in both live and replay).

        Using packet timestamps rather than wall-clock means a pcap replayed in
        milliseconds still reports the flow's true on-the-wire duration.
        """
        if self.first_ts is None or self.last_ts is None:
            return 0.0
        return max(0.0, self.last_ts - self.first_ts)

    def observe_time(self, ts: float):
        """Record a packet timestamp for duration accounting."""
        if self.first_ts is None:
            self.first_ts = ts
        self.last_ts = ts

    @property
    def land(self) -> int:
        """1 if source and destination IP are the same (LAND-attack indicator)."""
        return int(self.ip_a == self.ip_b)

    @property
    def kdd_flag(self) -> str:
        return derive_kdd_flag(
            self.seen_syn, self.seen_syn_ack, self.seen_fin,
            self.seen_rst_src, self.seen_rst_dst,
        )

    @property
    def service(self) -> str:
        return port_to_service(self.dst_port, self.proto)

    def update_tcp_flags(self, flags: int, from_initiator: bool):
        """Update connection state from a packet's TCP flags.

        from_initiator: True if the packet was sent by the flow initiator.
        """
        syn = bool(flags & 0x02)
        ack = bool(flags & 0x10)
        fin = bool(flags & 0x01)
        rst = bool(flags & 0x04)

        if syn and not ack:
            self.seen_syn = True       # SYN typically from initiator
        if syn and ack:
            self.seen_syn_ack = True   # SYN-ACK from responder
        if fin:
            self.seen_fin = True
        if rst:
            if from_initiator:
                self.seen_rst_src = True
            else:
                self.seen_rst_dst = True

    def build_feature_vector(self) -> np.ndarray:
        """Emit the live-computable numeric features in LIVE_NUMERIC_FEATURES order.

        Returns shape (1, len(LIVE_NUMERIC_FEATURES)).
        serror/rerror are derived from the flow's final TCP state — the honest
        live analogue of NSL-KDD's windowed error rates.
        """
        serror = 1.0 if (self.seen_syn and not self.seen_syn_ack) else 0.0
        rerror = 1.0 if self.kdd_flag in ("REJ", "RSTO", "RSTR") else 0.0

        values = {
            "duration":        self.duration,
            "src_bytes":       float(self.src_bytes),
            "dst_bytes":       float(self.dst_bytes),
            "land":            float(self.land),
            "count":           float(self.count),
            "srv_count":       float(self.count),   # service-count approximation
            "serror_rate":     serror,
            "srv_serror_rate": serror,
            "rerror_rate":     rerror,
            "srv_rerror_rate": rerror,
        }
        row = [values[name] for name in LIVE_NUMERIC_FEATURES]
        return np.array([row], dtype=np.float64)


# ─────────────────────────────────────────────
#  FLOW TABLE
# ─────────────────────────────────────────────
FlowKey = Tuple[str, str, str]   # (ip_a, ip_b, proto) with ip_a <= ip_b

flow_table: Dict[FlowKey, Flow] = {}
flow_lock = threading.Lock()


def _flow_key(src: str, dst: str, proto: str) -> FlowKey:
    a, b = sorted((src, dst))
    return (a, b, proto)


def get_or_create_flow(src: str, dst: str, proto: str, dst_port: int) -> Flow:
    key = _flow_key(src, dst, proto)
    with flow_lock:
        flow = flow_table.get(key)
        if flow is None:
            flow = Flow(
                ip_a=key[0], ip_b=key[1], proto=proto,
                dst_port=dst_port, initiator=src,
            )
            flow_table[key] = flow
        return flow


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
            print(f"[CLEANUP] Removed {len(expired)} expired flows. "
                  f"Active: {len(flow_table)}")


# ─────────────────────────────────────────────
#  MODEL
# ─────────────────────────────────────────────
model = None
encoder = None
_attack_index = 1          # index of the "attack" class in model.classes_
DECISION_THRESHOLD = 0.5   # overridden from model metadata in load_model()


def load_model():
    """Load the live-aligned model + encoder and validate feature alignment."""
    global model, encoder, _attack_index, DECISION_THRESHOLD
    print("[ML] Loading live-aligned model and encoder...")
    try:
        model = joblib.load(MODEL_PATH)
        encoder = joblib.load(ENCODER_PATH)
    except FileNotFoundError as e:
        raise SystemExit(
            f"[FATAL] Could not load model/encoder: {e}\n"
            f"        Train it first:  python train_random_forest_v2.py"
        )
    except Exception as e:
        raise SystemExit(f"[FATAL] Model load error: {e}")

    # Resolve which class index corresponds to "attack" (1).
    classes = list(getattr(model, "classes_", [0, 1]))
    _attack_index = classes.index(1) if 1 in classes else len(classes) - 1

    # Validate that the feature width we will emit matches the model.
    n_cat = sum(len(c) for c in encoder.categories_)
    expected = len(LIVE_NUMERIC_FEATURES) + n_cat
    n_in = getattr(model, "n_features_in_", expected)
    if n_in != expected:
        print(f"[ML][WARN] Feature width mismatch: model expects {n_in}, "
              f"sensor emits {expected}. Retrain with train_random_forest_v2.py.")
    else:
        print(f"[ML] Feature alignment OK ({expected} features: "
              f"{len(LIVE_NUMERIC_FEATURES)} numeric + {n_cat} one-hot).")

    # Apply the tuned decision threshold and surface the model's honest score.
    try:
        with open(META_PATH) as f:
            meta = json.load(f)
        DECISION_THRESHOLD = float(meta.get("decision_threshold", 0.5))
        print(f"[ML] Model KDDTest+ accuracy={meta.get('kddtest_accuracy'):.3f} "
              f"F1(attack)={meta.get('kddtest_f1_attack'):.3f} "
              f"FPR={meta.get('kddtest_fpr'):.3f}")
        print(f"[ML] Decision threshold: {DECISION_THRESHOLD:.2f} "
              f"(attack if p >= threshold)")
    except (FileNotFoundError, TypeError, ValueError):
        pass

    print("[ML] Model and encoder loaded successfully.")


def run_inference(flow: Flow) -> Tuple[Optional[str], Optional[float]]:
    """Build the live-aligned feature vector, predict, and return
    (label, attack_probability). Returns (None, None) on error."""
    try:
        numeric = flow.build_feature_vector()                      # (1, N_num)
        cat_values = {
            "protocol_type": flow.proto,
            "service":       flow.service,
            "flag":          flow.kdd_flag,
        }
        cat_row = np.array([[cat_values[c] for c in LIVE_CATEGORICAL_FEATURES]])
        cat_encoded = encoder.transform(cat_row)                   # (1, N_cat)
        features = np.hstack([numeric, cat_encoded])

        attack_prob = None
        if hasattr(model, "predict_proba"):
            attack_prob = float(model.predict_proba(features)[0][_attack_index])
            is_attack = attack_prob >= DECISION_THRESHOLD
        else:
            is_attack = int(model.predict(features)[0]) == 1

        label = "attack" if is_attack else "normal"
        conf = f" p={attack_prob:.2f}" if attack_prob is not None else ""
        print(f"[ML] {flow.src} → {flow.dst} | proto={flow.proto} "
              f"svc={flow.service} flag={flow.kdd_flag} | pred='{label}'{conf}")
        return label, attack_prob

    except Exception as e:
        print(f"[ML] Inference error for flow {flow.src}→{flow.dst}: {e}")
        return None, None


# ─────────────────────────────────────────────
#  ALERT
# ─────────────────────────────────────────────
def send_alert(flow: Flow, label: str, attack_prob: Optional[float]):
    severity = "high" if (attack_prob is not None and attack_prob >= HIGH_CONFIDENCE) else "medium"
    conf_txt = f"{attack_prob:.0%} confidence" if attack_prob is not None else "n/a"

    payload = {
        "src":        flow.src,
        "dst":        flow.dst,
        "proto":      flow.proto,
        "alert_type": "ML Anomaly",
        "severity":   severity,
        "message":    (
            f"ML model flagged flow as '{label}' ({conf_txt}): "
            f"{flow.src} → {flow.dst} ({flow.proto}/{flow.service}), "
            f"{flow.count} pkts, {flow.src_bytes}↑/{flow.dst_bytes}↓ bytes"
        ),
        "details": {
            "predicted_label": label,
            "attack_probability": round(attack_prob, 4) if attack_prob is not None else None,
            "service":         flow.service,
            "kdd_flag":        flow.kdd_flag,
            "duration":        round(flow.duration, 3),
            "src_bytes":       flow.src_bytes,
            "dst_bytes":       flow.dst_bytes,
            "packet_count":    flow.count,
        },
    }
    print(f"[ALERT] Sending to NIDS [{severity}]: {payload['message']}")
    try:
        resp = requests.post(
            NIDS_ALERT_URL,
            json=payload,
            timeout=ALERT_TIMEOUT_SEC,
            headers={"X-Sensor-Key": SENSOR_API_KEY},
        )
        if resp.status_code == 200:
            print(f"[ALERT] OK Accepted by NIDS (id={resp.json().get('alert_id')})")
        else:
            print(f"[ALERT] !! NIDS returned HTTP {resp.status_code}: {resp.text}")
    except requests.exceptions.ConnectionError:
        print(f"[ALERT] !! NIDS unreachable at {NIDS_ALERT_URL}. Is main.py running?")
    except requests.exceptions.Timeout:
        print(f"[ALERT] !! NIDS POST timed out after {ALERT_TIMEOUT_SEC}s")


# ─────────────────────────────────────────────
#  DISPATCH
# ─────────────────────────────────────────────
def dispatch(flow: Flow, results: Optional[List] = None):
    """Run inference on a flow and act on the result.

    Appends (flow, label, prob) to `results` when provided (replay reporting),
    and POSTs an alert when the flow is malicious and SEND_ALERTS is enabled.
    """
    label, attack_prob = run_inference(flow)
    if label is None:
        return
    if results is not None:
        results.append((flow, label, attack_prob))
    if label not in NORMAL_LABELS and SEND_ALERTS:
        send_alert(flow, label, attack_prob)


# ─────────────────────────────────────────────
#  PACKET PROCESSOR
# ─────────────────────────────────────────────
def process_packet(pkt):
    if IP not in pkt:
        return

    src   = pkt[IP].src
    dst   = pkt[IP].dst
    proto = "tcp" if TCP in pkt else "udp" if UDP in pkt else "icmp"

    dst_port = 0
    if TCP in pkt:
        dst_port = pkt[TCP].dport
    elif UDP in pkt:
        dst_port = pkt[UDP].dport

    flow = get_or_create_flow(src, dst, proto, dst_port)

    # ── Attribute the packet to a direction ───
    from_initiator = (src == flow.initiator)
    if from_initiator:
        flow.src_bytes += len(pkt)
    else:
        flow.dst_bytes += len(pkt)
    flow.count += 1
    flow.observe_time(float(getattr(pkt, "time", time.time())))

    if TCP in pkt:
        flags = int(pkt[TCP].flags)
        flow.update_tcp_flags(flags, from_initiator=from_initiator)

    # ── Live mode: infer inline once a flow matures ─────
    if INLINE_INFERENCE and flow.count >= MIN_PACKETS and not flow.inferenced:
        flow.inferenced = True   # mark before inference to avoid race on next packet
        dispatch(flow)


# ─────────────────────────────────────────────
#  PCAP REPLAY  (offline analysis)
# ─────────────────────────────────────────────
def replay_pcap(path: str):
    """Replay a pcap file through the detector and print a per-flow report.

    Unlike live mode, this evaluates each COMPLETE flow once (after all its
    packets are seen), giving the most accurate per-flow features.
    """
    from scapy.all import PcapReader

    flow_table.clear()
    n_pkts = 0
    print(f"[REPLAY] Reading {path} ...")
    try:
        with PcapReader(path) as reader:
            for pkt in reader:
                process_packet(pkt)
                n_pkts += 1
    except FileNotFoundError:
        raise SystemExit(f"[FATAL] pcap not found: {path}")

    results: List = []
    for flow in flow_table.values():
        if flow.count >= MIN_PACKETS:
            flow.inferenced = True
            dispatch(flow, results)

    _print_replay_report(path, n_pkts, results)
    return results


def _print_replay_report(path: str, n_pkts: int, results: List):
    attacks = [r for r in results if r[1] not in NORMAL_LABELS]
    print("\n" + "=" * 64)
    print(f"  REPLAY REPORT — {path}")
    print("=" * 64)
    print(f"  packets={n_pkts}  flows>= {MIN_PACKETS}pkts evaluated={len(results)}  "
          f"flagged={len(attacks)}")
    if attacks:
        print("\n  Flagged flows:")
        print(f"    {'source':<16}{'dest':<16}{'svc':<12}{'flag':<6}{'p':>6}")
        for flow, label, prob in sorted(attacks, key=lambda r: -(r[2] or 0)):
            p = f"{prob:.2f}" if prob is not None else " n/a"
            print(f"    {flow.src:<16}{flow.dst:<16}{flow.service:<12}"
                  f"{flow.kdd_flag:<6}{p:>6}")
    else:
        print("  No flows flagged as attacks.")
    print("=" * 64)


# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────
def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="NIDS ML sensor — live capture or offline pcap replay.",
    )
    src = parser.add_mutually_exclusive_group()
    src.add_argument("--pcap", metavar="FILE",
                     help="replay a pcap file offline instead of live capture")
    src.add_argument("--iface", default=NETWORK_INTERFACE,
                     help=f"live capture interface (default: {NETWORK_INTERFACE})")
    parser.add_argument("--min-packets", type=int, default=FLOW_MIN_PACKETS,
                        help=f"packets before a flow is evaluated "
                             f"(default: {FLOW_MIN_PACKETS})")
    parser.add_argument("--send-alerts", action="store_true",
                        help="POST alerts to main.py even in replay mode")
    args = parser.parse_args()

    global MIN_PACKETS, INLINE_INFERENCE, SEND_ALERTS
    MIN_PACKETS = args.min_packets

    load_model()

    if args.pcap:
        # Replay mode: evaluate complete flows; report only unless --send-alerts.
        INLINE_INFERENCE = False
        SEND_ALERTS = args.send_alerts
        replay_pcap(args.pcap)
        return

    # Live capture mode.
    if not SCAPY_AVAILABLE:
        raise SystemExit(
            "[FATAL] scapy not installed — live capture unavailable.\n"
            "        Install it (and Npcap on Windows / libpcap on Linux):\n"
            "          pip install scapy"
        )
    INLINE_INFERENCE = True
    SEND_ALERTS = True

    cleanup_thread = threading.Thread(target=cleanup_expired_flows, daemon=True)
    cleanup_thread.start()
    print(f"[+] Flow cleanup thread started (interval={FLOW_CLEANUP_INTERVAL}s)")

    print(f"[+] Monitoring interface: {args.iface}")
    print(f"[+] Inference triggers at {MIN_PACKETS} packets per flow")
    print(f"[+] Alerts -> {NIDS_ALERT_URL}")
    print("-" * 60)

    sniff(iface=args.iface, prn=process_packet, store=False)


if __name__ == "__main__":
    main()

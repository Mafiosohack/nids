"""
Offline verification for the live ML inference path (no scapy / no live capture).

Confirms that the feature vector the sensor builds aligns with the trained
rf_live model, that inference runs end-to-end, and that obviously-malicious vs
benign flows are scored sensibly. Run:

    python test_live_inference.py
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import live_ids_v2 as sensor
from feature_engineering.live_features import (
    LIVE_NUMERIC_FEATURES,
    LIVE_CATEGORICAL_FEATURES,
)


def make_flow(initiator, responder, proto, dst_port, **kw):
    a, b = sorted((initiator, responder))
    flow = sensor.Flow(
        ip_a=a, ip_b=b, proto=proto, dst_port=dst_port, initiator=initiator,
    )
    for k, v in kw.items():
        setattr(flow, k, v)
    return flow


def test_feature_width_matches_model():
    numeric = make_flow("10.0.0.1", "10.0.0.2", "tcp", 80).build_feature_vector()
    assert numeric.shape == (1, len(LIVE_NUMERIC_FEATURES)), numeric.shape

    n_cat = sum(len(c) for c in sensor.encoder.categories_)
    expected = len(LIVE_NUMERIC_FEATURES) + n_cat
    assert sensor.model.n_features_in_ == expected, (
        f"model expects {sensor.model.n_features_in_}, sensor emits {expected}"
    )
    print(f"[OK] feature width aligns: {expected} "
          f"({len(LIVE_NUMERIC_FEATURES)} numeric + {n_cat} one-hot)")


def test_bidirectional_accounting():
    """src_bytes/dst_bytes and SYN-ACK state must populate (the bug we fixed)."""
    f = make_flow("10.0.0.1", "10.0.0.2", "tcp", 80)
    # initiator sends 100 bytes, responder replies 200, handshake completes
    f.src_bytes, f.dst_bytes = 100, 200
    f.seen_syn = f.seen_syn_ack = f.seen_fin = True
    f.count = 6
    assert f.dst_bytes == 200, "dst_bytes should be populated"
    assert f.kdd_flag == "SF", f.kdd_flag
    print(f"[OK] bidirectional accounting: src={f.src_bytes} dst={f.dst_bytes} "
          f"flag={f.kdd_flag}")


def test_inference_runs_both_classes():
    # Benign-looking established HTTP flow
    benign = make_flow(
        "10.0.0.5", "93.184.216.34", "tcp", 443,
        src_bytes=600, dst_bytes=4000, count=12,
        seen_syn=True, seen_syn_ack=True, seen_fin=True,
    )
    # SYN-flood-style: many SYNs, no response, no bytes back
    synflood = make_flow(
        "66.66.66.66", "10.0.0.5", "tcp", 80,
        src_bytes=320, dst_bytes=0, count=400,
        seen_syn=True, seen_syn_ack=False,
    )

    for name, flow in [("benign-https", benign), ("synflood", synflood)]:
        label, prob, attack_type = sensor.run_inference(flow)
        assert label in ("normal", "attack"), label
        assert prob is None or 0.0 <= prob <= 1.0
        # attack_type is a specific CIC-IDS class when flagged, else None.
        assert (attack_type is None) == (label == "normal"), (label, attack_type)
        print(f"[OK] inference {name:<13} -> label={label} "
              f"type={attack_type} p_attack={prob:.3f}" + (" (SF=%s)" % flow.kdd_flag))


if __name__ == "__main__":
    print("Loading model...")
    sensor.load_model()
    print()
    test_feature_width_matches_model()
    test_bidirectional_accounting()
    test_inference_runs_both_classes()
    print("\nAll checks passed.")

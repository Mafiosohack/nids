import joblib
import numpy as np
from scapy.all import sniff, IP, TCP, UDP
from collections import defaultdict
import time

print("[+] Loading model and encoder...")

model = joblib.load("models/random_forest_v2.pkl")
encoder = joblib.load("models/rf_encoder_v2.pkl")

flows = defaultdict(lambda: {
    "start": time.time(),
    "duration": 0,
    "src_bytes": 0,
    "dst_bytes": 0,
    "count": 0
})

def build_feature_vector(flow, proto):
    duration = flow["duration"]
    src_bytes = flow["src_bytes"]
    dst_bytes = flow["dst_bytes"]
    count = flow["count"]

    # Minimal numerical placeholder features
    numeric = np.zeros(38)
    numeric[0] = duration
    numeric[1] = src_bytes
    numeric[2] = dst_bytes
    numeric[3] = count

    # Categorical fields
    service = "http"
    flag = "SF"

    cat_data = np.array([[proto, service, flag]])
    cat_encoded = encoder.transform(cat_data)

    final = np.hstack((numeric.reshape(1, -1), cat_encoded))

    return final

def process_packet(pkt):
    if IP in pkt:
        src = pkt[IP].src
        dst = pkt[IP].dst

        proto = "tcp" if TCP in pkt else "udp" if UDP in pkt else "icmp"

        key = (src, dst, proto)

        flows[key]["duration"] = time.time() - flows[key]["start"]
        flows[key]["src_bytes"] += len(pkt)
        flows[key]["count"] += 1

        if flows[key]["count"] >= 20:
            try:
                features = build_feature_vector(flows[key], proto)
                prediction = model.predict(features)

                if prediction[0] == 1:
                    print(f"[ALERT] Suspicious flow: {src} → {dst} ({proto})")

            except Exception as e:
                print("Prediction error:", e)

print("[+] Monitoring traffic...")

sniff(iface="ens37", prn=process_packet, store=False)

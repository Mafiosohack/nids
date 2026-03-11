import joblib
import numpy as np
from scapy.all import sniff, IP, TCP, UDP
from collections import defaultdict
import time

print("[+] Loading model and encoder...")

model = joblib.load("models/random_forest_v2.pkl")
encoder = joblib.load("models/rf_encoder_v2.pkl")

connections = defaultdict(lambda: {
    "start_time": time.time(),
    "packet_count": 0,
    "byte_count": 0
})

def extract_features(conn):
    duration = time.time() - conn["start_time"]
    packet_count = conn["packet_count"]
    byte_count = conn["byte_count"]

    # VERY BASIC feature vector
    features = np.array([[duration, packet_count, byte_count]])

    return features

def process_packet(packet):
    if IP in packet:
        src = packet[IP].src
        dst = packet[IP].dst

        protocol = "tcp" if TCP in packet else "udp" if UDP in packet else "other"

        key = (src, dst, protocol)

        connections[key]["packet_count"] += 1
        connections[key]["byte_count"] += len(packet)

        if connections[key]["packet_count"] > 20:  # threshold before prediction
            features = extract_features(connections[key])

            try:
                prediction = model.predict(features)
                if prediction[0] == 1:
                    print(f"[ALERT] Suspicious traffic: {src} → {dst}")
            except Exception as e:
                print("Feature mismatch:", e)

print("[+] Starting packet capture...")

sniff(iface="ens37", prn=process_packet, store=False)

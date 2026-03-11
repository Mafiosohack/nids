"""
═══════════════════════════════════════════════════════════════
REAL-TIME NETWORK INTRUSION DETECTION SYSTEM (NIDS)
═══════════════════════════════════════════════════════════════

Integrated FastAPI + Live Packet Capture + ML Detection

Architecture:
  1. FastAPI web server with REST endpoints
  2. Background packet capture thread (Scapy)
  3. Port scan detection (behavior-based)
  4. ML-based detection (Random Forest)
  5. Thread-safe alert storage
  6. Real-time monitoring dashboard support

Usage:
  sudo uvicorn main:app --host 0.0.0.0 --port 8000

Author: Claude & Warren
Date: 2026-02-19
═══════════════════════════════════════════════════════════════
"""

import threading
import time
import joblib
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict, deque
from typing import Dict, List, Optional, Deque
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Scapy imports
try:
    from scapy.all import sniff, IP, TCP, UDP, ICMP
    SCAPY_AVAILABLE = True
except ImportError:
    SCAPY_AVAILABLE = False
    print("⚠ WARNING: Scapy not installed. Live capture disabled.")
    print("Install with: pip install scapy")

# ═══════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════

NETWORK_INTERFACE = "ens37"  # Change this to your interface (eth0, wlan0, etc.)
PORT_SCAN_THRESHOLD = 5     # Ports contacted within time window
PORT_SCAN_WINDOW = 5         # Seconds
MAX_ALERTS_STORED = 200      # Maximum alerts kept in memory
FLOW_TIMEOUT = 60            # Seconds before flow expires
STEALTH_SCAN_THRESHOLD = 15
STEALTH_SCAN_WINDOW = 60  # seconds
DDOS_PACKET_RATE_THRESHOLD = 200  # packets per window
DDOS_WINDOW = 3  # seconds
# ═════════════════════════════════
# GLOBAL STATE (Thread-Safe)
# ═══════════════════════════════════════════════════════════════

alerts_lock = threading.Lock()
alerts: Deque[Dict] = deque(maxlen=MAX_ALERTS_STORED)

scan_tracker_lock = threading.Lock()
scan_tracker: Dict[str, Dict] = defaultdict(lambda: {
    'ports': set(),
    'first_seen': None,
    'last_seen': None
})

flow_tracker_lock = threading.Lock()
flow_tracker: Dict[str, Dict] = {}

sniffer_running = False
sniffer_thread: Optional[threading.Thread] = None

ml_model = None
ml_encoder = None
model_loaded = False

system_stats = {
    'packets_captured': 0,
    'alerts_generated': 0,
    'scans_detected': 0,
    'ml_detections': 0,
    'start_time': datetime.now()
}
# ─── Stealth Scan Tracker ─────────────────────────────────────
stealth_tracker_lock = threading.Lock()
stealth_tracker = defaultdict(lambda: {
    "ports": set(),
    "timestamps": []
})

# ─── DDoS Tracker ─────────────────────────────────────────────
ddos_tracker_lock = threading.Lock()
ddos_tracker = defaultdict(lambda: {
    "timestamps": []
})
# ═══════════════════════════════════════════════════════════════
# PYDANTIC SCHEMAS
# ═══════════════════════════════════════════════════════════════

class Alert(BaseModel):
    id: int
    type: str
    severity: str
    src: str
    dst: Optional[str] = None
    protocol: str
    message: str
    timestamp: str
    details: Optional[Dict] = None

class SystemStatus(BaseModel):
    sniffer_running: bool
    model_loaded: bool
    interface: str
    packets_captured: int
    alerts_generated: int
    scans_detected: int
    ml_detections: int
    uptime_seconds: float

# ═══════════════════════════════════════════════════════════════
# ML MODEL LOADING
# ═══════════════════════════════════════════════════════════════

def load_ml_models():
    """Load Random Forest model and OneHotEncoder at startup."""
    global ml_model, ml_encoder, model_loaded
    
    model_path = Path("models/random_forest_v2.pkl")
    encoder_path = Path("models/rf_encoder_v2.pkl")
    
    if not model_path.exists():
        print(f"⚠ Model not found: {model_path}")
        print("  ML detection disabled. Train model first.")
        return False
    
    if not encoder_path.exists():
        print(f"⚠ Encoder not found: {encoder_path}")
        print("  ML detection disabled. Save encoder first.")
        return False
    
    try:
        ml_model = joblib.load(model_path)
        ml_encoder = joblib.load(encoder_path)
        model_loaded = True
        print(f"✓ ML Model loaded: {model_path}")
        print(f"✓ Encoder loaded: {encoder_path}")
        return True
    except Exception as e:
        print(f"✗ Error loading ML models: {e}")
        return False

# ═══════════════════════════════════════════════════════════════
# ALERT GENERATION
# ═══════════════════════════════════════════════════════════════

def generate_alert(
    alert_type: str,
    severity: str,
    src: str,
    dst: Optional[str],
    protocol: str,
    message: str,
    details: Optional[Dict] = None
):
    """Thread-safe alert generation."""
    with alerts_lock:
        alert = {
            'id': system_stats['alerts_generated'] + 1,
            'type': alert_type,
            'severity': severity,
            'src': src,
            'dst': dst,
            'protocol': protocol,
            'message': message,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'details': details or {}
        }
        alerts.append(alert)
        system_stats['alerts_generated'] += 1
        
        # Log to console
        sev_icon = {
            'critical': '🔴',
            'high': '🟠',
            'medium': '🟡',
            'low': '🟢'
        }.get(severity, '⚪')
        
        print(f"{sev_icon} [{severity.upper()}] {alert_type}: {message}")

# ═══════════════════════════════════════════════════════════════
# PORT SCAN DETECTION (Behavior-Based)
# ═══════════════════════════════════════════════════════════════

def detect_port_scan(src_ip: str, dst_ip: str, dst_port: int):
    """
    Detect port scans by tracking SYN packets.
    Alert if a source IP contacts >10 different ports within 5 seconds.
    """
    with scan_tracker_lock:
        tracker = scan_tracker[src_ip]
        now = datetime.now()
        
        # Initialize first_seen
        if tracker['first_seen'] is None:
            tracker['first_seen'] = now
        
        tracker['last_seen'] = now
        tracker['ports'].add(dst_port)
        
        # Check if scan window expired
        time_elapsed = (now - tracker['first_seen']).total_seconds()
        
        if time_elapsed > PORT_SCAN_WINDOW:
            # Reset tracker
            tracker['ports'] = {dst_port}
            tracker['first_seen'] = now
            return
        
        # Check threshold
        num_ports = len(tracker['ports'])
        if num_ports > PORT_SCAN_THRESHOLD:
            # ALERT: Port scan detected!
            generate_alert(
                alert_type='Port Scan',
                severity='high',
                src=src_ip,
                dst=dst_ip,
                protocol='TCP',
                message=f'Port scan detected: {num_ports} ports in {time_elapsed:.1f}s',
                details={
                    'ports_scanned': num_ports,
                    'time_window': f'{time_elapsed:.1f}s',
                    'ports': sorted(list(tracker['ports']))[:20]  # First 20 ports
                }
            )
            system_stats['scans_detected'] += 1
            
            # Reset tracker after alert
            tracker['ports'].clear()
            tracker['first_seen'] = now

def detect_stealth_scan(src_ip: str, dst_port: int):
    """Detect slow stealth scans over longer time window."""
    with stealth_tracker_lock:
        tracker = stealth_tracker[src_ip]
        now = time.time()

        tracker["ports"].add(dst_port)
        tracker["timestamps"].append(now)

        # Remove timestamps outside window
        tracker["timestamps"] = [
            t for t in tracker["timestamps"]
            if now - t < STEALTH_SCAN_WINDOW
        ]

        if len(tracker["ports"]) >= STEALTH_SCAN_THRESHOLD:
            generate_alert(
                alert_type="Stealth Port Scan",
                severity="medium",
                src=src_ip,
                dst=None,
                protocol="TCP",
                message=f"Stealth scan detected: {len(tracker['ports'])} ports over {STEALTH_SCAN_WINDOW}s",
                details={
                    "ports_scanned": len(tracker["ports"]),
                    "window": STEALTH_SCAN_WINDOW
                }
            )

            tracker["ports"].clear()
            tracker["timestamps"].clear()

def detect_ddos(src_ip: str):
    """Detect high packet rate (possible DDoS or flood)."""
    with ddos_tracker_lock:
        tracker = ddos_tracker[src_ip]
        now = time.time()

        tracker["timestamps"].append(now)

        tracker["timestamps"] = [
            t for t in tracker["timestamps"]
            if now - t < DDOS_WINDOW
        ]

        if len(tracker["timestamps"]) >= DDOS_PACKET_RATE_THRESHOLD:
            generate_alert(
                alert_type="DDoS / Flood Detected",
                severity="critical",
                src=src_ip,
                dst=None,
                protocol="Multiple",
                message=f"High packet rate detected: {len(tracker['timestamps'])} packets in {DDOS_WINDOW}s",
                details={
                    "packet_rate": len(tracker["timestamps"]),
                    "window": DDOS_WINDOW
                }
            )

            tracker["timestamps"].clear()

# ═══════════════════════════════════════════════════════════════
# FEATURE EXTRACTION FOR ML
# ═══════════════════════════════════════════════════════════════

def extract_features_from_packet(packet) -> Optional[Dict]:
    """
    Extract features from a packet for ML inference.
    Returns feature dict or None if packet is incomplete.
    """
    if not packet.haslayer(IP):
        return None
    
    ip_layer = packet[IP]
    
    features = {
        'protocol_type': 'tcp' if packet.haslayer(TCP) else
                        'udp' if packet.haslayer(UDP) else
                        'icmp' if packet.haslayer(ICMP) else 'other',
        'src_bytes': len(packet),
        'dst_bytes': 0,  # Would need flow tracking
        'flag': 'S' if packet.haslayer(TCP) and packet[TCP].flags == 'S' else 'SF',
        'duration': 0,  # Would need flow tracking
    }
    
    return features

def build_feature_vector(features: Dict) -> Optional[np.ndarray]:
    """
    Build feature vector compatible with Random Forest model.
    Uses encoder to transform categorical features.
    """
    if not model_loaded:
        return None
    
    try:
        # Example feature vector structure
        # Adjust based on your actual model's expected features
        feature_list = [
            features.get('protocol_type', 'tcp'),
            features.get('flag', 'SF'),
            features.get('src_bytes', 0),
            features.get('dst_bytes', 0),
            features.get('duration', 0)
        ]
        
        # Transform using encoder (adjust indices based on your encoder)
        # This is a simplified example - adjust to your actual feature set
        categorical_features = [feature_list[0], feature_list[1]]
        encoded = ml_encoder.transform([categorical_features])
        
        # Combine with numerical features
        numerical = [feature_list[2], feature_list[3], feature_list[4]]
        feature_vector = np.concatenate([encoded[0], numerical]).reshape(1, -1)
        
        return feature_vector
    
    except Exception as e:
        print(f"⚠ Feature extraction error: {e}")
        return None

# ═══════════════════════════════════════════════════════════════
# ML-BASED DETECTION
# ═══════════════════════════════════════════════════════════════

def ml_detect(src_ip: str, dst_ip: str, protocol: str, features: Dict):
    """
    Run ML inference on extracted features.
    Generate alert if malicious prediction.
    """
    if not model_loaded:
        return
    
    feature_vector = build_feature_vector(features)
    if feature_vector is None:
        return
    
    try:
        prediction = ml_model.predict(feature_vector)[0]
        
        # Assuming binary: 0 = normal, 1 = malicious
        if prediction == 1:
            # Get probability if available
            try:
                proba = ml_model.predict_proba(feature_vector)[0]
                confidence = max(proba) * 100
            except:
                confidence = 0.0
            
            generate_alert(
                alert_type='ML Detection',
                severity='medium',
                src=src_ip,
                dst=dst_ip,
                protocol=protocol.upper(),
                message=f'Malicious traffic detected by ML model',
                details={
                    'prediction': 'malicious',
                    'confidence': f'{confidence:.1f}%',
                    'features': features
                }
            )
            system_stats['ml_detections'] += 1
    
    except Exception as e:
        print(f"⚠ ML inference error: {e}")

# ═══════════════════════════════════════════════════════════════
# PACKET PROCESSING CALLBACK
# ═══════════════════════════════════════════════════════════════

def process_packet(packet):
    """
    Main packet processing callback.
    Called by Scapy for every captured packet.
    """
    system_stats['packets_captured'] += 1
    
    # Only process IP packets
    if not packet.haslayer(IP):
        return
    
    ip_layer = packet[IP]
    src_ip = ip_layer.src
    dst_ip = ip_layer.dst
    # DDoS detection  
    detect_ddos(src_ip)
    
    # ─── TCP Processing ───────────────────────────────────────
    if packet.haslayer(TCP):
        tcp_layer = packet[TCP]
        dst_port = tcp_layer.dport
        
        # Port scan detection (SYN packets only)
        if tcp_layer.flags & 0x02:  # SYN flag
            detect_port_scan(src_ip, dst_ip, dst_port) 
            detect_stealth_scan(src_ip, dst_port)

        
        # ML detection
        features = extract_features_from_packet(packet)
        if features:
            ml_detect(src_ip, dst_ip, 'tcp', features)
    
    # ─── UDP Processing ───────────────────────────────────────
    elif packet.haslayer(UDP):
        features = extract_features_from_packet(packet)
        if features:
            ml_detect(src_ip, dst_ip, 'udp', features)
    
    # ─── ICMP Processing ──────────────────────────────────────
    elif packet.haslayer(ICMP):
        features = extract_features_from_packet(packet)
        if features:
            ml_detect(src_ip, dst_ip, 'icmp', features)

# ═══════════════════════════════════════════════════════════════
# PACKET SNIFFER (Background Thread)
# ═══════════════════════════════════════════════════════════════

def start_packet_capture():
    """
    Start packet capture in background thread.
    Runs indefinitely until stopped.
    """
    global sniffer_running
    
    if not SCAPY_AVAILABLE:
        print("✗ Scapy not available. Cannot start packet capture.")
        return
    
    print(f"🔍 Starting packet capture on interface: {NETWORK_INTERFACE}")
    print(f"   Port scan threshold: {PORT_SCAN_THRESHOLD} ports in {PORT_SCAN_WINDOW}s")
    print(f"   ML detection: {'enabled' if model_loaded else 'disabled'}")
    print()
    
    sniffer_running = True
    
    try:
        # Start sniffing (blocking call)
        sniff(
            iface=NETWORK_INTERFACE,
            prn=process_packet,
            store=False,           # Don't store packets in memory
            stop_filter=lambda x: not sniffer_running  # Stop condition
        )
    except PermissionError:
        print("✗ Permission denied. Run with sudo:")
        print("  sudo uvicorn main:app --host 0.0.0.0 --port 8000")
        sniffer_running = False
    except Exception as e:
        print(f"✗ Packet capture error: {e}")
        sniffer_running = False

def stop_packet_capture():
    """Stop packet capture gracefully."""
    global sniffer_running
    sniffer_running = False
    print("⏹ Packet capture stopped")

# ═══════════════════════════════════════════════════════════════
# FASTAPI APPLICATION
# ═══════════════════════════════════════════════════════════════

app = FastAPI(
    title="Real-Time NIDS API",
    description="Network Intrusion Detection System with Live Packet Capture",
    version="2.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ═══════════════════════════════════════════════════════════════
# STARTUP & SHUTDOWN EVENTS
# ═══════════════════════════════════════════════════════════════

@app.on_event("startup")
async def startup_event():
    """
    Startup tasks:
    1. Load ML models
    2. Start packet capture in background thread
    """
    global sniffer_thread
    
    print("="*70)
    print("🔐 REAL-TIME NETWORK INTRUSION DETECTION SYSTEM")
    print("="*70)
    
    # Load ML models
    load_ml_models()
    
    # Start packet sniffer in background thread
    if SCAPY_AVAILABLE:
        sniffer_thread = threading.Thread(
            target=start_packet_capture,
            daemon=True,
            name="PacketSniffer"
        )
        sniffer_thread.start()
        print("✓ Packet capture thread started")
    else:
        print("⚠ Packet capture disabled (Scapy not available)")
    
    print("="*70)
    print()

@app.on_event("shutdown")
async def shutdown_event():
    """Graceful shutdown."""
    print("\n🛑 Shutting down NIDS...")
    stop_packet_capture()
    if sniffer_thread:
        sniffer_thread.join(timeout=5)
    print("✓ Shutdown complete")

# ═══════════════════════════════════════════════════════════════
# API ENDPOINTS
# ═══════════════════════════════════════════════════════════════

@app.get("/")
async def root():
    """Welcome endpoint."""
    return {
        "message": "Real-Time NIDS API",
        "version": "2.0.0",
        "status": "online",
        "docs": "/docs"
    }

@app.get("/status", response_model=SystemStatus)
async def get_status():
    """Get system status and statistics."""
    uptime = (datetime.now() - system_stats['start_time']).total_seconds()
    
    return SystemStatus(
        sniffer_running=sniffer_running,
        model_loaded=model_loaded,
        interface=NETWORK_INTERFACE,
        packets_captured=system_stats['packets_captured'],
        alerts_generated=system_stats['alerts_generated'],
        scans_detected=system_stats['scans_detected'],
        ml_detections=system_stats['ml_detections'],
        uptime_seconds=round(uptime, 1)
    )

@app.get("/alerts")
async def get_alerts(limit: int = 50):
    """
    Get recent alerts.
    
    Args:
        limit: Maximum number of alerts to return (default: 50)
    
    Returns:
        List of alerts in reverse chronological order
    """
    with alerts_lock:
        alerts_list = list(alerts)
    
    return {
        "total_alerts": len(alerts_list),
        "alerts": alerts_list[:limit]
    }

@app.delete("/alerts")
async def clear_alerts():
    """Clear all alerts from memory."""
    with alerts_lock:
        alerts.clear()
    
    return {
        "message": "Alerts cleared",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "sniffer_running": sniffer_running,
        "model_loaded": model_loaded
    }

@app.post("/control/start")
async def start_capture():
    """Manually start packet capture (if stopped)."""
    global sniffer_thread, sniffer_running
    
    if sniffer_running:
        return {"message": "Sniffer already running"}
    
    if not SCAPY_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Scapy not installed. Cannot start capture."
        )
    
    sniffer_thread = threading.Thread(
        target=start_packet_capture,
        daemon=True,
        name="PacketSniffer"
    )
    sniffer_thread.start()
    
    return {
        "message": "Packet capture started",
        "interface": NETWORK_INTERFACE
    }

@app.post("/control/stop")
async def stop_capture():
    """Manually stop packet capture."""
    if not sniffer_running:
        return {"message": "Sniffer not running"}
    
    stop_packet_capture()
    
    return {
        "message": "Packet capture stopped"
    }

# ═══════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    
    print("⚠ WARNING: Run with sudo for packet capture:")
    print("  sudo uvicorn main:app --host 0.0.0.0 --port 8000")
    print()
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False  # Disable reload for production
    )

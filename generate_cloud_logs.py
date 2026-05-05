"""
generate_cloud_logs.py — VPC Flow Log Attack Simulator

Generates realistic AWS VPC Flow Log files simulating seven
attack scenarios. Drop the output files into cloud_logs/ and
cloud_log_sensor.py will detect them automatically.

Usage:
    python3 generate_cloud_logs.py              # generates all scenarios
    python3 generate_cloud_logs.py portscan     # single scenario
    python3 generate_cloud_logs.py exfil
    python3 generate_cloud_logs.py recon
    python3 generate_cloud_logs.py brute
    python3 generate_cloud_logs.py lateral
    python3 generate_cloud_logs.py udpamp
    python3 generate_cloud_logs.py travel

Output: cloud_logs/<scenario>.log
"""

import random
import sys
import time
from pathlib import Path

# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────
OUTPUT_DIR = Path("cloud_logs")
ACCOUNT_ID = "123456789012"
IFACE_ID   = "eni-0a1b2c3d4e5f"

# Internal network range (matches INTERNAL_PREFIX in sensor)
INTERNAL_NETS = [
    "192.168.96.",
    "192.168.1.",
    "10.0.1.",
]

# External IPs (attacker/exfil destination)
EXTERNAL_IPS = [
    "45.33.32.156",    # nmap host
    "185.220.101.45",  # Tor exit node
    "198.51.100.10",   # attacker
    "203.0.113.55",    # exfil server
    "91.108.4.172",    # external
    "77.88.55.66",     # scanner
    "104.21.45.8",     # external
]

def internal(prefix_idx: int = 0, last: int = None) -> str:
    prefix = INTERNAL_NETS[prefix_idx % len(INTERNAL_NETS)]
    return prefix + str(last if last else random.randint(10, 250))

def external(idx: int = None) -> str:
    if idx is not None:
        return EXTERNAL_IPS[idx % len(EXTERNAL_IPS)]
    return random.choice(EXTERNAL_IPS)

def flow_line(src, dst, sport, dport, proto, packets, nbytes, start, duration, action):
    """
    Build a single VPC Flow Log v2 line.
    proto: 6=TCP, 17=UDP, 1=ICMP
    action: ACCEPT or REJECT
    """
    end = int(start + duration)
    return (
        f"2 {ACCOUNT_ID} {IFACE_ID} "
        f"{src} {dst} {sport} {dport} {proto} "
        f"{packets} {nbytes} {int(start)} {end} "
        f"{action} OK\n"
    )

def write_log(filename: str, header: str, lines: list):
    OUTPUT_DIR.mkdir(exist_ok=True)
    path = OUTPUT_DIR / filename
    with open(path, "w") as f:
        # VPC Flow Log header
        f.write("version account-id interface-id srcaddr dstaddr "
                "srcport dstport protocol packets bytes "
                "start end action log-status\n")
        f.write(f"# Scenario: {header}\n")
        f.writelines(lines)
    print(f"  ✓ Written: {path}  ({len(lines)} flow records)")

# ─────────────────────────────────────────────
#  SCENARIO GENERATORS
# ─────────────────────────────────────────────

def gen_port_scan():
    """
    External attacker scanning many ports on an internal host.
    Produces 15 distinct destination ports → triggers Cloud Port Scan.
    """
    print("[+] Generating: Cloud Port Scan")
    src   = external(0)
    dst   = internal(0, 130)
    now   = time.time() - 30  # set in past so window check works

    ports = [22, 23, 25, 80, 110, 143, 443, 445, 1433, 3306, 3389,
             5432, 8080, 8443, 9200, 27017, 6379, 11211, 21, 53]

    lines = []
    for i, port in enumerate(ports):
        ts = now + i * 2  # 2 seconds apart
        lines.append(flow_line(
            src=src, dst=dst,
            sport=random.randint(40000, 60000), dport=port,
            proto=6, packets=1, nbytes=60,
            start=ts, duration=0, action="REJECT"
        ))

    # A few ACCEPTs on open ports
    for port in [22, 80, 443]:
        ts = now + random.randint(0, 30)
        lines.append(flow_line(
            src=src, dst=dst,
            sport=random.randint(40000, 60000), dport=port,
            proto=6, packets=5, nbytes=500,
            start=ts, duration=2, action="ACCEPT"
        ))

    write_log("port_scan.log", "Cloud Port Scan — External attacker scanning internal host", lines)


def gen_exfiltration():
    """
    Internal compromised host sending large amounts of data to external IP.
    Produces 60MB transfer → triggers Data Exfiltration.
    """
    print("[+] Generating: Data Exfiltration")
    src = internal(0, 150)   # compromised internal host
    dst = external(3)         # exfil server
    now = time.time() - 60

    lines = []
    # 30 large flows totalling ~60MB
    for i in range(30):
        ts     = now + i * 10
        nbytes = random.randint(1_500_000, 2_500_000)  # 1.5–2.5 MB per flow
        lines.append(flow_line(
            src=src, dst=dst,
            sport=random.randint(40000, 60000), dport=443,
            proto=6,
            packets=random.randint(1000, 2000),
            nbytes=nbytes,
            start=ts, duration=8, action="ACCEPT"
        ))

    # Normal background traffic to make it realistic
    for _ in range(10):
        ts = now + random.randint(0, 250)
        lines.append(flow_line(
            src=internal(0, 200), dst=external(0),
            sport=random.randint(40000, 60000), dport=80,
            proto=6, packets=10, nbytes=1500,
            start=ts, duration=1, action="ACCEPT"
        ))

    write_log("exfiltration.log", "Data Exfiltration — Large outbound transfer to external IP", lines)


def gen_recon():
    """
    External scanner hitting many distinct internal destinations.
    Produces 25 distinct destination IPs → triggers Cloud Recon.
    """
    print("[+] Generating: Cloud Reconnaissance")
    src = external(1)
    now = time.time() - 40

    lines = []
    for i in range(25):
        dst    = internal(0, 100 + i)
        ts     = now + i * 2
        lines.append(flow_line(
            src=src, dst=dst,
            sport=random.randint(40000, 60000), dport=443,
            proto=6, packets=2, nbytes=120,
            start=ts, duration=0, action="REJECT"
        ))

    write_log("recon.log", "Cloud Recon — External scanner probing many internal hosts", lines)


def gen_brute_force():
    """
    External IP making repeated failed connections to SSH port.
    Produces 15 REJECT flows to port 22 → triggers Cloud Brute Force.
    """
    print("[+] Generating: Cloud Brute Force (SSH)")
    src = external(2)
    dst = internal(0, 129)
    now = time.time() - 50

    lines = []
    for i in range(15):
        ts = now + i * 3
        lines.append(flow_line(
            src=src, dst=dst,
            sport=random.randint(40000, 60000), dport=22,
            proto=6, packets=4, nbytes=240,
            start=ts, duration=1, action="REJECT"
        ))

    # One successful login to make it realistic
    lines.append(flow_line(
        src=src, dst=dst,
        sport=random.randint(40000, 60000), dport=22,
        proto=6, packets=20, nbytes=3000,
        start=now + 50, duration=10, action="ACCEPT"
    ))

    write_log("brute_force.log", "Cloud Brute Force — Repeated REJECT flows to SSH port", lines)


def gen_lateral_movement():
    """
    Compromised internal host contacting many other internal hosts.
    Produces 10 distinct internal destinations → triggers Lateral Movement.
    """
    print("[+] Generating: Lateral Movement")
    src = internal(0, 150)  # compromised host
    now = time.time() - 80

    lines = []
    for i in range(12):
        dst = internal(0, 100 + i * 5)
        if dst == src:
            dst = internal(0, 200)
        ts  = now + i * 10
        lines.append(flow_line(
            src=src, dst=dst,
            sport=random.randint(40000, 60000),
            dport=random.choice([22, 445, 3389, 80, 8080]),
            proto=6, packets=15, nbytes=2000,
            start=ts, duration=3, action="ACCEPT"
        ))

    write_log("lateral.log", "Lateral Movement — Compromised internal host scanning internal network", lines)


def gen_udp_amplification():
    """
    External source sending many UDP flows to amplification ports.
    Produces 35 flows to port 53 → triggers Cloud UDP Amplification.
    """
    print("[+] Generating: UDP Amplification")
    src = external(5)
    dst = internal(0, 1)   # DNS server
    now = time.time() - 20

    lines = []
    for i in range(35):
        ts = now + i * 0.5
        lines.append(flow_line(
            src=src, dst=dst,
            sport=random.randint(1024, 65535), dport=53,
            proto=17,  # UDP
            packets=1, nbytes=512,
            start=ts, duration=0, action="ACCEPT"
        ))

    write_log("udp_amp.log", "UDP Amplification — High-volume UDP to DNS port from external IP", lines)


def gen_impossible_travel():
    """
    Same source IP appearing in two different simulated regions rapidly.
    Uses IPs whose third octet maps to different regions in the sensor.
    Triggers Impossible Travel alert.

    Region mapping (from cloud_log_sensor.py region_hint()):
        octet < 85   → us-east-1
        octet < 170  → eu-west-1
        octet >= 170 → ap-southeast-1
    """
    print("[+] Generating: Impossible Travel")

    # Two IPs that will be seen as from different regions
    # We use the SAME source IP but with different destination subnets
    # that map to different regions — simulating the same user appearing
    # in two cloud regions within a short window
    src_us  = "45.33.32.10"    # third octet 32 → us-east-1
    src_eu  = "45.33.200.10"   # third octet 200 → ap-southeast-1
    dst     = internal(0, 130)
    now     = time.time()

    lines = []

    # Login from US region
    lines.append(flow_line(
        src=src_us, dst=dst,
        sport=54321, dport=443,
        proto=6, packets=30, nbytes=5000,
        start=now - 400, duration=10, action="ACCEPT"
    ))

    # Login from AP region 5 minutes later — geographically impossible
    lines.append(flow_line(
        src=src_eu, dst=dst,
        sport=54322, dport=443,
        proto=6, packets=30, nbytes=5000,
        start=now - 100, duration=10, action="ACCEPT"
    ))

    # Add some normal background flows
    for _ in range(5):
        ts = now - random.randint(10, 390)
        lines.append(flow_line(
            src=internal(0, random.randint(100, 200)),
            dst=external(0),
            sport=random.randint(40000, 60000), dport=80,
            proto=6, packets=5, nbytes=800,
            start=ts, duration=1, action="ACCEPT"
        ))

    write_log("impossible_travel.log",
              "Impossible Travel — Same logical source in two regions within 5 minutes", lines)


def gen_all():
    print("[+] Generating all attack scenarios...")
    gen_port_scan()
    gen_exfiltration()
    gen_recon()
    gen_brute_force()
    gen_lateral_movement()
    gen_udp_amplification()
    gen_impossible_travel()


# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────
SCENARIOS = {
    "portscan": gen_port_scan,
    "exfil":    gen_exfiltration,
    "recon":    gen_recon,
    "brute":    gen_brute_force,
    "lateral":  gen_lateral_movement,
    "udpamp":   gen_udp_amplification,
    "travel":   gen_impossible_travel,
    "all":      gen_all,
}

if __name__ == "__main__":
    arg = sys.argv[1].lower() if len(sys.argv) > 1 else "all"

    if arg not in SCENARIOS:
        print(f"Unknown scenario '{arg}'. Choose from: {', '.join(SCENARIOS.keys())}")
        sys.exit(1)

    print("=" * 60)
    print("VPC FLOW LOG ATTACK SIMULATOR")
    print(f"Output directory: {OUTPUT_DIR.resolve()}")
    print("=" * 60)

    SCENARIOS[arg]()

    print()
    print("Done. Drop the generated files into cloud_logs/ and")
    print("cloud_log_sensor.py will detect them on the next poll.")

"""
make_sample_pcap.py — generate a small demo capture for offline replay.

Produces data/samples/demo_traffic.pcap containing a mix of benign and
malicious flows so the ML sensor's replay mode has something to chew on:

  - benign  : a complete HTTP session and a complete HTTPS session (SYN ->
              SYN-ACK -> data both ways -> FIN), and a normal DNS exchange
  - attack  : a SYN flood (many SYNs, no responses) and a port scan
              (SYNs fanned across ports, all rejected)

Run:
    python make_sample_pcap.py
    python live_ids_v2.py --pcap data/samples/demo_traffic.pcap
"""

from pathlib import Path

from scapy.all import Ether, IP, TCP, UDP, Raw, wrpcap

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "data" / "samples" / "demo_traffic.pcap"

ETH = lambda: Ether(src="00:11:22:33:44:55", dst="66:77:88:99:aa:bb")
packets = []
clock = [1000.0]  # monotonic fake capture time


def add(pkt, gap=0.001):
    clock[0] += gap
    pkt.time = clock[0]
    packets.append(pkt)


def tcp_session(client, server, dport, req_len, resp_len, sport=44000):
    """A well-formed TCP session: handshake, data both ways, graceful close."""
    add(ETH() / IP(src=client, dst=server) / TCP(sport=sport, dport=dport, flags="S"))
    add(ETH() / IP(src=server, dst=client) / TCP(sport=dport, dport=sport, flags="SA"))
    add(ETH() / IP(src=client, dst=server) / TCP(sport=sport, dport=dport, flags="A"))
    add(ETH() / IP(src=client, dst=server) / TCP(sport=sport, dport=dport, flags="PA")
        / Raw(b"X" * req_len))
    add(ETH() / IP(src=server, dst=client) / TCP(sport=dport, dport=sport, flags="PA")
        / Raw(b"Y" * resp_len), gap=0.05)
    add(ETH() / IP(src=client, dst=server) / TCP(sport=sport, dport=dport, flags="A"))
    add(ETH() / IP(src=client, dst=server) / TCP(sport=sport, dport=dport, flags="FA"))
    add(ETH() / IP(src=server, dst=client) / TCP(sport=dport, dport=sport, flags="FA"))
    add(ETH() / IP(src=client, dst=server) / TCP(sport=sport, dport=dport, flags="A"))


def dns_exchange(client, resolver, n=4):
    """Several DNS query/response pairs (benign UDP)."""
    for i in range(n):
        add(ETH() / IP(src=client, dst=resolver) / UDP(sport=50000 + i, dport=53)
            / Raw(b"q" * 30))
        add(ETH() / IP(src=resolver, dst=client) / UDP(sport=53, dport=50000 + i)
            / Raw(b"r" * 90), gap=0.02)


def syn_flood(attacker, victim, dport=80, n=60):
    """Many SYNs, no SYN-ACKs — classic half-open flood (flag S0)."""
    for i in range(n):
        add(ETH() / IP(src=attacker, dst=victim) / TCP(sport=30000 + i, dport=dport,
            flags="S"), gap=0.005)


def port_scan(scanner, target, ports):
    """SYN to many ports, each met with RST (rejected) — recon scan."""
    for i, p in enumerate(ports):
        add(ETH() / IP(src=scanner, dst=target) / TCP(sport=40000 + i, dport=p,
            flags="S"), gap=0.003)
        add(ETH() / IP(src=target, dst=scanner) / TCP(sport=p, dport=40000 + i,
            flags="RA"), gap=0.001)


def main():
    # Benign
    tcp_session("192.168.1.10", "93.184.216.34", 80, req_len=300, resp_len=4000)
    tcp_session("192.168.1.10", "140.82.121.4", 443, req_len=550, resp_len=8000,
                sport=44120)
    dns_exchange("192.168.1.10", "8.8.8.8", n=4)

    # Attack
    syn_flood("66.66.66.66", "192.168.1.10", dport=80, n=60)
    port_scan("10.0.0.99", "192.168.1.20",
              ports=[21, 22, 23, 25, 80, 110, 143, 443, 445, 3306, 3389, 8080])

    OUT.parent.mkdir(parents=True, exist_ok=True)
    wrpcap(str(OUT), packets)
    print(f"[+] Wrote {len(packets)} packets -> {OUT}")


if __name__ == "__main__":
    main()

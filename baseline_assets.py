"""Generate a starter `hosts` block for assets.json from observed traffic.

Tier 2's strongest rules diff against a known-good baseline, and hand-writing one
per host is the reason people skip it. This watches real traffic and writes the
block for you.

    # from a capture file
    python baseline_assets.py --pcap data/samples/quiet_hour.pcap

    # from live capture, 10 minutes
    python baseline_assets.py --iface eth0 --seconds 600

    # limit to the hosts you care about, and merge into an existing inventory
    python baseline_assets.py --pcap quiet.pcap \\
        --hosts 192.168.56.102,192.168.56.20 --merge assets.json -o assets.json

## Read this before trusting the output

**A baseline taken from a compromised host bakes the compromise in.** If the
attacker's C2 channel is active during the capture, this will faithfully record
it as normal and the sensor will never alert on it again. Capture from a host
you have reason to trust, during a quiet period, and read the output before
committing it — it is a starting point for review, not an answer.

The generated file is deliberately annotated with what was seen and how often,
so an entry that looks wrong is visible rather than buried in a list of ports.
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional, Set, Tuple

INTERNAL_PREFIXES = ("192.168.", "10.", "172.16.", "172.17.", "172.18.",
                     "172.19.", "172.20.", "172.21.", "172.22.", "172.23.",
                     "172.24.", "172.25.", "172.26.", "172.27.", "172.28.",
                     "172.29.", "172.30.", "172.31.")

# Ports seen outbound to MANY destinations get collapsed to [null, port] rather
# than listing every IP — otherwise a browser's traffic produces a useless list
# of hundreds of CDN addresses.
#
# THIS IS A REAL TRADEOFF, not a formatting choice. [null, 443] allows that port
# to ANY destination, so a reverse shell to tcp/443 from that host will no longer
# trip BEHAVIOR_UNEXPECTED_OUTBOUND. Keeping per-destination entries means a new
# 443 destination alerts immediately — at the cost of a long list and a false
# positive every time the host visits somewhere new.
#
# Whichever way you go, BEHAVIOR_INTERACTIVE_SHELL still catches the shell once
# someone types into it, because it does not care about the port at all.
COLLAPSE_AFTER_DESTINATIONS = 5


def is_internal(ip: str) -> bool:
    return ip.startswith(INTERNAL_PREFIXES)


class Observer:
    def __init__(self, hosts: Optional[Set[str]] = None,
                 collapse_after: int = COLLAPSE_AFTER_DESTINATIONS):
        self.only = hosts or set()
        self.collapse_after = collapse_after
        # host -> {(dst, port): count}
        self.outbound: Dict[str, Dict[Tuple[str, int], int]] = defaultdict(
            lambda: defaultdict(int))
        # host -> {port: count}   (proved listening by answering SYN-ACK)
        self.listening: Dict[str, Dict[int, int]] = defaultdict(
            lambda: defaultdict(int))
        self.packets = 0
        self.pending_syn: Set[Tuple[str, str, int]] = set()

    def _want(self, host: str) -> bool:
        return (not self.only) or host in self.only

    def handle(self, pkt) -> None:
        from scapy.all import IP, TCP
        self.packets += 1
        if not (pkt.haslayer(IP) and pkt.haslayer(TCP)):
            return
        ip, tcp = pkt[IP], pkt[TCP]
        flags = int(tcp.flags)
        syn, ack = bool(flags & 0x02), bool(flags & 0x10)

        if syn and not ack:
            # An outbound connection attempt: this is what the host normally reaches.
            if self._want(ip.src):
                self.outbound[ip.src][(ip.dst, int(tcp.dport))] += 1
            self.pending_syn.add((ip.src, ip.dst, int(tcp.dport)))
        elif syn and ack:
            # Answering SYN-ACK proves an accepting socket on src:sport.
            if (ip.dst, ip.src, int(tcp.sport)) in self.pending_syn and \
                    self._want(ip.src):
                self.listening[ip.src][int(tcp.sport)] += 1

    # ── output ────────────────────────────────────────────────────────────────
    def build(self, min_hits: int = 1) -> dict:
        hosts: Dict[str, dict] = {}
        for host in sorted(set(self.outbound) | set(self.listening)):
            entry: dict = {}

            ports = sorted(p for p, n in self.listening.get(host, {}).items()
                           if n >= min_hits)
            if ports:
                entry["listening_ports"] = ports

            seen = self.outbound.get(host, {})
            by_port: Dict[int, Set[str]] = defaultdict(set)
            for (dst, port), n in seen.items():
                if n >= min_hits:
                    by_port[port].add(dst)

            allowed, collapsed = [], []
            for port in sorted(by_port):
                dests = by_port[port]
                if self.collapse_after and len(dests) >= self.collapse_after:
                    allowed.append([None, port])
                    collapsed.append(port)
                else:
                    for dst in sorted(dests):
                        allowed.append([dst, port])
            if allowed or not ports:
                entry["allowed_outbound"] = allowed

            total_out = sum(seen.values())
            entry["_observed"] = (
                f"{total_out} outbound connection(s) to {len(seen)} "
                f"destination(s); {len(ports)} listening port(s) confirmed. "
                f"REVIEW before trusting: a baseline taken from a compromised "
                f"host records the compromise as normal."
            )
            if collapsed:
                entry["_collapsed_ports_warning"] = (
                    f"Port(s) {collapsed} were seen reaching {self.collapse_after}+ "
                    f"destinations and were widened to 'any destination'. A "
                    f"reverse shell from this host on {collapsed} will therefore "
                    f"NOT trip BEHAVIOR_UNEXPECTED_OUTBOUND — only "
                    f"BEHAVIOR_INTERACTIVE_SHELL will catch it, and only once "
                    f"someone types into it. Re-run with --collapse-after 0 to "
                    f"keep per-destination entries instead (noisier, stricter)."
                )
            if total_out == 0 and ports:
                entry["description"] = ("server-shaped: accepted connections, "
                                        "made none")
            hosts[host] = entry
        return hosts


def read_pcap(path: Path, obs: Observer) -> None:
    from scapy.all import PcapReader
    with PcapReader(str(path)) as reader:
        for pkt in reader:
            obs.handle(pkt)


def capture_live(iface: Optional[str], seconds: int, obs: Observer) -> None:
    from scapy.all import sniff
    print(f"[BASELINE] Capturing on {iface or '(scapy default)'} for {seconds}s. "
          f"Keep the network doing NORMAL things.", file=sys.stderr)
    sniff(iface=iface, prn=obs.handle, store=False, timeout=seconds)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="A baseline is only as trustworthy as the traffic it was taken from.")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--pcap", type=Path, help="read from a capture file")
    src.add_argument("--iface", help="capture live from this interface "
                                     "('auto' for scapy's default)")
    ap.add_argument("--seconds", type=int, default=300,
                    help="live capture duration (default 300)")
    ap.add_argument("--hosts", help="comma-separated IPs to baseline "
                                    "(default: every host seen)")
    ap.add_argument("--min-hits", type=int, default=1,
                    help="ignore destinations/ports seen fewer than N times")
    ap.add_argument("--collapse-after", type=int,
                    default=COLLAPSE_AFTER_DESTINATIONS, metavar="N",
                    help="widen a port to 'any destination' once it is seen "
                         "reaching N+ destinations (default 5). Use 0 to keep "
                         "every destination explicit: stricter, so a NEW "
                         "destination on that port alerts, but noisier")
    ap.add_argument("--merge", type=Path,
                    help="merge into this existing assets.json instead of "
                         "starting fresh")
    ap.add_argument("-o", "--output", type=Path,
                    help="write here (default: stdout)")
    args = ap.parse_args(argv)

    try:
        import scapy  # noqa: F401
    except ImportError:
        print("scapy is required: venv/Scripts/python.exe -m pip install scapy",
              file=sys.stderr)
        return 2

    only = {h.strip() for h in (args.hosts or "").split(",") if h.strip()}
    obs = Observer(only, collapse_after=args.collapse_after)

    if args.pcap:
        if not args.pcap.exists():
            print(f"No such capture: {args.pcap}", file=sys.stderr)
            return 2
        read_pcap(args.pcap, obs)
    else:
        iface = None if args.iface.lower() == "auto" else args.iface
        try:
            capture_live(iface, args.seconds, obs)
        except Exception as e:
            print(f"Capture failed ({type(e).__name__}: {e}). Live capture needs "
                  f"root/Administrator and a valid interface.", file=sys.stderr)
            return 1

    hosts = obs.build(min_hits=args.min_hits)
    print(f"[BASELINE] {obs.packets} packets -> {len(hosts)} host(s) baselined.",
          file=sys.stderr)
    if not hosts:
        print("[BASELINE] Nothing observed. Wrong interface, or no TCP traffic.",
              file=sys.stderr)

    if args.merge and args.merge.exists():
        doc = json.loads(args.merge.read_text(encoding="utf-8"))
        existing = dict(doc.get("hosts") or doc.get("servers") or {})
        for host, entry in hosts.items():
            if host in existing:
                # Never silently widen an allowlist somebody curated by hand.
                entry["_merge_note"] = ("an entry for this host already existed; "
                                        "the observed values are below for "
                                        "comparison, NOT merged automatically")
                existing[host] = {**existing[host], "_observed_rerun": entry}
            else:
                existing[host] = entry
        doc["hosts"] = existing
        doc.pop("servers", None)
    else:
        doc = {
            "_generated_by": "baseline_assets.py",
            "_warning": ("REVIEW BEFORE USE. These values were observed, not "
                         "verified. If any host was already compromised during "
                         "the capture, its attacker traffic is recorded here as "
                         "normal and will never alert again."),
            "hosts": hosts,
        }

    out = json.dumps(doc, indent=2)
    if args.output:
        args.output.write_text(out + "\n", encoding="utf-8")
        print(f"[BASELINE] Wrote {args.output}. Read it before deploying.",
              file=sys.stderr)
    else:
        print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())

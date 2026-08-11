"""Scan and brute-force rules — specific IDs, discriminating features on the alert.

What changed and why:

* Every stealth variant used to collapse into one "Stealth Scan" alert. An
  analyst could not tell a FIN scan from a NULL scan from an XMAS scan, and
  neither could the correlation engine or any future ATT&CK mapping. Each flag
  combination now has its own rule ID and the alert carries the decoded flags.

* "bruteforce" used to be one label regardless of pacing, so a hydra run and a
  deliberate one-attempt-per-minute grind were indistinguishable — even though
  they call for completely different responses. They are now split by MEASURED
  attempts/sec, and the measured rate is recorded on the alert so the
  classification can be checked rather than trusted.
"""

import time
from typing import Callable, Dict, List, Optional, Tuple

from .findings import Finding

# ── TCP flag bits ─────────────────────────────────────────────────────────────
FIN, SYN, RST, PSH, ACK, URG, ECE, CWR = (
    0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80)

_FLAG_NAMES = [(FIN, "FIN"), (SYN, "SYN"), (RST, "RST"), (PSH, "PSH"),
               (ACK, "ACK"), (URG, "URG"), (ECE, "ECE"), (CWR, "CWR")]


def decode_flags(flags: int) -> str:
    """0x29 -> 'FIN,PSH,URG'. This string goes on the alert."""
    flags = int(flags)
    names = [n for bit, n in _FLAG_NAMES if flags & bit]
    return ",".join(names) if names else "NONE"


# Exact flag value -> (rule_id, nmap technique). Order matters only for XMAS,
# which is checked as a mask below because -sX implementations vary.
_EXACT_SIGNATURES: Dict[int, Tuple[str, str]] = {
    0x00:        ("TCP_NULL_SCAN",   "nmap -sN (NULL)"),
    FIN:         ("TCP_FIN_SCAN",    "nmap -sF (FIN)"),
    SYN | FIN:   ("TCP_SYNFIN_SCAN", "SYN+FIN evasion"),
}

_XMAS_MASK = FIN | PSH | URG


def classify_tcp_signature(flags: int) -> Optional[Tuple[str, str]]:
    """Single-packet scan signature from the TCP flags, or None.

    Returns (rule_id, scan_technique). A bare SYN is deliberately NOT a
    signature — it is how every legitimate connection starts, so SYN scans are
    detected behaviourally (see PortScanDetector) rather than per-packet.
    """
    flags = int(flags)
    if flags & _XMAS_MASK == _XMAS_MASK and not flags & (SYN | RST | ACK):
        return ("TCP_XMAS_SCAN", "nmap -sX (XMAS)")
    return _EXACT_SIGNATURES.get(flags)


def signature_finding(src: str, dst: str, dport: int, flags: int,
                      ts: Optional[float] = None,
                      extra: Optional[dict] = None) -> Optional[Finding]:
    """Build the Finding for a flag-signature hit, carrying the decoded flags."""
    hit = classify_tcp_signature(flags)
    if hit is None:
        return None
    rule_id, technique = hit
    decoded = decode_flags(flags)
    details = {
        "tcp_flags":      decoded,
        "tcp_flags_hex":  f"0x{int(flags):02x}",
        "dst_port":       int(dport),
        "scan_technique": technique,
    }
    if extra:
        details.update(extra)
    return Finding(
        rule_id=rule_id, src=src, dst=dst, protocol="TCP", ts=ts,
        message=(f"{technique}: {src} -> {dst}:{dport} with TCP flags "
                 f"[{decoded}] (0x{int(flags):02x})"),
        details=details,
    )


# ──────────────────────────────────────────────────────────────────────────────
#  Port scan: SYN (half-open) vs CONNECT (full handshake)
# ──────────────────────────────────────────────────────────────────────────────
class PortScanDetector:
    """Distinguishes `nmap -sS` from `nmap -sT` by what the client does after
    the server answers.

    Both send bare SYNs to many ports, so port count alone cannot tell them
    apart. The discriminator is the client's third packet:

      * SYN -> SYN-ACK -> **RST**  = half-open, the connection was never
        completed. That is `-sS`.
      * SYN -> SYN-ACK -> **ACK**  = the handshake completed and a real socket
        existed (then usually closed immediately). That is `-sT`.

    Both counts go on the alert (`handshakes_completed`, `half_open_resets`) so
    the classification is auditable instead of asserted.
    """

    def __init__(
        self,
        port_threshold: int = 6,
        window_sec: float = 5.0,
        cooldown_sec: float = 30.0,
        max_sources: int = 4096,
        clock: Callable[[], float] = time.time,
    ):
        self.port_threshold = int(port_threshold)
        self.window_sec = float(window_sec)
        self.cooldown_sec = float(cooldown_sec)
        self.max_sources = int(max_sources)
        self.clock = clock
        # src -> {"ports": {port: ts}, "completed": set, "reset": set,
        #         "first": ts, "last_alert": ts}
        self._state: Dict[str, dict] = {}
        # (client, server, port) -> "syn" | "synack"
        self._handshakes: Dict[Tuple[str, str, int], str] = {}

    def _st(self, src: str) -> dict:
        st = self._state.get(src)
        if st is None:
            if len(self._state) >= self.max_sources:
                oldest = min(self._state, key=lambda s: self._state[s]["first"])
                del self._state[oldest]
            st = {"ports": {}, "completed": set(), "reset": set(),
                  "first": 0.0, "last_alert": 0.0, "dst": None}
            self._state[src] = st
        return st

    def observe_syn(self, src: str, dst: str, dport: int,
                    ts: Optional[float] = None) -> Optional[Finding]:
        now = self.clock() if ts is None else ts
        st = self._st(src)
        st["ports"] = {p: t for p, t in st["ports"].items()
                       if now - t <= self.window_sec}
        if not st["ports"]:
            st["first"] = now
            st["completed"].clear()
            st["reset"].clear()
        st["ports"][int(dport)] = now
        st["dst"] = dst
        self._handshakes[(src, dst, int(dport))] = "syn"

        if len(st["ports"]) < self.port_threshold:
            return None
        if now - st["last_alert"] < self.cooldown_sec:
            return None
        st["last_alert"] = now
        return self._emit(src, dst, st, now)

    def observe_synack(self, server: str, client: str, sport: int,
                       ts: Optional[float] = None) -> None:
        """Server answered — the handshake is now half-open, awaiting the client."""
        key = (client, server, int(sport))
        if self._handshakes.get(key) == "syn":
            self._handshakes[key] = "synack"

    def observe_client_ack(self, client: str, server: str, dport: int,
                           ts: Optional[float] = None) -> None:
        """Client completed the handshake => full connect()."""
        key = (client, server, int(dport))
        if self._handshakes.pop(key, None) == "synack":
            self._st(client)["completed"].add(int(dport))

    def observe_client_rst(self, client: str, server: str, dport: int,
                           ts: Optional[float] = None) -> None:
        """Client tore down after SYN-ACK without completing => half-open."""
        key = (client, server, int(dport))
        if self._handshakes.pop(key, None) == "synack":
            self._st(client)["reset"].add(int(dport))

    def _emit(self, src: str, dst: str, st: dict, now: float) -> Finding:
        ports = sorted(st["ports"])
        # 10 ms floor: packet timestamps have finite resolution, and dividing by
        # a near-zero span produced a nonsense "99469 ports/sec" on the alert.
        span = max(now - st["first"], 0.01)
        completed, resets = len(st["completed"]), len(st["reset"])
        if completed and completed >= resets:
            rule_id, technique = "TCP_CONNECT_SCAN", "nmap -sT (TCP connect)"
        else:
            rule_id, technique = "TCP_SYN_SCAN", "nmap -sS (SYN half-open)"
        return Finding(
            rule_id=rule_id, src=src, dst=dst, protocol="TCP", ts=now,
            message=(f"{technique}: {src} probed {len(ports)} distinct ports on "
                     f"{dst} in {span:.1f}s ({len(ports) / span:.1f} ports/sec, "
                     f"{completed} handshakes completed, {resets} half-open resets)"),
            details={
                "distinct_ports":       len(ports),
                "window_seconds":       round(span, 2),
                "ports_per_sec":        round(len(ports) / span, 2),
                "sample_ports":         ports[:20],
                "scan_technique":       technique,
                "handshakes_completed": completed,
                "half_open_resets":     resets,
            },
        )

    def reset(self) -> None:
        self._state.clear()
        self._handshakes.clear()


# ──────────────────────────────────────────────────────────────────────────────
#  Brute force: split by measured rate
# ──────────────────────────────────────────────────────────────────────────────
AUTH_SERVICES: Dict[int, str] = {
    21: "ftp", 22: "ssh", 23: "telnet", 25: "smtp", 110: "pop3", 143: "imap",
    389: "ldap", 445: "smb", 993: "imaps", 995: "pop3s", 1433: "mssql",
    3306: "mysql", 3389: "rdp", 5432: "postgres", 5900: "vnc", 8080: "http-alt",
}

# Ports actually watched for brute force. NOT the same as AUTH_SERVICES, which
# is only a display-name lookup.
#
# 8080 is deliberately excluded. It is overwhelmingly an application/API port,
# so counting repeated connections to it as login attempts turns ordinary
# polling into a HIGH "brute force" alert — a false positive caught by
# tests/test_benign_regression.py::test_polling_a_few_different_api_ports_is_clean.
# HTTP auth cannot be judged from connection counts anyway; that needs response
# codes, which this sensor does not parse.
DEFAULT_BRUTEFORCE_PORTS = frozenset(
    {21, 22, 23, 25, 110, 143, 389, 445, 993, 995, 1433, 3306, 3389, 5432, 5900})

# The line between the two rules. One attempt every 5 seconds: faster than a
# human retyping a password, slower than any tool running at default settings.
DEFAULT_RATE_SPLIT = 0.2   # attempts/sec


class BruteForceDetector:
    """Counts authentication attempts per (source, target, service) and classifies
    by the rate actually measured.

    A single long window is used for both rules; the classification comes from
    the observed attempts/sec, not from which window happened to trip. That way
    the alert can state *why* it is LOW_RATE — the measured figure is on it.

    Network-only caveat, stated on every alert: this counts connection attempts
    to an auth service. TCP completes for failed logins too, so the sensor
    cannot see success from the wire. Confirmation comes from host_log_sensor.py.
    """

    def __init__(
        self,
        min_attempts: int = 8,
        window_sec: float = 900.0,
        rate_split: float = DEFAULT_RATE_SPLIT,
        cooldown_sec: float = 60.0,
        ports: Optional[set] = None,
        max_keys: int = 8192,
        clock: Callable[[], float] = time.time,
    ):
        self.min_attempts = int(min_attempts)
        self.window_sec = float(window_sec)
        self.rate_split = float(rate_split)
        self.cooldown_sec = float(cooldown_sec)
        self.ports = set(ports) if ports else set(DEFAULT_BRUTEFORCE_PORTS)
        self.max_keys = int(max_keys)
        self.clock = clock
        self._attempts: Dict[Tuple[str, str, int], List[float]] = {}
        self._last_alert: Dict[Tuple[str, str, int], float] = {}

    def observe_attempt(self, src: str, dst: str, dport: int,
                        ts: Optional[float] = None) -> Optional[Finding]:
        dport = int(dport)
        if dport not in self.ports:
            return None
        now = self.clock() if ts is None else ts
        key = (src, dst, dport)

        hist = self._attempts.setdefault(key, [])
        hist[:] = [t for t in hist if now - t <= self.window_sec]
        hist.append(now)
        self._cap()

        if len(hist) < self.min_attempts:
            return None
        if now - self._last_alert.get(key, 0.0) < self.cooldown_sec:
            return None

        span = hist[-1] - hist[0]
        # (n-1) intervals over the span: the honest per-second attempt rate.
        rate = (len(hist) - 1) / span if span > 0 else float(len(hist))
        if rate >= self.rate_split:
            rule_id, rate_class = "BRUTEFORCE_STANDARD_RATE", "standard"
        else:
            rule_id, rate_class = "BRUTEFORCE_LOW_RATE", "low-and-slow"

        self._last_alert[key] = now
        service = AUTH_SERVICES.get(dport, f"tcp/{dport}")
        return Finding(
            rule_id=rule_id, src=src, dst=dst, protocol="TCP", ts=now,
            message=(f"Brute force ({rate_class}): {src} -> {dst}:{dport} "
                     f"({service}), {len(hist)} attempts over {span:.1f}s = "
                     f"{rate:.3f} attempts/sec "
                     f"[threshold {self.rate_split} /sec]"),
            details={
                "attempts":          len(hist),
                "attempts_per_sec":  round(rate, 4),
                "rate_class":        rate_class,
                "rate_split_per_sec": self.rate_split,
                "window_seconds":    round(span, 2),
                "observation_window_seconds": self.window_sec,
                "service":           service,
                "dst_port":          dport,
                "confidence": ("connection attempts only; TCP completes for failed "
                               "logins too, so success is not visible on the wire"),
            },
        )

    def _cap(self) -> None:
        if len(self._attempts) <= self.max_keys:
            return
        oldest = sorted(self._attempts, key=lambda k: self._attempts[k][-1])
        for k in oldest[: len(self._attempts) - self.max_keys]:
            del self._attempts[k]
            self._last_alert.pop(k, None)

    def reset(self) -> None:
        self._attempts.clear()
        self._last_alert.clear()

"""TIER 2 — a session on a TLS port that never negotiated TLS.

## The gap this closes

`BEHAVIOR_INTERACTIVE_SHELL` catches a shell on tcp/443 by its traffic shape,
but it needs ~40 packets over ~20 seconds of somebody actually typing before it
can say so. A beacon that checks in and leaves never produces that shape at all,
and an operator who runs three commands and disconnects may not either.

This rule answers a much narrower question with far less evidence: **did this
connection to a TLS port begin with a TLS handshake?** If it did not, whatever is
riding that port is not HTTPS, and it says so on the first data packet.

## Why it is cheap and high-fidelity

TLS is self-announcing. Every TLS 1.0-1.3 client opens with a record header:

    16 03 0X ...          content type 22 (handshake), version 3.x

Three bytes, at a fixed offset, on the first application byte the client sends.
There is no state to keep, no baseline to learn, and no threshold to tune — the
handshake is either there or it is not. Attackers pick 443 precisely because
egress filters allow it; they very often do not bother wrapping the channel in
real TLS, because the port is what gets them out, not the protocol.

## Why it only looks at the client, and only after a SYN

The verdict is only meaningful on the **first byte of the first client data
packet of a connection we watched open**. Two consequences, both deliberate:

  * **No SYN, no verdict.** If the sensor starts mid-session, the first packet it
    sees is somewhere in the middle of a stream, where an encrypted TLS record
    (content type 23) is indistinguishable from cleartext to this test. Rather
    than guess, flows with no observed SYN are ignored outright. This is the
    single biggest false-positive source and closing it costs one dict lookup.
  * **Client side only.** The server's reply is not checked: a server answering
    a cleartext request in cleartext tells us nothing the request did not.

## What the payload class buys

Plaintext HTTP on 443 is a real and fairly common misconfiguration — a listener
bound to the wrong port, a health check aimed at the wrong scheme. Calling that
"C2" would burn the rule's credibility. So the cleartext is classified:

  * looks like HTTP  -> `medium`, described as misconfiguration-or-tunnel
  * anything else    -> `critical`, because no benign service speaks an unknown
                        binary or shell-ish protocol on 443

The first bytes are recorded in hex on the finding either way, so an analyst can
confirm the call rather than take it on trust.

## Honest limits

  * **A C2 that does wrap itself in real TLS is invisible here**, and most mature
    frameworks do. This rule catches the lazy case, which is a large share of
    commodity tooling but not the careful operator. JA3/JA4 fingerprinting is
    what addresses that, and it is not implemented yet.
  * **It needs the client's FIRST data segment.** If that segment is lost or
    arrives out of order, the next one carries mid-stream bytes that would look
    like cleartext. The SYN's sequence number is recorded so this is detectable:
    a first data packet whose seq is past the expected one abandons the flow
    unjudged. Where the caller supplies no sequence numbers the check degrades
    to trusting arrival order, which is correct on the overwhelming majority of
    flows and wrong only on reordered ones.
  * **It cannot see inside a legitimate TLS session**, so an exploit or a shell
    tunnelled over genuine HTTPS is out of scope by construction.
"""

import time
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Optional, Set, Tuple

from .findings import Finding

# Ports where a TLS handshake is the convention, not merely one option. Ports
# with a cleartext-then-STARTTLS mode (587, 143, 21, 25) are deliberately NOT
# here: on those, opening in cleartext is correct behaviour.
DEFAULT_TLS_PORTS: Set[int] = {
    443,    # HTTPS
    465,    # SMTPS (implicit TLS)
    563,    # NNTPS
    636,    # LDAPS
    989,    # FTPS data
    990,    # FTPS control
    993,    # IMAPS
    995,    # POP3S
    5061,   # SIP-TLS
    6697,   # IRCS
    8443,   # HTTPS-alt
}

# Enough of the opening bytes to classify and to record as evidence.
SNIFF_BYTES = 16

_HTTP_OPENERS = (
    b"GET ", b"POST ", b"HEAD ", b"PUT ", b"DELETE ", b"OPTIONS ", b"PATCH ",
    b"TRACE ", b"CONNECT ", b"HTTP/",
)


def looks_like_tls_handshake(payload: bytes) -> bool:
    """True if `payload` opens with a TLS (or legacy SSLv2) handshake record.

    TLS 1.0 through 1.3 all begin the same way — content type 22, major version
    3 — because TLS 1.3 keeps the legacy record version on the wire for
    middlebox compatibility. So one check covers every version in use.
    """
    if len(payload) < 3:
        return False
    # SSLv2-style ClientHello: high bit set length prefix, msg type 1 at byte 2.
    # Effectively extinct, but cheap to allow so we never call it cleartext.
    if (payload[0] & 0x80) and payload[2] == 0x01:
        return True
    return payload[0] == 0x16 and payload[1] == 0x03 and payload[2] <= 0x04


def classify_cleartext(payload: bytes) -> Tuple[str, str]:
    """Label what the non-TLS opening bytes look like -> (class, description)."""
    head = payload[:SNIFF_BYTES]
    if any(head.startswith(v) for v in _HTTP_OPENERS):
        return ("http", "plaintext HTTP")
    printable = sum(1 for b in head if 32 <= b < 127 or b in (9, 10, 13))
    if head and printable / len(head) >= 0.9:
        return ("text", "an unidentified plaintext protocol")
    return ("binary", "an unidentified binary protocol")


@dataclass
class _PendingFlow:
    client: str
    server: str
    cport: int
    sport: int
    opened_ts: float
    # Sequence number the client's first data byte should carry (SYN's seq + 1).
    # None when the caller does not supply sequence numbers.
    expected_seq: Optional[int] = None


class TLSPortInspector:
    """Flag connections to TLS ports whose first client bytes are not a handshake."""

    def __init__(
        self,
        tls_ports: Optional[Iterable[int]] = None,
        cooldown_sec: float = 300.0,
        max_flows: int = 8192,
        clock: Callable[[], float] = time.time,
    ):
        self.tls_ports: Set[int] = set(int(p) for p in (tls_ports
                                                        if tls_ports is not None
                                                        else DEFAULT_TLS_PORTS))
        self.cooldown_sec = float(cooldown_sec)
        self.max_flows = int(max_flows)
        self.clock = clock
        # Keyed in the CLIENT->SERVER orientation, so a server's reply simply
        # fails the lookup and is ignored without an explicit direction test.
        self._pending: Dict[Tuple[str, int, str, int], _PendingFlow] = {}
        self._last_alert: Dict[Tuple[str, str, int], float] = {}

    # ── flow tracking ─────────────────────────────────────────────────────────
    def open_flow(self, client: str, server: str, cport: int, sport: int,
                  ts: Optional[float] = None, seq: Optional[int] = None) -> None:
        """Called on the client's SYN. Without this the flow is never judged.

        `seq` is the SYN's sequence number; the first data byte will carry
        `seq + 1`. Supplying it is what lets a reordered first segment be
        recognised instead of mistaken for the opening bytes.
        """
        sport = int(sport)
        if sport not in self.tls_ports:
            return
        now = self.clock() if ts is None else ts
        self._cap()
        key = (client, int(cport), server, sport)
        self._pending[key] = _PendingFlow(
            client=client, server=server, cport=int(cport), sport=sport,
            opened_ts=now,
            expected_seq=None if seq is None else (int(seq) + 1) & 0xFFFFFFFF)

    def close_flow(self, a: str, b: str, aport: int, bport: int) -> None:
        self._pending.pop((a, int(aport), b, int(bport)), None)
        self._pending.pop((b, int(bport), a, int(aport)), None)

    def observe_data(self, src: str, dst: str, sport: int, dport: int,
                     payload: bytes, ts: Optional[float] = None,
                     seq: Optional[int] = None) -> Optional[Finding]:
        """Judge the first data packet a tracked client sends. One verdict per flow."""
        if not payload:
            return None
        key = (src, int(sport), dst, int(dport))
        flow = self._pending.pop(key, None)
        if flow is None:
            return None          # server reply, mid-stream flow, or non-TLS port

        # Reordering guard: if this is not the segment carrying the connection's
        # first data byte, its bytes are mid-stream and prove nothing. Abandon
        # the flow rather than judge them — a wrong verdict here is precisely
        # the false positive the SYN requirement exists to prevent.
        if (flow.expected_seq is not None and seq is not None
                and (int(seq) & 0xFFFFFFFF) != flow.expected_seq):
            return None

        if looks_like_tls_handshake(payload):
            return None          # negotiated TLS: nothing to say

        now = self.clock() if ts is None else ts
        alert_key = (flow.client, flow.server, flow.sport)
        if now - self._last_alert.get(alert_key, 0.0) < self.cooldown_sec:
            return None
        self._last_alert[alert_key] = now

        payload_class, description = classify_cleartext(payload)
        head_hex = payload[:SNIFF_BYTES].hex()

        if payload_class == "http":
            severity = "medium"
            verdict = (f"This is {description}, which is most often a "
                       f"misconfiguration (a listener on the wrong port, a health "
                       f"check using the wrong scheme). It is also how HTTP-based "
                       f"C2 that does not bother with real TLS looks, so confirm "
                       f"what {flow.server} is before dismissing it.")
            indicator = "cleartext HTTP on a TLS port (misconfiguration or C2)"
        else:
            severity = "critical"
            verdict = (f"This is {description}. No legitimate service speaks "
                       f"anything but TLS on tcp/{flow.sport}, so a session that "
                       f"skips the handshake is a channel using the port purely "
                       f"because egress filters allow it.")
            indicator = "non-TLS session on a TLS port (C2 / reverse shell)"

        return Finding(
            rule_id="BEHAVIOR_CLEARTEXT_ON_TLS_PORT",
            src=flow.client, dst=flow.server, protocol="TCP", ts=now,
            severity_override=severity,
            message=(f"No TLS handshake: {flow.client} opened a connection to "
                     f"{flow.server}:{flow.sport} and its first data packet does "
                     f"not begin with a TLS record (first bytes {head_hex}). "
                     f"{verdict}"),
            details={
                "dst_port":        flow.sport,
                "destination":     flow.server,
                "payload_class":   payload_class,
                "first_bytes_hex": head_hex,
                "src_port":        flow.cport,
                "handshake_seen":  False,
                "seconds_to_first_byte": round(now - flow.opened_ts, 3),
                "indicator":       indicator,
                "tier":            2,
                "note": ("judged on the first client data packet of a connection "
                         "whose SYN was observed; mid-stream flows are never "
                         "judged. A C2 that wraps itself in real TLS defeats this "
                         "rule — JA3/JA4 fingerprinting is what addresses that."),
            },
        )

    # ── housekeeping ──────────────────────────────────────────────────────────
    def _cap(self) -> None:
        if len(self._pending) < self.max_flows:
            return
        oldest = sorted(self._pending, key=lambda k: self._pending[k].opened_ts)
        for k in oldest[: len(self._pending) - self.max_flows + 1]:
            del self._pending[k]

    def sweep(self, ts: Optional[float] = None, idle_sec: float = 300.0) -> int:
        """Drop flows that opened but never sent data. Returns how many were dropped."""
        now = self.clock() if ts is None else ts
        stale = [k for k, f in self._pending.items()
                 if now - f.opened_ts > idle_sec]
        for k in stale:
            del self._pending[k]
        return len(stale)

    @property
    def tracked_flows(self) -> int:
        return len(self._pending)

    def reset(self) -> None:
        self._pending.clear()
        self._last_alert.clear()

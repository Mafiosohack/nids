"""IPv4 fragment reassembly, run BEFORE rule matching.

## Why this exists

`nmap -f` splits the 20-byte TCP header across 8-byte IP fragments. Only the
first fragment carries source/destination ports (offsets 0-3); the flags byte
lives at TCP offset 13, which lands in the *second* fragment. So a scan-rule
engine that inspects fragments as they arrive sees:

  * fragment 0 — ports, but no flags
  * fragment 1 — flags, but no ports and no TCP layer at all

Neither matches a scan signature, which is why `-f` evaded detection entirely.
Reassembling first means a fragmented `-sF` scan normalises to exactly the same
`TCP_FIN_SCAN` signature as an unfragmented one.

## Design

Pure-bytes core (`FragmentReassembler`) so tests need no packet library, plus a
thin scapy adapter (`ScapyDefragmenter`) for the live sensor.

Fragments are tracked per datagram key `(src, dst, ip_id, proto)` — the correct
key per RFC 791; ip_id alone collides across hosts. Coverage is tracked as merged
byte intervals rather than a fragment count, so overlapping and duplicate
fragments (the classic evasion: send conflicting overlapping fragments and let
the host's reassembly policy pick) cannot fake completeness.

Bounded on every axis: per-datagram byte cap, buffer count cap, and a TTL sweep,
so a fragment flood cannot exhaust memory. Anything dropped is counted in
`stats` and reported through `/status` rather than disappearing silently.
"""

import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

DEFAULT_TIMEOUT_SEC = 30.0      # RFC 791 reassembly timeout is 15s; be generous
DEFAULT_MAX_BUFFERS = 1024      # concurrent partially-reassembled datagrams
DEFAULT_MAX_DATAGRAM = 65535    # an IPv4 datagram cannot exceed this
DEFAULT_MAX_FRAGMENTS = 128     # per datagram


@dataclass
class _Buffer:
    key: Tuple
    first_seen: float
    last_seen: float
    fragments: int = 0
    total_len: Optional[int] = None       # known once the MF=0 fragment arrives
    chunks: List[Tuple[int, bytes]] = field(default_factory=list)
    intervals: List[Tuple[int, int]] = field(default_factory=list)  # merged [start,end)
    header: Optional[bytes] = None        # first fragment's transport-header bytes

    def coverage_complete(self) -> bool:
        return (self.total_len is not None
                and len(self.intervals) == 1
                and self.intervals[0] == (0, self.total_len))


@dataclass
class Reassembled:
    """A datagram put back together. `payload` is the full transport-layer bytes."""
    src: str
    dst: str
    proto: int
    ip_id: int
    payload: bytes
    fragment_count: int
    elapsed_sec: float


class FragmentReassembler:
    def __init__(
        self,
        timeout_sec: float = DEFAULT_TIMEOUT_SEC,
        max_buffers: int = DEFAULT_MAX_BUFFERS,
        max_datagram_bytes: int = DEFAULT_MAX_DATAGRAM,
        max_fragments: int = DEFAULT_MAX_FRAGMENTS,
        clock: Callable[[], float] = time.time,
    ):
        self.timeout_sec = float(timeout_sec)
        self.max_buffers = int(max_buffers)
        self.max_datagram_bytes = int(max_datagram_bytes)
        self.max_fragments = int(max_fragments)
        self.clock = clock
        self._buffers: Dict[Tuple, _Buffer] = {}
        self.stats = {
            "fragments_seen": 0,
            "datagrams_reassembled": 0,
            "datagrams_timed_out": 0,
            "datagrams_dropped_oversize": 0,
            "buffers_evicted": 0,
            "overlaps_seen": 0,
        }

    def add_fragment(
        self,
        src: str,
        dst: str,
        proto: int,
        ip_id: int,
        frag_offset_units: int,   # IP header field: units of 8 bytes
        more_fragments: bool,
        payload: bytes,
        ts: Optional[float] = None,
    ) -> Optional[Reassembled]:
        """Feed one fragment. Returns the completed datagram, or None if still partial.

        `frag_offset_units` is the raw IP `frag` field (8-byte units), matching
        what scapy exposes as `pkt[IP].frag`.
        """
        now = self.clock() if ts is None else ts
        self.stats["fragments_seen"] += 1
        self.sweep(now)

        offset = int(frag_offset_units) * 8
        end = offset + len(payload)
        if end > self.max_datagram_bytes:
            self.stats["datagrams_dropped_oversize"] += 1
            return None

        key = (src, dst, int(proto), int(ip_id))
        buf = self._buffers.get(key)
        if buf is None:
            self._evict_if_full(now)
            buf = _Buffer(key=key, first_seen=now, last_seen=now)
            self._buffers[key] = buf

        buf.last_seen = now
        buf.fragments += 1
        if buf.fragments > self.max_fragments:
            del self._buffers[key]
            self.stats["datagrams_dropped_oversize"] += 1
            return None

        buf.chunks.append((offset, payload))
        self._merge(buf, offset, end)

        if not more_fragments:
            # Last fragment defines the datagram's total length.
            buf.total_len = end

        if not buf.coverage_complete():
            return None

        del self._buffers[key]
        self.stats["datagrams_reassembled"] += 1
        return Reassembled(
            src=src, dst=dst, proto=int(proto), ip_id=int(ip_id),
            payload=self._assemble(buf),
            fragment_count=buf.fragments,
            elapsed_sec=round(now - buf.first_seen, 4),
        )

    # ── internals ─────────────────────────────────────────────────────────────
    def _merge(self, buf: _Buffer, start: int, end: int) -> None:
        """Insert [start, end) into the merged interval list."""
        if end <= start:
            return
        merged: List[Tuple[int, int]] = []
        placed = False
        for s, e in buf.intervals:
            if e < start:
                merged.append((s, e))
            elif end < s:
                if not placed:
                    merged.append((start, end))
                    placed = True
                merged.append((s, e))
            else:
                self.stats["overlaps_seen"] += 1
                start, end = min(s, start), max(e, end)
        if not placed:
            merged.append((start, end))
        merged.sort()
        # Coalesce adjacent runs so a complete datagram collapses to one interval.
        out: List[Tuple[int, int]] = []
        for s, e in merged:
            if out and s <= out[-1][1]:
                out[-1] = (out[-1][0], max(out[-1][1], e))
            else:
                out.append((s, e))
        buf.intervals = out

    def _assemble(self, buf: _Buffer) -> bytes:
        """Join fragments. First-writer-wins on overlaps.

        This mirrors the BSD/Linux reassembly policy. Overlapping-fragment
        evasion works precisely because different stacks resolve conflicts
        differently; we pick one, record that an overlap happened in `stats`, and
        match rules against the result rather than dropping the datagram (which
        would hand the attacker an easy bypass).
        """
        out = bytearray(buf.total_len or 0)
        written = bytearray(len(out))
        for offset, data in buf.chunks:
            for i, byte in enumerate(data):
                pos = offset + i
                if pos < len(out) and not written[pos]:
                    out[pos] = byte
                    written[pos] = 1
        return bytes(out)

    def sweep(self, now: Optional[float] = None) -> int:
        """Drop datagrams that never completed. Returns how many were dropped."""
        now = self.clock() if now is None else now
        stale = [k for k, b in self._buffers.items()
                 if now - b.first_seen > self.timeout_sec]
        for k in stale:
            del self._buffers[k]
        self.stats["datagrams_timed_out"] += len(stale)
        return len(stale)

    def _evict_if_full(self, now: float) -> None:
        if len(self._buffers) < self.max_buffers:
            return
        oldest = min(self._buffers, key=lambda k: self._buffers[k].last_seen)
        del self._buffers[oldest]
        self.stats["buffers_evicted"] += 1

    @property
    def pending(self) -> int:
        return len(self._buffers)


# ──────────────────────────────────────────────────────────────────────────────
#  Scapy adapter — used by the live sensor, skipped entirely by the unit tests
#  that exercise the pure core above.
# ──────────────────────────────────────────────────────────────────────────────
class ScapyDefragmenter:
    """Wraps FragmentReassembler for scapy packets.

    Usage in the packet path:

        result = defrag.process(pkt)
        if result is FRAGMENT_PENDING:   # partial — nothing to match yet
            return
        pkt = result                     # either the original or the rebuilt one
    """

    PENDING = object()   # sentinel: fragment consumed, datagram not yet complete

    def __init__(self, **kwargs):
        self.core = FragmentReassembler(**kwargs)
        self.last_fragment_count = 0

    @staticmethod
    def is_fragment(pkt) -> bool:
        try:
            from scapy.all import IP
        except ImportError:      # pragma: no cover - live path only
            return False
        if not pkt.haslayer(IP):
            return False
        ip = pkt[IP]
        more_frags = bool(int(ip.flags) & 0x01)   # MF bit
        return more_frags or int(ip.frag) > 0

    def process(self, pkt, ts: Optional[float] = None):
        """Return the packet to match against, PENDING, or the original packet.

        Non-fragments pass straight through untouched, so this costs one flag
        test on the overwhelming majority of traffic.
        """
        from scapy.all import IP

        if not self.is_fragment(pkt):
            self.last_fragment_count = 0
            return pkt

        ip = pkt[IP]
        payload = bytes(ip.payload)
        done = self.core.add_fragment(
            src=ip.src, dst=ip.dst, proto=int(ip.proto), ip_id=int(ip.id),
            frag_offset_units=int(ip.frag),
            more_fragments=bool(int(ip.flags) & 0x01),
            payload=payload, ts=ts,
        )
        if done is None:
            return self.PENDING

        self.last_fragment_count = done.fragment_count
        # Rebuild a clean, unfragmented IP packet carrying the whole transport
        # payload. Re-parsing through IP() makes scapy dissect the TCP/UDP layer
        # normally, so every downstream rule sees an ordinary packet.
        rebuilt = IP(src=done.src, dst=done.dst, proto=done.proto,
                     id=done.ip_id, flags=0, frag=0) / done.payload
        return IP(bytes(rebuilt))

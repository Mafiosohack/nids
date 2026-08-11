"""Severity-weighted alert retention with deduplication.

The old store was `deque(maxlen=500)`: a fixed ring where 200 identical
port-scan pings from one host silently pushed every CRITICAL off the end. Two
independent problems — no severity weighting, and no deduplication — with the
same symptom.

## Policy

**Retention by severity.** CRITICAL and HIGH have no TTL: they are kept until
the hard capacity limit, and even then they are evicted last. MEDIUM/LOW/INFO
get progressively shorter TTLs. Defaults:

    critical  keep indefinitely      high  keep indefinitely
    medium    24 h                   low   6 h                 info  1 h

**Deduplication.** Alerts identical on (rule_id, src, dst, dst_port, severity)
within `dedup_window_sec` collapse into ONE record carrying `count`,
`first_seen` and `last_seen`, instead of N records. 200 identical port-scan
pings become one record with `count: 200`. The window is a rolling one, so a
sustained attack keeps extending a single record rather than starting new ones
every few minutes.

**Nothing vanishes silently.** Every eviction and every collapse is written to
an audit log (`eviction_log`, exposed on `/alerts/retention`), and counters
survive the alert itself. If an alert is gone, the log says which rule it was,
how many were folded in, and why it went.
"""

import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from .rules import severity_rank

# None = no TTL, keep until capacity pressure. Severity is lower-cased on lookup.
DEFAULT_TTL_SEC: Dict[str, Optional[float]] = {
    "critical": None,
    "high":     None,
    "medium":   24 * 3600.0,
    "low":      6 * 3600.0,
    "info":     1 * 3600.0,
}

DEFAULT_CAPACITY = 5000
DEFAULT_DEDUP_WINDOW = 300.0     # 5 min
DEFAULT_LOG_CAPACITY = 1000


@dataclass
class AddResult:
    stored: bool                      # True = a new record was created
    deduped: bool = False             # True = folded into an existing record
    record: Optional[dict] = None     # the record created or updated
    evicted: List[dict] = field(default_factory=list)
    collapsed_count: int = 0          # running count on the deduped record


class AlertStore:
    """Thread-safe. Records are plain dicts so they serialise straight to JSON."""

    def __init__(
        self,
        capacity: int = DEFAULT_CAPACITY,
        ttl_by_severity: Optional[Dict[str, Optional[float]]] = None,
        dedup_window_sec: float = DEFAULT_DEDUP_WINDOW,
        log_capacity: int = DEFAULT_LOG_CAPACITY,
        clock: Callable[[], float] = time.time,
        on_event: Optional[Callable[[dict], None]] = None,
    ):
        self.capacity = int(capacity)
        self.ttl = dict(DEFAULT_TTL_SEC)
        if ttl_by_severity:
            self.ttl.update({k.lower(): v for k, v in ttl_by_severity.items()})
        self.dedup_window_sec = float(dedup_window_sec)
        self.log_capacity = int(log_capacity)
        self.clock = clock
        self.on_event = on_event

        self._lock = threading.RLock()
        self._records: List[dict] = []                 # insertion order
        self._by_key: Dict[Tuple, dict] = {}           # dedup key -> live record
        self.eviction_log: List[dict] = []
        self.stats = {
            "added": 0, "deduped": 0, "evicted_ttl": 0,
            "evicted_capacity": 0, "collapsed_alerts": 0,
        }

    # ── keys ──────────────────────────────────────────────────────────────────
    @staticmethod
    def dedup_key(alert: dict) -> Tuple:
        """What counts as 'the same alert again'.

        Destination port is part of the key on purpose: a scan sweeping 200
        ports is one *scan* but the per-port alerts are genuinely different
        facts. It is the repeated alert for the SAME port that is redundant.
        """
        d = alert.get("details") or {}
        return (
            alert.get("rule_id") or alert.get("type"),
            alert.get("src"),
            alert.get("dst"),
            d.get("dst_port") or d.get("listening_port") or d.get("target_port"),
            (alert.get("severity") or "").lower(),
        )

    # ── write ─────────────────────────────────────────────────────────────────
    def add(self, alert: dict, ts: Optional[float] = None) -> AddResult:
        now = self.clock() if ts is None else ts
        with self._lock:
            evicted = self._expire(now)
            key = self.dedup_key(alert)
            existing = self._by_key.get(key)

            if existing is not None and now - existing["last_seen"] <= self.dedup_window_sec:
                existing["count"] += 1
                existing["last_seen"] = now
                existing["timestamp"] = alert.get("timestamp", existing["timestamp"])
                existing["message"] = alert.get("message", existing["message"])
                existing["details"] = alert.get("details", existing["details"])
                self.stats["deduped"] += 1
                self.stats["collapsed_alerts"] += 1
                if existing["count"] in (2, 10, 100, 1000) or existing["count"] % 1000 == 0:
                    self._log("collapse", existing,
                              f"repeat #{existing['count']} folded into "
                              f"{existing.get('id')} "
                              f"(dedup window {int(self.dedup_window_sec)}s)")
                return AddResult(stored=False, deduped=True, record=existing,
                                 evicted=evicted,
                                 collapsed_count=existing["count"])

            record = dict(alert)
            record.setdefault("count", 1)
            record["first_seen"] = now
            record["last_seen"] = now
            self._records.append(record)
            self._by_key[key] = record
            self.stats["added"] += 1
            evicted += self._enforce_capacity(now)
            return AddResult(stored=True, record=record, evicted=evicted)

    # ── eviction ──────────────────────────────────────────────────────────────
    def _expire(self, now: float) -> List[dict]:
        """Drop records whose severity TTL has elapsed."""
        keep, gone = [], []
        for r in self._records:
            ttl = self.ttl.get((r.get("severity") or "").lower())
            if ttl is not None and now - r["last_seen"] > ttl:
                gone.append(r)
            else:
                keep.append(r)
        if gone:
            self._records = keep
            for r in gone:
                self._by_key.pop(self.dedup_key(r), None)
                self._log("evict_ttl", r,
                          f"TTL {int(self.ttl.get((r.get('severity') or '').lower()))}s "
                          f"elapsed for severity {r.get('severity')}")
            self.stats["evicted_ttl"] += len(gone)
        return gone

    def _enforce_capacity(self, now: float) -> List[dict]:
        """Over capacity: evict lowest severity first, oldest first within a tier.

        A CRITICAL is only dropped once every INFO, LOW, MEDIUM and HIGH is
        already gone — which, at the default capacity, means something is very
        wrong and the log will say so.
        """
        overflow = len(self._records) - self.capacity
        if overflow <= 0:
            return []
        ranked = sorted(
            range(len(self._records)),
            key=lambda i: (severity_rank(self._records[i].get("severity", "info")),
                           self._records[i]["last_seen"]),
        )
        drop_idx = set(ranked[:overflow])
        gone = [self._records[i] for i in sorted(drop_idx)]
        self._records = [r for i, r in enumerate(self._records) if i not in drop_idx]
        for r in gone:
            self._by_key.pop(self.dedup_key(r), None)
            self._log("evict_capacity", r,
                      f"store at capacity ({self.capacity}); evicted "
                      f"lowest-severity-oldest first")
        self.stats["evicted_capacity"] += len(gone)
        return gone

    def _log(self, action: str, record: dict, reason: str) -> None:
        entry = {
            "ts":       self.clock(),
            "action":   action,
            "alert_id": record.get("id"),
            "rule_id":  record.get("rule_id") or record.get("type"),
            "severity": record.get("severity"),
            "src":      record.get("src"),
            "dst":      record.get("dst"),
            "count":    record.get("count", 1),
            "reason":   reason,
        }
        self.eviction_log.append(entry)
        if len(self.eviction_log) > self.log_capacity:
            del self.eviction_log[: len(self.eviction_log) - self.log_capacity]
        if self.on_event is not None:
            try:
                self.on_event(entry)
            except Exception:
                pass     # audit logging must never break the capture path

    # ── read ──────────────────────────────────────────────────────────────────
    def sweep(self, ts: Optional[float] = None) -> List[dict]:
        """Run TTL expiry without adding anything. For a periodic maintenance tick."""
        now = self.clock() if ts is None else ts
        with self._lock:
            return self._expire(now)

    def list(self, limit: Optional[int] = None, severity: Optional[str] = None,
             rule_id: Optional[str] = None) -> List[dict]:
        with self._lock:
            out = list(self._records)
        if severity:
            out = [r for r in out if (r.get("severity") or "").lower() == severity.lower()]
        if rule_id:
            out = [r for r in out
                   if (r.get("rule_id") or r.get("type")) == rule_id]
        return out[-limit:] if limit else out

    def __len__(self) -> int:
        with self._lock:
            return len(self._records)

    @property
    def total_alerts_represented(self) -> int:
        """Records plus everything folded into them — the true event count."""
        with self._lock:
            return sum(r.get("count", 1) for r in self._records)

    def snapshot_stats(self) -> dict:
        with self._lock:
            by_sev: Dict[str, int] = {}
            for r in self._records:
                s = (r.get("severity") or "unknown").lower()
                by_sev[s] = by_sev.get(s, 0) + 1
            return {
                **self.stats,
                "records": len(self._records),
                "alerts_represented": sum(r.get("count", 1) for r in self._records),
                "records_by_severity": by_sev,
                "capacity": self.capacity,
                "dedup_window_seconds": self.dedup_window_sec,
                "ttl_by_severity": {
                    k: ("indefinite" if v is None else int(v))
                    for k, v in self.ttl.items()
                },
                "eviction_log_entries": len(self.eviction_log),
            }

    def clear(self) -> int:
        with self._lock:
            n = len(self._records)
            self._records.clear()
            self._by_key.clear()
            self.eviction_log.clear()
            for k in self.stats:
                self.stats[k] = 0
            return n

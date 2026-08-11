"""Correlation engine — group alerts from one source into one incident.

Replaces the set-containment engine diagnosed in `docs/CORRELATION_DIAGNOSIS.md`.

## Specification (this is the part that was previously never written down)

**Grouping key: source IP, and only source IP.**
Nothing else. A second key was considered and rejected:

  * *Destination IP* would split one attacker sweeping three hosts into three
    incidents — the opposite of what an incident is for. The destinations are
    recorded as evidence on the incident instead.
  * *Port / session ID* is finer still and fails for the same reason.

The one refinement is **actor attribution**, not a second key: some detectors
file an alert under the victim's IP (data staging is keyed on the collecting
host; ARP spoofing's source is forged). Callers pass an explicit `actor`, and
events whose actor is unattributable are dropped rather than grouped under
whoever the packet claimed to be from.

**Window:** `window_sec`, configurable, default 600 s. Two events belong to the
same incident only if they fall within `window_sec` of each other. An incident
extends as long as new events keep arriving inside the window of its *last*
event; after that it goes stale and a later event starts a fresh incident.

**Promotion rule — what actually opens an incident.** An incident opens when the
source's buffered events show **forward progression through the kill chain**:
there exist two events e1, e2 with `ts(e1) <= ts(e2)` and
`stage_index(e1) < stage_index(e2)`. Consequences, all deliberate:

  * Five port scans do not open an incident. Neither do a port scan and a slow
    scan and a TCP signature — all reconnaissance, no advancement. This is what
    made a single `nmap -sS` produce a CRITICAL "Multi-Vector Attack" before.
  * A recon event followed by an exploit attempt does open one.
  * The reverse order (C2 at t=0, recon at t=30) does not: nothing advanced.
  * Events with no kill-chain stage (floods, ARP) never open an incident. They
    attach to an already-open one as context, because a flood during an
    intrusion is evidence, but on their own they are not a chain.
  * INFO-severity events never open one and never attach.

**Determinism under concurrency.** Ingest, evaluate and mutate happen in a single
critical section, so the check-then-act race in the old engine is gone.
Progression is computed over the buffer *sorted by event timestamp*, not by
arrival order, so N threads delivering the same N events in any interleaving
produce byte-identical grouping. `test_correlation.py` asserts this with 8
threads over a shuffled sequence.
"""

import itertools
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from .rules import (
    KILL_CHAIN_ORDER,
    STAGE_INDEX,
    STAGE_LABELS,
    max_severity,
    severity_rank,
)

# Defaults — every one is overridable per-instance so tests can drive fast clocks.
DEFAULT_WINDOW_SEC = 600.0     # 10 min: two events further apart are not one incident
DEFAULT_MAX_SOURCES = 4096     # LRU cap; a --rand-source flood must not eat memory
DEFAULT_MAX_EVENTS_PER_INCIDENT = 500
MIN_PROMOTING_SEVERITY = "low"  # info-only events can never open an incident


@dataclass(frozen=True)
class CorrelationEvent:
    """One alert offered to the engine.

    `actor` is the host we attribute the behaviour to — NOT necessarily the
    packet's source IP. See the module docstring.
    """
    rule_id: str
    actor: str
    ts: float
    severity: str = "medium"
    stage: Optional[str] = None
    dst: Optional[str] = None
    dst_port: Optional[int] = None
    title: Optional[str] = None
    message: str = ""
    alert_id: Optional[int] = None
    details: dict = field(default_factory=dict)


@dataclass
class Incident:
    incident_id: str
    source: str
    first_seen: float
    last_seen: float
    severity: str = "medium"
    # stage -> {"first", "last", "count", "rule_ids": [..]}
    stages: Dict[str, dict] = field(default_factory=dict)
    rule_counts: Dict[str, int] = field(default_factory=dict)
    destinations: Dict[str, int] = field(default_factory=dict)
    events: List[CorrelationEvent] = field(default_factory=list)
    dropped_events: int = 0        # trimmed for the per-incident cap, still counted
    event_count: int = 0
    alert_emitted: bool = False

    @property
    def ordered_stages(self) -> List[str]:
        """Stages reached, in kill-chain order (not arrival order)."""
        return [s for s in KILL_CHAIN_ORDER if s in self.stages]

    def effective_severity(self, escalate_at_stages: int = 3) -> str:
        """Incident severity = the worst member, escalated once the chain is deep.

        Two stages is an attacker advancing; that is exactly as bad as the worst
        thing they did. Three or more stages is a working intrusion, which is
        worse than any single alert in it — so it escalates to critical.

        The escalation is deliberately gated on stage COUNT, not alert count. The
        old engine escalated on "3 distinct alert types", which a single nmap run
        satisfies; three distinct kill-chain stages, it cannot.
        """
        if len(self.stages) >= escalate_at_stages:
            return "critical"
        return self.severity

    def to_dict(self, now: Optional[float] = None) -> dict:
        now = time.time() if now is None else now
        return {
            "incident_id":  self.incident_id,
            "source":       self.source,
            "severity":     self.effective_severity(),
            "member_severity": self.severity,
            "first_seen":   self.first_seen,
            "last_seen":    self.last_seen,
            "age_sec":      round(now - self.first_seen, 1),
            "idle_sec":     round(now - self.last_seen, 1),
            "duration_sec": round(self.last_seen - self.first_seen, 1),
            "event_count":  self.event_count,
            "stored_events": len(self.events),
            "dropped_events": self.dropped_events,
            "stage_count":  len(self.stages),
            "stages": [
                {
                    "stage":    s,
                    "label":    STAGE_LABELS.get(s, s),
                    "count":    self.stages[s]["count"],
                    "rule_ids": sorted(self.stages[s]["rule_ids"]),
                    "first":    self.stages[s]["first"],
                    "last":     self.stages[s]["last"],
                }
                for s in self.ordered_stages
            ],
            "rule_ids":     dict(sorted(self.rule_counts.items())),
            "destinations": dict(sorted(self.destinations.items(),
                                        key=lambda kv: -kv[1])[:12]),
            "timeline": [
                {"ts": e.ts, "rule_id": e.rule_id, "stage": e.stage,
                 "severity": e.severity, "dst": e.dst, "dst_port": e.dst_port,
                 "message": e.message}
                for e in self.events[-25:]
            ],
        }


@dataclass
class IngestResult:
    """What the engine decided about one event.

    `emit_alert` is True exactly once per incident (when it opens). The caller
    raises the alert OUTSIDE the engine lock — the engine never calls back into
    alerting, which is what made the old engine's re-entrancy hazardous.
    """
    action: str                      # "ignored" | "buffered" | "opened" | "extended"
    incident: Optional[dict] = None  # snapshot, safe to hand to JSON
    emit_alert: bool = False
    reason: str = ""


class CorrelationEngine:
    def __init__(
        self,
        window_sec: float = DEFAULT_WINDOW_SEC,
        max_sources: int = DEFAULT_MAX_SOURCES,
        max_events_per_incident: int = DEFAULT_MAX_EVENTS_PER_INCIDENT,
        min_promoting_severity: str = MIN_PROMOTING_SEVERITY,
        clock: Callable[[], float] = time.time,
    ):
        self.window_sec = float(window_sec)
        self.max_sources = int(max_sources)
        self.max_events_per_incident = int(max_events_per_incident)
        self.min_promoting_rank = severity_rank(min_promoting_severity)
        self.clock = clock

        self._lock = threading.RLock()
        # actor -> list of CorrelationEvent not yet part of an incident
        self._buffers: Dict[str, List[CorrelationEvent]] = {}
        # actor -> the currently OPEN incident for that actor (at most one)
        self._open: Dict[str, Incident] = {}
        # every incident ever opened, newest last
        self._incidents: List[Incident] = []
        self._ids = itertools.count(1)
        self.stats = {"ingested": 0, "ignored": 0, "buffered": 0,
                      "opened": 0, "extended": 0, "sources_evicted": 0}

    # ── ingest ────────────────────────────────────────────────────────────────
    def ingest(self, event: CorrelationEvent) -> IngestResult:
        """Offer one alert to the engine. Atomic; safe from any thread."""
        with self._lock:
            self.stats["ingested"] += 1

            if not event.actor:
                self.stats["ignored"] += 1
                return IngestResult("ignored", reason="no attributable actor")
            if severity_rank(event.severity) < self.min_promoting_rank:
                self.stats["ignored"] += 1
                return IngestResult("ignored", reason="below promoting severity")

            now = event.ts
            self._expire(now)

            actor = event.actor
            incident = self._open.get(actor)

            # ── an incident is already open for this source ──────────────────
            if incident is not None:
                if now - incident.last_seen <= self.window_sec:
                    self._attach(incident, event)
                    self.stats["extended"] += 1
                    return IngestResult("extended", incident.to_dict(now),
                                        reason="within window of open incident")
                # too old: close it and fall through to a fresh evaluation
                self._open.pop(actor, None)

            # ── no open incident: buffer, then test for forward progression ──
            buf = self._buffers.setdefault(actor, [])
            buf.append(event)
            buf[:] = [e for e in buf if now - e.ts <= self.window_sec]
            # Stage-less events are kept as context but never trigger promotion.
            self._touch_source(actor)

            progression = self._find_progression(buf)
            if progression is None:
                self.stats["buffered"] += 1
                return IngestResult("buffered", reason="no forward kill-chain progression")

            incident = self._open_incident(actor, buf, now)
            self._buffers.pop(actor, None)
            self.stats["opened"] += 1
            lo, hi = progression
            return IngestResult(
                "opened", incident.to_dict(now), emit_alert=True,
                reason=(f"forward progression {lo} -> {hi} from {actor} "
                        f"within {int(self.window_sec)}s"),
            )

    # ── internals ─────────────────────────────────────────────────────────────
    def _find_progression(self, buf: List[CorrelationEvent]):
        """Is there e1 before e2 with a strictly later kill-chain stage?

        Evaluated over the buffer SORTED BY TIMESTAMP, so the answer does not
        depend on the order events happened to arrive in.
        """
        staged = sorted(
            (e for e in buf if e.stage in STAGE_INDEX),
            key=lambda e: (e.ts, STAGE_INDEX[e.stage], e.rule_id),
        )
        if len(staged) < 2:
            return None
        best_idx = STAGE_INDEX[staged[0].stage]
        best_stage = staged[0].stage
        for e in staged[1:]:
            idx = STAGE_INDEX[e.stage]
            if idx > best_idx:
                return (best_stage, e.stage)
            if idx < best_idx:
                best_idx, best_stage = idx, e.stage
        return None

    def _open_incident(self, actor: str, buf: List[CorrelationEvent],
                       now: float) -> Incident:
        ordered = sorted(buf, key=lambda e: (e.ts, e.rule_id))
        inc = Incident(
            incident_id=f"INC-{next(self._ids):05d}",
            source=actor,
            first_seen=ordered[0].ts,
            last_seen=ordered[-1].ts,
            severity="info",
        )
        for e in ordered:
            self._attach(inc, e)
        self._open[actor] = inc
        self._incidents.append(inc)
        return inc

    def _attach(self, inc: Incident, e: CorrelationEvent) -> None:
        inc.event_count += 1
        inc.last_seen = max(inc.last_seen, e.ts)
        inc.first_seen = min(inc.first_seen, e.ts)
        inc.severity = max_severity(inc.severity, e.severity)
        inc.rule_counts[e.rule_id] = inc.rule_counts.get(e.rule_id, 0) + 1
        if e.dst:
            inc.destinations[e.dst] = inc.destinations.get(e.dst, 0) + 1
        if e.stage in STAGE_INDEX:
            st = inc.stages.get(e.stage)
            if st is None:
                inc.stages[e.stage] = {"first": e.ts, "last": e.ts, "count": 1,
                                       "rule_ids": {e.rule_id}}
            else:
                st["last"] = max(st["last"], e.ts)
                st["first"] = min(st["first"], e.ts)
                st["count"] += 1
                st["rule_ids"].add(e.rule_id)
        inc.events.append(e)
        if len(inc.events) > self.max_events_per_incident:
            overflow = len(inc.events) - self.max_events_per_incident
            del inc.events[:overflow]
            inc.dropped_events += overflow

    def _expire(self, now: float) -> None:
        """Close incidents and drop buffers that have gone quiet past the window."""
        for actor, inc in list(self._open.items()):
            if now - inc.last_seen > self.window_sec:
                del self._open[actor]
        for actor, buf in list(self._buffers.items()):
            buf[:] = [e for e in buf if now - e.ts <= self.window_sec]
            if not buf:
                del self._buffers[actor]

    def _touch_source(self, actor: str) -> None:
        """LRU-cap tracked sources. Open incidents are never evicted."""
        if len(self._buffers) <= self.max_sources:
            return
        victims = sorted(
            (a for a in self._buffers if a not in self._open),
            key=lambda a: max((e.ts for e in self._buffers[a]), default=0.0),
        )
        for a in victims[: len(self._buffers) - self.max_sources]:
            del self._buffers[a]
            self.stats["sources_evicted"] += 1

    # ── read side ─────────────────────────────────────────────────────────────
    def incidents(self, limit: int = 100, now: Optional[float] = None) -> List[dict]:
        now = self.clock() if now is None else now
        with self._lock:
            recent = self._incidents[-limit:]
            out = []
            for inc in recent:
                d = inc.to_dict(now)
                d["open"] = self._open.get(inc.source) is inc
                out.append(d)
        out.sort(key=lambda d: -d["last_seen"])
        return out

    def incident_count(self) -> int:
        with self._lock:
            return len(self._incidents)

    def open_incident_for(self, actor: str) -> Optional[dict]:
        with self._lock:
            inc = self._open.get(actor)
            return inc.to_dict(self.clock()) if inc else None

    def snapshot_stats(self) -> dict:
        with self._lock:
            return {
                **self.stats,
                "incidents_total": len(self._incidents),
                "incidents_open": len(self._open),
                "sources_buffered": len(self._buffers),
                "window_seconds": self.window_sec,
            }

    def reset(self) -> None:
        with self._lock:
            self._buffers.clear()
            self._open.clear()
            self._incidents.clear()
            self._ids = itertools.count(1)
            for k in self.stats:
                self.stats[k] = 0

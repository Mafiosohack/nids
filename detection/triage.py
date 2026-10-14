"""Analyst triage — how an alert leaves the review queue.

Before this, every alert stayed "new" forever. The sensor cannot tell a false
positive from a true one on its own: an admin's SSH session and an attacker's
shell are the same shape on the wire. So the decision is the analyst's, and
this module is where that decision is recorded and, when asked, remembered.

## States

    new             in the review queue. Every alert starts here.
    confirmed       reviewed: real activity, being worked. Leaves the queue.
    false_positive  reviewed: not malicious. Leaves the queue.
    resolved        handled and closed. Leaves the queue, but RE-ENTERS it if
                    the same alert recurs (see AlertStore.add) — a resolved
                    attack that starts again is news.

## Suppressions — a false positive, remembered

Marking an alert false positive is about that alert. A suppression is about
the future: "this rule, from this source, is known noise". It is deliberately
narrow — one (rule_id, source) pair, never a whole rule or a whole host — so a
suppressed admin box running a different rule (exfil, a new listener) still
alerts and still correlates.

A suppressed alert is NOT dropped. It is stored, counted and exported like any
other, arriving already marked false_positive with the suppression named, so the
forensic record is intact and the suppression is auditable. What it does not do
is enter the review queue or the correlation engine — keeping known noise out
of incidents is the reason to suppress at all.
"""

import threading
import time
from typing import Dict, List, Optional, Tuple

TRIAGE_NEW = "new"
TRIAGE_CONFIRMED = "confirmed"
TRIAGE_FALSE_POSITIVE = "false_positive"
TRIAGE_RESOLVED = "resolved"

TRIAGE_STATUSES = (TRIAGE_NEW, TRIAGE_CONFIRMED, TRIAGE_FALSE_POSITIVE,
                   TRIAGE_RESOLVED)

# Verdicts an analyst can set. "new" is included so a mistaken verdict can be
# undone by putting the alert back in the queue.
SETTABLE_STATUSES = frozenset(TRIAGE_STATUSES)


class SuppressionList:
    """Thread-safe set of (rule_id, source) pairs known to be noise."""

    def __init__(self, clock=time.time):
        self.clock = clock
        self._lock = threading.Lock()
        self._by_key: Dict[Tuple[str, str], dict] = {}
        self._last_id = 0

    @staticmethod
    def key(rule_id: Optional[str], src: Optional[str]) -> Tuple[str, str]:
        return (rule_id or "", src or "")

    def add(self, rule_id: str, src: str, created_by: str, note: str = "",
            suppression_id: Optional[int] = None,
            created_at: Optional[float] = None) -> dict:
        """Add or return the existing suppression for this pair (idempotent)."""
        k = self.key(rule_id, src)
        with self._lock:
            if k in self._by_key:
                return dict(self._by_key[k])
            # An id restored from disk keeps its number; new ones count on
            # from the highest seen, so ids never collide across restarts.
            sid = suppression_id if suppression_id is not None else self._last_id + 1
            self._last_id = max(self._last_id, sid)
            rec = {
                "id":         sid,
                "rule_id":    rule_id,
                "src":        src,
                "created_by": created_by,
                "created_at": created_at if created_at is not None else self.clock(),
                "note":       note or "",
                "hits":       0,
            }
            self._by_key[k] = rec
            return dict(rec)

    def remove(self, suppression_id: int) -> Optional[dict]:
        with self._lock:
            for k, rec in self._by_key.items():
                if rec["id"] == suppression_id:
                    del self._by_key[k]
                    return dict(rec)
        return None

    def match(self, rule_id: Optional[str], src: Optional[str]) -> Optional[dict]:
        """The suppression covering this alert, counting the hit, or None."""
        if not rule_id or not src:
            return None
        with self._lock:
            rec = self._by_key.get(self.key(rule_id, src))
            if rec is None:
                return None
            rec["hits"] += 1
            return dict(rec)

    def list(self) -> List[dict]:
        with self._lock:
            return sorted((dict(r) for r in self._by_key.values()),
                          key=lambda r: r["id"])

    def __len__(self) -> int:
        with self._lock:
            return len(self._by_key)

    def clear(self) -> None:
        with self._lock:
            self._by_key.clear()
            self._last_id = 0

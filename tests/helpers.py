"""Shared test fixtures: a fake clock and a rule-aware event builder."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from detection.correlation import CorrelationEvent      # noqa: E402
from detection.rules import RULES                        # noqa: E402


class FakeClock:
    """Deterministic clock. Tests advance time explicitly rather than sleeping."""

    def __init__(self, start: float = 1_000_000.0):
        self.now = float(start)

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> float:
        self.now += float(seconds)
        return self.now


def event(rule_id: str, actor: str, ts: float, dst: str = "192.168.56.102",
          dst_port: int = None, severity: str = None,
          stage: str = "__from_rule__", **details) -> CorrelationEvent:
    """Build a CorrelationEvent, taking stage/severity from the rule catalogue.

    `stage` and `severity` can be overridden to construct cases the catalogue
    does not contain (e.g. an INFO-severity benign event).
    """
    rule = RULES.get(rule_id)
    if stage == "__from_rule__":
        stage = rule.stage if rule else None
    if severity is None:
        severity = rule.severity if rule else "medium"
    return CorrelationEvent(
        rule_id=rule_id, actor=actor, ts=ts, severity=severity, stage=stage,
        dst=dst, dst_port=dst_port, message=f"{rule_id} from {actor}",
        details=details,
    )


# Lab topology (see memory: Kali attacker / Metasploitable target / Ubuntu sensor)
KALI = "192.168.56.101"
METASPLOITABLE = "192.168.56.102"
UBUNTU = "192.168.56.103"
ATTACKER_EXTERNAL = "203.0.113.50"

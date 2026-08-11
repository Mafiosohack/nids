"""The one shape every detector returns.

A detector never raises an alert itself — it returns a `Finding` and lets the
wiring layer (`main.py`) decide what to do with it. That keeps every detector
testable without a running server, and keeps alert formatting in one place.
"""

from dataclasses import dataclass, field
from typing import Optional

from .rules import RULES, validate_alert_details


@dataclass
class Finding:
    rule_id: str
    src: Optional[str]              # the actor, per the rule's src_semantics
    dst: Optional[str] = None
    protocol: str = "TCP"
    message: str = ""
    details: dict = field(default_factory=dict)
    ts: Optional[float] = None
    severity_override: Optional[str] = None

    # For alerts arriving from the host / cloud log sensors, which have no entry
    # in the packet-rule catalogue but still carry a kill-chain stage. Leaving
    # these unset means the catalogue is the authority, which is the normal case.
    title_override: Optional[str] = None
    stage_override: Optional[str] = None
    tactic_override: Optional[str] = None
    technique_override: Optional[str] = None
    src_semantics_override: Optional[str] = None

    @property
    def rule(self):
        return RULES.get(self.rule_id)

    @property
    def severity(self) -> str:
        if self.severity_override:
            return self.severity_override
        r = self.rule
        return r.severity if r else "medium"

    @property
    def title(self) -> str:
        if self.title_override:
            return self.title_override
        r = self.rule
        return r.title if r else self.rule_id

    @property
    def stage(self) -> Optional[str]:
        if self.stage_override:
            return self.stage_override
        r = self.rule
        return r.stage if r else None

    @property
    def tactic(self) -> Optional[str]:
        if self.tactic_override:
            return self.tactic_override
        r = self.rule
        return r.tactic if r else None

    @property
    def technique(self) -> Optional[str]:
        if self.technique_override:
            return self.technique_override
        r = self.rule
        return r.technique if r else None

    @property
    def src_semantics(self) -> str:
        if self.src_semantics_override:
            return self.src_semantics_override
        r = self.rule
        return r.src_semantics if r else "actor"

    def missing_discriminators(self):
        """Discriminator keys the catalogue requires that this finding lacks.

        Non-empty means the alert would go out as a generic bucket label — the
        exact regression the rule-ID work was meant to prevent. The efficacy
        harness fails on any non-empty result.
        """
        return validate_alert_details(self.rule_id, self.details)

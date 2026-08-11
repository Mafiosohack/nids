"""One place where the detectors are wired together.

`main.py` owns packet capture and HTTP; this owns "given a packet, which rules
fire, and what happens to the findings". The tests and the efficacy harness
drive this same object with synthetic packets, so what they exercise is what
runs in production rather than a parallel reimplementation.

Order of operations per TCP packet:

  1. fragment reassembly happens in the caller (needs scapy) — by the time a
     packet reaches here it is whole, so `-f` scans look like ordinary ones
  2. flag signature       -> TCP_{NULL,FIN,XMAS,SYNFIN}_SCAN
  3. behavioural scan     -> TCP_{SYN,CONNECT}_SCAN
  4. brute force          -> BRUTEFORCE_{STANDARD,LOW}_RATE
  5. Tier 1 payload sigs  -> EXPLOIT_*
  6. Tier 2 behaviour     -> BEHAVIOR_*
  7. every finding is published: deduped into the AlertStore, then offered to
     the correlation engine, which may open an incident

Tier 2 findings are published before Tier 1 findings from the same packet, so
the exploit-agnostic evidence leads.
"""

import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Dict, Iterable, List, Optional, Set, Tuple

from .correlation import CorrelationEngine, CorrelationEvent
from .exploit_tier1 import Tier1Engine
from .exploit_tier2 import (BeaconDetector, ListeningPortMonitor,
                            OutboundAnomalyDetector)
from .findings import Finding
from .interactive_shell import InteractiveShellDetector
from .retention import AlertStore
from .rules import RULES, TIER_CORRELATION, TIER_EXPLOIT_2, severity_rank
from .scan_rules import BruteForceDetector, PortScanDetector, signature_finding

FIN, SYN, RST, PSH, ACK, URG = 0x01, 0x02, 0x04, 0x08, 0x10, 0x20


@dataclass
class PipelineConfig:
    # Correlation
    correlation_window_sec: float = 600.0
    # Port scan
    scan_port_threshold: int = 6
    scan_window_sec: float = 5.0
    # Brute force
    bruteforce_min_attempts: int = 8
    bruteforce_window_sec: float = 900.0
    bruteforce_rate_split: float = 0.2
    # Tier 2
    monitored_servers: Set[str] = field(default_factory=set)
    outbound_baseline: Dict[str, Set[Tuple[str, int]]] = field(default_factory=dict)
    listening_baseline: Dict[str, Set[int]] = field(default_factory=dict)
    outbound_learn_sec: float = 300.0
    beacon_min_hits: int = 6
    beacon_max_cv: float = 0.25
    # Interactive-shell shape detection. Thresholds are conservative by default:
    # 40 packets over 20s of a turn-taking, keystroke-shaped conversation.
    shell_min_packets: int = 40
    shell_min_duration_sec: float = 20.0
    shell_expected_ports: Set[int] = field(default_factory=lambda: {22})
    # Standing (client, server) exemptions for hosts with no host_log_sensor.py.
    shell_authorized_pairs: Set[Tuple[str, str]] = field(default_factory=set)
    # Retention
    alert_capacity: int = 5000
    dedup_window_sec: float = 300.0
    # Hosts permitted to run reconnaissance (vulnerability scanners, monitoring).
    # Their recon findings are recorded at INFO rather than suppressed outright,
    # so the activity is still auditable. This is an allowlist, NOT a weakened
    # rule: an unlisted host running the same scan still alerts at full severity.
    authorized_scanners: Set[str] = field(default_factory=set)


class DetectionPipeline:
    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        clock: Callable[[], float] = time.time,
        on_alert: Optional[Callable[[dict], None]] = None,
    ):
        self.cfg = config or PipelineConfig()
        self.clock = clock
        self.on_alert = on_alert
        self._next_id = 0

        self.correlator = CorrelationEngine(
            window_sec=self.cfg.correlation_window_sec, clock=clock)
        self.alerts = AlertStore(capacity=self.cfg.alert_capacity,
                                 dedup_window_sec=self.cfg.dedup_window_sec,
                                 clock=clock)
        self.port_scan = PortScanDetector(
            port_threshold=self.cfg.scan_port_threshold,
            window_sec=self.cfg.scan_window_sec, clock=clock)
        self.brute_force = BruteForceDetector(
            min_attempts=self.cfg.bruteforce_min_attempts,
            window_sec=self.cfg.bruteforce_window_sec,
            rate_split=self.cfg.bruteforce_rate_split, clock=clock)
        self.tier1 = Tier1Engine(clock=clock)
        self.outbound = OutboundAnomalyDetector(
            monitored_hosts=self.cfg.monitored_servers,
            baseline=self.cfg.outbound_baseline,
            learn_sec=self.cfg.outbound_learn_sec, clock=clock)
        self.listening = ListeningPortMonitor(
            baseline=self.cfg.listening_baseline, clock=clock)
        self.beacon = BeaconDetector(min_hits=self.cfg.beacon_min_hits,
                                     max_cv=self.cfg.beacon_max_cv, clock=clock)
        self.shell = InteractiveShellDetector(
            min_packets=self.cfg.shell_min_packets,
            min_duration_sec=self.cfg.shell_min_duration_sec,
            expected_shell_ports=self.cfg.shell_expected_ports,
            authorized_pairs=self.cfg.shell_authorized_pairs, clock=clock)

        self.published: List[dict] = []     # every alert this pipeline emitted
        self.incidents_opened: List[dict] = []

    # ── packet entry point ────────────────────────────────────────────────────
    def on_tcp_packet(
        self,
        src: str, dst: str, sport: int, dport: int, flags: int,
        payload: bytes = b"",
        ts: Optional[float] = None,
        fragment_count: int = 0,
        is_outbound: bool = False,
        payload_len: Optional[int] = None,
    ) -> List[dict]:
        """Run every TCP rule. Returns the alerts published for this packet.

        `payload` may be truncated for signature matching; `payload_len` carries
        the TRUE length when it is, because the shell-shape rule measures packet
        sizes and a truncated length would corrupt the measurement.
        """
        now = self.clock() if ts is None else ts
        flags = int(flags)
        syn, ack, rst = bool(flags & SYN), bool(flags & ACK), bool(flags & RST)
        payload_len = len(payload) if payload_len is None else int(payload_len)
        tier1_hits: List[Finding] = []
        tier2_hits: List[Finding] = []
        other: List[Finding] = []

        # 2. single-packet flag signature
        extra = {"fragment_count": fragment_count} if fragment_count else None
        sig = signature_finding(src, dst, dport, flags, ts=now, extra=extra)
        if sig is not None:
            other.append(sig)

        if syn and not ack:
            # 3. behavioural scan
            f = self.port_scan.observe_syn(src, dst, dport, ts=now)
            if f is not None:
                other.append(f)
            # 4. brute force
            f = self.brute_force.observe_attempt(src, dst, dport, ts=now)
            if f is not None:
                other.append(f)
            # 5b. vsftpd second stage
            f = self.tier1.observe_syn(src, dst, dport, ts=now)
            if f is not None:
                tier1_hits.append(f)
            # 6. Tier 2 — outbound anomaly + beaconing
            f = self.outbound.observe_outbound_syn(src, dst, dport, ts=now)
            if f is not None:
                tier2_hits.append(f)
            if is_outbound:
                f = self.beacon.observe_connection(src, dst, dport, ts=now)
                if f is not None:
                    tier2_hits.append(f)
            # Record who initiated, so the shell detector can tell a reverse
            # shell (server types) from a bind shell (client types).
            self.shell.open_flow(src, dst, sport, dport, ts=now)
        elif syn and ack:
            self.port_scan.observe_synack(src, dst, sport, ts=now)
            # A SYN-ACK is proof the port is accepting: bind-shell check.
            f = self.listening.observe_listening(src, sport, ts=now)
            if f is not None:
                tier2_hits.append(f)
        elif rst:
            self.port_scan.observe_client_rst(src, dst, dport, ts=now)
        elif ack:
            self.port_scan.observe_client_ack(src, dst, dport, ts=now)

        # Interactive-shell shape. Fed on every data-bearing packet regardless of
        # flags, and only the LENGTH is used — never the content — so it works
        # identically on encrypted sessions.
        if payload_len > 0:
            f = self.shell.observe_data(src, dst, sport, dport, payload_len, ts=now)
            if f is not None:
                tier2_hits.append(f)
        if rst or (flags & FIN):
            self.shell.close_flow(src, dst, sport, dport)

        # 5. Tier 1 payload signatures
        if payload:
            for f in self.tier1.inspect_payload(src, dst, sport, dport, payload,
                                                ts=now):
                tier1_hits.append(f)
                if f.stage == "initial_access":
                    # Gives the bind-shell rule its "after exploit traffic" context.
                    self.listening.note_exploit_activity(dst, ts=now)

        # Tier 2 first: it generalises, so it leads the analyst's read.
        return [a for f in (tier2_hits + tier1_hits + other)
                for a in [self.publish(f)] if a is not None]

    # ── publication ───────────────────────────────────────────────────────────
    def publish(self, finding: Finding) -> Optional[dict]:
        """Turn a Finding into an alert: dedup, store, correlate."""
        rule = RULES.get(finding.rule_id)
        now = finding.ts if finding.ts is not None else self.clock()
        severity = finding.severity
        details = dict(finding.details)

        # Authorized scanners: recorded, not suppressed, and downgraded to INFO.
        if (rule is not None and rule.stage == "reconnaissance"
                and finding.src in self.cfg.authorized_scanners):
            severity = "info"
            details["suppressed_by"] = "authorized_scanner allowlist"
            details["original_severity"] = finding.severity

        alert = {
            # id is assigned below, only if the alert survives deduplication —
            # otherwise every collapsed repeat would burn an ID and leave gaps
            # in the sequence that look like lost alerts.
            "id":        None,
            "rule_id":   finding.rule_id,
            "type":      finding.title,
            "severity":  severity,
            "src":       finding.src,
            "dst":       finding.dst,
            "protocol":  finding.protocol,
            "message":   finding.message,
            "timestamp": datetime.fromtimestamp(now).strftime("%Y-%m-%d %H:%M:%S"),
            "details":   details,
            "tier":      rule.tier if rule else None,
            "generalizes": rule.generalizes if rule else None,
            "kill_chain_stage": finding.stage,
            "mitre_tactic":     finding.tactic,
            "mitre_technique":  finding.technique,
        }

        result = self.alerts.add(alert, ts=now)
        if not result.stored:
            # Deduped: the record's count went up, no new alert to publish.
            return None

        alert["id"] = self.next_alert_id()
        if result.record is not None:
            result.record["id"] = alert["id"]
        self.published.append(alert)
        if self.on_alert is not None:
            self.on_alert(alert)

        self._correlate(alert, finding, rule, now)
        return alert

    def next_alert_id(self) -> int:
        """Overridable so main.py can keep IDs unique against its persisted store."""
        self._next_id += 1
        return self._next_id

    def _correlate(self, alert, finding, rule, now) -> None:
        # Correlation OUTPUT must never become correlation INPUT. Without this,
        # the CORRELATED_INCIDENT alert announcing an incident gets ingested back
        # into that same incident, inflating its event_count by one and listing
        # itself among its own member rules.
        if rule is not None and rule.tier == TIER_CORRELATION:
            return

        # `src_semantics` decides who this event is attributed to. Events whose
        # source is forged (ARP) are never grouped under whoever they claimed
        # to be — the old engine's silent misattribution.
        if finding.src_semantics == "unknown":
            return
        actor = finding.src
        if not actor:
            return

        res = self.correlator.ingest(CorrelationEvent(
            rule_id=finding.rule_id, actor=actor, ts=now,
            severity=alert["severity"], stage=finding.stage,
            dst=finding.dst, dst_port=finding.details.get("dst_port"),
            title=alert["type"], message=finding.message,
            alert_id=alert["id"], details=finding.details,
        ))
        if not res.emit_alert:
            return

        inc = res.incident
        self.incidents_opened.append(inc)
        stages = [s["label"] for s in inc["stages"]]
        self.publish(Finding(
            rule_id="CORRELATED_INCIDENT", src=inc["source"], dst=None,
            protocol="Multiple", ts=now, severity_override=inc["severity"],
            message=(f"Incident {inc['incident_id']}: {inc['source']} advanced "
                     f"through [{' -> '.join(stages)}] across "
                     f"{inc['event_count']} alerts in {inc['duration_sec']:.0f}s"),
            details={
                "incident_id":    inc["incident_id"],
                "stages":         [s["stage"] for s in inc["stages"]],
                "stage_labels":   stages,
                "rule_ids":       inc["rule_ids"],
                "event_count":    inc["event_count"],
                "window_seconds": self.cfg.correlation_window_sec,
                "destinations":   list(inc["destinations"]),
                "promotion_reason": res.reason,
            },
        ))

    # ── introspection for tests / harness ─────────────────────────────────────
    def alerts_above(self, severity: str = "info") -> List[dict]:
        """Published alerts strictly above `severity`. Benign scenarios must be
        empty at 'info'."""
        floor = severity_rank(severity)
        return [a for a in self.published if severity_rank(a["severity"]) > floor]

    def reset(self) -> None:
        self.published.clear()
        self.incidents_opened.clear()
        self.correlator.reset()
        self.alerts.clear()
        self.port_scan.reset()
        self.brute_force.reset()
        self.tier1.reset()
        self.outbound.reset()
        self.listening.reset()
        self.beacon.reset()
        self._next_id = 0

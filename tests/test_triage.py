"""Triage, suppression, the single kill-chain engine, and the response playbook.

The queue is only useful if alerts can leave it, and only trustworthy if
leaving it never destroys evidence. These tests pin both halves down.
"""

import unittest

from helpers import ATTACKER_EXTERNAL, KALI, METASPLOITABLE, FakeClock, event

from detection.correlation import CorrelationEngine, is_active_intrusion
from detection.findings import Finding
from detection.pipeline import DetectionPipeline, PipelineConfig
from detection.response import (MITIGATION_NAMES, TECHNIQUES, playbook,
                                unmapped_techniques)
from detection.retention import AlertStore
from detection.rules import RULES

ADMIN_BOX = "192.168.56.10"


def finding(rule_id, src, dst=METASPLOITABLE, ts=None, **details):
    return Finding(rule_id=rule_id, src=src, dst=dst, ts=ts,
                   message=f"{rule_id} test", details=details)


class TestActiveIntrusionLatch(unittest.TestCase):
    """The old per-host tracker is gone; the correlation engine says it now."""

    def setUp(self):
        self.clock = FakeClock()
        self.engine = CorrelationEngine(window_sec=600, clock=self.clock)

    def test_condition(self):
        self.assertTrue(is_active_intrusion({"initial_access", "command_and_control"}))
        self.assertFalse(is_active_intrusion({"initial_access"}))
        self.assertFalse(is_active_intrusion({"reconnaissance", "command_and_control"}),
                         "post-access activity without a recorded way in is not a breach")

    def test_confirmed_exactly_once(self):
        t0 = self.clock.now
        r1 = self.engine.ingest(event("TCP_SYN_SCAN", KALI, t0))
        r2 = self.engine.ingest(event("EXPLOIT_VSFTPD_BACKDOOR", KALI, t0 + 10))
        r3 = self.engine.ingest(event("BEHAVIOR_C2_BEACON", KALI, t0 + 20,
                                      dst=ATTACKER_EXTERNAL))
        r4 = self.engine.ingest(event("EXFIL_BULK_OUTBOUND", KALI, t0 + 30,
                                      dst=ATTACKER_EXTERNAL))
        self.assertEqual([r.intrusion_confirmed for r in (r1, r2, r3, r4)],
                         [False, False, True, False])
        self.assertTrue(self.engine.incidents()[0]["active_intrusion"])
        self.assertEqual(self.engine.snapshot_stats()["active_intrusions"], 1)

    def test_windowed_unlike_the_old_tracker(self):
        """A login, then an unrelated alert an hour later, is not a breach in
        progress. The old tracker had no window and called it one."""
        t0 = self.clock.now
        self.engine.ingest(event("TCP_SYN_SCAN", KALI, t0))
        self.engine.ingest(event("EXPLOIT_SAMBA_USERMAP", KALI, t0 + 5))
        r = self.engine.ingest(event("BEHAVIOR_C2_BEACON", KALI, t0 + 3600,
                                     dst=ATTACKER_EXTERNAL))
        self.assertFalse(r.intrusion_confirmed)


class TestPipelineEmitsOneVerdict(unittest.TestCase):
    def test_kill_chain_produces_incident_then_intrusion(self):
        clock = FakeClock()
        pipe = DetectionPipeline(PipelineConfig(), clock=clock)
        t0 = clock.now
        pipe.publish(finding("TCP_SYN_SCAN", KALI, ts=t0, dst_port=21))
        pipe.publish(finding("EXPLOIT_VSFTPD_BACKDOOR", KALI, ts=t0 + 5, dst_port=21))
        pipe.publish(finding("EXPLOIT_VSFTPD_BACKDOOR_SHELL", KALI, ts=t0 + 8, dst_port=6200))
        rules = [a["rule_id"] for a in pipe.published]
        self.assertEqual(rules.count("CORRELATED_INCIDENT"), 1)
        self.assertEqual(rules.count("ACTIVE_INTRUSION"), 1)
        intr = next(a for a in pipe.published if a["rule_id"] == "ACTIVE_INTRUSION")
        self.assertEqual(intr["severity"], "critical")
        self.assertEqual(intr["src"], KALI)
        self.assertEqual(intr["details"]["incident_id"], "INC-00001")


class TestSuppression(unittest.TestCase):
    def setUp(self):
        self.clock = FakeClock()
        self.pipe = DetectionPipeline(PipelineConfig(), clock=self.clock)

    def test_suppressed_alert_is_stored_but_born_triaged(self):
        self.pipe.suppressions.add("BRUTEFORCE_STANDARD_RATE", ADMIN_BOX, "admin", "spray test")
        a = self.pipe.publish(finding("BRUTEFORCE_STANDARD_RATE", ADMIN_BOX, dst_port=22))
        self.assertIsNotNone(a, "a suppressed alert must still be recorded")
        self.assertEqual(a["triage_status"], "false_positive")
        self.assertIn("suppression #1", a["triage_by"])
        self.assertEqual(len(self.pipe.alerts.list(triage="new")), 0)
        self.assertEqual(len(self.pipe.alerts.list(triage="false_positive")), 1)

    def test_suppression_is_narrow(self):
        """Same source, different rule — still alerts. Same rule, different
        source — still alerts."""
        self.pipe.suppressions.add("BRUTEFORCE_STANDARD_RATE", ADMIN_BOX, "admin")
        other_rule = self.pipe.publish(finding("EXFIL_BULK_OUTBOUND", ADMIN_BOX,
                                               dst=ATTACKER_EXTERNAL))
        other_src = self.pipe.publish(finding("BRUTEFORCE_STANDARD_RATE", KALI, dst_port=22))
        self.assertEqual(other_rule["triage_status"], "new")
        self.assertEqual(other_src["triage_status"], "new")

    def test_suppressed_alerts_stay_out_of_correlation(self):
        t0 = self.clock.now
        self.pipe.suppressions.add("TCP_SYN_SCAN", KALI, "admin")
        self.pipe.suppressions.add("EXPLOIT_VSFTPD_BACKDOOR", KALI, "admin")
        self.pipe.publish(finding("TCP_SYN_SCAN", KALI, ts=t0, dst_port=21))
        self.pipe.publish(finding("EXPLOIT_VSFTPD_BACKDOOR", KALI, ts=t0 + 5, dst_port=21))
        self.assertEqual(self.pipe.correlator.incident_count(), 0)

    def test_correlation_alerts_cannot_be_suppressed(self):
        self.pipe.suppressions.add("ACTIVE_INTRUSION", KALI, "admin")
        t0 = self.clock.now
        self.pipe.publish(finding("TCP_SYN_SCAN", KALI, ts=t0, dst_port=21))
        self.pipe.publish(finding("EXPLOIT_VSFTPD_BACKDOOR", KALI, ts=t0 + 5, dst_port=21))
        self.pipe.publish(finding("EXPLOIT_VSFTPD_BACKDOOR_SHELL", KALI, ts=t0 + 8, dst_port=6200))
        intr = next(a for a in self.pipe.published if a["rule_id"] == "ACTIVE_INTRUSION")
        self.assertEqual(intr["triage_status"], "new")

    def test_ids_survive_restore(self):
        s = self.pipe.suppressions
        s.add("A", "1.1.1.1", "admin", suppression_id=7)
        self.assertEqual(s.add("B", "2.2.2.2", "admin")["id"], 8)
        self.assertEqual(s.add("A", "1.1.1.1", "x")["id"], 7, "re-adding is idempotent")


class TestTriageInTheStore(unittest.TestCase):
    def setUp(self):
        self.clock = FakeClock()
        self.store = AlertStore(dedup_window_sec=300, clock=self.clock)

    def _add(self, aid, **kw):
        a = {"id": aid, "rule_id": "TCP_SYN_SCAN", "type": "scan", "severity": "high",
             "src": KALI, "dst": METASPLOITABLE, "timestamp": "t", "message": "m",
             "details": {"dst_port": 22}}
        a.update(kw)
        r = self.store.add(a)
        if r.record is not None and r.stored:
            r.record["id"] = aid
        return r

    def test_new_by_default_and_verdict_moves_it(self):
        self._add(1)
        self.assertEqual(self.store.list(triage="new")[0]["id"], 1)
        updated = self.store.set_triage([1, 999], "confirmed", "ana", "real")
        self.assertEqual([r["id"] for r in updated], [1], "unknown ids are skipped")
        self.assertEqual(self.store.list(triage="new"), [])
        self.assertEqual(self.store.summary()["by_triage"], {"confirmed": 1})

    def test_resolved_alert_that_recurs_returns_to_the_queue(self):
        self._add(1)
        self.store.set_triage([1], "resolved", "ana")
        self.clock.advance(10)
        self._add(None)                          # same dedup key, folds in
        rec = self.store.list()[0]
        self.assertEqual(rec["triage_status"], "new")
        self.assertIn("recurred", rec["triage_note"])

    def test_false_positive_that_recurs_stays_judged(self):
        self._add(1)
        self.store.set_triage([1], "false_positive", "ana")
        self.clock.advance(10)
        self._add(None)
        self.assertEqual(self.store.list()[0]["triage_status"], "false_positive")

    def test_bad_status_rejected(self):
        self._add(1)
        with self.assertRaises(ValueError):
            self.store.set_triage([1], "ignored", "ana")

    def test_summary_counts_whole_store(self):
        for i in range(1, 251):                  # more than the dashboard's 200
            self._add(i, details={"dst_port": i})
        self._add(1000, rule_id="EXPLOIT_SAMBA_USERMAP", severity="critical",
                  details={"dst_port": 139})
        s = self.store.summary()
        self.assertEqual(s["by_severity"]["critical"], {"total": 1, "open": 1})
        self.assertEqual(s["by_severity"]["high"]["open"], 250)
        self.assertEqual(s["open_sources"], [{"ip": KALI, "open": 251}])

    def test_verdict_history(self):
        self._add(1); self._add(2, details={"dst_port": 23})
        self.store.set_triage([1, 2], "false_positive", "ana")
        self.assertEqual(self.store.verdicts_for("TCP_SYN_SCAN", KALI), {"false_positive": 2})


class TestResponsePlaybook(unittest.TestCase):
    def test_every_technique_in_use_is_mapped(self):
        self.assertEqual(unmapped_techniques(), [])

    def test_every_mitigation_id_has_a_name(self):
        for tid, t in TECHNIQUES.items():
            for mid, action in t["mitigations"]:
                self.assertIn(mid, MITIGATION_NAMES, f"{tid} -> {mid}")
                self.assertTrue(action)

    def test_every_rule_has_a_response(self):
        pb = playbook()["rules"]
        for rule_id, rule in RULES.items():
            self.assertIn(rule_id, pb)
            if rule.src_semantics == "unknown":
                continue      # spoofable/distributed source: naming it would mislead
            self.assertIn("{src}", pb[rule_id]["response"]["contain"] + pb[rule_id]["response"]["verify"],
                          f"{rule_id}: advice should name the address it is about")

    def test_victim_attributed_rules_never_say_block_the_source(self):
        """The bug this replaced: 'Block Source IP' on every alert."""
        pb = playbook()["rules"]
        for rule_id in ("BEHAVIOR_UNEXPECTED_OUTBOUND", "BEHAVIOR_NEW_LISTENING_PORT",
                        "COLLECTION_STAGING", "MITM_ARP_SPOOF"):
            contain = pb[rule_id]["response"]["contain"]
            self.assertFalse(contain.startswith("Block {src}"), rule_id)

    def test_sub_technique_urls(self):
        tech = playbook()["rules"]["MITM_ARP_SPOOF"]["technique"]
        self.assertEqual(tech["url"], "https://attack.mitre.org/techniques/T1557/002/")


if __name__ == "__main__":
    unittest.main()

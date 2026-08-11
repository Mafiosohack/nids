"""Correlation engine — grouping is specified here, not inferred from behaviour.

Every test feeds a synthetic sequence with explicit timestamps and asserts the
exact grouping out. The negative tests are the important ones: they are what
stop the engine drifting back into "fires after a few attacks occur".
"""

import random
import threading
import unittest

from helpers import (ATTACKER_EXTERNAL, KALI, METASPLOITABLE, UBUNTU,
                     FakeClock, event)

from detection.correlation import CorrelationEngine


class TestIncidentPromotion(unittest.TestCase):
    """Positive path: what SHOULD become one incident."""

    def setUp(self):
        self.clock = FakeClock()
        self.engine = CorrelationEngine(window_sec=600, clock=self.clock)

    def test_scan_then_exploit_then_beacon_is_ONE_incident(self):
        """The headline requirement: three stages, one source, one incident."""
        t0 = self.clock.now
        r1 = self.engine.ingest(event("TCP_SYN_SCAN", KALI, t0,
                                      dst=METASPLOITABLE, dst_port=22))
        r2 = self.engine.ingest(event("EXPLOIT_VSFTPD_BACKDOOR", KALI, t0 + 45,
                                      dst=METASPLOITABLE, dst_port=21))
        r3 = self.engine.ingest(event("BEHAVIOR_C2_BEACON", KALI, t0 + 120,
                                      dst=ATTACKER_EXTERNAL, dst_port=4444))

        self.assertEqual(r1.action, "buffered")   # recon alone is not an incident
        self.assertEqual(r2.action, "opened")     # recon -> initial_access advances
        self.assertTrue(r2.emit_alert)
        self.assertEqual(r3.action, "extended")   # NOT a second incident
        self.assertFalse(r3.emit_alert)

        self.assertEqual(self.engine.incident_count(), 1,
                         "three alerts must produce exactly one incident record")
        inc = self.engine.incidents()[0]
        self.assertEqual(inc["source"], KALI)
        self.assertEqual(inc["event_count"], 3)
        self.assertEqual([s["stage"] for s in inc["stages"]],
                         ["reconnaissance", "initial_access", "command_and_control"])
        self.assertEqual(sorted(inc["rule_ids"]),
                         ["BEHAVIOR_C2_BEACON", "EXPLOIT_VSFTPD_BACKDOOR",
                          "TCP_SYN_SCAN"])
        self.assertEqual(inc["severity"], "critical")   # max of members
        self.assertEqual(inc["duration_sec"], 120.0)

    def test_incident_extends_rather_than_duplicating(self):
        """20 further alerts must not create 20 incidents."""
        t0 = self.clock.now
        self.engine.ingest(event("TCP_SYN_SCAN", KALI, t0))
        self.engine.ingest(event("EXPLOIT_SAMBA_USERMAP", KALI, t0 + 10))
        emitted = 0
        for i in range(20):
            r = self.engine.ingest(
                event("BEHAVIOR_UNEXPECTED_OUTBOUND", KALI, t0 + 20 + i * 5))
            emitted += int(r.emit_alert)
        self.assertEqual(emitted, 0, "only the opening event raises an alert")
        self.assertEqual(self.engine.incident_count(), 1)
        self.assertEqual(self.engine.incidents()[0]["event_count"], 22)

    def test_grouping_key_is_source_ip_only(self):
        """One attacker hitting three targets = ONE incident, not three."""
        t0 = self.clock.now
        for i, target in enumerate(["10.0.0.1", "10.0.0.2", "10.0.0.3"]):
            self.engine.ingest(event("TCP_SYN_SCAN", KALI, t0 + i, dst=target))
            self.engine.ingest(event("EXPLOIT_DISTCC_CMDEXEC", KALI, t0 + 10 + i,
                                     dst=target))
        self.assertEqual(self.engine.incident_count(), 1)
        self.assertEqual(len(self.engine.incidents()[0]["destinations"]), 3)

    def test_two_sources_are_two_incidents(self):
        t0 = self.clock.now
        for src in (KALI, UBUNTU):
            self.engine.ingest(event("TCP_SYN_SCAN", src, t0))
            self.engine.ingest(event("EXPLOIT_DISTCC_CMDEXEC", src, t0 + 5))
        self.assertEqual(self.engine.incident_count(), 2)
        self.assertEqual({i["source"] for i in self.engine.incidents()},
                         {KALI, UBUNTU})

    def test_stageless_events_attach_as_context_but_never_open(self):
        """A SYN flood has no kill-chain stage: evidence, not a chain."""
        t0 = self.clock.now
        r1 = self.engine.ingest(event("FLOOD_SYN", KALI, t0))
        r2 = self.engine.ingest(event("FLOOD_PACKET_RATE", KALI, t0 + 5))
        self.assertEqual((r1.action, r2.action), ("buffered", "buffered"))
        self.assertEqual(self.engine.incident_count(), 0)

        # Now a real chain opens; the floods are folded in as evidence.
        self.engine.ingest(event("TCP_SYN_SCAN", KALI, t0 + 10))
        r4 = self.engine.ingest(event("EXPLOIT_JAVA_RMI_DESERIALIZATION",
                                      KALI, t0 + 20))
        self.assertEqual(r4.action, "opened")
        inc = self.engine.incidents()[0]
        self.assertEqual(inc["event_count"], 4)
        self.assertIn("FLOOD_SYN", inc["rule_ids"])
        self.assertEqual(inc["stage_count"], 2)   # floods add evidence, not stages


class TestNegativeCases(unittest.TestCase):
    """What must NEVER become an incident. These are the regression guards."""

    def setUp(self):
        self.clock = FakeClock()
        self.engine = CorrelationEngine(window_sec=600, clock=self.clock)

    def test_two_benign_events_from_same_ip_are_not_an_incident(self):
        """REQUIRED NEGATIVE TEST.

        Two unrelated INFO-severity events from one host — a routine DNS lookup
        and a routine HTTPS connection — must not be grouped. They share a source
        IP and nothing else.
        """
        t0 = self.clock.now
        r1 = self.engine.ingest(event("ML_FLOW_ANOMALY", UBUNTU, t0,
                                      severity="info", stage=None,
                                      predicted_class="BENIGN"))
        r2 = self.engine.ingest(event("ANOMALY_TRAFFIC_BASELINE", UBUNTU, t0 + 30,
                                      severity="info", stage=None))
        self.assertEqual(r1.action, "ignored")
        self.assertEqual(r2.action, "ignored")
        self.assertEqual(self.engine.incident_count(), 0,
                         "benign co-occurrence must not manufacture an incident")
        self.assertIsNone(self.engine.open_incident_for(UBUNTU))

    def test_repeated_reconnaissance_alone_never_opens_an_incident(self):
        """The old engine's worst bug: one nmap run trips >=3 distinct alert
        types and was promoted to a CRITICAL 'Multi-Vector Attack'."""
        t0 = self.clock.now
        for i, rid in enumerate(["TCP_SYN_SCAN", "TCP_FIN_SCAN", "SCAN_SLOW_RATE",
                                 "TCP_NULL_SCAN", "TCP_XMAS_SCAN",
                                 "SCAN_DISTRIBUTED"]):
            r = self.engine.ingest(event(rid, KALI, t0 + i * 2))
            self.assertFalse(r.emit_alert, f"{rid} must not open an incident")
        self.assertEqual(self.engine.incident_count(), 0)

    def test_events_outside_the_window_are_not_correlated(self):
        t0 = self.clock.now
        self.engine.ingest(event("TCP_SYN_SCAN", KALI, t0))
        r = self.engine.ingest(event("EXPLOIT_SAMBA_USERMAP", KALI, t0 + 601))
        self.assertEqual(r.action, "buffered")
        self.assertEqual(self.engine.incident_count(), 0)

    def test_backwards_stage_order_is_not_progression(self):
        """C2 at t=0 then recon at t=30 is not a kill chain advancing."""
        t0 = self.clock.now
        self.engine.ingest(event("BEHAVIOR_C2_BEACON", KALI, t0))
        r = self.engine.ingest(event("TCP_SYN_SCAN", KALI, t0 + 30))
        self.assertEqual(r.action, "buffered")
        self.assertEqual(self.engine.incident_count(), 0)

    def test_unattributable_source_is_dropped(self):
        """ARP spoofing forges the source; it must not be grouped under a victim."""
        r = self.engine.ingest(event("MITM_ARP_SPOOF", "", self.clock.now))
        self.assertEqual(r.action, "ignored")
        self.assertEqual(self.engine.incident_count(), 0)

    def test_same_stage_from_different_rules_is_not_progression(self):
        """Two credential-access rules are one stage, however different the IDs."""
        t0 = self.clock.now
        self.engine.ingest(event("BRUTEFORCE_STANDARD_RATE", KALI, t0))
        r = self.engine.ingest(event("BRUTEFORCE_LOW_RATE", KALI, t0 + 60))
        self.assertEqual(r.action, "buffered")
        self.assertEqual(self.engine.incident_count(), 0)


class TestWindowSemantics(unittest.TestCase):

    def test_window_is_configurable(self):
        clock = FakeClock()
        tight = CorrelationEngine(window_sec=30, clock=clock)
        wide = CorrelationEngine(window_sec=3600, clock=clock)
        t0 = clock.now
        for eng in (tight, wide):
            eng.ingest(event("TCP_SYN_SCAN", KALI, t0))
            eng.ingest(event("EXPLOIT_SAMBA_USERMAP", KALI, t0 + 120))
        self.assertEqual(tight.incident_count(), 0, "120s gap exceeds a 30s window")
        self.assertEqual(wide.incident_count(), 1, "120s gap fits a 3600s window")

    def test_incident_closes_after_idle_and_a_new_one_opens(self):
        clock = FakeClock()
        eng = CorrelationEngine(window_sec=300, clock=clock)
        t0 = clock.now
        eng.ingest(event("TCP_SYN_SCAN", KALI, t0))
        eng.ingest(event("EXPLOIT_SAMBA_USERMAP", KALI, t0 + 10))
        self.assertEqual(eng.incident_count(), 1)
        # Long silence, then a fresh chain.
        t1 = t0 + 5000
        eng.ingest(event("TCP_SYN_SCAN", KALI, t1))
        r = eng.ingest(event("EXPLOIT_DISTCC_CMDEXEC", KALI, t1 + 10))
        self.assertEqual(r.action, "opened")
        self.assertEqual(eng.incident_count(), 2,
                         "a chain after the window closes is a new incident")


class TestDeterminism(unittest.TestCase):

    def _sequence(self, t0):
        return [
            event("TCP_SYN_SCAN", KALI, t0 + 0, dst_port=22),
            event("TCP_FIN_SCAN", KALI, t0 + 5, dst_port=80),
            event("BRUTEFORCE_STANDARD_RATE", KALI, t0 + 30, dst_port=22),
            event("EXPLOIT_VSFTPD_BACKDOOR", KALI, t0 + 60, dst_port=21),
            event("BEHAVIOR_UNEXPECTED_OUTBOUND", METASPLOITABLE, t0 + 65),
            event("BEHAVIOR_C2_BEACON", KALI, t0 + 90),
        ]

    def test_arrival_order_does_not_change_grouping(self):
        """Same six events, five shuffles, identical grouping every time."""
        expected = None
        for seed in range(5):
            clock = FakeClock()
            eng = CorrelationEngine(window_sec=600, clock=clock)
            evts = self._sequence(clock.now)
            random.Random(seed).shuffle(evts)
            for e in evts:
                eng.ingest(e)
            got = sorted(
                (i["source"], i["event_count"], tuple(s["stage"] for s in i["stages"]))
                for i in eng.incidents()
            )
            if expected is None:
                expected = got
            self.assertEqual(got, expected,
                             f"shuffle seed {seed} produced different grouping")

    def test_concurrent_ingest_is_deterministic(self):
        """8 threads racing the same events must not double-open or drop."""
        clock = FakeClock()
        eng = CorrelationEngine(window_sec=600, clock=clock)
        evts = self._sequence(clock.now) * 4
        random.Random(7).shuffle(evts)

        opened = []
        lock = threading.Lock()

        def worker(chunk):
            for e in chunk:
                r = eng.ingest(e)
                if r.emit_alert:
                    with lock:
                        opened.append(r.incident["incident_id"])

        chunks = [evts[i::8] for i in range(8)]
        threads = [threading.Thread(target=worker, args=(c,)) for c in chunks]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # One incident per source (KALI and METASPLOITABLE both progress).
        self.assertEqual(len(opened), len(set(opened)),
                         "an incident must raise its alert exactly once")
        by_source = {i["source"] for i in eng.incidents()}
        self.assertEqual(by_source, {KALI})
        self.assertEqual(eng.incident_count(), 1)
        self.assertEqual(eng.incidents()[0]["event_count"],
                         sum(1 for e in evts if e.actor == KALI))


class TestBounds(unittest.TestCase):

    def test_source_table_is_bounded(self):
        """hping3 --rand-source must not grow the tracker without limit."""
        clock = FakeClock()
        eng = CorrelationEngine(window_sec=600, max_sources=50, clock=clock)
        for i in range(500):
            eng.ingest(event("TCP_SYN_SCAN", f"10.1.{i // 256}.{i % 256}",
                             clock.now + i * 0.01))
        self.assertLessEqual(eng.snapshot_stats()["sources_buffered"], 51)
        self.assertGreater(eng.snapshot_stats()["sources_evicted"], 0)

    def test_events_per_incident_are_capped_but_counted(self):
        clock = FakeClock()
        eng = CorrelationEngine(window_sec=6000, max_events_per_incident=10,
                                clock=clock)
        t0 = clock.now
        eng.ingest(event("TCP_SYN_SCAN", KALI, t0))
        eng.ingest(event("EXPLOIT_SAMBA_USERMAP", KALI, t0 + 1))
        for i in range(50):
            eng.ingest(event("BEHAVIOR_C2_BEACON", KALI, t0 + 2 + i))
        inc = eng.incidents()[0]
        self.assertEqual(inc["stored_events"], 10)
        self.assertEqual(inc["event_count"], 52, "the true count is preserved")
        self.assertEqual(inc["dropped_events"], 42)


if __name__ == "__main__":
    unittest.main(verbosity=2)

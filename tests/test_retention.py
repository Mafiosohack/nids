"""Alert retention: severity-weighted TTL, deduplication, audited eviction.

The finding being fixed: a fixed-capacity store with no eviction policy, so 200
identical port-scan pings silently pushed CRITICALs out of the ring.
"""

import unittest

from helpers import KALI, METASPLOITABLE, FakeClock

from detection.retention import AlertStore


def alert(rule_id, severity, aid=1, src=KALI, dst=METASPLOITABLE, port=22,
          message="x"):
    return {"id": aid, "rule_id": rule_id, "type": rule_id, "severity": severity,
            "src": src, "dst": dst, "protocol": "TCP", "message": message,
            "timestamp": "2026-08-11 12:00:00", "details": {"dst_port": port}}


class TestDeduplication(unittest.TestCase):

    def setUp(self):
        self.clock = FakeClock()
        self.store = AlertStore(capacity=1000, dedup_window_sec=300,
                                clock=self.clock)

    def test_200_identical_alerts_become_one_record_with_a_count(self):
        """The exact scenario from the brief."""
        for i in range(200):
            self.store.add(alert("TCP_SYN_SCAN", "high", aid=i),
                           ts=self.clock.now + i * 0.5)
        self.assertEqual(len(self.store), 1, "200 identical alerts = 1 record")
        rec = self.store.list()[0]
        self.assertEqual(rec["count"], 200, "nothing is lost — it is counted")
        self.assertEqual(self.store.total_alerts_represented, 200)
        self.assertEqual(self.store.stats["collapsed_alerts"], 199)

    def test_collapses_are_logged(self):
        for i in range(150):
            self.store.add(alert("TCP_SYN_SCAN", "high", aid=i),
                           ts=self.clock.now + i)
        collapses = [e for e in self.store.eviction_log if e["action"] == "collapse"]
        self.assertTrue(collapses, "collapsing must leave a trace")
        self.assertEqual(collapses[0]["rule_id"], "TCP_SYN_SCAN")
        self.assertIn("folded into", collapses[0]["reason"])

    def test_different_ports_are_different_facts(self):
        for i, port in enumerate([22, 23, 80, 443]):
            self.store.add(alert("TCP_SYN_SCAN", "high", aid=i, port=port),
                           ts=self.clock.now)
        self.assertEqual(len(self.store), 4)

    def test_different_sources_are_different_facts(self):
        for i, src in enumerate(["10.0.0.1", "10.0.0.2", "10.0.0.3"]):
            self.store.add(alert("TCP_SYN_SCAN", "high", aid=i, src=src),
                           ts=self.clock.now)
        self.assertEqual(len(self.store), 3)

    def test_repeat_outside_the_dedup_window_is_a_new_record(self):
        self.store.add(alert("TCP_SYN_SCAN", "high", aid=1), ts=self.clock.now)
        self.store.add(alert("TCP_SYN_SCAN", "high", aid=2),
                       ts=self.clock.now + 400)
        self.assertEqual(len(self.store), 2)

    def test_a_sustained_attack_keeps_extending_one_record(self):
        """Repeats inside the rolling window never start a second record."""
        t = self.clock.now
        for i in range(100):
            self.store.add(alert("FLOOD_SYN", "critical", aid=i), ts=t + i * 250)
        self.assertEqual(len(self.store), 1)
        self.assertEqual(self.store.list()[0]["count"], 100)


class TestSeverityWeightedRetention(unittest.TestCase):

    def setUp(self):
        self.clock = FakeClock()
        self.store = AlertStore(capacity=1000, dedup_window_sec=0,
                                clock=self.clock)

    def test_critical_and_high_are_kept_indefinitely(self):
        self.store.add(alert("FLOOD_SYN", "critical", aid=1), ts=self.clock.now)
        self.store.add(alert("TCP_SYN_SCAN", "high", aid=2), ts=self.clock.now)
        self.clock.advance(60 * 86400)                 # two months
        self.store.sweep(self.clock.now)
        self.assertEqual(len(self.store), 2)

    def test_low_and_info_expire_on_a_short_ttl(self):
        t = self.clock.now
        self.store.add(alert("TCP_FIN_SCAN", "low", aid=1), ts=t)
        self.store.add(alert("ML_FLOW_ANOMALY", "info", aid=2), ts=t)
        self.store.add(alert("FLOOD_SYN", "critical", aid=3), ts=t)

        self.clock.advance(2 * 3600)                   # 2h: info gone, low stays
        self.store.sweep(self.clock.now)
        self.assertEqual({r["severity"] for r in self.store.list()},
                         {"low", "critical"})

        self.clock.advance(8 * 3600)                   # 10h total: low gone too
        self.store.sweep(self.clock.now)
        self.assertEqual([r["severity"] for r in self.store.list()], ["critical"])

    def test_medium_ttl_is_a_day(self):
        t = self.clock.now
        self.store.add(alert("SCAN_SLOW_RATE", "medium", aid=1), ts=t)
        self.clock.advance(23 * 3600)
        self.store.sweep(self.clock.now)
        self.assertEqual(len(self.store), 1)
        self.clock.advance(2 * 3600)
        self.store.sweep(self.clock.now)
        self.assertEqual(len(self.store), 0)

    def test_ttl_evictions_are_logged(self):
        self.store.add(alert("ML_FLOW_ANOMALY", "info", aid=1), ts=self.clock.now)
        self.clock.advance(2 * 3600)
        self.store.sweep(self.clock.now)
        ev = [e for e in self.store.eviction_log if e["action"] == "evict_ttl"]
        self.assertEqual(len(ev), 1)
        self.assertEqual(ev[0]["severity"], "info")
        self.assertIn("TTL", ev[0]["reason"])


class TestCapacityPressure(unittest.TestCase):

    def test_criticals_survive_a_flood_of_low_severity_noise(self):
        """The original bug: a fixed ring where noise evicted the CRITICALs."""
        clock = FakeClock()
        store = AlertStore(capacity=50, dedup_window_sec=0, clock=clock)
        store.add(alert("FLOOD_SYN", "critical", aid=0, src="10.9.9.9"),
                  ts=clock.now)
        for i in range(500):
            store.add(alert("TCP_FIN_SCAN", "low", aid=100 + i,
                            src=f"10.0.{i // 254}.{i % 254}"),
                      ts=clock.now + i)
        remaining = store.list()
        self.assertEqual(len(remaining), 50)
        self.assertIn("critical", {r["severity"] for r in remaining},
                      "a CRITICAL must not be evicted while LOWs remain")

    def test_capacity_evictions_are_logged(self):
        clock = FakeClock()
        store = AlertStore(capacity=10, dedup_window_sec=0, clock=clock)
        for i in range(40):
            store.add(alert("TCP_FIN_SCAN", "low", aid=i, src=f"10.0.0.{i}"),
                      ts=clock.now + i)
        ev = [e for e in store.eviction_log if e["action"] == "evict_capacity"]
        self.assertEqual(len(ev), 30)
        self.assertIn("capacity", ev[0]["reason"])

    def test_eviction_log_is_itself_bounded(self):
        clock = FakeClock()
        store = AlertStore(capacity=5, dedup_window_sec=0, log_capacity=20,
                           clock=clock)
        for i in range(500):
            store.add(alert("TCP_FIN_SCAN", "low", aid=i, src=f"10.1.{i//254}.{i%254}"),
                      ts=clock.now + i)
        self.assertLessEqual(len(store.eviction_log), 20)


class TestStats(unittest.TestCase):

    def test_snapshot_reports_the_policy_and_the_true_event_count(self):
        clock = FakeClock()
        store = AlertStore(capacity=100, clock=clock)
        for i in range(30):
            store.add(alert("TCP_SYN_SCAN", "high", aid=i), ts=clock.now + i)
        store.add(alert("FLOOD_SYN", "critical", aid=99, port=None), ts=clock.now)
        s = store.snapshot_stats()
        self.assertEqual(s["records"], 2)
        self.assertEqual(s["alerts_represented"], 31)
        self.assertEqual(s["ttl_by_severity"]["critical"], "indefinite")
        self.assertEqual(s["ttl_by_severity"]["info"], 3600)
        self.assertEqual(s["records_by_severity"], {"high": 1, "critical": 1})


if __name__ == "__main__":
    unittest.main(verbosity=2)

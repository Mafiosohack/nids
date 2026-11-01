"""Alert History: the database-backed record behind the dashboard's History view.

The live store forgets MEDIUM alerts after 24 h and LOW after 6 h; the
database keeps the last 50,000 of every severity. These tests pin down the
query surface over that record (GET /alerts/history), that a verdict given
from History actually lands on a database-only alert, and that repeats folded
into an existing alert reach the database instead of living only in memory.
"""

import sqlite3
import time
import unittest

try:
    import scapy.all  # noqa: F401  (main.py imports it; mirrors other suites)
    SCAPY = True
except ImportError:                                   # pragma: no cover
    SCAPY = False

NOW = time.time()

# id, rule_id, type, severity, src, dst, message, triage_status, age_sec
ROWS = [
    (1, "TCP_SYN_SCAN", "TCP SYN (half-open) Scan", "high",
     "192.168.56.101", "192.168.56.102", "SYN scan of 1000 ports", None, 500),
    (2, "EXPLOIT_VSFTPD_BACKDOOR_SHELL", "vsftpd Backdoor Shell Opened", "critical",
     "192.168.56.101", "192.168.56.102", "backdoor shell on 6200", "new", 400),
    (3, "BRUTEFORCE_STANDARD_RATE", "Brute Force (standard rate)", "high",
     "192.168.56.103", "192.168.56.102", "20 failed SSH logins", "confirmed", 300),
    (4, "ML_FLOW_ANOMALY", "ML Anomaly", "medium",
     "192.168.56.1", "192.168.56.255", "flow looked like 50%_off promo", "false_positive", 200),
    (5, "ASSET_ROGUE_HOST", "Rogue Host", "medium",
     "192.168.56.130", None, "unknown host appeared", "resolved", 100),
    (7, "ML_FLOW_ANOMALY", "ML Anomaly", "medium",
     "192.168.56.102", "8.8.8.8", "flow was 50 percent off baseline", "new", 40),
]
LEGACY_ID = 6          # written before rule_id / count / triage existed


def seed(main):
    """Replace the alerts table with ROWS plus one legacy row."""
    with sqlite3.connect(str(main.DB_PATH)) as c:
        c.execute("DELETE FROM alerts")
    for (aid, rule, typ, sev, src, dst, msg, tri, age) in ROWS:
        main.db_insert_alert({"id": aid, "rule_id": rule, "type": typ,
                              "severity": sev, "src": src, "dst": dst,
                              "protocol": "TCP", "message": msg,
                              "timestamp": "2026-11-01 14:00:00", "details": {},
                              "count": 1, "triage_status": tri})
        with sqlite3.connect(str(main.DB_PATH)) as c:
            c.execute("UPDATE alerts SET created_at=? WHERE id=?", (NOW - age, aid))
    with sqlite3.connect(str(main.DB_PATH)) as c:
        c.execute("INSERT INTO alerts (id, type, severity, src, dst, protocol, message,"
                  " timestamp, details, created_at, rule_id, count, triage_status)"
                  " VALUES (?,?,?,?,?,?,?,?,?,?,NULL,NULL,NULL)",
                  (LEGACY_ID, "Port Scan", "low", "192.168.56.101", "192.168.56.102",
                   "TCP", "legacy row", "2026-07-01 10:00:00", "{}", NOW - 50))


@unittest.skipUnless(SCAPY, "main.py needs scapy")
class TestHistoryEndpoint(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from apiharness import LiveServer
        cls.srv = LiveServer().__enter__()
        cls.main = cls.srv.main
        seed(cls.main)
        cls.tok = cls.srv.login()

    @classmethod
    def tearDownClass(cls):
        cls.srv.__exit__(None, None, None)

    def hist(self, query="", status=200, token=None):
        r = self.srv.get("/alerts/history" + (("?" + query) if query else ""),
                         token=token or self.tok)
        self.assertEqual(r.status, status, r.body)
        return r.json()

    def ids(self, query=""):
        return [row["id"] for row in self.hist(query)["rows"]]

    def test_status_new_includes_legacy_null_rows(self):
        """Breaks if status=new matches only the literal 'new': rows written
        before triage existed (NULL) would silently leave the review list."""
        got = self.hist("status=new")
        self.assertEqual(got["total"], 4)
        self.assertEqual(sorted(self.ids("status=new")), [1, 2, LEGACY_ID, 7])

    def test_status_filters(self):
        self.assertEqual(self.ids("status=confirmed"), [3])
        self.assertEqual(self.ids("status=false_positive"), [4])
        self.assertEqual(self.ids("status=resolved"), [5])
        self.assertEqual(self.hist("status=all")["total"], 7)
        self.assertEqual(self.hist()["total"], 7, "default is all")

    def test_bad_status_is_400(self):
        self.hist("status=closed", status=400)

    def test_severity_repeatable(self):
        self.assertEqual(sorted(self.ids("severity=high&severity=critical")), [1, 2, 3])

    def test_bad_severity_is_400(self):
        self.hist("severity=urgent", status=400)

    def test_since_uses_last_seen(self):
        got = self.hist(f"since={NOW - 250}")
        self.assertEqual(sorted(r["id"] for r in got["rows"]), [4, 5, LEGACY_ID, 7])

    def test_ip_matches_src_or_dst_exactly(self):
        """192.168.56.1 must not match .101/.102/.103/.130 (no prefix LIKE)."""
        self.assertEqual(self.ids("ip=192.168.56.1"), [4])
        self.assertEqual(sorted(self.ids("ip=8.8.8.8")), [7], "matches destination too")

    def test_rule_id_filter(self):
        self.assertEqual(sorted(self.ids("rule_id=ML_FLOW_ANOMALY")), [4, 7])

    def test_q_is_literal(self):
        """Breaks if % and _ are passed to LIKE unescaped: '50%_off' would then
        also match 'flow was 50 percent off baseline'."""
        from urllib.parse import quote
        self.assertEqual(self.ids("q=" + quote("50%_off")), [4])
        got = self.hist("q=" + quote('"; DROP TABLE alerts;--'))
        self.assertEqual(got["total"], 0)
        self.assertEqual(self.hist()["total"], 7, "table survived")

    def test_sort_worst_then_newest(self):
        self.assertEqual(self.ids("status=new&sort=worst"), [2, 1, 7, LEGACY_ID])
        self.assertEqual(self.ids("status=new"), [7, LEGACY_ID, 2, 1], "default newest")

    def test_bad_sort_is_400(self):
        self.hist("sort=random", status=400)

    def test_paging_and_total(self):
        got = self.hist("limit=2&offset=2")
        self.assertEqual([r["id"] for r in got["rows"]], [5, 4])
        self.assertEqual((got["total"], got["offset"], got["limit"]), (7, 2, 2))

    def test_offset_past_end(self):
        got = self.hist("offset=10000")
        self.assertEqual(got["rows"], [])
        self.assertEqual(got["total"], 7)

    def test_limit_cap(self):
        self.hist("limit=201", status=422)
        self.hist("limit=0", status=422)

    def test_param_length_caps(self):
        self.hist("ip=" + "1" * 46, status=422)
        self.hist("rule_id=" + "A" * 65, status=422)
        self.hist("q=" + "x" * 201, status=422)

    def test_requires_auth(self):
        r = self.srv.get("/alerts/history")
        self.assertEqual(r.status, 401)

    def test_viewer_can_read(self):
        self.srv.post("/users", {"username": "vera", "password": "vera-password-1",
                                 "role": "viewer"}, token=self.tok)
        vtok = self.srv.login("vera", "vera-password-1")
        self.assertEqual(self.hist(token=vtok)["total"], 7)

    def test_legacy_row_shape(self):
        row = next(r for r in self.hist()["rows"] if r["id"] == LEGACY_ID)
        self.assertEqual(row["count"], 1)
        self.assertIsNone(row["rule_id"])
        self.assertEqual(row["triage_status"], "new")
        self.assertEqual(row["details"], {})
        self.assertAlmostEqual(row["last_seen"], NOW - 50, delta=1)
        self.assertNotIn("created_at", row)


@unittest.skipUnless(SCAPY, "main.py needs scapy")
class TestTriageReachesHistory(unittest.TestCase):
    """A verdict given from History on an alert the live store has already
    forgotten must land in the database, not be silently skipped."""

    @classmethod
    def setUpClass(cls):
        from apiharness import LiveServer
        cls.srv = LiveServer().__enter__()
        cls.main = cls.srv.main
        cls.tok = cls.srv.login()

    @classmethod
    def tearDownClass(cls):
        cls.srv.__exit__(None, None, None)

    def setUp(self):
        self.main.PIPELINE.reset()
        seed(self.main)                      # DB-only rows: none are in the live store

    def triage(self, ids, status, **kw):
        body = {"ids": ids, "status": status, "note": kw.pop("note", "checked")}
        body.update(kw)
        r = self.srv.post("/alerts/triage", body, token=self.tok)
        self.assertEqual(r.status, 200, r.body)
        return r.json()

    def status_of(self, aid):
        rows = self.srv.get("/alerts/history?limit=200", token=self.tok).json()["rows"]
        return next(r["triage_status"] for r in rows if r["id"] == aid)

    def publish_live(self):
        from detection.findings import Finding
        alert = self.main.publish_finding(Finding(
            rule_id="TCP_SYN_SCAN", src="192.168.56.120", dst="192.168.56.102",
            message="live scan", details={"dst_port": 22}))
        self.assertIsNotNone(alert)
        return alert["id"]

    def test_db_only_verdict_persists(self):
        self.assertIsNone(self.main.PIPELINE.alerts.get(1))
        got = self.triage([1], "resolved")
        self.assertEqual(got["updated"], 1)
        self.assertEqual(got["ids"], [1])
        self.assertEqual(self.status_of(1), "resolved")

    def test_mixed_batch(self):
        live = self.publish_live()
        got = self.triage([live, 2], "confirmed")
        self.assertEqual(got["updated"], 2)
        self.assertEqual(sorted(got["ids"]), sorted([live, 2]))
        self.assertEqual(self.status_of(live), "confirmed")
        self.assertEqual(self.status_of(2), "confirmed")

    def test_unknown_id_reported(self):
        got = self.triage([1, 999999], "resolved")
        self.assertEqual(got["not_found"], [999999])
        self.assertNotIn(999999, got["ids"])

    def test_admin_suppress_from_db_only_fp(self):
        got = self.triage([7], "false_positive", suppress=True, note="lab noise")
        pairs = {(s["rule_id"], s["src"]) for s in got["suppressions_created"]}
        self.assertIn(("ML_FLOW_ANOMALY", "192.168.56.102"), pairs)


class TestRepeatHook(unittest.TestCase):
    """The pipeline tells its owner when a finding folds into an existing alert,
    so the owner can persist the new count (and a reopen) instead of leaving
    the database at count 1 forever."""

    def setUp(self):
        from helpers import FakeClock
        from detection.pipeline import DetectionPipeline, PipelineConfig
        self.clock = FakeClock()
        self.seen = []
        self.pipe = DetectionPipeline(PipelineConfig(), clock=self.clock,
                                      on_repeat=self.seen.append)

    def scan(self, dt=0):
        from detection.findings import Finding
        return self.pipe.publish(Finding(
            rule_id="TCP_SYN_SCAN", src="192.168.56.101", dst="192.168.56.102",
            ts=self.clock.now + dt, message="scan", details={"dst_port": 22}))

    def test_new_record_does_not_call_on_repeat(self):
        self.scan()
        self.assertEqual(self.seen, [])

    def test_repeat_calls_on_repeat_with_count(self):
        first = self.scan()
        self.assertIsNone(self.scan(5), "a repeat publishes no new alert")
        self.assertEqual(len(self.seen), 1)
        self.assertEqual(self.seen[0]["id"], first["id"])
        self.assertEqual(self.seen[0]["count"], 2)

    def test_reopen_is_in_the_record(self):
        first = self.scan()
        self.pipe.alerts.set_triage([first["id"]], "resolved", "ana")
        self.scan(5)
        self.assertEqual(self.seen[-1]["triage_status"], "new")

    def test_reopen_is_flagged_and_plain_repeat_is_not(self):
        """Breaks if the owner cannot tell a reopen from a plain repeat: it must
        write triage fields only on a reopen, or a stale repeat copy can
        overwrite a verdict saved a moment earlier."""
        first = self.scan()
        self.scan(1)
        self.assertIs(self.seen[-1]["reopened"], False)
        self.pipe.alerts.set_triage([first["id"]], "resolved", "ana")
        self.scan(2)
        self.assertIs(self.seen[-1]["reopened"], True)


@unittest.skipUnless(SCAPY, "main.py needs scapy")
class TestRepeatPersisted(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from apiharness import LiveServer
        cls.srv = LiveServer().__enter__()
        cls.main = cls.srv.main
        cls.tok = cls.srv.login()

    @classmethod
    def tearDownClass(cls):
        cls.srv.__exit__(None, None, None)

    def setUp(self):
        self.main.PIPELINE.reset()
        with sqlite3.connect(str(self.main.DB_PATH)) as c:
            c.execute("DELETE FROM alerts")

    def publish(self):
        from detection.findings import Finding
        return self.main.publish_finding(Finding(
            rule_id="TCP_SYN_SCAN", src="192.168.56.121", dst="192.168.56.102",
            message="scan", details={"dst_port": 22}))

    def row(self, aid):
        rows = self.srv.get("/alerts/history", token=self.tok).json()["rows"]
        return next(r for r in rows if r["id"] == aid)

    def test_db_count_and_last_seen_follow_repeats(self):
        aid = self.publish()["id"]
        with sqlite3.connect(str(self.main.DB_PATH)) as c:
            c.execute("UPDATE alerts SET created_at=? WHERE id=?", (time.time() - 100, aid))
        self.assertIsNone(self.publish())
        row = self.row(aid)
        self.assertEqual(row["count"], 2)
        self.assertGreater(row["last_seen"], time.time() - 50)

    def test_reopen_persisted(self):
        aid = self.publish()["id"]
        r = self.srv.post("/alerts/triage", {"ids": [aid], "status": "resolved",
                                             "note": "done"}, token=self.tok)
        self.assertEqual(r.status, 200)
        self.assertEqual(self.row(aid)["triage_status"], "resolved")
        self.publish()
        self.assertEqual(self.row(aid)["triage_status"], "new")

    def test_repeat_flood_writes_are_bounded_and_history_is_exact(self):
        """Breaks if every repeat writes SQLite on the capture thread: an
        attacker replaying one packet would set the capture rate. History
        must still show the exact count when it is read."""
        calls = []
        real = self.main.db_touch_alert
        self.main.db_touch_alert = lambda *a, **k: (calls.append(a), real(*a, **k))
        try:
            aid = self.publish()["id"]
            for _ in range(60):
                self.publish()
        finally:
            self.main.db_touch_alert = real
        self.assertLessEqual(len(calls), 3, "repeat writes from the capture thread")
        self.assertEqual(self.row(aid)["count"], 61)

    def test_plain_repeat_does_not_overwrite_a_verdict(self):
        """Breaks if a repeat rewrites triage fields from a stale copy: the
        sniffer copies the record (new), the analyst confirms, the repeat
        write lands and the verdict is lost on restart."""
        aid = self.publish()["id"]
        r = self.srv.post("/alerts/triage", {"ids": [aid], "status": "confirmed"},
                          token=self.tok)
        self.assertEqual(r.status, 200)
        stale = {"id": aid, "count": 10, "triage_status": "new", "triage_by": None,
                 "triage_at": None, "triage_note": None, "reopened": False}
        self.main._on_repeat_alert(stale)
        row = self.row(aid)
        self.assertEqual(row["triage_status"], "confirmed")
        self.assertEqual(row["count"], 10)


if __name__ == "__main__":
    unittest.main()

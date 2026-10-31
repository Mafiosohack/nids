"""Regression tests for the 2026-10-31 security audit, step 1.

Each test replays one proof-of-concept from the audit and asserts the attack no
longer works. The production change that would make each test fail is named in
its docstring, so a future edit that reopens the hole is caught here.

Usability constraint these tests also hold: the sensors' real, valid payloads
must keep being accepted unchanged. Hardening that breaks the tool's core job
is not hardening.
"""

import json
import os
import re
import shutil
import subprocess
import tempfile
import unittest

from helpers import FakeClock

from detection.pipeline import DetectionPipeline
from detection.reassembly import FragmentReassembler

try:
    from scapy.all import IP, TCP, Raw
    SCAPY = True
except ImportError:                                   # pragma: no cover
    SCAPY = False

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DASHBOARD = os.path.join(REPO, "nids_dashboard.html")


def _feed_full_datagram(reasm, total, chunk=1480, ip_id=7):
    """Feed fragments covering [0, total) and return the last add_fragment result."""
    off, done = 0, None
    while off < total:
        n = min(chunk, total - off)
        done = reasm.add_fragment("203.0.113.9", "192.168.56.101", 6, ip_id,
                                  off // 8, off + n < total, b"A" * n, ts=0.0)
        off += n
    return done


# ── Chain 4: one oversized datagram must not blind the sensor ────────────────
class TestReassemblySizeCap(unittest.TestCase):
    def test_datagram_whose_ip_length_would_overflow_is_dropped(self):
        """Breaks if the cap goes back to 65535 bytes of TRANSPORT payload: the
        rebuilt IP header then needs a total length > 65535 and scapy raises."""
        reasm = FragmentReassembler()
        self.assertIsNone(_feed_full_datagram(reasm, 65520))
        self.assertGreaterEqual(reasm.stats["datagrams_dropped_oversize"], 1)

    def test_largest_legal_datagram_still_reassembles(self):
        """Usability: the cap must not reject anything a real stack can send."""
        reasm = FragmentReassembler()
        done = _feed_full_datagram(reasm, 65515)
        self.assertIsNotNone(done)
        self.assertEqual(len(done.payload), 65515)


@unittest.skipUnless(SCAPY, "scapy required")
class TestPerPacketIsolation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import main
        cls.main = main

    def test_a_detector_exception_does_not_escape_process_packet(self):
        """Breaks if process_packet loses its per-packet try/except: scapy then
        closes the capture socket and drops every packet queued behind it."""
        m = self.main
        real = m.detect_ddos

        def boom(*a, **k):
            raise ValueError("poisoned packet")
        m.detect_ddos = boom
        try:
            before = m.system_stats.get("packet_errors", 0)
            m.process_packet(IP(src="203.0.113.9", dst="192.168.56.101") / TCP())
            self.assertEqual(m.system_stats["packet_errors"], before + 1)
            self.assertIn("poisoned packet", m.system_stats["last_packet_error"])
        finally:
            m.detect_ddos = real

    def test_status_reports_packet_errors(self):
        """Breaks if the error count is not surfaced: a silently skipped packet
        is the same blind spot as a dropped one, just harder to notice."""
        st = self.main.status({"username": "t", "role": "admin"})
        self.assertIn("packet_errors", st)
        self.assertIn("last_packet_error", st)

    def test_oversized_fragmented_datagram_is_harmless_end_to_end(self):
        """The audit PoC itself: 45 fragments reassembling past 65515 bytes."""
        off, total = 0, 65520
        while off < total:
            n = min(1480, total - off)
            frag = IP(src="203.0.113.9", dst="192.168.56.101", proto=6, id=4243,
                      flags=1 if off + n < total else 0, frag=off // 8) / Raw(b"A" * n)
            self.main.process_packet(IP(bytes(frag)))      # must not raise
            off += n


# ── Chains 1 and 3: the sensor intake trusts nothing it does not need to ─────
@unittest.skipUnless(SCAPY, "main.py needs scapy for a meaningful import")
class TestSensorIntake(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import main
        cls.main = main
        cls.tmp = tempfile.mkdtemp()
        cls._db = main.DB_PATH
        main.DB_PATH = main.Path(cls.tmp) / "test.db"
        main.ADMIN_SEED_PASSWORD = "test-only-password"
        main.init_db()

    @classmethod
    def tearDownClass(cls):
        cls.main.DB_PATH = cls._db
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def setUp(self):
        self.main.PIPELINE.reset()

    def model(self, **kw):
        base = {"src": "192.168.56.50", "alert_type": "ML Anomaly",
                "severity": "high", "message": "m"}
        base.update(kw)
        return self.main.ExternalAlert(**base)

    def assertRejected(self, **kw):
        from pydantic import ValidationError
        with self.assertRaises(ValidationError):
            self.model(**kw)

    def test_html_in_severity_is_rejected(self):
        """Breaks if severity is free text again - the stored-XSS vector."""
        self.assertRejected(severity='"><img src=x onerror=alert(1)>')

    def test_uppercase_severity_is_normalised_not_rejected(self):
        """Usability: a sensor sending HIGH is right, just loud."""
        self.assertEqual(self.model(severity="HIGH").severity, "high")

    def test_unknown_alert_type_is_rejected(self):
        self.assertRejected(alert_type="<script>")

    def test_every_alert_type_the_bundled_sensors_send_is_accepted(self):
        """Usability: none of the three shipped sensors may start getting 422s."""
        sent = ["ML Anomaly", "Host Brute Force", "SSH Login", "Root Access",
                "Sensitive File Access", "Persistence", "Cloud Port Scan",
                "Cloud Recon", "Cloud Brute Force", "Cloud UDP Amplification",
                "Data Exfiltration", "Lateral Movement", "Impossible Travel"]
        for t in sent:
            self.model(alert_type=t)

    def test_src_must_be_an_address_or_hostname(self):
        self.assertRejected(src="<b>x</b>")
        self.model(src="203.0.113.66")
        self.model(src="web-prod-01")          # host sensor's fallback actor

    def test_oversized_details_are_rejected(self):
        self.assertRejected(details={"blob": "x" * 20000})

    def test_long_message_is_truncated_not_rejected(self):
        """Usability: keep the alert, drop the excess."""
        self.assertLessEqual(len(self.model(message="x" * 5000).message), 1000)

    def test_sensor_cannot_claim_a_packet_rule_via_details(self):
        """Breaks if /alert honours details.rule_id again: that let a key-holder
        forge EXPLOIT_* rules and build an ACTIVE_INTRUSION against any IP."""
        a = self.model(details={"rule_id": "EXPLOIT_VSFTPD_BACKDOOR"})
        out = self.main.receive_external_alert(a, sender="ml")
        rec = self.main.PIPELINE.alerts.get(out["alert_id"])
        self.assertEqual(rec["rule_id"], "ML_FLOW_ANOMALY")


# ── Clear Alerts clears alerts, not what the sensor has learned ──────────────
class TestClearAlertsKeepsLearnedState(unittest.TestCase):
    def setUp(self):
        self.clock = FakeClock()
        self.p = DetectionPipeline(clock=self.clock)

    def test_clear_alerts_keeps_profiles_and_confirmed_logins(self):
        """Breaks if Clear Alerts goes back to pipeline.reset(): that wiped the
        host baseline (and the saver then persisted the empty store)."""
        self.p.on_tcp_packet("192.168.56.10", "192.168.56.101", 51000, 22, 0x02,
                             ts=self.clock.now)
        self.p.shell.note_authorized_session("192.168.56.10", "192.168.56.101")
        for port in range(20, 40):                          # make some alerts
            self.p.on_tcp_packet("192.168.56.66", "192.168.56.101", 40000, port,
                                 0x02, ts=self.clock.now)
        self.assertGreater(len(self.p.alerts), 0)

        self.p.clear_alerts()

        self.assertEqual(len(self.p.alerts), 0)
        self.assertEqual(self.p.correlator.incident_count(), 0)
        self.assertIsNotNone(self.p.profiles.profile("192.168.56.10"))
        self.assertIn(("192.168.56.10", "192.168.56.101"), self.p.shell._authorized)


# ── Host-log sensor: attacker-typed usernames must not steer attribution ─────
class TestHostLogInjection(unittest.TestCase):
    def setUp(self):
        import host_log_sensor as h
        self.h = h
        self.sent = []
        self._real = h.send_alert
        h.send_alert = lambda *a, **k: self.sent.append(a)
        h.failed_tracker.clear(); h.active_sessions.clear(); h._recent.clear()

    def tearDown(self):
        self.h.send_alert = self._real

    def feed(self, msg, n=1):
        for i in range(n):
            self.h.process_line(f"Oct 31 10:00:{i:02d} web-prod-01 {msg}")

    def test_username_cannot_frame_another_ip(self):
        """Breaks if RE_FAILED is unanchored again: the username
        'root from 10.0.0.5' made the brute force look like it came from 10.0.0.5."""
        self.feed("sshd[1]: Failed password for invalid user root from 10.0.0.5 "
                  "from 203.0.113.9 port 4242 ssh2", n=5)
        srcs = {a[2] for a in self.sent if a[0] == "Host Brute Force"}
        self.assertEqual(srcs, {"203.0.113.9"})

    def test_invalid_user_line_cannot_forge_root_or_login(self):
        """Breaks if the sudo/useradd/Accepted patterns can match anywhere in a
        line: an SSH username is attacker text inside an sshd line."""
        self.feed("sshd[1]: Invalid user x sudo: a : USER=root ; COMMAND=/bin/bash "
                  "from 203.0.113.9 port 1")
        self.feed("sshd[1]: Invalid user sshd[2]: Accepted password for a from "
                  "10.0.0.9 port 1 ssh2 from 203.0.113.9 port 2")
        self.feed("sshd[1]: Invalid user useradd[3]: new user: name=evil, UID=0 "
                  "from 203.0.113.9 port 3")
        self.assertEqual(self.sent, [])

    def test_the_generated_demo_log_still_produces_the_full_chain(self):
        """Usability: the shipped demo (host_logs/fullchain.log) must still work."""
        path = os.path.join(REPO, "host_logs", "fullchain.log")
        if not os.path.exists(path):
            self.skipTest("demo log not generated")
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                self.h.process_line(line)
        kinds = {a[0] for a in self.sent}
        self.assertTrue({"Host Brute Force", "SSH Login", "Root Access",
                         "Persistence"} <= kinds, kinds)


# ── Dashboard: data from the server is data, never markup ────────────────────
NODE = shutil.which("node")


def _js_function(src: str, name: str) -> str:
    """Extract `function name(...) {...}` from the dashboard by brace matching."""
    m = re.search(r"function\s+" + re.escape(name) + r"\s*\(", src)
    if not m:
        raise AssertionError(f"function {name} not found in dashboard")
    i = src.index("{", m.end())
    depth = 0
    for j in range(i, len(src)):
        depth += {"{": 1, "}": -1}.get(src[j], 0)
        if depth == 0:
            return src[m.start(): j + 1]
    raise AssertionError(f"unbalanced braces in {name}")


@unittest.skipUnless(NODE, "node required for dashboard JS tests")
class TestDashboardEscaping(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(DASHBOARD, encoding="utf-8") as fh:
            cls.src = fh.read()

    def run_js(self, fn_names, expr):
        code = "\n".join(_js_function(self.src, n) for n in fn_names)
        code += f"\nprocess.stdout.write(JSON.stringify({expr}));"
        out = subprocess.run([NODE, "-e", code], capture_output=True, text=True,
                             timeout=20)
        self.assertEqual(out.returncode, 0, out.stderr)
        return json.loads(out.stdout)

    def test_esc_escapes_single_quotes(self):
        self.assertNotIn("'", self.run_js(["esc"], "esc(\"x');alert(1)//\")"))

    def test_severity_is_whitelisted_before_rendering(self):
        """Breaks if the alert list renders raw severity again (stored XSS)."""
        self.assertEqual(self.run_js(["sevKey"], "sevKey('\"><img src=x>')"), "low")
        self.assertEqual(self.run_js(["sevKey"], "sevKey('HIGH')"), "high")

    def test_alert_list_uses_the_whitelist(self):
        """Breaks if `sev` in the alert row is read from the record directly
        again; every use of it (class name and label) must go through sevKey."""
        body = _js_function(self.src, "renderAlerts")
        self.assertRegex(body, r"\bsev\s*=\s*sevKey\(a\.severity\)")
        self.assertNotRegex(body, r"\bsev\s*=\s*\(?a\.severity")

    def test_api_override_ignored_on_a_served_page(self):
        """Breaks if ?api= is honoured on the hosted dashboard: a crafted link
        would post the operator's password to another origin."""
        loc = ("{protocol:'http:', origin:'http://nids:8000', "
               "search:'?api=https://evil.example'}")
        self.assertEqual(self.run_js(["resolveApiBase"], f"resolveApiBase({loc})"),
                         "http://nids:8000")

    def test_api_override_still_works_from_a_local_file(self):
        """Usability: opening the HTML file locally against a remote sensor."""
        loc = "{protocol:'file:', origin:'null', search:'?api=http://10.0.0.2:8000/'}"
        self.assertEqual(self.run_js(["resolveApiBase"], f"resolveApiBase({loc})"),
                         "http://10.0.0.2:8000")

    def test_user_table_has_no_inline_handlers_built_from_usernames(self):
        self.assertNotIn("onclick=\"deleteUser(", self.src)
        self.assertNotIn("onchange=\"changeRole(", self.src)


if __name__ == "__main__":
    unittest.main()

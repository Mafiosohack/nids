"""Benign traffic regression lock.

Efficacy testing found benign traffic was not misclassified — but nothing held
that in place. These tests replay the four hand-tested benign scenarios through
the real pipeline and assert BOTH conditions:

  1. no alert above INFO severity, and
  2. no incident created by the correlation engine.

Condition 2 is the one that would previously have failed: the old engine
promoted any three distinct alert types from one source to a CRITICAL
"Multi-Vector Attack", so benign activity that tripped three low-value rules
manufactured an incident even when no individual alert was wrong.
"""

import unittest

from helpers import FakeClock

from detection.pipeline import DetectionPipeline, PipelineConfig

FIN, SYN, RST, PSH, ACK = 0x01, 0x02, 0x04, 0x08, 0x10

ADMIN = "192.168.56.10"
WORKSTATION = "192.168.56.20"
SERVER = "192.168.56.102"
WEB = "93.184.216.34"


class BenignScenario(unittest.TestCase):

    def setUp(self):
        self.clock = FakeClock()
        self.pipe = DetectionPipeline(
            PipelineConfig(
                monitored_servers={SERVER},
                outbound_baseline={SERVER: set()},
                listening_baseline={SERVER: {21, 22, 23, 25, 80, 139, 445,
                                             3306, 3632, 5432, 6667}},
                authorized_scanners={ADMIN},
            ),
            clock=self.clock,
        )

    def assert_clean(self, scenario: str):
        """Both halves of the requirement, with a readable failure message."""
        noisy = self.pipe.alerts_above("info")
        self.assertEqual(
            noisy, [],
            f"{scenario}: expected no alert above INFO, got "
            f"{[(a['rule_id'], a['severity']) for a in noisy]}")
        self.assertEqual(
            self.pipe.correlator.incident_count(), 0,
            f"{scenario}: correlation engine manufactured a false incident "
            f"{self.pipe.incidents_opened}")

    def handshake(self, client, server, dport, sport=50000, t=None):
        """A complete, ordinary TCP connection setup."""
        t = self.clock.now if t is None else t
        self.pipe.on_tcp_packet(client, server, sport, dport, SYN, ts=t)
        self.pipe.on_tcp_packet(server, client, dport, sport, SYN | ACK, ts=t + 0.01)
        self.pipe.on_tcp_packet(client, server, sport, dport, ACK, ts=t + 0.02)

    def teardown_conn(self, client, server, dport, sport=50000, t=None):
        t = self.clock.now if t is None else t
        self.pipe.on_tcp_packet(client, server, sport, dport, FIN | ACK, ts=t)
        self.pipe.on_tcp_packet(server, client, dport, sport, FIN | ACK, ts=t + 0.01)
        self.pipe.on_tcp_packet(client, server, sport, dport, ACK, ts=t + 0.02)


class TestLegitimateSubnetScan(BenignScenario):
    """An authorised admin sweeping their own subnet."""

    def test_service_check_across_the_subnet_is_clean(self):
        """One port per host is a health check, not a scan — no rule should care."""
        t = self.clock.now
        for i in range(1, 40):
            host = f"192.168.56.{i}"
            self.handshake(ADMIN, host, 22, sport=40000 + i, t=t + i * 0.5)
            self.teardown_conn(ADMIN, host, 22, sport=40000 + i, t=t + i * 0.5 + 0.1)
        self.assert_clean("subnet service check")

    def test_authorized_scanner_full_scan_is_recorded_at_INFO_only(self):
        """A real nmap from the allowlisted scanner. The activity is still
        recorded — at INFO, with the original severity kept on the alert — so it
        stays auditable without paging anyone."""
        t = self.clock.now
        for i, port in enumerate([21, 22, 23, 25, 80, 110, 139, 443, 445, 3306]):
            self.pipe.on_tcp_packet(ADMIN, SERVER, 41000 + i, port, SYN,
                                    ts=t + i * 0.05)
        self.assert_clean("authorized nmap")
        recon = [a for a in self.pipe.published
                 if a["kill_chain_stage"] == "reconnaissance"]
        self.assertTrue(recon, "the scan must still be recorded, not dropped")
        for a in recon:
            self.assertEqual(a["severity"], "info")
            self.assertEqual(a["details"]["suppressed_by"],
                             "authorized_scanner allowlist")
            self.assertIn("original_severity", a["details"])

    def test_an_UNauthorized_host_running_the_same_scan_still_alerts(self):
        """Proof the allowlist is an allowlist and not a weakened rule."""
        t = self.clock.now
        for i, port in enumerate([21, 22, 23, 25, 80, 110, 139, 443, 445, 3306]):
            self.pipe.on_tcp_packet(WORKSTATION, SERVER, 41000 + i, port, SYN,
                                    ts=t + i * 0.05)
        alerts = self.pipe.alerts_above("info")
        self.assertTrue(alerts, "an unlisted host scanning must alert")
        self.assertIn("TCP_SYN_SCAN", {a["rule_id"] for a in alerts})


class TestFailedLoginNoise(BenignScenario):
    """A user fat-fingering their password. Common, and not an attack."""

    def test_three_failed_ssh_attempts_are_clean(self):
        t = self.clock.now
        for i in range(3):
            self.handshake(WORKSTATION, SERVER, 22, sport=51000 + i,
                           t=t + i * 12)
            self.teardown_conn(WORKSTATION, SERVER, 22, sport=51000 + i,
                               t=t + i * 12 + 4)
        self.assert_clean("three failed SSH logins")

    def test_seven_attempts_stay_under_the_threshold(self):
        """One below the documented min_attempts of 8 — the boundary case."""
        t = self.clock.now
        for i in range(7):
            self.pipe.on_tcp_packet(WORKSTATION, SERVER, 52000 + i, 22, SYN,
                                    ts=t + i * 10)
        self.assert_clean("seven failed logins")

    def test_the_eighth_attempt_does_alert(self):
        """Boundary proof in the other direction: the rule still works."""
        t = self.clock.now
        for i in range(8):
            self.pipe.on_tcp_packet(WORKSTATION, SERVER, 52000 + i, 22, SYN,
                                    ts=t + i * 10)
        rule_ids = {a["rule_id"] for a in self.pipe.alerts_above("info")}
        self.assertIn("BRUTEFORCE_LOW_RATE", rule_ids)


class TestBurstyCurlTraffic(BenignScenario):
    """Rapid repeated HTTPS fetches — a script, a CI job, a page load."""

    def test_sixty_rapid_https_fetches_are_clean(self):
        t = self.clock.now
        for i in range(60):
            self.handshake(WORKSTATION, WEB, 443, sport=53000 + i, t=t + i * 0.1)
            self.teardown_conn(WORKSTATION, WEB, 443, sport=53000 + i,
                               t=t + i * 0.1 + 0.05)
        self.assert_clean("bursty curl to one host")

    def test_bursty_traffic_is_not_mistaken_for_beaconing(self):
        """Regular-ish but fast: the min-interval guard is what saves this."""
        t = self.clock.now
        for i in range(30):
            self.pipe.on_tcp_packet(WORKSTATION, WEB, 54000 + i, 443, SYN,
                                    ts=t + i * 0.5, is_outbound=True)
        self.assert_clean("fast regular fetches")

    def test_polling_a_few_different_api_ports_is_clean(self):
        """Under the 6-port threshold, however often it repeats."""
        t = self.clock.now
        for cycle in range(20):
            for i, port in enumerate([80, 443, 8080]):
                self.handshake(WORKSTATION, WEB, port, sport=55000 + cycle * 3 + i,
                               t=t + cycle * 2 + i * 0.1)
        self.assert_clean("multi-port API polling")


class TestNormalSshSession(BenignScenario):
    """A real interactive session: handshake, banner, keystrokes, clean close."""

    def test_full_ssh_session_is_clean(self):
        t = self.clock.now
        self.handshake(WORKSTATION, SERVER, 22, sport=56000, t=t)
        self.pipe.on_tcp_packet(SERVER, WORKSTATION, 22, 56000, PSH | ACK,
                                payload=b"SSH-2.0-OpenSSH_9.6\r\n", ts=t + 0.1)
        self.pipe.on_tcp_packet(WORKSTATION, SERVER, 56000, 22, PSH | ACK,
                                payload=b"SSH-2.0-OpenSSH_9.6\r\n", ts=t + 0.2)
        for i in range(200):
            self.pipe.on_tcp_packet(WORKSTATION, SERVER, 56000, 22, PSH | ACK,
                                    payload=b"\x00" * 48, ts=t + 1 + i * 0.3)
            self.pipe.on_tcp_packet(SERVER, WORKSTATION, 22, 56000, PSH | ACK,
                                    payload=b"\x00" * 96, ts=t + 1.1 + i * 0.3)
        self.teardown_conn(WORKSTATION, SERVER, 22, sport=56000, t=t + 120)
        self.assert_clean("interactive SSH session")

    def test_server_answering_on_a_baselined_port_is_not_a_bind_shell(self):
        t = self.clock.now
        for port in (21, 22, 80, 445, 3306):
            self.pipe.on_tcp_packet(SERVER, WORKSTATION, port, 57000,
                                    SYN | ACK, ts=t)
        self.assert_clean("server answering on baseline ports")

    def test_long_running_session_does_not_drift_into_an_incident(self):
        """The old engine had no time window, so a long session eventually
        accumulated enough distinct alert types to promote. This asserts the
        window actually bounds accumulation."""
        t = self.clock.now
        for hour in range(8):
            base = t + hour * 3600
            self.handshake(WORKSTATION, SERVER, 22, sport=58000 + hour, t=base)
            for i in range(5):
                self.pipe.on_tcp_packet(WORKSTATION, SERVER, 58000 + hour, 22,
                                        PSH | ACK, payload=b"\x00" * 32,
                                        ts=base + i * 60)
        self.assert_clean("8-hour SSH session")


class TestCorrelationFeedback(unittest.TestCase):
    """Correlation output must not become correlation input."""

    def test_incident_does_not_count_its_own_announcement(self):
        clock = FakeClock()
        pipe = DetectionPipeline(PipelineConfig(), clock=clock)
        t = clock.now
        for i, port in enumerate([21, 22, 23, 25, 80, 445, 3306]):
            pipe.on_tcp_packet(WORKSTATION, SERVER, 60000 + i, port, SYN,
                               ts=t + i * 0.1)
        pipe.on_tcp_packet(WORKSTATION, SERVER, 60100, 21, PSH | ACK,
                           payload=b"USER pwn:)\r\n", ts=t + 10)

        self.assertEqual(pipe.correlator.incident_count(), 1)
        inc = pipe.correlator.incidents()[0]
        self.assertNotIn("CORRELATED_INCIDENT", inc["rule_ids"],
                         "an incident must not list its own announcement as a "
                         "member alert")
        real = [a for a in pipe.published if a["rule_id"] != "CORRELATED_INCIDENT"]
        self.assertEqual(inc["event_count"], len(real),
                         "event_count must equal the real member alerts")


class TestBenignSuiteIntegrity(unittest.TestCase):
    """Meta-test: the benign suite must actually be capable of failing."""

    def test_a_real_attack_through_the_same_pipeline_is_detected(self):
        clock = FakeClock()
        pipe = DetectionPipeline(
            PipelineConfig(monitored_servers={SERVER},
                           outbound_baseline={SERVER: set()}),
            clock=clock)
        t = clock.now
        for i, port in enumerate([21, 22, 23, 25, 80, 445, 3306]):
            pipe.on_tcp_packet(WORKSTATION, SERVER, 60000 + i, port, SYN,
                               ts=t + i * 0.1)
        pipe.on_tcp_packet(WORKSTATION, SERVER, 60100, 21, PSH | ACK,
                           payload=b"USER pwn:)\r\n", ts=t + 10)
        pipe.on_tcp_packet(SERVER, "203.0.113.9", 40000, 4444, SYN, ts=t + 20)

        self.assertTrue(pipe.alerts_above("info"),
                        "the benign assertions would pass trivially if the "
                        "pipeline never alerted at all")
        self.assertGreaterEqual(pipe.correlator.incident_count(), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)

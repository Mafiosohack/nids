"""Spoofed traffic must not be able to exhaust the sensor: 2026-10-31 audit, step 4.

Measured before this change (scratchpad poc_state_growth.py): every spoofed
source left ~1 KB of per-source state behind for good - ddos/syn/udp-amp/
slow-scan/baseline/handshake trackers grew one entry per fake address, because
their pruning only ran when the SAME key came back, and a spoofed source never
does. Ten million free packets is ~10 GB.

The caps that did exist evicted with a full scan or sort on every insert once
full, which turns the memory attack into a CPU one.

The contract:
  * every per-source table has a ceiling
  * evicting is cheap (amortised), so a full table does not slow the sensor
  * the least recently SEEN key goes first, so an active attacker stays hot
  * evictions are counted and shown in /status - pressure is visible, not silent
"""

import os
import random
import time
import unittest

try:
    import scapy.all  # noqa: F401
    SCAPY = True
except ImportError:                                   # pragma: no cover
    SCAPY = False


class TestBoundedDict(unittest.TestCase):
    def make(self, n=3, factory=list):
        from utils.bounded import BoundedDict
        return BoundedDict(factory, max_keys=n, name="t")

    def test_behaves_like_defaultdict(self):
        d = self.make()
        d["a"].append(1)
        self.assertEqual(d["a"], [1])
        self.assertIn("a", d)
        self.assertNotIn("zz", d)
        self.assertEqual(len(d), 1, "membership test must not create keys")

    def test_never_exceeds_its_ceiling(self):
        d = self.make(3)
        for i in range(100):
            d[i].append(i)
        self.assertEqual(len(d), 3)
        self.assertEqual(d.evictions, 97)

    def test_recently_seen_key_survives(self):
        """An attacker who keeps sending stays tracked through a flood of
        one-shot spoofed sources."""
        d = self.make(3)
        d["attacker"].append(0)
        for i in range(50):
            d[f"spoof{i}"].append(i)
            d["attacker"].append(i)          # touched every round
        self.assertIn("attacker", d)
        self.assertEqual(len(d["attacker"]), 51)

    def test_explicit_assignment_and_removal(self):
        d = self.make(2, factory=dict)
        d["x"] = {"v": 1}
        self.assertEqual(d.get("x"), {"v": 1})
        self.assertIsNone(d.get("nope"))
        self.assertEqual(d.pop("x"), {"v": 1})
        self.assertIsNone(d.pop("x", None))
        d["y"]["k"] = 2
        del d["y"]
        self.assertEqual(len(d), 0)

    def test_limit_can_be_lowered_live(self):
        d = self.make(10)
        for i in range(10):
            d[i]
        d.set_limit(4)
        self.assertEqual(len(d), 4)
        self.assertEqual(sorted(d.keys()), [6, 7, 8, 9])

    def test_iteration_is_safe_to_mutate_from_a_snapshot(self):
        d = self.make(5)
        for i in range(5):
            d[i]
        for k, _ in list(d.items()):
            if k % 2:
                del d[k]
        self.assertEqual(sorted(d), [0, 2, 4])


class TestCapsLeaveHeadroom(unittest.TestCase):
    """A full table must not re-sort on every new key."""

    def test_interactive_shell_cap_is_amortised(self):
        from detection.interactive_shell import InteractiveShellDetector
        det = InteractiveShellDetector(max_flows=100)
        for i in range(101):                       # one past the ceiling
            det.open_flow("10.0.0.1", f"10.1.0.{i % 250}", 1000 + i, 4444, ts=float(i))
        self.assertLessEqual(len(det._flows), 100)
        # The next new flow must fit without another eviction pass.
        before = len(det._flows)
        det.open_flow("10.0.0.1", "10.9.9.9", 9999, 4444, ts=999.0)
        self.assertEqual(len(det._flows), before + 1,
                         "evicted again on the very next flow: cost is not amortised")
        # Oldest-first is preserved: flow 0 went, the newest stayed.
        self.assertEqual(min(f.first_ts for f in det._flows.values()) > 0.0, True)
        self.assertIn(999.0, [f.first_ts for f in det._flows.values()])


@unittest.skipUnless(SCAPY, "scapy required")
class TestSpoofedFloodIsBounded(unittest.TestCase):
    LIMIT = 300

    def setUp(self):
        import tempfile
        import main
        self.main = main
        # Alerts raised by the flood are persisted: never into the real DB.
        self.tmp = tempfile.mkdtemp()
        self.saved_db = main.DB_PATH
        main.DB_PATH = main.Path(self.tmp) / "flood.db"
        main.init_db()
        self.saved = main.tracker_limit()
        main.set_tracker_limit(self.LIMIT)

    def tearDown(self):
        import shutil
        self.main.set_tracker_limit(self.saved)
        self.main.PIPELINE.reset()
        self.main.DB_PATH = self.saved_db
        shutil.rmtree(self.tmp, ignore_errors=True)

    def flood(self, n):
        from scapy.all import DNS, DNSQR, IP, TCP, UDP, Ether, Raw
        rnd = random.Random(11)
        def rip():
            return f"{rnd.randint(1, 223)}.{rnd.randint(0, 255)}.{rnd.randint(0, 255)}.{rnd.randint(1, 254)}"
        dsts = ["192.168.56.102", "192.168.56.20"]
        for i in range(n):
            src, dst = rip(), rnd.choice(dsts)
            k = i % 4
            if k == 0:
                p = Ether()/IP(src=src, dst=dst)/TCP(sport=rnd.randint(1024, 65535), dport=rnd.randint(1, 65535), flags="S")
            elif k == 1:
                p = Ether()/IP(src=src, dst=dst)/UDP(sport=rnd.randint(1024, 65535), dport=rnd.choice([123, 1900, 11211]))/Raw(b"x" * 200)
            elif k == 2:
                p = Ether()/IP(src=dst, dst=src)/TCP(sport=rnd.choice([22, 4444]), dport=rnd.randint(1024, 65535), flags="SA")
            else:
                p = Ether()/IP(src=src, dst=dst)/UDP(sport=rnd.randint(1024, 65535), dport=53)/DNS(rd=1, qd=DNSQR(qname=f"{i:x}.example.com"))
            self.main.process_packet(Ether(bytes(p)))

    def test_every_per_source_table_stays_under_its_ceiling(self):
        import contextlib, io
        errors_before = self.main.system_stats.get("packet_errors", 0) or 0
        with contextlib.redirect_stdout(io.StringIO()):
            self.flood(2000)
        sizes = self.main.state_table_sizes()
        over = {k: v for k, v in sizes.items() if v["size"] > v["limit"]}
        self.assertEqual(over, {}, f"tables over their ceiling: {over}")
        # Delta, not total: other tests raise packet errors on purpose.
        self.assertEqual((self.main.system_stats.get("packet_errors", 0) or 0) - errors_before, 0)

    def test_pressure_is_reported(self):
        import contextlib, io
        with contextlib.redirect_stdout(io.StringIO()):
            self.flood(1500)
        ev = self.main.state_evictions()
        self.assertGreater(sum(ev.values()), 0, "a flood past the limit evicted nothing?")


@unittest.skipUnless(SCAPY, "scapy required")
class TestStatusShowsStatePressure(unittest.TestCase):
    def test_status_has_state_tables(self):
        from apiharness import LiveServer
        with LiveServer() as srv:
            body = srv.get("/status", token=srv.login()).json()
        self.assertIn("state_tables", body)
        some = next(iter(body["state_tables"].values()))
        self.assertEqual(set(some), {"size", "limit", "evictions"})


if __name__ == "__main__":
    unittest.main()

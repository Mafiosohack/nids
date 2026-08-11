"""Interactive shell detection by traffic shape.

The rule exists to close one specific hole: a shell on a non-baselined host
calling out to tcp/443, which every port- and baseline-driven rule misses.
`test_reverse_shell_on_443_is_caught` is that case.

The negative tests carry equal weight. This rule flags the shape of a human
typing, and legitimate SSH has exactly that shape, so the tests must prove it
stays quiet when the host sensor confirms a real login — and that it does not
mistake a file transfer or a web session for a shell.
"""

import unittest

from helpers import ATTACKER_EXTERNAL, KALI, METASPLOITABLE, FakeClock

from detection.interactive_shell import InteractiveShellDetector

WORKSTATION = "192.168.56.20"
ADMIN = "192.168.56.10"


class ShellCase(unittest.TestCase):

    def setUp(self):
        self.clock = FakeClock()
        self.det = InteractiveShellDetector(min_packets=40, min_duration_sec=20.0,
                                            clock=self.clock)

    def type_session(self, typist, listener, tport, commands=15,
                     cport=51000, output_bytes=400, keystroke=6,
                     gap=2.0, opened_by=None, t0=None):
        """Simulate someone typing commands and getting output back.

        `opened_by` is whoever sent the SYN — the victim for a reverse shell,
        the attacker for a bind shell.
        """
        t = self.clock.now if t0 is None else t0
        opened_by = opened_by or typist
        other = listener if opened_by == typist else typist
        if opened_by == typist:
            self.det.open_flow(typist, listener, cport, tport, ts=t)
        else:
            self.det.open_flow(listener, typist, cport, tport, ts=t)

        found = None
        for i in range(commands):
            # keystrokes: several tiny packets, one per keypress
            for _ in range(6):
                t += 0.15
                found = self.det.observe_data(typist, listener, cport, tport,
                                              keystroke, ts=t) or found
            # output: a couple of larger packets back
            for _ in range(2):
                t += 0.05
                found = self.det.observe_data(listener, typist, tport, cport,
                                              output_bytes, ts=t) or found
            t += gap
        return found


class TestReverseShellDetection(ShellCase):

    def test_reverse_shell_on_443_is_caught(self):
        """THE GAP. Victim dials out to 443; attacker types into it.

        No baseline for this host, and 443 is an allowed egress port, so
        BEHAVIOR_UNEXPECTED_OUTBOUND and BEHAVIOR_UNCOMMON_EGRESS_PORT both miss
        it. The shape does not.
        """
        f = self.type_session(typist=ATTACKER_EXTERNAL, listener=WORKSTATION,
                              tport=443, opened_by=WORKSTATION)
        self.assertIsNotNone(f, "a shell on 443 must not be invisible")
        self.assertEqual(f.rule_id, "BEHAVIOR_INTERACTIVE_SHELL")
        self.assertEqual(f.severity, "critical")
        self.assertEqual(f.details["shell_port"], 443)
        self.assertGreater(f.details["small_packet_ratio"], 0.6)
        self.assertGreater(f.details["output_input_ratio"], 2.0)
        self.assertEqual(f.missing_discriminators(), [])

    def test_reverse_shell_on_4444_is_caught(self):
        f = self.type_session(typist=ATTACKER_EXTERNAL, listener=WORKSTATION,
                              tport=4444, opened_by=WORKSTATION)
        self.assertIsNotNone(f)
        self.assertEqual(f.severity, "critical")

    def test_bind_shell_is_caught_too(self):
        """Attacker connects IN and types. Opposite direction, same rule."""
        f = self.type_session(typist=KALI, listener=METASPLOITABLE, tport=4444,
                              opened_by=KALI)
        self.assertIsNotNone(f)
        self.assertEqual(f.rule_id, "BEHAVIOR_INTERACTIVE_SHELL")

    def test_the_typist_is_attributed_as_the_source(self):
        """Correlation keys on the actor, so the operator must be the src."""
        f = self.type_session(typist=ATTACKER_EXTERNAL, listener=WORKSTATION,
                              tport=443, opened_by=WORKSTATION)
        self.assertEqual(f.src, ATTACKER_EXTERNAL)
        self.assertEqual(f.dst, WORKSTATION)

    def test_works_without_seeing_the_handshake(self):
        """Sensor started mid-session: no SYN, so no recorded initiator."""
        t = self.clock.now
        found = None
        for i in range(15):
            for _ in range(6):
                t += 0.15
                found = self.det.observe_data(ATTACKER_EXTERNAL, WORKSTATION,
                                              443, 51000, 6, ts=t) or found
            for _ in range(2):
                t += 0.05
                found = self.det.observe_data(WORKSTATION, ATTACKER_EXTERNAL,
                                              51000, 443, 400, ts=t) or found
            t += 2.0
        self.assertIsNotNone(found)


class TestLegitimateSshSuppression(ShellCase):
    """The load-bearing negative case: real SSH looks identical, by nature."""

    def test_ssh_confirmed_by_host_sensor_is_suppressed(self):
        self.det.note_authorized_session(ADMIN, METASPLOITABLE,
                                         ts=self.clock.now)
        f = self.type_session(typist=ADMIN, listener=METASPLOITABLE, tport=22,
                              opened_by=ADMIN)
        self.assertIsNone(
            f, "a host-sensor-confirmed login must not raise a shell alert")

    def test_ssh_without_a_confirmed_login_is_flagged_but_hedged(self):
        """No host sensor evidence: report it, but as missing evidence rather
        than as a verdict, and at medium not critical."""
        f = self.type_session(typist=ADMIN, listener=METASPLOITABLE, tport=22,
                              opened_by=ADMIN)
        self.assertIsNotNone(f)
        self.assertEqual(f.severity, "medium")
        self.assertTrue(f.details["on_known_shell_port"])
        self.assertIn("host_log_sensor.py is not deployed", f.message)

    def test_authorization_expires(self):
        self.det.note_authorized_session(ADMIN, METASPLOITABLE,
                                         ts=self.clock.now)
        self.clock.advance(7200)          # past the 1h TTL
        f = self.type_session(typist=ADMIN, listener=METASPLOITABLE, tport=22,
                              opened_by=ADMIN, t0=self.clock.now)
        self.assertIsNotNone(f, "a stale authorisation must not suppress forever")

    def test_authorization_is_per_host_pair(self):
        """Confirming admin->target must not silence attacker->target."""
        self.det.note_authorized_session(ADMIN, METASPLOITABLE,
                                         ts=self.clock.now)
        f = self.type_session(typist=KALI, listener=METASPLOITABLE, tport=22,
                              opened_by=KALI)
        self.assertIsNotNone(f)


class TestNonShellTrafficIsNotFlagged(ShellCase):

    def test_file_transfer_is_not_a_shell(self):
        """Same packet count, one direction, high throughput."""
        t = self.clock.now
        self.det.open_flow(WORKSTATION, METASPLOITABLE, 52000, 21, ts=t)
        found = None
        for i in range(200):
            t += 0.01
            found = self.det.observe_data(METASPLOITABLE, WORKSTATION, 21, 52000,
                                          1400, ts=t) or found
        self.assertIsNone(found)

    def test_web_browsing_is_not_a_shell(self):
        """Requests are large-ish, connection is short, little turn-taking."""
        t = self.clock.now
        self.det.open_flow(WORKSTATION, "93.184.216.34", 53000, 443, ts=t)
        found = None
        for i in range(30):
            t += 0.05
            found = self.det.observe_data(WORKSTATION, "93.184.216.34", 53000,
                                          443, 620, ts=t) or found
            for _ in range(4):
                t += 0.02
                found = self.det.observe_data("93.184.216.34", WORKSTATION, 443,
                                              53000, 1400, ts=t) or found
        self.assertIsNone(found)

    def test_one_way_stream_is_not_a_shell(self):
        t = self.clock.now
        found = None
        for i in range(120):
            t += 0.5
            found = self.det.observe_data(WORKSTATION, ATTACKER_EXTERNAL, 54000,
                                          443, 8, ts=t) or found
        self.assertIsNone(found, "no return traffic means no session")

    def test_short_session_is_not_enough_evidence(self):
        f = self.type_session(typist=KALI, listener=METASPLOITABLE, tport=4444,
                              commands=2, gap=0.5)
        self.assertIsNone(f)

    def test_too_few_packets_is_not_enough_evidence(self):
        det = InteractiveShellDetector(min_packets=200, min_duration_sec=20.0,
                                       clock=self.clock)
        self.det = det
        f = self.type_session(typist=KALI, listener=METASPLOITABLE, tport=4444)
        self.assertIsNone(f)

    def test_beacon_style_polling_is_not_a_shell(self):
        """Regular small check-ins with small replies: that is BEHAVIOR_C2_BEACON's
        job, and the output ratio keeps this rule off it."""
        t = self.clock.now
        self.det.open_flow(METASPLOITABLE, ATTACKER_EXTERNAL, 55000, 443, ts=t)
        found = None
        for i in range(60):
            t += 30.0
            found = self.det.observe_data(METASPLOITABLE, ATTACKER_EXTERNAL,
                                          55000, 443, 12, ts=t) or found
            t += 0.1
            found = self.det.observe_data(ATTACKER_EXTERNAL, METASPLOITABLE,
                                          443, 55000, 10, ts=t) or found
        self.assertIsNone(found)


class TestHousekeeping(ShellCase):

    def test_alert_fires_once_per_session(self):
        first = self.type_session(typist=ATTACKER_EXTERNAL, listener=WORKSTATION,
                                  tport=443, opened_by=WORKSTATION)
        self.assertIsNotNone(first)
        again = self.type_session(typist=ATTACKER_EXTERNAL, listener=WORKSTATION,
                                  tport=443, opened_by=WORKSTATION, cport=51001,
                                  t0=self.clock.now + 60)
        self.assertIsNone(again, "cooldown must stop per-packet alert spam")

    def test_flow_table_is_bounded(self):
        det = InteractiveShellDetector(max_flows=50, clock=self.clock)
        for i in range(400):
            det.observe_data(f"10.5.{i // 254}.{i % 254}", WORKSTATION,
                             40000 + i, 443, 10, ts=self.clock.now + i)
        self.assertLessEqual(det.tracked_flows, 50)

    def test_idle_flows_are_swept(self):
        self.det.observe_data(KALI, METASPLOITABLE, 56000, 4444, 10,
                              ts=self.clock.now)
        self.assertEqual(self.det.tracked_flows, 1)
        self.clock.advance(1000)
        self.assertEqual(self.det.sweep(self.clock.now), 1)
        self.assertEqual(self.det.tracked_flows, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)

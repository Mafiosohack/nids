"""Per-host profile store.

The load-bearing tests here are the ones about the store's own honesty. A
baseline that reports "first contact ever" from a profile that is four minutes
old, or after it has quietly evicted the peer it is asked about, is worse than
no baseline at all - it manufactures confident-looking evidence out of when the
sensor happened to start. So `confident` is asserted at least as hard as the
facts it qualifies.
"""

import unittest

from helpers import ATTACKER_EXTERNAL, KALI, METASPLOITABLE, UBUNTU, FakeClock

from detection.host_profile import (ESTABLISHED, LEARNING, HostProfileStore,
                                    Volume)

FILE_SERVER = "192.168.56.120"


def fixed_hour(hour):
    """An hour-of-day function that ignores the clock, for deterministic tests."""
    return lambda ts: hour


class ProfileTestCase(unittest.TestCase):
    """A store whose warm-up gates are small enough to cross inside a test."""

    def setUp(self):
        self.clock = FakeClock()
        self.store = HostProfileStore(
            min_observations=5, min_age_sec=60.0, period_sec=3600.0,
            clock=self.clock)

    def establish(self, host, peer=METASPLOITABLE, port=443, hits=6, gap=15.0):
        """Push `host` past both warm-up gates against one boring peer."""
        for _ in range(hits):
            self.store.observe_connection(host, peer, port, ts=self.clock.now)
            self.clock.advance(gap)
        return self.store.profile(host)


class TestWarmUp(ProfileTestCase):
    """Nothing the store says is trustworthy until it has watched for a while."""

    def test_a_brand_new_profile_is_learning_and_says_so(self):
        n = self.store.observe_connection(KALI, METASPLOITABLE, 445)
        self.assertEqual(n.profile_state, LEARNING)
        self.assertTrue(n.new_peer)          # the fact is right ...
        self.assertFalse(n.confident)        # ... but it means nothing yet
        self.assertIn("still learning", n.caveat)

    def test_both_gates_must_pass(self):
        """Count alone is not enough, and age alone is not enough."""
        # Chatty host: clears the count in one second, still too young.
        for _ in range(20):
            self.store.observe_connection(KALI, METASPLOITABLE, 443,
                                          ts=self.clock.now)
        self.assertFalse(self.store.is_established(KALI, ts=self.clock.now))

        # Quiet host: old enough, but has barely been seen.
        self.store.observe_connection(UBUNTU, METASPLOITABLE, 443,
                                      ts=self.clock.now)
        self.clock.advance(3600)
        self.store.observe_connection(UBUNTU, METASPLOITABLE, 443,
                                      ts=self.clock.now)
        self.assertFalse(self.store.is_established(UBUNTU, ts=self.clock.now))

    def test_established_after_both_gates(self):
        self.establish(KALI)
        self.assertTrue(self.store.is_established(KALI, ts=self.clock.now))

    def test_details_carry_the_evidence_grade(self):
        n = self.store.observe_connection(KALI, METASPLOITABLE, 445)
        self.assertEqual(n.as_details()["profile_evidence"], "degraded")
        self.establish(UBUNTU)
        n = self.store.observe_connection(UBUNTU, FILE_SERVER, 445,
                                          ts=self.clock.now)
        self.assertEqual(n.as_details()["profile_evidence"], "confident")


class TestConnectionNovelty(ProfileTestCase):

    def test_first_contact_with_a_new_peer(self):
        """The single deliberate hop LATERAL_THRESHOLD=6 structurally misses."""
        self.establish(KALI)
        n = self.store.observe_connection(KALI, FILE_SERVER, 445,
                                          ts=self.clock.now)
        self.assertTrue(n.new_peer)
        self.assertTrue(n.confident)
        self.assertIsNone(n.caveat)
        self.assertEqual(n.dst_port, 445)
        self.assertEqual(n.peer_connections, 0)

    def test_a_known_peer_on_a_new_port_is_still_novel(self):
        """Peer-level novelty alone would miss the interesting half of this:
        A has talked to B for months, but never on 445."""
        self.establish(KALI, peer=FILE_SERVER, port=443)
        n = self.store.observe_connection(KALI, FILE_SERVER, 445,
                                          ts=self.clock.now)
        self.assertFalse(n.new_peer)
        self.assertTrue(n.new_port_for_peer)
        self.assertTrue(n.new_port_for_host)
        self.assertTrue(n.confident)

    def test_routine_traffic_is_not_novel(self):
        self.establish(KALI, peer=METASPLOITABLE, port=443)
        n = self.store.observe_connection(KALI, METASPLOITABLE, 443,
                                          ts=self.clock.now)
        self.assertFalse(n.new_peer)
        self.assertFalse(n.new_port_for_peer)
        self.assertFalse(n.new_port_for_host)
        self.assertGreater(n.peer_connections, 0)

    def test_novelty_is_judged_before_the_packet_is_recorded(self):
        """Otherwise every connection would be 'already known' - by itself."""
        self.establish(KALI)
        first = self.store.observe_connection(KALI, FILE_SERVER, 445,
                                              ts=self.clock.now)
        second = self.store.observe_connection(KALI, FILE_SERVER, 445,
                                               ts=self.clock.now)
        self.assertTrue(first.new_peer)
        self.assertFalse(second.new_peer)

    def test_the_target_is_profiled_too_but_only_the_actor_gets_an_answer(self):
        n = self.store.observe_connection(KALI, METASPLOITABLE, 445)
        self.assertEqual(n.host, KALI)
        target = self.store.profile(METASPLOITABLE)
        self.assertIn(445, target.server_ports)      # a port it serves ...
        self.assertNotIn(445, target.client_ports)   # ... not one it dialled
        self.assertEqual(target.peers[KALI].connections_in, 1)
        self.assertEqual(target.peers[KALI].connections_out, 0)

    def test_host_filter_excludes_the_internet(self):
        store = HostProfileStore(host_filter=lambda ip: ip.startswith("192.168."),
                                 clock=self.clock)
        self.assertIsNone(store.observe_connection(ATTACKER_EXTERNAL, KALI, 443))
        self.assertIsNone(store.profile(ATTACKER_EXTERNAL))
        self.assertIsNotNone(store.profile(KALI))    # the internal end is kept


class TestEvictionHonesty(ProfileTestCase):
    """Once history has been dropped, 'never seen before' is not a claim the
    store is entitled to make."""

    def test_peer_eviction_degrades_first_contact_claims(self):
        store = HostProfileStore(min_observations=2, min_age_sec=1.0,
                                 max_peers_per_host=4, clock=self.clock)
        for i in range(10):
            store.observe_connection(KALI, "10.0.0.%d" % i, 443, ts=self.clock.now)
            self.clock.advance(5)
        p = store.profile(KALI)
        self.assertEqual(len(p.peers), 4)
        self.assertGreater(p.peers_evicted, 0)

        n = store.observe_connection(KALI, FILE_SERVER, 445, ts=self.clock.now)
        self.assertEqual(n.profile_state, ESTABLISHED)
        self.assertTrue(n.new_peer)
        self.assertFalse(n.confident)
        self.assertIn("evicted", n.caveat)

    def test_eviction_does_not_degrade_answers_about_known_peers(self):
        """Forgetting peers costs the store first-contact claims, nothing more."""
        store = HostProfileStore(min_observations=2, min_age_sec=1.0,
                                 max_peers_per_host=4, clock=self.clock)
        for i in range(10):
            store.observe_connection(KALI, "10.0.0.%d" % i, 443, ts=self.clock.now)
            self.clock.advance(5)
        n = store.observe_connection(KALI, "10.0.0.9", 443, ts=self.clock.now)
        self.assertFalse(n.new_peer)
        self.assertTrue(n.confident)

    def test_hosts_are_capped_and_the_least_recent_goes_first(self):
        store = HostProfileStore(max_hosts=3, clock=self.clock)
        for i in range(6):
            store.observe_connection("10.0.0.%d" % i, METASPLOITABLE, 443,
                                     ts=self.clock.now)
            self.clock.advance(1)
        # Seven profiles were created, not six: the target endpoint of every
        # connection is profiled too, and being touched every iteration it is
        # also the one guaranteed to survive.
        self.assertEqual(store.stats()["hosts"], 3)
        self.assertEqual(store.stats()["hosts_evicted"], 4)
        self.assertIsNotNone(store.profile(METASPLOITABLE))
        self.assertIsNone(store.profile("10.0.0.0"))
        self.assertIsNotNone(store.profile("10.0.0.5"))

    def test_stale_hosts_are_swept_without_a_maintenance_tick(self):
        store = HostProfileStore(host_ttl_sec=100.0, sweep_interval_sec=10.0,
                                 clock=self.clock)
        store.observe_connection(KALI, METASPLOITABLE, 443, ts=self.clock.now)
        self.clock.advance(500)
        # Any observe call drives the sweep; nothing external has to remember to.
        store.observe_connection(UBUNTU, METASPLOITABLE, 443, ts=self.clock.now)
        self.assertIsNone(store.profile(KALI))
        self.assertIsNotNone(store.profile(UBUNTU))


class TestVolumeAndDirection(ProfileTestCase):

    def test_bytes_are_attributed_in_the_right_direction(self):
        self.store.observe_bytes(KALI, METASPLOITABLE, 1500, ts=self.clock.now)
        self.assertEqual(self.store.profile(KALI).lifetime.sent, 1500)
        self.assertEqual(self.store.profile(KALI).lifetime.recv, 0)
        self.assertEqual(self.store.profile(METASPLOITABLE).lifetime.recv, 1500)
        self.assertEqual(self.store.profile(METASPLOITABLE).lifetime.sent, 0)

    def test_ratio_floor_avoids_infinity(self):
        self.assertEqual(Volume(sent=5000, recv=0).ratio, 5000.0)
        self.assertEqual(Volume(sent=100, recv=10).ratio, 10.0)
        self.assertEqual(Volume().ratio, 0.0)

    def test_ratio_flip_against_the_hosts_own_baseline(self):
        """A workstation that normally pulls, and is now pushing. No absolute
        threshold sees this; the fold change does."""
        self.establish(KALI)
        for _ in range(4):                       # a period of ordinary browsing
            self.store.observe_bytes(METASPLOITABLE, KALI, 1_000_000,
                                     ts=self.clock.now)
            self.store.observe_bytes(KALI, METASPLOITABLE, 50_000,
                                     ts=self.clock.now)
            self.clock.advance(60)
        self.clock.advance(3600)                 # next period
        self.store.observe_bytes(KALI, ATTACKER_EXTERNAL, 4_000_000,
                                 ts=self.clock.now)

        shift = self.store.ratio_shift(KALI, ts=self.clock.now)
        self.assertEqual(shift.baseline_periods, 1)
        self.assertAlmostEqual(shift.baseline_ratio, 0.05, places=4)
        self.assertTrue(shift.confident)
        self.assertGreater(shift.fold_change, 50)

    def test_ratio_shift_is_not_confident_without_a_completed_period(self):
        self.establish(KALI)
        self.store.observe_bytes(KALI, METASPLOITABLE, 10_000, ts=self.clock.now)
        shift = self.store.ratio_shift(KALI, ts=self.clock.now)
        self.assertEqual(shift.baseline_periods, 0)
        self.assertIsNone(shift.baseline_ratio)
        self.assertFalse(shift.confident)
        self.assertIn("no completed period", shift.caveat)

    def test_slow_drip_accumulates_where_a_window_would_forget(self):
        """2 MB every ten minutes trips no 50 MB / 300 s window, and is 288 MB
        by the end of the day."""
        store = HostProfileStore(min_observations=1, min_age_sec=0.0,
                                 period_sec=86400.0, clock=self.clock)
        self.clock.now = 86400.0 * 12       # start on a period boundary
        store.observe_connection(KALI, ATTACKER_EXTERNAL, 443, ts=self.clock.now)
        last_ts = self.clock.now
        for _ in range(144):
            last_ts = self.clock.now
            store.observe_bytes(KALI, ATTACKER_EXTERNAL, 2_000_000, ts=last_ts)
            self.clock.advance(600)
        # Asked WITHIN the period. One second later the day has rolled and the
        # honest answer is zero again, which is the point of a period bucket.
        vol = store.period_volume(KALI, peer=ATTACKER_EXTERNAL, ts=last_ts)
        self.assertEqual(vol.sent_bytes, 288_000_000)
        self.assertEqual(vol.peer, ATTACKER_EXTERNAL)

    def test_period_rollover_moves_current_into_baseline(self):
        store = HostProfileStore(period_sec=100.0, clock=self.clock)
        store.observe_bytes(KALI, METASPLOITABLE, 400, ts=self.clock.now)
        self.clock.advance(150)
        store.observe_bytes(KALI, METASPLOITABLE, 30, ts=self.clock.now)
        p = store.profile(KALI)
        self.assertEqual(p.current.sent, 30)
        self.assertEqual(p.baseline.sent, 400)
        self.assertEqual(p.baseline_periods, 1)
        self.assertEqual(p.lifetime.sent, 430)

    def test_idle_periods_do_not_dilute_the_baseline(self):
        store = HostProfileStore(period_sec=100.0, clock=self.clock)
        store.observe_bytes(KALI, METASPLOITABLE, 400, ts=self.clock.now)
        self.clock.advance(10_000)               # a hundred silent periods
        store.observe_bytes(KALI, METASPLOITABLE, 30, ts=self.clock.now)
        self.assertEqual(store.profile(KALI).baseline_periods, 1)


class TestHourNovelty(ProfileTestCase):

    def test_a_never_before_active_hour(self):
        store = HostProfileStore(min_observations=3, min_age_sec=1.0,
                                 period_sec=100.0, hour_of_day=fixed_hour(14),
                                 clock=self.clock)
        for _ in range(5):
            store.observe_connection(KALI, METASPLOITABLE, 443, ts=self.clock.now)
            store.observe_bytes(KALI, METASPLOITABLE, 100, ts=self.clock.now)
            self.clock.advance(60)               # crosses a period boundary

        at_2pm = store.hour_novelty(KALI, ts=self.clock.now)
        self.assertFalse(at_2pm.new_hour)
        self.assertTrue(at_2pm.confident)

        store.hour_of_day = fixed_hour(3)
        at_3am = store.hour_novelty(KALI, ts=self.clock.now)
        self.assertTrue(at_3am.new_hour)
        self.assertTrue(at_3am.confident)
        self.assertEqual(at_3am.active_hours, 1)

    def test_no_hour_verdict_before_a_full_cycle(self):
        """Otherwise 'never active at 03:00' is a fact about the sensor's
        uptime, not about the host."""
        self.establish(KALI)
        answer = self.store.hour_novelty(KALI, ts=self.clock.now)
        self.assertFalse(answer.confident)
        self.assertIn("completed period", answer.caveat)


class TestFingerprintNovelty(ProfileTestCase):

    def test_new_fingerprint_for_this_host(self):
        self.establish(KALI)
        first = self.store.observe_tls_fingerprint(KALI, "ja3-firefox",
                                                   ts=self.clock.now)
        self.assertTrue(first.new_fingerprint)
        again = self.store.observe_tls_fingerprint(KALI, "ja3-firefox",
                                                   ts=self.clock.now)
        self.assertFalse(again.new_fingerprint)
        self.assertEqual(again.occurrences, 1)
        other = self.store.observe_tls_fingerprint(KALI, "ja3-implant",
                                                   ts=self.clock.now)
        self.assertTrue(other.new_fingerprint)
        self.assertTrue(other.confident)

    def test_empty_fingerprint_is_ignored(self):
        self.assertIsNone(self.store.observe_tls_fingerprint(KALI, ""))


class TestPersistence(ProfileTestCase):
    """A baseline that dies with the process is not a baseline."""

    def test_round_trip_preserves_state_and_history(self):
        self.establish(KALI, peer=METASPLOITABLE, port=443)
        self.store.observe_bytes(KALI, METASPLOITABLE, 1234, ts=self.clock.now)
        blob = self.store.to_dict()

        restored = HostProfileStore(min_observations=5, min_age_sec=60.0,
                                    period_sec=3600.0, clock=self.clock)
        self.assertEqual(restored.load(blob), 2)   # KALI and its peer
        self.assertTrue(restored.is_established(KALI, ts=self.clock.now))

        n = restored.observe_connection(KALI, METASPLOITABLE, 443,
                                        ts=self.clock.now)
        self.assertFalse(n.new_peer)               # history survived the restart
        self.assertTrue(n.confident)
        self.assertEqual(restored.profile(KALI).lifetime.sent, 1234)

    def test_serialization_is_json_safe(self):
        import json
        self.establish(KALI)
        self.store.observe_bytes(KALI, METASPLOITABLE, 10, ts=self.clock.now)
        self.store.observe_tls_fingerprint(KALI, "ja3-x", ts=self.clock.now)
        json.dumps(self.store.to_dict())           # raises if it is not
        json.dumps(self.store.snapshot(KALI, ts=self.clock.now))
        json.dumps(self.store.summary(ts=self.clock.now))

    def test_a_corrupt_profile_file_does_not_stop_the_sensor(self):
        store = HostProfileStore(clock=self.clock)
        loaded = store.load({"hosts": {"1.2.3.4": {"nope": True},
                                       "5.6.7.8": None}})
        self.assertEqual(loaded, 0)
        self.assertEqual(store.stats()["hosts"], 0)

    def test_load_of_empty_or_missing_data_is_a_no_op(self):
        store = HostProfileStore(clock=self.clock)
        self.assertEqual(store.load({}), 0)
        self.assertEqual(store.load(None), 0)


class TestPipelineWiring(unittest.TestCase):
    """The store must be fed by the same packet path production uses, and must
    not change what the pipeline alerts on."""

    def setUp(self):
        from detection.pipeline import DetectionPipeline, PipelineConfig
        self.clock = FakeClock()
        self.pipeline = DetectionPipeline(
            PipelineConfig(profile_min_observations=3, profile_min_age_sec=10.0),
            clock=self.clock)

    def test_a_syn_populates_the_profile(self):
        self.pipeline.on_tcp_packet(KALI, METASPLOITABLE, 51000, 445, 0x02,
                                    ts=self.clock.now)
        p = self.pipeline.profiles.profile(KALI)
        self.assertIsNotNone(p)
        self.assertEqual(p.connections, 1)
        self.assertIn(445, p.client_ports)

    def test_payload_bytes_reach_the_profile_in_both_directions(self):
        self.pipeline.on_tcp_packet(KALI, METASPLOITABLE, 51000, 445, 0x18,
                                    payload=b"x" * 40, ts=self.clock.now)
        self.pipeline.on_tcp_packet(METASPLOITABLE, KALI, 445, 51000, 0x18,
                                    payload=b"y" * 900, ts=self.clock.now)
        self.assertEqual(self.pipeline.profiles.profile(KALI).lifetime.sent, 40)
        self.assertEqual(self.pipeline.profiles.profile(KALI).lifetime.recv, 900)

    def test_true_payload_length_is_used_not_the_truncated_copy(self):
        """main.py truncates the payload for signature matching; the profile
        must bill the real size or every volume figure is wrong."""
        self.pipeline.on_tcp_packet(KALI, METASPLOITABLE, 51000, 443, 0x18,
                                    payload=b"x" * 100, payload_len=64000,
                                    ts=self.clock.now)
        self.assertEqual(self.pipeline.profiles.profile(KALI).lifetime.sent, 64000)

    def test_feeding_the_store_emits_no_alerts(self):
        for i in range(10):
            self.pipeline.on_tcp_packet(KALI, METASPLOITABLE, 51000 + i, 443,
                                        0x02, ts=self.clock.now)
            self.clock.advance(30)
        self.assertEqual(self.pipeline.alerts_above("info"), [])

    def test_reset_clears_the_profiles(self):
        self.pipeline.on_tcp_packet(KALI, METASPLOITABLE, 51000, 445, 0x02,
                                    ts=self.clock.now)
        self.pipeline.reset()
        self.assertIsNone(self.pipeline.profiles.profile(KALI))


if __name__ == "__main__":
    unittest.main()

"""Regression tests for the 2026-10-31 audit, step 2: login, sessions, audit.

Usability is part of the spec here, not an afterthought: a legitimate operator
who mistypes a password a few times must never notice the throttle, and an
attacker hammering the admin account must not be able to lock the real admin
out for more than a short wait.
"""

import hashlib
import sqlite3
import tempfile
import time
import unittest

from helpers import FakeClock

try:
    import scapy.all  # noqa: F401  (main.py imports it; mirrors other suites)
    SCAPY = True
except ImportError:                                   # pragma: no cover
    SCAPY = False

from utils.auth_throttle import LoginThrottle


# ── Throttle policy (pure) ────────────────────────────────────────────────────
class TestLoginThrottle(unittest.TestCase):
    def setUp(self):
        self.clock = FakeClock()
        self.t = LoginThrottle(clock=self.clock)

    def fail(self, n, ip="10.0.0.1", user="admin"):
        for _ in range(n):
            self.t.failure(ip, user)

    def test_a_few_typos_cost_nothing(self):
        """Usability: five wrong passwords in a row is a bad morning, not an attack."""
        self.fail(5)
        self.assertEqual(self.t.retry_after("10.0.0.1", "admin"), 0)

    def test_sustained_guessing_from_one_ip_is_slowed(self):
        self.fail(6)
        first = self.t.retry_after("10.0.0.1", "admin")
        self.assertGreater(first, 0)
        self.fail(4)
        self.assertGreater(self.t.retry_after("10.0.0.1", "admin"), first)

    def test_ip_delay_is_capped(self):
        self.fail(60)
        self.assertLessEqual(self.t.retry_after("10.0.0.1", "admin"), 300)

    def test_the_delay_elapses(self):
        self.fail(6)
        self.clock.advance(self.t.retry_after("10.0.0.1", "admin") + 0.01)
        self.assertEqual(self.t.retry_after("10.0.0.1", "admin"), 0)

    def test_attacker_cannot_lock_the_admin_out_for_long(self):
        """Many IPs failing against one username: the per-USER delay is short,
        so the real admin, from a clean IP, waits seconds at worst."""
        for i in range(200):
            self.t.failure(f"10.9.{i // 250}.{i % 250}", "admin")
        self.assertLessEqual(self.t.retry_after("192.168.56.10", "admin"), 30)

    def test_success_clears_the_ip(self):
        self.fail(8)
        self.t.success("10.0.0.1", "admin")
        self.assertEqual(self.t.retry_after("10.0.0.1", "admin"), 0)

    def test_old_failures_are_forgotten(self):
        self.fail(8)
        self.clock.advance(3600)
        self.assertEqual(self.t.retry_after("10.0.0.1", "admin"), 0)

    def test_memory_is_bounded(self):
        """A spray from spoofed/rotating sources must not grow without limit."""
        t = LoginThrottle(clock=self.clock, max_keys=100)
        for i in range(5000):
            t.failure(f"10.{i // 65536}.{(i // 256) % 256}.{i % 256}", f"u{i}")
        self.assertLessEqual(t.tracked_keys, 200)


# ── API behaviour ─────────────────────────────────────────────────────────────
@unittest.skipUnless(SCAPY, "main.py needs scapy")
class TestLoginAndSessions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from apiharness import LiveServer
        cls.srv = LiveServer().__enter__()
        cls.main = cls.srv.main

    @classmethod
    def tearDownClass(cls):
        cls.srv.__exit__(None, None, None)

    def setUp(self):
        self.main.LOGIN_THROTTLE.reset()

    def test_throttled_login_gets_429_even_with_the_right_password(self):
        """Breaks if the throttle is checked AFTER the password: a correct guess
        would then still succeed, and the throttle only slows wrong answers."""
        for _ in range(6):
            self.srv.post("/auth/login", {"username": "admin", "password": "nope"})
        r = self.srv.post("/auth/login", {"username": "admin",
                                          "password": "harness-admin-password"})
        self.assertEqual(r.status, 429)
        self.assertIn("retry-after", {k.lower() for k in r.headers})

    def test_unknown_user_costs_the_same_as_a_wrong_password(self):
        """Breaks if a missing user returns before the PBKDF2 work: the timing
        difference (~10ms vs ~95ms) names which accounts exist."""
        def t(user):
            s = time.perf_counter()
            self.srv.post("/auth/login", {"username": user, "password": "x"})
            return time.perf_counter() - s
        unknown = min(t("no-such-user") for _ in range(3))
        self.main.LOGIN_THROTTLE.reset()
        known = min(t("admin") for _ in range(3))
        self.assertGreater(unknown, known * 0.5)

    def test_tokens_are_not_stored_in_plaintext(self):
        """Breaks if sessions.token holds the bearer token itself: anyone who can
        read the DB file (or a backup) gets live sessions."""
        tok = self.srv.login()
        with sqlite3.connect(str(self.main.DB_PATH)) as c:
            stored = [r[0] for r in c.execute("SELECT token FROM sessions")]
        self.assertNotIn(tok, stored)
        self.assertIn(hashlib.sha256(tok.encode()).hexdigest(), stored)
        self.assertEqual(self.srv.get("/status", token=tok).status, 200)

    def test_password_change_ends_every_other_session(self):
        """Breaks if a stolen token survives the victim changing the password."""
        stolen = self.srv.login()
        mine = self.srv.login()
        r = self.srv.post("/auth/change-password",
                          {"current_password": "harness-admin-password",
                           "new_password": "harness-admin-password-2"}, token=mine)
        self.assertEqual(r.status, 200)
        self.assertEqual(self.srv.get("/status", token=stolen).status, 401)
        self.assertEqual(self.srv.get("/status", token=mine).status, 200)
        # put it back for the other tests
        self.srv.post("/auth/change-password",
                      {"current_password": "harness-admin-password-2",
                       "new_password": "harness-admin-password"}, token=mine)


@unittest.skipUnless(SCAPY, "main.py needs scapy")
class TestAuditLog(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from apiharness import LiveServer
        cls.srv = LiveServer().__enter__()
        cls.main = cls.srv.main

    @classmethod
    def tearDownClass(cls):
        cls.srv.__exit__(None, None, None)

    def actions(self, token):
        r = self.srv.get("/audit?limit=200", token=token)
        self.assertEqual(r.status, 200)
        return [(e["actor"], e["action"], e["target"]) for e in r.json()["entries"]]

    def test_privileged_actions_are_recorded(self):
        """Breaks if a privileged action stops being written to the audit log:
        with a stolen admin token, this is the only record of what it did."""
        self.main.LOGIN_THROTTLE.reset()
        self.srv.post("/auth/login", {"username": "admin", "password": "wrong"})
        tok = self.srv.login()
        self.srv.post("/users", {"username": "ana", "password": "ana-password-1",
                                 "role": "analyst"}, token=tok)
        self.srv.patch("/users/ana/role", {"role": "viewer"}, token=tok)
        self.srv.post("/suppressions", {"rule_id": "TCP_SYN_SCAN",
                                        "src": "192.168.56.9", "note": "scanner"},
                      token=tok)
        self.srv.delete("/alerts", token=tok)
        got = self.actions(tok)
        for want in [("admin", "login_failure", "admin"),
                     ("admin", "login", "admin"),
                     ("admin", "user_create", "ana"),
                     ("admin", "role_change", "ana"),
                     ("admin", "suppression_add", "TCP_SYN_SCAN from 192.168.56.9"),
                     ("admin", "alerts_clear", "")]:
            self.assertIn(want, got)

    def test_audit_log_is_admin_only(self):
        self.main.LOGIN_THROTTLE.reset()
        tok = self.srv.login()
        self.srv.post("/users", {"username": "vic", "password": "vic-password-1",
                                 "role": "analyst"}, token=tok)
        analyst = self.srv.login("vic", "vic-password-1")
        self.assertEqual(self.srv.get("/audit", token=analyst).status, 403)


@unittest.skipUnless(SCAPY, "main.py needs scapy")
class TestSessionTableUpgrade(unittest.TestCase):
    def test_plaintext_sessions_from_an_old_db_are_dropped_once(self):
        """An existing DB holds raw tokens. They cannot be verified against
        hashes, so they are cleared once on upgrade (everyone logs in again)."""
        import main
        tmp = tempfile.mkdtemp()
        saved = main.DB_PATH
        try:
            main.DB_PATH = main.Path(tmp) / "old.db"
            with sqlite3.connect(str(main.DB_PATH)) as c:
                c.execute("CREATE TABLE sessions (token TEXT PRIMARY KEY, username "
                          "TEXT NOT NULL, role TEXT NOT NULL, expires_at TEXT NOT "
                          "NULL, created_at TEXT NOT NULL)")
                c.execute("INSERT INTO sessions VALUES ('raw','admin','admin',"
                          "'2999-01-01','2026-01-01')")
            main.ADMIN_SEED_PASSWORD = "x-test-password"
            main.init_db()
            with sqlite3.connect(str(main.DB_PATH)) as c:
                self.assertEqual(c.execute("SELECT COUNT(*) FROM sessions")
                                  .fetchone()[0], 0)
            main.init_db()       # idempotent: a second start must not log people out
            tok = main.db_create_session("admin", "admin")
            main.init_db()
            self.assertIsNotNone(main.db_get_session(tok))
        finally:
            main.DB_PATH = saved


if __name__ == "__main__":
    unittest.main()


# ── Sensor identity: each sensor may only say what it exists to say ──────────
import json as _json       # noqa: E402
import os as _os           # noqa: E402

from utils import sensor_keys as sk   # noqa: E402


class TestSensorKeyFile(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.path = _os.path.join(self.tmp, "sensor_keys.json")
        self._env = _os.environ.pop("NIDS_SENSOR_KEY", None)

    def tearDown(self):
        if self._env is not None:
            _os.environ["NIDS_SENSOR_KEY"] = self._env

    def test_first_run_creates_one_distinct_key_per_sensor(self):
        """Usability: no config to write. Security: no shared key to leak."""
        keys = sk.load_or_create(self.path)
        self.assertEqual(set(keys), set(sk.SENSOR_SCOPES))
        self.assertEqual(len(set(keys.values())), len(keys))
        self.assertTrue(all(len(k) >= 32 for k in keys.values()))
        self.assertEqual(sk.load_or_create(self.path), keys, "must be stable")

    @unittest.skipIf(_os.name == "nt", "POSIX permissions")
    def test_key_file_is_private(self):
        sk.load_or_create(self.path)
        self.assertEqual(_os.stat(self.path).st_mode & 0o077, 0)

    def test_a_local_sensor_finds_its_key_with_no_configuration(self):
        keys = sk.load_or_create(self.path)
        self.assertEqual(sk.key_for("host", self.path), keys["host"])

    def test_env_var_overrides_the_file(self):
        """Usability: a sensor on another machine has no key file."""
        _os.environ["NIDS_SENSOR_KEY"] = "remote-sensor-key-from-env-0123456789"
        self.assertEqual(sk.key_for("ml", self.path),
                         "remote-sensor-key-from-env-0123456789")
        del _os.environ["NIDS_SENSOR_KEY"]

    def test_missing_key_explains_the_fix(self):
        with self.assertRaises(sk.SensorKeyError) as ctx:
            sk.key_for("ml", _os.path.join(self.tmp, "absent.json"))
        self.assertIn("NIDS_SENSOR_KEY", str(ctx.exception))
        self.assertIn("sensor_keys", str(ctx.exception))


@unittest.skipUnless(SCAPY, "main.py needs scapy")
class TestSensorScopes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from apiharness import LiveServer
        cls.srv = LiveServer().__enter__()
        cls.main = cls.srv.main
        cls.keys = sk.load_or_create(str(cls.main.SENSOR_KEYS_PATH))

    @classmethod
    def tearDownClass(cls):
        cls.srv.__exit__(None, None, None)

    def setUp(self):
        self.main.PIPELINE.reset()
        self.main.HOST_ADDRESSES.clear()

    def send(self, sensor_or_key, **body):
        key = self.keys.get(sensor_or_key, sensor_or_key)
        payload = {"src": "192.168.56.50", "dst": "192.168.56.101",
                   "alert_type": "ML Anomaly", "severity": "high", "message": "m"}
        payload.update(body)
        return self.srv.post("/alert", payload, headers={"X-Sensor-Key": key})

    def test_public_default_key_is_refused(self):
        """Breaks if the key printed in this repository works again."""
        self.assertEqual(self.send(sk.PUBLIC_DEFAULT_KEY).status, 401)

    def test_each_sensor_reports_its_own_alerts(self):
        """Usability: every bundled sensor's real alert types still go through."""
        for sensor, types in sk.SENSOR_SCOPES.items():
            for t in types:
                r = self.send(sensor, alert_type=t)
                self.assertEqual(r.status, 200, f"{sensor} / {t}: {r.body!r}")

    def test_ml_key_cannot_vouch_for_a_login(self):
        """Breaks if any key but the host sensor's can send SSH Login: that is
        the 'hide my shell for an hour' forgery from the audit."""
        self.assertEqual(self.send("ml", alert_type="SSH Login").status, 403)
        self.assertEqual(self.send("cloud", alert_type="SSH Login").status, 403)
        self.assertEqual(self.main.PIPELINE.shell._authorized, {})

    def test_ml_key_cannot_build_a_kill_chain(self):
        """Breaks if a sensor can claim packet-rule types ('Port Scan', 'Breach'):
        that forged an ACTIVE_INTRUSION against any IP even after step 1."""
        for t in ("Port Scan", "Breach", "Lateral Movement"):
            self.assertEqual(self.send("ml", alert_type=t, src="192.168.56.8").status,
                             403, t)
        self.assertEqual(self.main.PIPELINE.correlator.incident_count(), 0)

    def test_alert_records_which_sensor_reported_it(self):
        r = self.send("cloud", alert_type="Cloud Recon")
        rec = self.main.PIPELINE.alerts.get(r.json()["alert_id"])
        self.assertEqual(rec["details"]["reported_by"], "cloud")

    def test_host_login_for_a_hostname_maps_to_its_ip(self):
        """The exemption never worked before: the host sensor reports the syslog
        HOSTNAME, flows are keyed by IP. assets.json host_addresses bridges it."""
        self.main.HOST_ADDRESSES["web-prod-01"] = "192.168.56.101"
        r = self.send("host", alert_type="SSH Login", src="192.168.56.10",
                      dst="web-prod-01", severity="medium")
        self.assertEqual(r.status, 200, r.body)
        self.assertIn(("192.168.56.10", "192.168.56.101"),
                      self.main.PIPELINE.shell._authorized)


@unittest.skipUnless(SCAPY, "main.py needs scapy")
class TestOperatorChosenSharedKey(unittest.TestCase):
    def test_a_private_NIDS_SENSOR_KEY_keeps_working(self):
        """Usability: deployments that already set their own shared key must
        not break on upgrade. It is a secret the operator chose, not the
        published default, so it keeps full sensor rights."""
        from apiharness import LiveServer
        key = "operator-chosen-shared-key-0123456789"
        with LiveServer(env={"NIDS_SENSOR_KEY": key}) as srv:
            r = srv.post("/alert", {"src": "192.168.56.50", "alert_type": "SSH Login",
                                    "dst": "192.168.56.101", "severity": "medium",
                                    "message": "m"},
                         headers={"X-Sensor-Key": key})
            self.assertEqual(r.status, 200, r.body)

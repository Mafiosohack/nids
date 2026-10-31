"""TLS by default, with no setup: 2026-10-31 audit, step 2.

The threat: the NIDS dashboard is reached across the very network it monitors.
On plain HTTP, anyone positioned on that segment (ARP spoofing, a mirror port)
copies an admin login and gets the defender's whole view - including which of
their own actions were detected and which sources are suppressed.

The usability contract these tests hold:
  * first start makes a certificate; nobody runs openssl
  * a sensor on the same machine needs no TLS configuration at all
  * a remote sensor needs one value (the fingerprint), printed at startup
  * plain HTTP stays possible, but only when explicitly asked for
"""

import json
import os
import shutil
import socket
import ssl
import tempfile
import threading
import time
import unittest

try:
    import cryptography  # noqa: F401
    CRYPTO = True
except ImportError:                                   # pragma: no cover
    CRYPTO = False

try:
    import scapy.all  # noqa: F401
    SCAPY = True
except ImportError:                                   # pragma: no cover
    SCAPY = False


@unittest.skipUnless(CRYPTO, "cryptography required")
class TestCertificate(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_first_start_creates_a_certificate(self):
        from utils.tls import ensure_certificate
        m = ensure_certificate(self.dir)
        self.assertTrue(m.created)
        self.assertTrue(os.path.exists(m.cert_path) and os.path.exists(m.key_path))
        self.assertRegex(m.fingerprint, r"^([0-9A-F]{2}:){31}[0-9A-F]{2}$")

    def test_certificate_is_reused_so_the_fingerprint_is_stable(self):
        """Usability: a remote sensor pins the fingerprint once; restarting the
        server must not invalidate it."""
        from utils.tls import ensure_certificate
        a = ensure_certificate(self.dir)
        b = ensure_certificate(self.dir)
        self.assertFalse(b.created)
        self.assertEqual(a.fingerprint, b.fingerprint)

    def test_certificate_names_loopback(self):
        """Local sensors and the browser on the box connect to 127.0.0.1."""
        from cryptography import x509
        from utils.tls import ensure_certificate
        m = ensure_certificate(self.dir)
        with open(m.cert_path, "rb") as fh:
            cert = x509.load_pem_x509_certificate(fh.read())
        san = cert.extensions.get_extension_for_class(x509.SubjectAlternativeName).value
        self.assertIn("localhost", san.get_values_for_type(x509.DNSName))
        self.assertIn("127.0.0.1",
                      [str(i) for i in san.get_values_for_type(x509.IPAddress)])

    @unittest.skipIf(os.name == "nt", "POSIX permissions")
    def test_private_key_is_owner_only(self):
        from utils.tls import ensure_certificate
        m = ensure_certificate(self.dir)
        self.assertEqual(os.stat(m.key_path).st_mode & 0o077, 0)


@unittest.skipUnless(CRYPTO and SCAPY, "cryptography + scapy required")
class TestTransportChoice(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def choose(self, **env):
        import main
        env.setdefault("NIDS_TLS_DIR", self.dir)
        return main.transport_config(env)

    def test_tls_is_the_default(self):
        """Breaks if the server goes back to plain HTTP unless asked otherwise."""
        t = self.choose()
        self.assertEqual(t["scheme"], "https")
        self.assertTrue(os.path.exists(t["ssl_certfile"]))

    def test_plain_http_only_when_explicitly_requested(self):
        t = self.choose(NIDS_INSECURE_HTTP="1")
        self.assertEqual(t["scheme"], "http")
        self.assertIsNone(t["ssl_certfile"])

    def test_operator_certificate_is_used_as_is(self):
        """Usability: someone with a real certificate must not get ours."""
        from utils.tls import ensure_certificate
        own = ensure_certificate(os.path.join(self.dir, "own"))
        t = self.choose(NIDS_TLS_CERT=own.cert_path, NIDS_TLS_KEY=own.key_path)
        self.assertEqual(t["ssl_certfile"], own.cert_path)
        self.assertEqual(t["fingerprint"], own.fingerprint)


def _serve_tls(cert, key, port_box, stop):
    """Minimal HTTPS endpoint answering /health, for the client tests."""
    import http.server

    class H(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            body = json.dumps({"status": "ok"}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0))
            self.rfile.read(n)
            self.do_GET()

        def log_message(self, *a):
            pass

    srv = http.server.HTTPServer(("127.0.0.1", 0), H)
    if cert:
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.load_cert_chain(cert, key)
        srv.socket = ctx.wrap_socket(srv.socket, server_side=True)
    port_box.append(srv.server_address[1])
    srv.timeout = 0.2
    while not stop.is_set():
        srv.handle_request()
    srv.server_close()


def _issuer_on_the_wire(port):
    """(sha256 hex, issuer) of the certificate a client at 127.0.0.1:port
    actually receives - which is not always the one the server loaded."""
    import hashlib
    from cryptography import x509
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    with socket.create_connection(("127.0.0.1", port), timeout=5) as s, \
            ctx.wrap_socket(s) as t:
        der = t.getpeercert(True)
    return (hashlib.sha256(der).hexdigest(),
            x509.load_der_x509_certificate(der).issuer.rfc4514_string())


@unittest.skipUnless(CRYPTO, "cryptography required")
class TestSensorClient(unittest.TestCase):
    def setUp(self):
        from utils.tls import ensure_certificate
        self.dir = tempfile.mkdtemp()
        self.mat = ensure_certificate(self.dir)
        self.stop = threading.Event()
        self.port = []
        self.t = threading.Thread(target=_serve_tls, daemon=True,
                                  args=(self.mat.cert_path, self.mat.key_path,
                                        self.port, self.stop))
        self.t.start()
        while not self.port:
            time.sleep(0.01)
        # Some endpoint software (seen: AVG Web Shield) intercepts TLS even on
        # loopback and re-signs with its own root. Pinning then *correctly*
        # refuses the connection, so a "connects" test cannot pass here.
        wire, issuer = _issuer_on_the_wire(self.port[0])
        self.intercepted = None if wire == self.mat.fingerprint.replace(
            ":", "").lower() else issuer

    def needs_clean_path(self):
        if self.intercepted:
            self.skipTest(f"TLS on this machine is intercepted by [{self.intercepted}]; "
                          f"the pin refuses it (see test_interception_is_named)")

    def tearDown(self):
        self.stop.set()
        self.t.join(timeout=5)
        shutil.rmtree(self.dir, ignore_errors=True)

    def url(self):
        return f"https://127.0.0.1:{self.port[0]}"

    def test_pinned_fingerprint_connects(self):
        self.needs_clean_path()
        from utils.sensor_client import pinned_session
        s = pinned_session(self.url(), fingerprint=self.mat.fingerprint)
        self.assertEqual(s.get(self.url() + "/health", timeout=5).status_code, 200)

    def test_local_sensor_pins_from_the_cert_file_with_no_config(self):
        """Usability: a sensor on the same box reads the fingerprint itself."""
        self.needs_clean_path()
        from utils.sensor_client import pinned_session
        s = pinned_session(self.url(), cert_path=self.mat.cert_path)
        self.assertEqual(s.get(self.url() + "/health", timeout=5).status_code, 200)

    def test_a_pin_matching_the_presented_certificate_connects(self):
        """The accept path, checkable on any machine: pin whatever certificate
        this client really receives (ours, or an interceptor's)."""
        from utils.sensor_client import pinned_session
        wire, _ = _issuer_on_the_wire(self.port[0])
        s = pinned_session(self.url(), fingerprint=wire)
        self.assertEqual(s.get(self.url() + "/health", timeout=5).status_code, 200)

    def test_interception_is_named(self):
        """Usability: 'fingerprints did not match' alone cannot tell an operator
        whether it is their antivirus or an attacker. The error names who
        issued the certificate the sensor was actually shown.

        Pinned to the wrong value, the sensor is shown our own certificate, so
        its issuer must appear; on an intercepted machine the correct pin
        produces the same error naming the interceptor instead."""
        from utils.sensor_client import AlertClient, TransportError
        os.environ["NIDS_SENSOR_KEY"] = "test-key-not-used-for-anything"
        try:
            c = AlertClient("ml", url=self.url(), fingerprint=":".join(["00"] * 32))
            with self.assertRaises(TransportError) as ctx:
                c.post({"x": 1}, timeout=5)
        finally:
            del os.environ["NIDS_SENSOR_KEY"]
        msg = str(ctx.exception)
        _, issuer = _issuer_on_the_wire(self.port[0])
        self.assertIn("issued by", msg)
        self.assertIn(issuer, msg)

    def test_health_failure_is_explained_too(self):
        from utils.sensor_client import AlertClient, TransportError
        os.environ["NIDS_SENSOR_KEY"] = "test-key-not-used-for-anything"
        try:
            c = AlertClient("ml", url=self.url(), fingerprint=":".join(["00"] * 32))
            with self.assertRaises(TransportError):
                c.health(timeout=5)
        finally:
            del os.environ["NIDS_SENSOR_KEY"]

    def test_wrong_fingerprint_is_refused(self):
        """Breaks if pinning is not enforced: a man in the middle presenting his
        own certificate would receive the sensor key and every alert."""
        from utils.sensor_client import pinned_session
        bad = ":".join(["00"] * 32)
        s = pinned_session(self.url(), fingerprint=bad)
        with self.assertRaises(Exception):
            s.get(self.url() + "/health", timeout=5)

    def test_https_without_any_pin_explains_the_fix(self):
        from utils.sensor_client import TransportError, pinned_session
        with self.assertRaises(TransportError) as ctx:
            pinned_session(self.url(), cert_path=os.path.join(self.dir, "none.pem"))
        self.assertIn("NIDS_TLS_FINGERPRINT", str(ctx.exception))


@unittest.skipUnless(shutil.which("node"), "node required")
class TestDashboardTransportBanner(unittest.TestCase):
    def run_js(self, expr):
        import subprocess
        from test_security_hardening import DASHBOARD, _js_function
        with open(DASHBOARD, encoding="utf-8") as fh:
            src = fh.read()
        code = _js_function(src, "insecureTransport") + \
            f"\nprocess.stdout.write(JSON.stringify({expr}));"
        out = subprocess.run(["node", "-e", code], capture_output=True, text=True)
        self.assertEqual(out.returncode, 0, out.stderr)
        return json.loads(out.stdout)

    def test_plain_http_across_the_network_is_flagged(self):
        self.assertTrue(self.run_js(
            "insecureTransport({protocol:'http:', hostname:'192.168.56.10'})"))

    def test_https_and_loopback_are_not_flagged(self):
        """Usability: no nagging when nothing crosses the wire in the clear."""
        self.assertFalse(self.run_js(
            "insecureTransport({protocol:'https:', hostname:'192.168.56.10'})"))
        self.assertFalse(self.run_js(
            "insecureTransport({protocol:'http:', hostname:'127.0.0.1'})"))
        self.assertFalse(self.run_js(
            "insecureTransport({protocol:'http:', hostname:'localhost'})"))


if __name__ == "__main__":
    unittest.main()


class TestSensorsUseTheSharedClient(unittest.TestCase):
    """The three sensors used to each hardcode http://127.0.0.1:8000 and the
    published key 'sensor-key-change-me-in-production'. One client now."""

    SENSORS = ("live_ids_v2.py", "host_log_sensor.py", "cloud_log_sensor.py")

    def source(self, name):
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        with open(os.path.join(root, name), encoding="utf-8") as fh:
            return fh.read()

    def test_no_sensor_ships_the_published_key(self):
        for s in self.SENSORS:
            with self.subTest(sensor=s):
                self.assertFalse("sensor-key-change-me-in-production" in self.source(s),
                                 f"{s} still ships the published sensor key")

    def test_no_sensor_defaults_to_plain_http(self):
        for s in self.SENSORS:
            with self.subTest(sensor=s):
                self.assertFalse("http://127.0.0.1:8000" in self.source(s),
                                 f"{s} still defaults to plain HTTP")

    def test_each_sensor_asks_for_its_own_identity(self):
        for s, who in zip(self.SENSORS, ("ml", "host", "cloud")):
            with self.subTest(sensor=s):
                self.assertTrue(f'client_for("{who}")' in self.source(s),
                                f"{s} does not use client_for({who!r})")


class TestClientFor(unittest.TestCase):
    def setUp(self):
        import utils.sensor_client as sc
        self.sc = sc
        sc._CLIENTS.clear()
        self.dir = tempfile.mkdtemp()
        self.saved = {k: os.environ.pop(k, None) for k in
                      ("NIDS_SENSOR_KEY", "NIDS_SENSOR_KEYS", "NIDS_TLS_FINGERPRINT")}

    def tearDown(self):
        self.sc._CLIENTS.clear()
        for k, v in self.saved.items():
            if v is not None:
                os.environ[k] = v
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_missing_setup_is_explained_not_raised(self):
        """A sensor started before main.py must keep running and say why its
        alerts are not going anywhere yet."""
        client, why = self.sc.client_for(
            "ml", keys_path=os.path.join(self.dir, "none.json"))
        self.assertIsNone(client)
        self.assertTrue(why)

    def test_failure_is_not_cached(self):
        """Usability: starting main.py afterwards fixes it without restarting
        the sensor."""
        self.sc.client_for("ml", keys_path=os.path.join(self.dir, "none.json"))
        os.environ["NIDS_SENSOR_KEY"] = "k"
        os.environ["NIDS_TLS_FINGERPRINT"] = ":".join(["AB"] * 32)
        try:
            client, why = self.sc.client_for("ml")
        finally:
            del os.environ["NIDS_SENSOR_KEY"], os.environ["NIDS_TLS_FINGERPRINT"]
        self.assertIsNotNone(client, why)
        self.assertIs(self.sc.client_for("ml")[0], client)


class TestTransportIsVisible(unittest.TestCase):
    def dashboard(self):
        from test_security_hardening import DASHBOARD
        with open(DASHBOARD, encoding="utf-8") as fh:
            return fh.read()

    def test_banner_element_sits_above_the_login(self):
        """The warning must be on screen while credentials are being typed."""
        import re
        src = self.dashboard()
        self.assertTrue(re.search(r'id="transport-warning"', src),
                        "no #transport-warning element")
        self.assertLess(src.index('id="transport-warning"'),
                        src.index('id="login-overlay"'),
                        "warning must come before the login overlay")

    def test_banner_starts_hidden_for_real(self):
        """This dashboard scopes .hidden per component (.login-error.hidden,
        ...); there is no global rule. Without its own, the banner would show
        on HTTPS too - nagging every correct deployment."""
        import re
        self.assertTrue(re.search(r"\.transport-warning\.hidden\s*\{\s*display\s*:\s*none",
                                  self.dashboard()),
                        "no .transport-warning.hidden{display:none} rule")

    def test_banner_is_driven_by_insecure_transport(self):
        import re
        self.assertTrue(re.search(
            r"if\s*\(\s*insecureTransport\(\s*location\s*\)\s*\)", self.dashboard()),
            "nothing calls insecureTransport(location)")

    @unittest.skipUnless(SCAPY, "scapy required")
    def test_health_reports_whether_tls_is_on(self):
        from apiharness import LiveServer
        with LiveServer() as srv:
            body = srv.get("/health").json()
        self.assertIn("tls", body)
        self.assertFalse(body["tls"])      # the harness serves plain HTTP


@unittest.skipUnless(SCAPY, "scapy required")
class TestAutostartCapture(unittest.TestCase):
    """start.sh used to `curl -X POST /control/start` with no login: a 401 it
    discarded, so capture never started from the script. Autostart is now a
    server-side setting, and its outcome is audited under actor "system"."""

    def entries(self, srv):
        tok = srv.login()
        return srv.get("/audit?limit=200", token=tok).json()["entries"]

    def test_autostart_attempt_is_audited_with_its_reason(self):
        from unittest import mock
        import main
        from apiharness import LiveServer
        # NIDS_IFACE is read at import; patch the constant it lands in.
        with mock.patch.object(main, "NETWORK_INTERFACE", "nids-no-such-iface"),                 LiveServer(env={"NIDS_AUTOSTART_CAPTURE": "1"}) as srv:
            mine = [e for e in self.entries(srv) if e["actor"] == "system"
                    and e["action"].startswith("sniffer_start")]
        self.assertTrue(mine, "autostart left no audit trace")
        self.assertEqual(mine[0]["action"], "sniffer_start_failed")
        self.assertIn("nids-no-such-iface", mine[0]["detail"])

    def test_no_autostart_unless_asked(self):
        from apiharness import LiveServer
        with LiveServer() as srv:
            mine = [e for e in self.entries(srv) if e["actor"] == "system"]
        self.assertEqual(mine, [])


class TestDefaultsFollowTLS(unittest.TestCase):
    def test_local_file_dashboard_defaults_to_https(self):
        import subprocess
        if not shutil.which("node"):
            self.skipTest("node required")
        from test_security_hardening import DASHBOARD, _js_function
        with open(DASHBOARD, encoding="utf-8") as fh:
            code = _js_function(fh.read(), "resolveApiBase")
        code += ("\nprocess.stdout.write(resolveApiBase("
                 "{protocol:'file:', origin:'null', search:''}));")
        out = subprocess.run(["node", "-e", code], capture_output=True, text=True)
        self.assertEqual(out.stdout, "https://127.0.0.1:8000", out.stderr)

    @unittest.skipUnless(SCAPY, "scapy required")
    def test_default_cors_origins_include_https(self):
        import main
        if os.environ.get("NIDS_ALLOWED_ORIGINS"):
            self.skipTest("operator override in environment")
        self.assertIn("https://127.0.0.1:8000", main.ALLOWED_ORIGINS)
        self.assertIn("https://localhost:8000", main.ALLOWED_ORIGINS)

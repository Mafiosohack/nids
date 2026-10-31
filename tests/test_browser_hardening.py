"""The browser boundary: 2026-10-31 audit, step 3.

Step 1 escaped what the dashboard renders. That fixes the bugs that were found;
this file is about the next one nobody has found yet. Two layers:

  * CSP: injected markup cannot execute. Only the dashboard's own inline
    script, identified by its SHA-256, runs; no inline event handlers; no
    third-party script host (Chart.js is served from here).
  * HttpOnly session cookie: even if script does run, it cannot read the
    session token and carry it off - it dies with the tab.

The cookie brings CSRF with it, so a cookie-authenticated request that changes
state must come from our own origin. Bearer tokens (curl, scripts) are
unchanged: a browser never attaches one cross-site by itself.
"""

import base64
import hashlib
import os
import re
import unittest

try:
    import scapy.all  # noqa: F401
    SCAPY = True
except ImportError:                                   # pragma: no cover
    SCAPY = False

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DASHBOARD = os.path.join(REPO, "nids_dashboard.html")
VENDORED_CHART = os.path.join(REPO, "static", "vendor", "chart.umd.min.js")


def dashboard_source():
    with open(DASHBOARD, encoding="utf-8") as fh:
        return fh.read()


def inline_scripts(html):
    return re.findall(r"<script(?![^>]*\bsrc=)[^>]*>(.*?)</script>", html, re.S)


def csp_hash(body):
    return "'sha256-" + base64.b64encode(
        hashlib.sha256(body.encode("utf-8")).digest()).decode() + "'"


def directives(csp):
    out = {}
    for part in csp.split(";"):
        bits = part.split()
        if bits:
            out[bits[0]] = bits[1:]
    return out


class TestDashboardMarkup(unittest.TestCase):
    """Static properties of the HTML that the CSP depends on."""

    def test_no_inline_event_handlers(self):
        """CSP blocks onclick="..." attributes. Any left would be dead
        buttons - and allowing them ('unsafe-hashes') would re-open the hole."""
        src = dashboard_source()
        found = re.findall(r"<[^>]*\son[a-z]+\s*=", src)
        self.assertEqual(found, [], f"{len(found)} inline handler(s) remain")

    def test_every_button_action_is_wired(self):
        """Buttons now name an action (data-action="...") that one listener
        looks up in a fixed table. A name missing from the table is a dead
        button; this is the check that the conversion lost none."""
        src = dashboard_source()
        used = set(re.findall(r'data-action="([A-Za-z]+)"', src))
        self.assertGreaterEqual(len(used), 10, "expected the converted buttons")
        m = re.search(r"const ACTIONS\s*=\s*\{(.*?)\n\};", src, re.S)
        self.assertTrue(m, "no ACTIONS table")
        table = set(re.findall(r"^\s*([A-Za-z]+)\s*:", m.group(1), re.M))
        self.assertEqual(used - table, set(), "buttons with no action")

    def test_no_third_party_script(self):
        src = dashboard_source()
        srcs = re.findall(r"<script[^>]*\bsrc=\"([^\"]+)\"", src)
        for s in srcs:
            with self.subTest(src=s):
                self.assertFalse(re.match(r"^(https?:)?//", s),
                                 f"script loaded from another host: {s}")

    def test_chart_js_is_vendored(self):
        self.assertTrue(os.path.exists(VENDORED_CHART), "static/vendor/chart.umd.min.js missing")
        with open(VENDORED_CHART, encoding="utf-8") as fh:
            head = fh.read(400)
        self.assertIn("Chart.js v4.4.1", head)

    def test_token_never_reaches_web_storage(self):
        """Breaks if the dashboard goes back to keeping the session where any
        script on the page can read it."""
        src = dashboard_source()
        self.assertIsNone(re.search(r"(session|local)Storage\.setItem\(\s*'nids_token'", src))


@unittest.skipUnless(SCAPY, "scapy required")
class TestServedHeaders(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from apiharness import LiveServer
        cls.cm = LiveServer()
        cls.srv = cls.cm.__enter__()
        # ~108 KB: large enough to hit the loopback stall, see get_large().
        # Fetched once; the tests below only read it.
        cls.dashboard = cls.srv.get_large("/")

    @classmethod
    def tearDownClass(cls):
        cls.cm.__exit__(None, None, None)

    def header(self, r, name):
        return {k.lower(): v for k, v in r.headers.items()}.get(name.lower())

    def test_dashboard_is_the_real_page(self):
        """Run from the repo root: the dashboard path is relative to the working
        directory, and from elsewhere / is a 562-byte 404 with no inline script,
        which would make the CSP checks below pass vacuously."""
        self.assertEqual(self.dashboard.status, 200)
        self.assertTrue(inline_scripts(self.dashboard.body.decode("utf-8")))

    def test_dashboard_csp_allows_only_its_own_scripts(self):
        r = self.dashboard
        csp = self.header(r, "Content-Security-Policy")
        self.assertTrue(csp, "no CSP on the dashboard")
        d = directives(csp)
        scripts = d.get("script-src", [])
        self.assertNotIn("'unsafe-inline'", scripts)
        self.assertNotIn("'unsafe-eval'", scripts)
        self.assertNotIn("'unsafe-hashes'", scripts)
        wanted = {csp_hash(b) for b in inline_scripts(r.body.decode("utf-8"))}
        allowed = {s for s in scripts if s.startswith("'sha256-")}
        # Exactly ours: a missing hash is a dead dashboard, an extra one is a
        # script the policy permits that the page does not contain.
        self.assertEqual(allowed, wanted)
        self.assertNotIn(csp_hash(""), allowed, "empty inline script permitted")
        self.assertFalse([s for s in scripts if s.startswith(("http", "https:"))],
                         f"script-src allows a remote host: {scripts}")
        self.assertEqual(d.get("frame-ancestors"), ["'none'"])
        self.assertEqual(d.get("object-src") or d.get("default-src"), ["'none'"])
        self.assertEqual(d.get("base-uri"), ["'none'"])

    def test_injected_script_would_not_match(self):
        """The property that matters: a script that is not byte-identical to
        ours has a different hash and is not in the policy."""
        csp = self.header(self.dashboard, "Content-Security-Policy")
        self.assertNotIn(csp_hash("alert(document.cookie)"), csp)

    def test_baseline_headers_on_api_responses(self):
        r = self.srv.get("/health")
        self.assertEqual(self.header(r, "X-Content-Type-Options"), "nosniff")
        self.assertEqual(self.header(r, "Referrer-Policy"), "no-referrer")
        self.assertEqual(self.header(r, "X-Frame-Options"), "DENY")

    def test_no_hsts_with_a_self_signed_certificate(self):
        """Usability: HSTS makes browsers refuse the click-through on a
        self-signed certificate - the operator could not reach the dashboard."""
        self.assertIsNone(self.header(self.dashboard, "Strict-Transport-Security"))

    def test_vendored_chart_is_served(self):
        r = self.srv.get_large("/static/vendor/chart.umd.min.js")
        self.assertEqual(r.status, 200)
        self.assertIn("javascript", self.header(r, "Content-Type"))

    def test_static_cannot_escape_its_directory(self):
        for p in ("/static/../main.py", "/static/%2e%2e/main.py",
                  "/static/..%2fmain.py", "/static/vendor/../../nids_users.db"):
            with self.subTest(path=p):
                r = self.srv.get(p)
                self.assertNotEqual(r.status, 200, p)


@unittest.skipUnless(SCAPY, "scapy required")
class TestCookieSession(unittest.TestCase):
    def setUp(self):
        from apiharness import LiveServer, ADMIN_PASSWORD
        self.cm = LiveServer()
        self.srv = self.cm.__enter__()
        self.password = ADMIN_PASSWORD
        self.origin = f"http://127.0.0.1:{self.srv.port}"

    def tearDown(self):
        self.cm.__exit__(None, None, None)

    def browser_login(self):
        r = self.srv.post("/auth/login", {"username": "admin", "password": self.password})
        self.assertEqual(r.status, 200)
        set_cookie = {k.lower(): v for k, v in r.headers.items()}.get("set-cookie", "")
        m = re.match(r"nids_session=([^;]+)", set_cookie)
        self.assertTrue(m, f"no nids_session cookie: {set_cookie!r}")
        return m.group(1), set_cookie

    def test_login_sets_an_httponly_strict_cookie(self):
        _, sc = self.browser_login()
        low = sc.lower()
        self.assertIn("httponly", low)
        self.assertIn("samesite=strict", low)
        self.assertIn("path=/", low)

    def test_cookie_alone_authenticates_reads(self):
        tok, _ = self.browser_login()
        r = self.srv.get("/status", headers={"Cookie": f"nids_session={tok}"})
        self.assertEqual(r.status, 200)

    def test_session_endpoint_restores_the_dashboard_after_reload(self):
        """The page can no longer read the token, so it asks who it is."""
        tok, _ = self.browser_login()
        r = self.srv.get("/auth/session", headers={"Cookie": f"nids_session={tok}"})
        self.assertEqual(r.status, 200)
        self.assertEqual(r.json()["username"], "admin")
        self.assertEqual(r.json()["role"], "admin")
        self.assertNotIn("token", r.json())
        self.assertEqual(self.srv.get("/auth/session").status, 401)

    def test_cookie_write_from_another_origin_is_refused(self):
        """CSRF. SameSite=Strict stops other *sites*, but another service on the
        same IP and a different port is the same site - so check Origin."""
        tok, _ = self.browser_login()
        for origin in ("http://evil.example", f"http://127.0.0.1:{self.srv.port + 1}"):
            with self.subTest(origin=origin):
                r = self.srv.delete("/alerts", headers={
                    "Cookie": f"nids_session={tok}", "Origin": origin})
                self.assertEqual(r.status, 403)

    def test_cookie_write_with_no_origin_is_refused(self):
        tok, _ = self.browser_login()
        r = self.srv.post("/control/stop", headers={"Cookie": f"nids_session={tok}"})
        self.assertEqual(r.status, 403)

    def test_cookie_write_from_our_own_origin_works(self):
        tok, _ = self.browser_login()
        r = self.srv.delete("/alerts", headers={
            "Cookie": f"nids_session={tok}", "Origin": self.origin})
        self.assertEqual(r.status, 200)

    def test_bearer_scripts_are_unaffected(self):
        """Usability: curl/scripts send no Origin and must keep working."""
        tok = self.srv.login()
        self.assertEqual(self.srv.delete("/alerts", token=tok).status, 200)

    def test_logout_clears_the_cookie_and_revokes_it(self):
        tok, _ = self.browser_login()
        r = self.srv.post("/auth/logout", headers={
            "Cookie": f"nids_session={tok}", "Origin": self.origin})
        self.assertEqual(r.status, 200)
        sc = {k.lower(): v for k, v in r.headers.items()}.get("set-cookie", "")
        self.assertRegex(sc.lower(), r"nids_session=(\"\")?;.*max-age=0")
        self.assertEqual(self.srv.get("/status", headers={
            "Cookie": f"nids_session={tok}"}).status, 401)


@unittest.skipUnless(SCAPY, "scapy required")
class TestCookieFlags(unittest.TestCase):
    def test_secure_flag_follows_the_transport(self):
        """Secure on HTTPS. Not on NIDS_INSECURE_HTTP, where a Secure cookie
        would never be stored and login would silently loop."""
        import main
        self.assertTrue(main.session_cookie_params("https")["secure"])
        self.assertFalse(main.session_cookie_params("http")["secure"])


if __name__ == "__main__":
    unittest.main()

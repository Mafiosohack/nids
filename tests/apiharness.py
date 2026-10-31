"""Run the real FastAPI app on a random loopback port, against throwaway state.

Used by tests that need actual HTTP behaviour (status codes, headers, client
IP, dependency injection) rather than calling endpoint functions directly.
starlette's TestClient would do, but it needs httpx, which this project does
not otherwise depend on; uvicorn and urllib are already here.

Every path main.py writes to is redirected into a temp directory, so a test run
can never touch the operator's real nids_users.db or sensor keys.
"""

import json
import os
import shutil
import tempfile
import threading
import time
import urllib.error
import urllib.request

ADMIN_PASSWORD = "harness-admin-password"


class Response:
    def __init__(self, status, body, headers):
        self.status = status
        self.body = body
        self.headers = headers

    def json(self):
        return json.loads(self.body or b"null")


class LiveServer:
    """Context manager: `with LiveServer() as srv: srv.post(...)`."""

    def __init__(self, env=None):
        self.env = dict(env or {})

    def __enter__(self):
        import main
        import uvicorn
        self.main = main
        self.tmp = tempfile.mkdtemp(prefix="nids-test-")
        self._saved = {k: getattr(main, k) for k in
                       ("DB_PATH", "ADMIN_SEED_PASSWORD", "SENSOR_KEYS_PATH")
                       if hasattr(main, k)}
        main.DB_PATH = main.Path(self.tmp) / "users.db"
        main.ADMIN_SEED_PASSWORD = ADMIN_PASSWORD
        if hasattr(main, "SENSOR_KEYS_PATH"):
            main.SENSOR_KEYS_PATH = main.Path(self.tmp) / "sensor_keys.json"
        self._old_env = {k: os.environ.get(k) for k in self.env}
        os.environ.update(self.env)
        main.PIPELINE.reset()

        cfg = uvicorn.Config(main.app, host="127.0.0.1", port=0, log_level="error")
        self.server = uvicorn.Server(cfg)
        self.thread = threading.Thread(target=self.server.run, daemon=True)
        self.thread.start()
        deadline = time.time() + 15
        while not self.server.started:
            if time.time() > deadline:
                raise RuntimeError("test server did not start")
            time.sleep(0.02)
        self.port = self.server.servers[0].sockets[0].getsockname()[1]
        self.stalls = []        # (path, reason) for every get_large() retry
        return self

    def __exit__(self, *exc):
        self.server.should_exit = True
        self.thread.join(timeout=10)
        for k, v in self._saved.items():
            setattr(self.main, k, v)
        for k, v in self._old_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        shutil.rmtree(self.tmp, ignore_errors=True)

    # ── HTTP ──────────────────────────────────────────────────────────────────
    def request(self, method, path, body=None, headers=None, token=None,
                timeout=15):
        h = {"Content-Type": "application/json"}
        h.update(headers or {})
        if token:
            h["Authorization"] = f"Bearer {token}"
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(f"http://127.0.0.1:{self.port}{path}",
                                     data=data, headers=h, method=method)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return Response(r.status, r.read(), dict(r.headers))
        except urllib.error.HTTPError as e:
            return Response(e.code, e.read(), dict(e.headers))

    def get(self, path, **kw):
        return self.request("GET", path, **kw)

    def get_large(self, path, attempts=4, timeout=4, **kw):
        """GET for bodies over ~64 KB (the dashboard, vendored JS).

        Measured 2026-10-31 on the dev box: about 1 in 10 such transfers over
        loopback hangs after the first ~64 KB receive window. It does it
        identically with uvicorn, stdlib http.server and a raw socket client,
        so it is outside this app; AVG's network filter is the running
        suspect. A GET is safe to repeat, so retry with a short timeout -
        and report every retry on stderr and in self.stalls, never silently.
        Small JSON responses never stall and keep using get()."""
        import http.client
        import socket
        import sys
        reasons = []
        for _ in range(attempts):
            try:
                return self.request("GET", path, timeout=timeout, **kw)
            except (socket.timeout, TimeoutError, ConnectionResetError,
                    http.client.IncompleteRead) as e:
                reasons.append(type(e).__name__)
                self.stalls.append((path, type(e).__name__))
                print(f"[apiharness] GET {path} stalled ({type(e).__name__}); "
                      f"retrying - see get_large()", file=sys.stderr)
        raise AssertionError(
            f"GET {path} stalled on all {attempts} attempts ({reasons}). "
            "Not seen from the app itself; check the local network filter "
            "(AVG Web Shield) before suspecting the server.")

    def post(self, path, body=None, **kw):
        return self.request("POST", path, body=body, **kw)

    def delete(self, path, **kw):
        return self.request("DELETE", path, **kw)

    def patch(self, path, body=None, **kw):
        return self.request("PATCH", path, body=body, **kw)

    def login(self, username="admin", password=ADMIN_PASSWORD):
        r = self.post("/auth/login", {"username": username, "password": password})
        if r.status != 200:
            raise AssertionError(f"login failed: {r.status} {r.body!r}")
        return r.json()["token"]

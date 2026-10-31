"""The harness's large-body GET: retries a stalled transfer, says so, and
gives up loudly. Measured 2026-10-31 on the dev box: ~10% of loopback
transfers over ~64 KB hang after the first receive window - with uvicorn,
with stdlib http.server and with a raw socket client alike, so it is not
the app. AVG's network filter is running there. The retry keeps the suite
honest about headers/CSP without depending on that."""

import socket
import threading
import time
import unittest

from apiharness import LiveServer

BODY = b"y" * 100_000
STALL_AFTER = 57_344


class StubServer:
    """Serves BODY. The first `stalls` connections send headers and part of
    the body, then go silent - the shape of the stall seen in the wild."""

    def __init__(self, stalls):
        self.stalls = stalls
        self.connections = 0
        self.sock = socket.socket()
        self.sock.bind(("127.0.0.1", 0))
        self.sock.listen()
        self.port = self.sock.getsockname()[1]
        self._open = []
        threading.Thread(target=self._serve, daemon=True).start()

    def _serve(self):
        while True:
            try:
                c, _ = self.sock.accept()
            except OSError:
                return
            self._open.append(c)
            c.recv(4096)
            self.connections += 1
            head = (b"HTTP/1.1 200 OK\r\nContent-Length: %d\r\n"
                    b"X-Probe: yes\r\n\r\n" % len(BODY))
            if self.connections <= self.stalls:
                c.sendall(head + BODY[:STALL_AFTER])   # ...and never the rest
            else:
                c.sendall(head + BODY)
                c.close()

    def close(self):
        self.sock.close()
        for c in self._open:
            try:
                c.close()
            except OSError:
                pass


def harness_for(stub):
    srv = LiveServer.__new__(LiveServer)
    srv.port = stub.port
    srv.stalls = []
    return srv


class TestGetLarge(unittest.TestCase):
    def test_a_stalled_transfer_is_retried_and_recorded(self):
        stub = StubServer(stalls=1)
        try:
            srv = harness_for(stub)
            r = srv.get_large("/", timeout=0.5)
            self.assertEqual(r.status, 200)
            self.assertEqual(r.body, BODY)
            self.assertEqual(r.headers.get("X-Probe"), "yes")
            self.assertEqual(stub.connections, 2)
            self.assertEqual(len(srv.stalls), 1, "the retry must be visible")
        finally:
            stub.close()

    def test_gives_up_loudly_with_the_diagnosis(self):
        stub = StubServer(stalls=99)
        try:
            srv = harness_for(stub)
            t0 = time.monotonic()
            with self.assertRaises(AssertionError) as cm:
                srv.get_large("/", attempts=3, timeout=0.3)
            self.assertLess(time.monotonic() - t0, 5)
            self.assertEqual(stub.connections, 3)
            self.assertIn("stalled", str(cm.exception))
        finally:
            stub.close()

    def test_a_clean_transfer_is_not_retried(self):
        stub = StubServer(stalls=0)
        try:
            srv = harness_for(stub)
            self.assertEqual(srv.get_large("/", timeout=0.5).body, BODY)
            self.assertEqual(stub.connections, 1)
            self.assertEqual(srv.stalls, [])
        finally:
            stub.close()


if __name__ == "__main__":
    unittest.main()

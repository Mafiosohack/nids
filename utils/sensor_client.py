"""How every sensor talks to main.py: its own key, over a pinned TLS channel.

One implementation for all three sensors, so the transport cannot drift
between them (it did: two of them used to hardcode the published key).

Configuration, in the order a sensor looks for it:

  NIDS_URL               default https://127.0.0.1:8000
  NIDS_SENSOR_KEY        else this sensor's entry in sensor_keys.json
  NIDS_TLS_FINGERPRINT   else computed from the local tls/nids-cert.pem

So a sensor on the same machine as main.py needs none of them, and a sensor on
another machine needs NIDS_URL, NIDS_SENSOR_KEY and NIDS_TLS_FINGERPRINT - all
three printed by main.py at startup.

There is deliberately no automatic fallback from HTTPS to HTTP. A sensor that
silently downgrades is one an attacker can downgrade; a clear error is better.
"""

import os
from typing import Dict, Optional, Tuple

import requests
from requests.adapters import HTTPAdapter

from utils import sensor_keys

DEFAULT_URL = "https://127.0.0.1:8000"
ENV_URL = "NIDS_URL"
ENV_FINGERPRINT = "NIDS_TLS_FINGERPRINT"
DEFAULT_CERT = os.path.join(os.environ.get("NIDS_TLS_DIR", "tls"), "nids-cert.pem")


class TransportError(RuntimeError):
    """The sensor cannot reach main.py securely. The message says how to fix it."""


class _PinnedAdapter(HTTPAdapter):
    """Accept exactly one certificate, identified by its SHA-256 fingerprint.

    Chain validation is replaced, not skipped: urllib3 compares the server's
    certificate against the pin on every connection and aborts on mismatch.
    """

    def __init__(self, fingerprint: str, **kw):
        self._pin = fingerprint.replace(":", "").lower()
        super().__init__(**kw)

    def init_poolmanager(self, *args, **kw):
        kw["assert_fingerprint"] = self._pin
        kw["cert_reqs"] = "CERT_NONE"
        kw["assert_hostname"] = False
        return super().init_poolmanager(*args, **kw)

    def send(self, request, **kw):
        # requests re-derives cert_reqs from `verify` on every send (True ->
        # CERT_REQUIRED -> CA chain validation, which a self-signed cert fails).
        # The pin above is the verification; assert_fingerprint stays in the
        # pool kwargs and is checked on every handshake regardless.
        kw["verify"] = False
        return super().send(request, **kw)


def pinned_session(url: str, fingerprint: Optional[str] = None,
                   cert_path: Optional[str] = None) -> requests.Session:
    """A requests Session for `url`. HTTPS is pinned; plain HTTP is passed
    through (only reachable when the operator chose NIDS_INSECURE_HTTP)."""
    s = requests.Session()
    if not url.lower().startswith("https://"):
        return s
    fp = (fingerprint or os.environ.get(ENV_FINGERPRINT, "")).strip()
    if not fp:
        path = cert_path or DEFAULT_CERT
        try:
            from utils.tls import fingerprint_of
            fp = fingerprint_of(path)
        except (OSError, ValueError, ImportError) as e:
            raise TransportError(
                f"Cannot verify the NIDS server: no {ENV_FINGERPRINT} set and no "
                f"local certificate at {path} ({type(e).__name__}). On the server's "
                f"machine this file exists after main.py first starts; on another "
                f"machine set {ENV_FINGERPRINT} to the fingerprint main.py prints "
                f"at startup.") from e
    s.mount("https://", _PinnedAdapter(fp))
    return s


def presented_certificate(url: str, timeout: float = 3.0) -> Optional[str]:
    """Who the certificate at `url` claims to be and who signed it, as seen by
    this client. Used only to explain a failed pin - never to decide trust."""
    import socket
    import ssl
    from urllib.parse import urlsplit
    try:
        from cryptography import x509
        u = urlsplit(url)
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        with socket.create_connection((u.hostname, u.port or 443), timeout=timeout) as s, \
                ctx.wrap_socket(s, server_hostname=u.hostname) as t:
            cert = x509.load_der_x509_certificate(t.getpeercert(True))
        return (f"'{cert.subject.rfc4514_string()}' issued by "
                f"'{cert.issuer.rfc4514_string()}'")
    except Exception:            # explanation is best-effort; the refusal is not
        return None


class AlertClient:
    """POSTs one sensor's alerts to main.py's /alert endpoint."""

    def __init__(self, sensor: str, url: Optional[str] = None,
                 fingerprint: Optional[str] = None, keys_path=None):
        self.sensor = sensor
        self.base = (url or os.environ.get(ENV_URL, DEFAULT_URL)).rstrip("/")
        self.alert_url = self.base + "/alert"
        self.key = (sensor_keys.key_for(sensor) if keys_path is None
                    else sensor_keys.key_for(sensor, keys_path))
        self.session = pinned_session(self.base, fingerprint=fingerprint)

    def _explain(self, e: Exception) -> TransportError:
        text = str(e)
        if "WRONG_VERSION_NUMBER" in text or "wrong version number" in text:
            return TransportError(
                f"{self.base} answered in plain HTTP. The server was started "
                f"with NIDS_INSECURE_HTTP=1; set NIDS_URL=http://... here too, "
                f"or (better) restart the server without it.")
        if "Fingerprints did not match" in text:
            shown = presented_certificate(self.base) or "an unreadable certificate"
            return TransportError(
                f"Refused {self.base}: it presented {shown}, which is not the "
                f"pinned NIDS certificate. Nothing was sent. Either the server's "
                f"certificate was regenerated (update {ENV_FINGERPRINT} to the "
                f"fingerprint main.py prints), or something is intercepting TLS "
                f"on this path - antivirus 'HTTPS scanning' does this, so does an "
                f"attacker. If the issuer is your own security software, exclude "
                f"this address from its HTTPS scanning.")
        return TransportError(f"TLS to {self.base} failed: {e}")

    def post(self, payload: dict, timeout: float = 3.0) -> requests.Response:
        try:
            return self.session.post(self.alert_url, json=payload, timeout=timeout,
                                     headers={"X-Sensor-Key": self.key})
        except requests.exceptions.SSLError as e:
            raise self._explain(e) from e

    def health(self, timeout: float = 3.0) -> requests.Response:
        try:
            return self.session.get(self.base + "/health", timeout=timeout)
        except requests.exceptions.SSLError as e:
            raise self._explain(e) from e


_CLIENTS: Dict[str, AlertClient] = {}


def client_for(sensor: str, keys_path=None) -> Tuple[Optional[AlertClient], str]:
    """(client, "") or (None, why). Never raises, so a sensor started before
    main.py keeps running. Only successes are cached: once main.py has created
    the key file and certificate, the next alert just works."""
    if sensor in _CLIENTS:
        return _CLIENTS[sensor], ""
    try:
        client = AlertClient(sensor, keys_path=keys_path)
    except (sensor_keys.SensorKeyError, TransportError) as e:
        return None, str(e)
    _CLIENTS[sensor] = client
    return client, ""


def report_connection(sensor: str) -> bool:
    """Startup check for a sensor: print, in one place, whether its alerts will
    land and if not, exactly why. Returns True when main.py answered 200."""
    client, why = client_for(sensor)
    if client is None:
        print(f"[!] Alerts will not be sent yet: {why}")
        return False
    try:
        r = client.health()
    except TransportError as e:
        print(f"[!] {e}")
        return False
    except requests.exceptions.RequestException:
        print(f"[!] Cannot reach main.py at {client.base}")
        print("    Start main.py first, or alerts will not be stored.")
        return False
    if r.status_code == 200:
        print(f"[+] main.py is reachable at {client.base} (TLS pinned). "
              f"Starting watch loop...")
        return True
    print(f"[!] main.py returned HTTP {r.status_code} - alerts may fail")
    return False

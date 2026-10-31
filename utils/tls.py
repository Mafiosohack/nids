"""Self-signed TLS for the dashboard and sensor API, created on first start.

## Why TLS is the default

The dashboard is usually reached across the network the NIDS is watching. On
plain HTTP, anyone positioned on that segment copies an admin login in transit
and gets the defender's view: which of their actions were detected, the learned
map of who talks to whom, and the suppression list. The NIDS would even flag
the ARP spoof used to do it - after the credential was already gone.

## Why self-signed, and why that is not "insecure"

A CA-issued certificate needs a domain name and renewal, which a lab sensor on
192.168.56.x does not have. A self-signed certificate gives the same
encryption; what it lacks is a third party vouching for the server's identity.
That is replaced by the FINGERPRINT, printed at every start:

  * browser: accept the warning once, after checking the fingerprint matches
  * local sensors: read the certificate file directly - nothing to configure
  * remote sensors: pin the fingerprint (NIDS_TLS_FINGERPRINT)

Pinning is stricter than CA trust: a certificate any CA would sign still fails
if it is not this exact one.

The certificate is reused across restarts (so pinned fingerprints stay valid)
and regenerated only when missing or within 30 days of expiring.
"""

import datetime
import hashlib
import ipaddress
import os
import socket
from dataclasses import dataclass
from pathlib import Path
from typing import List

CERT_NAME = "nids-cert.pem"
KEY_NAME = "nids-key.pem"
VALIDITY_DAYS = 825          # the longest lifetime browsers accept
RENEW_BEFORE_DAYS = 30


@dataclass
class TLSMaterial:
    cert_path: str
    key_path: str
    fingerprint: str          # SHA-256 of the DER certificate, AA:BB:... form
    created: bool


def fingerprint_of(cert_path) -> str:
    """SHA-256 fingerprint of a PEM certificate, colon-separated upper hex -
    the same form browsers show, so the operator can compare by eye."""
    from cryptography import x509
    from cryptography.hazmat.primitives.serialization import Encoding
    with open(cert_path, "rb") as fh:
        der = x509.load_pem_x509_certificate(fh.read()).public_bytes(Encoding.DER)
    digest = hashlib.sha256(der).hexdigest().upper()
    return ":".join(digest[i:i + 2] for i in range(0, len(digest), 2))


def local_addresses() -> List[str]:
    """Addresses this host is likely reached at, for the certificate's names."""
    found = {"127.0.0.1", "::1"}
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None):
            found.add(info[4][0])
    except OSError:
        pass
    # The primary outbound address. A UDP "connect" sends nothing; it only asks
    # the OS which local address it would route from.
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("192.0.2.1", 9))
            found.add(s.getsockname()[0])
    except OSError:
        pass
    out = []
    for a in found:
        try:
            out.append(str(ipaddress.ip_address(a.split("%")[0])))
        except ValueError:
            pass
    return sorted(set(out))


def _still_valid(cert_path: Path) -> bool:
    from cryptography import x509
    try:
        with open(cert_path, "rb") as fh:
            cert = x509.load_pem_x509_certificate(fh.read())
    except (OSError, ValueError):
        return False
    left = cert.not_valid_after_utc - datetime.datetime.now(datetime.timezone.utc)
    return left > datetime.timedelta(days=RENEW_BEFORE_DAYS)


def ensure_certificate(directory="tls") -> TLSMaterial:
    """Return the certificate in `directory`, creating it if absent or expiring."""
    d = Path(directory)
    cert_path, key_path = d / CERT_NAME, d / KEY_NAME
    if cert_path.exists() and key_path.exists() and _still_valid(cert_path):
        return TLSMaterial(str(cert_path), str(key_path),
                           fingerprint_of(cert_path), created=False)

    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.x509.oid import NameOID

    d.mkdir(parents=True, exist_ok=True)
    key = ec.generate_private_key(ec.SECP256R1())
    host = socket.gethostname() or "nids"
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, f"NIDS {host}"),
                      x509.NameAttribute(NameOID.ORGANIZATION_NAME,
                                         "NIDS self-signed (verify by fingerprint)")])
    dns = sorted({"localhost", host})
    sans = [x509.DNSName(n) for n in dns] + \
           [x509.IPAddress(ipaddress.ip_address(a)) for a in local_addresses()]
    now = datetime.datetime.now(datetime.timezone.utc)
    cert = (x509.CertificateBuilder()
            .subject_name(name).issuer_name(name)
            .public_key(key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(now - datetime.timedelta(minutes=5))
            .not_valid_after(now + datetime.timedelta(days=VALIDITY_DAYS))
            .add_extension(x509.SubjectAlternativeName(sans), critical=False)
            .add_extension(x509.BasicConstraints(ca=False, path_length=None),
                           critical=True)
            .sign(key, hashes.SHA256()))

    # Key first, owner-only from the moment it exists.
    fd = os.open(str(key_path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "wb") as fh:
        fh.write(key.private_bytes(serialization.Encoding.PEM,
                                   serialization.PrivateFormat.PKCS8,
                                   serialization.NoEncryption()))
    try:
        os.chmod(key_path, 0o600)
    except OSError:
        pass
    with open(cert_path, "wb") as fh:
        fh.write(cert.public_bytes(serialization.Encoding.PEM))
    return TLSMaterial(str(cert_path), str(key_path),
                       fingerprint_of(cert_path), created=True)

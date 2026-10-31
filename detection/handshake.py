"""Is the address on this TCP payload really the one holding the connection?

A packet's source address is whatever its sender wrote. That matters most where
the NIDS names someone: an exploit signature attributes the attack to the
packet's source, and correlation can escalate that name to ACTIVE_INTRUSION.
Forged one-way packets did exactly that to an innocent host (audit 2026-10-31).

What an off-path forger cannot produce is knowledge of the real server's
initial sequence number: the server's SYN-ACK goes to the claimed client, not
to the forger. So a connection verifies when, as seen by this sensor:

  1. the client sent a SYN                         (we know its ISN)
  2. exactly one server ISN answered it            (a second, different SYN-ACK
                                                    is someone forging one)
  3. the client did not reset it                   (a victim's stack RSTs a
                                                    SYN-ACK for a SYN it never sent)
  4. the payload acknowledges the peer's sequence  (within a window; a guess
     space                                          lands outside it)

Limits, stated: an attacker ON the path (who can read the victim's traffic,
e.g. after ARP spoofing - which the ARP rules flag) sees the real ISN and can
forge a verifying session. And a sensor that starts mid-connection, or sees
only one direction, cannot verify anything: those findings still alert, they
just do not name anyone in an incident.
"""

from typing import Optional, Tuple

from utils.bounded import BoundedDict

SYN, ACK, RST = 0x02, 0x10, 0x04
_MOD = 2 ** 32
# How far past the peer's ISN an acknowledgement may plausibly be: 16 MiB of
# data from the peer. A blind guess lands in this window with p = 2^-8; in
# practice the forger also has to beat a "two SYN-ACKs" or "client RST" tell.
ACK_WINDOW = 2 ** 24


def _ahead(ack: int, isn: int) -> int:
    """Bytes `ack` is past isn+1, in sequence space (wraps at 2^32)."""
    return (int(ack) - (int(isn) + 1)) % _MOD


class HandshakeVerifier:
    def __init__(self, max_flows: int = 65536):
        # (client, server, cport, sport) -> state dict
        self._flows = BoundedDict(None, max_keys=max_flows, name="handshakes.verify")

    def __len__(self) -> int:
        return len(self._flows)

    def stats(self) -> dict:
        return self._flows.stats()

    def reset(self) -> None:
        self._flows.clear()

    def observe(self, src: str, dst: str, sport: int, dport: int, flags: int,
                seq: Optional[int], ack: Optional[int]) -> None:
        flags = int(flags)
        syn, is_ack, rst = bool(flags & SYN), bool(flags & ACK), bool(flags & RST)
        if syn and not is_ack:
            if seq is None:
                return
            self._flows[(src, dst, int(sport), int(dport))] = {
                "client_isn": int(seq), "server_isn": None,
                "conflict": False, "reset": False}
            return
        if syn and is_ack:
            st = self._flows.get((dst, src, int(dport), int(sport)))
            if st is None or seq is None or ack is None:
                return
            if int(ack) != (st["client_isn"] + 1) % _MOD:
                return                      # not an answer to the SYN we saw
            if st["server_isn"] is None:
                st["server_isn"] = int(seq)
            elif st["server_isn"] != int(seq):
                st["conflict"] = True       # retransmissions repeat the ISN
            return
        if rst:
            st = self._flows.get((src, dst, int(sport), int(dport)))
            if st is not None and st["server_isn"] is not None:
                st["reset"] = True

    def verify(self, src: str, dst: str, sport: int, dport: int,
               ack: Optional[int]) -> Tuple[bool, str]:
        """Does a payload from src (to dst) belong to a connection src holds?
        Returns (verified, reason) - the reason is written for the analyst."""
        st = self._flows.get((src, dst, int(sport), int(dport)))
        if st is not None:                       # src is the client
            peer_isn, side = st["server_isn"], "server"
        else:
            st = self._flows.get((dst, src, int(dport), int(sport)))
            if st is None:
                return False, ("no TCP handshake observed for this connection: the "
                               "source address may be forged (or the sensor started "
                               "mid-connection / sees one direction only)")
            peer_isn, side = st["client_isn"], "client"
        if st["server_isn"] is None:
            return False, "the server never answered the SYN: no connection was established"
        if st["conflict"]:
            return False, ("two different SYN-ACKs were seen for this connection: "
                           "someone forged one")
        if st["reset"]:
            return False, ("the claimed client reset this connection: its host did "
                           "not open it, so the address is likely forged")
        if ack is None:
            return False, "no acknowledgement number on this packet to check"
        if _ahead(ack, peer_isn) > ACK_WINDOW:
            return False, (f"the payload acknowledges a {side} sequence number this "
                           f"sender could not have seen: likely forged")
        return True, "TCP handshake observed and acknowledged by this sender"

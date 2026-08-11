"""TIER 2 — interactive shell session detection by traffic SHAPE.

## The gap this closes

Every other reverse-shell rule keys off *where* the connection goes:
`BEHAVIOR_UNEXPECTED_OUTBOUND` needs the host in a baseline,
`BEHAVIOR_UNCOMMON_EGRESS_PORT` needs a non-standard port. So a shell on a
non-baselined host calling out to tcp/443 fires nothing at all — and that is
exactly what an attacker picks, because it blends into normal egress.

This rule ignores where the connection goes and looks at what it *is*.

## The signature

An interactive shell has a shape no bulk protocol has:

  * **tiny packets one way** — keystrokes. A human typing `whoami\\n` is 1-7 bytes
    of payload, often one packet per keypress on an unbuffered pty.
  * **larger bursts the other way** — command output. `ls -la` returns kilobytes.
  * **turn-taking** — command, output, command, output. Directions alternate,
    unlike a download (one direction) or a chat protocol (irregular both ways).
  * **long-lived and low-volume** — minutes of connection carrying a few KB. A
    file transfer moves the same bytes in under a second.
  * **human pacing** — gaps between commands are seconds, and irregular. This
    is what separates it from an automated agent polling.

The rule is direction-agnostic on purpose. In a **reverse** shell the victim
connects out, so keystrokes arrive server->client; in a **bind** shell the
attacker connects in, so keystrokes go client->server. The detector measures
both directions and calls whichever side sends mostly-small packets the
"command side", so one rule covers both.

## Why it needs the host sensor

A legitimate SSH session IS an interactive shell — it has the same shape, because
it is the same thing. No amount of threshold tuning separates them, so the rule
does not try. Instead it correlates: `note_authorized_session()` is called when
`host_log_sensor.py` reports an `SSH Login` for that host pair, and a matching
flow is then suppressed. Without that confirmation the rule still fires, but on
a known shell port it says so explicitly rather than claiming certainty — the
alert reads "no corresponding host-sensor login", which is a statement about
evidence, not a verdict.

## Honest limits

  * **Encrypted shells with heavy padding blunt the small-packet signal.** SSH
    pads keystrokes to the cipher block size, and Meterpreter frames its own
    packets. The default `small_payload_bytes` of 64 catches raw shells (netcat,
    `bash -i >& /dev/tcp/...`, python pty) and padded keystrokes, but a transport
    that pads to 512 bytes will not match.
  * **A shell that is never typed into produces no shape.** This rule needs an
    operator at the keyboard; automated tasking looks like beaconing instead,
    which is `BEHAVIOR_C2_BEACON`'s job.
  * **It needs payload lengths, not payloads.** It never inspects content, so
    encryption does not blind it — but it does need to see both directions of
    the flow, which means a SPAN/TAP position that sees return traffic.
"""

import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Set, Tuple

from .findings import Finding

# A payload at or below this many bytes counts as a "keystroke-sized" packet.
# 64 covers raw shells (1-8 bytes) and block-padded interactive SSH keystrokes.
DEFAULT_SMALL_PAYLOAD = 64


@dataclass
class _Side:
    """One direction of a flow."""
    packets: int = 0
    small: int = 0
    payload_bytes: int = 0

    @property
    def small_ratio(self) -> float:
        return self.small / self.packets if self.packets else 0.0


@dataclass
class _Flow:
    client: str            # who sent the SYN
    server: str
    cport: int
    sport: int
    first_ts: float
    last_ts: float
    c2s: _Side = field(default_factory=_Side)
    s2c: _Side = field(default_factory=_Side)
    turns: int = 0          # direction alternations
    last_dir: Optional[str] = None
    gaps: List[float] = field(default_factory=list)
    fired: bool = False

    @property
    def packets(self) -> int:
        return self.c2s.packets + self.s2c.packets

    @property
    def duration(self) -> float:
        return self.last_ts - self.first_ts

    @property
    def turn_ratio(self) -> float:
        return self.turns / self.packets if self.packets else 0.0


class InteractiveShellDetector:
    def __init__(
        self,
        min_packets: int = 40,
        min_duration_sec: float = 20.0,
        small_payload_bytes: int = DEFAULT_SMALL_PAYLOAD,
        min_small_ratio: float = 0.6,
        min_output_ratio: float = 2.0,
        # A real shell alternates roughly once per command, not once per packet:
        # ~6 keystroke packets then ~2 output packets gives 2 turns per 8 packets
        # = 0.24, so a 0.25 floor rejects the very thing being detected. This is
        # a weak secondary signal — it only rules out "burst one way, burst back"
        # — and the discrimination is actually carried by min_small_ratio and
        # min_output_ratio. Kept low deliberately.
        min_turn_ratio: float = 0.15,
        max_bytes_per_sec: float = 2048.0,
        expected_shell_ports: Optional[Set[int]] = None,
        # Static (client, server) pairs that are always allowed to hold an
        # interactive session — an admin jump box to its servers, say. This is
        # the escape hatch for hosts where host_log_sensor.py is NOT deployed:
        # without it, every legitimate SSH session to such a host raises a
        # medium alert. Configure via `authorized_shell_sessions` in assets.json.
        # It is a standing exemption, so keep it narrow — a shell from an
        # allowlisted pair is invisible to this rule.
        authorized_pairs: Optional[Set[Tuple[str, str]]] = None,
        authorized_session_ttl: float = 3600.0,
        cooldown_sec: float = 600.0,
        max_flows: int = 8192,
        clock: Callable[[], float] = time.time,
    ):
        self.min_packets = int(min_packets)
        self.min_duration_sec = float(min_duration_sec)
        self.small_payload_bytes = int(small_payload_bytes)
        self.min_small_ratio = float(min_small_ratio)
        self.min_output_ratio = float(min_output_ratio)
        self.min_turn_ratio = float(min_turn_ratio)
        # A shell is chatty but not fast. A bulk transfer with the same packet
        # count moves orders of magnitude more data.
        self.max_bytes_per_sec = float(max_bytes_per_sec)
        self.expected_shell_ports = set(expected_shell_ports or {22})
        self.authorized_pairs = set(authorized_pairs or ())
        self.authorized_session_ttl = float(authorized_session_ttl)
        self.cooldown_sec = float(cooldown_sec)
        self.max_flows = int(max_flows)
        self.clock = clock

        self._flows: Dict[Tuple, _Flow] = {}
        self._authorized: Dict[Tuple[str, str], float] = {}
        self._last_alert: Dict[Tuple[str, str], float] = {}

    # ── session authorisation (from host_log_sensor.py) ──────────────────────
    def note_authorized_session(self, client: str, server: str,
                                ts: Optional[float] = None) -> None:
        """Record that the host sensor confirmed a real login for this pair.

        This is the correlation that keeps legitimate SSH quiet. It is
        deliberately keyed on the host PAIR and not the 4-tuple: the auth log
        knows who logged in from where, not which ephemeral port they used.
        """
        now = self.clock() if ts is None else ts
        self._authorized[(client, server)] = now

    def authorize_pair(self, client: str, server: str) -> None:
        """Add a standing exemption. Loaded from assets.json at startup."""
        self.authorized_pairs.add((client, server))

    def _is_authorized(self, flow: _Flow, now: float) -> bool:
        if (flow.client, flow.server) in self.authorized_pairs:
            return True
        seen = self._authorized.get((flow.client, flow.server))
        return seen is not None and now - seen <= self.authorized_session_ttl

    # ── flow tracking ─────────────────────────────────────────────────────────
    def open_flow(self, client: str, server: str, cport: int, sport: int,
                  ts: Optional[float] = None) -> None:
        """Called on the SYN, so we know which end initiated the connection."""
        now = self.clock() if ts is None else ts
        key = self._key(client, server, cport, sport)
        self._cap()
        self._flows[key] = _Flow(client=client, server=server, cport=int(cport),
                                 sport=int(sport), first_ts=now, last_ts=now)

    def close_flow(self, a: str, b: str, aport: int, bport: int) -> None:
        self._flows.pop(self._key(a, b, aport, bport), None)
        self._flows.pop(self._key(b, a, bport, aport), None)

    @staticmethod
    def _key(a: str, b: str, aport: int, bport: int) -> Tuple:
        """Direction-independent flow key, so either packet direction finds it."""
        end1, end2 = (a, int(aport)), (b, int(bport))
        return (end1, end2) if end1 <= end2 else (end2, end1)

    def observe_data(self, src: str, dst: str, sport: int, dport: int,
                     payload_len: int, ts: Optional[float] = None
                     ) -> Optional[Finding]:
        """Feed one data-bearing packet. Content is never inspected — only size."""
        if payload_len <= 0:
            return None
        now = self.clock() if ts is None else ts
        key = self._key(src, dst, sport, dport)
        flow = self._flows.get(key)
        if flow is None:
            # No SYN seen (sensor started mid-session). Assume the lower-numbered
            # port is the server, which is right for the overwhelming majority of
            # services and only affects which side we label "client".
            if int(dport) <= int(sport):
                client, server, cport, sport_ = src, dst, int(sport), int(dport)
            else:
                client, server, cport, sport_ = dst, src, int(dport), int(sport)
            flow = _Flow(client=client, server=server, cport=cport,
                         sport=sport_, first_ts=now, last_ts=now)
            self._cap()
            self._flows[key] = flow

        direction = "c2s" if src == flow.client else "s2c"
        side = flow.c2s if direction == "c2s" else flow.s2c
        side.packets += 1
        side.payload_bytes += payload_len
        if payload_len <= self.small_payload_bytes:
            side.small += 1

        if flow.last_dir is not None and flow.last_dir != direction:
            flow.turns += 1
        flow.last_dir = direction
        gap = now - flow.last_ts
        if gap > 0:
            flow.gaps.append(gap)
            if len(flow.gaps) > 200:
                del flow.gaps[:100]
        flow.last_ts = now

        return self._evaluate(flow, now)

    # ── verdict ───────────────────────────────────────────────────────────────
    def _evaluate(self, flow: _Flow, now: float) -> Optional[Finding]:
        if flow.fired:
            return None
        if flow.packets < self.min_packets:
            return None
        if flow.duration < self.min_duration_sec:
            return None
        if not (flow.c2s.packets and flow.s2c.packets):
            return None                      # one-way: a transfer, not a session

        # Whichever side sends mostly small packets is typing. This is what makes
        # one rule cover both reverse shells (server types) and bind shells
        # (client types).
        if flow.c2s.small_ratio >= flow.s2c.small_ratio:
            cmd_side, out_side, cmd_dir = flow.c2s, flow.s2c, "client->server"
            typist, listener = flow.client, flow.server
        else:
            cmd_side, out_side, cmd_dir = flow.s2c, flow.c2s, "server->client"
            typist, listener = flow.server, flow.client

        if cmd_side.small_ratio < self.min_small_ratio:
            return None
        if cmd_side.payload_bytes <= 0:
            return None
        output_ratio = out_side.payload_bytes / cmd_side.payload_bytes
        if output_ratio < self.min_output_ratio:
            return None
        if flow.turn_ratio < self.min_turn_ratio:
            return None

        total_bytes = flow.c2s.payload_bytes + flow.s2c.payload_bytes
        throughput = total_bytes / max(flow.duration, 1e-6)
        if throughput > self.max_bytes_per_sec:
            return None                      # bulk transfer, not a conversation

        if self._is_authorized(flow, now):
            # The host sensor confirmed a real login for this pair. This is
            # somebody's actual SSH session; staying quiet is the correct call.
            flow.fired = True
            return None

        pair = (flow.client, flow.server)
        if now - self._last_alert.get(pair, 0.0) < self.cooldown_sec:
            return None
        self._last_alert[pair] = now
        flow.fired = True

        on_shell_port = flow.sport in self.expected_shell_ports
        if on_shell_port:
            severity = "medium"
            caveat = (f"tcp/{flow.sport} is a normal remote-access port and no "
                      f"corresponding host-sensor login was seen for "
                      f"{flow.client}->{flow.server}. That means either "
                      f"host_log_sensor.py is not deployed on {flow.server}, or "
                      f"this session did not come from a logged login.")
        else:
            severity = "critical"
            caveat = (f"tcp/{flow.sport} is not a remote-access service, so an "
                      f"interactive session here has no legitimate explanation.")

        return Finding(
            rule_id="BEHAVIOR_INTERACTIVE_SHELL",
            src=typist, dst=listener, protocol="TCP", ts=now,
            severity_override=severity,
            message=(f"Interactive shell session: {typist} is typing commands to "
                     f"{listener} over tcp/{flow.sport} "
                     f"({flow.packets} packets across {flow.duration:.0f}s, "
                     f"{cmd_side.small_ratio:.0%} keystroke-sized {cmd_dir}, "
                     f"{output_ratio:.1f}x more output than input). {caveat}"),
            details={
                "shell_port":          flow.sport,
                "command_direction":   cmd_dir,
                "small_packet_ratio":  round(cmd_side.small_ratio, 3),
                "output_input_ratio":  round(output_ratio, 2),
                "turn_ratio":          round(flow.turn_ratio, 3),
                "packets":             flow.packets,
                "duration_sec":        round(flow.duration, 1),
                "bytes_per_sec":       round(throughput, 1),
                "command_bytes":       cmd_side.payload_bytes,
                "output_bytes":        out_side.payload_bytes,
                "on_known_shell_port": on_shell_port,
                "host_sensor_login_seen": False,
                "indicator": "interactive shell (port- and payload-agnostic)",
                "tier": 2,
                "note": ("detected from traffic shape only; no payload was "
                         "inspected, so encryption does not defeat it. Heavy "
                         "packet padding can."),
            },
        )

    # ── housekeeping ──────────────────────────────────────────────────────────
    def _cap(self) -> None:
        if len(self._flows) < self.max_flows:
            return
        oldest = sorted(self._flows, key=lambda k: self._flows[k].last_ts)
        for k in oldest[: len(self._flows) - self.max_flows + 1]:
            del self._flows[k]

    def sweep(self, ts: Optional[float] = None, idle_sec: float = 900.0) -> int:
        """Drop flows that have gone quiet. Returns how many were dropped."""
        now = self.clock() if ts is None else ts
        stale = [k for k, f in self._flows.items() if now - f.last_ts > idle_sec]
        for k in stale:
            del self._flows[k]
        return len(stale)

    @property
    def tracked_flows(self) -> int:
        return len(self._flows)

    def reset(self) -> None:
        self._flows.clear()
        self._authorized.clear()
        self._last_alert.clear()

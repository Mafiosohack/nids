"""Per-host behavioural profile — the substrate the novelty rules stand on.

## Why this exists

Every threshold in this NIDS is currently *absolute*: six admin-port targets is
lateral movement, 50 MB in 300 seconds is exfiltration. Absolute thresholds have
two failure modes that no amount of tuning fixes.

  * **They miss the deliberate operator.** `LATERAL_THRESHOLD = 6` structurally
    cannot see an attacker who makes ONE hop, from a workstation to the one file
    server they actually wanted. That single connection is far more anomalous
    than six connections from the vulnerability scanner - but only if you know
    the workstation has never spoken to that server before.
  * **They fire on the boring outlier.** A nightly backup pushing 80 GB trips
    the exfil rule every night, forever, because 50 MB is 50 MB whether or not
    this host moves 50 MB every day of its life.

Both are the same missing fact: *what is normal for THIS host*. This module
records that, so a rule can ask "is this new?" instead of "is this big?".

## What it is not

It is **not a detector.** Nothing here returns a `Finding` or decides a
severity. It answers questions and reports how much it trusts its own answer;
the rules that consume it (first-contact-on-admin-port, ratio flip, slow-drip
accounting, JA3 novelty) live elsewhere and own those decisions. Keeping the
memory separate from the verdicts is the whole point - there are already four
independent per-host dicts in `main.py`, and a fifth one welded to one rule
would be the fifth mistake rather than the fix.

## Honesty about its own reliability

A profile that started ninety seconds ago thinks every peer is new. That is not
a detection, it is an artefact of when the sensor booted, and shipping it as an
alert would be the classic baseline-tool failure. So every answer carries
`confident` and, when false, a `caveat` saying why:

  * **learning** - fewer than `min_observations` connections, or younger than
    `min_age_sec`. Both gates matter: a chatty host clears the count in a minute
    without having seen a single overnight batch job, and a quiet host ages into
    the window having learned almost nothing.
  * **degraded** - the LRU caps evicted peers, ports or fingerprints for this
    host, so "never seen before" is no longer a claim the store can make. It
    says so rather than quietly lying; a consumer may still fire, but the alert
    should carry the caveat.

`as_details()` on every result exists for exactly that: merge it into a
`Finding`'s details and the evidence about the evidence travels with the alert.

## Periods, not sliding windows

Volume is bucketed into fixed periods (a day by default) rather than a rolling
window, because the question the slow-drip rule needs answered is "how much has
this host sent to that destination *today*, in total" - a slow drip of 2 MB
every ten minutes never trips a windowed threshold but is 288 MB by midnight.

Empty periods are not counted: a host idle for a week does not accumulate seven
periods of zeroes that dilute its baseline. `baseline_periods` is therefore the
number of periods in which the host actually did something.

## Bounds

Everything is capped and evicted least-recently/least-used: hosts, peers per
host, ports per host and per peer, fingerprints per host. A stale-host sweep
runs from inside the observe path on a throttle, so there is no maintenance
tick to forget to call and no docstring promising one that does not exist.

## What it does not see

  * **TCP only, and TCP payload bytes only.** It is fed from
    `DetectionPipeline.on_tcp_packet`, so UDP volume (including DNS) and packet
    framing overhead are absent from every byte figure here. A host exfiltrating
    over UDP is invisible to the volume side of this store; DNS tunnelling is
    `EXFIL_DNS_TUNNEL`'s job, not this one.
  * **Only what crosses the monitored segment.** Two hosts on the same switch
    that the sensor does not span will never appear as peers, so "first contact"
    means first contact *as seen from here*.
  * **It cannot tell a new peer from a re-used address.** DHCP churn and NAT
    both make one IP mean different machines over time. Profiles are keyed on
    IP because that is what the wire carries.
"""

import time
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

LEARNING = "learning"
ESTABLISHED = "established"


def _hour_local(ts: float) -> int:
    return time.localtime(ts).tm_hour


def _bump(counter: Dict[int, int], key: int, cap: int) -> Tuple[bool, int]:
    """Increment `counter[key]`, evicting the least-used keys past `cap`.

    Returns (was_new, evicted). `was_new` is judged BEFORE the increment, which
    is what makes it a novelty answer rather than a count.
    """
    was_new = key not in counter
    counter[key] = counter.get(key, 0) + 1
    evicted = 0
    while len(counter) > cap:
        victim = min(counter, key=lambda k: counter[k])
        del counter[victim]
        evicted += 1
    return was_new, evicted


@dataclass
class Volume:
    """Bytes in each direction, from the profiled host's point of view."""
    sent: int = 0
    recv: int = 0

    @property
    def total(self) -> int:
        return self.sent + self.recv

    @property
    def ratio(self) -> float:
        """sent / received, with a floor of one received byte.

        The floor avoids an infinity that no JSON encoder and no comparison
        wants to handle. It costs nothing in meaning: a host that has sent 5 GB
        and received nothing reports a ratio of 5e9, which is not a number any
        threshold will mistake for balanced traffic.
        """
        return self.sent / max(self.recv, 1)


@dataclass
class PeerRecord:
    """One host this profile has exchanged traffic with.

    `out_ports` are ports on the PEER that this host dialled - the dimension a
    lateral-movement rule needs ("A has talked to B for months, but never on
    445"). Ports the peer dialled on this host are not tracked per peer; they
    are the same handful of service ports for every peer, so they live once at
    profile level in `server_ports`.
    """
    peer: str
    first_seen: float
    last_seen: float
    connections_out: int = 0
    connections_in: int = 0
    out_ports: Dict[int, int] = field(default_factory=dict)
    lifetime: Volume = field(default_factory=Volume)
    current: Volume = field(default_factory=Volume)
    current_period: int = -1
    baseline: Volume = field(default_factory=Volume)
    baseline_periods: int = 0
    ports_evicted: int = 0

    def roll(self, period: int) -> None:
        if period == self.current_period:
            return
        if self.current.total:
            self.baseline.sent += self.current.sent
            self.baseline.recv += self.current.recv
            self.baseline_periods += 1
        self.current = Volume()
        self.current_period = period


@dataclass
class HostProfile:
    """Everything the store remembers about one host."""
    host: str
    first_seen: float
    last_seen: float
    connections: int = 0
    peers: Dict[str, PeerRecord] = field(default_factory=dict)
    client_ports: Dict[int, int] = field(default_factory=dict)
    server_ports: Dict[int, int] = field(default_factory=dict)
    active_hours: List[int] = field(default_factory=lambda: [0] * 24)
    tls_fingerprints: Dict[str, int] = field(default_factory=dict)
    lifetime: Volume = field(default_factory=Volume)
    current: Volume = field(default_factory=Volume)
    current_period: int = -1
    baseline: Volume = field(default_factory=Volume)
    baseline_periods: int = 0
    peers_evicted: int = 0
    ports_evicted: int = 0
    fingerprints_evicted: int = 0

    def roll(self, period: int) -> None:
        if period == self.current_period:
            return
        if self.current.total:
            self.baseline.sent += self.current.sent
            self.baseline.recv += self.current.recv
            self.baseline_periods += 1
        self.current = Volume()
        self.current_period = period

    @property
    def age_sec(self) -> float:
        return self.last_seen - self.first_seen


# ── query results ─────────────────────────────────────────────────────────────
@dataclass
class _Answer:
    """Base for every answer this store gives. `confident` is load-bearing."""
    host: str
    profile_state: str
    confident: bool
    caveat: Optional[str] = None

    def as_details(self) -> dict:
        """Flat, JSON-safe dict to merge into a Finding's `details`.

        `profile_evidence` is added as a single word a dashboard can sort or
        filter on without parsing the caveat prose.
        """
        out = {k: (round(v, 4) if isinstance(v, float) else v)
               for k, v in asdict(self).items()}
        out["profile_evidence"] = "confident" if self.confident else "degraded"
        return out


@dataclass
class ConnectionNovelty(_Answer):
    peer: str = ""
    dst_port: int = 0
    new_peer: bool = False
    new_port_for_peer: bool = False
    new_port_for_host: bool = False
    known_peers: int = 0
    known_ports: int = 0
    peer_first_seen: Optional[float] = None
    peer_connections: int = 0
    profile_age_sec: float = 0.0
    profile_connections: int = 0


@dataclass
class RatioShift(_Answer):
    """This period's send/receive balance against the host's own baseline."""
    current_ratio: float = 0.0
    baseline_ratio: Optional[float] = None
    fold_change: Optional[float] = None
    baseline_periods: int = 0
    current_sent_bytes: int = 0
    current_recv_bytes: int = 0
    period_seconds: float = 0.0


@dataclass
class PeriodVolume(_Answer):
    """Cumulative bytes this period, for slow-drip accounting."""
    peer: Optional[str] = None
    sent_bytes: int = 0
    recv_bytes: int = 0
    period_seconds: float = 0.0
    period_index: int = 0
    baseline_periods: int = 0


@dataclass
class HourNovelty(_Answer):
    hour: int = 0
    hour_observations: int = 0
    total_observations: int = 0
    hour_share: float = 0.0
    new_hour: bool = False
    active_hours: int = 0


@dataclass
class FingerprintNovelty(_Answer):
    fingerprint: str = ""
    new_fingerprint: bool = False
    known_fingerprints: int = 0
    occurrences: int = 0


class HostProfileStore:
    """Remembers what each host normally does. Answers, never alerts.

    `host_filter` decides who is worth profiling - `main.py` passes
    `is_internal`, so the store holds the network being defended rather than
    every address on the Internet that ever sent a packet at it. Without a
    filter every host seen is profiled, bounded by `max_hosts`.
    """

    def __init__(
        self,
        min_observations: int = 20,
        min_age_sec: float = 900.0,
        period_sec: float = 86400.0,
        max_hosts: int = 4096,
        max_peers_per_host: int = 512,
        max_ports_per_host: int = 128,
        max_ports_per_peer: int = 16,
        max_fingerprints_per_host: int = 32,
        host_ttl_sec: float = 30 * 86400.0,
        sweep_interval_sec: float = 300.0,
        host_filter: Optional[Callable[[str], bool]] = None,
        hour_of_day: Callable[[float], int] = _hour_local,
        clock: Callable[[], float] = time.time,
    ):
        self.min_observations = int(min_observations)
        self.min_age_sec = float(min_age_sec)
        self.period_sec = float(period_sec)
        self.max_hosts = int(max_hosts)
        self.max_peers_per_host = int(max_peers_per_host)
        self.max_ports_per_host = int(max_ports_per_host)
        self.max_ports_per_peer = int(max_ports_per_peer)
        self.max_fingerprints_per_host = int(max_fingerprints_per_host)
        self.host_ttl_sec = float(host_ttl_sec)
        self.sweep_interval_sec = float(sweep_interval_sec)
        self.host_filter = host_filter
        self.hour_of_day = hour_of_day
        self.clock = clock
        self._hosts: Dict[str, HostProfile] = {}
        self._last_sweep = 0.0
        self.hosts_evicted = 0

    # ── ingest ────────────────────────────────────────────────────────────────
    def observe_connection(self, src: str, dst: str, dport: int,
                           ts: Optional[float] = None
                           ) -> Optional[ConnectionNovelty]:
        """Record a connection initiation and report what was new about it.

        Both endpoints are updated - the initiator gains an outbound peer and a
        client port, the target gains an inbound peer and a served port - but
        only the INITIATOR gets a novelty answer, because the initiator is the
        actor. Returns None when the initiator is not profiled (filtered out).
        """
        now = self.clock() if ts is None else ts
        dport = int(dport)
        self._maybe_sweep(now)

        answer = None
        p = self._touch(src, now)
        if p is not None:
            state = self._state(p, now)
            rec = p.peers.get(dst)
            new_peer = rec is None
            new_port_for_host = dport not in p.client_ports
            new_port_for_peer = rec is None or dport not in rec.out_ports
            confident, caveat = self._confidence(p, state, new_peer=new_peer)
            answer = ConnectionNovelty(
                host=src, profile_state=state, confident=confident, caveat=caveat,
                peer=dst, dst_port=dport,
                new_peer=new_peer,
                new_port_for_peer=new_port_for_peer,
                new_port_for_host=new_port_for_host,
                known_peers=len(p.peers), known_ports=len(p.client_ports),
                peer_first_seen=None if rec is None else rec.first_seen,
                peer_connections=0 if rec is None else rec.connections_out,
                profile_age_sec=round(p.age_sec, 1),
                profile_connections=p.connections,
            )
            # Record only after the answer is built, so novelty is judged
            # against history rather than against this very packet.
            p.connections += 1
            p.active_hours[self.hour_of_day(now) % 24] += 1
            _, ev = _bump(p.client_ports, dport, self.max_ports_per_host)
            p.ports_evicted += ev
            rec = self._touch_peer(p, dst, now)
            rec.connections_out += 1
            _, ev = _bump(rec.out_ports, dport, self.max_ports_per_peer)
            rec.ports_evicted += ev

        target = self._touch(dst, now)
        if target is not None:
            _, ev = _bump(target.server_ports, dport, self.max_ports_per_host)
            target.ports_evicted += ev
            self._touch_peer(target, src, now).connections_in += 1

        return answer

    def observe_bytes(self, src: str, dst: str, payload_bytes: int,
                      ts: Optional[float] = None) -> None:
        """Attribute payload bytes to both endpoints, in the right direction."""
        payload_bytes = int(payload_bytes)
        if payload_bytes <= 0:
            return
        now = self.clock() if ts is None else ts
        period = self._period(now)

        sender = self._touch(src, now)
        if sender is not None:
            sender.roll(period)
            sender.lifetime.sent += payload_bytes
            sender.current.sent += payload_bytes
            rec = self._touch_peer(sender, dst, now)
            rec.roll(period)
            rec.lifetime.sent += payload_bytes
            rec.current.sent += payload_bytes

        receiver = self._touch(dst, now)
        if receiver is not None:
            receiver.roll(period)
            receiver.lifetime.recv += payload_bytes
            receiver.current.recv += payload_bytes
            rec = self._touch_peer(receiver, src, now)
            rec.roll(period)
            rec.lifetime.recv += payload_bytes
            rec.current.recv += payload_bytes

    def observe_tls_fingerprint(self, host: str, fingerprint: str,
                                ts: Optional[float] = None
                                ) -> Optional[FingerprintNovelty]:
        """Record a TLS client fingerprint (JA3/JA4) seen from `host`.

        Deliberately per-host novelty and nothing else. "This fingerprint
        matches no known browser" needs a curated corpus this project does not
        have; "this host has never presented this fingerprint before" needs only
        the host's own history, which is exactly what is here.
        """
        if not fingerprint:
            return None
        now = self.clock() if ts is None else ts
        p = self._touch(host, now)
        if p is None:
            return None
        state = self._state(p, now)
        new_fp = fingerprint not in p.tls_fingerprints
        confident, caveat = self._confidence(
            p, state, degraded=(p.fingerprints_evicted > 0 and new_fp),
            degraded_reason="TLS fingerprints were evicted from this profile, so "
                            "'never seen before' is no longer certain")
        answer = FingerprintNovelty(
            host=host, profile_state=state, confident=confident, caveat=caveat,
            fingerprint=fingerprint, new_fingerprint=new_fp,
            known_fingerprints=len(p.tls_fingerprints),
            occurrences=p.tls_fingerprints.get(fingerprint, 0),
        )
        p.tls_fingerprints[fingerprint] = p.tls_fingerprints.get(fingerprint, 0) + 1
        while len(p.tls_fingerprints) > self.max_fingerprints_per_host:
            victim = min(p.tls_fingerprints, key=lambda k: p.tls_fingerprints[k])
            del p.tls_fingerprints[victim]
            p.fingerprints_evicted += 1
        return answer

    # ── queries ───────────────────────────────────────────────────────────────
    def profile(self, host: str) -> Optional[HostProfile]:
        return self._hosts.get(host)

    def is_established(self, host: str, ts: Optional[float] = None) -> bool:
        p = self._hosts.get(host)
        if p is None:
            return False
        now = self.clock() if ts is None else ts
        return self._state(p, now) == ESTABLISHED

    def ratio_shift(self, host: str, ts: Optional[float] = None
                    ) -> Optional[RatioShift]:
        """This period's send/receive balance against previous periods.

        A workstation that normally pulls ten times what it pushes and is now
        pushing is the shape of exfiltration at any volume - which is the point,
        since the absolute rule only sees it at 50 MB.
        """
        p = self._hosts.get(host)
        if p is None:
            return None
        now = self.clock() if ts is None else ts
        p.roll(self._period(now))
        state = self._state(p, now)
        base_ratio = p.baseline.ratio if p.baseline_periods > 0 else None
        cur_ratio = p.current.ratio
        fold = None
        if base_ratio is not None and base_ratio > 0:
            fold = cur_ratio / base_ratio
        confident, caveat = self._confidence(p, state)
        if confident and p.baseline_periods < 1:
            confident = False
            caveat = ("no completed period to compare against yet - this host has "
                      "only been observed within the current period")
        return RatioShift(
            host=host, profile_state=state, confident=confident, caveat=caveat,
            current_ratio=cur_ratio, baseline_ratio=base_ratio, fold_change=fold,
            baseline_periods=p.baseline_periods,
            current_sent_bytes=p.current.sent, current_recv_bytes=p.current.recv,
            period_seconds=self.period_sec,
        )

    def period_volume(self, host: str, peer: Optional[str] = None,
                      ts: Optional[float] = None) -> Optional[PeriodVolume]:
        """Cumulative bytes this period, for the whole host or one destination.

        This is the slow-drip accounting surface: 2 MB every ten minutes never
        trips a 50 MB / 300 s window, but it is 288 MB against this number.
        """
        p = self._hosts.get(host)
        if p is None:
            return None
        now = self.clock() if ts is None else ts
        period = self._period(now)
        p.roll(period)
        state = self._state(p, now)
        confident, caveat = self._confidence(p, state)
        if peer is None:
            vol, baseline_periods = p.current, p.baseline_periods
        else:
            rec = p.peers.get(peer)
            if rec is None:
                vol, baseline_periods = Volume(), 0
            else:
                rec.roll(period)
                vol, baseline_periods = rec.current, rec.baseline_periods
        return PeriodVolume(
            host=host, profile_state=state, confident=confident, caveat=caveat,
            peer=peer, sent_bytes=vol.sent, recv_bytes=vol.recv,
            period_seconds=self.period_sec, period_index=period,
            baseline_periods=baseline_periods,
        )

    def hour_novelty(self, host: str, ts: Optional[float] = None
                     ) -> Optional[HourNovelty]:
        """Has this host ever been active at this hour of the day?

        Needs at least one completed period behind it: judging "never active at
        03:00" from a profile that has not lived through an 03:00 is not a
        finding about the host, it is a finding about when the sensor started.
        """
        p = self._hosts.get(host)
        if p is None:
            return None
        now = self.clock() if ts is None else ts
        hour = self.hour_of_day(now) % 24
        state = self._state(p, now)
        total = sum(p.active_hours)
        seen = p.active_hours[hour]
        confident, caveat = self._confidence(p, state)
        if confident and p.baseline_periods < 1:
            confident = False
            caveat = ("fewer than one completed period observed - the profile has "
                      "not lived through a full daily cycle yet")
        return HourNovelty(
            host=host, profile_state=state, confident=confident, caveat=caveat,
            hour=hour, hour_observations=seen, total_observations=total,
            hour_share=(seen / total) if total else 0.0,
            new_hour=(seen == 0),
            active_hours=sum(1 for h in p.active_hours if h),
        )

    # ── introspection / persistence ───────────────────────────────────────────
    def snapshot(self, host: str, top_peers: int = 10,
                 ts: Optional[float] = None) -> Optional[dict]:
        """JSON-safe view of one profile, for the API and the dashboard."""
        p = self._hosts.get(host)
        if p is None:
            return None
        now = self.clock() if ts is None else ts
        peers = sorted(p.peers.values(), key=lambda r: r.last_seen, reverse=True)
        return {
            "host":             p.host,
            "state":            self._state(p, now),
            "first_seen":       p.first_seen,
            "last_seen":        p.last_seen,
            "age_seconds":      round(p.age_sec, 1),
            "connections":      p.connections,
            "known_peers":      len(p.peers),
            "client_ports":     sorted(p.client_ports),
            "server_ports":     sorted(p.server_ports),
            "active_hours":     list(p.active_hours),
            "tls_fingerprints": sorted(p.tls_fingerprints),
            "lifetime_sent_bytes": p.lifetime.sent,
            "lifetime_recv_bytes": p.lifetime.recv,
            "period_sent_bytes":   p.current.sent,
            "period_recv_bytes":   p.current.recv,
            "baseline_periods":    p.baseline_periods,
            "evicted": {"peers": p.peers_evicted, "ports": p.ports_evicted,
                        "fingerprints": p.fingerprints_evicted},
            "top_peers": [
                {"peer": r.peer, "connections_out": r.connections_out,
                 "connections_in": r.connections_in,
                 "ports": sorted(r.out_ports), "last_seen": r.last_seen,
                 "period_sent_bytes": r.current.sent,
                 "period_recv_bytes": r.current.recv}
                for r in peers[:top_peers]],
        }

    def summary(self, limit: int = 100, ts: Optional[float] = None) -> List[dict]:
        """Compact roster, most recently seen first."""
        now = self.clock() if ts is None else ts
        rows = sorted(self._hosts.values(), key=lambda p: p.last_seen, reverse=True)
        return [{
            "host": p.host, "state": self._state(p, now),
            "connections": p.connections, "known_peers": len(p.peers),
            "client_ports": len(p.client_ports), "server_ports": len(p.server_ports),
            "period_sent_bytes": p.current.sent, "period_recv_bytes": p.current.recv,
            "baseline_periods": p.baseline_periods, "last_seen": p.last_seen,
        } for p in rows[:limit]]

    def stats(self) -> dict:
        return {
            "hosts": len(self._hosts),
            "peers": sum(len(p.peers) for p in self._hosts.values()),
            "hosts_evicted": self.hosts_evicted,
            "established": sum(1 for p in self._hosts.values()
                               if self._state(p, p.last_seen) == ESTABLISHED),
            "min_observations": self.min_observations,
            "min_age_sec": self.min_age_sec,
            "period_sec": self.period_sec,
        }

    def to_dict(self) -> dict:
        """Serialize for persistence.

        A baseline that dies with the process is not a baseline - every restart
        would put every host back in `learning` and blind the novelty rules for
        the length of the warm-up. `main.py` owns where this goes.
        """
        return {"version": 1, "period_sec": self.period_sec,
                "hosts": {h: asdict(p) for h, p in self._hosts.items()}}

    def load(self, data: dict) -> int:
        """Restore from `to_dict()`. Returns the number of profiles loaded.

        Unknown or malformed entries are skipped rather than raising: a corrupt
        profile file must not stop the sensor from starting.
        """
        hosts = (data or {}).get("hosts") or {}
        loaded = 0
        for host, raw in hosts.items():
            try:
                peers = {}
                for peer, praw in (raw.get("peers") or {}).items():
                    peers[peer] = PeerRecord(
                        peer=praw["peer"], first_seen=praw["first_seen"],
                        last_seen=praw["last_seen"],
                        connections_out=praw.get("connections_out", 0),
                        connections_in=praw.get("connections_in", 0),
                        out_ports={int(k): v for k, v in
                                   (praw.get("out_ports") or {}).items()},
                        lifetime=Volume(**(praw.get("lifetime") or {})),
                        current=Volume(**(praw.get("current") or {})),
                        current_period=praw.get("current_period", -1),
                        baseline=Volume(**(praw.get("baseline") or {})),
                        baseline_periods=praw.get("baseline_periods", 0),
                        ports_evicted=praw.get("ports_evicted", 0))
                hours = list(raw.get("active_hours") or [0] * 24)[:24]
                hours += [0] * (24 - len(hours))
                self._hosts[host] = HostProfile(
                    host=raw["host"], first_seen=raw["first_seen"],
                    last_seen=raw["last_seen"],
                    connections=raw.get("connections", 0), peers=peers,
                    client_ports={int(k): v for k, v in
                                  (raw.get("client_ports") or {}).items()},
                    server_ports={int(k): v for k, v in
                                  (raw.get("server_ports") or {}).items()},
                    active_hours=hours,
                    tls_fingerprints=dict(raw.get("tls_fingerprints") or {}),
                    lifetime=Volume(**(raw.get("lifetime") or {})),
                    current=Volume(**(raw.get("current") or {})),
                    current_period=raw.get("current_period", -1),
                    baseline=Volume(**(raw.get("baseline") or {})),
                    baseline_periods=raw.get("baseline_periods", 0),
                    peers_evicted=raw.get("peers_evicted", 0),
                    ports_evicted=raw.get("ports_evicted", 0),
                    fingerprints_evicted=raw.get("fingerprints_evicted", 0))
                loaded += 1
            except (AttributeError, KeyError, TypeError, ValueError):
                continue
        return loaded

    def reset(self) -> None:
        self._hosts.clear()
        self._last_sweep = 0.0
        self.hosts_evicted = 0

    # ── internals ─────────────────────────────────────────────────────────────
    def _period(self, now: float) -> int:
        return int(now // self.period_sec) if self.period_sec > 0 else 0

    def _state(self, p: HostProfile, now: float) -> str:
        """Both gates must pass. Count alone lets a chatty host graduate in a
        minute having seen nothing but one burst; age alone lets a near-silent
        host graduate having learned nothing at all."""
        if p.connections >= self.min_observations and p.age_sec >= self.min_age_sec:
            return ESTABLISHED
        return LEARNING

    def _confidence(self, p: HostProfile, state: str, new_peer: bool = False,
                    degraded: bool = False, degraded_reason: str = ""
                    ) -> Tuple[bool, Optional[str]]:
        if state != ESTABLISHED:
            return False, (f"profile still learning: {p.connections} of "
                           f"{self.min_observations} connections, "
                           f"{p.age_sec:.0f}s of {self.min_age_sec:.0f}s")
        if new_peer and p.peers_evicted:
            return False, (f"{p.peers_evicted} peer(s) were evicted from this "
                           f"profile, so 'first contact' cannot be asserted")
        if degraded:
            return False, degraded_reason
        return True, None

    def _touch(self, host: str, now: float) -> Optional[HostProfile]:
        if not host:
            return None
        if self.host_filter is not None and not self.host_filter(host):
            return None
        p = self._hosts.get(host)
        if p is None:
            p = HostProfile(host=host, first_seen=now, last_seen=now,
                            current_period=self._period(now))
            self._hosts[host] = p
            self._cap_hosts()
        elif now > p.last_seen:
            p.last_seen = now
        return p

    def _touch_peer(self, p: HostProfile, peer: str, now: float) -> PeerRecord:
        rec = p.peers.get(peer)
        if rec is None:
            rec = PeerRecord(peer=peer, first_seen=now, last_seen=now,
                             current_period=self._period(now))
            p.peers[peer] = rec
            while len(p.peers) > self.max_peers_per_host:
                victim = min(p.peers, key=lambda k: p.peers[k].last_seen)
                del p.peers[victim]
                p.peers_evicted += 1
        elif now > rec.last_seen:
            rec.last_seen = now
        return rec

    def _cap_hosts(self) -> None:
        while len(self._hosts) > self.max_hosts:
            victim = min(self._hosts, key=lambda k: self._hosts[k].last_seen)
            del self._hosts[victim]
            self.hosts_evicted += 1

    def _maybe_sweep(self, now: float) -> None:
        """Drop hosts unseen for `host_ttl_sec`.

        Driven from the observe path on a throttle rather than from a
        maintenance tick, so there is nothing for a caller to forget to call.
        """
        if self.host_ttl_sec <= 0:
            return
        if now - self._last_sweep < self.sweep_interval_sec:
            return
        self._last_sweep = now
        stale = [h for h, p in self._hosts.items()
                 if now - p.last_seen > self.host_ttl_sec]
        for h in stale:
            del self._hosts[h]
            self.hosts_evicted += 1

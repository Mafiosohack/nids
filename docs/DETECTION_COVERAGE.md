# Detection Coverage

What this sensor detects, what it does not, and how confident each answer is.
Generated live at `GET /rules/coverage` and rendered in the dashboard's **Rule
Coverage** panel; this file is the narrative version.

---

## Scope limit — read this first

**This is network-only detection.** It sees packets on the monitored segment and
infers everything from them. It has **no host, process, or EDR visibility**.

It therefore cannot see:

- code executing in memory, or any action that produces no packets
- an exploit delivered inside a TLS session it cannot decrypt
- local privilege escalation
- anything on a host whose traffic never crosses the monitored segment
- whether a login actually **succeeded** — TCP completes for failed logins too

Host-level evidence comes from `host_log_sensor.py`, which reads auth logs and is
an independent source. Where a network rule guesses at something a host would
know for certain, the alert says so in its `confidence` field rather than
implying certainty (`BREACH_SESSION` and `BRUTEFORCE_*` both do this).

---

## The two exploit tiers, and why they are not one number

The five lab exploits have payload signatures. Three behavioural rules detect
what a successful exploit *causes*. These are different kinds of claim and the
report keeps them apart, because "8 exploit rules" would be a misleading number.

### Tier 1 — payload signatures · NARROW

Fires on the exact wire format of five specific exploits. An exploit variant that
changes the trigger string, or any exploit not on this list, **will not match**.

| rule_id | Exploit | Trigger |
|---|---|---|
| `EXPLOIT_VSFTPD_BACKDOOR` | vsftpd 2.3.4 (OSVDB-73573) | FTP `USER` containing `:)` |
| `EXPLOIT_VSFTPD_BACKDOOR_SHELL` | ″ (confirmation) | connection to tcp/6200 after the trigger |
| `EXPLOIT_SAMBA_USERMAP` | CVE-2007-2447 | shell metachars in the SMB username |
| `EXPLOIT_DISTCC_CMDEXEC` | CVE-2004-2687 | shell strings in the cleartext distcc request |
| `EXPLOIT_UNREALIRCD_BACKDOOR` | CVE-2010-2075 | `AB;` command marker in IRC |
| `EXPLOIT_JAVA_RMI_DESERIALIZATION` | Java RMI gadget chain | `AC ED 00 05` stream header |

Every one carries `generalizes=False` in the catalogue, which is what drives the
"these exploits only" badge on the dashboard.

### Tier 2 — post-exploitation behaviour · BROAD, and ranked above Tier 1

Detects the *consequences* of execution, so it holds regardless of which exploit
produced them — including exploits with no signature written for them.

| rule_id | Indicator | Method |
|---|---|---|
| `BEHAVIOR_INTERACTIVE_SHELL` | someone is typing commands | traffic **shape**: keystroke-sized packets one way, output bursts the other, turn-taking, long-lived, low-volume |
| `BEHAVIOR_UNEXPECTED_OUTBOUND` | reverse shell | a host whose baseline outbound is ~zero initiates a connection |
| `BEHAVIOR_NEW_LISTENING_PORT` | bind shell | a port answering SYN-ACK that is absent from the known-good snapshot |
| `BEHAVIOR_C2_BEACON` | C2 check-in | connections at a regular interval (low interval CV), plus low variation in per-check-in payload size |
| `BEHAVIOR_CLEARTEXT_ON_TLS_PORT` | channel using 443 for egress, not for HTTPS | the first client data packet on a TLS port does not open with a TLS handshake record |
| `BEHAVIOR_UNCOMMON_EGRESS_PORT` | reverse shell (fallback) | outbound to a non-client port, for hosts with no baseline |

### Two features, not one threshold twice — `BEHAVIOR_C2_BEACON`

Timing regularity alone has real benign lookalikes: NTP, health checks,
monitoring pollers and update agents are all metronomic *by design*, and they are
this rule's dominant false-positive source. What separates them from an implant
is what they send. A beacon asking "any tasks?" emits a near-identical request
every check-in, because that request is a fixed structure with a fixed-size
session id. A poller's body tracks whatever it is reporting.

So the rule computes a second coefficient of variation, over the client bytes
sent per check-in, and lets the two features disagree:

| interval CV | payload-size CV | verdict |
|---|---|---|
| low | low | **critical** — two independent regularities agreeing |
| low | high | **medium** — reads as a scheduled job, still recorded |
| low | not measured | **high** (catalogue) — `size_evidence: unavailable` |

The interval CV remains the gate: identical payloads at irregular times are a
client that always sends the same request, not a heartbeat. Byte *counts* are
measured without reading byte *content*, so the second feature survives
encryption exactly as the first one does. A deployment that never feeds sizes in
behaves as the rule did before the feature existed, and says so rather than
implying a measurement it never took.

### "They already have a shell" — how the layers cover it

The port- and baseline-driven rules all key off *where* a connection goes, which
leaves one specific hole: **a shell on a non-baselined host calling out to
tcp/443**. `BEHAVIOR_UNEXPECTED_OUTBOUND` only watches hosts listed in
`assets.json`, and `BEHAVIOR_UNCOMMON_EGRESS_PORT` exempts `COMMON_OUTBOUND_PORTS`
— exactly the ports an attacker picks to blend in.

`BEHAVIOR_INTERACTIVE_SHELL` closes it by ignoring the destination entirely and
matching the shape of a human at a keyboard. Measured behaviour, from the
efficacy harness:

| Setup | On the callback SYN | On the first data packet | Once the attacker types |
|---|---|---|---|
| host not baselined | *nothing* | `BEHAVIOR_CLEARTEXT_ON_TLS_PORT`¹ | `BEHAVIOR_INTERACTIVE_SHELL` |
| baselined, port collapsed to any-destination | *nothing* | `BEHAVIOR_CLEARTEXT_ON_TLS_PORT`¹ | `BEHAVIOR_INTERACTIVE_SHELL` |
| baselined per-destination | `BEHAVIOR_UNEXPECTED_OUTBOUND` | + `BEHAVIOR_CLEARTEXT_ON_TLS_PORT`¹ | + `BEHAVIOR_INTERACTIVE_SHELL` |

¹ only if the channel is not wrapped in real TLS. Commodity tooling frequently
is not — 443 is chosen for the egress filter, not the protocol — but a careful
operator defeats this rule, and nothing here should be read as covering that
case. JA3/JA4 fingerprinting is what addresses it, and it is **not implemented**.

The three are complementary and worth having all of them, because they need
different amounts of evidence:

- a **baseline** catches the callback *immediately*, before a single byte of
  data, but only if the destination is genuinely new for that host;
- the **TLS-handshake check** needs one data packet and no baseline at all, but
  only sees the case where the attacker did not bother with real TLS;
- the **shape rule** always catches it whatever the encryption, but not until
  somebody actually sits down and uses the shell.

The TLS check only judges flows whose SYN it observed. Mid-stream, an encrypted
TLS record (content type 23) is indistinguishable from cleartext to a first-byte
test, so a sensor that started mid-session renders no verdict rather than a
guess. That single guard is what keeps a three-byte test from becoming an
alert-fatigue machine, and the efficacy harness has a case pinning it.

**Legitimate SSH is the unavoidable false positive**, because an SSH session *is*
an interactive shell — same shape, same thing. No threshold separates them, so
the rule does not try. It stays quiet only when something vouches for the
session:

1. `host_log_sensor.py` reports an `SSH Login` for that host pair (preferred —
   `main.py` feeds this back automatically), or
2. the pair is listed in `authorized_shell_sessions` in `assets.json` (the
   escape hatch for hosts with no host sensor).

With neither, a legitimate SSH session raises a **medium** alert whose text says
it could not confirm a login — a statement about missing evidence, not a verdict.
If that is noisy in your environment, the fix is to deploy the host sensor or add
the pair; turning the rule off would restore the tcp/443 hole.

Tier 2 is listed **first** in the coverage report and its findings are published
**before** Tier 1 findings from the same packet, because the generalising
evidence is the one an analyst should read first.

`efficacy_harness.py` proves this rather than asserting it: the case
*"Reverse shell, UNKNOWN exploit"* forbids every `EXPLOIT_*` rule from firing and
still requires `BEHAVIOR_UNEXPECTED_OUTBOUND`.

**Tier 2 needs a baseline.** Without a `servers` block in `assets.json` it falls
back to a learning window, which bakes in whatever was happening at the time.
Findings from a learned baseline carry `baseline_source: "learned"` — treat them
accordingly. Take the port snapshot (`ss -ltnp`) from a host you trust, **before**
testing, not after.

---

## Rule IDs and discriminating features

Every rule has a unique, stable ID, and the catalogue declares the detail keys
each alert **must** carry. `Finding.missing_discriminators()` enforces it; the
efficacy harness fails a case if any alert lacks them, even when the detection
itself succeeded. That is what stops a rule decaying back into a generic label.

### Scans — one ID per technique

Previously every stealth variant produced one `"Stealth Scan"` alert.

| rule_id | nmap | Discriminators on the alert |
|---|---|---|
| `TCP_NULL_SCAN` | `-sN` | `tcp_flags`, `tcp_flags_hex`, `dst_port`, `scan_technique` |
| `TCP_FIN_SCAN` | `-sF` | ″ |
| `TCP_XMAS_SCAN` | `-sX` | ″ |
| `TCP_SYNFIN_SCAN` | — | ″ |
| `TCP_SYN_SCAN` | `-sS` | `distinct_ports`, `ports_per_sec`, `handshakes_completed`, `half_open_resets` |
| `TCP_CONNECT_SCAN` | `-sT` | ″ |

`-sS` and `-sT` both send bare SYNs, so port count cannot separate them. The
discriminator is the client's third packet: **RST** after SYN-ACK = half-open
(`-sS`); **ACK** = the handshake completed (`-sT`). Both counts go on the alert so
the classification is auditable, not asserted.

### Brute force — split on the measured rate

| rule_id | Condition | On the alert |
|---|---|---|
| `BRUTEFORCE_STANDARD_RATE` | ≥ 0.2 attempts/sec | `attempts_per_sec`, `rate_class`, `service`, `dst_port`, `window_seconds` |
| `BRUTEFORCE_LOW_RATE` | < 0.2 attempts/sec | ″ |

The threshold is one attempt per five seconds: faster than a human retyping a
password, slower than any tool at default settings. One long observation window
(900 s) feeds both; the classification comes from the rate actually measured, and
the figure is on the alert so it can be checked.

`tcp/8080` was deliberately **removed** from the watched ports. It is an
application port, so counting connections to it as login attempts turned ordinary
API polling into a HIGH brute-force alert — a false positive the benign
regression suite caught.

### Fragment reassembly

`nmap -f` splits the TCP header across 8-byte fragments: fragment 0 carries the
ports, fragment 1 carries the flags byte (TCP offset 13). Neither matches a scan
signature on its own, which is why `-f` previously evaded detection entirely.

`detection/reassembly.py` reassembles before any rule runs, so a fragmented `-sF`
normalises to exactly the same `TCP_FIN_SCAN` output as an unfragmented one —
asserted field-by-field in `tests/test_reassembly.py`. Overlapping fragments
resolve first-writer-wins (the BSD/Linux policy) and the overlap is counted in
`stats` rather than dropping the datagram, which would hand the attacker a bypass.

---

## Correlation

Grouping key: **source IP, and only source IP.** Window: configurable,
default 600 s (`NIDS_CORRELATION_WINDOW`).

An incident opens only on **forward progression through the kill chain**: two
events from one source where the later one sits at a later stage. Consequences:

- repeated reconnaissance never opens an incident (one `nmap -sS` trips five
  distinct rules, and under the old engine that alone produced a CRITICAL
  "Multi-Vector Attack")
- backwards order (C2 then recon) is not progression
- stage-less events (floods, ARP) attach to an open incident as evidence but
  never open one
- INFO-severity events neither open nor attach

Severity is the worst member alert, escalated to CRITICAL at three or more
stages. Grouping is computed over events sorted by **timestamp**, not arrival
order, so concurrent delivery cannot change the result.

Full before/after analysis: `docs/CORRELATION_DIAGNOSIS.md`.

---

## Alert retention

| Severity | TTL |
|---|---|
| CRITICAL, HIGH | indefinite (evicted last, only under capacity pressure) |
| MEDIUM | 24 h |
| LOW | 6 h |
| INFO | 1 h |

Identical alerts — same `(rule_id, src, dst, dst_port, severity)` — inside the
dedup window (default 300 s) collapse into **one record with a `count`**. 200
identical port-scan pings become one record with `count: 200`, not 200 records.

Every eviction and collapse is written to an audit log exposed at
`GET /alerts/retention`, so nothing disappears without a trace. `/status` reports
both `alerts_stored` (records) and `alerts_represented` (true event count) —
showing only the record count would under-report an attack.

---

## Verifying all of it

```bash
venv/Scripts/python.exe -m unittest discover -s tests -t tests   # 112 tests
venv/Scripts/python.exe efficacy_harness.py                      # 25 TP/FP/FN cases
```

The harness has a case per efficacy finding, including the benign scenarios,
and exits non-zero on any regression.

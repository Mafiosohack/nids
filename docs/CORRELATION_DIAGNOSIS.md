# Correlation Engine — Diagnosis (pre-redesign)

Read of `main.py` as of commit `6c29c7b`, lines 464–465, 741–744, 752–813, 1965–1966.
This documents what the **old** engine did and why its output looked arbitrary.
The replacement is `detection/correlation.py`; see `DETECTION_COVERAGE.md`.

---

## What the old engine actually was

```python
correlation_tracker = defaultdict(lambda: {"events": []})      # line 464

# inside generate_alert(), line 741:
if not _from_correlation and src:
    with correlation_lock:
        correlation_tracker[src]["events"].append(alert_type)   # <- a STRING
    correlate_events(src)
```

`correlate_events(src)` then tested four hardcoded `set.issubset` rules and one
catch-all (`len(unique) >= 3`), and cleared the source's event list on any match.

## Answers to the five diagnostic questions

### 1. What key does it group on?

**Source IP only** — `correlation_tracker[src]`, where `src` is whatever the
firing detector passed as the alert's `src` argument. Grouping on source IP is
the right choice; the problem is that `src` is **not consistently the actor**:

| Detector | line | value passed as `src` | is that the attacker? |
|---|---|---|---|
| `detect_port_scan` | 854 | scanning host | yes |
| `detect_brute_force` | 926 | attacking host | yes |
| `detect_data_staging` | 1152 | `dst` — the *collecting* host | it's the victim/compromised host |
| `detect_distributed_scan` | 1329 | `None` | never correlates at all |
| `detect_arp_spoofing` | 1291 | the **spoofed** IP | no — the attacker forged it |
| `detect_c2_beacon` | 1102 | internal `initiator` | yes (compromised host) |

So the same tracker bucket mixes attacker-attributed and victim-attributed
events. An ARP-spoofing alert filed under the victim's IP can combine with the
victim's own benign traffic alerts to trip the ≥3 rule.

### 2. What time window does it use?

**None. There is no time window anywhere in the correlation path.**

`events` is a plain `list` of strings. Nothing carries a timestamp, nothing is
ever pruned by age. The list is only emptied when:

- a correlation rule fires (line 795 / 807), or
- it exceeds `CORRELATION_EVENT_LIMIT = 15` entries (line 811), or
- an admin calls `DELETE /alerts` (line 1966).

This is the single biggest cause of the "random" behaviour. A port scan at
09:00 and a brute force at 17:00 are correlated into one `Attack Chain`
CRITICAL, because from the engine's point of view they are adjacent list
entries. Incident membership is a function of *how much unrelated traffic
happened since the last flush*, not of time.

### 3. Does it account for attack-stage ordering (recon → exploit → C2)?

**No.** The check is `required_events.issubset(unique)` — pure set containment
over an unordered `set`. Ordering is never inspected, yet the alert text asserts
one:

```python
({"Port Scan", "Brute Force Attempt"}, "Attack Chain", "critical",
 "Port Scan followed by Brute Force")          # line 753-758
```

Brute force at t=0 and a port scan at t=60 produce the message "Port Scan
**followed by** Brute Force". The claim is not backed by anything the code
checked. Note also that the kill-chain machinery already exists elsewhere
(`KILL_CHAIN_ORDER`, `ATTACK_MAP`, `update_intrusion_state`, line 555–661) —
the correlation engine simply never consults it.

### 4. Is grouping deterministic under concurrent events?

**No**, for two separate reasons.

**(a) Non-atomic check-then-clear.** `correlate_events` snapshots the list under
the lock (line 780-781), *releases* it, evaluates, calls `generate_alert`, and
only then re-acquires the lock to clear (line 794-795). Two threads can both
pass `issubset` on the same state before either clears, emitting two identical
CRITICAL incidents for one chain. This is a live race, not theoretical: alerts
arrive from at least three concurrent producers —

- the sniffer thread (`_sniffer_worker` → `process_packet`),
- FastAPI threadpool workers serving `POST /alert` from `live_ids_v2.py`,
- the same endpoint from `host_log_sensor.py` and `cloud_log_sensor.py`.

**(b) Order-dependent state.** Because a fired rule *clears the whole list*,
the result depends on arrival order, not event content. Sequence
`[Port Scan, SYN Flood, Brute Force]` fires `Coordinated Attack` and discards
the brute force; `[Port Scan, Brute Force, SYN Flood]` fires `Attack Chain` and
discards the SYN flood. Same three facts, two different incidents, and in both
cases evidence is silently thrown away.

### 5. Why does it "look random" in practice?

Three compounding effects, all visible in the code:

1. **The catch-all fires on ordinary recon.** Line 799: `if len(unique) >= 3` →
   `Multi-Vector Attack`, CRITICAL. But a *single* `nmap -sS` against one host
   trips `TCP Signature` (line 1437), `Port Scan` (1444), `Stealth Scan` (1445),
   `Slow Scan` (1448) and often `Traffic Anomaly` (1449) — **five distinct
   alert types from one command**. So one scan reliably manufactures a
   CRITICAL "multi-vector attack." That is the behaviour reported as "fires
   after a few attacks occur."
2. **No window** (Q2) means the buckets accumulate across the whole session, so
   which combination trips first depends on unrelated background traffic.
3. **Clear-on-fire** means after the first incident the source starts from
   empty, so a *genuine* later chain (exploit → C2) often does **not**
   correlate. The engine is loudest on the least interesting event and silent
   on the most interesting one.

## Secondary defects found in the same read

- **Unbounded memory.** `correlation_tracker` is a `defaultdict` keyed by source
  IP with no eviction. `hping3 --rand-source` (already a supported test case)
  creates one dict entry per spoofed source, forever.
- **Evidence loss.** Only the alert *type string* is stored — no timestamp, no
  destination, no port, no `details`. The incident record therefore cannot show
  an analyst *what* happened, and nothing downstream (MITRE mapping, reporting)
  has a specific feature to work from.
- **`CORRELATION_EVENT_LIMIT` is a memory guard masquerading as logic.** Hitting
  15 events silently deletes the source's entire history mid-attack.

## What the redesign must therefore fix

| # | Defect | Fix in `detection/correlation.py` |
|---|---|---|
| 1 | No time window | configurable `window_sec`, events pruned by age on every ingest |
| 2 | `src` is sometimes the victim | callers pass an explicit `actor`; rule catalogue declares actor semantics |
| 3 | No stage ordering | incident requires **forward progression** through `KILL_CHAIN_ORDER` |
| 4 | Recon-only trips CRITICAL | same-stage repetition can never open an incident |
| 5 | Check-then-clear race | ingest + evaluate + state mutation in one critical section |
| 6 | Order-dependent result | advancement computed over the buffer **sorted by event timestamp** |
| 7 | Clear-on-fire loses history | incidents are long-lived records that *extend*, never reset |
| 8 | Evidence discarded | full event objects retained (rule_id, ts, dst, port, details) |
| 9 | Unbounded growth | LRU cap on tracked sources and on events per incident |

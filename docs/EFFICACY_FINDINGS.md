# Rule-Layer Efficacy Findings — and what closed each one

Findings from the manual TP/FP/FN testing round (repeated trials per attack type
against the lab: Kali attacker, Metasploitable target, Ubuntu sensor), and the
change that closed each. Every row has an automated case in
`efficacy_harness.py` so it cannot regress silently.

| # | Finding | Status | Closed by | Test |
|---|---|---|---|---|
| 1 | Stealth scans all collapsed into one generic "stealth scan" alert | fixed | `TCP_FIN_SCAN` / `TCP_NULL_SCAN` / `TCP_XMAS_SCAN` / `TCP_SYNFIN_SCAN`, each carrying decoded flags | `test_scan_rules.py::TestScanSignatureIDs` |
| 2 | `-sS` and `-sT` indistinguishable | fixed | `TCP_SYN_SCAN` vs `TCP_CONNECT_SCAN`, split on the client's third packet (RST = half-open, ACK = completed) | `test_scan_rules.py::TestPortScanDetector` |
| 3 | Fragmented scans (`-f`) evaded detection entirely | fixed | IPv4 reassembly before rule matching (`detection/reassembly.py`) | `test_reassembly.py::test_fragmented_fin_scan_normalises_to_the_same_signature` |
| 4 | Standard-rate and low-rate brute force produced the same generic alert | fixed | `BRUTEFORCE_STANDARD_RATE` / `BRUTEFORCE_LOW_RATE`, split on measured attempts/sec, figure recorded on the alert | `test_scan_rules.py::TestBruteForceRateSplit` |
| 5 | DDoS / flood detection correct | kept as-is | `FLOOD_SYN`, `FLOOD_PACKET_RATE` — given rule IDs and discriminators, logic unchanged | — |
| 6 | Zero exploit-stage detection | fixed | Tier 1: five payload signatures. Tier 2: three behavioural rules that generalise beyond them | `test_exploit_tier1.py`, `test_exploit_tier2.py` |
| 7 | Benign traffic correct but nothing locked it in | fixed | four benign scenarios asserting zero non-INFO alerts **and** zero incidents | `test_benign_regression.py` |
| 8 | Correlation engine grouped on an unintentional basis | fixed | rebuilt on an explicit spec after diagnosis | `test_correlation.py` (17 cases) |
| 9 | Alert store had fixed capacity, no eviction policy | fixed | severity-weighted TTL + dedup-with-count + audited eviction | `test_retention.py` |

## Regressions found *by* this work

Two real defects that the new tests caught, neither previously known:

**`tcp/8080` in the brute-force watch list.** Repeated connections to an
application/API port on 8080 were counted as login attempts, producing a HIGH
`BRUTEFORCE_STANDARD_RATE` on ordinary polling. Caught by
`test_benign_regression.py::test_polling_a_few_different_api_ports_is_clean`.
Fixed by removing 8080 from `DEFAULT_BRUTEFORCE_PORTS` — HTTP auth cannot be
judged from connection counts anyway, that needs response codes this sensor does
not parse.

**Non-ASCII characters back in alert text.** An arrow in the "Active Intrusion"
message raised `UnicodeEncodeError` on a cp1252 stdout — the same class of bug
fixed in July, reintroduced because the fix was "keep the strings ASCII", a
promise every future call site has to remember. Now handled once, at the single
point alerts are written out (`main.py::safe_print`), so it cannot recur.

## Not addressed

- **Distributed scan / slow scan / ARP / DNS-tunnel / exfil rules** were outside
  the scope of this round. They now have rule IDs and discriminators, but their
  detection logic is unchanged and untested by the harness.
- **UDP and ICMP paths** carry no Tier 1/Tier 2 inspection. Tier 1 signatures are
  TCP-payload only.
- **`BEHAVIOR_NEW_LISTENING_PORT` needs a baseline snapshot** to say anything at
  all. Without a `servers` block in `assets.json` it is silent by design —
  silence beats guessing at what "normal" was.
- **Encrypted payloads** defeat Tier 1 entirely. That is the structural reason
  Tier 2 is ranked above it.

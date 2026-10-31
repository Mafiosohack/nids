# NIDS Systems Map

Scope: capture → features → detection → alert → storage → correlation → dashboard, plus the cloud and host log sensors where they POST into `/alert`. Code at `89f6d06`.
Tag meanings: **[SIM]** = I checked the claim by driving the real classes with synthetic input (scripts kept outside the repo, venv python). Anything else is from reading the code.

---

## 1. Inventory

### Control flow

| Stage | Entry → path |
|---|---|
| Capture | `POST /control/start` main.py:2308 → thread `_sniffer_worker` main.py:1675 → `sniff(prn=process_packet, timeout=1)` in a loop main.py:1693-1700. Handling is synchronous, in the capture thread. |
| Features (packet) | Parsed inline in `process_packet` main.py:1537-1602. IP fragments are reassembled first main.py:1556-1563. Payload is cut to 4096 bytes main.py:1586. |
| Features (ML) | Separate process. `live_ids_v2.process_packet` live_ids_v2.py:570 → `Flow`, keyed `(ip_a, ip_b, proto)` with **no ports** live_ids_v2.py:327-335 → `build_feature_vector` live_ids_v2.py:298. |
| Detection | The remaining detectors in main.py (1031-1531), plus `PIPELINE.on_tcp_packet` pipeline.py:151-266. ML runs `run_inference` live_ids_v2.py:436 at packet 5 live_ids_v2.py:600-602. Cloud runs `process_flow` cloud_log_sensor.py:354. Host runs `process_line` host_log_sensor.py:298. |
| Alert gen | `Finding` → `PIPELINE.publish` pipeline.py:269. Legacy callers use `generate_alert` main.py:980, which passes no `ts`, so the alert is stamped with arrival time (pipeline.py:272). Sensors reach it via `POST /alert` main.py:2281. |
| Storage | `AlertStore.add` pipeline.py:304 (in memory). Then `on_alert` → `db_insert_alert` main.py:942, which opens a new sqlite connection and commits under a global `db_lock` (main.py:302-315), still in the capture thread. |
| Correlation | Two independent stocks: `CorrelationEngine.ingest` via pipeline.py:316,341, and `intrusion_tracker` via main.py:952 → 847. |
| Dashboard | Polls `/alerts?limit=200` and `/intrusions` every 2 s, and `/status` and `/incidents` every 4 s (nids_dashboard.html:640,790-793,821). |

### Buffers between stages

| Between | Buffer | Cite |
|---|---|---|
| NIC → detection | libpcap/Npcap ring only. No application queue. | main.py:1694-1700 |
| ML sensor → main | None. The POST is synchronous inside the capture callback, with a 2 s timeout. | live_ids_v2.py:526-531 |
| Log files → sensors | The file on disk. The read offset `processed_lines` lives only in memory. | cloud_log_sensor.py:118,413-426; host_log_sensor.py:110,349-359 |
| publish → correlation | None. Correlation runs inline, and only for alerts that were *stored* (not deduped). | pipeline.py:304-316 |
| Alerts → UI | `AlertStore` (≤5000 records) and sqlite (≤50 000 rows). | main.py:85,495 |

### Constants on the path (value @ line)

- **main.py:** NIDS_IFACE "ens37" @61. Port scan 6 ports/5 s @63-64. DDoS 200 pkts/3 s @70-71. SYN flood 100, window 5, ratio 5 @73-75. Brute-force threshold 8 @77. UDP amp 50/10 s @82-83. MAX_ALERTS_STORED 5000 @85. CORRELATION_WINDOW 600 @90. DEDUP_WINDOW 300 @95. Brute-force rate split 0.2, long window 900 @98-99. BREACH_MEMORY 180 @119. Lateral 6 targets/120 s @124-125. Exfil 50 MB/300 s @130-131. Beacon 6 hits / 900 / 0.25 / 2.0 @134-137. Staging 5 sources / 20 MB / 300 s @146-148. Handshake TTL 30, cap 20 000 @151-152. IMDS 6/20 s @164-165. DNS 60 s window, 15 queries, label 45, qname 60, entropy 3.6 @169-173. ARP cooldown 60 @178. Profile min obs 20, min age 900, period 86400, max hosts 4096, save every 300 @199-208. Distributed scan 20 s / 4 sources / 3 ports @212-214. Slow scan 1800/12 @218-219. Baseline bucket 10, α 0.3, min samples 6, k 4.0, min count 20 @223-227. BREACH_SESSION_BYTES 12000 @1128. ALERT_RETENTION 50 000 @495; prune runs when an alert id is a multiple of 1000 @943. Inline cooldowns: breach 180 @1169, lateral 90 @1195, reverse shell 120 @1221, exfil 120 @1262, staging 180 @1287, IMDS 90 @1369, DNS 120 @1392, distributed 60 @1465, slow scan 300 @1486, baseline 120 @1513.
  - **Inert:** `BEACON_WINDOW` @135 and `BEACON_MIN_INTERVAL` @137 are never read. `SYN_FLOOD_WINDOW` @74 is only echoed into alert text @1061.
- **detection/:** retention TTLs: info 1 h, low 6 h, medium 24 h, high/critical none (retention.py:38-44). Eviction log 1000 retention.py:48. Correlation max sources 4096, 500 events per incident (correlation.py:65-66). Port scan cooldown 30, 4096 sources (scan_rules.py:109-110). Brute force cooldown 60, 8192 keys (scan_rules.py:252-254). Outbound learn 300, cooldown 120 (exploit_tier2.py:67-68). Listening-port context 300, cooldown 300 (exploit_tier2.py:164-165). Beacon: window 900, min interval 2.0, cooldown 300, 8192 pairs, 4 size samples, size CV 0.15 (exploit_tier2.py:296-309). Shell: 40 pkts, 20 s, 64 B small packet, 0.6 small ratio, 2.0 output ratio, 0.15 turn ratio, 2048 B/s, auth TTL 3600, cooldown 600, 8192 flows (pipeline.py:74-75; interactive_shell.py:115-137). TLS cooldown 300, 8192 flows (tls_shape.py:146-147). Profile caps: 512 peers, 128 ports, 16 per peer, 30-day TTL, sweep 300 (host_profile.py:295-300). Reassembly: 30 s timeout, 1024 buffers, 128 fragments (reassembly.py:37-40). vsftpd confirm window 180 (exploit_tier1.py:243).
- **live_ids_v2.py:** POST timeout 2 @91. Min packets 5, max flow age 120, cleanup every 30 @97-99. HIGH_CONFIDENCE 0.80 @101. Decision threshold 0.5, overridable from metadata @373,421.
- **cloud_log_sensor.py:** URL and key hardcoded @39-40. Poll 5 @47. Port scan 10/60 @53-54. Exfil 50 MB/300 @56-57. Recon 20/60 @59-60. Brute 10/60 @62-63. Lateral 8/120 @65-66. UDP 30/30 @68-69. Travel gap 600 @71.
- **host_log_sensor.py:** URL and key hardcoded @42-43. Poll 5 @50. Failed logins 5/120 @53-54. Breach after 3 failures @58. Session TTL 3600 @66.
- **Dashboard:** POLL_MS 2000 (nids_dashboard.html:640). Fetch abort 4000 (nids_dashboard.html:682). Row limit 200 (nids_dashboard.html:821).

### Drop / evict / discard points

| Where | What is lost | Observable? |
|---|---|---|
| Kernel/pcap buffer | Packets the capture thread could not keep up with | **Silent.** `packets_captured` counts callbacks only (main.py:1538). |
| Reassembly timeout / eviction / oversize | Fragment buffers | Counted (reassembly.py:87-94), but `/status` exposes only `datagrams_reassembled` (main.py:1990). The docstring at reassembly.py:29-30 says "reported through /status". |
| Payload > 4096 bytes | Signature bytes | Silent (main.py:1586) |
| Dedup collapse | The event becomes `count++`, and its message/details overwrite the record's | Logged at repeats 2, 10, 100, 1000, … (retention.py:124-128). **The collapsed event never reaches correlation** (pipeline.py:305-307). |
| TTL / capacity eviction | Alert records | Logged to `eviction_log` (retention.py:157,183) |
| sqlite prune | Rows older than max id − 50 000 | Silent (main.py:543-549) |
| Correlation LRU / per-incident cap | Buffers / events | Counted (correlation.py:317,339) |
| Detector `_cap()` (brute force, beacon, shell, TLS, port scan) | Tracking state | **Silent** (scan_rules.py:127-129,315-321; exploit_tier2.py:424-430; interactive_shell.py:335-340; tls_shape.py:258-263) |
| `src_semantics == "unknown"` | Event skipped for correlation | Silent (pipeline.py:335-336) |
| ML flow expiry by `start_time` | Flows, including still-active ones | Printed only (live_ids_v2.py:357-362) |
| ML / log-sensor POST failure | The alert | Printed to the sensor's stdout, not retried (live_ids_v2.py:542-546; cloud_log_sensor.py:152-157) |
| Log truncation or rotation | Lines, until the file grows past the old line count | **Silent** (cloud_log_sensor.py:413-414; host_log_sensor.py:350-351) |
| Profile save failure | Learned baseline | Only OSError, TypeError and ValueError are caught (main.py:792). Any other exception ends the saver thread (main.py:804-805). |

---

## 2. Stocks

| Stock | Unit | Bound | Inflow | Outflow | On overflow |
|---|---|---|---|---|---|
| `AlertStore._records` | records | 5000 (main.py:85) plus a TTL by severity | new dedup key, retention.py:137 | TTL retention.py:144; capacity retention.py:163 | Lowest severity first, oldest `last_seen` within a severity. Logged. The capacity check sorts every record on each add (retention.py:173-177). |
| sqlite `alerts` | rows | 50 000, pruned every 1000 ids (main.py:495,943) | main.py:942 | main.py:543-549; `DELETE /alerts` main.py:2252 | Rows are dropped oldest-id first, silently. `count` is written once and never updated (main.py:498-511). |
| `CorrelationEngine._buffers` | events per actor | 600 s window; 4096 actors LRU (correlation.py:237,329-339) | correlation.py:236 | correlation.py:237,319-327 | LRU drop, counted |
| `CorrelationEngine._incidents` | incidents | **Unbounded** (correlation.py:200,292) | correlation.py:292 | `reset()` only | Grows with uptime |
| `DetectionPipeline.published`, `.incidents_opened` | alert dicts | **Unbounded** (pipeline.py:147-148,312,352) | every stored alert | `reset()` only | Grows with uptime. Only tests and the harness read it (efficacy_harness.py:582; tests/test_benign_regression.py:47). |
| `intrusion_tracker` | hosts × stages | **Unbounded, no time outflow** (main.py:811) | main.py:860-873 | `DELETE /alerts` main.py:2253-2254; process restart | Status only ever ratchets up. `announced` is set once per host (main.py:875-877). |
| main.py per-IP trackers (`ddos`, `syn`, `udp_amp`, `lateral`, `exfil`, `staging`, `metadata`, `dns`, `distributed`, `slowscan`, `baseline`, `breach_hot_targets`, `arp_ip_mac`, `_rogue_seen`, `_recent_pc_alerts`) | keys | **Unbounded key count** (main.py:596-632,614) | first packet from a source | none (inner lists are pruned, keys never are) | Memory grows with the number of distinct sources |
| `syn_tracker[src]` counts | SYNs, ACKs | **Unbounded in time.** The 5 s window is never applied. | main.py:1052,1629 | reset only when the alert fires (main.py:1064-1065) | A lifetime counter behaves like a rate |
| `lateral.dests`, `staging.srcs`, `exfil.bytes` | hosts / bytes | Only reset after a quiet gap longer than the window, or on alert (main.py:1193,1256-1259,1281-1284) | per packet | alert or quiet gap | The "window" really means "since the last quiet gap" |
| `handshake_tracker` | pending SYNs | TTL applied only once the table reaches 20 000 (main.py:1313-1317) | main.py:1621 | SYN-ACK pop main.py:1625; prune | At the cap, every SYN rescans the whole table |
| `PortScanDetector._handshakes` | pending SYNs | **Unbounded** (scan_rules.py:122,147) | each SYN | client ACK/RST after a SYN-ACK (scan_rules.py:167,174) | Unanswered SYNs are never removed |
| `HostProfileStore._hosts` | profiles | 4096 LRU + 30-day TTL (host_profile.py:725-746) | pipeline.py:211,237 | LRU / TTL | Counted. **Nothing consumes it** (pipeline.py:207-210). |
| ML `flow_table` | flows | Age 120 s from `start_time`, swept every 30 s, no size cap (live_ids_v2.py:351-359) | live_ids_v2.py:338-348 | sweep | Unbounded between sweeps |
| Cloud trackers, `processed_lines` | keys / sets / bytes | **Unbounded.** `ports`/`dests` sets are never windowed; `exfil.bytes` is only cleared on alert. | cloud_log_sensor.py:192,211,229,266 | reset on alert only | Thresholds become lifetime counts |

---

## 3. Loops

| # | Type | Path | Stock | Metric pair (expected sign) | Falsifier |
|---|---|---|---|---|---|
| L1 | Balancing | retention.py:114-131 dedup | AlertStore records | `stats.deduped` ↑ while `stats.added` stays flat during a sustained single-key attack (retention.py:85-88, `/alerts/retention`) | `added` increments on repeats inside 300 s |
| L2 | Balancing | retention.py:140 → 163-187 | AlertStore records | `records` pinned at 5000 while `evicted_capacity` ↑ | `records` > capacity |
| L3 | Balancing, **unmeasured delay** | TTL runs only inside `add()` (retention.py:112). `sweep()` retention.py:211 has no caller. | AlertStore records | age of the oldest INFO record vs time since the last stored alert (+) | No INFO record ever older than 3600 s on a quiet sensor |
| L4 | Balancing | per-key cooldowns, e.g. scan_rules.py:151,282; main.py:1115-1123 | alert emission rate | minimum gap between same-(rule, src) alert timestamps ≥ cooldown | a gap smaller than the cooldown |
| L5 | Balancing | open incident absorbs new events (correlation.py:225-230) | open incidents | `stats.extended` ≫ `stats.opened` per actor (`/incidents`) | two open incidents for one source inside the window |
| L6 | Balancing, **unmeasured delay** | EWMA/var update runs whether or not an alert fired (main.py:1522-1531). A bucket closes only on that source's *next* SYN (main.py:1504-1506). | per-source baseline | `baseline_mean` in consecutive `ANOMALY_TRAFFIC_BASELINE` alerts for one source rises toward `burst`, and alerts stop (details main.py:1519-1520) | repeated alerts at a constant `burst` with a flat `baseline_mean` |
| L7 | Balancing (load shedding by silent drop) | synchronous callback (main.py:1696). Per-packet O(n) list rebuilds (main.py:1255,1280). Capacity sort per add (retention.py:173). sqlite commit per alert (main.py:942). | pcap buffer | `packets_captured`/s vs `alerts_generated`/s from `/status` (−) under constant offered load | processed pps unchanged during an alert storm. **Partly SPECULATIVE:** offered load and kernel drops are not measured. |
| L8 | Balancing (silent) | ML POST inside the capture callback (live_ids_v2.py:600-602 → 526-531) | ML capture throughput | "NIDS POST timed out" lines vs "[ML]" inference lines per minute (live_ids_v2.py:480,546) (−) | inference rate unchanged while timeouts occur |
| L9 | Balancing, intended, **broken** | host "SSH Login" → `note_authorized_session(src, dst)` main.py:1007-1008 → suppression interactive_shell.py:176-180,279-283 | shell alerts | SSH Login count per (client, server) vs `BEHAVIOR_INTERACTIVE_SHELL` per the same pair (−) | Both fire for one session. **The code guarantees this:** `dst` is the syslog *hostname* (host_log_sensor.py:302,211) and suppression compares against the server *IP*. **[SIM]:** suppressed when keyed by IP, alert fires when keyed by hostname. |
| L10 | Reinforcing, **cut (gain 0)** | `CORRELATED_INCIDENT` → publish → return early (pipeline.py:354→329-330). `ACTIVE_INTRUSION` → `generate_alert` → `update_intrusion_state` → return, because stage is None (main.py:883→952→855). | correlation input | correlator `stats.ingested` vs non-correlation stored alerts (equal) | `ingested` exceeds that count |
| L11 | Reinforcing | attacker traffic is learned into profiles (pipeline.py:211,237) | host profiles | **SPECULATIVE.** There is no consumer, so no metric pair exists. | — |

**Missing loops** (a quantity that drifts with nothing measuring or correcting it):
- The per-IP key counts of every stock marked *Unbounded* in §2.
- `intrusion_tracker` stage age.
- Kernel packet drops.
- ML train/serve feature divergence.
- The sqlite `count` column vs the in-memory count (main.py:498-511).
- Whether the profile-saver thread is still alive (main.py:804-805).
- Log-sensor 401s, which appear only in sensor stdout.

---

## 4. Delays

| Delay | Between | Length | In loop |
|---|---|---|---|
| pcap buffering | NIC → `process_packet` | UNMEASURED | L7 |
| ML maturity | first packet → verdict | 5 packets (live_ids_v2.py:97); wall time UNMEASURED | L8 |
| ML POST | verdict → stored | ≤2 s (live_ids_v2.py:91); typical UNMEASURED | L8 |
| Log write + poll | auth.log / VPC log → `/alert` | 5 s poll (host_log_sensor.py:50; cloud_log_sensor.py:47) + write latency UNMEASURED (AWS delivery delay is external) | L9 |
| Arrival-time stamping | event time → correlation `ts` | = the two delays above, because sensor alerts carry no `ts` (main.py:1010-1019; pipeline.py:272) | L5 |
| Dedup window (rolling) | first alert → next correlatable alert | 300 s, extended by every repeat (retention.py:116-118) | L1 → starves L5 |
| Correlation window | event → buffer drop | 600 s (main.py:90) | L5 |
| Breach arming | brute-force alert → hot target expires | 180 s (main.py:119) | — |
| TTL expiry | `last_seen` + TTL → eviction | TTL + time until the next `add()` (UNMEASURED) | **L3** |
| EWMA bucket close | bucket end → evaluation | 10 s + time until that source's next SYN (UNMEASURED) | **L6** |
| Shell verdict | SYN → evaluation | ≥20 s and ≥40 packets (pipeline.py:74-75) | L9 |
| Profile flush | learn → disk | 300 s (main.py:208) | — |
| Dashboard | store → screen | 2–4 s (nids_dashboard.html:790-793) | — |

**Balancing loops that contain an unmeasured delay:**
- **L3 (TTL):** eviction lag is proportional to how quiet the network is.
- **L6 (EWMA):** a gap of hours counts as one sample, so raising `BASELINE_ALPHA` makes an attacker's rate become "normal" in fewer SYNs, not in fewer seconds. This is the only loop here with a goal-seeking controller where tuning can plausibly overshoot.
- **L8 (ML backpressure).**
- **L9 (shell suppression):** its delay also races the 20 s verdict.

L7 and L8 correct by losing data, not by oscillating.

---

## 5. Layer map

| # | Decisions on this path |
|---|---|
| 12 | All thresholds and cooldowns in §1. The dedup window (main.py:95), correlation window (main.py:90), BREACH_MEMORY (main.py:119), FLOW_MIN_PACKETS (live_ids_v2.py:97). |
| 11 | MAX_ALERTS_STORED (main.py:85). ALERT_RETENTION (main.py:495). Dashboard `limit=200` (nids_dashboard.html:821). Correlation 4096/500 (correlation.py:65-66). Detector caps of 8192. HANDSHAKE_MAX_ENTRIES (main.py:152). Payload cut at 4096 (main.py:1586). |
| 10 | Dedup runs *before* correlation (pipeline.py:304-316). Capture handling is synchronous with no queue (main.py:1694-1700). Sensors connect over a synchronous HTTP POST (main.py:2281). ML flow key without ports (live_ids_v2.py:327-335). Two separate correlation stocks (main.py:811 vs correlation.py:196-200). |
| 9 | Rolling dedup (retention.py:116-118). TTL checked on write only (retention.py:112). Incident extends from its last event (correlation.py:226). ML flow expiry by start time, not activity (live_ids_v2.py:357). |
| 8 | Cooldowns / `_pc_should_fire` (main.py:1115). Capacity eviction (retention.py:163). LRU caps. No capture backpressure. ML POST timeout (live_ids_v2.py:91). |
| 7 | Correlation output excluded from input (pipeline.py:329-330). **Beacon fed twice** per connection (pipeline.py:202-205 and main.py:1308→1244). |
| 6 | Eviction log (main.py:2130-2149). `/status` has no drop count (main.py:1971-1993). Dashboard redraws only when the id set changes, so count increases never repaint (nids_dashboard.html:826-828). Alert text quotes windows that are not enforced (main.py:1061,1199). |
| 5 | Rule catalogue (rules.py:111-418). Forward-progression promotion (correlation.py:257-277). `intrusion_status` (main.py:892-905). `src_semantics` (pipeline.py:335). |
| 4 | `NIDS_*` env overrides, `assets.json` (main.py:635-743), `PipelineConfig` (pipeline.py:50-95). The log sensors cannot be configured (cloud_log_sensor.py:39-40; host_log_sensor.py:42-43). |
| 3 | Per actor, forward kill-chain progression on a monitored segment (correlation.py:26-40; main.py:847-852) |
| 2 | Source IP = actor (correlation.py:7). RFC1918 = trusted internal (main.py:112-115); the cloud sensor treats all of 172/8 as internal (cloud_log_sensor.py:163). One shared pre-shared key for all sensors (main.py:1774-1786). Network-only visibility (rules.py:18-21). |
| 1 | UNVERIFIED from code |

### Layer inversions (a shallow knob that is load-bearing for a deep capability)

| Knob (#11/#12) | Deep capability it silently decides | Evidence |
|---|---|---|
| `ALERT_DEDUP_WINDOW_SEC` 300, rolling | **Whether correlation and breach detection see a persistent attack at all.** Repeats extend one record forever, so correlation sees only the first event. That event leaves the 600 s buffer while the attack continues. `note_bruteforce_target` is never refreshed, so the hot target expires after 180 s. | retention.py:116-118; pipeline.py:305-307; correlation.py:237; main.py:1604-1608,1164. **[SIM]:** 20 min of brute-force repeats → 1 stored record; brute force at t=0 plus login at t=660 → `buffered`, no incident. |
| `NIDS_SENSOR_KEY` | Whether the host and cloud sensors are connected. They ignore the env var, so following the startup warning (main.py:2391-2393) turns every one of their POSTs into a 401. | cloud_log_sensor.py:40; host_log_sensor.py:43; main.py:244,1781 |
| `FLOW_MIN_PACKETS` 5 + infer-once | Whether `count`/`srv_count` carry information. Live, they always equal 5 (live_ids_v2.py:313-314,600-602). Training uses whole-flow packet counts (collector/cicids_loader.py:249,264-265), and `count` has 48 % relative importance (models/cicids_live_meta.json:232). | as cited |
| Dashboard `limit=200`, insertion order | Whether a retained CRITICAL is *visible*. Each port gets its own dedup key (retention.py:104) and signature findings have no cooldown (scan_rules.py:61-83), so any scan of more than 200 ports pushes it off screen. | main.py:2127; nids_dashboard.html:821 |
| `BEACON_MIN_INTERVAL` 2.0 (the detector default) | Whether C2 beacon detection works in production: the second feed at SYN-ACK inserts an RTT-sized interval. | exploit_tier2.py:351-352. **[SIM]:** 12 check-ins at a 30 s period → 1 finding pipeline-only, **0** with the production double feed. |
| `HANDSHAKE_MAX_ENTRIES` 20 000 | Per-SYN cost under flood: O(table) on every SYN once full | main.py:1313-1317 |

---

## 6. Targeted questions

1. **Separate event source, gated by the store.** Correlation is fed inline from `publish`, not read from `AlertStore` (pipeline.py:316,341). The store cap (5000) and the sqlite cap (50 000) do not bound it. What bounds it is dedup: only non-deduped alerts are ingested (pipeline.py:305-307). The dedup window is rolling (300 s, retention.py:116-118) and the correlation window is 600 s (main.py:90). So under sustained activity the effective correlation horizon for a stage is 600 s from its *first* alert, however long that stage lasts. Correlation state is not persisted or restored; restart restores alerts while bypassing it (main.py:2378-2383).
2. **Priority-aware, then LRU by `last_seen`** in `AlertStore` (TTL by severity; capacity sort on severity rank, then `last_seen`; retention.py:38-44,173-177). sqlite: FIFO by id (main.py:543-549). Correlation: LRU by newest event, open incidents exempt (correlation.py:333-336). Profiles: LRU for hosts and peers, LFU for ports (host_profile.py:95-108,718,727).
3. **Written to the store, not re-correlated.** `CORRELATED_INCIDENT` and `ACTIVE_INTRUSION` are stored and persisted (pipeline.py:304,313-314). Both have tier `correlation` (rules.py:398,407) and are skipped at pipeline.py:329-330. They do re-enter `update_intrusion_state` (main.py:952) but are ignored there because they have no stage (main.py:855).
4. **Silently, for capture.** No kernel/pcap drop counter exists. `packets_captured` counts processed packets (main.py:1538). Fragment drop counters exist but are not surfaced (main.py:1990).
5. **No.** `/model/info` serves training-time metadata only (main.py:2074-2109), and live_ids_v2.py records no feature statistics. Divergence exists by construction (§5, `count`). The metadata's `serving_note` is stale: it says the sensor loads `rf_live.pkl` (models/cicids_live_meta.json:83), but it loads `cicids_live.pkl` (live_ids_v2.py:59,83). The module docstring claiming "no distribution mismatch" (live_ids_v2.py:16-21) is stale too.
6. **No.** The `alerts` table has no status, owner or ack column (main.py:349-371). No endpoint mutates a single alert; the only write is a bulk `DELETE /alerts` (main.py:2245-2261).

---

## 7. Findings (ranked by cost × likelihood)

| # | Finding | Cite | What an analyst sees | Cheapest fix |
|---|---|---|---|---|
| 1 | Beacon detector is blinded by a double feed | pipeline.py:202-205 + main.py:1308→1244 → exploit_tier2.py:351-352 **[SIM]** | No `BEHAVIOR_C2_BEACON`, ever, for any beacon whose SYN-ACK is seen with RTT < 2 s. Tests stay green because they drive the pipeline without main.py. | Delete the main.py feed (§8). |
| 2 | Dedup starves correlation and breach detection | §5 row 1 **[SIM]** | An 11-minute hydra run followed by a login produces a HIGH brute force and a login alert but **no incident** and no `BREACH_SESSION`. | #9: compare dedup to `first_seen` instead of `last_seen` (retention.py:116), so a sustained stream re-publishes every 300 s (< 600 s correlation window). Set BREACH_MEMORY_SEC ≥ 360 (dedup 300 + brute-force cooldown 60). Cost: one record per 5 min per stream. |
| 3 | Log sensors ignore `NIDS_SENSOR_KEY` and `NIDS_URL` | cloud_log_sensor.py:39-40; host_log_sensor.py:42-43 | After securing the deployment as the startup warning asks, host and cloud stages vanish. The dashboard looks quiet; a 401 appears only in the sensor terminal. | #12: read the env vars as live_ids_v2.py:89-95 does. |
| 4 | Legitimate admin SSH escalates to ACTIVE INTRUSION | L9 key mismatch **[SIM]**; "SSH Login" maps to `initial_access` (rules.py:453); shell alert src = typist (interactive_shell.py:306); main.py:897; correlation.py:257-277 | With the host sensor deployed: SSH Login (medium), then Interactive Shell (medium), then a CORRELATED_INCIDENT and **"ACTIVE INTRUSION: <admin IP>" critical**. Whether real SSH meets the shape thresholds is UNVERIFIED. | #12: add a host-IP setting to host_log_sensor.py and send it as `dst` (host_log_sensor.py:211). |
| 5 | `intrusion_tracker` has no outflow; lateral/staging/exfil windows are fictional | main.py:811,860-878,1193,1256-1259,1281-1284 | With uptime: any host with one past `initial_access` (e.g. a login) plus any later post-access alert becomes critical. An admin touching 6 internal hosts over days gets "Lateral movement … in 120s". A host that was announced once is **never announced again**. | #9: prune `lateral.dests` with its timestamps. For the tracker, see §8. |
| 6 | Per-port signature storm stalls capture and hides criticals | scan_rules.py:61-83 (no cooldown); retention.py:104,173-177; main.py:942; nids_dashboard.html:821 **[SIM]:** 20 000 records took 46 s in `AlertStore` alone | During `nmap -sF -p-` the sensor falls behind (silent drops). The dashboard's 200 rows are all FIN-scan mediums, and an earlier CRITICAL is not on screen. | #8: add a per-(rule, src, dst) cooldown on signature findings, as `PortScanDetector` has (scan_rules.py:151). #11: dashboard also fetches `/alerts?severity=critical`. |
| 7 | FLOOD_SYN uses a lifetime counter | main.py:1050-1065 | `nmap -sS` on more than 100 ports raises "SYN flood … window_seconds 5" at critical. `-sS` sends RSTs, not ACKs, so the ratio check passes. | #9: timestamp-window it like `detect_ddos` (main.py:1035). |
| 8 | ML scores a 5-packet prefix of a host-pair aggregate | §5 row 3; live_ids_v2.py:327-335 | One ML verdict per host pair per ~120–150 s. A 1000-port scan is judged once, on 5 packets. | **#10 required:** any fixed N makes `count` a constant, so no #12 value fixes it. Score on flow expiry or FIN, which replay mode already does (live_ids_v2.py:628-631), and add ports to the key. |
| 9 | Unbounded stocks | §2 rows marked *Unbounded* | Memory grows with uptime and distinct sources; nothing reports it. | #11: `deque(maxlen=…)` for `published`, `incidents_opened` and `_incidents`. TTL-prune `_handshakes`. |
| 10 | Restart distorts history | main.py:498-511 (`count` never updated), main.py:2383 → retention.py:110,135-136 (TTL clock reset); the cloud sensor replays every file from line 0 (cloud_log_sensor.py:118) with arrival-time stamps | After a restart, `alerts_represented` under-reports, INFO alerts live another hour, and old cloud events appear as new. | #9: pass the stored `created_at` as `ts` on restore. Persist the count on dedup. |
| 11 | Profile saver can die | main.py:787-798,804-805; no lock in host_profile.py:318 | Baseline silently stops persisting. PLAUSIBLE: a dict can change size during `asdict` while capture writes to it. | #8: catch `Exception` in the saver loop. |
| 12 | Breach bytes are summed across all sessions | main.py:1154-1168 vs the docstring "single session" (main.py:1145-1146) | Brute-force traffic itself may trip "SUSPECTED COMPROMISE" (UNVERIFIED). | #12/#5: key the counter by source port. |

---

## 8. Deletion

| Remove | Why it is safe or better |
|---|---|
| **`detect_c2_beacon` and its call** (main.py:1235-1246,1308) | A coupling to sever. The pipeline already feeds the beacon detector on every outbound SYN (pipeline.py:202-205). The second feed is what makes finding #1 happen. Deleting it is the whole fix. |
| **`intrusion_tracker`, `update_intrusion_state`, the `ACTIVE_INTRUSION` emission** (main.py:808-905,951-953); rebuild `/intrusions` (main.py:2007-2044) from `correlator.incidents()` | This stock duplicates `CorrelationEngine`, with no window, no `src_semantics`, no LRU and no persistence. The engine already escalates to critical at 3 or more stages (correlation.py:111-124). **Trade-off:** the tracker's only unique ability is joining stages more than 600 s apart. That horizon is really "since last restart", and it is the source of finding #5. |
| **Impossible Travel** (cloud_log_sensor.py:298-318,369-370) | Cannot fire. `region_hint` is a pure function of the source IP (cloud_log_sensor.py:168-183), so one IP always yields one region. |
| **Inert knobs** `BEACON_WINDOW`, `BEACON_MIN_INTERVAL` (main.py:135,137), `SYN_FLOOD_WINDOW` (main.py:74) | Nothing reads the first two. The third is only printed and misstates behaviour. |
| **`SCAN_FRAGMENTED` catalogue entry** (rules.py:174-180) | No code emits it; fragments are annotated onto other alerts (main.py:1611-1613). It still counts toward `total_rules` (rules.py:559), overstating coverage. |
| **Uncalled `sweep()` methods** (retention.py:211; interactive_shell.py:342; tls_shape.py:265) and `beacon_tracker` (main.py:610), `tracker_lock` (cloud_log_sensor.py:119) | Dead code. The sweeps imply a maintenance tick that does not exist (L3). Delete them, or call them; do not keep both. |

---

## What I could not determine

- The pcap/Npcap ring size and real drop rate on the sensor VM. L7 cannot be quantified without it.
- sqlite commit latency on the sensor host, i.e. the real per-alert cost at main.py:942.
- Whether real interactive SSH or Meterpreter traffic meets the shell-shape thresholds (finding #4 depends on it). This needs a pcap.
- Whether failed SSH attempts exceed 12 000 bytes within 180 s (finding #12).
- Whether any deployed auth.log uses an IP as the syslog hostname. The repo's generator uses `web-prod-01` (generate_host_logs.py:26).
- AWS VPC flow-log delivery latency (external to the code).
- Whether the profile-saver `RuntimeError` happens in practice; it is timing-dependent.
- Whether a live attack has ever run long enough for the dedup/correlation gap to matter. No runtime logs were available to me.
- Actual memory growth rate of the unbounded stocks under real traffic.
- Whether anything other than the dashboard consumes `/alerts` and would also need the severity fix.
- Meadows #1 (whether to build this at all) and #3 (true goals): code cannot answer these. The docstrings are claims, not evidence.
- Thread-safety of the unlocked main.py trackers against the FastAPI threadpool. They appear to be touched only by the capture thread; I did not trace every endpoint for indirect access.

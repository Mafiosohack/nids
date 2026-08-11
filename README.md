# Network Intrusion Detection System (NIDS)

## 🔐 Overview

A modular Network-Based Intrusion Detection System built in Python. It captures
network traffic, extracts flow-based features, applies machine-learning models,
and generates security alerts for anomalous or malicious behaviour.

Designed for:
- Educational security research
- Lab-based attack simulation
- Cloud log analysis extension
- Future IPS (Intrusion Prevention System) expansion

---

## ⚠️ What is / isn't in the repo

The trained CIC-IDS2017 models (~7 MB) **are committed**, so a fresh clone runs
after just setting up the venv — no dataset required. Excluded (large or
regenerable):

| Path | What | How to get it |
|------|------|---------------|
| `venv/` | Python virtual environment | Create it (see Setup) |
| `data/cicids2017/` | CIC-IDS2017 dataset (~2 GB) | Download — only needed to **retrain** |
| `models/rf_live*.pkl` | Old NSL-KDD model (unused) | Regenerate with `train_random_forest_v2.py` |
| `*.db` | User / alert databases | Auto-created on first run |

So after cloning you only need to (1) set up the venv — then it runs. You need
the dataset **only if you want to retrain**.

---

## 🛠️ Setup

Requires Python 3.10+ (developed on 3.14).

```powershell
# Windows (PowerShell)
python -m venv venv
venv\Scripts\python.exe -m pip install --upgrade pip
venv\Scripts\python.exe -m pip install -r requirements.txt
```

```bash
# Linux / macOS
python3 -m venv venv
venv/bin/python -m pip install --upgrade pip
venv/bin/python -m pip install -r requirements.txt
```

### "ModuleNotFoundError: No module named 'numpy'"

This means you ran the **global** `python` instead of the venv's. Always use the
venv interpreter. Either activate it once per shell:

```powershell
venv\Scripts\Activate.ps1      # Windows  → prompt shows (venv), then `python …` works
source venv/bin/activate       # Linux/macOS
```

…or call it explicitly every time: `venv\Scripts\python.exe …` (Windows) /
`venv/bin/python …` (Linux). If activation *and* the module is still missing,
you skipped `pip install -r requirements.txt`.

---

## ▶️ Running

### 1. Dashboard + API (main.py)

```powershell
$env:NIDS_IFACE = "auto"          # or a real interface name — see below
venv\Scripts\python.exe main.py
```

Then open **http://127.0.0.1:8000**. A default admin is seeded on first run:

- **user:** `admin`  **password:** `nids@admin123`

Change it immediately, or set `NIDS_ADMIN_PASSWORD` before the first run so the
default is never used. (The user DB is `nids_users.db`, created automatically.)

#### Capture interface (`NIDS_IFACE`)

The built-in sniffer defaults to `ens37`, which **only exists on the Ubuntu
sensor VM**. On any other host, set `NIDS_IFACE` or capture silently yields
nothing:

| Value | Meaning |
|---|---|
| `auto` | let scapy pick the default interface (easiest) |
| `ens37` | the lab sensor VM's monitoring NIC |
| `\Device\NPF_{...}` | a specific Windows adapter |

List what this host actually has:

```powershell
venv\Scripts\python.exe -c "from scapy.arch import get_if_list; print(get_if_list())"
```

`POST /control/start` now **rejects** an interface that doesn't exist (listing
the valid ones) instead of reporting `started` and capturing zero packets, and
`GET /status` returns `sniffer_error` whenever the capture thread has died.

#### Reaching the dashboard from another machine

The dashboard talks to whichever origin served it, so browsing to
`http://<sensor-ip>:8000` from another host just works. Two things to set when
the backend is not on your own machine:

```powershell
$env:NIDS_ALLOWED_ORIGINS = "http://192.168.1.10:8000"   # only if you host the HTML elsewhere
```

To point the page at a backend on a different host than the one serving it, add
`?api=http://192.168.1.10:8000` to the dashboard URL.

### 2. ML sensor (live_ids_v2.py)

The sensor serves the CIC-IDS2017 **live-schema multiclass** model
(`models/cicids_live.pkl`) and predicts the specific attack type per flow.

**Offline replay** (no capture stack needed — good for a first test). The demo
pcap is gitignored, so generate it once, then replay it:

```powershell
venv\Scripts\python.exe make_sample_pcap.py                 # writes data/samples/demo_traffic.pcap
venv\Scripts\python.exe live_ids_v2.py --pcap data/samples/demo_traffic.pcap
```

**Live capture** (POSTs alerts to a running main.py):

```powershell
venv\Scripts\python.exe live_ids_v2.py --iface "Ethernet"
```

Live capture needs packet-capture drivers and privileges:
- **Windows:** install [Npcap](https://npcap.com/); run the shell as Administrator;
  pass your real interface name to `--iface` (list them with
  `venv\Scripts\python.exe -c "from scapy.all import get_if_list; print(get_if_list())"`),
  or use `--iface auto`.
- **Linux:** run with `sudo`; the default interface in the code is `ens37`.

If the sensor runs on a **different host than the dashboard**, point it at the
dashboard — otherwise it POSTs alerts to its own loopback and nothing arrives:

```powershell
$env:NIDS_URL = "http://192.168.1.10:8000"   # dashboard host
$env:NIDS_SENSOR_KEY = "<same value main.py uses>"
```

> `start.sh` automates main.py + sniffer + sensor, but is **Linux-only** (uses
> `sudo`, `venv/bin/python`). On Windows, run `main.py` and `live_ids_v2.py` in
> two terminals as shown above.

---

## 🎯 Rule engine, correlation and coverage

Detection logic lives in `detection/` — pure, clock-injectable, unit-tested, with
no scapy or FastAPI imports. `main.py` is the wiring layer (capture in, HTTP out)
and `detection/pipeline.py` is where the detectors are composed. The tests and
the efficacy harness drive that same pipeline, so what they exercise is what
runs live.

**Detection is network-only.** The sensor sees packets on the monitored segment
and infers everything from them: no host, process, or EDR visibility. It cannot
see in-memory execution, an exploit inside TLS it cannot decrypt, local privilege
escalation, or whether a login actually succeeded. Host evidence comes from
`host_log_sensor.py`, a separate source. Full statement of limits:
[`docs/DETECTION_COVERAGE.md`](docs/DETECTION_COVERAGE.md).

### Running the checks

```powershell
venv\Scripts\python.exe -m unittest discover -s tests -t tests    # 112 unit tests
venv\Scripts\python.exe efficacy_harness.py                       # TP/FP/FN report
venv\Scripts\python.exe efficacy_harness.py -v --json             # detail / machine-readable
```

The harness carries one case per efficacy-testing finding — including the benign
scenarios — and exits non-zero on regression. Beyond firing, it also fails any
alert that lacks the discriminating features its rule declares, so a rule cannot
quietly decay back into a generic bucket label.

### Rule IDs

Every alert names a stable `rule_id` from the catalogue in `detection/rules.py`,
and carries the feature that triggered it:

- scans: `TCP_SYN_SCAN`, `TCP_CONNECT_SCAN`, `TCP_FIN_SCAN`, `TCP_NULL_SCAN`,
  `TCP_XMAS_SCAN` — with decoded flags, port counts and handshake evidence,
  instead of one generic "stealth scan"
- brute force: `BRUTEFORCE_STANDARD_RATE` / `BRUTEFORCE_LOW_RATE`, split on the
  **measured** attempts/sec (threshold 0.2/s), with the figure on the alert
- `nmap -f` scans are reassembled before rule matching, so they normalise to the
  same signature as unfragmented ones

### Exploit detection is two tiers, and they are not equivalent

- **Tier 1** — payload signatures for five specific exploits (vsftpd 2.3.4,
  Samba usermap_script, distccd, UnrealIRCd, Java RMI). Narrow by construction:
  change the trigger string and the match is gone.
- **Tier 2** — behavioural, ranked **above** Tier 1: interactive shell sessions,
  unexpected outbound connections (reverse shell), new listening ports vs a
  known-good baseline (bind shell), and interval-variance beaconing (C2). These
  fire on exploitation techniques beyond the five, including ones with no
  signature.

The distinction is visible in `GET /rules/coverage`, in the dashboard's **Rule
Coverage** panel, and in the harness output — never merged into a single number.

### "Has someone already got a shell on our machine?"

Three layers answer this, and they are complementary:

| Situation | What catches it |
|---|---|
| Box dials out to a destination unusual **for that host** | `BEHAVIOR_UNEXPECTED_OUTBOUND` — fires on the callback SYN, before a command is typed |
| Box dials out on an unusual **port** | `BEHAVIOR_UNCOMMON_EGRESS_PORT` — fallback for hosts with no baseline |
| Shell on an **allowed** port (tcp/443) from an unbaselined host | `BEHAVIOR_INTERACTIVE_SHELL` — ignores the destination and matches the *shape* of someone typing |
| Attacker connects **in** to a bind shell | `BEHAVIOR_NEW_LISTENING_PORT`, then `BEHAVIOR_INTERACTIVE_SHELL` |
| Confirmed root / persistence on the box | `host_log_sensor.py` — the only source that *knows* rather than infers |

`BEHAVIOR_INTERACTIVE_SHELL` measures keystroke-sized packets one way, larger
output bursts the other, turn-taking, long-lived and low-volume. It never
inspects payload, so encryption does not defeat it — heavy packet padding can.

**A legitimate SSH session has the same shape, because it is the same thing.**
The rule stays quiet only when something vouches for the session: either
`host_log_sensor.py` reports a matching `SSH Login` (fed back automatically), or
the pair is in `authorized_shell_sessions` in `assets.json`. With neither, real
SSH raises a **medium** alert saying it could not confirm a login. If that is
noisy, deploy the host sensor or allowlist the pair — disabling the rule would
reopen the tcp/443 hole.

### Baselines (`assets.json`)

Tier 2's strongest rules diff against a known-good baseline. Add a `hosts` block
(see `assets.example.json`) — **not just servers**: a workstation with no entry
falls back to the port heuristic, which exempts exactly the ports an attacker
would choose.

Generate a starter inventory instead of hand-writing it:

```powershell
venv\Scripts\python.exe baseline_assets.py --pcap data\samples\quiet_hour.pcap
venv\Scripts\python.exe baseline_assets.py --iface eth0 --seconds 600 -o assets.json
venv\Scripts\python.exe baseline_assets.py --pcap quiet.pcap --collapse-after 0   # stricter
```

Two things to know before trusting the output:

- **A baseline taken from a compromised host bakes the compromise in.** Capture
  from a host you have reason to trust, during a quiet period, and read the file
  before deploying it.
- By default a port seen reaching 5+ destinations is widened to "any
  destination", which keeps browser traffic manageable but means a reverse shell
  on that port no longer trips the outbound rule. `--collapse-after 0` keeps
  every destination explicit — stricter, noisier. The generated file flags which
  ports were widened and what it cost.

### Correlation

Alerts group into incidents by **source IP only**, inside a configurable window
(`NIDS_CORRELATION_WINDOW`, default 600 s). An incident opens only when a
source's alerts show **forward progression through the kill chain** — so repeated
reconnaissance never manufactures one, and a scan → exploit → beacon sequence
becomes a single incident record rather than three unrelated alerts.

See `GET /incidents`, the dashboard's **Correlated Incidents** panel, and
[`docs/CORRELATION_DIAGNOSIS.md`](docs/CORRELATION_DIAGNOSIS.md) for what the
previous engine did and why its output looked arbitrary.

### Alert retention

CRITICAL/HIGH are kept indefinitely; MEDIUM 24 h, LOW 6 h, INFO 1 h. Identical
repeated alerts collapse into one record with a `count` (200 identical scan pings
= one record, `count: 200`). Every eviction and collapse is logged and readable
at `GET /alerts/retention`, so nothing disappears silently.

### New endpoints

| Endpoint | Purpose |
|---|---|
| `GET /incidents` | correlated incidents + the grouping rules in force |
| `GET /rules/coverage` | per-tier rule coverage and the scope limit |
| `GET /alerts/retention` | retention policy, occupancy, eviction/collapse audit log |

| Env var | Default | Purpose |
|---|---|---|
| `NIDS_CORRELATION_WINDOW` | `600` | correlation window, seconds |
| `NIDS_ALERT_DEDUP_WINDOW` | `300` | dedup window, seconds |
| `NIDS_AUTHORIZED_SCANNERS` | *(empty)* | comma-separated IPs whose recon is recorded at INFO |

| `assets.json` key | Purpose |
|---|---|
| `hosts` | per-host `listening_ports`, `allowed_outbound`, or `learn_outbound: true` |
| `authorized_scanners` | IPs whose reconnaissance is recorded at INFO |
| `authorized_shell_sessions` | `[client, server]` pairs exempt from the interactive-shell rule |

---

## 🤖 Retraining the models

Two models are trained on **CIC-IDS2017**:

- **Model A — full flow** (`train_cicids_flow.py` → `models/cicids_flow.pkl`):
  all 78 flow features. Highest accuracy; a reference model (needs a flow-feature
  extractor to serve live).
- **Model B — live schema** (`train_cicids_live.py` → `models/cicids_live.pkl`):
  the 78 features mapped onto the 10+3 packet-derivable features the live sensor
  computes. **This is the model the sensor serves.**

### 1. Get the dataset

Download CIC-IDS2017 from the
[Canadian Institute for Cybersecurity](https://www.unb.ca/cic/datasets/ids-2017.html)
(the `MachineLearningCSV` and `GeneratedLabelledFlows` releases) and place the
8 day-CSVs of each into:

```
data/cicids2017/
├── features/   ← MachineLearningCSV       (78 features + Label)
└── labelled/   ← GeneratedLabelledFlows   (same + Flow ID / IPs / ports / Protocol / Timestamp)
```

### 2. Train

```powershell
venv\Scripts\python.exe train_cicids_flow.py      # Model A  (writes models/cicids_flow*.{pkl,json})
venv\Scripts\python.exe train_cicids_live.py      # Model B  (writes models/cicids_live*.{pkl,json} + encoder)
```

Each does a stratified 75/25 hold-out and prints an honest per-class report.
Quick smoke run on a subsample: add `--nrows 200000`. Evaluate without saving:
add `--no-save`.

### 3. Verify

```powershell
venv\Scripts\python.exe test_live_inference.py                       # offline sensor check
venv\Scripts\python.exe live_ids_v2.py --pcap data/samples/demo_traffic.pcap
```

The dashboard's **Model Comparison** panel reads the freshly written metadata via
`/model/info`, so retraining updates the UI numbers automatically.

> The older NSL-KDD pipeline (`train_random_forest_v2.py` → `models/rf_live.pkl`)
> is retained but no longer wired into the sensor.

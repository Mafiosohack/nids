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

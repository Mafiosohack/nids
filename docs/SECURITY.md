# Security notes for this repository

This is a defensive security tool published in a public repository. Two
different things need care: what the tool does on your network, and what this
*repository* discloses about the network it was developed on. This file covers
the second.

---

## Known exposure: `nids_users.db` in git history

**Status: present in public history. Not remediated by rewrite. Impact assessed
as low — see below.**

The SQLite user database was committed before `*.db` was added to `.gitignore`:

| | |
|---|---|
| Blob | `2d36330ad5f9ef83920301cd09e2ce2bc42b4cc0` (28 KB) |
| Commits | `53d177f` "Updated NIDS project", `0a744d8` "NIDS cleanup + honest ML pipeline" |
| Current tree | removed |
| History | **still reachable** — `git cat-file -p 2d36330` on any clone |

Contents:

- `users`: one row — `admin`, role `admin`, password stored as an **unsalted
  SHA-256** hash (the legacy format, predating the current PBKDF2 scheme)
- `sessions`: one bearer token, expired 2026-05-04

**Why the impact is low.** The stored hash is the SHA-256 of `nids@admin123` —
the default password that was documented in the README at the time. It therefore
discloses nothing that the repository did not already state in plain text, and
the session token is long expired.

**Why it still matters.** Had the password been changed before that commit, a
real credential would now be public permanently, and unsalted SHA-256 is
recovered offline in seconds. This is the standard failure mode: `.gitignore`
prevents *future* commits, it does not remove what is already published. Adding
the ignore rule felt like a fix and was not one.

### Why history was not rewritten

Purging the blob requires `git filter-repo` or BFG plus a force-push. That
rewrites every commit SHA in the repository, breaking existing clones, forks and
any references to commit hashes. Given the leaked value is a documented default
and the token is expired, the disruption outweighs the benefit.

**This calculus changes** if the DB had ever held a real password, a valid
token, or a non-default account. If you ever commit a live secret, do not weigh
convenience: rotate the secret first (rotation is what actually fixes it — the
copy is already distributed), then decide about the rewrite separately.

To purge it if you later decide to:

```bash
pip install git-filter-repo
git filter-repo --path nids_users.db --invert-paths
git push --force origin main      # breaks every existing clone and fork
```

---

## Changed 2026-08-11: no more published default password

The seeded `admin` account previously used a hardcoded `nids@admin123`.

A default credential in a public repository is not a default — it is a published
credential. Every deployment that did not override it could be logged into by
anyone who read the README, and the leaked DB above confirms the default was
actually in use.

Now: when `NIDS_ADMIN_PASSWORD` is unset, `main.py` generates a 20-character
random password per deployment (~115 bits, ambiguous characters excluded so it
can be transcribed) and prints it **once** at startup. Only the PBKDF2-HMAC-SHA256
hash (200k iterations) is stored, so it cannot be recovered afterwards — if you
miss it, delete `nids_users.db` and restart.

## Unchanged: the sensor pre-shared key

`NIDS_SENSOR_KEY` still falls back to a fixed default
(`sensor-key-change-me-in-production`), deliberately. `main.py` and the three
sensors must agree on this value, and randomising it server-side would silently
break every sensor not restarted with the new one.

Scope of the exposure: the key authenticates `POST /alert` only. Knowing it lets
someone **inject false alerts**; it does not let them read alerts, which requires
a user session. On an isolated lab segment that is acceptable. On any routable
network, set `NIDS_SENSOR_KEY` on the server and all sensors. Startup warns
whenever the default is live.

---

## Files that must never be committed

`.gitignore` enforces these. The reasoning matters more than the list:

| Path | Why |
|---|---|
| `assets.json`, `assets.*.json` | The real inventory: internal IPs, gateway/host **MAC addresses**, per-host **listening-port snapshots**, permitted egress, and which host pairs may hold a shell. This is precisely the reconnaissance an attacker would otherwise have to run a scan to obtain — and catching that scan is what this tool is for. Commit `assets.example.json` (placeholders only) instead. |
| `baseline.json`, `*.assets.json` | Output of `baseline_assets.py` — same content, same reason. |
| `*.db`, `*.sqlite*` | User accounts, password hashes, session tokens, alert history. |
| `*.pcap`, `*.pcapng`, `*.cap` | Captured traffic can contain credentials, session tokens and real payloads. |
| `.env` | Secrets. |
| `host_logs/`, `*.log` | Real auth logs — usernames, source IPs, login times. |

Before pushing, a quick check that nothing sensitive slipped in:

```bash
git status --short                       # untracked files about to be added
git diff --cached --name-only            # what is actually staged
git log --all --pretty=format: --name-only --diff-filter=A \
  | sort -u | grep -iE '\.db$|\.env$|\.pem$|\.key$|assets\.json'
```

The third command searches **all history**, not just the working tree, which is
the check that would have caught `nids_users.db` at the time.

---

## What this repository intentionally does contain

- `data/raw/KDDTrain+.txt`, `KDDTest+.txt` (22 MB) — the NSL-KDD public research
  dataset. Committed on purpose: `compare_models.py` and
  `collector/dataset_loader.py` need it, and a fresh clone should be able to run
  them. Publicly redistributable.
- `models/cicids_*.pkl` (~7 MB) — trained CIC-IDS2017 models, so a clone runs
  without the ~2 GB dataset.
- Lab IP addresses (`192.168.56.0/24`) in tests, docs and
  `assets.example.json` — RFC1918 host-only addresses that describe nothing
  reachable.

## Reporting

This is a lab/educational project with no production deployment and no security
contact. If you find something, open an issue.

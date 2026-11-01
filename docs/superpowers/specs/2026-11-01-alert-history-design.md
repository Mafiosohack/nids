# Live alert feed + Alert History page — design

Date: 2026-11-01 · Status: awaiting review

## Why

The user, after testing the dashboard:

- The alert column is too small to read. Its alerts are fine; the space is the problem.
- Alerts pile up in that column and cause confusion. There is no way to clear it
  short of the admin-only **Clear Alerts**, which deletes every record permanently.
- Every alert is recorded anyway, so the column does not need to be the record.
  A separate page should show the whole history, and the column should be
  clearable.
- The attack (exploit) popup stays exactly as it is.

## Decisions already made

| Question | Decision |
|---|---|
| What does **Clear** do to unreviewed alerts? | **Nothing.** Clear changes only what the column shows. No verdict is written. Unreviewed alerts stay "Needs review" everywhere else (stat card, kill chain, exploit popup, History). |
| Where does review (triage) happen? | In both places: the queue keeps its review tabs, History has them over the whole database, and the alert popup gives verdicts from either. |
| Layout | The AFTER layout stays. Only the queue grows a little (§1). A full-width variant was rejected on the canvas. |
| Separate URL or in-page view? | **In-page view.** Since the 2026-10-31 audit the auth token lives only in page memory; a separate `/history` URL would force a second login or a weaker token store. |
| Where does History read from? | **SQLite** (`alerts` table, last 50 000 records), not the in-memory `AlertStore`, which drops MEDIUM after 24 h and LOW after 6 h. |

Out of scope: changing how the "Needs Review" stat card counts, the admin
**Clear Alerts** button, suppression, correlation, and anything ML-related (a
separate discussion).

## 1. Main page — Live Alerts column

**Layout: the AFTER (e1e5e56) dashboard is kept as it is.** The user
reviewed a full-width variant on the design canvas and rejected it. They want
every existing panel (timeline, Attack Distribution donut, Sources in Question,
Targeted Ports, Model Comparison, Feature Importance, System Controls) where
it is, with the Alert Queue only *a little* bigger. Canvas board:
`ProposedFeed.dc.html` ("PROPOSED v2").

**Size** (current → new):

| What | Now | New |
|---|---|---|
| `.row-2col` columns (timeline : queue) | `1.7fr 1fr` | `1.25fr 1fr` (queue ≈ 510 → 610px at 1440) |
| `.alerts-container` `max-height` | 400px | 560px |
| `.alert-msg` | 12px | 13px |
| `.alert-src`, `.alert-ts`, `.alert-type-tag`, `.alert-count-badge` | 10px | 11px |
| `.sev-tag`, `.alert-triage` | 9px | 10px |
| `.alert-item` padding | 9px 12px | 11px 14px |
| `.alert-sev-dot` | 7px | 8px |
| `.chart-wrap-lg` (timeline) | — | grows to match the queue's height |

Severity encoding (stripe width, solid/tint/outline/dashed, red only for
critical) is unchanged.

**Content.** The panel keeps its title (Alert Queue) and its review-status
tabs, with their counts. Within whichever tab is selected, it shows only
alerts with `id > clearedThroughId`. Tab counts are not affected by Clear,
because they count review status. The existing ×N bundling and the IP filter
chip are unchanged.

**Clear button** in the panel header, styled `btn-neutral`, not `btn-danger`:
it is reversible and destroys nothing.

- Click → `clearedThroughId = max id currently held` and save it to
  `localStorage` under `nids.feed.clearedThrough`. All access is wrapped in
  try/catch; if storage is unavailable, the clear lasts until reload.
- A **Show all** link next to it resets `clearedThroughId` to 0.
- The empty state after a clear: *"Cleared at 14:32. N alerts still need
  review → History"*. N comes from `/alerts/history?status=new` → `total`
  (the same number the History page shows). The link opens History filtered
  to Needs review. When N is 0 it reads *"Cleared at 14:32. Nothing awaiting
  review."*
- After Admin **Clear Alerts** (ids may restart), `clearedThroughId` resets
  to 0. On load, if `clearedThroughId` is greater than the highest id the
  server returns, it also resets to 0.

**Unchanged:** clicking a row opens the alert popup with its verdict
controls; the exploit popup, kill chain, IP strip and stat cards behave as
they do today.

## 2. History view

**Navigation.** A **Dashboard | History** segmented control in the top bar.
Switching hides one `<main>` section and shows the other. No reload, and the
token stays in memory. The choice is kept in `location.hash` (`#history`), so
a reload returns to the same view after login. Polling for the dashboard
keeps running while History is open; History itself fetches only on filter
change, page change, or a **Refresh** click. It does not auto-poll, so the
rows don't move while being read.

**Filters** (a bar above the table; any change resets to page 1):

| Filter | Control | Server param |
|---|---|---|
| Review status | tabs: Needs review · Confirmed · False positive · Resolved · All | `status` |
| Severity | multi-select chips: critical / high / medium / low / info | `severity` (repeatable) |
| Time | Last 1 h · 24 h · 7 d · All | `since` (epoch seconds) |
| IP | text, matches src **or** dst exactly | `ip` |
| Rule | select, filled from `GET /rules/coverage` ids | `rule_id` |
| Text | text, substring of message | `q` |

The default view is **Needs review**, **All time**, sorted worst first.

**Table columns:** Time · Severity · Rule (`type`, `rule_id` in the
tooltip) · Source → Destination · ×count · Status · Reviewed by. 100 rows per
page, with **‹ Prev** / **Next ›** and *"1–100 of 3,412"*. Newest first,
except the Needs review tab sorts by severity rank, then newest.

**Row click** opens the existing alert popup with that one record, so verdicts
work from History. If `/alerts/{id}/context` returns 404 (the record has left
the live store), the evidence section reads *"Live evidence has expired for
this alert (older than the live store). The verdict still applies."*. The
verdict buttons stay enabled. The suppression checkbox stays available to
admins; see §3.

**Export CSV** (analyst+, `export_alerts`) exports *all rows matching the
filters*. It pages through `/alerts/history` with `limit=200`, stopping at
10 000 rows and saying so if it does. Cells starting with `= + - @`, tab or
CR get a leading `'`. This blocks CSV formula injection: alert messages
contain attacker-controlled text (HTTP payloads, user agents). The same
neutralization is applied to the existing main-page export.

**Retention note**, one line under the filter bar: *"The live view drops
medium alerts after 24 h and low after 6 h. History keeps the last 50,000
alerts of every severity."*

## 3. Server

### 3a. `GET /alerts/history` (new, `view_alerts`)

```
?status=new|confirmed|false_positive|resolved|all   (default all)
&severity=high&severity=critical                    (repeatable; default any)
&since=<epoch float>                                (default none)
&ip=<string, ≤45 chars>
&rule_id=<string, ≤64 chars>
&q=<string, ≤200 chars>
&sort=newest|worst                                  (default newest)
&offset=<int ≥0>&limit=<1..200, default 100>
→ {"total": int, "offset": int, "limit": int, "rows": [alert, ...]}
```

- Every filter goes in as a bound `?` parameter. The WHERE clause is built
  only from fixed column fragments. Nothing from the request is ever
  formatted into SQL. `q` uses `message LIKE ? ESCAPE '\'` with `% _ \`
  escaped in the value.
- `status=new` matches `triage_status IS NULL OR triage_status = 'new'`
  (rows written before triage existed are NULL).
- `severity` values are validated against the known set (400 otherwise).
- `sort=worst` orders by a `CASE severity` rank, then `id DESC`. `newest` is
  `id DESC`.
- `details` is JSON-decoded per row, as `db_load_recent_alerts` already does.
- `total` comes from a `COUNT(*)` with the same WHERE.
- The time filter uses `created_at`, which §3c keeps updated as **last seen**.

### 3b. `POST /alerts/triage` reaches DB-only records

Today `PIPELINE.alerts.set_triage` updates only records still in memory and
silently skips the rest, so a verdict given from History on an old alert
would be lost. Change:

1. Run `set_triage` as now → `updated` (in-memory records).
2. `missing = ids − {r.id for r in updated}`. If any, a new `db_set_triage(ids,
   status, by, at, note)` runs one parameterized `UPDATE … WHERE id IN (…)`
   and returns those rows (`rule_id, type, src`) so suppression pairs can be
   built from them too.
3. The response's `updated` counts both. Ids found in neither place are
   reported in a new `not_found` list rather than silently dropped.

### 3c. Repeats reach the database

Deduped repeats raise `count` on the in-memory record but never reach SQLite
(`pipeline.publish` returns before `on_alert`). History would show ×1 for an
alert the live store knows as ×300. A repeat into a *resolved* record also
reopens it in memory but not on disk. Change: `DetectionPipeline` gains an
optional `on_repeat(record)` callback, called when `alerts.add` folds a
finding into an existing record that has an id. `main.py` wires it to
`db_touch_alert(id, count, triage_status, triage_by, triage_at, triage_note)`,
which also sets `created_at = now`. `AddResult` already returns the existing
record on a dedup (`retention.py:143`), so `AlertStore` needs no change.

Write volume: one small UPDATE per collapsed repeat. Under a flood this is
the same order as today's inserts, which the store already handles. If it
shows up in the systems-map L8 measurement, batch it later. No batching now.

## 4. Security notes

- No new permission. History reads require `view_alerts`, the same data
  `/alerts` already exposes to every role. Verdicts and suppression keep
  their existing permission checks.
- Bounded inputs (lengths above, `limit ≤ 200`) and a single query per
  request. The 10 000-row export cap bounds the client.
- All rendered alert fields go through the existing `esc()` / `textContent`
  path. No new `innerHTML` with record data.
- `localStorage` holds only an integer id. No token, no alert content.

## 5. Testing

Unit (venv python, alongside the existing suite):

- `/alerts/history`: each filter alone and combined; `status=new` includes
  NULL rows; `worst` ordering; offset/limit and `total`; `limit=201` → 422;
  bad severity → 400; `q` containing `%`, `_`, `'`, `"; DROP TABLE alerts;--`
  is matched literally and the table survives; viewer role can read.
- Triage of an id present only in SQLite persists and is returned. A mixed
  in-memory + DB-only batch updates both. An unknown id appears in `not_found`.
  Admin suppression from a DB-only false positive creates the suppression.
- A deduped repeat updates SQLite `count` and `created_at`. A repeat into a
  resolved record persists the reopen.
- CSV neutralization helper: `=1+1`, `+x`, `-x`, `@x`, `\tx` are prefixed and
  ordinary text is untouched.

Then the full suite plus `efficacy_harness.py` (expect 36/36, see the
verify-with-venv note).

Headless UI check (Playwright, mocked API, 1366×768 and 1920×1080):

- Every existing panel is still present. The queue column is about 610px at
  1440 wide, and the computed sizes match the table above.
- Clear empties the feed and survives a reload, the empty state shows the
  Needs-review total and its link opens History on Needs review, and Show
  all restores the feed.
- History: switching views keeps the session, filters change the request
  params, paging labels are correct, a row click opens the popup, the expired
  evidence message appears on a 404 context, and the exploit popup still
  appears over either view.

## 6. Files touched

- `nids_dashboard.html`: layout, type scale, feed + Clear, view switch,
  History view, CSV neutralization.
- `main.py`: `GET /alerts/history`, `db_set_triage`, `db_touch_alert`,
  triage fallback, `on_repeat` wiring.
- `detection/pipeline.py`: `on_repeat` callback.
- `tests/`: new tests listed above.
- `docs/superpowers/specs/`: this file. After shipping, the AFTER row of
  the design canvas gets updated (the canvas has gone stale before).

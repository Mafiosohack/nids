# Alert History + clearable Alert Queue — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the dashboard's Alert Queue a little bigger and clearable (view-only), and add an in-page History view that reads every recorded alert from SQLite.

**Architecture:** Three server changes in `main.py` and one in `detection/pipeline.py`: a parameterized history query + endpoint, a triage fallback that reaches DB-only rows, and an `on_repeat` hook that persists dedup repeats. Two dashboard changes in `nids_dashboard.html`: the sized queue with a localStorage clear marker, and a History view toggled from the top bar in the same page.

**Tech Stack:** Python 3 / FastAPI / sqlite3, unittest (`tests/`, `tests/apiharness.LiveServer` for real HTTP), single-file vanilla JS dashboard under a hash-based CSP, Playwright (system Python, `channel="chrome"`) for the headless UI checks.

**Spec:** `docs/superpowers/specs/2026-11-01-alert-history-design.md`. The design canvas boards `ProposedFeed`, `ProposedFeedCleared` and `ProposedHistory` at https://claude.ai/artifact/2gHddcDP4fBAwpobHjKMew show the approved look.

## Global Constraints

- Run Python with the venv only: `venv/Scripts/python.exe` (system Python lacks scapy and fakes failures).
- Unit suite: `venv/Scripts/python.exe -m unittest discover -s tests -t tests` — currently passes; must still pass after every task.
- Harness: `venv/Scripts/python.exe efficacy_harness.py` must report 36/36 at the end.
- **No `git commit` or `git push` unless the user says so** (repo rule: commit only when asked, straight to `main`, push needs its own go-ahead). Each task ends with a checkpoint, not a commit.
- `live_ids_v2.py` has an unrelated uncommitted change. Leave it alone and keep it out of any commit.
- SQL: every request value is a bound `?`; WHERE clauses are joined from fixed string fragments only.
- Dashboard: no inline handlers (CSP). Buttons use `data-action` + an entry in `ACTIONS`; inputs get listeners in JS. Record text reaches the DOM only via `textContent` or `esc()`.
- Clear never writes a verdict and never calls the server. localStorage key `nids.feed.clearedThrough` holds only `{"id": <int>, "at": <epoch ms>}`, wrapped in try/catch.
- Sizes (exact): `.row-2col` `1.25fr 1fr`; `.alerts-container` `max-height:560px`; `.alert-msg` 13px; `.alert-src`/`.alert-ts`/`.alert-type-tag`/`.alert-count-badge` 11px; `.sev-tag`/`.alert-triage` 10px; `.alert-item` padding `11px 14px`; `.alert-sev-dot` 8px. Every other panel stays as it is.
- History copy (exact): retention line *"The live view drops medium alerts after 24 h and low after 6 h. History keeps the last 50,000 alerts of every severity."*; expired evidence *"Live evidence has expired for this alert (older than the live store). The verdict still applies."*

## Review Focus

1. **Rows written before triage/rule_id/count existed** (NULL `triage_status`, `rule_id`, `count`) must show as Needs review, rule = `type`, ×1 — not vanish from the Needs review tab. → test in Task 1.
2. **A stale clear marker** (admin Clear Alerts, or a fresh DB whose ids restart below the stored id) must not hide every future alert forever. → reset rule + UI check in Task 4.
3. **localStorage throwing** (private window, blocked storage) must not break the queue; Clear just lasts until reload. → UI check in Task 4.
4. **`offset` past the end** returns `rows: []` with the true `total`, and the pager says so instead of showing a blank table. → test in Task 1, UI check in Task 5.
5. **CSV cells that start with `=`, `+`, `-`, `@`, tab or CR**, including the existing main-page export, are neutralized; ordinary negative-looking text like `-` alone is still prefixed (safe, accepted). → UI check in Task 5.

---

### Task 1: History query + `GET /alerts/history`

**Files:**
- Modify: `main.py` — new `db_query_alert_history` beside `db_load_recent_alerts` (~line 655); new endpoint right after `alerts_summary` (~line 2530).
- Test: `tests/test_alert_history.py` (new)

**Interfaces:**
- Produces: `db_query_alert_history(status: str = "all", severities: Optional[List[str]] = None, since: Optional[float] = None, ip: Optional[str] = None, rule_id: Optional[str] = None, q: Optional[str] = None, sort: str = "newest", offset: int = 0, limit: int = 100) -> Tuple[int, List[Dict]]` — `(total, rows)`; rows shaped like `db_load_recent_alerts` output (details decoded, `created_at` kept as `last_seen`, NULL `count` → 1, NULL `triage_status` → `"new"`).
- Produces: `GET /alerts/history` → `{"total", "offset", "limit", "rows"}`, permission `view_alerts`. Params exactly as spec §3a.

- [ ] **Step 1: Write the failing tests** in `tests/test_alert_history.py`. Use one `LiveServer` per class (`setUpClass`/`tearDownClass` as in `test_access_hardening.py`). Seed rows with `srv.main.db_insert_alert({...})` (ids 1..N, set `created_at` by updating the row after insert where a test needs a time), plus one legacy row inserted by raw SQL with NULL `rule_id`, `count`, `triage_status`.

```python
class TestHistoryEndpoint(unittest.TestCase):
    def test_status_new_includes_legacy_null_rows(self):       # legacy id in rows
    def test_status_filters(self):                             # confirmed/false_positive/resolved/all totals
    def test_severity_repeatable(self):                        # ?severity=high&severity=critical
    def test_bad_severity_is_400(self):                        # ?severity=urgent -> 400
    def test_since_uses_last_seen(self):                       # created_at older than since excluded
    def test_ip_matches_src_or_dst_exactly(self):              # "192.168.56.1" does not match .101
    def test_rule_id_filter(self):
    def test_q_is_literal(self):                               # q="50%_off" matches only that message;
                                                               # q="\"; DROP TABLE alerts;--" -> 200, table still has rows
    def test_sort_worst_then_newest(self):                     # critical before high; equal sev by id desc
    def test_paging_and_total(self):                           # limit=2 offset=2 -> 2 rows, total unchanged
    def test_offset_past_end(self):                            # offset=10_000 -> rows == [], total == N
    def test_limit_cap(self):                                  # limit=201 -> 422
    def test_param_length_caps(self):                          # ip 46 chars / rule_id 65 / q 201 -> 422
    def test_viewer_can_read(self):                            # viewer role -> 200
    def test_legacy_row_shape(self):                           # count == 1, rule_id None, triage_status "new"
```

- [ ] **Step 2: Run to verify they fail**
Run: `venv/Scripts/python.exe -m unittest discover -s tests -t tests -p test_alert_history.py -v`
Expected: FAIL (404 on `/alerts/history`). Later tasks run their tests with the same command.

- [ ] **Step 3: Implement `db_query_alert_history(...)` in `main.py`.** Severity rank for `sort="worst"`: `CASE lower(severity) WHEN 'critical' THEN 5 WHEN 'high' THEN 4 WHEN 'medium' THEN 3 WHEN 'low' THEN 2 WHEN 'info' THEN 1 ELSE 0 END DESC, id DESC`. `status="new"` → `(triage_status IS NULL OR triage_status = 'new')`. `q` → `message LIKE ? ESCAPE '\'` with `\`, `%`, `_` escaped in the value and wrapped in `%…%`. `total` from `SELECT COUNT(*)` with the same WHERE and params.

- [ ] **Step 4: Implement the endpoint.** `severity: Optional[List[str]] = Query(None)`, validated against `{"critical","high","medium","low","info"}` (400 otherwise); `status` validated against `SETTABLE_STATUSES | {"new","all"}` (400); `sort` in `{"newest","worst"}` (400); `offset: int = Query(0, ge=0)`, `limit: int = Query(100, ge=1, le=200)`; `ip` `max_length=45`, `rule_id` `max_length=64`, `q` `max_length=200`.

- [ ] **Step 5: Run to verify they pass**, then the whole unit suite.
Expected: new tests PASS; suite still green.

- [ ] **Step 6: Checkpoint** — report to the user; do not commit.

---

### Task 2: Verdicts reach DB-only alerts

**Files:**
- Modify: `main.py` — new `db_set_triage` beside `db_update_triage` (~line 623); `triage_alerts` (~line 2561).
- Test: `tests/test_alert_history.py` (add class)

**Interfaces:**
- Consumes: `PIPELINE.alerts.set_triage(ids, status, by, note) -> List[dict]` (existing; skips unknown ids).
- Produces: `db_set_triage(ids: List[int], status: str, by: str, at: float, note: str) -> List[Dict]` — updates the given ids in one `UPDATE … WHERE id IN (?,?,…)` and returns `[{id, rule_id, type, src}]` for the ids that exist.
- Produces: `POST /alerts/triage` response gains `"not_found": [int]`; `"updated"` counts in-memory + DB-only records.

- [ ] **Step 1: Write the failing tests**

```python
class TestTriageReachesHistory(unittest.TestCase):
    def test_db_only_verdict_persists(self):       # row in DB, not in PIPELINE.alerts -> verdict 200,
                                                   # /alerts/history?status=resolved contains it
    def test_mixed_batch(self):                    # one live + one DB-only id -> updated == 2, both stick
    def test_unknown_id_reported(self):            # id 999999 -> in "not_found", not in "ids"
    def test_admin_suppress_from_db_only_fp(self): # suppress=true on DB-only false positive ->
                                                   # suppressions_created has (rule_id, src)
```

- [ ] **Step 2: Run to verify they fail.** Expected: the DB-only row still reads `new`; `not_found` missing.

- [ ] **Step 3: Implement.** In `triage_alerts`, after `set_triage`: `missing = sorted(set(req.ids) - {r["id"] for r in updated})`; `db_rows = db_set_triage(missing, req.status, session["username"], time.time(), note)` when `missing`; `not_found = sorted(set(missing) - {r["id"] for r in db_rows})`. Suppression pairs come from `updated + db_rows`. Keep the existing `db_update_triage(updated)` call and its try/except.

- [ ] **Step 4: Run the tests and the suite.** Expected: PASS.

- [ ] **Step 5: Checkpoint.**

---

### Task 3: Dedup repeats reach the database

**Files:**
- Modify: `detection/pipeline.py` — `DetectionPipeline.__init__` (~line 102) and `publish` (~line 331).
- Modify: `main.py` — new `db_touch_alert` beside `db_insert_alert`; `_on_repeat_alert` beside `_on_published_alert` (~line 1073); `_wire_pipeline` (~line 1100).
- Test: `tests/test_alert_history.py` (add classes)

**Interfaces:**
- Produces: `DetectionPipeline(..., on_repeat: Optional[Callable[[dict], None]] = None)`, stored as `self.on_repeat`. Called with `dict(result.record)` when `alerts.add` folds a finding into an existing record whose `id` is not None. Not called for a new record. `reset()` leaves it in place, as it does `on_alert`.
- Produces: `db_touch_alert(rec: Dict) -> None` — `UPDATE alerts SET count=?, triage_status=?, triage_by=?, triage_at=?, triage_note=?, created_at=? WHERE id=?` with `created_at = time.time()`.

- [ ] **Step 1: Write the failing tests**

```python
class TestRepeatHook(unittest.TestCase):           # pure pipeline, FakeClock, no server
    def test_repeat_calls_on_repeat_with_count(self):   # same finding twice -> one call, record["count"] == 2
    def test_new_record_does_not_call_on_repeat(self):
    def test_reopen_is_in_the_record(self):              # resolve, repeat -> record["triage_status"] == "new"

class TestRepeatPersisted(unittest.TestCase):      # LiveServer
    def test_db_count_and_last_seen_follow_repeats(self):  # srv.main.publish_finding(f) twice ->
                                                           # history row count == 2, last_seen increased
    def test_reopen_persisted(self):                       # resolve via API, publish again ->
                                                           # history row triage_status == "new"
```

- [ ] **Step 2: Run to verify they fail.**

- [ ] **Step 3: Implement** the hook in `publish` (in the `if not result.stored:` branch, before `return None`), `db_touch_alert`, and `_on_repeat_alert(rec)` = `db_touch_alert(rec)` inside the same try/except + `safe_print("[DB][WARN] …")` pattern as `_on_published_alert`. Wire `PIPELINE.on_repeat = _on_repeat_alert` in `_wire_pipeline`.

- [ ] **Step 4: Run the tests, the suite, and the harness** (`efficacy_harness.py`, 36/36 — the pipeline changed).

- [ ] **Step 5: Checkpoint.**

---

### Task 4: Alert Queue — a little bigger, and clearable

**Files:**
- Modify: `nids_dashboard.html` — CSS lines ~112, 119–149; Alert Queue panel header ~483–501; `renderAlerts` ~1042; `clearAlerts` ~1489 (reset marker after the admin wipe); `ACTIONS` ~1771.
- UI check: `<scratchpad>/ui_check_queue.py` (not committed)

**Interfaces:**
- Consumes: `GET /alerts/history?status=new&limit=1` → `total` (Task 1).
- Produces (JS, file scope): `let feedCleared = {id: 0, at: 0}`; `loadFeedCleared()`, `saveFeedCleared()`; `clearFeed()`, `showAllFeed()`; `historyNewTotal` (number | null) refreshed by `refreshHistoryNewTotal()` at most once per 15 s; `ACTIONS.clearFeed`, `ACTIONS.showAllFeed`.

- [ ] **Step 1: Write the headless check first** (`ui_check_queue.py`): Playwright + `channel="chrome"`, `wait_until="domcontentloaded"`, page routed to a mocked API at `http://nids.test` (serve `nids_dashboard.html` at `/`, JSON fixtures for `/auth/login`, `/status`, `/alerts`, `/alerts/summary`, `/incidents`, `/model/info`, `/alerts/history`). Assertions at 1440×900:
  - queue column width 590–630px; `.alert-msg` computed 13px, `.sev-tag` 10px; `.alerts-container` max-height 560px; the donut, Model Comparison and Feature Importance panels still exist.
  - Clear → list shows "Cleared at HH:MM" and "N alerts still need review" with N from the mocked history `total`; tab counts unchanged; nothing POSTed.
  - reload → still cleared. A new alert with a higher id appears. Show all → everything back.
  - stored marker `{"id": 99999}` while the server's max id is 40 → marker reset, queue full (Review Focus 2).
  - `localStorage.getItem`/`setItem` stubbed to throw via `add_init_script` → page loads, Clear works, no console errors (Review Focus 3).
- [ ] **Step 2: Run it to verify it fails.** Run: `python <scratchpad>/ui_check_queue.py` — expect failures on width/size/Clear.
- [ ] **Step 3: Implement.** CSS values from Global Constraints. The timeline chart grows to the queue's height (`.chart-wrap-lg` height to fill the panel). Header gets `Clear` (`btn btn-neutral`, `data-action="clearFeed"`), a `Show all` link (`data-action="showAllFeed"`, hidden unless cleared) and `History →` (`data-action="showView" data-arg="history"`, wired in Task 5; until then it may be omitted). `renderAlerts` filters `id > feedCleared.id`, adds `feedCleared.id` to `sig`, and renders the cleared empty state (spec §1 copy). Reset the marker to 0 when it is above the highest id in `allAlerts` after a successful poll, and inside `clearAlerts()` after the admin wipe succeeds.
- [ ] **Step 4: Run the UI check** — expect all assertions to pass, and the unit suite still green.
- [ ] **Step 5: Checkpoint.**

---

### Task 5: History view

**Files:**
- Modify: `nids_dashboard.html` — top bar (~402–430) gets the view switch; wrap the existing `<div class="main">` as `id="view-dashboard"`; add `<div class="main hidden" id="view-history">`; new CSS for `.views`, history filters, table and pager (reuse existing tokens); JS: history state, loader, renderer, CSV helper; `loadEvidence` catch (~1644); `exportAlerts` (~1503); `ACTIONS`.
- UI check: `<scratchpad>/ui_check_history.py` (not committed)

**Interfaces:**
- Consumes: `GET /alerts/history` (Task 1), `POST /alerts/triage` incl. `not_found` (Task 2), `GET /rules/coverage` → `tiers[].rules[].rule_id`, existing `openModal(grp)` with `grp = {rep: row, count: row.count, alerts: [row]}`.
- Produces (JS): `showView(name: 'dashboard'|'history')` (sets `location.hash` to `#history` or clears it; called after login with the hash's view); `hist = {status: 'new', severities: Set, since: null|seconds, ip: '', rule: '', q: '', offset: 0}`; `loadHistory()`; `loadHistoryCounts()` (five `limit=1` calls, status only, other filters ignored); `csvCell(v: any) -> string` (prefixes `'` when the text starts with `= + - @`, tab or CR, then CSV-quotes); `exportHistoryCsv()` (pages `limit=200` up to 10,000 rows, toast if capped); `ACTIONS.showView`, `ACTIONS.histPage` (`arg` = `prev`|`next`), `ACTIONS.histStatus`, `ACTIONS.histSeverity`, `ACTIONS.histRefresh`, `ACTIONS.exportHistory`.

- [ ] **Step 1: Write the headless check** (`ui_check_history.py`, same harness as Task 4) asserting:
  - the switch shows History and hides the dashboard without a reload; `#history` survives a reload after login; Dashboard switches back; the exploit popup still appears over History when the mocked `/alerts` adds a critical exploit alert.
  - default request is `status=new&sort=worst&offset=0&limit=100`; changing severity, time, IP, rule and text produces exactly the matching query params (text inputs debounced 300 ms) and resets `offset` to 0.
  - pager label "1–100 of 3,412"; Next → `offset=100`; with a mocked `offset` past the end it shows "No rows on this page" and Prev is enabled (Review Focus 4).
  - retention line text matches the Global Constraints copy exactly.
  - row click opens the alert popup; with `/alerts/{id}/context` mocked to 404 `"Alert 7 is no longer in the live store."` the evidence list contains the expired-evidence copy and verdict buttons stay enabled; a verdict reloads the table.
  - export: the downloaded CSV has `'=1+1`, `'+x`, `'-x`, `'@x`, `'\tx` for those messages, and the main-page Export CSV neutralizes the same (Review Focus 5); exporting 0 rows toasts instead of downloading.
  - row text with `<img src=x onerror=alert(1)>` renders as text.
- [ ] **Step 2: Run it to verify it fails.**
- [ ] **Step 3: Implement** per spec §2. Table columns: Time · Severity · Rule (`type`, `rule_id` in `title`) · Source → Destination · Count · Status · Reviewed by. Sort param is `worst` on the Needs review tab and `newest` otherwise. History does not poll; it fetches on filter, page or Refresh only. In `loadEvidence`'s catch, use the expired-evidence copy when the message contains `no longer in the live store`, otherwise keep the existing text. `exportAlerts` switches to `csvCell`.
- [ ] **Step 4: Run both UI checks and the unit suite.** Expected: all pass.
- [ ] **Step 5: Checkpoint.**

---

### Task 6: Final verification and records

**Files:**
- Modify: `README.md` test count line (~227) if the count changed; design canvas AFTER boards (via the Artifact tool), only after the user confirms the shipped look.

- [ ] **Step 1:** Run the unit suite and `efficacy_harness.py`. Record the exact counts (expect 36/36 harness; suite = previous count + the new tests, 0 failures, 0 skips).
- [ ] **Step 2:** Start the real server from the venv against a scratch copy of the DB, log in, and confirm by eye at 1366×768 and 1920×1080: queue size, Clear, History filters, a verdict on an old alert.
- [ ] **Step 3:** Update the README test count. Ask the user before updating the canvas and before any commit.

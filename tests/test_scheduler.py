"""The PG-native ingest ticker (``scheduler``).

The ticker ENQUEUEs claim-able ``ingest_jobs`` rows on an injected schedule
(plan §1f — the engine ships the machinery; the host supplies the SCHEDULE +
task map). These tests register a fake schedule + fake ingest tasks, then pin:

  * the pure next-fire / soonest schedule math (injected clock, no real sleep);
  * the enqueue side against the DB;
  * the boot-kick (non-freshness entries only);
  * one run_forever loop tick with a virtual clock.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

import queue_workflows
from queue_workflows import scheduler
from queue_workflows.db import connection


# A fake host schedule, mirroring the shape ai_leads uses (fetch@:37,
# load@:47, freshness@:07) but registered via the public hook.
_FAKE_SCHEDULE = [
    scheduler.ScheduleEntry("ingest-fetch-hourly", 37, "run_fetch_all", "fetch"),
    scheduler.ScheduleEntry("ingest-load-hourly", 47, "run_load_all", "load"),
    scheduler.ScheduleEntry("ingest-freshness-hourly", 7, "audit_freshness", "load"),
]


@pytest.fixture(autouse=True)
def _register_ingest():
    """Register the fake ingest tasks (so enqueue_ingest_job accepts them) +
    the fake schedule (so the Ticker / boot-kick fire it)."""
    for name in ("run_fetch_all", "run_load_all", "audit_freshness"):
        queue_workflows.register_ingest_task(name, lambda reason: {"ok": True})
    queue_workflows.set_ingest_schedule(_FAKE_SCHEDULE)
    yield


# ── next-fire computation (pure) ─────────────────────────────────────────────


def test_next_fire_same_hour_when_before_minute():
    e = scheduler.ScheduleEntry("x", 37, "run_fetch_all", "fetch")
    now = datetime(2026, 5, 24, 10, 5, 0, tzinfo=timezone.utc)
    assert e.next_fire_at(now) == datetime(2026, 5, 24, 10, 37, 0, tzinfo=timezone.utc)


def test_next_fire_rolls_to_next_hour_when_past_minute():
    e = scheduler.ScheduleEntry("x", 37, "run_fetch_all", "fetch")
    now = datetime(2026, 5, 24, 10, 40, 0, tzinfo=timezone.utc)
    assert e.next_fire_at(now) == datetime(2026, 5, 24, 11, 37, 0, tzinfo=timezone.utc)


def test_next_fire_exactly_on_minute_rolls_forward():
    e = scheduler.ScheduleEntry("x", 37, "run_fetch_all", "fetch")
    now = datetime(2026, 5, 24, 10, 37, 0, tzinfo=timezone.utc)
    assert e.next_fire_at(now) == datetime(2026, 5, 24, 11, 37, 0, tzinfo=timezone.utc)


# ── hour-restricted cadence (G4) ─────────────────────────────────────────────


def test_next_fire_hours_none_is_hourly():
    # explicit None == original hourly behaviour (backward-compat regression).
    e = scheduler.ScheduleEntry("x", 37, "run_fetch_all", "fetch", hours=None)
    now = datetime(2026, 5, 24, 10, 5, 0, tzinfo=timezone.utc)
    assert e.next_fire_at(now) == datetime(2026, 5, 24, 10, 37, 0, tzinfo=timezone.utc)


def test_next_fire_daily_hour_rolls_to_next_day():
    # daily at 06:30 — from 10:05 the next fire is tomorrow 06:30.
    e = scheduler.ScheduleEntry("glofas", 30, "run_load_all", "load", hours=frozenset({6}))
    now = datetime(2026, 5, 24, 10, 5, 0, tzinfo=timezone.utc)
    assert e.next_fire_at(now) == datetime(2026, 5, 25, 6, 30, 0, tzinfo=timezone.utc)


def test_next_fire_daily_hour_same_day_when_before():
    e = scheduler.ScheduleEntry("glofas", 30, "run_load_all", "load", hours=frozenset({6}))
    now = datetime(2026, 5, 24, 3, 0, 0, tzinfo=timezone.utc)
    assert e.next_fire_at(now) == datetime(2026, 5, 24, 6, 30, 0, tzinfo=timezone.utc)


def test_next_fire_multiple_hours_picks_next():
    # four-times-a-day at :00 — from 10:05 the next is 16:00 (10:00 already past).
    e = scheduler.ScheduleEntry(
        "icon", 0, "run_fetch_all", "fetch", hours=frozenset({4, 10, 16, 22}),
    )
    now = datetime(2026, 5, 24, 10, 5, 0, tzinfo=timezone.utc)
    assert e.next_fire_at(now) == datetime(2026, 5, 24, 16, 0, 0, tzinfo=timezone.utc)


def test_soonest_next_fire_across_schedule():
    now = datetime(2026, 5, 24, 10, 10, 0, tzinfo=timezone.utc)
    soonest, due = scheduler._soonest(_FAKE_SCHEDULE, now)
    assert soonest == datetime(2026, 5, 24, 10, 37, 0, tzinfo=timezone.utc)
    assert [e.name for e in due] == ["ingest-fetch-hourly"]


def test_soonest_groups_simultaneous_entries():
    e1 = scheduler.ScheduleEntry("a", 37, "run_fetch_all", "fetch")
    e2 = scheduler.ScheduleEntry("b", 37, "run_load_all", "load")
    now = datetime(2026, 5, 24, 10, 0, 0, tzinfo=timezone.utc)
    soonest, due = scheduler._soonest([e1, e2], now)
    assert soonest == datetime(2026, 5, 24, 10, 37, 0, tzinfo=timezone.utc)
    assert {e.name for e in due} == {"a", "b"}


# ── enqueue (against the DB) ─────────────────────────────────────────────────


def _ingest_rows():
    with connection() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT task_name, queue, reason, status FROM ingest_jobs "
            "ORDER BY created_at"
        )
        return list(cur.fetchall())


def test_enqueue_due_inserts_claimable_rows():
    e_fetch = scheduler.ScheduleEntry("ingest-fetch-hourly", 37, "run_fetch_all", "fetch")
    scheduler.enqueue_due([e_fetch], reason="tick")
    rows = _ingest_rows()
    assert len(rows) == 1
    assert rows[0]["task_name"] == "run_fetch_all"
    assert rows[0]["queue"] == "fetch"
    assert rows[0]["reason"] == "tick"
    assert rows[0]["status"] == "queued"


def test_boot_kick_enqueues_fetch_and_load_only():
    """The boot kick fires the non-freshness entries (one fetch + one load),
    tagged reason='boot'. Operates on the host-injected schedule."""
    scheduler.enqueue_boot_kick()
    rows = _ingest_rows()
    tasks = {(r["task_name"], r["queue"], r["reason"]) for r in rows}
    assert tasks == {
        ("run_fetch_all", "fetch", "boot"),
        ("run_load_all", "load", "boot"),
    }


# ── the ticker loop (injected clock + sleep, no real waiting) ────────────────


def test_run_forever_enqueues_on_each_tick_then_stops():
    state = {"now": datetime(2026, 5, 24, 10, 30, 0, tzinfo=timezone.utc)}

    def fake_now():
        return state["now"]

    def fake_sleep(seconds):
        state["now"] = state["now"] + timedelta(seconds=seconds)

    tick = scheduler.Ticker(now_fn=fake_now, sleep_fn=fake_sleep)
    orig = scheduler.enqueue_due

    def counting_enqueue(due, *, reason="tick"):
        orig(due, reason=reason)
        if reason == "tick":
            tick.stop()

    import queue_workflows.scheduler as sched_mod
    sched_mod.enqueue_due = counting_enqueue
    try:
        tick.run_forever()
    finally:
        sched_mod.enqueue_due = orig

    rows = _ingest_rows()
    reasons = [r["reason"] for r in rows]
    assert reasons.count("boot") == 2
    assert "tick" in reasons
    tick_rows = [r for r in rows if r["reason"] == "tick"]
    assert tick_rows[0]["task_name"] == "run_fetch_all"


def test_run_forever_stop_during_wait_fires_no_tick():
    """A stop() observed DURING the bounded sleep wait must short-circuit BEFORE
    enqueue_due(due, reason='tick') runs — shutting down mid-wait must not emit a
    half-due ingest row. This pins the ``if self._stop.is_set(): break`` guard at
    scheduler.py:175-176: without it, a worker stopped while waiting for a
    far-future fire would spuriously enqueue a 'tick' on the way out. Only the
    post-enqueue stop path is otherwise covered, so a regression dropping this
    guard would slip through silently."""
    # Single far-future entry; clock starts before its minute so wait_s > 0 and
    # the loop enters the bounded-sleep wait (never reaching enqueue_due/tick).
    far = scheduler.ScheduleEntry("ingest-fetch-hourly", 37, "run_fetch_all", "fetch")
    state = {"now": datetime(2026, 5, 24, 10, 5, 0, tzinfo=timezone.utc)}

    def fake_now():
        return state["now"]

    calls: list[str] = []

    def fake_sleep(seconds):
        # Stop lands DURING the wait; crucially, do NOT advance the clock — so
        # wait_s stays > 0 and only the stop flag ends the wait loop.
        tick.stop()

    tick = scheduler.Ticker(schedule=[far], now_fn=fake_now, sleep_fn=fake_sleep)

    import queue_workflows.scheduler as sched_mod
    orig = scheduler.enqueue_due

    def recording_enqueue(due, *, reason="tick"):
        calls.append(reason)
        return orig(due, reason=reason)

    sched_mod.enqueue_due = recording_enqueue
    try:
        tick.run_forever()
    finally:
        sched_mod.enqueue_due = orig

    # The only enqueue_due call was the on-boot kick — never a 'tick'.
    assert calls == ["boot"]
    assert [r for r in _ingest_rows() if r["reason"] == "tick"] == []

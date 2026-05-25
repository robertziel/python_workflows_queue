"""Postgres-native periodic ticker — the ingest scheduler.

Owns the schedule as plain data and ENQUEUEs a claim-able ``ingest_jobs`` row
per fire — which a fetch/load ``claim_worker`` then picks up via ``LISTEN
ingest_job_ready``. No new Postgres extension (not pg_cron): it's a
long-running Python loop that sleeps to the next scheduled instant.

The SCHEDULE is HOST-INJECTED (plan §1f): the engine ships the generic
``Ticker``/``ScheduleEntry``/``enqueue_due``/``enqueue_boot_kick`` machinery but
takes the actual schedule from ``config.ingest_schedule`` (empty by default).
The host sets it via ``queue_workflows.set_ingest_schedule([...])``. The
``Ticker`` reads that config default unless a schedule is passed explicitly.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Callable

from queue_workflows import node_queue

log = logging.getLogger(__name__)


# ── schedule as data ─────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ScheduleEntry:
    """One hourly periodic fire. ``minute`` is the minute-of-hour it fires
    (mirroring ``crontab(minute=N)``); ``task_name`` / ``queue`` name the
    claim-able ingest job it enqueues."""

    name: str
    minute: int
    task_name: str
    queue: str

    def next_fire_at(self, now: datetime) -> datetime:
        """The next instant strictly AFTER ``now`` at which this entry fires.
        Strict-after so a fire that just happened at exactly ``self.minute``
        rolls to the next hour rather than re-firing in a tight loop."""
        candidate = now.replace(
            minute=self.minute, second=0, microsecond=0,
        )
        if candidate <= now:
            candidate = candidate + timedelta(hours=1)
        return candidate


def _configured_schedule() -> list[ScheduleEntry]:
    """The host-injected schedule (``config.ingest_schedule``). Empty unless a
    host called ``queue_workflows.set_ingest_schedule``."""
    from queue_workflows.config import get_config
    return list(get_config().ingest_schedule)


def _soonest(
    schedule: list[ScheduleEntry], now: datetime,
) -> tuple[datetime, list[ScheduleEntry]]:
    """Return ``(soonest_fire_instant, entries_firing_then)``. Entries that
    share the same next-fire instant are grouped so a single wake enqueues all
    of them."""
    fires = [(e.next_fire_at(now), e) for e in schedule]
    soonest = min(t for t, _ in fires)
    due = [e for t, e in fires if t == soonest]
    return soonest, due


# ── enqueue ──────────────────────────────────────────────────────────────────


def enqueue_due(due: list[ScheduleEntry], *, reason: str = "tick") -> list[str]:
    """Enqueue a claim-able ``ingest_jobs`` row for each due entry. Returns the
    inserted row ids."""
    ids: list[str] = []
    for e in due:
        jid = node_queue.enqueue_ingest_job(
            task_name=e.task_name, queue=e.queue, reason=reason,
        )
        ids.append(jid)
        log.info(
            "[scheduler] enqueued %s (task=%s queue=%s reason=%s) -> %s",
            e.name, e.task_name, e.queue, reason, jid,
        )
    return ids


def enqueue_boot_kick(schedule: list[ScheduleEntry] | None = None) -> list[str]:
    """On-boot kick: enqueue every non-freshness entry (one per
    fetch/load-style task), tagged ``reason='boot'`` so a reboot pulls
    immediately. "Freshness" entries (those whose task_name contains
    ``freshness``) are excluded — they're cheap reads that don't need a boot
    kick. Operates on the host-injected schedule when none is passed."""
    sched = schedule if schedule is not None else _configured_schedule()
    boot = [e for e in sched if "freshness" not in e.task_name]
    return enqueue_due(boot, reason="boot")


# ── the ticker loop ──────────────────────────────────────────────────────────


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Ticker:
    """Long-running periodic enqueuer. Sleeps to the soonest scheduled instant,
    enqueues the due rows, repeats. ``now_fn`` / ``sleep_fn`` are injectable so
    the schedule math is unit-testable with a virtual clock (no real waiting).
    A dropped sleep (clock jitter) is harmless — the next iteration recomputes
    from the live clock.

    ``schedule`` defaults to the host-injected ``config.ingest_schedule``."""

    #: Cap a single sleep so a stop() signal is observed within this many
    #: seconds even when the next fire is far away.
    MAX_SLEEP_S = 30.0

    def __init__(
        self, *, schedule: list[ScheduleEntry] | None = None,
        now_fn: Callable[[], datetime] = _utcnow,
        sleep_fn: Callable[[float], None] | None = None,
    ) -> None:
        self.schedule = schedule if schedule is not None else _configured_schedule()
        self._now_fn = now_fn
        self._stop = threading.Event()
        # Default real sleep is interruptible by stop() via the Event wait.
        self._sleep_fn = sleep_fn or (lambda s: self._stop.wait(s))

    def run_forever(self) -> None:
        log.info(
            "[scheduler] starting PG-native ingest ticker (%d entries)",
            len(self.schedule),
        )
        if not self.schedule:
            log.warning(
                "[scheduler] no ingest schedule configured "
                "(queue_workflows.set_ingest_schedule); ticker idles"
            )
        # Boot kick once on startup (module-level so a test can swap
        # enqueue_due/enqueue_boot_kick).
        import queue_workflows.scheduler as _self
        _self.enqueue_boot_kick(self.schedule)
        while not self._stop.is_set():
            if not self.schedule:
                # Nothing to fire — just honour stop() promptly.
                self._stop.wait(self.MAX_SLEEP_S)
                continue
            now = self._now_fn()
            soonest, due = _soonest(self.schedule, now)
            wait_s = max(0.0, (soonest - now).total_seconds())
            # Sleep in bounded slices so stop() is observed promptly and a
            # far-future fire doesn't block shutdown.
            while wait_s > 0 and not self._stop.is_set():
                slice_s = min(wait_s, self.MAX_SLEEP_S)
                self._sleep_fn(slice_s)
                now = self._now_fn()
                wait_s = max(0.0, (soonest - now).total_seconds())
            if self._stop.is_set():
                break
            _self.enqueue_due(due, reason="tick")

    def stop(self) -> None:
        self._stop.set()


# ── entrypoint ───────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    import argparse
    import signal

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    argparse.ArgumentParser(prog="queue-scheduler").parse_args(argv)

    ticker = Ticker()

    def _handler(signum, _frame):
        log.info("[scheduler] signal %s; stopping", signum)
        ticker.stop()

    signal.signal(signal.SIGTERM, _handler)
    signal.signal(signal.SIGINT, _handler)
    ticker.run_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

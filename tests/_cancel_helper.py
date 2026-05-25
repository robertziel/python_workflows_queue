"""Cancel-and-wait helpers for the run-cancel propagation tests.

The host cancel handler flips ``workflow_runs.status='cancelled'`` and stops
there — workers notice by reading the run row. These helpers poll Postgres
(the same source of truth the production cancel-watcher polls), not pg_notify.
"""

from __future__ import annotations

import time
from typing import Any, Callable

from queue_workflows import node_queue, run_store
from queue_workflows.db import connection


def cancel_run_via_rails(run_id: str, *, error: str = "cancelled by user") -> None:
    """Issue the same UPDATE the host's cancel handler issues: set
    ``status='cancelled'``, ``error``, and ``finished_at``. Does NOT touch
    ``workflow_node_jobs`` — that's the dispatcher's job."""
    with connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            UPDATE workflow_runs
               SET status = 'cancelled',
                   error  = %s,
                   finished_at = now()
             WHERE id = %s
            """,
            (error, run_id),
        )
        conn.commit()


def wait_for_predicate(
    predicate: Callable[[], bool],
    *,
    timeout_s: float = 10.0,
    interval_s: float = 0.05,
    description: str = "predicate",
) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if predicate():
            return
        time.sleep(interval_s)
    raise AssertionError(
        f"timed out after {timeout_s}s waiting for {description}"
    )


def wait_for_run_status(
    run_id: str, statuses: set[str], *, timeout_s: float = 10.0,
) -> dict[str, Any]:
    last: dict[str, Any] = {}

    def check() -> bool:
        nonlocal last
        run = run_store.get_run(run_id) or {}
        last = run
        return run.get("status") in statuses

    try:
        wait_for_predicate(
            check, timeout_s=timeout_s,
            description=f"run {run_id} status ∈ {sorted(statuses)}",
        )
    except AssertionError:
        raise AssertionError(
            f"run {run_id} did not reach {sorted(statuses)!r} within "
            f"{timeout_s}s; last status={last.get('status')!r} "
            f"error={last.get('error')!r}"
        )
    return last


def wait_for_job_status(
    job_id: str, statuses: set[str], *, timeout_s: float = 10.0,
) -> dict[str, Any]:
    last: dict[str, Any] = {}

    def check() -> bool:
        nonlocal last
        row = node_queue.get_node_job(job_id) or {}
        last = row
        return row.get("status") in statuses

    try:
        wait_for_predicate(
            check, timeout_s=timeout_s,
            description=f"job {job_id} status ∈ {sorted(statuses)}",
        )
    except AssertionError:
        raise AssertionError(
            f"job {job_id} did not reach {sorted(statuses)!r} within "
            f"{timeout_s}s; last status={last.get('status')!r}"
        )
    return last

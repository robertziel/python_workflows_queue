"""Minimal run-row read/write over ``workflow_runs`` — the engine's run-state
seam (the ``repo`` inversion, plan §1d-A).

``workflow_runs`` is the DAG-run record and the queue's substrate: the engine
created it (migration 0001) and several engine modules read/write it —
``dispatcher`` (get/update on completion/failure), ``node_executor`` (get for
the out_dir + context), ``cancel_watcher`` (poll status), ``node_pool``
(update on expand). This module is that minimal psycopg surface. It is a LEAF:
it imports only ``db`` (the pool) — nothing up the engine stack — so
``dispatcher``/``node_pool`` can import it without a cycle.

A host that joins ``workflow_runs`` to its own tables (parcels, Rails views)
keeps its own ORM views of the SAME table; this is the project's established
"two ORMs, one Postgres" pattern. The engine deliberately treats ``parcel_id``
as an opaque nullable column (the engine's migration 0001 drops the parcels
FK), so ``run_store`` never needs to know about the host's domain.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from queue_workflows.db import connection


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _as_json(value: Any):
    from psycopg.types.json import Jsonb
    return Jsonb(value if value is not None else {})


def get_run(run_id: str) -> dict[str, Any] | None:
    """Return the ``workflow_runs`` row as a dict, or ``None`` if missing."""
    with connection() as conn, conn.cursor() as cur:
        cur.execute("SELECT * FROM workflow_runs WHERE id = %s", (run_id,))
        return cur.fetchone()


# Whitelist of columns update_run() accepts. Typos surface at call time
# instead of silently dropping. Mirrors ai_leads' queries._UPDATABLE for the
# columns the engine touches (parcel_id stays writable so a host that keeps
# the run_store path for inserts can still set it).
_UPDATABLE = frozenset({
    "status", "priority", "current_step_id",
    "progress_pct", "steps_done", "context", "input_spec", "error",
    "out_dir", "resume_count", "parcel_id", "mode",
    "queued_at", "started_at", "finished_at",
})

# Columns that are JSONB in workflow_runs — wrapped with Jsonb on write.
_JSON_COLS = frozenset({"steps_done", "context", "input_spec"})


def update_run(run_id: str, **fields: Any) -> dict[str, Any] | None:
    """Update whitelisted columns on a run row; returns the updated row dict
    (or ``None`` if the row is missing). Always bumps ``updated_at``."""
    if not fields:
        return get_run(run_id)
    bad = set(fields) - _UPDATABLE
    if bad:
        raise ValueError(f"unknown columns: {sorted(bad)!r}")
    cols = list(fields.keys())
    set_frags = []
    params: list[Any] = []
    for c in cols:
        set_frags.append(f"{c} = %s")
        v = fields[c]
        params.append(_as_json(v) if c in _JSON_COLS else v)
    set_frags.append("updated_at = %s")
    params.append(_now())
    params.append(run_id)
    sql = (
        f"UPDATE workflow_runs SET {', '.join(set_frags)} "
        "WHERE id = %s RETURNING *"
    )
    with connection() as conn, conn.cursor() as cur:
        cur.execute(sql, params)
        return cur.fetchone()


def insert_run(
    *,
    run_id: str,
    workflow_name: str,
    parcel_id: str | None = None,
    out_dir: str | None = None,
    status: str = "queued",
    priority: int = 100,
    mode: str = "node",
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Insert a fresh ``workflow_runs`` row. ``mode`` defaults to ``node``
    (the only live mode post-Phase-5). Returns the inserted row dict."""
    with connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO workflow_runs
                (id, parcel_id, workflow_name, status, priority,
                 out_dir, mode, context, queued_at, created_at, updated_at)
            VALUES
                (%s, %s, %s, %s, %s,
                 %s, %s, %s, %s, now(), now())
            RETURNING *
            """,
            (
                run_id, parcel_id, workflow_name, status, priority,
                out_dir, mode, _as_json(context or {}),
                _now() if status == "queued" else None,
            ),
        )
        return cur.fetchone()


def delete_run(run_id: str) -> None:
    """Hard-delete one run. ``workflow_run_files`` / ``workflow_node_jobs`` /
    ``workflow_dispatch_events`` / ``workflow_input_submissions`` cascade via
    their FKs."""
    with connection() as conn, conn.cursor() as cur:
        cur.execute("DELETE FROM workflow_runs WHERE id = %s", (run_id,))


def reenqueue_running_for_resume() -> int:
    """Startup hook: orphan ``running`` rows flip back to ``queued`` at
    priority=10. Cap at 5 auto-resumes — beyond that the run moves to
    ``failed`` to break crash loops. Returns the number of rows touched.

    Plain-SQL port of ai_leads' ``queries.reenqueue_running_for_resume``."""
    with connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            UPDATE workflow_runs
            SET status = CASE WHEN resume_count >= 5 THEN 'failed'
                              ELSE 'queued' END,
                error = CASE WHEN resume_count >= 5
                             THEN COALESCE(error, '') || ' [auto-resume cap reached]'
                             ELSE error END,
                priority = 10,
                resume_count = resume_count + 1,
                queued_at = CASE WHEN resume_count >= 5 THEN queued_at
                                 ELSE now() END,
                finished_at = CASE WHEN resume_count >= 5 THEN now()
                                   ELSE finished_at END,
                updated_at = now()
            WHERE status = 'running'
            RETURNING id
            """,
        )
        return len(cur.fetchall())


def claim_next_queued() -> dict[str, Any] | None:
    """Atomically pick the oldest queued ``mode='step'`` run, flip to
    ``running``. ``FOR UPDATE SKIP LOCKED`` keeps concurrent workers from
    claiming the same row. Node-mode runs are expanded by the dispatcher, not
    claimed here — so this excludes them (parity with ai_leads' query).

    Kept for parity / standalone step-mode use; the live node engine doesn't
    call it."""
    with connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            UPDATE workflow_runs
            SET status = 'running',
                started_at = COALESCE(started_at, now()),
                updated_at = now()
            WHERE id = (
                SELECT id FROM workflow_runs
                WHERE status = 'queued' AND mode = 'step'
                ORDER BY priority ASC, queued_at ASC
                FOR UPDATE SKIP LOCKED
                LIMIT 1
            )
            RETURNING *
            """,
        )
        return cur.fetchone()

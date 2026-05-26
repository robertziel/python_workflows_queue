"""Node-per-job queue — DB helpers.

Thin SQL layer on top of the ``workflow_node_jobs`` table. All queries
use the psycopg connection pool established in :mod:`queue_workflows.db`
so they interoperate with the rest of the engine.

This module is pure plumbing — no subprocess spawning, no torch. Used
by:

- the dispatcher (to enqueue newly-satisfied nodes),
- the CPU pool (to claim + finalise rows),
- the GPU pool (to claim + finalise rows, with a "prefer current
  model" order),
- Rails' queue-snapshot route via a summary query.

The ingest task set (``run_fetch_all`` / etc.) is HOST-CONFIGURABLE (plan
§1f): ``enqueue_ingest_job`` validates ``task_name`` against the host's
registered ``config.ingest_task_map`` (an empty map by default), not a
hard-coded ai_leads-domain frozenset. ``INGEST_QUEUES`` (``fetch``/``load``)
stays an engine constant — it matches the migration-0007 queue CHECK, which is
generic.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any, Iterable

from queue_workflows.db import connection


# ── Types ────────────────────────────────────────────────────────────────


QueueName = str  # 'cpu' | 'gpu' — soft type; enforced by CHECK in DB.
NodeStatus = str  # queued|running|completed|failed|cancelled|awaiting_input


# ── Helpers ──────────────────────────────────────────────────────────────


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _as_json(value: Any) -> Any:
    """psycopg Jsonb shortcut — kept tiny so callers don't import psycopg.types."""
    from psycopg.types.json import Jsonb
    return Jsonb(value or {})


# ── Enqueue ──────────────────────────────────────────────────────────────


def enqueue_node_job(
    *,
    run_id: str,
    node_id: str,
    node_module: str,
    queue: QueueName,
    required_model: str | None = None,
    inputs: dict[str, Any] | None = None,
    priority: int = 100,
    pipeline_name: str | None = None,
) -> str:
    """Insert a fresh ``queued`` node-job row. Returns the row id.

    Invariants:
        - queue in {'cpu', 'gpu'}.
        - queue='cpu' ⇒ required_model IS NULL (CPU tasks don't use
          the model cache).
        - queue='gpu' MAY or MAY NOT have required_model.
        - (run_id, node_id) UNIQUE — one row per DAG cell.

    Raises ``ValueError`` before hitting the DB on a bad queue name
    or a CPU row with a stray model.
    """
    if queue not in ("cpu", "gpu"):
        raise ValueError(f"queue must be 'cpu' or 'gpu', got {queue!r}")
    if queue == "cpu" and required_model:
        raise ValueError("cpu node-job must not set required_model")

    row_id = str(uuid.uuid4())
    with connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO workflow_node_jobs
                (id, run_id, pipeline_name, node_id, node_module,
                 queue, required_model,
                 status, priority, inputs, context_delta,
                 created_at)
            VALUES
                (%s, %s, %s, %s, %s,
                 %s, %s,
                 'queued', %s, %s, '{}'::jsonb,
                 now())
            """,
            (
                row_id, run_id, pipeline_name, node_id, node_module,
                queue, required_model,
                priority, _as_json(inputs or {}),
            ),
        )
    return row_id


def insert_skipped_job(
    *,
    run_id: str,
    node_id: str,
    pipeline_name: str | None = None,
) -> str:
    """Insert a status='skipped' marker row for a node whose
    ``skip_if`` evaluated to true at ready-check time. No worker
    touches it; it exists so dependents see a satisfied predecessor
    via ``_find_ready_nodes``.

    queue='cpu' + required_model NULL satisfies the
    ``workflow_node_jobs_required_model_check`` CHECK; node_module
    stays empty (no Python import) — the row is purely bookkeeping.
    """
    row_id = str(uuid.uuid4())
    with connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO workflow_node_jobs
                (id, run_id, pipeline_name, node_id, node_module,
                 queue, required_model,
                 status, priority, inputs, context_delta,
                 created_at, finished_at)
            VALUES
                (%s, %s, %s, %s, '',
                 'cpu', NULL,
                 'skipped', 100, '{}'::jsonb, '{}'::jsonb,
                 now(), now())
            """,
            (row_id, run_id, pipeline_name, node_id),
        )
    return row_id


def get_node_job(job_id: str) -> dict[str, Any] | None:
    with connection() as conn, conn.cursor() as cur:
        cur.execute("SELECT * FROM workflow_node_jobs WHERE id = %s", (job_id,))
        return cur.fetchone()


def list_jobs_for_run(run_id: str) -> list[dict[str, Any]]:
    with connection() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT * FROM workflow_node_jobs "
            "WHERE run_id = %s ORDER BY created_at ASC",
            (run_id,),
        )
        return list(cur.fetchall())


# ── Claim / update ───────────────────────────────────────────────────────
#
# The live claim path for the Postgres-as-queue backend. These ``claim_next_*``
# helpers are the ``SELECT … FOR UPDATE SKIP LOCKED`` queue: a single atomic
# ``queued → running`` UPDATE that also stamps a lease and folds in the
# run-cancel guard. The ``claim_worker`` loops call them on every wake.
#
# Default lease: 600 s. Independent of job duration — a live worker
# renews its lease (~every 10 s) while running, so a multi-hour job
# keeps its lease via renewal; only a dead/wedged worker lets it
# lapse, at which point ``reclaim_expired_leases`` re-queues the row.
DEFAULT_LEASE_S = 600


def _host_dir(host_priority: int) -> int:
    """Direction sign for the ``created_at`` tiebreak, derived from the
    claiming worker's ``host_priority``.

    The baseline (``host_priority >= 0``) walks the band **oldest-first**
    → claims the **head** of the queue (``+1``). Only an explicit overflow
    host (``host_priority < 0``) sorts last: it walks newest-first → yields
    the head to the priority hosts and claims the **tail** (``-1``).

    Returns ``+1`` or ``-1``; the value is multiplied into
    ``EXTRACT(EPOCH FROM created_at)`` in the ORDER BY.
    """
    return -1 if host_priority < 0 else 1


# The claim is one statement. The SKIP-LOCKED subselect picks the next
# claimable row; the outer UPDATE stamps running + lease + claimed_by.
# ``{order}`` is the only interpolation and is built from validated ints
# / fixed fragments below — never from caller-supplied strings.
#
# Run-cancel guard: the subselect additionally requires the parent run is
# NOT terminal-cancelled/failed, so a worker can't even claim a job whose
# run was cancelled out from under it.
_CLAIM_SQL = """
UPDATE workflow_node_jobs AS j
SET status = 'running',
    started_at = now(),
    worker_lane = %(worker_lane)s,
    claimed_by = %(host)s,
    lease_expires_at = now() + make_interval(secs => %(lease_s)s)
WHERE j.id = (
    SELECT c.id FROM workflow_node_jobs c
    WHERE c.queue = %(queue)s
      AND c.status = 'queued'
      AND EXISTS (
          SELECT 1 FROM workflow_runs r
          WHERE r.id = c.run_id
            AND r.status NOT IN ('cancelled', 'failed')
      )
      {capability}
    ORDER BY {order}
    FOR UPDATE SKIP LOCKED
    LIMIT 1
)
RETURNING *
"""

# Warm-model affinity tiebreak (GPU): rows whose ``required_model``
# equals the worker's currently-loaded model jump to the head of their
# priority band so consecutive same-model jobs don't reload. ``IS NOT
# DISTINCT FROM`` (not ``=``) so NULL=NULL counts as a match.
_AFFINITY_TERM = "(c.required_model IS NOT DISTINCT FROM %(current_model)s) DESC"

# host_priority tiebreak: multiply the creation epoch by ±1 so a
# high-priority host takes the oldest (head) and an overflow host takes
# the newest (tail). See :func:`_host_dir`.
_HOST_DIR_TERM = "(EXTRACT(EPOCH FROM c.created_at) * %(host_dir)s) ASC"


def claim_next_cpu_job(
    worker_lane: int = 0,
    *,
    host: str | None = None,
    lease_s: int = DEFAULT_LEASE_S,
    host_priority: int = 0,
) -> dict[str, Any] | None:
    """Atomically grab the next queued CPU job (the CPU claim worker's
    claim).

    Stamps the lease (``claimed_by=host``,
    ``lease_expires_at=now()+lease_s``) and applies the run-cancel guard.
    CPU has no model cache, so there's no warm-affinity term; ordering
    is ``priority ASC`` then the ``host_priority``-directed creation
    tiebreak."""
    order = f"c.priority ASC, {_HOST_DIR_TERM}"
    # CPU jobs carry no required_model, so no capability gate applies.
    sql = _CLAIM_SQL.format(order=order, capability="")
    params = {
        "worker_lane": worker_lane,
        "host": host,
        "lease_s": int(lease_s),
        "queue": "cpu",
        "host_dir": _host_dir(host_priority),
    }
    with connection() as conn, conn.cursor() as cur:
        cur.execute(sql, params)
        return cur.fetchone()


def claim_next_gpu_job(
    worker_lane: int = 0,
    current_model: str | None = None,
    *,
    host: str | None = None,
    lease_s: int = DEFAULT_LEASE_S,
    host_priority: int = 0,
    known_models: Iterable[str] | None = None,
) -> dict[str, Any] | None:
    """Atomically grab the next queued GPU job (the GPU claim worker's
    claim).

    Same lease + run-cancel guard as :func:`claim_next_cpu_job`, plus:

    * **Capability gate** — only claim a job whose ``required_model`` this
      worker can serve (in ``known_models``) or that needs no model. Restores
      the gate the old Celery ``_gpu_should_accept`` enforced (reject+requeue),
      as a cleaner claim-time filter so an incapable worker never grabs the row
      — a capable peer does. With no ``known_models`` (worker hasn't advertised
      its registry yet) it falls back to claim-any so a cold worker can't wedge
      the queue.
    * **Warm-model affinity tiebreak** — rows whose ``required_model`` matches
      the worker's ``current_model`` (``IS NOT DISTINCT FROM``) sort first
      within their priority band so consecutive same-model jobs don't reload.
      ``host_priority`` then breaks the creation-order tie exactly as on CPU."""
    order = (
        f"{_AFFINITY_TERM}, "
        f"c.priority ASC, "
        f"{_HOST_DIR_TERM}"
    )
    known = [m for m in (known_models or []) if m]
    capability = (
        "AND (c.required_model IS NULL "
        "OR c.required_model = ANY(%(known_models)s::text[]))"
        if known else ""
    )
    sql = _CLAIM_SQL.format(order=order, capability=capability)
    params = {
        "worker_lane": worker_lane,
        "host": host,
        "lease_s": int(lease_s),
        "queue": "gpu",
        "current_model": current_model,
        "host_dir": _host_dir(host_priority),
    }
    if known:
        params["known_models"] = known
    with connection() as conn, conn.cursor() as cur:
        cur.execute(sql, params)
        return cur.fetchone()


# ── Ingest queue ────────────────────────────────────────────────────────────
#
# The periodic ingest work rides a DEDICATED ``ingest_jobs`` table (migration
# 0007), not ``workflow_node_jobs``: a periodic ingest unit is a standalone
# callable with no parent run, so it can't satisfy the NOT NULL ``run_id`` FK
# nor the claim SQL's run-cancel join. These helpers mirror the cpu/gpu
# claim/lease shape (DRY at the SQL level) minus the run-existence guard —
# there's no run.

#: Default ingest queues (ai_leads byte-compat). The LIVE allow-list is
#: ``config.ingest_queues`` (host-configurable, G1) — see :func:`_ingest_queues`.
#: Migration 0008 dropped the DB CHECK that pinned ``ingest_jobs.queue`` to these.
INGEST_QUEUES: frozenset[str] = frozenset({"fetch", "load"})


def _ingest_queues() -> frozenset[str]:
    """The valid ingest queue set — host-configurable via
    ``queue_workflows.configure(ingest_queues=...)`` (default {'fetch','load'}).
    Migration 0008 moved this allow-list from a DB CHECK to here (mirrors the
    task_name gate in :func:`_ingest_tasks`)."""
    from queue_workflows.config import get_config
    return get_config().ingest_queues


def _ingest_tasks() -> frozenset[str]:
    """The valid ingest ``task_name`` set — the keys of the host-registered
    ``config.ingest_task_map`` (plan §1f). Empty until a host registers tasks
    via ``queue_workflows.register_ingest_task``."""
    from queue_workflows.config import get_config
    return frozenset(get_config().ingest_task_map.keys())


def enqueue_ingest_job(
    *, task_name: str, queue: str, reason: str = "tick", priority: int = 100,
    args: dict[str, Any] | None = None, conn: Any = None,
) -> str:
    """Insert a fresh ``queued`` ingest-job row. Returns the row id.

    Raises ``ValueError`` before touching the DB on an unknown queue or
    task_name (must be a registered ingest task), matching
    :func:`enqueue_node_job`'s fail-before-write contract.

    ``args`` (migration 0008) is an optional JSON-able dict of per-job
    arguments — persisted to the ``args`` column and handed to the registered
    callable — so a host can enqueue a *parametrised* ingest task (e.g.
    ``run_scenario`` with a scenario id), not only parameterless periodic
    sweeps. Defaults to ``{}``.

    ``conn`` is an optional host psycopg connection. When given, the INSERT runs
    on it so the **caller controls the transaction** — the job row and the
    host's own domain row (e.g. ``scenario_runs``) commit atomically, and the
    ``ingest_job_ready`` NOTIFY rides the same txn. When ``None``, a pooled
    connection is borrowed and autocommits on success.
    """
    iq = _ingest_queues()
    if queue not in iq:
        raise ValueError(f"ingest queue must be in {sorted(iq)}, got {queue!r}")
    tasks = _ingest_tasks()
    if task_name not in tasks:
        raise ValueError(
            f"task_name must be a registered ingest task {sorted(tasks)}, got "
            f"{task_name!r} (register via queue_workflows.register_ingest_task)"
        )
    row_id = str(uuid.uuid4())
    sql = """
        INSERT INTO ingest_jobs
            (id, task_name, queue, reason, args, status, priority, created_at)
        VALUES (%s, %s, %s, %s, %s, 'queued', %s, now())
    """
    params = (row_id, task_name, queue, reason, _as_json(args or {}), priority)
    if conn is not None:
        with conn.cursor() as cur:
            cur.execute(sql, params)
    else:
        with connection() as own, own.cursor() as cur:
            cur.execute(sql, params)
    return row_id


def get_ingest_job(job_id: str) -> dict[str, Any] | None:
    with connection() as conn, conn.cursor() as cur:
        cur.execute("SELECT * FROM ingest_jobs WHERE id = %s", (job_id,))
        return cur.fetchone()


def claim_next_ingest_job(
    queue: str, *, host: str | None = None, lease_s: int = DEFAULT_LEASE_S,
) -> dict[str, Any] | None:
    """Atomically grab the next queued ingest job on ``queue``.

    Mirrors :func:`claim_next_cpu_job` — a single ``SELECT … FOR UPDATE
    SKIP LOCKED`` claim stamping the lease — but WITHOUT the run-cancel join
    (an ingest job has no parent run). Ordered ``priority ASC`` then FIFO
    on creation. Returns the claimed row, or ``None`` when the queue had
    nothing claimable.
    """
    with connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            UPDATE ingest_jobs AS j
            SET status = 'running',
                started_at = now(),
                claimed_by = %(host)s,
                lease_expires_at = now() + make_interval(secs => %(lease_s)s)
            WHERE j.id = (
                SELECT c.id FROM ingest_jobs c
                WHERE c.queue = %(queue)s
                  AND c.status = 'queued'
                ORDER BY c.priority ASC, c.created_at ASC
                FOR UPDATE SKIP LOCKED
                LIMIT 1
            )
            RETURNING *
            """,
            {"host": host, "lease_s": int(lease_s), "queue": queue},
        )
        return cur.fetchone()


def mark_ingest_completed(
    job_id: str, *, result: dict[str, Any] | None = None, seconds: float | None = None,
) -> dict[str, Any] | None:
    """Transition an ingest row to ``completed`` (idempotent — returns
    ``None`` if already terminal). Same WHERE-narrowed contract as
    :func:`mark_completed`."""
    import json
    json.dumps(result or {})
    with connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            UPDATE ingest_jobs
            SET status = 'completed', finished_at = now(),
                result = %s, seconds = %s
            WHERE id = %s
              AND status NOT IN ('completed', 'failed', 'cancelled')
            RETURNING *
            """,
            (_as_json(result or {}), seconds, job_id),
        )
        return cur.fetchone()


def mark_ingest_failed(
    job_id: str, *, error: str, seconds: float | None = None,
) -> dict[str, Any] | None:
    """Transition an ingest row to ``failed`` (idempotent)."""
    with connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            UPDATE ingest_jobs
            SET status = 'failed', finished_at = now(),
                error = %s, seconds = %s
            WHERE id = %s
              AND status NOT IN ('completed', 'failed', 'cancelled')
            RETURNING *
            """,
            (error[:8000] if error else error, seconds, job_id),
        )
        return cur.fetchone()


def reclaim_expired_ingest_leases() -> list[dict[str, Any]]:
    """Re-queue ``running`` ingest rows whose lease has lapsed.

    The ingest-table twin of :func:`reclaim_expired_leases`: a live claim
    worker renews its lease while running, so a lapsed lease means the
    owner died/wedged. Flip back to ``queued``, clear the lease, and bump
    priority to the front — the status flip fires the ``ingest_job_ready``
    NOTIFY so an idle worker picks it straight back up. Returns the
    reclaimed rows' id / task_name / queue.
    """
    with connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            UPDATE ingest_jobs
            SET status = 'queued',
                started_at = NULL,
                claimed_by = NULL,
                lease_expires_at = NULL,
                priority = LEAST(priority, 10)
            WHERE status = 'running'
              AND lease_expires_at IS NOT NULL
              AND lease_expires_at < now()
            RETURNING id, task_name, queue
            """,
        )
        return list(cur.fetchall())


def reclaim_expired_leases() -> list[dict[str, Any]]:
    """Re-queue ``running`` rows whose lease has lapsed.

    A live worker renews its lease while running, so a lapsed lease means
    the owner died or wedged. Flip such rows back to ``queued``, clear
    the lease bookkeeping, and bump priority to the front
    (``LEAST(priority, 10)``) so the recovered work jumps the queue. The
    status flip fires the ``node_job_ready`` NOTIFY (migration 0006) so an
    idle worker picks it straight back up.

    Only touches rows that carry a lease (``lease_expires_at IS NOT NULL``).

    Returns the reclaimed rows' ``id`` / ``run_id`` / ``node_id`` so the
    caller can re-dispatch or log them."""
    with connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            UPDATE workflow_node_jobs
            SET status = 'queued',
                started_at = NULL,
                claimed_by = NULL,
                lease_expires_at = NULL,
                priority = LEAST(priority, 10)
            WHERE status = 'running'
              AND lease_expires_at IS NOT NULL
              AND lease_expires_at < now()
            RETURNING id, run_id, node_id
            """,
        )
        return list(cur.fetchall())


def reclaim_all_running_for_resume() -> list[dict[str, Any]]:
    """Re-queue EVERY ``running`` node job back to ``queued`` — the restart
    hook, not the lease-expiry path (:func:`reclaim_expired_leases`).

    Called on orchestrator boot: a force-recreate restart (or a crash) has just
    bounced the whole fleet, so any ``running`` row is orphaned — its worker is
    gone. Flip them all back to ``queued`` immediately (clearing the lease
    bookkeeping, jumping the queue via ``LEAST(priority, 10)``) instead of
    waiting up to the full 600 s lease for the expiry sweep, so the fresh
    workers pick the work straight back up. The status flip fires the
    ``node_job_ready`` NOTIFY so an idle worker grabs it at once.

    Safe against a worker that somehow survived the restart and still holds a
    row: clearing ``claimed_by`` trips that worker's :class:`JobStatusWatcher`
    (it polls its row and hard-exits the instant the row is no longer
    claimed-by-it), so the row is never run twice.

    Returns the reclaimed rows' ``id`` / ``run_id`` / ``node_id``."""
    with connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            UPDATE workflow_node_jobs
            SET status = 'queued',
                started_at = NULL,
                claimed_by = NULL,
                lease_expires_at = NULL,
                priority = LEAST(priority, 10)
            WHERE status = 'running'
            RETURNING id, run_id, node_id
            """,
        )
        return list(cur.fetchall())


# ── Worker capacity heartbeat (DRY upsert) ──────────────────────────────────
#
# Single home for the ``worker_heartbeats`` INSERT … ON CONFLICT so the two
# emitters (the claim worker's ``HeartbeatEmitter`` for capacity + the GPU
# model cache's ``_publish_current_model`` for the model slot) share ONE
# statement rather than each carrying its own copy.


def upsert_worker_heartbeat(
    *,
    host_label: str,
    queue: str,
    concurrency: int,
    current_model: str | None = None,
    known_models: Iterable[str] | None = None,
    update_current_model: bool = True,
) -> None:
    """Upsert this worker's ``(host_label, queue)`` capacity row, refreshing
    ``last_seen`` to ``now()``.

    Any queue family may upsert (migration 0008 dropped the cpu/gpu-only
    CHECK), so ingest workers heartbeat too. The GPU heartbeat passes
    ``current_model`` (the gauge's busy signal); CPU + ingest heartbeats leave
    it NULL. ``known_models`` is the capability list advertised for affinity
    routing; ``None`` is normalised to an empty array so the column is never
    left stale.

    ``update_current_model`` controls whether the ON CONFLICT path
    overwrites ``current_model``.
    """
    known = list(known_models) if known_models is not None else []
    model_set = (
        "current_model = EXCLUDED.current_model," if update_current_model else ""
    )
    with connection() as conn, conn.cursor() as cur:
        cur.execute(
            f"""
            INSERT INTO worker_heartbeats
                (host_label, queue, concurrency, last_seen,
                 current_model, known_models)
            VALUES (%s, %s, %s, now(), %s, %s)
            ON CONFLICT (host_label, queue) DO UPDATE
                SET concurrency   = EXCLUDED.concurrency,
                    {model_set}
                    known_models  = EXCLUDED.known_models,
                    last_seen     = EXCLUDED.last_seen
            """,
            (host_label, queue, int(concurrency), current_model, known),
        )


def mark_completed_in_txn(
    cur,
    job_id: str,
    *,
    context_delta: dict[str, Any],
    seconds: float,
    vm_rss_mb_peak: int | None = None,
) -> dict[str, Any] | None:
    """Same as :func:`mark_completed` but runs on a caller-supplied
    cursor — lets the worker write the dispatch-event row in the same
    transaction (outbox atomicity contract)."""
    import json
    json.dumps(context_delta or {})
    cur.execute(
        """
        UPDATE workflow_node_jobs
        SET status = 'completed',
            finished_at = now(),
            context_delta = %s,
            seconds = %s,
            vm_rss_mb_peak = %s
        WHERE id = %s
          AND status NOT IN ('completed', 'failed', 'cancelled')
        RETURNING *
        """,
        (_as_json(context_delta), seconds, vm_rss_mb_peak, job_id),
    )
    return cur.fetchone()


def mark_completed(
    job_id: str,
    *,
    context_delta: dict[str, Any],
    seconds: float,
    vm_rss_mb_peak: int | None = None,
) -> dict[str, Any] | None:
    """Transition a row from a non-terminal state to ``completed``.

    Returns the updated row, or ``None`` when the row was already
    in a terminal state. The WHERE clause is the load-bearing piece —
    without it, a stray second call would silently overwrite a
    freshly-finalised ``context_delta`` with whatever the second call
    computed (often ``{}``).

    The allowlist is "any non-terminal state": ``queued``, ``running``,
    ``awaiting_input``."""
    with connection() as conn, conn.cursor() as cur:
        return mark_completed_in_txn(
            cur, job_id, context_delta=context_delta,
            seconds=seconds, vm_rss_mb_peak=vm_rss_mb_peak,
        )


def mark_failed_in_txn(
    cur,
    job_id: str,
    *,
    error: str,
    seconds: float | None = None,
) -> dict[str, Any] | None:
    """Same-txn variant of :func:`mark_failed`."""
    cur.execute(
        """
        UPDATE workflow_node_jobs
        SET status = 'failed',
            finished_at = now(),
            error = %s,
            seconds = %s
        WHERE id = %s
          AND status NOT IN ('completed', 'failed', 'cancelled')
        RETURNING *
        """,
        (error[:8000] if error else error, seconds, job_id),
    )
    return cur.fetchone()


def mark_failed(
    job_id: str, *, error: str, seconds: float | None = None,
) -> dict[str, Any] | None:
    """Transition a row from a non-terminal state to ``failed``.

    Returns the updated row, or ``None`` when the row was already
    in a terminal state. Same idempotency reasoning as
    :func:`mark_completed`."""
    with connection() as conn, conn.cursor() as cur:
        return mark_failed_in_txn(
            cur, job_id, error=error, seconds=seconds,
        )


def mark_awaiting_input_in_txn(cur, job_id: str) -> dict[str, Any] | None:
    """Same-txn variant of :func:`mark_awaiting_input`."""
    cur.execute(
        "UPDATE workflow_node_jobs "
        "SET status = 'awaiting_input' "
        "WHERE id = %s "
        "  AND status NOT IN ('completed', 'failed', 'cancelled') "
        "RETURNING *",
        (job_id,),
    )
    return cur.fetchone()


def mark_awaiting_input(job_id: str) -> dict[str, Any] | None:
    with connection() as conn, conn.cursor() as cur:
        return mark_awaiting_input_in_txn(cur, job_id)


def set_input_spec(run_id: str, node_id: str, spec: dict | None) -> None:
    """Persist a per-job ``input_spec`` so the frontend can render
    N awaiting inputs side-by-side. Idempotent — re-calling with the
    same spec is a no-op write."""
    import json
    payload = json.dumps(spec) if spec is not None else None
    with connection() as conn, conn.cursor() as cur:
        cur.execute(
            "UPDATE workflow_node_jobs "
            "SET input_spec = %s::jsonb "
            "WHERE run_id = %s AND node_id = %s "
            "  AND status = 'awaiting_input'",
            (payload, run_id, node_id),
        )


# ── Dispatch-event outbox ─────────────────────────────────────────────────


def enqueue_dispatch_event_in_txn(
    cur, run_id: str, node_id: str, kind: str,
) -> int:
    """Insert a dispatch event in the caller's transaction. ``kind``
    is one of 'completed' / 'failed' / 'awaiting_input' — DB CHECK
    enforces the set."""
    cur.execute(
        """
        INSERT INTO workflow_dispatch_events (run_id, node_id, kind)
        VALUES (%s, %s, %s)
        RETURNING id
        """,
        (run_id, node_id, kind),
    )
    row = cur.fetchone()
    return int(row["id"])


def list_unprocessed_dispatch_events(
    *, limit: int = 50,
) -> list[dict[str, Any]]:
    """Return up-to-``limit`` rows where ``processed_at IS NULL``.

    Read-only inspection helper for tests, the snapshot endpoint, and
    the ``DispatchDriver.drain_until_quiescent`` no-progress probe.
    The drain path lives in :meth:`NodePool._drain_dispatch_events`."""
    with connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, run_id, node_id, kind, attempts, error
              FROM workflow_dispatch_events
             WHERE processed_at IS NULL
             ORDER BY created_at ASC
             LIMIT %s
            """,
            (int(limit),),
        )
        return list(cur.fetchall())


def mark_dispatch_event_processed(event_id: int) -> None:
    with connection() as conn, conn.cursor() as cur:
        cur.execute(
            "UPDATE workflow_dispatch_events "
            "SET processed_at = now() WHERE id = %s",
            (event_id,),
        )


def record_dispatch_event_failure(event_id: int, error: str) -> None:
    """Increment ``attempts`` and record the error text without
    setting ``processed_at`` so the next tick retries."""
    with connection() as conn, conn.cursor() as cur:
        cur.execute(
            "UPDATE workflow_dispatch_events "
            "SET attempts = attempts + 1, error = %s "
            "WHERE id = %s",
            (error[:8000] if error else error, event_id),
        )


def count_unprocessed_dispatch_events() -> int:
    """Cheap COUNT for the queue snapshot endpoint + startup health log."""
    with connection() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT COUNT(*) AS n FROM workflow_dispatch_events "
            "WHERE processed_at IS NULL"
        )
        return int(cur.fetchone()["n"])


def cancel_queued_jobs_for_run(run_id: str) -> int:
    """Flip all ``queued`` jobs for a run to ``cancelled``. Running jobs
    are left alone — workers notice the run's cancel flag between jobs.
    Returns the number of rows touched.
    """
    with connection() as conn, conn.cursor() as cur:
        cur.execute(
            "UPDATE workflow_node_jobs "
            "SET status = 'cancelled', finished_at = now() "
            "WHERE run_id = %s AND status = 'queued'",
            (run_id,),
        )
        return cur.rowcount or 0


def cancel_siblings_after_failure(run_id: str) -> int:
    """When a node fails, cancel any still-queued jobs for the run —
    downstream work is moot. Returns rows touched.
    """
    return cancel_queued_jobs_for_run(run_id)


def set_resolved_inputs(job_id: str, resolved_inputs: dict[str, Any]) -> None:
    """Write the execution-time snapshot of resolved inputs into the
    ``resolved_inputs`` column. Called by the worker just before
    invoking the node module so the snapshot reflects exactly what
    the node received — useful for forensics when the upstream
    context changes between enqueue and execute.

    Pre-validates JSON serialisation so a bad input fails before any
    state mutation, matching :func:`mark_completed`'s contract."""
    import json
    json.dumps(resolved_inputs or {})
    with connection() as conn, conn.cursor() as cur:
        cur.execute(
            "UPDATE workflow_node_jobs "
            "SET resolved_inputs = %s WHERE id = %s",
            (_as_json(resolved_inputs or {}), job_id),
        )


# ── Snapshot for Rails ───────────────────────────────────────────────────


def snapshot() -> dict[str, Any]:
    """Return counts and running/queued rows per queue, for the
    queue-indicator UI. Keeps the payload small — at most 50 rows per
    section.
    """
    with connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT queue, status, COUNT(*) AS n
            FROM workflow_node_jobs
            GROUP BY queue, status
            """
        )
        counts: dict[tuple[str, str], int] = {}
        for row in cur.fetchall():
            counts[(row["queue"], row["status"])] = int(row["n"])

        def _top(queue: str, status: str, limit: int = 50) -> list[dict[str, Any]]:
            cur.execute(
                "SELECT * FROM workflow_node_jobs "
                "WHERE queue = %s AND status = %s "
                "ORDER BY created_at ASC LIMIT %s",
                (queue, status, limit),
            )
            return list(cur.fetchall())

        return {
            "cpu": {
                "running": _top("cpu", "running"),
                "queued":  _top("cpu", "queued"),
            },
            "gpu": {
                "running": _top("gpu", "running"),
                "queued":  _top("gpu", "queued"),
            },
            "counts": {
                f"{q}_{s}": n for (q, s), n in counts.items()
            },
        }


def ingest_snapshot() -> dict[str, Any]:
    """Per-queue depth + live-worker counts for the INGEST path — the
    ``ingest_jobs`` twin of :func:`snapshot` (which covers only the cpu/gpu DAG
    queues). ``queues[q]`` carries the status counts plus ``workers`` = the
    number of ``worker_heartbeats`` rows on that queue still fresh (< 30 s,
    matching the claim worker's 10 s refresh). A host maps queued+running →
    "messages" and ``workers`` → "consumers" for its queue-indicator UI.

    NB: ``worker_heartbeats`` is keyed ``(host_label, queue)``, so ``workers``
    counts live worker *hosts* per queue, not processes — enough to drive the
    "no consumer → starvation" warning."""
    with connection() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT queue, status, COUNT(*) AS n FROM ingest_jobs "
            "GROUP BY queue, status"
        )
        counts: dict[str, dict[str, int]] = {}
        for row in cur.fetchall():
            counts.setdefault(row["queue"], {})[row["status"]] = int(row["n"])
        cur.execute(
            "SELECT queue, COUNT(*) AS n FROM worker_heartbeats "
            "WHERE last_seen > now() - interval '30 seconds' GROUP BY queue"
        )
        workers = {row["queue"]: int(row["n"]) for row in cur.fetchall()}

    queues: dict[str, dict[str, int]] = {}
    for q in set(counts) | set(workers):
        st = counts.get(q, {})
        queues[q] = {
            "queued": st.get("queued", 0),
            "running": st.get("running", 0),
            "completed": st.get("completed", 0),
            "failed": st.get("failed", 0),
            "workers": workers.get(q, 0),
        }
    return {"queues": queues}

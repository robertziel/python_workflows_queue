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
import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Any, Iterable

from queue_workflows.db import connection

log = logging.getLogger(__name__)


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


def _project(project: str | None) -> str:
    """Resolve the tenant tag for an enqueue/claim (migration 0017).

    ``None`` ⇒ this client's configured ``config.project`` (the common path: a
    per-project client sets it once at startup and every enqueue/claim inherits
    it). An explicit string overrides it (the dispatcher passes the parent run's
    project; tests pin it). Default ``""`` is the single-tenant sentinel — see
    :attr:`queue_workflows.config.EngineConfig.project`."""
    if project is not None:
        return project
    from queue_workflows.config import get_config
    return get_config().project or ""


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
    project: str | None = None,
) -> str:
    """Insert a fresh ``queued`` node-job row. Returns the row id.

    ``project`` (migration 0017) is the tenant tag stamped on the row; ``None``
    ⇒ this client's ``config.project``. The dispatcher passes the parent run's
    project so the node inherits it. See :func:`_project`.

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
                 queue, required_model, project,
                 status, priority, inputs, context_delta,
                 created_at)
            VALUES
                (%s, %s, %s, %s, %s,
                 %s, %s, %s,
                 'queued', %s, %s, '{}'::jsonb,
                 now())
            """,
            (
                row_id, run_id, pipeline_name, node_id, node_module,
                queue, required_model, _project(project),
                priority, _as_json(inputs or {}),
            ),
        )
    return row_id


def insert_skipped_job(
    *,
    run_id: str,
    node_id: str,
    pipeline_name: str | None = None,
    project: str | None = None,
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
                 queue, required_model, project,
                 status, priority, inputs, context_delta,
                 created_at, finished_at)
            VALUES
                (%s, %s, %s, %s, '',
                 'cpu', NULL, %s,
                 'skipped', 100, '{}'::jsonb, '{}'::jsonb,
                 now(), now())
            """,
            (row_id, run_id, pipeline_name, node_id, _project(project)),
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


def prioritize_node_job(job_id: str) -> dict[str, Any] | None:
    """Flag a QUEUED node job to "run next" — set ``is_priority`` so the next
    worker asking for a node in its queue + capability claims it before older +
    default-priority peers (``is_priority`` sorts FIRST in the claim ORDER BY,
    ahead of the priority band and the GPU warm-model affinity tiebreak).

    A no-op on a non-queued row — a job that's already running/terminal can't be
    re-ordered. Returns the updated row, or ``None`` if nothing was queued under
    that id (so the caller can tell the flag didn't take).
    """
    with connection() as conn, conn.cursor() as cur:
        cur.execute(
            "UPDATE workflow_node_jobs SET is_priority = TRUE "
            "WHERE id = %s AND status = 'queued' "
            "RETURNING *",
            (job_id,),
        )
        return cur.fetchone()


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

# Staleness threshold for the orchestrator-side dead-worker detector
# (:func:`flag_stale_workers_holding_running_jobs`). A live claim worker
# refreshes its ``worker_heartbeats`` row every ``HEARTBEAT_INTERVAL_S`` (10 s);
# 3× that (30 s) is the same window Rails' queue gauge uses to call a row stale,
# so a heartbeat older than this is a worker that stopped beating — wedged or
# dead. Env-overridable for ops tuning.
STALE_WORKER_AFTER_S = 30


def _stale_worker_after_s() -> int:
    raw = (os.environ.get("AI_LEADS_STALE_WORKER_AFTER_S", "") or "").strip()
    if not raw:
        return STALE_WORKER_AFTER_S
    try:
        return max(1, int(float(raw)))
    except (TypeError, ValueError):
        return STALE_WORKER_AFTER_S


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
      AND c.project = %(project)s
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
    project: str | None = None,
) -> dict[str, Any] | None:
    """Atomically grab the next queued CPU job (the CPU claim worker's
    claim).

    Stamps the lease (``claimed_by=host``,
    ``lease_expires_at=now()+lease_s``) and applies the run-cancel guard.
    CPU has no model cache, so there's no warm-affinity term; ordering
    is ``is_priority DESC`` (the operator "run next" flag jumps the queue),
    then ``priority ASC``, then the ``host_priority``-directed creation
    tiebreak."""
    order = f"c.is_priority DESC, c.priority ASC, {_HOST_DIR_TERM}"
    # CPU jobs carry no required_model, so no capability gate applies.
    sql = _CLAIM_SQL.format(order=order, capability="")
    params = {
        "worker_lane": worker_lane,
        "host": host,
        "lease_s": int(lease_s),
        "queue": "cpu",
        "project": _project(project),
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
    require_model: bool | None = None,
    pool_modules: Iterable[str] | None = None,
    project: str | None = None,
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
      the worker's ``current_model`` (``IS NOT DISTINCT FROM``) sort ahead of
      the integer ``priority`` band (only the operator "run next" ``is_priority``
      flag outranks affinity), so consecutive same-model jobs don't reload — a
      warm-model job preempts a colder one even from a worse priority band.
      ``host_priority`` then breaks the creation-order tie exactly as on CPU.
    * **Model-presence lane filter** (``require_model``) — splits the GPU queue
      into two disjoint claim sets so a two-lane GPU worker (inline warm-model
      diffusion lane + a PAR-sized no-model VLM pool lane) never over-claims or
      steals the other lane's rows. ``None`` (default) keeps the existing
      claim-any behaviour; ``True`` adds ``AND c.required_model IS NOT NULL``
      (model-backed diffusion jobs only); ``False`` adds
      ``AND c.required_model IS NULL`` (no-model GPU jobs — VLM/HTTP — only).
      Orthogonal to and ANDed with the capability gate above.
    * **VLM-pool eligibility** (``pool_modules``) — when non-empty, only these
      node modules are genuine VLM-facade (pool-safe). The POOL lane then claims
      a no-model job ONLY if its ``node_module`` is in ``pool_modules``; the
      INLINE lane additionally claims no-model jobs whose module is NOT in the
      set, so heavy in-process GPU work (erasers, detectors, builders) runs on
      the conc-1 serial lane instead of PAR-concurrently in the pool. Empty /
      unset ⇒ legacy split (every no-model GPU job is pool-eligible)."""
    order = (
        # The operator "run next" flag jumps the whole queue — ahead of the
        # warm-model affinity tiebreak too (a flagged cold-model node preempts a
        # warm one; the reload is the accepted cost of "run this next").
        f"c.is_priority DESC, "
        f"{_AFFINITY_TERM}, "
        f"c.priority ASC, "
        f"{_HOST_DIR_TERM}"
    )
    known = [m for m in (known_models or []) if m]
    pool = [m for m in (pool_modules or []) if m]
    capability_terms = []
    if known:
        capability_terms.append(
            "AND (c.required_model IS NULL "
            "OR c.required_model = ANY(%(known_models)s::text[]))"
        )
    if require_model is True:
        if pool:
            # Inline lane: model-backed jobs PLUS no-model jobs whose module is
            # NOT VLM-pool-eligible — heavy in-process GPU work (erasers,
            # detectors, builders) runs conc-1 here, never concurrently in the
            # pool.
            capability_terms.append(
                "AND (c.required_model IS NOT NULL "
                "OR NOT (c.node_module = ANY(%(pool_modules)s::text[])))"
            )
        else:
            capability_terms.append("AND c.required_model IS NOT NULL")
    elif require_model is False:
        if pool:
            # Pool lane: no-model jobs whose module IS VLM-pool-eligible only.
            capability_terms.append(
                "AND c.required_model IS NULL "
                "AND c.node_module = ANY(%(pool_modules)s::text[])"
            )
        else:
            capability_terms.append("AND c.required_model IS NULL")
    capability = " ".join(capability_terms)
    sql = _CLAIM_SQL.format(order=order, capability=capability)
    params = {
        "worker_lane": worker_lane,
        "host": host,
        "lease_s": int(lease_s),
        "queue": "gpu",
        "project": _project(project),
        "current_model": current_model,
        "host_dir": _host_dir(host_priority),
    }
    if known:
        params["known_models"] = known
    if pool:
        params["pool_modules"] = pool
    with connection() as conn, conn.cursor() as cur:
        cur.execute(sql, params)
        return cur.fetchone()


def vlm_pool_should_defer(
    host_label: str, par: int, *, stale_s: int = STALE_WORKER_AFTER_S,
    project: str | None = None,
) -> bool:
    """FILL-BEFORE-SPILL gate for the no-model GPU (VLM) pool lane.

    A vLLM/ollama machine's pool feeder calls this BEFORE each no-model claim to
    decide whether to *defer* this cycle. The fleet ranks GPU machines by
    advertised PAR — ``(worker_heartbeats.concurrency DESC, host_label ASC)`` —
    so a high-`--max-num-seqs` vLLM box ranks first and a PAR-1 ollama box ranks
    last. This machine ``M = (host_label, par)`` defers IFF some FRESH gpu worker
    ``R`` ranked STRICTLY ABOVE it still has free VLM capacity:

      * ranked above:  ``R.concurrency > M.par
                         OR (R.concurrency = M.par AND R.host_label < M.host)``
      * free capacity: ``(running no-model gpu jobs claimed_by R) < R.concurrency``

    The effect is bin-packing: VLM jobs fill the highest-ranked box's PAR slots
    first and spill to the next box only when the higher one is full — under
    light load they consolidate onto one machine (freeing the others for
    diffusion / idle-unload), under heavy load they still spill (no throughput
    loss). Invariants (all enforced by the EXISTS query, no Python branching):

      * **Top-ranked never defers.** The max-PAR / earliest-host machine has no
        ``R`` above it → the EXISTS is empty → it always fills first. No global
        starvation.
      * **Single box / no fresh higher peer → never defers** (returns ``False``)
        ⇒ behaviour byte-identical to today. SAFE default that protects
        single-box fleets and other library consumers.
      * **Stale peers don't count.** ``R`` must be fresh
        (``last_seen > now() - stale_s``); a dead top box can't block everyone.
      * **Pool lane only.** Capacity counts ``queue='gpu' AND status='running'
        AND required_model IS NULL`` — the inline diffusion lane (a warm-model
        job) is neither counted nor affected.

    A single cheap query: ``worker_heartbeats`` (fresh gpu rows ranked above M)
    LEFT JOINed to a per-``claimed_by`` COUNT of its running no-model gpu jobs,
    short-circuited by ``EXISTS``. ``stale_s`` mirrors the heartbeat freshness
    window (default :data:`STALE_WORKER_AFTER_S` = 30 s)."""
    sql = """
        SELECT EXISTS (
            SELECT 1
              FROM worker_heartbeats r
              LEFT JOIN (
                  SELECT claimed_by, COUNT(*) AS running_no_model
                    FROM workflow_node_jobs
                   WHERE queue = 'gpu'
                     AND status = 'running'
                     AND required_model IS NULL
                     AND project = %(project)s
                   GROUP BY claimed_by
              ) j ON j.claimed_by = r.host_label
             WHERE r.queue = 'gpu'
               AND r.project = %(project)s
               AND r.last_seen > now() - make_interval(secs => %(stale_s)s)
               -- ranked STRICTLY above M = (host_label, par)
               AND (
                    r.concurrency > %(par)s
                 OR (r.concurrency = %(par)s AND r.host_label < %(host)s)
               )
               -- R still has free VLM capacity
               AND COALESCE(j.running_no_model, 0) < r.concurrency
        ) AS should_defer
    """
    params = {
        "host": host_label,
        "par": int(par),
        "stale_s": max(1, int(stale_s)),
        "project": _project(project),
    }
    with connection() as conn, conn.cursor() as cur:
        cur.execute(sql, params)
        return bool(cur.fetchone()["should_defer"])


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
    project: str | None = None,
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
            (id, task_name, queue, reason, args, project, status, priority, created_at)
        VALUES (%s, %s, %s, %s, %s, %s, 'queued', %s, now())
    """
    params = (
        row_id, task_name, queue, reason, _as_json(args or {}),
        _project(project), priority,
    )
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
    project: str | None = None,
) -> dict[str, Any] | None:
    """Atomically grab the next queued ingest job on ``queue``.

    Mirrors :func:`claim_next_cpu_job` — a single ``SELECT … FOR UPDATE
    SKIP LOCKED`` claim stamping the lease — but WITHOUT the run-cancel join
    (an ingest job has no parent run). Ordered ``priority ASC`` then FIFO
    on creation. Project-scoped (migration 0017): only rows whose ``project``
    matches this client's (``None`` ⇒ ``config.project``). Returns the claimed
    row, or ``None`` when the queue had nothing claimable.
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
                  AND c.project = %(project)s
                ORDER BY c.priority ASC, c.created_at ASC
                FOR UPDATE SKIP LOCKED
                LIMIT 1
            )
            RETURNING *
            """,
            {
                "host": host, "lease_s": int(lease_s), "queue": queue,
                "project": _project(project),
            },
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
    """Re-queue ``running`` rows whose lease has lapsed — UNLESS the
    parent run is no longer active.

    A live worker renews its lease while running, so a lapsed lease
    means the owner died or wedged. Two outcomes depending on the
    parent run's status:

      * ``running`` parent — flip the row to ``queued``, clear the
        lease bookkeeping, bump priority to the front
        (``LEAST(priority, 10)``) so the recovered work jumps the
        queue. The status flip fires the ``node_job_ready`` NOTIFY
        (migration 0006) so an idle worker picks it straight back up.
      * ``cancelled`` / ``failed`` / ``completed`` parent — flip the
        row to ``cancelled`` instead. The claim SQL filters out jobs
        whose parent is non-running, so re-queuing here would create
        a ghost: the row sits in ``queued`` forever (no worker will
        ever claim it) while the queue popover keeps reporting "+1
        queued." Regression: cancelled run
        ``41570ecd-566e-4281-8b12-4e925fceebd1`` left
        ``reconstruct/render_hunyuan_i2v`` orphaned for 20+ minutes
        after a worker restart killed the in-flight render. Pinned
        by ``test_lease_reclaim_cancels_running_row_on_cancelled_parent``.

    Only touches rows that carry a lease (``lease_expires_at IS NOT NULL``).

    Returns the reclaimed rows' ``id`` / ``run_id`` / ``node_id`` so the
    caller can re-dispatch or log them."""
    with connection() as conn, conn.cursor() as cur:
        # Single UPDATE branches on the parent run's status. The
        # CASE-on-target keeps this atomic — no race window between an
        # is-parent-active check and the lease flip.
        cur.execute(
            """
            UPDATE workflow_node_jobs j
            SET status = CASE WHEN r.status = 'running'
                              THEN 'queued'
                              ELSE 'cancelled'
                         END,
                started_at = CASE WHEN r.status = 'running'
                                  THEN NULL
                                  ELSE j.started_at
                             END,
                claimed_by = NULL,
                lease_expires_at = NULL,
                priority = CASE WHEN r.status = 'running'
                                THEN LEAST(j.priority, 10)
                                ELSE j.priority
                           END,
                finished_at = CASE WHEN r.status = 'running'
                                   THEN j.finished_at
                                   ELSE now()
                              END
            FROM workflow_runs r
            WHERE r.id::text = j.run_id
              AND j.status = 'running'
              AND j.lease_expires_at IS NOT NULL
              AND j.lease_expires_at < now()
            RETURNING j.id, j.run_id, j.node_id
            """,
        )
        return list(cur.fetchall())


def requeue_job_for_retry_in_txn(cur, job_id: str) -> dict[str, Any] | None:
    """Re-queue ONE ``running`` node-job for a watchdog retry — same-cursor
    variant of :func:`requeue_job_for_retry`.

    The watchdog-trip twin of the lease-reclaim mechanic, scoped to a single
    row by id. In one statement it:

      * flips ``running`` → ``queued`` (which fires the migration-0006
        ``node_job_ready`` NOTIFY so an idle worker re-claims it at once);
      * clears the lease bookkeeping (``started_at``/``claimed_by``/
        ``lease_expires_at`` → NULL) so a fresh worker can stamp its own;
      * bumps ``priority`` to the FRONT with ``LEAST(priority, 10)`` — the EXACT
        mechanic :func:`reclaim_expired_leases` uses — so the retry runs
        promptly ahead of newer work;
      * increments ``watchdog_retries`` (migration 0010), the per-job re-queue
        counter the trip site reads to enforce the retry cap.

    CAS-guarded + idempotent like the other transitions: the WHERE narrows to
    ``status = 'running'``, so a row that's already terminal (completed / failed
    / cancelled / skipped) or already ``queued``/``awaiting_input`` is left
    untouched and the function returns ``None``. Writes NO dispatch event — the
    run stays ``running``; only the node re-runs.

    Returns the updated row (with the incremented ``watchdog_retries``), or
    ``None`` on no-match."""
    cur.execute(
        """
        UPDATE workflow_node_jobs
        SET status = 'queued',
            started_at = NULL,
            claimed_by = NULL,
            lease_expires_at = NULL,
            priority = LEAST(priority, 10),
            watchdog_retries = watchdog_retries + 1
        WHERE id = %s
          AND status = 'running'
        RETURNING *
        """,
        (job_id,),
    )
    return cur.fetchone()


def requeue_job_for_retry(job_id: str) -> dict[str, Any] | None:
    """Re-queue a watchdog-tripped node-job so a fresh worker retries it,
    instead of failing the whole run.

    Opens its own connection and delegates to
    :func:`requeue_job_for_retry_in_txn` (one ``UPDATE … RETURNING`` ⇒ one
    transaction). See that function for the full re-queue mechanic + CAS /
    idempotency contract. Returns the updated row, or ``None`` when the row was
    not ``running`` (already terminal / already re-queued)."""
    with connection() as conn, conn.cursor() as cur:
        return requeue_job_for_retry_in_txn(cur, job_id)


def flag_stale_workers_holding_running_jobs(
    *, stale_after_s: int | None = None, project: str | None = None,
) -> list[dict[str, Any]]:
    """Detect + flag workers that have gone SILENT while still owning a
    ``running`` job — the GPU-hardware-hang recovery the lease-reclaim alone
    can't reach.

    THE GAP. A GPU hardware-hang (e.g. a ROCr "GPU Hang" HW exception on a
    wan_i2v render) can leave the claim-worker PROCESS wedged: a torch/HIP call
    blocked in the dead GPU context, with the worker's in-process
    :class:`~queue_workflows.claim_worker.GpuHealthWatchdog` unable to make its
    trip — either because the trip signal is unobservable from inside (on a ROCm
    box the box-level GPU probe can read non-idle even while THIS render is
    wedged, while the hung render holds its weights resident so RAM is static, so
    "GPU idle AND RAM static" is never both-true) or, on a GIL-holding hang,
    because the daemon thread can't run at all. Either way the worker stops
    refreshing its ``worker_heartbeats`` row — it silently quits claiming
    overflow work. The lease-reclaim re-queues the JOB onto a healthy host
    (good), but nothing flags the dead PROCESS so it can be bounced.

    THE DETECTOR. The orchestrator (:class:`~queue_workflows.node_pool.NodePool`)
    runs in a SEPARATE process, GIL-independent of any wedged worker, so it CAN
    observe the frozen heartbeat. This finds every ``worker_heartbeats`` row
    whose ``last_seen`` is older than ``stale_after_s`` (default 30 s = 3× the
    10 s heartbeat cadence) THAT still owns ≥ 1 ``running``
    ``workflow_node_jobs`` row. The job→worker join is BOTH
    ``j.claimed_by = wh.host_label`` AND ``j.queue = wh.queue``: the claim stamps
    ``claimed_by`` with the worker's host label (the value ``worker_heartbeats``
    is keyed on), and the queue match attributes the job to the right worker
    PROCESS on a host that runs several (e.g. host-c runs a cpu AND a gpu worker
    under one ``host_label`` — a wedged gpu worker must not flag the healthy cpu
    worker's row, and vice-versa). It stamps ``last_flagged_dead_at = now()`` on
    the matching rows and RETURNS them so the caller logs a clear, actionable
    line for an operator / host-supervisor to bounce the container.

    IDEMPOTENT. Only rows whose ``last_flagged_dead_at`` is NULL or itself older
    than ``stale_after_s`` are (re)flagged, so the 0.5 s orchestrator tick
    doesn't relog every pass — it flags once, then stays quiet until the worker
    recovers (a fresh heartbeat clears the flag via
    :func:`upsert_worker_heartbeat`) and goes stale again.

    SAFE / NON-DESTRUCTIVE. This does NOT touch the job rows (the lease-reclaim
    owns re-queuing) and does NOT kill anything — a cross-host container kill
    isn't feasible from the orchestrator (no docker socket, different host). It
    surfaces a durable, queryable signal; the host-supervisor hook acts on it.

    Returns the flagged rows' ``host_label`` / ``queue`` / ``last_seen`` /
    ``running_jobs`` (the count of running jobs that worker still owns)."""
    threshold = (
        _stale_worker_after_s() if stale_after_s is None else max(1, int(stale_after_s))
    )
    proj = _project(project)
    with connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            WITH stale AS (
                SELECT wh.host_label, wh.queue, wh.project, wh.last_seen,
                       COUNT(j.id) AS running_jobs
                  FROM worker_heartbeats wh
                  JOIN workflow_node_jobs j
                    ON j.claimed_by = wh.host_label
                   AND j.queue = wh.queue
                   AND j.project = wh.project
                   AND j.status = 'running'
                 WHERE wh.project = %(project)s
                   AND wh.last_seen < now() - make_interval(secs => %(thr)s)
                   AND (
                        wh.last_flagged_dead_at IS NULL
                     OR wh.last_flagged_dead_at < now() - make_interval(secs => %(thr)s)
                   )
                 GROUP BY wh.host_label, wh.queue, wh.project, wh.last_seen
            )
            UPDATE worker_heartbeats wh
               SET last_flagged_dead_at = now()
              FROM stale
             WHERE wh.host_label = stale.host_label
               AND wh.queue = stale.queue
               AND wh.project = stale.project
            RETURNING wh.host_label, wh.queue, wh.project, stale.last_seen,
                      stale.running_jobs
            """,
            {"thr": threshold, "project": proj},
        )
        return list(cur.fetchall())


def reclaim_all_running_for_resume(
    *, project: str | None = None,
) -> list[dict[str, Any]]:
    """Re-queue ``running`` node jobs whose CLAIMING WORKER is gone — the
    orchestrator-boot recovery hook, not the lease-expiry path
    (:func:`reclaim_expired_leases`).

    Project-scoped (migration 0017): only THIS orchestrator's project's jobs
    (``None`` ⇒ ``config.project``). The outer ``UPDATE`` filters
    ``j.project = %(project)s`` — NOT just the heartbeat sub-join — so on a
    shared broker project A's restart can't clear project B's ``claimed_by``
    (which would trip B's live worker's ``JobStatusWatcher`` and kill B's render,
    e.g. a GIL-stalled long CUDA job whose heartbeat froze >30 s while still
    rendering). Default ``""`` matches every job (single-tenant byte-compatible).

    Scoped to jobs whose ``claimed_by`` host has **no fresh heartbeat** on the
    job's queue (last beat older than ``STALE_WORKER_AFTER_S``, or no row at
    all). Such a row is genuinely orphaned — its worker died across the restart
    — so flip it back to ``queued`` at once (clearing the lease bookkeeping,
    jumping the queue via ``LEAST(priority, 10)``) instead of idling out its
    up-to-600 s lease. The status flip fires the ``node_job_ready`` NOTIFY so a
    fresh worker grabs it immediately.

    Why the heartbeat scope (was: re-queue EVERY running row): the original
    assumed a force-recreate had bounced the WHOLE fleet. But the orchestrator/
    dispatcher container restarts independently of the GPU claim workers
    (deploys, hot-fixes, a single-container bounce). Re-queuing *all* running
    rows then yanked HEALTHY in-flight jobs on workers that never restarted
    (e.g. box-a2 mid-diffusion) — clearing ``claimed_by`` trips that live
    worker's :class:`JobStatusWatcher`, which hard-exits it (operator report:
    "box-a2 stopped taking GPU tasks" after an unrelated dispatcher restart). A
    live worker beats every ``HEARTBEAT_INTERVAL_S`` (10 s), so a fresh
    heartbeat reliably means "still running its job — leave it alone".

    Correctness doesn't hinge on perfect timing: any orphan this skips (e.g. a
    full-fleet restart where the new workers beat before this runs) is still
    caught by :func:`reclaim_expired_leases` once the dead lease lapses. So this
    is a fast-path optimisation that is now also SAFE for partial restarts.

    Returns the reclaimed rows' ``id`` / ``run_id`` / ``node_id``."""
    with connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            UPDATE workflow_node_jobs j
            SET status = 'queued',
                started_at = NULL,
                claimed_by = NULL,
                lease_expires_at = NULL,
                priority = LEAST(j.priority, 10)
            WHERE j.status = 'running'
              AND j.project = %(project)s
              AND NOT EXISTS (
                SELECT 1 FROM worker_heartbeats h
                WHERE h.host_label = j.claimed_by
                  AND h.queue = j.queue
                  AND h.project = j.project
                  AND h.last_seen > now() - make_interval(secs => %(stale_s)s)
              )
            RETURNING id, run_id, node_id
            """,
            {"stale_s": float(STALE_WORKER_AFTER_S), "project": _project(project)},
        )
        return list(cur.fetchall())


def requeue_running_for_worker(
    host_label: str, queue: str, *, project: str | None = None,
) -> int:
    """Re-queue every ``running`` row a SPECIFIC worker still owns — the
    per-``(host_label, queue, project)``-scoped twin of
    :func:`reclaim_all_running_for_resume`.

    Project-scoped (migration 0017): ``project`` defaults to this worker's
    ``config.project``. On a shared broker ``host_label`` is no longer globally
    unique (two projects' workers can share a machine + queue), so without the
    project term an operator hard-stop of project A's worker on ``spark2/gpu``
    would also yank project B's running job there. The hard-stop runs inside the
    worker's own ``WorkerControlWatcher``, so the default ``config.project`` is
    exactly that worker's tenant.

    Used when an operator turns a machine's cpu/gpu (or ingest) worker OFF via the
    ``worker_controls`` hard-stop: the in-flight job is released back to the queue
    so a healthy peer (or this worker once it's turned back ON) picks it up — the
    point is to stop the WORKER, not to fail the WORK.

    RESUME-STYLE, not a watchdog retry: flips ``running`` → ``queued``, clears the
    lease bookkeeping, and bumps priority to the front (``LEAST(priority, 10)``)
    WITHOUT incrementing ``watchdog_retries`` (turning a machine off is an
    operational redistribution, not a node failure — it must not burn the retry
    cap). The status flip fires the migration-0006/0007 ``node_job_ready`` /
    ``ingest_job_ready`` NOTIFY so an idle worker re-claims at once.

    Targets ``ingest_jobs`` when ``queue`` is a host-configured ingest queue
    (``config.ingest_queues``), else ``workflow_node_jobs`` (cpu/gpu) — matching
    the table that worker draws from. The table is chosen by this fixed branch
    (never interpolated from caller data); ``host_label`` / ``queue`` are bound
    params. Returns the number of rows re-queued.

    SAFETY (no double-run): clearing ``claimed_by`` is exactly what trips a
    still-running worker's :class:`~queue_workflows.claim_worker.JobStatusWatcher`
    (it hard-exits the instant its row is no longer claimed-by-it) — the same
    guarantee :func:`reclaim_all_running_for_resume` relies on, so the row is never
    run twice across the hand-off."""
    table = "ingest_jobs" if queue in _ingest_queues() else "workflow_node_jobs"
    with connection() as conn, conn.cursor() as cur:
        cur.execute(
            f"""
            UPDATE {table}
            SET status = 'queued',
                started_at = NULL,
                claimed_by = NULL,
                lease_expires_at = NULL,
                priority = LEAST(priority, 10)
            WHERE status = 'running'
              AND claimed_by = %s
              AND queue = %s
              AND project = %s
            """,
            (host_label, queue, _project(project)),
        )
        return cur.rowcount or 0


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
    llm_servers_available: Iterable[str] | None = None,
    vram_total_mb: int | None = None,
    fits_models: Iterable[str] | None = None,
    update_current_model: bool = True,
    project: str | None = None,
) -> None:
    """Upsert this worker's ``(host_label, queue, project)`` capacity row,
    refreshing ``last_seen`` to ``now()``.

    ``project`` (migration 0017) is the worker's tenant tag; ``None`` ⇒
    ``config.project``. It's part of the worker identity so two projects' clients
    on the SAME machine + queue (one shared broker) don't clobber each other's
    heartbeat (the PK is ``(host_label, queue, project)``).

    Any queue family may upsert (migration 0008 dropped the cpu/gpu-only
    CHECK), so ingest workers heartbeat too. The GPU heartbeat passes
    ``current_model`` (the gauge's busy signal); CPU + ingest heartbeats leave
    it NULL. ``known_models`` is the capability list advertised for affinity
    routing; ``None`` is normalised to an empty array so the column is never
    left stale.

    ``llm_servers_available`` is the OBSERVED LLM-server capability (migration
    0014) — which server types this host can actually run (e.g. ``['ollama']`` on
    an AMD box, ``['ollama', 'vllm']`` on an NVIDIA host with the vllm sidecar).
    ``None`` is normalised to the ``['ollama']`` baseline so the column is never
    left stale and a caller that doesn't care (cpu/ingest workers, other consumer
    projects) keeps the safe default.

    ``update_current_model`` controls whether the ON CONFLICT path
    overwrites ``current_model``.

    A live refresh ALSO clears ``last_flagged_dead_at`` (migration 0009): if the
    orchestrator's stale-worker detector had flagged this worker as wedged, a
    fresh heartbeat means it (or its replacement after a bounce) is alive again,
    so the dead-flag is reset — a future hang then re-flags cleanly instead of
    staying latched from the previous incident.
    """
    known = list(known_models) if known_models is not None else []
    llm_servers = (
        list(llm_servers_available) if llm_servers_available is not None else ["ollama"]
    )
    # Capacity advertisement (migration 0015). ``fits_models`` is the worker-
    # computed subset of known ids whose est_vram_gb fits this machine, used by
    # the claim gate and the fleet unassignable sweep; ``None`` ⇒ empty array
    # (advertise no capacity claim) so a non-GPU / pre-capacity worker leaves the
    # column at its '{}' default rather than stale.
    fits = list(fits_models) if fits_models is not None else []
    vram = int(vram_total_mb) if vram_total_mb is not None else None
    model_set = (
        "current_model = EXCLUDED.current_model," if update_current_model else ""
    )
    with connection() as conn, conn.cursor() as cur:
        cur.execute(
            f"""
            INSERT INTO worker_heartbeats
                (host_label, queue, project, concurrency, last_seen,
                 current_model, known_models, llm_servers_available,
                 vram_total_mb, fits_models)
            VALUES (%s, %s, %s, %s, now(), %s, %s, %s, %s, %s)
            ON CONFLICT (host_label, queue, project) DO UPDATE
                SET concurrency   = EXCLUDED.concurrency,
                    {model_set}
                    known_models  = EXCLUDED.known_models,
                    llm_servers_available = EXCLUDED.llm_servers_available,
                    vram_total_mb = EXCLUDED.vram_total_mb,
                    fits_models   = EXCLUDED.fits_models,
                    last_seen     = EXCLUDED.last_seen,
                    last_flagged_dead_at = NULL
            """,
            (host_label, queue, _project(project), int(concurrency), current_model,
             known, llm_servers, vram, fits),
        )


def clear_worker_current_model(
    host_label: str, queue: str, *, mark_stale: bool = True,
    project: str | None = None,
) -> dict[str, Any] | None:
    """Clear a worker's ``current_model`` (its GPU busy signal) and, by default,
    age its ``last_seen`` out of the live window — the "don't leave a busy ghost"
    fix for a HARD-exiting worker.

    A watchdog trip exits via ``os._exit``, which SKIPS the ``_run_node``
    ``finally`` (so ``ModelCache.mark_idle`` and the heartbeat refresh never run).
    The worker's last-written ``worker_heartbeats`` row therefore keeps advertising
    a ``current_model`` even though the process is gone — inflating Rails' "N/M GPU
    busy" gauge (the user's observed "3/2 GPU busy" after a kill). Before
    hard-exiting, the trip path calls this to null out ``current_model`` and, when
    ``mark_stale`` is set, push ``last_seen`` ~10× the heartbeat cadence into the
    past so the gauge — which counts only rows fresh within 30 s — drops the dead
    worker immediately rather than waiting up to 30 s for the heartbeat to age out.
    (A replacement worker's fresh heartbeat re-establishes the row normally.)

    Best-effort + idempotent: scoped by the ``(host_label, queue)`` primary key,
    a no-op (returns ``None``) when the row is absent. The caller swallows any
    error — the hard-exit must happen regardless. Returns the updated row, or
    ``None`` on no-match."""
    # 100 s ≈ 10× the 10 s heartbeat cadence, comfortably past the 30 s gauge
    # window so the dead worker drops out at once.
    stale_secs = 100 if mark_stale else 0
    with connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            UPDATE worker_heartbeats
            SET current_model = NULL,
                last_seen = CASE WHEN %s > 0
                                 THEN now() - make_interval(secs => %s)
                                 ELSE last_seen END
            WHERE host_label = %s AND queue = %s AND project = %s
            RETURNING *
            """,
            (stale_secs, stale_secs, host_label, queue, _project(project)),
        )
        return cur.fetchone()


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
            vm_rss_mb_peak = %s,
            -- Stamp the executing machine on the terminal row so per-host error /
            -- log queries work off workflow_node_jobs too (claimed_by is the real
            -- host identity; host_label was left NULL in practice). COALESCE keeps
            -- any value a worker already set.
            host_label = COALESCE(host_label, claimed_by)
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
            seconds = %s,
            -- Stamp the failing machine on the terminal row (claimed_by = the real
            -- host) so "which machine failed?" is answerable from workflow_node_jobs,
            -- not just the events table. COALESCE keeps an already-set value.
            host_label = COALESCE(host_label, claimed_by)
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


# ── Node-event history (per-node, per-attempt) ─────────────────────────────
#
# Append-only log (migration 0011). ``record_node_event_in_txn`` rides the SAME
# transaction as the terminal / requeue state change (outbox-atomicity, exactly
# like ``enqueue_dispatch_event_in_txn``). ``record_node_event`` opens its OWN
# connection and is BEST-EFFORT — it swallows every error so an event-write blip
# can never fail the load-bearing claim / terminal / watchdog path.

#: The migration-0011 CHECK set, mirrored here so a bad type fails loudly in
#: Python (before the surrounding txn aborts on the DB CHECK).
NODE_EVENT_TYPES = frozenset({
    "claimed", "model_load_start", "model_load_done", "progress_beat",
    "stall_suspected", "stall_trip", "gpu_health_trip", "budget_trip",
    "requeued", "reassigned", "lease_renew", "completed", "failed",
    "cancelled", "error",
})


def record_node_event_in_txn(
    cur,
    *,
    run_id: str,
    node_id: str,
    event_type: str,
    job_id: str | None = None,
    attempt: int = 0,
    host_label: str | None = None,
    queue: str | None = None,
    model: str | None = None,
    elapsed_s: float | None = None,
    error: str | None = None,
    detail: dict[str, Any] | None = None,
) -> int:
    """Append one node event in the caller's transaction (outbox-atomicity).

    Used by the terminal (completed/failed/cancelled) + requeue paths so the
    event lands atomically with the state change — mirroring
    :func:`enqueue_dispatch_event_in_txn`. ``event_type`` must be in
    :data:`NODE_EVENT_TYPES` (validated here so a bad type raises in Python
    rather than aborting the surrounding txn on the DB CHECK). ``attempt`` is
    the node-job's ``watchdog_retries`` at emit time — the cross-attempt key."""
    if event_type not in NODE_EVENT_TYPES:
        raise ValueError(f"unknown node event_type: {event_type!r}")
    cur.execute(
        """
        INSERT INTO workflow_node_events
            (run_id, node_id, job_id, attempt, event_type,
             host_label, queue, model, elapsed_s, error, detail)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        RETURNING id
        """,
        (
            run_id, node_id, job_id, int(attempt or 0), event_type,
            host_label, queue, model, elapsed_s,
            (error[:8000] if error else error),
            _as_json(detail or {}),
        ),
    )
    return int(cur.fetchone()["id"])


def record_node_event(
    *,
    run_id: str,
    node_id: str,
    event_type: str,
    job_id: str | None = None,
    attempt: int = 0,
    host_label: str | None = None,
    queue: str | None = None,
    model: str | None = None,
    elapsed_s: float | None = None,
    error: str | None = None,
    detail: dict[str, Any] | None = None,
) -> int | None:
    """Best-effort append of one node event on its OWN connection.

    For the non-terminal emit sites (claim, model-load, watchdog
    suspected/trip, reassign, lease-renew) that must NOT widen or fail the
    load-bearing path. Swallows EVERY exception (logs once) and returns ``None``
    on failure — an event-history blip can never take down a node run. The
    terminal / requeue sites use :func:`record_node_event_in_txn` instead
    (atomic with the state change)."""
    try:
        with connection() as conn, conn.cursor() as cur:
            return record_node_event_in_txn(
                cur, run_id=run_id, node_id=node_id, event_type=event_type,
                job_id=job_id, attempt=attempt, host_label=host_label,
                queue=queue, model=model, elapsed_s=elapsed_s, error=error,
                detail=detail,
            )
    except Exception:  # noqa: BLE001 — best-effort; never propagate
        log.exception(
            "[node-event] failed to record %s for run=%s node=%s (ignored)",
            event_type, run_id, node_id,
        )
        return None


def prune_node_events(older_than_days: int = 30) -> int:
    """Delete node events older than ``older_than_days`` (append-only growth
    control; ``ON DELETE CASCADE`` already covers run-delete / purge). Called
    from an interval-gated NodePool sweep. Returns rows deleted."""
    days = max(int(older_than_days), 1)
    with connection() as conn, conn.cursor() as cur:
        cur.execute(
            "DELETE FROM workflow_node_events "
            "WHERE created_at < now() - make_interval(days => %s)",
            (days,),
        )
        return cur.rowcount or 0


def flag_unassignable_gpu_jobs(
    *, stale_s: int | None = None, project: str | None = None,
) -> list[dict[str, Any]]:
    """Capacity-aware fleet sweep (migration 0015): red-flag queued GPU model-jobs
    that **no live machine can hold**, and clear the flag when they become
    assignable again. Returns the rows NEWLY flagged this call (the NULL→now
    transition) so the caller emits one ``unassignable`` event each.

    A queued ``gpu`` job with a ``required_model`` is *unassignable* iff NO fresh
    GPU worker advertises that model in its ``fits_models`` (the worker-computed
    set of ids whose ``est_vram_gb`` fits its VRAM). The decision is pure SQL over
    ``worker_heartbeats`` — the worker pushed the fit computation into the
    heartbeat precisely so the orchestrator (which holds no model registry) can
    decide here.

    CRITICAL liveness guard: flagging requires at least ONE fresh GPU heartbeat.
    "No machine has enough VRAM" is a CAPACITY verdict, not a liveness one — if
    the whole GPU fleet is momentarily down (a deploy / all-workers bounce) we
    must NOT red-flag every job as unassignable (that's the dead-worker sweep's
    concern). With zero fresh GPU workers the sweep is a no-op.

    Idempotent: the flag UPDATE is guarded ``unassignable_at IS NULL`` so a job
    already flagged is not re-returned (no duplicate event) until it clears and
    re-flags. The clear path un-flags rows that became fittable OR left ``queued``
    (claimed / cancelled), so a stale red flag never lingers.

    ``stale_s`` overrides the freshness window (default ``_stale_worker_after_s``,
    30 s = 3× the 10 s heartbeat).
    """
    window = int(stale_s) if stale_s is not None else _stale_worker_after_s()
    proj = _project(project)
    with connection() as conn, conn.cursor() as cur:
        # CLEAR first: un-flag any flagged job that is now fittable, or that has
        # left the queued state (claimed / cancelled / terminal) — so a red flag
        # never outlives the condition that set it. Project-scoped (migration
        # 0017): a per-project orchestrator only judges its OWN project's jobs
        # against its OWN project's fleet — another project's worker fitting the
        # model must NOT mask this project's stuck job (exact-match claim means
        # that worker can never take it).
        cur.execute(
            """
            WITH live_gpu AS (
                SELECT fits_models FROM worker_heartbeats
                WHERE queue = 'gpu'
                  AND project = %(project)s
                  AND last_seen > now() - make_interval(secs => %(window)s)
            )
            UPDATE workflow_node_jobs j
            SET unassignable_at = NULL, unassignable_reason = NULL
            WHERE j.unassignable_at IS NOT NULL
              AND j.project = %(project)s
              AND (
                    j.status <> 'queued'
                 OR EXISTS (
                        SELECT 1 FROM live_gpu lg
                        WHERE j.required_model = ANY(lg.fits_models)
                    )
              )
            """,
            {"window": window, "project": proj},
        )
        # FLAG: queued gpu model-jobs that no fresh GPU worker can hold. Guarded
        # on at least one fresh GPU heartbeat existing (liveness vs capacity).
        cur.execute(
            """
            WITH live_gpu AS (
                SELECT fits_models, vram_total_mb FROM worker_heartbeats
                WHERE queue = 'gpu'
                  AND project = %(project)s
                  AND last_seen > now() - make_interval(secs => %(window)s)
            ),
            fleet AS (
                SELECT count(*) AS n, max(vram_total_mb) AS max_vram
                FROM live_gpu
            )
            UPDATE workflow_node_jobs j
            SET unassignable_at = now(),
                unassignable_reason =
                    'no live GPU machine can hold model ' || j.required_model
                    || ' — none of ' || (SELECT n FROM fleet)::text
                    || ' live GPU machine(s) has enough VRAM (max '
                    || COALESCE((SELECT max_vram FROM fleet)::text, 'unknown')
                    || 'MB)'
            WHERE j.queue = 'gpu'
              AND j.project = %(project)s
              AND j.status = 'queued'
              AND j.required_model IS NOT NULL
              AND j.unassignable_at IS NULL
              AND (SELECT n FROM fleet) > 0
              AND NOT EXISTS (
                    SELECT 1 FROM live_gpu lg
                    WHERE j.required_model = ANY(lg.fits_models)
              )
            RETURNING j.id, j.run_id, j.node_id, j.required_model,
                      j.unassignable_reason
            """,
            {"window": window, "project": proj},
        )
        return list(cur.fetchall())


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


def delete_non_terminal_jobs_for_run(run_id: str) -> list[str]:
    """Restart primitive: delete every job for ``run_id`` whose status is NOT
    ``completed`` / ``skipped``. Returns the list of deleted ``node_id``s so
    the host can cascade into its own artefacts (on-disk node dirs, input
    submissions). Idempotent — a second call returns ``[]``.

    The engine's ``dispatcher._find_ready_nodes`` treats surviving
    ``completed`` / ``skipped`` rows as cursors when re-expanding the DAG: only
    nodes WITHOUT a row whose deps are completed/skipped get enqueued. So a
    follow-up ``dispatcher.start_run`` after this call resumes the run from
    exactly the deleted set (plus any downstream that never got enqueued
    because of the original failure) — the host's "retry whole run" button no
    longer re-does the completed prefix.

    CALLER POLICY: this primitive INCLUDES ``running`` rows in the deletion
    set. If the caller's retry contract needs to refuse while a worker is
    mid-flight (the typical case — the cancel-watcher polls at 5 s, so a
    just-terminated run can briefly still have ``running`` children winding
    down), it must check ``WHERE status='running'`` itself BEFORE calling this.
    The engine's lease-reclaim sweep eventually re-queues a stranded
    ``running`` row, so the worst case here is a single straggler producing
    output to a path the caller then `rm -rf`'s — recoverable, but worth a
    409 from the caller.
    """
    with connection() as conn, conn.cursor() as cur:
        cur.execute(
            "DELETE FROM workflow_node_jobs "
            "WHERE run_id = %s "
            "  AND status NOT IN ('completed', 'skipped') "
            "RETURNING node_id",
            (run_id,),
        )
        return [r["node_id"] for r in cur.fetchall()]


def cancel_orphaned_queued_jobs(*, project: str | None = None) -> int:
    """Flip every ``queued`` job whose parent run is already terminal
    (``cancelled`` / ``failed``) to ``cancelled``. Returns the number of rows
    touched.

    The host's cancel handler typically only updates ``workflow_runs.status``;
    it does NOT cascade into ``workflow_node_jobs``. The claim SQL's run-cancel
    guard prevents such jobs from ever running, but they linger in ``queued``
    forever — operator-facing queue gauges then misleadingly suggest a worker
    stall. This sweep is the cleanup. Only ``status='queued'`` rows are touched
    so we don't race the cancel-watcher's cooperative ``running`` cancel.

    Project-scoped (migration 0017): only this orchestrator's project's jobs
    (``None`` ⇒ ``config.project``). The flip is correctness-neutral (the parent
    run is already terminal, so the job could never run), but a per-project
    orchestrator must not write another tenant's rows — kept consistent with the
    other sweeps. Default ``""`` matches all (single-tenant byte-compatible).

    Idempotent: a second call after the first returns 0.
    """
    with connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            UPDATE workflow_node_jobs j
               SET status = 'cancelled', finished_at = now()
              FROM workflow_runs r
             WHERE j.run_id = r.id
               AND j.project = %s
               AND j.status = 'queued'
               AND r.status IN ('cancelled', 'failed')
            """,
            (_project(project),),
        )
        return cur.rowcount or 0


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


def snapshot(*, project: str | None = None) -> dict[str, Any]:
    """Return counts and running/queued rows per queue, for the
    queue-indicator UI. Keeps the payload small — at most 50 rows per
    section.

    ``project`` (migration 0017) filters to one tenant; ``None`` (default) is
    the broker-wide view across all projects — byte-compatible with the pre-0017
    single-tenant deploy (every row is ``''``). Each returned row carries its
    ``project`` (``SELECT *``) so a multi-tenant caller can group client-side."""
    pred = "" if project is None else " WHERE project = %(project)s"
    pf = {} if project is None else {"project": project}
    with connection() as conn, conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT queue, status, COUNT(*) AS n
            FROM workflow_node_jobs{pred}
            GROUP BY queue, status
            """,
            pf,
        )
        counts: dict[tuple[str, str], int] = {}
        for row in cur.fetchall():
            counts[(row["queue"], row["status"])] = int(row["n"])

        def _top(queue: str, status: str, limit: int = 50) -> list[dict[str, Any]]:
            cur.execute(
                "SELECT * FROM workflow_node_jobs "
                "WHERE queue = %(queue)s AND status = %(status)s"
                + (" AND project = %(project)s" if project is not None else "")
                + " ORDER BY created_at ASC LIMIT %(limit)s",
                {"queue": queue, "status": status, "limit": limit, **pf},
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


def ingest_snapshot(*, project: str | None = None) -> dict[str, Any]:
    """Per-queue depth + live-worker counts for the INGEST path — the
    ``ingest_jobs`` twin of :func:`snapshot` (which covers only the cpu/gpu DAG
    queues). ``queues[q]`` carries the status counts plus ``workers`` = the
    number of ``worker_heartbeats`` rows on that queue still fresh (< 30 s,
    matching the claim worker's 10 s refresh). A host maps queued+running →
    "messages" and ``workers`` → "consumers" for its queue-indicator UI.

    NB: ``worker_heartbeats`` is keyed ``(host_label, queue, project)``, so
    ``workers`` counts live worker *hosts* per queue, not processes — enough to
    drive the "no consumer → starvation" warning.

    ``project`` (migration 0017) filters to one tenant; ``None`` (default) is the
    broker-wide view (byte-compatible with single-tenant)."""
    jpred = "" if project is None else " WHERE project = %(project)s"
    hpred = "" if project is None else " AND project = %(project)s"
    pf = {} if project is None else {"project": project}
    with connection() as conn, conn.cursor() as cur:
        cur.execute(
            f"SELECT queue, status, COUNT(*) AS n FROM ingest_jobs{jpred} "
            "GROUP BY queue, status",
            pf,
        )
        counts: dict[str, dict[str, int]] = {}
        for row in cur.fetchall():
            counts.setdefault(row["queue"], {})[row["status"]] = int(row["n"])
        cur.execute(
            "SELECT queue, COUNT(*) AS n FROM worker_heartbeats "
            "WHERE last_seen > now() - interval '30 seconds'" + hpred
            + " GROUP BY queue",
            pf,
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


def fleet_snapshot(
    *, stale_after_s: float = 30.0, project: str | None = None,
) -> list[dict[str, Any]]:
    """Read-only per-``(host_label, queue)`` fleet capacity view — the observed
    ``worker_heartbeats`` rows, the telemetry read model a fleet view / operator
    control plane consumes.

    Unlike :func:`snapshot` / :func:`ingest_snapshot` (which only *count*
    heartbeats or join them internally for claim decisions), this returns the
    per-worker rows with their advertised capability (``current_model``,
    ``known_models``, ``llm_servers_available`` [0014], ``vram_total_mb`` /
    ``fits_models`` [0015]). It deliberately surfaces **stale and dead-flagged**
    workers too — that's the point of an observability read — so it returns ALL
    rows ordered by ``(queue, host_label)``, each augmented with two derived
    flags rather than filtering:

      * ``fresh``        — ``last_seen`` within ``stale_after_s`` (default 30 s,
                           matching the claim worker's 10 s refresh × 3);
      * ``flagged_dead`` — the orchestrator's stale-worker detector stamped
                           ``last_flagged_dead_at`` (migration 0009) and no fresh
                           heartbeat has cleared it.

    ``project`` (migration 0017) filters to one tenant's workers; ``None``
    (default) returns the whole fleet across all projects (each row carries its
    ``project`` via ``SELECT *``).

    Pure read; no host coupling. Returns ``[]`` on an empty fleet.
    """
    pred = "" if project is None else " WHERE project = %(project)s"
    with connection() as conn, conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT *,
                   (last_seen > now() - (%(stale)s * interval '1 second')) AS fresh,
                   (last_flagged_dead_at IS NOT NULL)               AS flagged_dead
            FROM worker_heartbeats{pred}
            ORDER BY queue, host_label
            """,
            {"stale": float(stale_after_s),
             **({} if project is None else {"project": project})},
        )
        return [dict(row) for row in cur.fetchall()]

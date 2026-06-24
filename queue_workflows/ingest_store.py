"""Backend-agnostic ingest-job store — the ``db_backend`` seam for the *flat*
ingest-family queue (``ingest_jobs``).

WHY this module exists
----------------------
The engine's ingest path (scheduler → claim worker → ``ingest_executor``) talks
to Postgres' ``ingest_jobs`` table directly via :mod:`queue_workflows.node_queue`.
The pluggable :class:`~queue_workflows.backends.base.StorageBackend` SPI is a
generic durable queue, but it keeps its OWN tables (``qw_jobs`` …), so routing
the ingest path straight through ``get_backend()`` would, for ``db_backend="pg"``,
silently move ingest work off ``ingest_jobs`` and break every existing pg deploy
(the dashboard's ``ingest_snapshot``, a host's caller-txn enqueue, …).

So this facade switches on ``config.db_backend``:

* ``pg`` (default) → delegates to the existing ``node_queue.*ingest*`` functions
  against ``ingest_jobs`` — **byte-identical** to today.
* ``redis`` / ``mongodb`` → maps the ingest job onto the StorageBackend SPI
  (``payload = {task_name, reason, args}``), so the ingest family can run on a
  non-PG backend with no ``ingest_jobs`` table at all.

The ingest job ⇄ SPI ``Job`` mapping
-------------------------------------
* ``payload`` carries the ingest-specific fields ``{task_name, reason, args}``;
  every claim is re-inflated to the flat ingest-row dict shape the rest of the
  engine (``ingest_executor.execute_ingest_job``) already expects.
* **Priority direction is inverted.** Ingest orders ``priority ASC`` (lower =
  sooner); the SPI claims ``priority DESC`` (higher = sooner). The seam stores
  ``spi_priority = -ingest_priority`` so "lower ingest number runs first" holds
  on either backend, and un-negates it on read.

This module is ADDITIVE: it imports the engine's pg path rather than replacing
it, and nothing in the engine calls it yet — wiring the live scheduler / claim
worker / reclaim sweep through it is the next, separately-audited slice.
"""

from __future__ import annotations

import uuid
from typing import Any

from queue_workflows.config import get_config
from queue_workflows.node_queue import DEFAULT_LEASE_S

# ── backend selection ───────────────────────────────────────────────────────


def _use_spi() -> bool:
    """True when ``db_backend`` is a non-pg StorageBackend (redis/mongodb)."""
    return get_config().db_backend != "pg"


def _backend():
    from queue_workflows.backends import get_backend

    return get_backend()


def _ingest_queues() -> frozenset[str]:
    return frozenset(get_config().ingest_queues)


def _ingest_tasks() -> dict[str, Any]:
    return get_config().ingest_task_map


def _validate(task_name: str, queue: str) -> None:
    """Fail BEFORE any write on an unknown queue / unregistered task — mirrors
    :func:`node_queue.enqueue_ingest_job`'s fail-before-write contract so the
    guard is identical on every backend."""
    iq = _ingest_queues()
    if queue not in iq:
        raise ValueError(f"ingest queue must be in {sorted(iq)}, got {queue!r}")
    tasks = _ingest_tasks()
    if task_name not in tasks:
        raise ValueError(
            f"task_name must be a registered ingest task {sorted(tasks)}, got "
            f"{task_name!r} (register via queue_workflows.register_ingest_task)"
        )


# ── SPI Job ⇄ ingest-row mapping ────────────────────────────────────────────


def _job_to_ingest_row(job: dict[str, Any] | None) -> dict[str, Any] | None:
    """Re-inflate a generic SPI :class:`Job` into the flat ingest-row dict shape
    the engine already passes around (``execute_ingest_job`` reads ``id`` /
    ``task_name`` / ``reason`` / ``args``)."""
    if job is None:
        return None
    payload = job.get("payload") or {}
    return {
        "id": job["id"],
        "task_name": payload.get("task_name"),
        "queue": job.get("queue"),
        "reason": payload.get("reason", "tick"),
        "args": payload.get("args") or {},
        "status": job.get("status"),
        "claimed_by": job.get("claimed_by"),
        "priority": (
            -int(job["priority"]) if job.get("priority") is not None else None
        ),
        "result": job.get("result"),
        "error": job.get("error"),
        "lease_expires_at": job.get("lease_expires_at"),
        "created_at": job.get("created_at"),
    }


# ── public API (mirrors node_queue's ingest surface) ────────────────────────


def enqueue_ingest_job(
    *, task_name: str, queue: str, reason: str = "tick", priority: int = 100,
    args: dict[str, Any] | None = None, conn: Any = None,
    project: str | None = None,
) -> str:
    """Insert a fresh ``queued`` ingest job; return its id.

    ``conn`` (a host psycopg connection for a caller-controlled transaction) is
    only meaningful for the pg path and is ignored by the SPI backends, which
    have no equivalent cross-row transaction with a host's domain table.

    ``project`` (migration 0017) is the tenant tag on the pg path (``None`` ⇒
    ``config.project``); ignored on the SPI path, whose tenancy is ``db_namespace``
    (see docs/multitenant_broker.md)."""
    if not _use_spi():
        from queue_workflows import node_queue

        return node_queue.enqueue_ingest_job(
            task_name=task_name, queue=queue, reason=reason,
            priority=priority, args=args, conn=conn, project=project,
        )
    _validate(task_name, queue)
    job_id = str(uuid.uuid4())
    return _backend().enqueue(
        queue,
        {"task_name": task_name, "reason": reason, "args": args or {}},
        job_id=job_id,
        priority=-int(priority),  # ingest ASC ⇄ SPI DESC
    )


def get_ingest_job(job_id: str) -> dict[str, Any] | None:
    if not _use_spi():
        from queue_workflows import node_queue

        return node_queue.get_ingest_job(job_id)
    return _job_to_ingest_row(_backend().get(job_id))


def claim_next_ingest_job(
    queue: str, *, host: str | None = None, lease_s: int = DEFAULT_LEASE_S,
) -> dict[str, Any] | None:
    """Atomically claim the next queued ingest job on ``queue`` (exactly-once
    under contention), or ``None``."""
    if not _use_spi():
        from queue_workflows import node_queue

        return node_queue.claim_next_ingest_job(queue, host=host, lease_s=lease_s)
    job = _backend().claim(queue, host or "", lease_s=float(lease_s))
    return _job_to_ingest_row(job)


def renew_ingest_lease(
    job_id: str, host: str, *, lease_s: int = DEFAULT_LEASE_S,
) -> bool:
    """Extend the lease iff the row is still ``running`` and owned by ``host``.
    Returns whether it renewed (the LeaseRenewer's heartbeat)."""
    if not _use_spi():
        from queue_workflows.db import connection

        with connection() as conn, conn.cursor() as cur:
            cur.execute(
                "UPDATE ingest_jobs "
                "SET lease_expires_at = now() + make_interval(secs => %s) "
                "WHERE id = %s AND status = 'running' AND claimed_by = %s "
                "RETURNING id",
                (int(lease_s), job_id, host),
            )
            return cur.fetchone() is not None
    return _backend().renew_lease(job_id, host, lease_s=float(lease_s))


def mark_ingest_completed(
    job_id: str, *, result: dict[str, Any] | None = None,
    seconds: float | None = None,
) -> dict[str, Any] | None:
    """Transition to ``completed`` (idempotent — ``None`` if already terminal)."""
    if not _use_spi():
        from queue_workflows import node_queue

        return node_queue.mark_ingest_completed(
            job_id, result=result, seconds=seconds,
        )
    return _job_to_ingest_row(
        _backend().mark_completed(job_id, result=result or {})
    )


def mark_ingest_failed(
    job_id: str, *, error: str, seconds: float | None = None,
) -> dict[str, Any] | None:
    """Transition to ``failed`` (idempotent — ``None`` if already terminal)."""
    if not _use_spi():
        from queue_workflows import node_queue

        return node_queue.mark_ingest_failed(job_id, error=error, seconds=seconds)
    return _job_to_ingest_row(_backend().mark_failed(job_id, error=error))


def reclaim_expired_ingest_leases() -> list[dict[str, Any]]:
    """Re-queue ``running`` ingest jobs whose lease lapsed — the sole recovery
    path for an orphaned ingest row. Returns ``{id, task_name, queue}`` per row."""
    if not _use_spi():
        from queue_workflows import node_queue

        return node_queue.reclaim_expired_ingest_leases()
    be = _backend()
    reclaimed = []
    for jid in be.reclaim_expired():
        row = _job_to_ingest_row(be.get(jid))
        if row is not None:
            reclaimed.append(
                {"id": row["id"], "task_name": row["task_name"], "queue": row["queue"]}
            )
    return reclaimed


def ingest_snapshot() -> dict[str, Any]:
    """Per-queue depth + live-worker counts (the dashboard's queue indicator)."""
    if not _use_spi():
        from queue_workflows import node_queue

        return node_queue.ingest_snapshot()
    be = _backend()
    queues: dict[str, dict[str, int]] = {}
    for q in sorted(_ingest_queues()):
        c = be.counts(q)
        queues[q] = {
            "queued": int(c.get("queued", 0)),
            "running": int(c.get("running", 0)),
            "completed": int(c.get("completed", 0)),
            "failed": int(c.get("failed", 0)),
            "workers": len(be.workers(q)),
        }
    return {"queues": queues}


__all__ = [
    "enqueue_ingest_job",
    "get_ingest_job",
    "claim_next_ingest_job",
    "renew_ingest_lease",
    "mark_ingest_completed",
    "mark_ingest_failed",
    "reclaim_expired_ingest_leases",
    "ingest_snapshot",
]

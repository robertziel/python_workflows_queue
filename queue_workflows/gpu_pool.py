"""Shared GPU pool (pivot B) — a namespace-scoped durable queue of *self-contained*
GPU tasks that pooled GPU workers across apps claim and execute, while each app
keeps its own database for run/DAG state.

WHY this exists
---------------
The feasibility audit (worklog/redis-broker-rearchitecture.md) rejected "one Redis
as the sole bus for everything" but endorsed a **shared GPU fleet**: pool the GPU
boxes so any box can serve GPU work from any app, without collapsing the per-app
Postgres-as-bus design. A normal DAG node-job can't be run by a pooled worker — it
is bound to its app's DB (run row, node_jobs, dispatch outbox, node_events,
run_files) and a filesystem out_dir (see node_executor). So the pool trades in a
different unit: a **PoolTask** that carries everything needed to run
(``{model, handler, inputs, output_dir, params}``), where inputs/output_dir are
references into **shared NFS** (the same storage the apps already use). Pooled
workers touch ONLY the shared pool store + NFS — never an app's database.

Design (DDD)
------------
- **Store:** the shared pool is a :class:`StorageBackend` (redis) addressed
  INDEPENDENTLY of ``config.db_backend`` — an app keeps ``db_backend="pg"`` for its
  own DAG while the pool lives on a separate redis store
  (``gpu_pool_backend`` / ``gpu_pool_url_env`` / ``gpu_pool_namespace``). Every app
  + GPU box sharing a fleet uses the same namespace.
- **Capability routing by QUEUE NAME** (no SPI change): a task is submitted to a
  queue naming the capability it needs (a model id, or a box-class channel like
  ``gpu:box-a`` / ``gpu:box-b``). A pooled worker serves an ORDERED set of queues
  (its warm-model queue first ⇒ affinity; its box-class queues ⇒ box-b/box-a
  separation), reusing per-queue claim verbatim.
- **Submitter (app):** :func:`submit_pool_task` then :func:`await_pool_result` —
  the app's GPU step submits, blocks on the result, then writes outputs into its
  OWN db/out_dir exactly as before.
- **Worker (GPU box):** ``register_pool_handler(name, fn)`` deploys the op CODE on
  the box; :func:`run_pool_worker_once` claims a capable task, resolves the handler,
  runs ``fn(*, inputs, output_dir, params) -> dict``, and writes the result back
  atomically (idempotent terminal).
"""

from __future__ import annotations

import os
import threading
import time
from typing import Any

from queue_workflows.config import get_config


class PoolTaskFailed(RuntimeError):
    """Raised by :func:`await_pool_result` when the awaited task ended ``failed``
    (or is unknown). ``task_id`` / ``error`` carry the detail."""

    def __init__(self, task_id: str, error: str | None) -> None:
        super().__init__(f"pool task {task_id} failed: {error}")
        self.task_id = task_id
        self.error = error


# ── pool backend (separate from the app's db_backend) ───────────────────────

_POOL: dict[str, Any] = {"key": None, "be": None}
_LOCK = threading.RLock()


def _pool_backend():
    """The shared-pool :class:`StorageBackend`, built + cached independently of
    ``config.db_backend``. Rebuilds when the configured backend/url/namespace
    changes (so a test pointing at a fresh namespace gets a fresh handle)."""
    cfg = get_config()
    url = os.environ.get(cfg.gpu_pool_url_env)
    if not url:
        raise RuntimeError(
            f"shared GPU pool requires {cfg.gpu_pool_url_env} to hold the pool DSN "
            f"(e.g. a redis URL); set it on every app + GPU box sharing the fleet"
        )
    key = (cfg.gpu_pool_backend, url, cfg.gpu_pool_namespace or "")
    with _LOCK:
        if _POOL["key"] != key:
            _close_locked()
            from queue_workflows import backends

            be = backends.build_backend(
                cfg.gpu_pool_backend, url=url, namespace=cfg.gpu_pool_namespace or "",
            )
            be.ensure_schema()
            _POOL["key"] = key
            _POOL["be"] = be
        return _POOL["be"]


def _close_locked() -> None:
    be = _POOL.get("be")
    if be is not None:
        try:
            be.close()
        except Exception:
            pass
    _POOL["key"] = None
    _POOL["be"] = None


def close_pool_backend() -> None:
    """Close + drop the cached pool backend (shutdown / test teardown)."""
    with _LOCK:
        _close_locked()


# ── submitter side (the app's GPU step) ─────────────────────────────────────


def submit_pool_task(
    *, queue: str, handler: str, model: str | None = None,
    inputs: dict[str, Any] | None = None, output_dir: str | None = None,
    params: dict[str, Any] | None = None, priority: int = 0,
) -> str:
    """Enqueue a self-contained GPU task onto the shared pool; return its id.

    ``queue`` is the capability channel (a model id or box-class like
    ``gpu:box-a``); ``handler`` names the registered op a pooled worker will run;
    ``inputs`` / ``output_dir`` reference shared NFS. The submitter need NOT have
    the handler registered locally — only the worker resolves it."""
    payload = {
        "handler": handler,
        "model": model,
        "inputs": inputs or {},
        "output_dir": output_dir,
        "params": params or {},
    }
    return _pool_backend().enqueue(queue, payload, priority=int(priority))


def get_pool_task(task_id: str) -> dict[str, Any] | None:
    """The task's current row (status / payload / result / error), or ``None``."""
    return _pool_backend().get(task_id)


def await_pool_result(
    task_id: str, *, timeout_s: float = 300.0, poll_s: float = 0.5,
) -> dict[str, Any]:
    """Block until the task is terminal: return its ``result`` dict on
    ``completed``; raise :class:`PoolTaskFailed` on ``failed`` (or unknown task);
    raise ``TimeoutError`` if it doesn't finish within ``timeout_s``."""
    be = _pool_backend()
    deadline = time.monotonic() + float(timeout_s)
    while True:
        job = be.get(task_id)
        if job is None:
            raise PoolTaskFailed(task_id, "task not found (lost / wrong namespace)")
        status = job.get("status")
        if status == "completed":
            return job.get("result") or {}
        if status == "failed":
            raise PoolTaskFailed(task_id, job.get("error"))
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"pool task {task_id} not terminal within {timeout_s:.0f}s "
                f"(status={status!r})"
            )
        time.sleep(float(poll_s))


# ── worker side (the GPU box) ───────────────────────────────────────────────


def claim_pool_task(
    *, queues: list[str], worker: str, lease_s: float = 300.0,
) -> dict[str, Any] | None:
    """Claim the next task from the first non-empty queue in ``queues`` order
    (capability/affinity = the order), or ``None`` if all are empty."""
    be = _pool_backend()
    for q in queues:
        job = be.claim(q, worker, lease_s=float(lease_s))
        if job is not None:
            return job
    return None


def renew_pool_lease(task_id: str, worker: str, *, lease_s: float = 300.0) -> bool:
    """Extend the lease while a handler runs (the renewer heartbeat)."""
    return _pool_backend().renew_lease(task_id, worker, lease_s=float(lease_s))


def execute_pool_task(job: dict[str, Any]) -> str:
    """Run one claimed PoolTask to a terminal state via its registered handler.
    Returns ``"completed"`` / ``"failed"`` / ``"skipped"`` (already terminal).

    The result + terminal flip ride the backend's atomic outbox; an unknown
    handler or a raising handler marks the task ``failed`` so the submitter's
    :func:`await_pool_result` surfaces it (never a silent vanish)."""
    be = _pool_backend()
    job_id = job["id"]
    payload = job.get("payload") or {}
    handler = payload.get("handler")
    fn = get_config().gpu_pool_handlers.get(handler)
    if fn is None:
        row = be.fail_with_event(
            job_id, "failed",
            error=(
                f"unknown pool handler {handler!r} on this worker "
                f"(registered: {sorted(get_config().gpu_pool_handlers)})"
            ),
        )
        return "failed" if row is not None else "skipped"
    try:
        result = fn(
            inputs=payload.get("inputs") or {},
            output_dir=payload.get("output_dir"),
            params=payload.get("params") or {},
        )
        if not isinstance(result, dict):
            result = {"result": result}
    except Exception as exc:
        row = be.fail_with_event(job_id, "failed", error=f"{type(exc).__name__}: {exc}")
        return "failed" if row is not None else "skipped"
    row = be.complete_with_event(job_id, "completed", result=result)
    return "completed" if row is not None else "skipped"


def run_pool_worker_once(
    *, queues: list[str], worker: str, lease_s: float = 300.0,
) -> str | None:
    """Claim one capable task and run it to terminal. Returns the terminal status,
    or ``None`` when every served queue was empty (caller blocks on the wake)."""
    job = claim_pool_task(queues=queues, worker=worker, lease_s=lease_s)
    if job is None:
        return None
    return execute_pool_task(job)


def reclaim_expired_pool_leases() -> list[str]:
    """Re-queue pool tasks whose lease lapsed (a dead/wedged GPU box) — the sole
    recovery path for an orphaned ``running`` pool task. Returns reclaimed ids."""
    return _pool_backend().reclaim_expired()


__all__ = [
    "PoolTaskFailed",
    "submit_pool_task",
    "get_pool_task",
    "await_pool_result",
    "claim_pool_task",
    "renew_pool_lease",
    "execute_pool_task",
    "run_pool_worker_once",
    "reclaim_expired_pool_leases",
    "close_pool_backend",
]

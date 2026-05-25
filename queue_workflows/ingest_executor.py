"""Execute one claimed ``ingest_jobs`` row.

The Source-Ingestion twin of ``node_executor.execute_node``. A claimed ingest
row names a periodic callable rather than a DAG node, so the body is far
smaller: no ``$from`` input resolution, no model cache, no dispatch-event
outbox, no thumbnail. It maps ``task_name`` → the host-registered callable,
runs it, and writes the terminal row.

GENERIC dispatch (plan §1f): the engine does NOT know the task-name→callable
map. It looks up ``task_name`` in the host-registered
``config.ingest_task_map`` (populated via
``queue_workflows.register_ingest_task(name, callable)``). Each callable takes
the ``reason`` string and returns a JSON-able result dict.
"""

from __future__ import annotations

import logging
import time
import traceback

from queue_workflows import node_queue

log = logging.getLogger(__name__)


def _run_task(task_name: str, reason: str) -> dict:
    """Dispatch a periodic ingest callable by name → its result dict. Looks the
    name up in the host-registered ingest dispatch map."""
    from queue_workflows.config import get_config
    task_map = get_config().ingest_task_map
    fn = task_map.get(task_name)
    if fn is None:
        raise ValueError(
            f"unknown ingest task_name {task_name!r} "
            f"(registered: {sorted(task_map)}; register via "
            f"queue_workflows.register_ingest_task)"
        )
    result = fn(reason)
    return result if isinstance(result, dict) else {"result": result}


def execute_ingest_job(job: dict) -> str:
    """Run one already-claimed ``ingest_jobs`` row to a terminal state.
    Returns ``"completed"`` / ``"failed"`` / ``"skipped"``.

    Contract (mirrors ``execute_node``): the row MUST already be ``running``
    (the caller's claim is the queued→running transition). ``mark_ingest_*``
    returns ``None`` when the row is already terminal (a duplicate/raced claim)
    — in that case we return ``"skipped"``.
    """
    job_id = job["id"]
    t0 = time.time()
    try:
        result = _run_task(job["task_name"], job.get("reason") or "tick")
    except Exception as exc:
        err = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
        log.error("[ingest-executor] %s %s", job_id, err)
        row = node_queue.mark_ingest_failed(
            job_id, error=err, seconds=time.time() - t0,
        )
        return "failed" if row is not None else "skipped"

    row = node_queue.mark_ingest_completed(
        job_id, result=result, seconds=time.time() - t0,
    )
    if row is None:
        log.warning(
            "[ingest-executor] %s already terminal; skipping completion", job_id,
        )
        return "skipped"
    return "completed"


__all__ = ["execute_ingest_job"]

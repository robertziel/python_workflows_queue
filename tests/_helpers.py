"""Shared engine-test helpers (domain-free).

The ai_leads engine tests built a parent run via ``repo.upsert_parcel`` +
``repo.insert_run`` (a parcel FK + the SQLModel ORM). The engine suite must
stay domain-free, so we insert the run through the engine's own
:func:`queue_workflows.run_store.insert_run` — ``workflow_runs`` is parcel-
agnostic in the engine schema (no parcels table, ``parcel_id`` a plain nullable
column).
"""

from __future__ import annotations

import uuid

from queue_workflows import run_store
from queue_workflows.db import connection


def make_run(run_id: str | None = None, *, status: str = "running",
             workflow_name: str = "_test_wf", out_dir: str = "/tmp/out") -> str:
    """Insert a parent ``workflow_runs`` row in ``mode='node'``.

    Defaults to ``status='running'`` so the claim-path run-cancel guard (run NOT
    IN cancelled/failed) is satisfied — cancel-guard tests flip it explicitly.
    """
    run_id = run_id or str(uuid.uuid4())
    run_store.insert_run(
        run_id=run_id, workflow_name=workflow_name,
        out_dir=out_dir, status=status, mode="node",
    )
    return run_id


def set_run_status(run_id: str, status: str) -> None:
    with connection() as c, c.cursor() as cur:
        cur.execute(
            "UPDATE workflow_runs SET status=%s WHERE id=%s", (status, run_id),
        )


def row(job_id: str) -> dict:
    from queue_workflows import node_queue
    r = node_queue.get_node_job(job_id)
    assert r is not None
    return r


def force_lease(job_id: str, *, expires_in_s: float) -> None:
    """Mark a row ``running`` with a lease that expires ``expires_in_s`` from
    now (negative ⇒ already expired)."""
    with connection() as c, c.cursor() as cur:
        cur.execute(
            "UPDATE workflow_node_jobs "
            "SET status='running', started_at=now(), claimed_by='host-x', "
            "    lease_expires_at = now() + make_interval(secs => %s) "
            "WHERE id=%s",
            (float(expires_in_s), job_id),
        )

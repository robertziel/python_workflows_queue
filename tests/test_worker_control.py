"""Worker ON/OFF control — state accessors + the scoped requeue.

Covers (the operator-write side of the feature):
- set/get round-trip + upsert idempotency + the on/hard defaults;
- desired_state_for is default-ON (no row / table absent) and reads OFF only on an
  explicit OFF row;
- validation: a bad desired_state / an unregistered stop_policy fails BEFORE any
  write (fail-before-write), matching enqueue_ingest_job;
- get_worker_control never raises UndefinedTable on a pre-0012 DB (⇒ None ⇒ ON);
- the migration-0012 trigger fires the worker_control NOTIFY (payload host:queue);
- node_queue.requeue_running_for_worker: resume-style (no watchdog_retries bump),
  scoped to (host_label, queue) for BOTH the node and ingest tables.
"""

from __future__ import annotations

import contextlib
import uuid

import pytest

from queue_workflows import node_queue, worker_control
from queue_workflows.db import connection, db_url
from tests._helpers import make_run


# ── helpers ────────────────────────────────────────────────────────────────


def _running_node_job(host: str, *, queue: str = "gpu", model: str | None = None) -> str:
    """A ``running`` workflow_node_jobs row claimed_by ``host`` on ``queue``."""
    run_id = make_run(workflow_name="_wc_test")
    job_id = node_queue.enqueue_node_job(
        run_id=run_id, node_id="n", node_module="x", queue=queue,
        required_model=model,
    )
    with connection() as c, c.cursor() as cur:
        cur.execute(
            "UPDATE workflow_node_jobs "
            "SET status='running', started_at=now(), claimed_by=%s, "
            "    lease_expires_at = now() + interval '600 seconds' "
            "WHERE id=%s",
            (host, job_id),
        )
    return job_id


def _running_ingest_job(host: str, *, queue: str = "fetch") -> str:
    """A ``running`` ingest_jobs row claimed_by ``host`` on ``queue`` (inserted
    directly so the test needn't register a host ingest task)."""
    job_id = str(uuid.uuid4())
    with connection() as c, c.cursor() as cur:
        cur.execute(
            "INSERT INTO ingest_jobs "
            "(id, task_name, queue, status, started_at, claimed_by, lease_expires_at) "
            "VALUES (%s, 'run_x', %s, 'running', now(), %s, "
            "        now() + interval '600 seconds')",
            (job_id, queue, host),
        )
    return job_id


# ── state accessors ──────────────────────────────────────────────────────────


def test_set_and_get_round_trip():
    worker_control.set_worker_control(
        "host-c", "gpu", desired_state="off", stop_policy="hard",
        requested_by="op@example",
    )
    row = worker_control.get_worker_control("host-c", "gpu")
    assert row is not None
    assert row["host_label"] == "host-c"
    assert row["queue"] == "gpu"
    assert row["desired_state"] == "off"
    assert row["stop_policy"] == "hard"
    assert row["requested_by"] == "op@example"


def test_upsert_updates_in_place():
    worker_control.set_worker_control("h", "cpu", desired_state="off")
    worker_control.set_worker_control("h", "cpu", desired_state="on")
    row = worker_control.get_worker_control("h", "cpu")
    assert row["desired_state"] == "on"
    # one row, updated — not a duplicate.
    with connection() as c, c.cursor() as cur:
        cur.execute(
            "SELECT COUNT(*) AS n FROM worker_controls "
            "WHERE host_label='h' AND queue='cpu'"
        )
        assert cur.fetchone()["n"] == 1


def test_get_absent_returns_none():
    assert worker_control.get_worker_control("nobody", "gpu") is None


def test_desired_state_for_defaults_on_when_absent():
    assert worker_control.desired_state_for("nobody", "gpu") == "on"


def test_desired_state_for_reads_off_and_on():
    worker_control.disable_worker("host-a", "gpu")
    assert worker_control.desired_state_for("host-a", "gpu") == "off"
    worker_control.enable_worker("host-a", "gpu")
    assert worker_control.desired_state_for("host-a", "gpu") == "on"


def test_enable_disable_helpers():
    worker_control.disable_worker("host-b", "gpu", requested_by="me")
    r = worker_control.get_worker_control("host-b", "gpu")
    assert r["desired_state"] == "off" and r["stop_policy"] == "hard"
    worker_control.enable_worker("host-b", "gpu")
    assert worker_control.get_worker_control("host-b", "gpu")["desired_state"] == "on"


# ── validation (fail-before-write) ────────────────────────────────────────────


def test_invalid_desired_state_rejected():
    with pytest.raises(ValueError):
        worker_control.set_worker_control("h", "cpu", desired_state="paused")
    # nothing written
    assert worker_control.get_worker_control("h", "cpu") is None


def test_unregistered_stop_policy_rejected():
    with pytest.raises(ValueError):
        worker_control.set_worker_control(
            "h", "cpu", desired_state="off", stop_policy="drain",
        )
    assert worker_control.get_worker_control("h", "cpu") is None


def test_stop_policies_registry_has_hard_only_for_now():
    assert "hard" in worker_control.STOP_POLICIES
    # drain/pause are reserved names not yet implemented (the seam for later).
    assert "drain" not in worker_control.STOP_POLICIES
    assert "pause" not in worker_control.STOP_POLICIES


# ── backward-compat: pre-0012 DB has no table ─────────────────────────────────


def test_get_worker_control_table_absent_returns_none(monkeypatch):
    """A consumer DB that predates migration 0012 has no worker_controls table;
    get_worker_control must swallow UndefinedTable and return None (⇒ treated as
    ON), so the engine runs unchanged before the migration is applied."""
    import psycopg

    class _FakeCur:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def execute(self, *a, **k):
            raise psycopg.errors.UndefinedTable(
                'relation "worker_controls" does not exist'
            )

        def fetchone(self):
            return None

    class _FakeConn:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def cursor(self):
            return _FakeCur()

    @contextlib.contextmanager
    def _fake_connection():
        yield _FakeConn()

    monkeypatch.setattr(worker_control, "connection", _fake_connection)
    assert worker_control.get_worker_control("h", "gpu") is None
    assert worker_control.desired_state_for("h", "gpu") == "on"


# ── NOTIFY trigger ─────────────────────────────────────────────────────────────


def test_set_worker_control_fires_notify():
    """The migration-0012 trigger NOTIFYs ``worker_control`` with ``host:queue``
    on write, so a plain INSERT/UPDATE (Python or Rails) wakes the worker."""
    import psycopg

    with psycopg.connect(db_url(), autocommit=True) as conn:
        conn.execute("LISTEN worker_control")
        worker_control.set_worker_control("host-c", "gpu", desired_state="off")
        payloads = [n.payload for n in conn.notifies(timeout=3.0, stop_after=1)]
    assert payloads == ["host-c:gpu"]


# ── node_queue.requeue_running_for_worker (resume-style, scoped) ───────────────


def test_requeue_running_for_worker_node():
    job_id = _running_node_job("host-a", queue="gpu")
    n = node_queue.requeue_running_for_worker("host-a", "gpu")
    assert n == 1
    row = node_queue.get_node_job(job_id)
    assert row["status"] == "queued"
    assert row["claimed_by"] is None
    assert row["started_at"] is None
    assert row["lease_expires_at"] is None


def test_requeue_does_not_increment_watchdog_retries():
    """Resume-style redistribution, NOT a watchdog retry — turning a machine off
    must not burn the per-job retry cap."""
    job_id = _running_node_job("host-a", queue="gpu")
    node_queue.requeue_running_for_worker("host-a", "gpu")
    assert (node_queue.get_node_job(job_id).get("watchdog_retries") or 0) == 0


def test_requeue_scoped_to_host():
    mine = _running_node_job("host-a", queue="gpu")
    theirs = _running_node_job("host-c", queue="gpu")
    assert node_queue.requeue_running_for_worker("host-a", "gpu") == 1
    assert node_queue.get_node_job(mine)["status"] == "queued"
    assert node_queue.get_node_job(theirs)["status"] == "running"  # untouched


def test_requeue_scoped_to_queue():
    """host-c runs a cpu AND a gpu worker under one host_label: turning OFF the
    gpu worker must not release the cpu worker's in-flight job."""
    gpu_job = _running_node_job("host-c", queue="gpu")
    cpu_job = _running_node_job("host-c", queue="cpu")
    assert node_queue.requeue_running_for_worker("host-c", "gpu") == 1
    assert node_queue.get_node_job(gpu_job)["status"] == "queued"
    assert node_queue.get_node_job(cpu_job)["status"] == "running"  # untouched


def test_requeue_targets_ingest_table_for_ingest_queue():
    job_id = _running_ingest_job("host-c", queue="fetch")
    n = node_queue.requeue_running_for_worker("host-c", "fetch")
    assert n == 1
    assert node_queue.get_ingest_job(job_id)["status"] == "queued"
    assert node_queue.get_ingest_job(job_id)["claimed_by"] is None


def test_requeue_no_match_returns_zero():
    assert node_queue.requeue_running_for_worker("ghost", "gpu") == 0

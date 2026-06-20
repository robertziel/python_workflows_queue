"""Resume policy for orphaned ``running`` runs (``reenqueue_running_for_resume``).

Startup hook: a run left in ``running`` by a worker death (crash, watchdog
hard-exit, or an operator fleet-restart) must GO BACK TO THE QUEUE so it can
finish — it must NOT be auto-failed. Earlier the hook capped at 5 resumes then
marked the run ``failed`` ("[auto-resume cap reached]"); that conflated two
unrelated things — a poison-pill run that crashes every worker, and a healthy
run that simply rode through N fleet restarts or a host-specific hang (e.g. the
Blackwell qwen stall, which the worker watchdog fails on a box-c but which would
complete on host-c). The cap killed the healthy case. Policy now: always
re-queue; ``resume_count`` still climbs for observability, but never auto-fails.
Genuine node failures still fail the run via the normal node-failure path.
"""

from __future__ import annotations

import pytest

from queue_workflows import run_store
from queue_workflows.db import connection
from tests._helpers import make_run


def _set(run_id: str, **cols) -> None:
    sets = ", ".join(f"{k} = %s" for k in cols)
    with connection() as c, c.cursor() as cur:
        cur.execute(
            f"UPDATE workflow_runs SET {sets} WHERE id = %s",
            (*cols.values(), run_id),
        )


def test_reenqueue_requeues_orphan_running_run():
    run_id = make_run(status="running")  # resume_count defaults to 0
    n = run_store.reenqueue_running_for_resume()
    assert n >= 1
    r = run_store.get_run(run_id)
    assert r["status"] == "queued"
    assert r["resume_count"] == 1
    assert r["priority"] == 10


def test_reenqueue_never_fails_even_far_past_old_cap():
    """The crux: a run resumed many times still goes BACK TO QUEUE, never
    ``failed``. resume_count keeps climbing (visibility) but trips no auto-fail."""
    run_id = make_run(status="running")
    _set(run_id, resume_count=9)  # well past the old cap of 5
    n = run_store.reenqueue_running_for_resume()
    assert n >= 1
    r = run_store.get_run(run_id)
    assert r["status"] == "queued", "must re-queue regardless of resume_count, never fail"
    assert r["resume_count"] == 10
    assert "auto-resume cap" not in (r["error"] or "")
    assert r["finished_at"] is None


def test_update_run_rejects_unknown_column():
    """``update_run``'s ``_UPDATABLE`` whitelist makes a column typo a loud
    ``ValueError`` at call time instead of silently dropping the whole UPDATE
    (a misspelled kwarg in a caller would otherwise be a no-op that looks like
    a successful write). Guards against re-loosening that validation."""
    run_id = make_run(status="queued")

    # Reject path: an off-whitelist column name (here a typo of ``status``)
    # must raise — never reach the SQL with an un-vetted identifier.
    with pytest.raises(ValueError, match="unknown columns"):
        run_store.update_run(run_id, statuss="running")

    # The rejected call must have written nothing: status is unchanged.
    assert run_store.get_run(run_id)["status"] == "queued"


def test_update_run_no_fields_is_noop_returning_current_row():
    """Calling ``update_run`` with no fields is a read-through no-op: it returns
    the current row unchanged rather than emitting an empty/invalid UPDATE."""
    run_id = make_run(status="queued")
    row = run_store.update_run(run_id)
    assert row is not None
    assert row["id"] == run_id
    assert row["status"] == "queued"


def test_update_run_writes_whitelisted_column():
    """Positive control: a whitelisted column DOES get written, so the reject
    test above is proving a guard, not a broken update path."""
    run_id = make_run(status="queued")
    row = run_store.update_run(run_id, status="running")
    assert row["status"] == "running"
    assert run_store.get_run(run_id)["status"] == "running"

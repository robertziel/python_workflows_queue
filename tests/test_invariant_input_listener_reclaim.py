"""Input listener reclaim of stuck ``processing`` rows.

The contract: no row stays in ``processing`` longer than the reclaim threshold
(``INPUT_CLAIM_RECLAIM_S``).
"""

from __future__ import annotations

import uuid

import pytest

from queue_workflows import dispatcher, input_listener, run_store
from queue_workflows.db import connection


def _make_run() -> str:
    run_id = str(uuid.uuid4())
    run_store.insert_run(
        run_id=run_id, workflow_name="_listener_reclaim_test",
        out_dir="/tmp/out", status="awaiting_input", mode="node",
    )
    return run_id


def _insert_pending(run_id: str, node_id: str, value=None) -> str:
    sub_id = str(uuid.uuid4())
    with connection() as c, c.cursor() as cur:
        cur.execute(
            """
            INSERT INTO workflow_input_submissions
                (id, run_id, node_id, value, status)
            VALUES (%s, %s, %s, %s::jsonb, 'pending')
            """,
            (sub_id, run_id, node_id, '"x"' if value is None else value),
        )
    return sub_id


def _force_status(sub_id: str, *, status: str,
                  claimed_at_seconds_ago: int | None = None) -> None:
    with connection() as c, c.cursor() as cur:
        if claimed_at_seconds_ago is None:
            cur.execute(
                "UPDATE workflow_input_submissions SET status=%s WHERE id=%s",
                (status, sub_id),
            )
        else:
            cur.execute(
                "UPDATE workflow_input_submissions "
                "SET status=%s, claimed_at = now() - make_interval(secs => %s) "
                "WHERE id=%s",
                (status, claimed_at_seconds_ago, sub_id),
            )


def _get_status(sub_id: str) -> str:
    with connection() as c, c.cursor() as cur:
        cur.execute(
            "SELECT status FROM workflow_input_submissions WHERE id=%s",
            (sub_id,),
        )
        return cur.fetchone()["status"]


def test_invariant_pending_rows_still_picked_up():
    run_id = _make_run()
    sub_id = _insert_pending(run_id, "pick_pano", value='"sv1"')
    rows = input_listener.InputListener._claim_pending()
    assert sub_id in [r["id"] for r in rows]
    assert _get_status(sub_id) == "processing"


def test_invariant_processing_within_threshold_not_reclaimed():
    run_id = _make_run()
    sub_id = _insert_pending(run_id, "pick_pano")
    rows = input_listener.InputListener._claim_pending()
    assert any(r["id"] == sub_id for r in rows)

    rows2 = input_listener.InputListener._claim_pending()
    assert all(r["id"] != sub_id for r in rows2)
    assert _get_status(sub_id) == "processing"


def test_crash_recovery_processing_reclaimed_after_threshold():
    run_id = _make_run()
    sub_id = _insert_pending(run_id, "pick_pano")
    _force_status(sub_id, status="processing",
                  claimed_at_seconds_ago=int(input_listener.INPUT_CLAIM_RECLAIM_S * 2))
    rows = input_listener.InputListener._claim_pending()
    assert sub_id in [r["id"] for r in rows]


def test_invariant_no_orphan_processing_after_recovery_window():
    run_id = _make_run()
    sub_ids = [
        _insert_pending(run_id, f"input_{i}", value=f'"v{i}"') for i in range(3)
    ]
    for sid in sub_ids:
        _force_status(sid, status="processing",
                      claimed_at_seconds_ago=int(input_listener.INPUT_CLAIM_RECLAIM_S * 2))

    listener = input_listener.InputListener()
    listener._poll_once()

    with connection() as c, c.cursor() as cur:
        cur.execute(
            "SELECT id, status FROM workflow_input_submissions WHERE id = ANY(%s)",
            (sub_ids,),
        )
        rows = list(cur.fetchall())
    bad = [r for r in rows if r["status"] == "processing"]
    assert not bad


def test_race_listener_crash_between_claim_and_process(monkeypatch):
    run_id = _make_run()
    sub_id = _insert_pending(run_id, "pick_pano", value='"v1"')

    calls: list[tuple[str, str]] = []

    def stub_resume(rid: str, nid: str, value=None) -> int:
        calls.append((rid, nid))
        return 1

    monkeypatch.setattr(dispatcher, "resume_after_input", stub_resume)

    listener = input_listener.InputListener()
    listener._poll_once()
    assert _get_status(sub_id) == "processed"
    assert calls == [(run_id, "pick_pano")]

    _force_status(sub_id, status="processing",
                  claimed_at_seconds_ago=int(input_listener.INPUT_CLAIM_RECLAIM_S * 2))

    listener._poll_once()
    assert _get_status(sub_id) == "processed"
    assert len(calls) == 2


def test_invariant_reclaim_threshold_is_tunable():
    assert hasattr(input_listener, "INPUT_CLAIM_RECLAIM_S")
    assert isinstance(input_listener.INPUT_CLAIM_RECLAIM_S, (int, float))
    assert input_listener.INPUT_CLAIM_RECLAIM_S > 0

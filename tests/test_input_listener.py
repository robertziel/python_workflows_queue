"""Unit tests for ``input_listener``.

Drives the durable input-submission path end-to-end against the real test DB,
with ``dispatcher.resume_after_input`` monkeypatched so we observe what the
poller forwards without actually walking a DAG.
"""

from __future__ import annotations

import uuid

import pytest

from queue_workflows import input_listener, run_store
from queue_workflows.db import connection


def _mk_run(workflow_name: str = "with_input") -> str:
    run_id = str(uuid.uuid4())
    run_store.insert_run(
        run_id=run_id, workflow_name=workflow_name,
        out_dir="/tmp/out", status="awaiting_input", mode="node",
    )
    return run_id


def _mk_submission(run_id: str, node_id: str, value) -> str:
    import json
    sub_id = str(uuid.uuid4())
    with connection() as conn, conn.cursor() as cur:
        cur.execute(
            "INSERT INTO workflow_input_submissions "
            "(id, run_id, node_id, value, status) "
            "VALUES (%s, %s, %s, %s::jsonb, 'pending')",
            (sub_id, run_id, node_id, json.dumps(value)),
        )
        conn.commit()
    return sub_id


def _status_of(sub_id: str) -> str:
    with connection() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT status FROM workflow_input_submissions WHERE id=%s",
            (sub_id,),
        )
        return cur.fetchone()["status"]


# ── Claim / dispatch / terminal state ───────────────────────────────────


def test_poll_claims_pending_and_marks_processed(monkeypatch):
    run_id = _mk_run()
    sub_id = _mk_submission(run_id, "ask_pano", "PANO_42")

    calls: list[tuple] = []

    def fake_resume(r, n, value=None):
        calls.append((r, n, value))
        return 1

    monkeypatch.setattr(
        "queue_workflows.dispatcher.resume_after_input", fake_resume,
    )

    listener = input_listener.InputListener()
    listener._poll_once()

    assert calls == [(run_id, "ask_pano", "PANO_42")]
    assert _status_of(sub_id) == "processed"


def test_poll_marks_failed_on_dispatcher_exception(monkeypatch):
    run_id = _mk_run()
    sub_id = _mk_submission(run_id, "ask_pano", "x")

    def fake_resume(r, n, value=None):
        raise RuntimeError("deliberate")

    monkeypatch.setattr(
        "queue_workflows.dispatcher.resume_after_input", fake_resume,
    )

    listener = input_listener.InputListener()
    listener._poll_once()

    assert _status_of(sub_id) == "failed"
    with connection() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT error FROM workflow_input_submissions WHERE id=%s",
            (sub_id,),
        )
        row = cur.fetchone()
    assert "deliberate" in row["error"]


def test_poll_is_idempotent_on_already_processed_rows(monkeypatch):
    run_id = _mk_run()
    sub_id = _mk_submission(run_id, "ask_pano", "v")

    calls: list = []
    monkeypatch.setattr(
        "queue_workflows.dispatcher.resume_after_input",
        lambda r, n, value=None: calls.append(value),
    )

    listener = input_listener.InputListener()
    listener._poll_once()
    listener._poll_once()

    assert len(calls) == 1
    assert _status_of(sub_id) == "processed"


def test_claim_pending_skip_locked_isolates_concurrent_pollers():
    """Two pollers claiming in parallel must not both pick up the same row.
    ``FOR UPDATE SKIP LOCKED`` is the guard — assert it by holding a row lock
    in one transaction while another tries to claim."""
    run_id = _mk_run()
    sub_a = _mk_submission(run_id, "node_a", "a")
    sub_b = _mk_submission(run_id, "node_b", "b")

    with connection() as held_conn:
        held_conn.autocommit = False
        with held_conn.cursor() as hc:
            hc.execute(
                "SELECT id FROM workflow_input_submissions "
                "WHERE id = %s FOR UPDATE",
                (sub_a,),
            )
            claimed = input_listener.InputListener._claim_pending()
            claimed_ids = {r["id"] for r in claimed}
            assert sub_a not in claimed_ids
            assert sub_b in claimed_ids
        held_conn.rollback()

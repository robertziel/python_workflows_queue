"""Invariant: ``NODE_EVENT_TYPES`` (the Python validation frozenset) MUST equal the
``workflow_node_events_type_check`` DB CHECK set — exactly.

Why this is load-bearing: :func:`node_queue.record_node_event` is BEST-EFFORT — it
swallows the ``ValueError`` that :func:`record_node_event_in_txn` raises for an unknown
``event_type``. So if the frozenset and the DB CHECK drift, the engine **silently drops**
events the DB would happily accept — with no error surfaced on the load-bearing path.

This test exists because of exactly that regression: migration 0015 added
``'unassignable'`` to the DB CHECK (and ``node_pool._sweep_unassignable_jobs`` emits it),
but it was NOT added to ``NODE_EVENT_TYPES`` — so every unassignable event was rejected in
Python and swallowed, and the durable forensic event never persisted despite the CHECK
allowing it. Keeping the two allow-lists in lock-step is the contract.
"""

from __future__ import annotations

import re

from queue_workflows.db import connection
from queue_workflows.node_queue import NODE_EVENT_TYPES


def test_node_event_types_match_db_check():
    with connection() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT pg_get_constraintdef(oid) AS def FROM pg_constraint "
            "WHERE conname = 'workflow_node_events_type_check'"
        )
        row = cur.fetchone()

    assert row, (
        "workflow_node_events_type_check not found — migration 0011/0015 not applied?"
    )
    db_types = set(re.findall(r"'([a-z_]+)'", row["def"]))
    py_types = set(NODE_EVENT_TYPES)

    assert db_types == py_types, (
        "NODE_EVENT_TYPES drifted from the DB CHECK — events will be silently dropped. "
        f"In DB CHECK only: {sorted(db_types - py_types)}; "
        f"in NODE_EVENT_TYPES only: {sorted(py_types - db_types)}"
    )
    # the specific type whose omission motivated this test
    assert "unassignable" in py_types

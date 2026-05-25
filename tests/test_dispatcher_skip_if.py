"""Unit tests for ``dispatcher._should_skip_node`` + the skipped-node flow
through ``_find_ready_nodes``.

Pure DAG-walk logic in isolation — no DB inserts, no real workflow registry.
We construct the workflow dict + the ``existing`` jobs map by hand.
"""

from __future__ import annotations

from queue_workflows import dispatcher


# ── _should_skip_node ────────────────────────────────────────────────────


def test_no_skip_if_field_means_never_skip():
    node = {"id": "edit", "depends_on": ["turnout"]}
    assert dispatcher._should_skip_node(node, run=None, existing={}) is False


def test_skip_if_eq_true_skips():
    node = {
        "id": "rotate",
        "skip_if": {"$from": "turnout.path", "$ne": "rotate"},
    }
    run = {"context": {}}
    existing = {
        "turnout": {"status": "completed", "context_delta": {"path": "direct"}},
    }
    assert dispatcher._should_skip_node(node, run, existing) is True


def test_skip_if_eq_false_runs():
    node = {
        "id": "rotate",
        "skip_if": {"$from": "turnout.path", "$ne": "rotate"},
    }
    run = {"context": {}}
    existing = {
        "turnout": {"status": "completed", "context_delta": {"path": "rotate"}},
    }
    assert dispatcher._should_skip_node(node, run, existing) is False


def test_skip_if_unresolvable_ref_does_not_skip():
    """Fail-safe: a malformed ref / missing context value falls back to 'don't
    skip' so a typo can't dead-end the whole branch."""
    node = {
        "id": "rotate",
        "skip_if": {"$from": "no_such_step.no_such_key", "$eq": "x"},
    }
    run = {"context": {}}
    assert dispatcher._should_skip_node(node, run, existing={}) is False


def test_eval_skip_context_merges_completed_sibling_deltas():
    run = {"context": {"parcel": {"label": "p1"}}}
    existing = {
        "a": {"status": "completed", "context_delta": {"v": 1}},
        "b": {"status": "queued", "context_delta": {"v": 2}},
        "c": {"status": "skipped", "context_delta": {}},
        "d": {"status": "completed", "context_delta": {"v": 4}},
    }
    ctx = dispatcher._eval_skip_context(run, existing)
    assert ctx["parcel"] == {"label": "p1"}
    assert ctx["a"] == {"v": 1}
    assert ctx["d"] == {"v": 4}
    assert "b" not in ctx


# ── _find_ready_nodes treats 'skipped' as satisfied ──────────────────────


def test_find_ready_treats_skipped_dep_as_satisfied():
    workflow = {
        "name": "wf",
        "mode": "node",
        "steps": [
            {"id": "a", "kind": "input", "widget": "confirm"},
            {"id": "b", "kind": "input", "widget": "confirm",
             "depends_on": ["a"]},
        ],
    }
    existing = {"a": {"status": "skipped"}}
    ready = dispatcher._find_ready_nodes(workflow, existing)
    ids = [n["id"] for n in ready]
    assert "b" in ids


def test_find_ready_blocks_on_running_dep():
    workflow = {
        "name": "wf",
        "mode": "node",
        "steps": [
            {"id": "a", "kind": "input", "widget": "confirm"},
            {"id": "b", "kind": "input", "widget": "confirm",
             "depends_on": ["a"]},
        ],
    }
    existing = {"a": {"status": "running"}}
    ready = dispatcher._find_ready_nodes(workflow, existing)
    ids = [n["id"] for n in ready]
    assert "b" not in ids


# ── _nodes_of propagates skip_if ──────────────────────────────────────────


def test_step_level_skip_if_propagates_to_input_node():
    workflow = {
        "name": "wf",
        "mode": "node",
        "steps": [
            {
                "id": "pick_ref_direct",
                "kind": "input",
                "widget": "pick_or_upload",
                "library": "fence_references",
                "skip_if": {"$from": "turnout.path", "$ne": "direct"},
                "depends_on": ["turnout"],
            },
        ],
    }
    nodes = dispatcher._nodes_of(workflow)
    pick = next(n for n in nodes if n["id"] == "pick_ref_direct")
    assert pick["skip_if"] == {"$from": "turnout.path", "$ne": "direct"}

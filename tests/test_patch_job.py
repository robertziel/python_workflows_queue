"""PATCH jobs — the operator's "fix an earlier step in place" (mode B).

A patch job is a synthetic, NON-DAG node-job row: it re-runs one node's module
against the CURRENT input image while the run sits parked at an awaiting-input
node. Its ``inputs`` carry a ``__patch__`` marker::

    {"__patch__": {"target_input": "<input node_id>", "source_node": "<node_id>"}}

Dispatcher contract under test:

* ``on_node_completed`` for a patch job makes the patch outputs CANONICAL —
  each ``context_delta`` key shared with the source node whose value is a
  changed file path under ``out_dir`` is copied over the source node's
  original path (downstream ``$from`` refs and the parked spec keep their
  paths; the bytes change underneath) — then bumps the parked input's
  ``input_spec.patched_at`` and returns WITHOUT cascading the DAG, without a
  workflow provider, and without completing the run.
* ``on_node_failed`` for a patch job leaves the run alive (no run-fail, no
  sibling cancel) and surfaces the error on the spec as ``patch_error``.
"""

from __future__ import annotations

import uuid
from pathlib import Path

import pytest

from queue_workflows import dispatcher, node_queue, run_store
from queue_workflows.db import connection


def _insert_run(out_dir: Path) -> str:
    run_id = str(uuid.uuid4())
    run_store.insert_run(
        run_id=run_id,
        # Deliberately NOT registered with any workflow provider: the patch
        # path must never need _load_workflow.
        workflow_name="wf_patch_test_unregistered",
        out_dir=str(out_dir),
        status="running",
        mode="node",
    )
    return run_id


def _job(run_id: str, node_id: str) -> dict:
    with connection() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT status, input_spec FROM workflow_node_jobs "
            "WHERE run_id = %s AND node_id = %s",
            (run_id, node_id),
        )
        row = cur.fetchone()
        assert row is not None, f"no job row for {node_id}"
        return {"status": row["status"], "input_spec": row["input_spec"]}


def _job_count(run_id: str) -> int:
    with connection() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT count(*) AS n FROM workflow_node_jobs WHERE run_id = %s",
            (run_id,),
        )
        return int(cur.fetchone()["n"])


@pytest.fixture
def patched_run(tmp_path):
    """Run with: completed source node (file + delta), parked input whose spec
    points at the source file, and a queued PATCH job whose own output file
    (different bytes) is already on disk."""
    out = tmp_path / "run"
    src_dir = out / "remove_car" / "erase"
    src_dir.mkdir(parents=True)
    orig = src_dir / "no_car.jpg"
    orig.write_bytes(b"ORIGINAL")

    run_id = _insert_run(out)

    src_job_id = node_queue.enqueue_node_job(
        run_id=run_id, node_id="remove_car/erase",
        node_module="car_remove", queue="cpu", inputs={},
    )
    node_queue.mark_completed(
        src_job_id,
        context_delta={"no_car_path": str(orig), "note": "kept"},
        seconds=0.0,
    )

    inp_job_id = node_queue.enqueue_node_job(
        run_id=run_id, node_id="pick_fence",
        node_module="__input__pick_fence", queue="cpu",
        inputs={"widget": "pick_fence", "target": "selected_mask_path"},
    )
    node_queue.mark_awaiting_input(inp_job_id)
    node_queue.set_input_spec(run_id, "pick_fence", {
        "widget": "pick_fence",
        "source_abs_path": str(orig),
        "source_rel_path": "remove_car/erase/no_car.jpg",
        # The consumer (Rails patcher) sets this when it enqueues the patch so
        # every polling UI disables the input while the re-run is in flight;
        # the engine MUST clear it on patch completion AND failure.
        "patch_pending": True,
        "patch_node_id": "__patch__/p1",
    })

    patch_dir = out / "__patch__" / "p1"
    patch_dir.mkdir(parents=True)
    patched = patch_dir / "no_car.jpg"
    patched.write_bytes(b"PATCHED")

    patch_job_id = node_queue.enqueue_node_job(
        run_id=run_id, node_id="__patch__/p1",
        node_module="car_remove", queue="cpu",
        inputs={
            "sv_path": str(orig),
            "fix_comment": "there is still a car on the right",
            "__patch__": {
                "target_input": "pick_fence",
                "source_node": "remove_car/erase",
            },
        },
    )

    return {
        "run_id": run_id,
        "orig": orig,
        "patched": patched,
        "patch_job_id": patch_job_id,
    }


def test_patch_completion_overwrites_original_and_bumps_spec(patched_run):
    run_id = patched_run["run_id"]
    node_queue.mark_completed(
        patched_run["patch_job_id"],
        context_delta={"no_car_path": str(patched_run["patched"]), "note": "kept"},
        seconds=1.0,
    )

    new_jobs = dispatcher.on_node_completed(run_id, "__patch__/p1")

    assert new_jobs == 0, "a patch must never cascade the DAG"
    # canonical content replaced at the ORIGINAL path
    assert patched_run["orig"].read_bytes() == b"PATCHED"
    # parked input untouched, spec bumped (same paths, new patched_at)
    inp = _job(run_id, "pick_fence")
    assert inp["status"] == "awaiting_input"
    spec = inp["input_spec"]
    assert spec["source_abs_path"] == str(patched_run["orig"])
    assert spec.get("patched_at"), "spec must carry a patched_at cache-buster"
    assert "patch_error" not in spec
    # input re-enabled: the pending flag set at enqueue time must be cleared
    assert "patch_pending" not in spec
    assert "patch_node_id" not in spec
    # run alive, no extra rows
    assert (run_store.get_run(run_id) or {}).get("status") == "running"
    assert _job_count(run_id) == 3


def test_patch_failure_leaves_run_alive_and_flags_spec(patched_run):
    run_id = patched_run["run_id"]
    node_queue.mark_failed(patched_run["patch_job_id"], error="CUDA exploded")

    dispatcher.on_node_failed(run_id, "__patch__/p1")

    # the run survives; the parked input is NOT cancelled
    assert (run_store.get_run(run_id) or {}).get("status") == "running"
    inp = _job(run_id, "pick_fence")
    assert inp["status"] == "awaiting_input"
    spec = inp["input_spec"]
    assert "CUDA exploded" in (spec.get("patch_error") or "")
    # input re-enabled even on failure — the operator continues or retries
    assert "patch_pending" not in spec
    assert "patch_node_id" not in spec
    # original bytes untouched
    assert patched_run["orig"].read_bytes() == b"ORIGINAL"


def test_patch_completion_ignores_non_file_and_unmatched_keys(patched_run):
    """Only same-key, changed, under-out_dir file paths are copied — scalar
    keys and keys absent on the source delta must be ignored, not crash."""
    run_id = patched_run["run_id"]
    node_queue.mark_completed(
        patched_run["patch_job_id"],
        context_delta={
            "no_car_path": str(patched_run["patched"]),
            "note": "changed scalar",          # scalar — ignored
            "extra_path": str(patched_run["patched"]),  # no source twin — ignored
        },
        seconds=1.0,
    )

    dispatcher.on_node_completed(run_id, "__patch__/p1")

    assert patched_run["orig"].read_bytes() == b"PATCHED"

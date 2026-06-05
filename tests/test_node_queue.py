"""Unit tests for node-per-job queue DB plumbing + DAG dispatcher.

Covers:
- routing by queue + required_model
- DB invariants (CHECK constraints, UNIQUE)
- claim ordering (cpu: FIFO by priority; gpu: prefer current model)
- DAG scheduling (enqueue initial, enqueue dependents, diamond, fanout)
- failure cancellation of queued siblings
- awaiting_input transitions + resume
- run status transitions (queued→running→completed/failed)

The dispatcher DAG tests drive an in-test dict-backed workflow provider (via
``queue_workflows.set_workflow_provider``) instead of the ai_leads filesystem
registry — keeping the engine suite domain-free.
"""

from __future__ import annotations

import uuid

import psycopg.errors as pge
import pytest

import queue_workflows
from queue_workflows import dispatcher, node_queue, run_store
from tests._helpers import make_run as _make_run


# ── Enqueue + DB invariants ──────────────────────────────────────────────


def test_enqueue_cpu_job_without_model_succeeds():
    run_id = _make_run()
    job_id = node_queue.enqueue_node_job(
        run_id=run_id, node_id="n1", node_module="smoke_heartbeat", queue="cpu",
    )
    row = node_queue.get_node_job(job_id)
    assert row["status"] == "queued"
    assert row["queue"] == "cpu"
    assert row["required_model"] is None


def test_enqueue_gpu_job_with_model_succeeds():
    run_id = _make_run()
    job_id = node_queue.enqueue_node_job(
        run_id=run_id, node_id="n1", node_module="fence_install",
        queue="gpu", required_model="sdxl_ipadapter",
    )
    row = node_queue.get_node_job(job_id)
    assert row["queue"] == "gpu"
    assert row["required_model"] == "sdxl_ipadapter"


def test_enqueue_gpu_without_model_allowed():
    run_id = _make_run()
    job_id = node_queue.enqueue_node_job(
        run_id=run_id, node_id="n1", node_module="x", queue="gpu",
    )
    row = node_queue.get_node_job(job_id)
    assert row["queue"] == "gpu"
    assert row["required_model"] is None


def test_enqueue_cpu_with_model_rejected():
    run_id = _make_run()
    with pytest.raises(ValueError, match="required_model"):
        node_queue.enqueue_node_job(
            run_id=run_id, node_id="n1", node_module="x",
            queue="cpu", required_model="whatever",
        )


def test_enqueue_invalid_queue_rejected():
    run_id = _make_run()
    with pytest.raises(ValueError, match="queue must be"):
        node_queue.enqueue_node_job(
            run_id=run_id, node_id="n1", node_module="x", queue="tpu",
        )


def test_duplicate_node_id_rejected_by_unique_index():
    run_id = _make_run()
    node_queue.enqueue_node_job(
        run_id=run_id, node_id="n1", node_module="x", queue="cpu",
    )
    with pytest.raises(pge.UniqueViolation):
        node_queue.enqueue_node_job(
            run_id=run_id, node_id="n1", node_module="x", queue="cpu",
        )


# ── Claim ordering ────────────────────────────────────────────────────────


def test_claim_cpu_returns_none_when_empty():
    assert node_queue.claim_next_cpu_job(worker_lane=0) is None


def test_claim_cpu_is_priority_then_fifo():
    run_id = _make_run()
    a = node_queue.enqueue_node_job(run_id=run_id, node_id="a", node_module="x",
                                    queue="cpu", priority=100)
    b = node_queue.enqueue_node_job(run_id=run_id, node_id="b", node_module="x",
                                    queue="cpu", priority=50)
    node_queue.enqueue_node_job(run_id=run_id, node_id="c", node_module="x",
                                queue="cpu", priority=100)
    first = node_queue.claim_next_cpu_job(0)
    assert first["id"] == b
    second = node_queue.claim_next_cpu_job(0)
    assert second["id"] == a


def test_claim_gpu_without_model_uses_priority_fifo():
    run_id = _make_run()
    a = node_queue.enqueue_node_job(run_id=run_id, node_id="a", node_module="x",
                                    queue="gpu", required_model="sdxl")
    node_queue.enqueue_node_job(run_id=run_id, node_id="b", node_module="x",
                                queue="gpu", required_model="flux")
    claimed = node_queue.claim_next_gpu_job(worker_lane=0, current_model=None)
    assert claimed["id"] == a


def test_claim_gpu_prefers_current_model_over_fifo():
    run_id = _make_run()
    node_queue.enqueue_node_job(run_id=run_id, node_id="a", node_module="x",
                                queue="gpu", required_model="sdxl")
    b = node_queue.enqueue_node_job(run_id=run_id, node_id="b", node_module="x",
                                    queue="gpu", required_model="flux")
    claimed = node_queue.claim_next_gpu_job(worker_lane=0, current_model="flux")
    assert claimed["id"] == b


def test_claim_gpu_falls_back_to_other_model_when_no_match():
    run_id = _make_run()
    node_queue.enqueue_node_job(run_id=run_id, node_id="a", node_module="x",
                                queue="gpu", required_model="sdxl")
    claimed = node_queue.claim_next_gpu_job(worker_lane=0, current_model="flux")
    assert claimed["required_model"] == "sdxl"


# ── Mark completed / failed / awaiting_input ─────────────────────────────


def test_mark_completed_sets_finished_and_context_delta():
    run_id = _make_run()
    job_id = node_queue.enqueue_node_job(
        run_id=run_id, node_id="a", node_module="x", queue="cpu",
    )
    node_queue.claim_next_cpu_job(0)
    row = node_queue.mark_completed(
        job_id, context_delta={"k": "v"}, seconds=1.5, vm_rss_mb_peak=123,
    )
    assert row["status"] == "completed"
    assert row["seconds"] == 1.5
    assert row["vm_rss_mb_peak"] == 123
    assert row["context_delta"] == {"k": "v"}
    assert row["finished_at"] is not None


def test_mark_failed_stores_error_truncated():
    run_id = _make_run()
    job_id = node_queue.enqueue_node_job(
        run_id=run_id, node_id="a", node_module="x", queue="cpu",
    )
    node_queue.claim_next_cpu_job(0)
    long_err = "x" * 20000
    row = node_queue.mark_failed(job_id, error=long_err, seconds=0.1)
    assert row["status"] == "failed"
    assert len(row["error"]) == 8000


def test_terminal_jobs_stamp_host_label_from_claimed_by():
    """A terminal job records its executing machine (claimed_by) on the row so
    per-host error/log queries work off workflow_node_jobs, not just events —
    host_label was left NULL in practice. (The test env configures no host, so we
    set claimed_by directly to exercise the COALESCE stamp.)"""
    from queue_workflows import db

    def _set_claimed_by(job_id, host):
        with db.connection() as conn, conn.cursor() as cur:
            cur.execute("UPDATE workflow_node_jobs SET claimed_by=%s WHERE id=%s", (host, job_id))

    run_id = _make_run()
    # failed → host_label stamped from claimed_by
    f_id = node_queue.enqueue_node_job(run_id=run_id, node_id="f", node_module="x", queue="cpu")
    node_queue.claim_next_cpu_job(0)
    _set_claimed_by(f_id, "box-a2")
    frow = node_queue.mark_failed(f_id, error="boom", seconds=0.1)
    assert frow["host_label"] == "box-a2"
    # completed → same stamp
    c_id = node_queue.enqueue_node_job(run_id=run_id, node_id="c", node_module="x", queue="cpu")
    node_queue.claim_next_cpu_job(0)
    _set_claimed_by(c_id, "box-b")
    crow = node_queue.mark_completed(c_id, context_delta={}, seconds=0.1)
    assert crow["host_label"] == "box-b"


def test_cancel_queued_jobs_for_run_leaves_running_alone():
    run_id = _make_run()
    node_queue.enqueue_node_job(run_id=run_id, node_id="a", node_module="x",
                                queue="cpu")
    node_queue.enqueue_node_job(run_id=run_id, node_id="b", node_module="x",
                                queue="cpu")
    node_queue.claim_next_cpu_job(0)
    n = node_queue.cancel_queued_jobs_for_run(run_id)
    assert n == 1
    rows = {j["node_id"]: j for j in node_queue.list_jobs_for_run(run_id)}
    assert rows["a"]["status"] == "running"
    assert rows["b"]["status"] == "cancelled"


# ── Dispatcher: DAG expansion (in-test dict-backed workflow provider) ────────


@pytest.fixture
def temp_workflow():
    """Install an ephemeral workflow + matching pipeline schema via the engine's
    injected workflow provider (no filesystem registry). Returns a factory
    ``_make(schema_nodes) -> workflow_name``. Node ids in the queued rows come
    out as ``p/<schema_node_id>`` (step_id is ``p``)."""
    workflows: dict[str, dict] = {}
    pipelines: dict[str, dict] = {}

    def _make(schema_nodes: list[dict]) -> str:
        name = f"_test_{uuid.uuid4().hex[:8]}"
        workflows[name] = {
            "name": name, "mode": "node",
            "steps": [{"id": "p", "kind": "pipeline", "pipeline": name, "inputs": {}}],
        }
        pipelines[name] = {"name": name, "nodes": schema_nodes}
        return name

    queue_workflows.set_workflow_provider(
        lambda n: workflows[n], lambda n: pipelines[n],
    )
    # Expose the dicts so input-step tests (below) can register their own
    # workflow shapes against the SAME provider.
    _make.workflows = workflows  # type: ignore[attr-defined]
    _make.pipelines = pipelines  # type: ignore[attr-defined]
    yield _make


def _nid(s: str) -> str:
    return f"p/{s}"


def test_enqueued_node_job_carries_pipeline_name(temp_workflow):
    name = temp_workflow([
        {"id": "only", "node": "smoke", "depends_on": [], "gpu": False},
    ])
    run_id = _make_run(workflow_name=name)
    dispatcher.start_run(run_id)
    jobs = node_queue.list_jobs_for_run(run_id)
    assert len(jobs) == 1
    assert jobs[0]["pipeline_name"] == name


def test_input_step_job_has_null_pipeline_name(temp_workflow):
    name = f"_test_{uuid.uuid4().hex[:8]}"
    temp_workflow.workflows[name] = {
        "name": name, "mode": "node",
        "steps": [
            {"id": "upload", "kind": "input", "widget": "file_upload", "target": "x"},
        ],
    }
    temp_workflow.pipelines[name] = {"name": name, "nodes": []}
    run_id = _make_run(workflow_name=name)
    dispatcher.start_run(run_id)
    jobs = node_queue.list_jobs_for_run(run_id)
    assert len(jobs) == 1
    assert jobs[0]["pipeline_name"] is None


def test_start_run_enqueues_only_source_nodes(temp_workflow):
    name = temp_workflow([
        {"id": "a", "node": "smoke", "depends_on": [], "gpu": False},
        {"id": "b", "node": "smoke", "depends_on": ["a"], "gpu": False},
    ])
    run_id = _make_run(workflow_name=name)
    n = dispatcher.start_run(run_id)
    assert n == 1
    jobs = {j["node_id"]: j for j in node_queue.list_jobs_for_run(run_id)}
    assert set(jobs) == {_nid("a")}


def test_start_run_is_idempotent(temp_workflow):
    name = temp_workflow([
        {"id": "a", "node": "smoke", "depends_on": [], "gpu": False},
    ])
    run_id = _make_run(workflow_name=name)
    assert dispatcher.start_run(run_id) == 1
    assert dispatcher.start_run(run_id) == 0


def test_on_node_completed_enqueues_dependents(temp_workflow):
    name = temp_workflow([
        {"id": "a", "node": "smoke", "depends_on": [], "gpu": False},
        {"id": "b", "node": "smoke", "depends_on": ["a"], "gpu": False},
    ])
    run_id = _make_run(workflow_name=name)
    dispatcher.start_run(run_id)
    jobs = {j["node_id"]: j for j in node_queue.list_jobs_for_run(run_id)}
    node_queue.claim_next_cpu_job(0)
    node_queue.mark_completed(jobs[_nid("a")]["id"], context_delta={}, seconds=0.1)
    n = dispatcher.on_node_completed(run_id, _nid("a"))
    assert n == 1
    jobs = {j["node_id"]: j for j in node_queue.list_jobs_for_run(run_id)}
    assert set(jobs) == {_nid("a"), _nid("b")}
    assert jobs[_nid("b")]["status"] == "queued"


def test_diamond_dag_joins_once(temp_workflow):
    name = temp_workflow([
        {"id": "a", "node": "smoke", "depends_on": [], "gpu": False},
        {"id": "b", "node": "smoke", "depends_on": ["a"], "gpu": False},
        {"id": "c", "node": "smoke", "depends_on": ["a"], "gpu": False},
        {"id": "d", "node": "smoke", "depends_on": ["b", "c"], "gpu": False},
    ])
    run_id = _make_run(workflow_name=name)
    dispatcher.start_run(run_id)
    jobs = {j["node_id"]: j for j in node_queue.list_jobs_for_run(run_id)}
    node_queue.claim_next_cpu_job(0)
    node_queue.mark_completed(jobs[_nid("a")]["id"], context_delta={}, seconds=0.1)
    dispatcher.on_node_completed(run_id, _nid("a"))
    jobs = {j["node_id"]: j for j in node_queue.list_jobs_for_run(run_id)}
    assert set(jobs) == {_nid("a"), _nid("b"), _nid("c")}
    node_queue.claim_next_cpu_job(0)
    node_queue.mark_completed(jobs[_nid("b")]["id"], context_delta={}, seconds=0.1)
    dispatcher.on_node_completed(run_id, _nid("b"))
    jobs = {j["node_id"]: j for j in node_queue.list_jobs_for_run(run_id)}
    assert _nid("d") not in jobs
    node_queue.mark_completed(jobs[_nid("c")]["id"], context_delta={}, seconds=0.1)
    dispatcher.on_node_completed(run_id, _nid("c"))
    jobs = {j["node_id"]: j for j in node_queue.list_jobs_for_run(run_id)}
    assert _nid("d") in jobs
    assert jobs[_nid("d")]["status"] == "queued"


def test_fanout_routes_gpu_vs_cpu_correctly(temp_workflow):
    name = temp_workflow([
        {"id": "root", "node": "smoke", "depends_on": [], "gpu": False},
        {"id": "cpu_child", "node": "smoke", "depends_on": ["root"], "gpu": False},
        {"id": "gpu_child", "node": "fence_install", "depends_on": ["root"],
         "gpu": True, "model": "sdxl_ipadapter"},
    ])
    run_id = _make_run(workflow_name=name)
    dispatcher.start_run(run_id)
    jobs = {j["node_id"]: j for j in node_queue.list_jobs_for_run(run_id)}
    node_queue.claim_next_cpu_job(0)
    node_queue.mark_completed(jobs[_nid("root")]["id"], context_delta={}, seconds=0.1)
    dispatcher.on_node_completed(run_id, _nid("root"))
    jobs = {j["node_id"]: j for j in node_queue.list_jobs_for_run(run_id)}
    assert jobs[_nid("cpu_child")]["queue"] == "cpu"
    assert jobs[_nid("cpu_child")]["required_model"] is None
    assert jobs[_nid("gpu_child")]["queue"] == "gpu"
    assert jobs[_nid("gpu_child")]["required_model"] == "sdxl_ipadapter"


def test_failure_cancels_queued_siblings_and_marks_run_failed(temp_workflow):
    name = temp_workflow([
        {"id": "a", "node": "smoke", "depends_on": [], "gpu": False},
        {"id": "b", "node": "smoke", "depends_on": [], "gpu": False},
    ])
    run_id = _make_run(workflow_name=name)
    dispatcher.start_run(run_id)
    jobs = {j["node_id"]: j for j in node_queue.list_jobs_for_run(run_id)}
    node_queue.claim_next_cpu_job(0)
    node_queue.mark_failed(jobs[_nid("a")]["id"], error="boom", seconds=0.1)
    dispatcher.on_node_failed(run_id, _nid("a"))
    jobs = {j["node_id"]: j for j in node_queue.list_jobs_for_run(run_id)}
    assert jobs[_nid("a")]["status"] == "failed"
    assert jobs[_nid("b")]["status"] == "cancelled"
    run = run_store.get_run(run_id)
    assert run["status"] == "failed"


def test_run_completed_when_all_nodes_done(temp_workflow):
    name = temp_workflow([
        {"id": "a", "node": "smoke", "depends_on": [], "gpu": False},
        {"id": "b", "node": "smoke", "depends_on": ["a"], "gpu": False},
    ])
    run_id = _make_run(workflow_name=name)
    dispatcher.start_run(run_id)
    jobs = {j["node_id"]: j for j in node_queue.list_jobs_for_run(run_id)}
    node_queue.claim_next_cpu_job(0)
    node_queue.mark_completed(jobs[_nid("a")]["id"], context_delta={}, seconds=0.1)
    dispatcher.on_node_completed(run_id, _nid("a"))
    jobs = {j["node_id"]: j for j in node_queue.list_jobs_for_run(run_id)}
    node_queue.claim_next_cpu_job(0)
    node_queue.mark_completed(jobs[_nid("b")]["id"], context_delta={}, seconds=0.1)
    dispatcher.on_node_completed(run_id, _nid("b"))
    run = run_store.get_run(run_id)
    assert run["status"] == "completed"


def test_input_step_parks_run_in_awaiting_input(temp_workflow):
    name = f"_test_{uuid.uuid4().hex[:8]}"
    temp_workflow.workflows[name] = {
        "name": name, "mode": "node",
        "steps": [
            {"id": "upload", "kind": "input", "widget": "file_upload",
             "target": "image_path"},
            {"id": "p", "kind": "pipeline", "pipeline": name,
             "depends_on": ["upload"], "inputs": {}},
        ],
    }
    temp_workflow.pipelines[name] = {
        "name": name,
        "nodes": [{"id": "parse", "node": "smoke", "depends_on": [], "gpu": False}],
    }
    run_id = _make_run(workflow_name=name)
    dispatcher.start_run(run_id)
    jobs = {j["node_id"]: j for j in node_queue.list_jobs_for_run(run_id)}
    assert jobs["upload"]["queue"] == "cpu"
    assert jobs["upload"]["node_module"].startswith("__input__")
    node_queue.claim_next_cpu_job(0)
    node_queue.mark_awaiting_input(jobs["upload"]["id"])
    dispatcher.on_node_awaiting_input(run_id, "upload")
    upload_job = next(
        j for j in node_queue.list_jobs_for_run(run_id) if j["node_id"] == "upload"
    )
    assert upload_job["status"] == "awaiting_input"

    jobs = {j["node_id"]: j for j in node_queue.list_jobs_for_run(run_id)}
    assert _nid("parse") not in jobs

    dispatcher.resume_after_input(run_id, "upload")
    jobs = {j["node_id"]: j for j in node_queue.list_jobs_for_run(run_id)}
    assert _nid("parse") in jobs
    assert jobs[_nid("parse")]["status"] == "queued"


# ── Snapshot ─────────────────────────────────────────────────────────────


def test_snapshot_splits_cpu_vs_gpu():
    run_id = _make_run()
    node_queue.enqueue_node_job(run_id=run_id, node_id="c1", node_module="x",
                                queue="cpu")
    node_queue.enqueue_node_job(run_id=run_id, node_id="c2", node_module="x",
                                queue="cpu")
    node_queue.enqueue_node_job(run_id=run_id, node_id="g1", node_module="x",
                                queue="gpu", required_model="sdxl")
    snap = node_queue.snapshot()
    assert len(snap["cpu"]["queued"]) == 2
    assert len(snap["gpu"]["queued"]) == 1
    assert snap["counts"]["cpu_queued"] == 2
    assert snap["counts"]["gpu_queued"] == 1

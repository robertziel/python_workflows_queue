"""The per-node ``is_priority`` "run next" flag.

An operator can flag a QUEUED node so the next worker asking for a node in that
queue/capability claims it BEFORE older + default peers — `is_priority` sorts
FIRST in the claim ORDER BY (ahead of the integer priority band and, on GPU,
ahead of the warm-model affinity tiebreak). Only queued rows can be re-ordered.
"""
from __future__ import annotations

from queue_workflows import node_queue
from tests._helpers import make_run as _make_run


def test_cpu_priority_flag_jumps_ahead_of_older_default():
    run_id = _make_run()
    a = node_queue.enqueue_node_job(run_id=run_id, node_id="a", node_module="x", queue="cpu")
    node_queue.enqueue_node_job(run_id=run_id, node_id="b", node_module="x", queue="cpu")
    # A LATER job, flagged priority → claimed FIRST despite a, b being older.
    c = node_queue.enqueue_node_job(run_id=run_id, node_id="c", node_module="x", queue="cpu")
    row = node_queue.prioritize_node_job(c)
    assert row is not None and row["is_priority"] is True

    first = node_queue.claim_next_cpu_job(0)
    assert first["id"] == c
    assert first["is_priority"] is True
    # FIFO resumes for the rest.
    assert node_queue.claim_next_cpu_job(0)["id"] == a


def test_gpu_priority_flag_overrides_warm_model_affinity():
    run_id = _make_run()
    # A warm-model job (worker has 'flux' loaded) — normally claimed first.
    node_queue.enqueue_node_job(run_id=run_id, node_id="warm", node_module="x",
                                queue="gpu", required_model="flux")
    # A cold-model job flagged priority → must jump ahead of the warm one.
    cold = node_queue.enqueue_node_job(run_id=run_id, node_id="cold", node_module="x",
                                       queue="gpu", required_model="sdxl")
    node_queue.prioritize_node_job(cold)
    claimed = node_queue.claim_next_gpu_job(
        worker_lane=0, current_model="flux", known_models=["flux", "sdxl"],
    )
    assert claimed["id"] == cold
    assert claimed["is_priority"] is True


def test_default_jobs_are_not_priority():
    run_id = _make_run()
    j = node_queue.enqueue_node_job(run_id=run_id, node_id="j", node_module="x", queue="cpu")
    assert node_queue.get_node_job(j)["is_priority"] is False


def test_prioritize_only_affects_queued_jobs():
    run_id = _make_run()
    j = node_queue.enqueue_node_job(run_id=run_id, node_id="j", node_module="x", queue="cpu")
    node_queue.claim_next_cpu_job(0)  # j is now running
    # Can't re-order a job that's already been claimed.
    assert node_queue.prioritize_node_job(j) is None
    assert node_queue.get_node_job(j)["is_priority"] is False

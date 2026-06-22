"""Shared GPU pool (pivot B) — a namespace-scoped durable queue of self-contained
GPU tasks that pooled GPU workers across apps claim + execute, while each app keeps
its own DB for run/DAG state.

Contract pinned here:
  * submit → (capability-routed) claim → execute (registered handler) → the app
    awaits the result; the worker touches ONLY the shared pool store + the handler
    (which reads/writes shared NFS) — never an app DB;
  * the pool store is INDEPENDENT of ``config.db_backend`` (ai_leads keeps pg for
    its DAG; the pool is redis);
  * capability routing is by QUEUE NAME (box-b/box-a box-class + warm-model
    affinity as the worker's ordered queue set);
  * idempotent terminal + a failure path the submitter observes.

Redis-only by design; SKIPs when QUEUE_WORKFLOWS_TEST_REDIS_URL is unset.
"""

from __future__ import annotations

import os
import uuid

import pytest

import queue_workflows
from queue_workflows import gpu_pool


@pytest.fixture
def pool():
    """Point the shared pool at a fresh redis namespace; register two handlers.
    Skips if redis is down."""
    url = os.environ.get("QUEUE_WORKFLOWS_TEST_REDIS_URL")
    if not url:
        pytest.skip("set QUEUE_WORKFLOWS_TEST_REDIS_URL to exercise the GPU pool")
    os.environ["QUEUE_WORKFLOWS_GPU_POOL_URL"] = url
    ns = f"pool_{uuid.uuid4().hex[:12]}"
    # db_backend stays "pg" (the app's DAG store) — the pool must NOT use it.
    queue_workflows.configure(db_backend="pg", gpu_pool_namespace=ns)

    ran: list[dict] = []

    def upscale(*, inputs, output_dir, params):
        ran.append({"inputs": inputs, "output_dir": output_dir, "params": params})
        return {"out": f"{output_dir}/upscaled.png", "scale": params.get("scale")}

    def boom(*, inputs, output_dir, params):
        raise RuntimeError("handler exploded")

    queue_workflows.register_pool_handler("upscale", upscale)
    queue_workflows.register_pool_handler("boom", boom)
    try:
        gpu_pool._pool_backend().ensure_schema()
    except Exception as exc:
        pytest.skip(f"redis unreachable: {type(exc).__name__}: {exc}")
    yield ran
    gpu_pool.close_pool_backend()
    from queue_workflows import backends
    backends.close_all()
    os.environ.pop("QUEUE_WORKFLOWS_GPU_POOL_URL", None)


def test_submit_claim_execute_await_roundtrip(pool):
    ran = pool
    tid = gpu_pool.submit_pool_task(
        queue="gpu:box-a", handler="upscale", model="sr",
        inputs={"src": "/nfs/runs/123/in.png"}, output_dir="/nfs/runs/123/out",
        params={"scale": 4},
    )
    assert gpu_pool.get_pool_task(tid)["status"] == "queued"

    # A pooled worker (separate box in prod; here: same process) claims + runs it.
    assert gpu_pool.run_pool_worker_once(queues=["gpu:box-a"], worker="box-a1") == "completed"
    assert ran == [{"inputs": {"src": "/nfs/runs/123/in.png"},
                    "output_dir": "/nfs/runs/123/out", "params": {"scale": 4}}]

    result = gpu_pool.await_pool_result(tid, timeout_s=5, poll_s=0.01)
    assert result == {"out": "/nfs/runs/123/out/upscaled.png", "scale": 4}


def test_pool_is_independent_of_db_backend(pool):
    # db_backend is "pg" (the app's DAG store), but the pool runs on redis.
    assert queue_workflows.get_config().db_backend == "pg"
    assert gpu_pool._pool_backend().name == "redis"
    tid = gpu_pool.submit_pool_task(queue="gpu:box-a", handler="upscale",
                                    inputs={}, output_dir="/nfs/o", params={})
    gpu_pool.run_pool_worker_once(queues=["gpu:box-a"], worker="w")
    assert gpu_pool.await_pool_result(tid, timeout_s=5, poll_s=0.01)["out"] == "/nfs/o/upscaled.png"


def test_capability_routing_by_queue(pool):
    """A box-a box serving only gpu:box-a must NOT claim a box-b task."""
    box-a_t = gpu_pool.submit_pool_task(queue="gpu:box-a", handler="upscale",
                                        inputs={}, output_dir="/o", params={})
    box-b_t = gpu_pool.submit_pool_task(queue="gpu:box-b", handler="upscale",
                                          inputs={}, output_dir="/o", params={})
    # box-a worker drains its lane only.
    assert gpu_pool.run_pool_worker_once(queues=["gpu:box-a"], worker="box-a1") == "completed"
    assert gpu_pool.run_pool_worker_once(queues=["gpu:box-a"], worker="box-a1") is None
    assert gpu_pool.get_pool_task(box-a_t)["status"] == "completed"
    assert gpu_pool.get_pool_task(box-b_t)["status"] == "queued"   # untouched
    # A box-b worker then takes its own.
    assert gpu_pool.run_pool_worker_once(queues=["gpu:box-b"], worker="bl1") == "completed"
    assert gpu_pool.get_pool_task(box-b_t)["status"] == "completed"


def test_warm_model_affinity_is_queue_order(pool):
    """The worker's ordered queue set = warm-model-first affinity: a worker that
    lists its warm model's queue first claims that lane before the colder one."""
    cold = gpu_pool.submit_pool_task(queue="gpu:modelB", handler="upscale",
                                     inputs={}, output_dir="/o", params={})
    warm = gpu_pool.submit_pool_task(queue="gpu:modelA", handler="upscale",
                                     inputs={}, output_dir="/o", params={})
    # Worker warm on modelA lists it first → claims the modelA task first.
    gpu_pool.run_pool_worker_once(queues=["gpu:modelA", "gpu:modelB"], worker="w")
    assert gpu_pool.get_pool_task(warm)["status"] == "completed"
    assert gpu_pool.get_pool_task(cold)["status"] == "queued"


def test_failure_path_surfaces_to_submitter(pool):
    tid = gpu_pool.submit_pool_task(queue="gpu:box-a", handler="boom",
                                    inputs={}, output_dir="/o", params={})
    assert gpu_pool.run_pool_worker_once(queues=["gpu:box-a"], worker="w") == "failed"
    with pytest.raises(gpu_pool.PoolTaskFailed, match="handler exploded"):
        gpu_pool.await_pool_result(tid, timeout_s=5, poll_s=0.01)


def test_idempotent_terminal_no_clobber(pool):
    tid = gpu_pool.submit_pool_task(queue="gpu:box-a", handler="upscale",
                                    inputs={}, output_dir="/o", params={"scale": 2})
    job = gpu_pool._pool_backend().claim("gpu:box-a", "w", lease_s=30)
    assert gpu_pool.execute_pool_task(job) == "completed"
    # A duplicate execute of the same (already-terminal) task is a no-op.
    assert gpu_pool.execute_pool_task(job) == "skipped"
    assert gpu_pool.await_pool_result(tid, timeout_s=5, poll_s=0.01)["scale"] == 2


def test_unknown_handler_marks_failed(pool):
    """A task whose handler isn't registered on THIS worker fails loudly (visible
    to the submitter) rather than silently vanishing."""
    tid = gpu_pool.submit_pool_task(queue="gpu:box-a", handler="ghost_op",
                                    inputs={}, output_dir="/o", params={})
    assert gpu_pool.run_pool_worker_once(queues=["gpu:box-a"], worker="w") == "failed"
    with pytest.raises(gpu_pool.PoolTaskFailed, match="ghost_op"):
        gpu_pool.await_pool_result(tid, timeout_s=5, poll_s=0.01)


def test_await_times_out_when_task_never_worked(pool):
    tid = gpu_pool.submit_pool_task(queue="gpu:box-a", handler="upscale",
                                    inputs={}, output_dir="/o", params={})
    # No worker runs it → await must raise TimeoutError (not hang, not return).
    with pytest.raises(TimeoutError, match="not terminal"):
        gpu_pool.await_pool_result(tid, timeout_s=0.2, poll_s=0.05)
    assert gpu_pool.get_pool_task(tid)["status"] == "queued"  # still claimable


def test_await_unknown_task_raises(pool):
    with pytest.raises(gpu_pool.PoolTaskFailed, match="not found"):
        gpu_pool.await_pool_result("no-such-task-id", timeout_s=1, poll_s=0.05)


def test_reclaim_expired_pool_lease(pool):
    import time
    tid = gpu_pool.submit_pool_task(queue="gpu:box-a", handler="upscale",
                                    inputs={}, output_dir="/o", params={})
    gpu_pool._pool_backend().claim("gpu:box-a", "dead-worker", lease_s=1)
    assert gpu_pool.get_pool_task(tid)["status"] == "running"
    time.sleep(1.2)
    reclaimed = gpu_pool.reclaim_expired_pool_leases()
    assert tid in reclaimed
    assert gpu_pool.get_pool_task(tid)["status"] == "queued"   # recoverable

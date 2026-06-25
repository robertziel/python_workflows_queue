"""One queue, all 3 projects, via the CLIENT LIBRARY — end-to-end + isolation.

The headline guarantee of the consolidation, proven at the *full lifecycle* level
(not just the queue primitives): three real projects — ai_leads, alpha,
beta — each wired ONLY through the public client API
(``configure(project=…)`` + the workflow/node hooks + ``dispatcher.start_run`` +
``ClaimWorker`` + ``NodePool``) share ONE broker database and each runs a
node-mode run to completion, while a project's worker claims ONLY its own jobs.

Faithful to the deployment shape: one process hosts one ``EngineConfig``, so each
"client" is ``configure(project=X)`` against the shared broker; ``ClaimWorker``
claims for ``config.project`` and the outbox drain correlates to the parent run's
project — exactly how each project runs its own orchestrator+worker on the broker.

Runs on both backends (the conftest test DB == the shared broker).
"""

from __future__ import annotations

import sys
import threading
import types
import uuid

import pytest

import queue_workflows
from queue_workflows import dispatcher, node_pool, node_queue, run_store
from queue_workflows.claim_worker import ClaimWorker
from queue_workflows.db import connection
from tests._helpers import force_lease, make_run, row

PROJECTS = ["ai_leads", "alpha", "beta"]

# ── shared client wiring (one fake workflow + node, used by every project) ──
_RAN: list[str] = []  # the project each worker was configured as when it executed


def _wire_client_hooks() -> None:
    """What every project's startup does once: a node-module package + a
    one-node workflow/pipeline provider. Shared across the 3 projects (they run
    the same trivial DAG; only their `project` tag + their runs differ)."""
    _RAN.clear()
    mod = types.ModuleType("qwf_3p_nodes.echo_node")

    def echo_run(inputs, out=None):
        # Records the project the EXECUTING worker is configured as — proving
        # project X's worker ran X's node under X's config.
        _RAN.append(queue_workflows.get_config().project)
        return {"context_delta": {"echoed": True}}

    mod.run = echo_run
    sys.modules["qwf_3p_nodes.echo_node"] = mod
    queue_workflows.set_node_module_package("qwf_3p_nodes")
    workflows = {"_3p_wf": {"name": "_3p_wf",
                            "steps": [{"id": "p", "kind": "pipeline", "pipeline": "_3p_pipe"}]}}
    pipelines = {"_3p_pipe": {"name": "_3p_pipe", "nodes": [{"id": "n1", "node": "echo_node"}]}}
    queue_workflows.set_workflow_provider(lambda n: workflows[n], lambda n: pipelines[n])


def _client_enqueue_run(project: str) -> str:
    """Project ``project``'s client: configure ITS project, insert a node-mode run
    on the SHARED broker, expand the DAG → one queued cpu node-job tagged project."""
    queue_workflows.configure(project=project)
    rid = str(uuid.uuid4())
    run_store.insert_run(run_id=rid, workflow_name="_3p_wf", out_dir=None,
                         status="queued", mode="node")  # project from config
    assert dispatcher.start_run(rid) == 1
    return rid


def _client_drain_queue(project: str) -> int:
    """Project ``project``'s claim worker: claim + EXECUTE only its own cpu jobs
    until the queue is empty (for it). Returns how many it ran."""
    queue_workflows.configure(project=project)
    worker = ClaimWorker(queue="cpu", host=f"{project}-cpu")
    n = 0
    while worker.run_once():
        n += 1
    return n


def test_three_projects_one_broker_full_lifecycle():
    _wire_client_hooks()
    runs = {p: _client_enqueue_run(p) for p in PROJECTS}

    # 1) ONE shared table / one cpu queue holds all 3 projects' jobs.
    with connection() as c, c.cursor() as cur:
        cur.execute("SELECT project, queue, COUNT(*) AS n FROM workflow_node_jobs "
                    "GROUP BY project, queue ORDER BY project")
        rows = cur.fetchall()
    assert {r["project"] for r in rows} == set(PROJECTS)
    assert all(r["queue"] == "cpu" and r["n"] == 1 for r in rows)

    # 2) ISOLATION: ai_leads' worker runs ONLY ai_leads' job; alpha &
    #    beta jobs stay queued (the worker never claimed them).
    assert _client_drain_queue("ai_leads") == 1
    assert _RAN == ["ai_leads"]                                   # only ai_leads ran
    by_project = {p: node_queue.list_jobs_for_run(runs[p])[0] for p in PROJECTS}
    assert by_project["ai_leads"]["status"] == "completed"
    assert by_project["ai_leads"]["project"] == "ai_leads"
    assert by_project["alpha"]["status"] == "queued"          # untouched
    assert by_project["beta"]["status"] == "queued"

    # 3) the other two projects' workers each run their own job.
    assert _client_drain_queue("alpha") == 1
    assert _client_drain_queue("beta") == 1
    assert set(_RAN) == set(PROJECTS)                             # all 3 executed
    for p in PROJECTS:
        job = node_queue.list_jobs_for_run(runs[p])[0]
        assert job["status"] == "completed" and job["project"] == p

    # 4) each project's orchestrator drains ITS outbox → ITS run completes.
    for p in PROJECTS:
        queue_workflows.configure(project=p)
        node_pool.NodePool(register_builtins=None)._drain_dispatch_events()
        assert run_store.get_run(runs[p])["status"] == "completed"

    # 5) broker-wide consolidated view sees all 3 on the one queue.
    assert set(node_queue.list_projects()) >= set(PROJECTS)
    assert node_queue.snapshot()["counts"].get("cpu_completed") >= 3  # None = broker-wide
    for p in PROJECTS:
        assert node_queue.snapshot(project=p)["counts"].get("cpu_completed") == 1


@pytest.mark.pg_only
def test_concurrent_cross_project_claims_no_theft():
    """A genuinely CONCURRENT multi-project claim race on the shared queue — the
    serialized lifecycle test can't exhibit this. N threads per project,
    barrier-released, each claiming ITS OWN project's cpu jobs. Asserts zero
    double-claims and zero cross-project claims: a permanent regression guard on
    the claim WHERE-clause (the sole cross-tenant isolation boundary)."""
    PER, THREADS = 12, 3
    for p in PROJECTS:
        queue_workflows.configure(project=p)
        for i in range(PER):
            rid = str(uuid.uuid4())
            run_store.insert_run(run_id=rid, workflow_name="w", out_dir="/t",
                                 status="running", mode="node")
            node_queue.enqueue_node_job(run_id=rid, node_id=f"n{i}",
                                        node_module="m", queue="cpu")
    claimed: list[tuple[str, str]] = []
    lock = threading.Lock()
    barrier = threading.Barrier(len(PROJECTS) * THREADS)

    def worker(project: str) -> None:
        barrier.wait()  # release all threads together → real contention
        while True:
            job = node_queue.claim_next_cpu_job(0, host=f"{project}-w", project=project)
            if job is None:
                break
            assert job["project"] == project          # never another project's row
            with lock:
                claimed.append((project, job["id"]))

    threads = [threading.Thread(target=worker, args=(p,))
               for p in PROJECTS for _ in range(THREADS)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    ids = [j for _, j in claimed]
    assert len(ids) == len(set(ids))                  # NO double-claim
    assert len(ids) == PER * len(PROJECTS)            # all claimed, none lost
    assert all(row(j)["project"] == p for p, j in claimed)  # NO cross-project claim


def test_reclaim_preserves_project_tag_on_shared_broker():
    """`reclaim_expired_leases` is INTENTIONALLY broker-wide — so it MUST leave a
    re-queued orphan's `project` tag intact, else a recovered job could be claimed
    by the wrong tenant. With 2 projects on the shared broker: reclaim A's expired
    orphan, assert its tag survives, and that B's worker claims only B's job and
    NOT the re-queued A orphan."""
    queue_workflows.configure(project="ai_leads")
    a_job = node_queue.enqueue_node_job(run_id=make_run(), node_id="a",
                                        node_module="m", queue="cpu")
    force_lease(a_job, expires_in_s=-30)              # running + expired lease
    queue_workflows.configure(project="alpha")
    b_job = node_queue.enqueue_node_job(run_id=make_run(), node_id="b",
                                        node_module="m", queue="cpu")

    reclaimed = [r["id"] for r in node_queue.reclaim_expired_leases()]  # broker-wide
    assert a_job in reclaimed
    assert row(a_job)["status"] == "queued"
    assert row(a_job)["project"] == "ai_leads"        # TAG PRESERVED

    # alpha's worker claims ONLY its own job, never the re-queued ai_leads orphan
    got = node_queue.claim_next_cpu_job(0, host="alpha-w", project="alpha")
    assert got["id"] == b_job
    assert node_queue.claim_next_cpu_job(0, host="alpha-w", project="alpha") is None
    assert row(a_job)["status"] == "queued"           # still waiting for ai_leads


def test_no_cross_project_claim_on_shared_queue():
    """Tighter isolation lock: with 3 projects' jobs live on the one queue, a
    worker configured as project X claims X's job and NEVER another's — even when
    X's queue is then empty (it parks, it does not steal)."""
    _wire_client_hooks()
    runs = {p: _client_enqueue_run(p) for p in PROJECTS}
    # a worker for a project with NO jobs claims nothing (no cross-tenant theft).
    queue_workflows.configure(project="beta")
    w = ClaimWorker(queue="cpu", host="lm-cpu")
    assert w.run_once() is True                       # its own job
    assert w.run_once() is False                      # empty for it now
    # ai_leads & alpha jobs are still queued — lm's worker never touched them.
    assert node_queue.list_jobs_for_run(runs["ai_leads"])[0]["status"] == "queued"
    assert node_queue.list_jobs_for_run(runs["alpha"])[0]["status"] == "queued"

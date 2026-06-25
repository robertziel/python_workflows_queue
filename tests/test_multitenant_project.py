"""Multi-tenant broker: project-scoped queue (migration 0017).

The engine runs ONE cpu + ONE gpu (+ ingest) queue on a SHARED broker Postgres;
every queue record carries a ``project`` tenant tag, and a per-project client
enqueues + claims ONLY its own project's rows. These tests pin that contract on
the live claim path:

- a client claims only its own project's node-jobs (cpu + gpu) / ingest jobs,
  never another project's — two projects' jobs coexisting on the same queue;
- the default empty project (``""``) round-trips, so a single-Postgres-per-
  project deploy is byte-compatible (pre-0017 behaviour);
- ``config.project`` is the implicit tag for enqueue AND claim, so one
  ``configure(project=...)`` makes a client self-consistent;
- two projects' workers on the SAME ``(host_label, queue)`` keep distinct
  heartbeats (the PK now includes project) and ``fleet_snapshot`` filters by it.
"""

from __future__ import annotations

import uuid

import queue_workflows
from queue_workflows import node_queue, run_store
from queue_workflows.db import connection


def _run(project: str, *, status: str = "running") -> str:
    """Insert a parent run tagged with ``project`` (status running so the claim
    run-cancel guard is satisfied)."""
    rid = str(uuid.uuid4())
    run_store.insert_run(
        run_id=rid, workflow_name="_mt_wf", out_dir="/tmp/out",
        status=status, mode="node", project=project,
    )
    return rid


# ── cpu claim isolation ────────────────────────────────────────────────────


def test_cpu_claim_is_scoped_to_project():
    a = node_queue.enqueue_node_job(
        run_id=_run("alpha"), node_id="a", node_module="x", queue="cpu",
        project="alpha",
    )
    b = node_queue.enqueue_node_job(
        run_id=_run("beta"), node_id="b", node_module="x", queue="cpu",
        project="beta",
    )

    # A client on project=beta never sees alpha's job, even though both sit on
    # the SAME cpu queue in one DB.
    first_beta = node_queue.claim_next_cpu_job(host="h", project="beta")
    assert first_beta is not None and first_beta["id"] == b
    assert node_queue.claim_next_cpu_job(host="h", project="beta") is None

    # alpha's job is still claimable by an alpha client.
    first_alpha = node_queue.claim_next_cpu_job(host="h", project="alpha")
    assert first_alpha is not None and first_alpha["id"] == a


# ── gpu claim isolation ────────────────────────────────────────────────────


def test_gpu_claim_is_scoped_to_project():
    a = node_queue.enqueue_node_job(
        run_id=_run("alpha"), node_id="g", node_module="x", queue="gpu",
        required_model="sdxl", project="alpha",
    )
    b = node_queue.enqueue_node_job(
        run_id=_run("beta"), node_id="g", node_module="x", queue="gpu",
        required_model="sdxl", project="beta",
    )

    claimed = node_queue.claim_next_gpu_job(
        0, current_model="sdxl", host="h", known_models=["sdxl"], project="alpha",
    )
    assert claimed is not None and claimed["id"] == a
    # beta's row is untouched and only an beta client can take it.
    assert node_queue.claim_next_gpu_job(
        0, current_model="sdxl", host="h", known_models=["sdxl"], project="alpha",
    ) is None
    claimed_b = node_queue.claim_next_gpu_job(
        0, current_model="sdxl", host="h", known_models=["sdxl"], project="beta",
    )
    assert claimed_b is not None and claimed_b["id"] == b


# ── ingest claim isolation ─────────────────────────────────────────────────


def test_ingest_claim_is_scoped_to_project():
    queue_workflows.register_ingest_task("noop", lambda reason: {})
    a = node_queue.enqueue_ingest_job(task_name="noop", queue="fetch", project="alpha")
    b = node_queue.enqueue_ingest_job(task_name="noop", queue="fetch", project="beta")

    got = node_queue.claim_next_ingest_job("fetch", host="h", project="alpha")
    assert got is not None and got["id"] == a
    assert node_queue.claim_next_ingest_job("fetch", host="h", project="alpha") is None
    got_b = node_queue.claim_next_ingest_job("fetch", host="h", project="beta")
    assert got_b is not None and got_b["id"] == b


# ── back-compat: empty project (single-tenant) round-trips ──────────────────


def test_empty_project_is_single_tenant_backcompat():
    # No project set anywhere → everything is the "" sentinel and claims work
    # exactly as pre-0017 (one Postgres per project, no host wiring).
    jid = node_queue.enqueue_node_job(
        run_id=_run(""), node_id="a", node_module="x", queue="cpu",
    )
    got = node_queue.claim_next_cpu_job(host="h")
    assert got is not None and got["id"] == jid and got["project"] == ""


def test_config_project_is_implicit_tag_for_enqueue_and_claim():
    # One configure(project=...) makes a client self-consistent: it stamps the
    # tag on enqueue and filters by it on claim without passing project= each call.
    queue_workflows.configure(project="gamma")
    jid = node_queue.enqueue_node_job(
        run_id=_run("gamma"), node_id="a", node_module="x", queue="cpu",
    )
    assert node_queue.get_node_job(jid)["project"] == "gamma"
    # A different project's client can't take it; the gamma client can.
    assert node_queue.claim_next_cpu_job(host="h", project="other") is None
    got = node_queue.claim_next_cpu_job(host="h")  # defaults to config.project
    assert got is not None and got["id"] == jid


# ── heartbeat identity now includes project ─────────────────────────────────


def test_two_projects_share_host_queue_without_heartbeat_clobber():
    # The shared broker can run two projects' gpu clients on the SAME machine.
    node_queue.upsert_worker_heartbeat(
        host_label="host-a", queue="gpu", project="alpha", concurrency=1,
    )
    node_queue.upsert_worker_heartbeat(
        host_label="host-a", queue="gpu", project="beta", concurrency=2,
    )
    with connection() as c, c.cursor() as cur:
        cur.execute(
            "SELECT project, concurrency FROM worker_heartbeats "
            "WHERE host_label='host-a' AND queue='gpu' ORDER BY project"
        )
        rows = cur.fetchall()
    # Two distinct rows — neither clobbered the other (old PK would have).
    assert [(r["project"], r["concurrency"]) for r in rows] == [
        ("alpha", 1), ("beta", 2),
    ]

    # fleet_snapshot filters by project; broker-wide sees both.
    assert {r["project"] for r in node_queue.fleet_snapshot()} >= {"alpha", "beta"}
    only_alpha = node_queue.fleet_snapshot(project="alpha")
    assert only_alpha and all(r["project"] == "alpha" for r in only_alpha)


# ── snapshot filtering ──────────────────────────────────────────────────────


def test_snapshot_counts_filter_by_project():
    node_queue.enqueue_node_job(
        run_id=_run("alpha"), node_id="a", node_module="x", queue="cpu",
        project="alpha",
    )
    node_queue.enqueue_node_job(
        run_id=_run("beta"), node_id="b", node_module="x", queue="cpu",
        project="beta",
    )
    snap = node_queue.snapshot(project="alpha")
    assert snap["counts"].get("cpu_queued") == 1  # only alpha's row counted
    # broker-wide sees both.
    assert node_queue.snapshot()["counts"].get("cpu_queued") == 2


# ── orchestrator row-pickup paths are project-scoped too (audit fixes) ───────


def test_dispatch_run_selection_is_project_scoped():
    # NodePool._tick must expand ONLY its own project's queued runs — else a
    # shared-broker orchestrator expands another project's run under its own DAG.
    ra = _run("alpha", status="queued")
    rb = _run("beta", status="queued")
    assert run_store.list_queued_node_run_ids(project="alpha") == [ra]
    assert run_store.list_queued_node_run_ids(project="beta") == [rb]
    assert set(run_store.list_queued_node_run_ids(project="")) == set()  # neither is ""


def test_unassignable_sweep_is_project_scoped():
    # beta's gpu worker can't hold 'big'; alpha's CAN. beta's queued model-job
    # must be flagged unassignable by beta's sweep — NOT masked by alpha's worker
    # (which, exact-match, can never claim beta's job).
    node_queue.upsert_worker_heartbeat(
        host_label="a", queue="gpu", project="alpha", concurrency=1,
        fits_models=["big"], vram_total_mb=80000,
    )
    node_queue.upsert_worker_heartbeat(
        host_label="b", queue="gpu", project="beta", concurrency=1,
        fits_models=[], vram_total_mb=8000,
    )
    jid = node_queue.enqueue_node_job(
        run_id=_run("beta"), node_id="g", node_module="x", queue="gpu",
        required_model="big", project="beta",
    )
    flagged = node_queue.flag_unassignable_gpu_jobs(project="beta")
    assert [r["id"] for r in flagged] == [jid]
    # alpha's sweep must neither flag nor clear beta's job.
    assert node_queue.flag_unassignable_gpu_jobs(project="alpha") == []
    assert node_queue.get_node_job(jid)["unassignable_at"] is not None


def test_input_listener_claim_is_project_scoped():
    import json
    from queue_workflows import input_listener

    ra = _run("alpha", status="awaiting_input")
    rb = _run("beta", status="awaiting_input")

    def _sub(run_id: str) -> str:
        sid = str(uuid.uuid4())
        with connection() as c, c.cursor() as cur:
            cur.execute(
                "INSERT INTO workflow_input_submissions "
                "(id, run_id, node_id, value, status) "
                "VALUES (%s, %s, 'ask', %s::jsonb, 'pending')",
                (sid, run_id, json.dumps("v")),
            )
            c.commit()
        return sid

    sa, sb = _sub(ra), _sub(rb)
    queue_workflows.configure(project="alpha")
    claimed = {r["id"] for r in input_listener.InputListener._claim_pending()}
    assert sa in claimed and sb not in claimed


def test_requeue_running_for_worker_is_project_scoped():
    # Two projects' workers share host_label 'host-a' + gpu, one running job each.
    a = node_queue.enqueue_node_job(
        run_id=_run("alpha"), node_id="g", node_module="x", queue="gpu",
        required_model="m", project="alpha",
    )
    b = node_queue.enqueue_node_job(
        run_id=_run("beta"), node_id="g", node_module="x", queue="gpu",
        required_model="m", project="beta",
    )
    with connection() as c, c.cursor() as cur:
        cur.execute(
            "UPDATE workflow_node_jobs SET status='running', claimed_by='host-a', "
            "lease_expires_at = now() + interval '600 seconds' WHERE id IN (%s, %s)",
            (a, b),
        )
        c.commit()
    # Hard-stop alpha's worker on host-a/gpu re-queues ONLY alpha's running job.
    n = node_queue.requeue_running_for_worker("host-a", "gpu", project="alpha")
    assert n == 1
    assert node_queue.get_node_job(a)["status"] == "queued"
    assert node_queue.get_node_job(b)["status"] == "running"  # beta untouched


# ── orchestrator startup + periodic recovery paths are scoped (audit round 2) ─


def _running_job(project: str, claimed_by: str = "hostX") -> str:
    """A running node-job tagged `project`, claimed_by `claimed_by`, no heartbeat
    (so the resume-reclaim treats it as orphaned)."""
    jid = node_queue.enqueue_node_job(
        run_id=_run(project), node_id="g", node_module="x", queue="gpu",
        required_model="m", project=project,
    )
    with connection() as c, c.cursor() as cur:
        cur.execute(
            "UPDATE workflow_node_jobs SET status='running', claimed_by=%s, "
            "lease_expires_at = now() + interval '600 seconds' WHERE id=%s",
            (claimed_by, jid),
        )
        c.commit()
    return jid


def test_reenqueue_running_for_resume_is_project_scoped():
    # Orchestrator-startup run resume must touch ONLY its own project's runs.
    ra = _run("alpha", status="running")
    rb = _run("beta", status="running")
    n = run_store.reenqueue_running_for_resume(project="alpha")
    assert n == 1
    assert run_store.get_run(ra)["status"] == "queued"
    assert run_store.get_run(rb)["status"] == "running"  # beta untouched


def test_reclaim_all_running_for_resume_is_project_scoped():
    # The outer UPDATE (not just the heartbeat sub-join) must be project-scoped,
    # else A's restart clears B's claimed_by and trips B's live worker.
    a = _running_job("alpha")
    b = _running_job("beta")
    reclaimed = node_queue.reclaim_all_running_for_resume(project="alpha")
    assert {r["id"] for r in reclaimed} == {a}
    assert node_queue.get_node_job(a)["status"] == "queued"
    assert node_queue.get_node_job(a)["claimed_by"] is None
    assert node_queue.get_node_job(b)["status"] == "running"   # beta's render safe
    assert node_queue.get_node_job(b)["claimed_by"] == "hostX"


def test_stuck_run_selection_is_project_scoped():
    # Phantom runs (queued/running, no live node-job) reconciled only per-project.
    ra = _run("alpha", status="running")
    rb = _run("beta", status="running")
    assert run_store.list_stuck_node_run_ids(project="alpha") == [ra]
    assert run_store.list_stuck_node_run_ids(project="beta") == [rb]


def test_dispatch_outbox_drain_is_project_scoped(monkeypatch):
    from queue_workflows import dispatcher, node_pool

    ra = _run("alpha")
    rb = _run("beta")
    with connection() as c, c.cursor() as cur:
        node_queue.enqueue_dispatch_event_in_txn(cur, ra, "n", "completed")
        node_queue.enqueue_dispatch_event_in_txn(cur, rb, "n", "completed")
        c.commit()

    seen: list[str] = []
    monkeypatch.setattr(dispatcher, "on_node_completed", lambda rid, nid: seen.append(rid))

    queue_workflows.configure(project="alpha")
    node_pool.NodePool(register_builtins=None)._drain_dispatch_events()

    assert seen == [ra]  # only alpha's event drained; beta's left for beta's orchestrator
    unprocessed = {e["run_id"] for e in node_queue.list_unprocessed_dispatch_events()}
    assert rb in unprocessed and ra not in unprocessed


def test_cancel_orphaned_queued_jobs_is_project_scoped():
    # queued jobs of an already-terminal run, cancelled only within the project.
    ra, rb = _run("alpha", status="failed"), _run("beta", status="failed")
    node_queue.enqueue_node_job(run_id=ra, node_id="q", node_module="x", queue="cpu", project="alpha")
    node_queue.enqueue_node_job(run_id=rb, node_id="q", node_module="x", queue="cpu", project="beta")
    n = node_queue.cancel_orphaned_queued_jobs(project="alpha")
    assert n == 1


def test_flag_stale_workers_is_project_scoped():
    # Each orchestrator flags only its own project's dead workers.
    for proj in ("alpha", "beta"):
        jid = _running_job(proj, claimed_by="boxshared")
        node_queue.upsert_worker_heartbeat(
            host_label="boxshared", queue="gpu", project=proj, concurrency=1,
        )
        # age the heartbeat out of the fresh window
        with connection() as c, c.cursor() as cur:
            cur.execute(
                "UPDATE worker_heartbeats SET last_seen = now() - interval '120 seconds' "
                "WHERE host_label='boxshared' AND queue='gpu' AND project=%s",
                (proj,),
            )
            c.commit()
    flagged = node_queue.flag_stale_workers_holding_running_jobs(
        stale_after_s=30, project="alpha",
    )
    assert [r["project"] for r in flagged] == ["alpha"]

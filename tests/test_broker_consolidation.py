"""Broker consolidation — PROOF that one broker holds one queue for all projects.

The headline of the new model ("one queue per cpu/gpu, NOT per project; keep the
project name in the record for filtering; the queue is on the broker side"). This
drives the WHOLE thing against ONE database (the conftest test DB == the broker)
and asserts:

  * two projects' jobs coexist on the SAME workflow_node_jobs table / one shared
    cpu + gpu queue (not a queue per project);
  * each project's client claims ONLY its own project's rows (isolation on the
    shared broker);
  * the broker-wide snapshot shows BOTH projects; a per-project snapshot shows
    only that project; list_projects() enumerates the tenants;
  * the ``queue-broker`` console bootstraps + reports the consolidated view.

Runs on both backends (Postgres + SQLite) — pure engine primitives.
"""

from __future__ import annotations

import os
import threading
import urllib.parse
import uuid

import pytest

import queue_workflows
from queue_workflows import broker, db, node_queue, run_store
from queue_workflows.db import connection


def _client_enqueue(project: str, *, cpu: int = 0, gpu: int = 0) -> list[str]:
    """Simulate project ``project``'s client: it configures ITS project once and
    enqueues onto the SHARED broker (jobs get tagged with its project)."""
    queue_workflows.configure(project=project)
    ids = []
    for i in range(cpu):
        rid = str(uuid.uuid4())
        run_store.insert_run(run_id=rid, workflow_name="w", out_dir="/t",
                             status="running", mode="node")  # project from config
        ids.append(node_queue.enqueue_node_job(
            run_id=rid, node_id=f"c{i}", node_module="m", queue="cpu"))
    for i in range(gpu):
        rid = str(uuid.uuid4())
        run_store.insert_run(run_id=rid, workflow_name="w", out_dir="/t",
                             status="running", mode="node")
        ids.append(node_queue.enqueue_node_job(
            run_id=rid, node_id=f"g{i}", node_module="m", queue="gpu",
            required_model="sdxl"))
    return ids


def _drain(project: str, queue: str) -> list[str]:
    """Simulate project ``project``'s claim worker draining ``queue`` — it claims
    ONLY rows tagged with its project."""
    claim = (node_queue.claim_next_cpu_job if queue == "cpu"
             else lambda **k: node_queue.claim_next_gpu_job(0, "sdxl", **k))
    out = []
    while True:
        job = claim(host=f"{project}-worker", project=project)
        if job is None:
            break
        assert job["project"] == project          # never another project's row
        out.append(job["id"])
    return out


def test_two_projects_share_one_broker_one_queue():
    a = set(_client_enqueue("ai_leads", cpu=3, gpu=2))
    b = set(_client_enqueue("pic_to_3d", cpu=1, gpu=1))

    # ONE shared table holds BOTH projects' jobs on the shared cpu/gpu queue.
    with connection() as c, c.cursor() as cur:
        cur.execute("SELECT COUNT(*) AS n FROM workflow_node_jobs")
        assert cur.fetchone()["n"] == 7          # ai_leads 3+2 + pic_to_3d 1+1
        cur.execute("SELECT DISTINCT project FROM workflow_node_jobs ORDER BY project")
        assert [r["project"] for r in cur.fetchall()] == ["ai_leads", "pic_to_3d"]

    # broker-wide view sees BOTH; per-project scopes to one.
    assert node_queue.snapshot()["counts"].get("cpu_queued") == 4   # 3 + 1
    assert node_queue.snapshot(project="ai_leads")["counts"].get("cpu_queued") == 3
    assert node_queue.snapshot(project="pic_to_3d")["counts"].get("cpu_queued") == 1
    assert set(node_queue.list_projects()) >= {"ai_leads", "pic_to_3d"}

    # each project's worker claims ONLY its own rows off the shared queue.
    a_cpu = set(_drain("ai_leads", "cpu"))
    b_cpu = set(_drain("pic_to_3d", "cpu"))
    assert a_cpu == {i for i in a if node_queue.get_node_job(i)["queue"] == "cpu"}
    assert b_cpu == {i for i in b if node_queue.get_node_job(i)["queue"] == "cpu"}
    assert a_cpu.isdisjoint(b_cpu)               # no cross-tenant claim
    # ai_leads draining gpu never grabs pic_to_3d's gpu row, and vice versa
    a_gpu = set(_drain("ai_leads", "gpu"))
    assert all(node_queue.get_node_job(i)["project"] == "ai_leads" for i in a_gpu)


def test_queue_broker_bootstraps_and_reports(capsys):
    _client_enqueue("ai_leads", cpu=2)
    _client_enqueue("pic_to_3d", gpu=1)
    # default: bootstrap (idempotent on the already-migrated broker) + status
    rc = broker.main([])
    assert rc == 0
    out = capsys.readouterr().out
    assert "broker bootstrapped" in out
    assert "ai_leads" in out and "pic_to_3d" in out      # consolidated view
    # --status only
    assert broker.main(["--status"]) == 0
    out2 = capsys.readouterr().out
    assert "broker schema version" in out2 and "projects on this broker" in out2


@pytest.mark.pg_only
def test_concurrent_bootstrap_on_shared_broker_is_safe():
    """The recipe points EVERY project's orchestrator at one broker; each calls
    db.bootstrap() on boot. A Postgres advisory lock must make concurrent
    bootstraps on the SAME DB safe — no UniqueViolation on the version-ledger PK
    and no DuplicateColumn from a double-applied migration. Proven on a FRESH DB
    raced from scratch by N threads (so it never touches the shared test DB)."""
    parsed = urllib.parse.urlparse(db.db_url())
    maint = urllib.parse.urlunparse(parsed._replace(path="/postgres"))
    fresh = "qw_concurrent_boot_test"
    import psycopg
    with psycopg.connect(maint, autocommit=True) as c:
        c.execute(f"DROP DATABASE IF EXISTS {fresh}")
        c.execute(f"CREATE DATABASE {fresh}")
    os.environ["QW_CONCURRENT_BOOT_DSN"] = urllib.parse.urlunparse(
        parsed._replace(path="/" + fresh))
    db.close_pool()
    queue_workflows.configure(db_url_env="QW_CONCURRENT_BOOT_DSN")
    errors: list = []
    barrier = threading.Barrier(5)

    def boot():
        try:
            barrier.wait()
            db.bootstrap()
        except Exception as e:  # noqa: BLE001 — collect any race error
            errors.append(repr(e))

    threads = [threading.Thread(target=boot) for _ in range(5)]
    try:
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)
        assert errors == [], f"concurrent bootstrap raced: {errors}"
        assert db.current_schema_version() >= 17     # applied the full chain once
    finally:
        db.close_pool()
        queue_workflows.configure(db_url_env="QUEUE_WORKFLOWS_TEST_DB_URL")
        with psycopg.connect(maint, autocommit=True) as c:
            c.execute(f"DROP DATABASE IF EXISTS {fresh}")

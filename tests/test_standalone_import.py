"""Standalone gate (plan acceptance §7) — the engine runs with only psycopg +
Postgres, no host on the path.

``import queue_workflows; queue_workflows.configure(...)`` succeeds, all the
registration hooks are callable, the safe defaults are in place, and a real
round-trip against the engine's own schema works (insert a run via run_store,
enqueue a node job, claim it, mark it complete, drain a dispatch event).

These exercise the engine end-to-end WITHOUT any ai_leads node module / model
spec — the workflow provider + a fake node module are supplied in-test, keeping
the engine suite domain-free.
"""

from __future__ import annotations

import sys
import types
import uuid

import pytest

import queue_workflows
from queue_workflows import dispatcher, node_pool, node_queue, run_store


# ── import + configure surface ─────────────────────────────────────────────


def test_import_and_configure_with_safe_defaults():
    cfg = queue_workflows.configure()  # no args — all defaults
    # Defaults keep ai_leads byte-compat env names.
    assert cfg.db_url_env  # set by conftest to the test env
    assert cfg.container_prefix == "ai_leads-"
    # Safe defaults: empty video set, no node package, no-op registrar.
    assert isinstance(cfg.video_model_ids, frozenset)
    # builtin_model_registrar default is a no-op (callable, returns None).
    assert cfg.builtin_model_registrar() is None
    # resolve_ref defaults to the engine's own.
    from queue_workflows.refs import resolve_ref
    assert cfg.get_resolve_ref() is resolve_ref


def test_all_registration_hooks_callable():
    queue_workflows.set_node_module_package("pkg.nodes")
    assert queue_workflows.get_config().node_module_package == "pkg.nodes"

    queue_workflows.set_node_resolver(lambda name: None)
    assert queue_workflows.get_config().node_resolver is not None

    reg = lambda: None
    queue_workflows.set_builtin_model_registrar(reg)
    assert queue_workflows.get_config().builtin_model_registrar is reg

    queue_workflows.set_workflow_provider(lambda n: {}, lambda n: {})
    assert queue_workflows.get_config().workflow_loader is not None
    assert queue_workflows.get_config().pipeline_schema_loader is not None

    queue_workflows.register_ingest_task("t1", lambda reason: {"ok": True})
    assert "t1" in queue_workflows.get_config().ingest_task_map

    from queue_workflows.scheduler import ScheduleEntry
    queue_workflows.set_ingest_schedule(
        [ScheduleEntry("e", 5, "t1", "fetch")]
    )
    assert len(queue_workflows.get_config().ingest_schedule) == 1


def test_migrations_dir_ships_the_sql():
    from queue_workflows import migrations
    d = migrations.dir()
    names = sorted(p.name for p in d.glob("*.sql") if not p.name.endswith(".down.sql"))
    assert names == [
        "0001_queue_runs.sql",
        "0002_node_jobs.sql",
        "0003_input_submissions.sql",
        "0004_dispatch_events.sql",
        "0005_worker_heartbeats.sql",
        "0006_pg_queue_lease.sql",
        "0007_ingest_jobs.sql",
    ]


# ── a real end-to-end round-trip against the engine schema ──────────────────


def _install_fake_node(name: str, run_fn):
    """A tiny in-test node module so the engine suite stays domain-free."""
    mod = types.ModuleType(f"qwf_fake_nodes.{name}")
    mod.run = run_fn
    sys.modules[f"qwf_fake_nodes.{name}"] = mod


def _fake_workflow_provider():
    """A one-pipeline workflow whose single node runs the fake module."""
    workflows = {
        "_standalone_wf": {
            "name": "_standalone_wf",
            "steps": [{"id": "p", "kind": "pipeline", "pipeline": "_standalone_pipe"}],
        }
    }
    pipelines = {
        "_standalone_pipe": {
            "name": "_standalone_pipe",
            "nodes": [{"id": "n1", "node": "echo_node"}],
        }
    }
    queue_workflows.set_workflow_provider(
        lambda name: workflows[name],
        lambda name: pipelines[name],
    )


def test_end_to_end_run_through_dispatcher_and_node_pool():
    # Wire the node-module resolver at a fake package + a fake node.
    queue_workflows.set_node_module_package("qwf_fake_nodes")
    ran: list[str] = []

    def echo_run(inputs: dict, out=None):
        ran.append("echo")
        return {"context_delta": {"echoed": True}}

    _install_fake_node("echo_node", echo_run)
    _fake_workflow_provider()

    # 1) Insert a node-mode run via the engine's run_store (no parcels).
    run_id = str(uuid.uuid4())
    run_store.insert_run(
        run_id=run_id, workflow_name="_standalone_wf",
        out_dir=None, status="queued", mode="node",
    )

    # 2) Expand the DAG (dispatcher.start_run) → one queued node-job.
    n = dispatcher.start_run(run_id)
    assert n == 1
    jobs = node_queue.list_jobs_for_run(run_id)
    assert len(jobs) == 1
    assert jobs[0]["status"] == "queued"
    assert jobs[0]["node_module"] == "echo_node"

    # 3) Claim + execute it via the claim worker's run_once (cpu).
    from queue_workflows.claim_worker import ClaimWorker
    worker = ClaimWorker(queue="cpu", host="standalone-test")
    assert worker.run_once() is True
    assert ran == ["echo"]

    job = node_queue.list_jobs_for_run(run_id)[0]
    assert job["status"] == "completed"
    assert job["context_delta"] == {"echoed": True}

    # 4) A completed-event landed in the outbox; draining it completes the run.
    events = node_queue.list_unprocessed_dispatch_events()
    assert any(e["kind"] == "completed" for e in events)
    pool = node_pool.NodePool(register_builtins=None)
    pool._drain_dispatch_events()
    assert run_store.get_run(run_id)["status"] == "completed"

"""``queue-conductor`` — the operator-facing READ side of the GO-half conductor:
a single-DB fleet capacity view rendering ``node_queue.fleet_snapshot()``.

Read-only by design; worker ON/OFF control stays in ``queue-worker-control`` (no
duplication). Single-DB like every other console script — the operator points it
at one DSN via the engine's ``db_url_env`` env; the networked multi-DB daemon +
web UI that would wrap this are a separate, human-gated build.
"""

from __future__ import annotations

import json

from queue_workflows import node_queue
from queue_workflows_conductor import conductor


# ── pure formatter (no DB) ────────────────────────────────────────────────────


def test_render_empty_says_no_workers() -> None:
    out = conductor.render_fleet([], as_json=False)
    assert "no workers" in out.lower()


def test_render_json_roundtrips() -> None:
    rows = [
        {
            "host_label": "box-a", "queue": "gpu", "current_model": "qwen",
            "fresh": True, "flagged_dead": False, "vram_total_mb": 24000,
            "llm_servers_available": ["ollama", "vllm"],
        }
    ]
    parsed = json.loads(conductor.render_fleet(rows, as_json=True))
    assert parsed[0]["host_label"] == "box-a"
    assert parsed[0]["queue"] == "gpu"


def test_render_table_marks_dead_and_stale() -> None:
    rows = [
        {"host_label": "wedged", "queue": "gpu", "current_model": None,
         "fresh": False, "flagged_dead": True, "vram_total_mb": None,
         "llm_servers_available": []},
        {"host_label": "slow", "queue": "cpu", "current_model": None,
         "fresh": False, "flagged_dead": False, "vram_total_mb": None,
         "llm_servers_available": []},
    ]
    out = conductor.render_fleet(rows, as_json=False)
    assert "wedged" in out and "DEAD" in out
    assert "slow" in out and "stale" in out


# ── main() against the test DB (conftest points connection() at it) ───────────


def test_main_prints_seeded_worker(capsys) -> None:
    node_queue.upsert_worker_heartbeat(
        host_label="box-live", queue="gpu", concurrency=1, current_model="qwen",
    )
    rc = conductor.main([])
    assert rc == 0
    assert "box-live" in capsys.readouterr().out


def test_main_json_flag_emits_list(capsys) -> None:
    node_queue.upsert_worker_heartbeat(host_label="box-json", queue="gpu", concurrency=1)
    rc = conductor.main(["--json"])
    assert rc == 0
    data = json.loads(capsys.readouterr().out)
    assert any(r["host_label"] == "box-json" for r in data)


def test_main_queue_filter(capsys) -> None:
    node_queue.upsert_worker_heartbeat(host_label="g", queue="gpu", concurrency=1)
    node_queue.upsert_worker_heartbeat(host_label="c", queue="cpu", concurrency=1)
    conductor.main(["--queue", "gpu"])
    out = capsys.readouterr().out
    assert "g" in out and "cpu" not in out


def test_cli_entrypoint_releases_pool(capsys) -> None:
    # The console entry closes the pool for a clean exit; the pool re-inits
    # lazily, so engine calls after it still work (no broken global state).
    node_queue.upsert_worker_heartbeat(host_label="cli-box", queue="gpu", concurrency=1)
    assert conductor.cli([]) == 0
    assert "cli-box" in capsys.readouterr().out
    # pool was dropped, but the next engine call transparently re-opens it.
    assert any(r["host_label"] == "cli-box" for r in node_queue.fleet_snapshot())

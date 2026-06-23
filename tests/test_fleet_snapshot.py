"""``node_queue.fleet_snapshot`` — the read-only per-(host,queue) fleet capacity
view (the observed ``worker_heartbeats`` rows, with their capability columns).

This is the telemetry read model a fleet view / conductor consumes: the existing
``snapshot``/``ingest_snapshot`` only *count* heartbeats or join them internally
for claim decisions — neither returns the per-worker rows. The view must surface
STALE and dead-flagged workers too (that's the point of an observability read),
so it returns ALL rows with derived ``fresh`` / ``flagged_dead`` flags rather
than filtering.
"""

from __future__ import annotations

from queue_workflows import node_queue
from queue_workflows.db import connection


def _by_host(rows: list[dict], host: str, queue: str) -> dict:
    [row] = [r for r in rows if r["host_label"] == host and r["queue"] == queue]
    return row


def test_empty_fleet_is_empty_list() -> None:
    assert node_queue.fleet_snapshot() == []


def test_snapshot_returns_seeded_capacity_rows() -> None:
    node_queue.upsert_worker_heartbeat(
        host_label="box-a",
        queue="gpu",
        concurrency=1,
        current_model="qwen",
        known_models=["qwen", "sdxl"],
        llm_servers_available=["ollama", "vllm"],
        vram_total_mb=24000,
        fits_models=["qwen"],
    )
    node_queue.upsert_worker_heartbeat(
        host_label="box-b", queue="cpu", concurrency=4,
    )

    rows = node_queue.fleet_snapshot()
    assert {(r["host_label"], r["queue"]) for r in rows} == {
        ("box-a", "gpu"),
        ("box-b", "cpu"),
    }

    gpu = _by_host(rows, "box-a", "gpu")
    assert gpu["current_model"] == "qwen"
    assert "vllm" in gpu["llm_servers_available"]
    assert gpu["vram_total_mb"] == 24000
    assert gpu["fits_models"] == ["qwen"]
    assert gpu["fresh"] is True
    assert gpu["flagged_dead"] is False


def test_ordered_by_queue_then_host() -> None:
    node_queue.upsert_worker_heartbeat(host_label="z", queue="gpu", concurrency=1)
    node_queue.upsert_worker_heartbeat(host_label="a", queue="gpu", concurrency=1)
    node_queue.upsert_worker_heartbeat(host_label="m", queue="cpu", concurrency=1)
    rows = node_queue.fleet_snapshot()
    assert [(r["queue"], r["host_label"]) for r in rows] == [
        ("cpu", "m"),
        ("gpu", "a"),
        ("gpu", "z"),
    ]


def test_stale_worker_still_listed_but_not_fresh() -> None:
    node_queue.upsert_worker_heartbeat(host_label="old", queue="gpu", concurrency=1)
    # Backdate last_seen well past the freshness window.
    with connection() as conn, conn.cursor() as cur:
        cur.execute(
            "UPDATE worker_heartbeats SET last_seen = now() - interval '120 seconds' "
            "WHERE host_label = %s AND queue = %s",
            ("old", "gpu"),
        )
    rows = node_queue.fleet_snapshot(stale_after_s=30.0)
    row = _by_host(rows, "old", "gpu")
    assert row["fresh"] is False  # surfaced, but flagged not-fresh


def test_dead_flag_surfaced() -> None:
    node_queue.upsert_worker_heartbeat(host_label="wedged", queue="gpu", concurrency=1)
    with connection() as conn, conn.cursor() as cur:
        cur.execute(
            "UPDATE worker_heartbeats SET last_flagged_dead_at = now() "
            "WHERE host_label = %s AND queue = %s",
            ("wedged", "gpu"),
        )
    row = _by_host(node_queue.fleet_snapshot(), "wedged", "gpu")
    assert row["flagged_dead"] is True

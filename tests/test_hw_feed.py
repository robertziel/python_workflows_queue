"""Centralized hw-metrics: the broker metrics-DSN seam + the HwFeed reader.

The hw sampler publishes ``NOTIFY hw_metrics`` to the metrics DSN (the shared
broker); HwFeed is the reusable client-lib consumer every project imports so all
projects show the SAME fleet-wide hardware view. Runs against the test Postgres
(NOTIFY/LISTEN is Postgres-only, as hw-metrics always is).
"""

from __future__ import annotations

import time

import queue_workflows
from queue_workflows import hw_metrics
from queue_workflows.db import db_url
from queue_workflows.hw_feed import HwFeed


def test_metrics_dsn_resolution(monkeypatch):
    # default: metrics_db_url_env unset → falls back to the queue db_url_env
    queue_workflows.configure()
    assert hw_metrics._uses_dedicated_metrics_dsn() is False
    assert hw_metrics.metrics_dsn() == db_url()  # == the queue DSN
    # set a DISTINCT metrics env → dedicated broker target
    monkeypatch.setenv("BROKER_METRICS_DSN", "postgresql://broker/x")
    queue_workflows.configure(metrics_db_url_env="BROKER_METRICS_DSN")
    assert hw_metrics._uses_dedicated_metrics_dsn() is True
    assert hw_metrics.metrics_dsn() == "postgresql://broker/x"


def test_hwfeed_receives_broadcast():
    """End-to-end: a published hw sample lands in HwFeed.latest_by_host(),
    keyed by host, fresh (not stale)."""
    feed = HwFeed(stale_after_s=60, dsn=db_url()).start()
    try:
        hosts: dict = {}
        # broadcast repeatedly while polling — robust against the LISTEN-setup race
        for _ in range(40):
            hw_metrics._broadcast({"cpu": {"pct": 12}, "gpus": []})
            time.sleep(0.15)
            hosts = feed.latest_by_host()
            if hosts:
                break
        assert hosts, "HwFeed did not receive any hw_metrics NOTIFY"
        host = next(iter(hosts))
        assert hosts[host]["stale"] is False
        assert "cpu" in hosts[host] and "host" in hosts[host]
    finally:
        feed.stop()

"""Reusable consumer for the centralized hardware-metrics feed.

The hw sampler (:mod:`queue_workflows.hw_metrics`) fires ``NOTIFY hw_metrics``
on the shared **broker** (``config.metrics_db_url_env`` → :func:`hw_metrics.
metrics_dsn`). :class:`HwFeed` is the matching reader — it generalizes the
per-project ``hw_listener`` so **every project imports one wrapper** and shows
the SAME broker-sourced, fleet-wide hardware view instead of each project
sampling its own DB.

Design (mirrors the engine's other LISTEN consumers):

  * A single **daemon thread** on a **dedicated autocommit psycopg LISTEN
    connection** — NOT the engine pool (a LISTEN connection is long-lived).
  * **Never fatal** to the host app: if the DSN is missing or the connection
    drops, it logs + reconnects with capped backoff; the host keeps serving.
  * Holds the **latest sample per host** in memory; :meth:`latest_by_host`
    returns them read-only, each marked ``stale`` past ``stale_after_s``.

Typical use in a project's web process::

    from queue_workflows import hw_feed
    feed = hw_feed.HwFeed().start()          # background; reads the broker
    ...
    return {"hosts": feed.latest_by_host()}  # serve to the dashboard
"""

from __future__ import annotations

import json
import logging
import threading
import time
from typing import Any

from queue_workflows.hw_metrics import NOTIFY_CHANNEL, metrics_dsn

log = logging.getLogger(__name__)

_RECONNECT_MAX_S = 30.0
_POLL_TIMEOUT_S = 1.0  # wake the LISTEN loop ~1×/s so stop() is responsive


class HwFeed:
    """Background ``LISTEN hw_metrics`` reader holding the latest sample per host.

    ``dsn`` overrides the resolved :func:`metrics_dsn` (the broker) — mainly for
    tests. ``stale_after_s`` marks a host's last sample stale once telemetry
    stops arriving (the sampler emits ~every 5 s)."""

    def __init__(self, *, stale_after_s: float = 15.0, dsn: str | None = None):
        self.stale_after_s = float(stale_after_s)
        self._dsn = dsn
        self._latest: dict[str, dict[str, Any]] = {}   # host -> {"sample":…, "at": ts}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # ── lifecycle ────────────────────────────────────────────────────────────
    def start(self) -> "HwFeed":
        """Start the background reader (idempotent). Returns self for chaining."""
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, name="hw-feed", daemon=True)
            self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()

    # ── read model ───────────────────────────────────────────────────────────
    def latest_by_host(self) -> dict[str, dict[str, Any]]:
        """The latest sample per host (a copy), each augmented with a ``stale``
        flag (no fresh sample within ``stale_after_s``). ``{}`` until the first
        sample arrives."""
        now = time.time()
        with self._lock:
            return {
                host: {**rec["sample"], "stale": (now - rec["at"]) > self.stale_after_s}
                for host, rec in self._latest.items()
            }

    # ── internals ────────────────────────────────────────────────────────────
    def _store(self, payload: str) -> None:
        try:
            sample = json.loads(payload)
        except Exception:
            return
        host = str(sample.get("host") or "?")
        with self._lock:
            self._latest[host] = {"sample": sample, "at": time.time()}

    def _run(self) -> None:
        import psycopg

        backoff = 1.0
        while not self._stop.is_set():
            dsn = self._dsn or metrics_dsn()
            if not dsn:
                log.warning("[hw_feed] no metrics DSN configured "
                            "(config.metrics_db_url_env / db_url_env unset)")
                if self._stop.wait(5.0):
                    return
                continue
            try:
                with psycopg.connect(dsn, autocommit=True) as conn:
                    conn.execute(f"LISTEN {NOTIFY_CHANNEL}")
                    log.info("[hw_feed] LISTEN %s on the metrics DSN", NOTIFY_CHANNEL)
                    backoff = 1.0
                    while not self._stop.is_set():
                        for note in conn.notifies(timeout=_POLL_TIMEOUT_S):
                            self._store(note.payload)
            except Exception as exc:  # noqa: BLE001 — telemetry must never be fatal
                log.warning("[hw_feed] listener dropped (%s); retry in %.0fs",
                            exc, backoff)
                if self._stop.wait(backoff):
                    return
                backoff = min(backoff * 2, _RECONNECT_MAX_S)

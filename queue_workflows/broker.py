"""``queue-broker`` — stand up / own THE shared broker.

The multitenant model (migration 0017) is "shared DB + per-project clients": ONE
broker Postgres holds ONE ``cpu`` + ONE ``gpu`` (+ ingest) queue across ALL
projects, every record tagged with ``project``; each project's client
(orchestrator + workers + scheduler) connects to the SAME broker and
enqueues/claims only its own project's rows. This console is the explicit
"bootstrap the broker once, then point every project at it" entry point — the
thing that makes *one consolidated queue for all projects* a config flip:

    # 1. stand up the broker schema once (idempotent), against the broker DSN:
    BROKER_DSN=postgresql://…/broker  queue-broker

    # 2. every process of every project points at that broker + names itself:
    #    queue_workflows.configure(project="ai_leads",  db_url_env="BROKER_DSN")
    #    queue_workflows.configure(project="pic_to_3d",  db_url_env="BROKER_DSN")
    #    … each enqueues/claims ONLY its own project's rows on the shared queue.

    # 3. watch the consolidated queue across all projects:
    BROKER_DSN=…  queue-broker --status        # per-project breakdown
    BROKER_DSN=…  queue-conductor-web           # the web view

This is the orchestrator's ``db.bootstrap`` step made an explicit, inspectable
entry point. You needn't run it before the projects: ``db.bootstrap`` takes a
Postgres advisory lock, so concurrent orchestrator boots against one shared
broker are safe (the lock serializes; a late bootstrap that finds the chain
already applied is a no-op). It imports ONLY the client primitives — never the
conductor (the client→conductor boundary).
"""

from __future__ import annotations

import argparse

from queue_workflows import config as _config
from queue_workflows import db, node_queue


def _status() -> int:
    """Print the consolidated, broker-wide view: schema version + the projects
    sharing the broker + each one's cpu/gpu queue depth."""
    version = db.current_schema_version()
    if version == 0:
        print("broker schema NOT bootstrapped (version 0) — run `queue-broker`")
        return 1
    projects = node_queue.list_projects()
    print(f"broker schema version: {version}")
    print(f"projects on this broker ({len(projects)}):")
    if not projects:
        print("  (none — no queue records yet)")
    for p in projects:
        snap = node_queue.snapshot(project=p)
        c = snap.get("counts", {})

        def n(q: str, s: str) -> int:
            return int(c.get(f"{q}_{s}", 0))

        label = "(default)" if p == "" else p
        print(
            f"  {label:24s}  cpu[q{n('cpu','queued')} r{n('cpu','running')}]"
            f"  gpu[q{n('gpu','queued')} r{n('gpu','running')}]"
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="queue-broker",
        description="Stand up / inspect THE shared broker (one queue for all "
        "projects). Default: bootstrap the broker schema (idempotent).",
    )
    parser.add_argument(
        "--db-url-env", default=None,
        help="env var holding the broker DSN (default: the configured "
        f"{_config.get_config().db_url_env}).",
    )
    parser.add_argument(
        "--status", action="store_true",
        help="only print the consolidated broker view (do not bootstrap).",
    )
    args = parser.parse_args(argv)

    if args.db_url_env:
        import queue_workflows
        queue_workflows.configure(db_url_env=args.db_url_env)

    try:
        if args.status:
            return _status()
        db.bootstrap()  # idempotent + concurrency-safe: applies the engine chain
        print(f"broker bootstrapped — schema version {db.current_schema_version()}")
        return _status()
    finally:
        # Drain the pool so a short-lived console doesn't print psycopg_pool
        # "couldn't stop thread" warnings at interpreter shutdown.
        db.close_pool()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.0] — 2026-05-27

### Added
- **Pluggable storage backends** — a `StorageBackend` SPI
  (`queue_workflows.backends`) makes the durable-queue store selectable via
  `configure(db_backend="pg"|"redis"|"mongodb")`, one provider per file. `pg`
  (default) is byte-compatible and unchanged; `redis` (atomic claim/terminal via
  Lua + pub/sub wake) and `mongodb` (`find_one_and_update` + multi-doc-txn outbox
  + change-stream wake, replica set) reproduce the same contract — claim
  exactly-once, lease/reclaim, idempotent terminals, the atomic outbox, and
  per-namespace tenant isolation — pinned by a parametrized contract suite run
  against all three live servers. Optional drivers: `queue_workflows[redis]` /
  `[mongodb]`. See `docs/storage_backends.md`.
- Operator worker ON/OFF control plane: a `worker_controls` table + a
  per-`(host, queue)` `WorkerControlWatcher` that hard-stops (kill in-flight +
  free RAM/VRAM) or parks a worker on command (migration 0012). See
  `docs/worker_control.md`.
- Durable `workflow_node_events` store with lifecycle/trip/terminal event
  writers emitted from the worker, plus a NodePool retention sweep
  (migration 0011).
- Orchestrator-side dead-worker sweep that flags a worker whose heartbeat is
  stale while it still holds a `running` job; health-gated stall trip with
  re-queue-and-retry (migrations 0009/0010).
- No-progress `StallWatchdog` for GPU nodes; re-queue all running jobs on
  restart and worker self-kill when reassigned; always re-queue orphaned runs.
- `invoke_context` host hook — a per-node setup/teardown context manager whose
  `finalize(context_delta)` is applied only on success.
- MIT `LICENSE` and PEP 639 license metadata, keywords, and trove classifiers
  in `pyproject.toml`.
- `README.md`: a "local-cluster swift management" purpose note, an Architecture
  diagram (Mermaid), an "Example dashboard" section, a "Turning workers on/off"
  section, and a Docs index.
- `AGENTS.md` as a byte-identical mirror of `CLAUDE.md`, and a documented
  changelog convention.
- This `CHANGELOG.md`.

### Changed
- GPU guard is now health-driven (GPU-idle **and** static-RAM) instead of a
  fixed wall-clock cap.

### Fixed
- GPU claim: restored the capability-routing gate.
- `__input__` nodes are parked via the dispatch outbox instead of importing the
  sentinel.

### Removed
- `uv.lock` is no longer tracked — this is a library; consumers resolve against
  `pyproject.toml`. The lockfile stays local for dev/CI reproducibility.

## [0.2.0] — 2026-05-25

### Added
- Multi-tenant ingest path: host-defined ingest queues and per-job `args`, with
  host-side `task_name`/queue validation (migration 0008). A second consumer
  (e.g. `lm_flood`) can route its own queue names and carry per-job arguments
  without forking the schema.

## [0.1.0] — 2026-05-25

### Added
- Initial standalone release — a Postgres-as-queue workflow engine extracted
  from the `ai_leads` stack (Phase 6): a `SELECT … FOR UPDATE SKIP LOCKED` claim
  loop woken by `LISTEN`/`NOTIFY`, lease reclaim, a DAG dispatcher with a durable
  dispatch-event outbox, a GPU warm-model cache, periodic ingest work + a
  PG-native scheduler, and per-host hw-metrics telemetry. Migrations 0001–0007.

[Unreleased]: https://github.com/robertziel/python_workflows_queue/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/robertziel/python_workflows_queue/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/robertziel/python_workflows_queue/releases/tag/v0.2.0
[0.1.0]: https://github.com/robertziel/python_workflows_queue/commit/9ddaf4ae80d906e9d286403bab015e56ba9899ed

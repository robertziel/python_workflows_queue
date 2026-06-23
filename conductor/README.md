# queue-workflows-conductor

The **conductor** (control-plane) distribution for `queue-workflows` — deployed
**apart from** the per-project client.

It depends on the client distribution (`queue_workflows`) and consumes its primitives
(`node_queue.fleet_snapshot`, `worker_control`). The dependency edge points one
way — **conductor → client** — so the client (worker / orchestrator) never
imports the conductor.

Today it ships the read-only fleet view:

```bash
queue-conductor                 # table of every reporting worker (worker_heartbeats)
queue-conductor --queue gpu     # filter to one queue
queue-conductor --json          # machine-readable
```

The operator supplies the DB DSN via the client's `db_url_env` (e.g.
`AI_LEADS_DB_URL`), exactly like the client console scripts — single-DB, no
stored fleet credentials. The networked multi-DB daemon + web UI + inference
proxy will accrete into this package as separate, deployable surfaces.

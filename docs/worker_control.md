# Worker ON/OFF control

An operator turns a machine's **cpu** / **gpu** (or ingest) worker ON or OFF.
The first — and so far only — policy is a **hard stop**: stop the in-flight work
immediately and free RAM/VRAM now. The design leaves a clean seam for softer
policies (e.g. *drain*: finish the current task, then stop) without a schema or
API change.

Postgres is the only moving part. The control plane is one table; any process
that can write a row (the engine's helpers, the `queue-worker-control` CLI, or a
host app such as ai_leads' Rails over the shared DB) can flip a worker.

## Data model — `worker_controls` (migration 0012)

```
worker_controls(
  host_label    text,
  queue         text,            -- 'cpu' | 'gpu' | <ingest queue>
  desired_state text,            -- 'on' | 'off'   (CHECK)
  stop_policy   text,            -- 'hard' (default); free-form, validated in Python
  requested_by  text,
  updated_at    timestamptz,
  PRIMARY KEY (host_label, queue)
)
```

This is **desired** state, written by an operator. It is deliberately a separate
table from `worker_heartbeats` (which is *observed*, ephemeral state a live worker
upserts every ~10 s): an OFF state must persist precisely while the worker is NOT
beating — exactly when its heartbeat row is aging out — so it cannot live there.

Keyed `(host_label, queue)` — the same identity the heartbeat and the claim's
`claimed_by`/`queue` use. A host runs several workers under one `host_label`
(host-c runs a cpu *and* a gpu worker), so control is **per-queue**: turning off
"host-a gpu" never touches "host-a cpu".

A row trigger fires `pg_notify('worker_control', '<host>:<queue>')` on every
INSERT/UPDATE, so a worker wakes immediately and a plain SQL write from any
consumer (no app-side NOTIFY code) suffices — mirroring the `node_job_ready` /
`ingest_job_ready` triggers.

## Control flow

```
operator/Rails/CLI ──INSERT/UPDATE worker_controls (desired_state='off')──► trigger ──NOTIFY 'worker_control'─┐
                                                                                                              ▼
running worker: WorkerControlWatcher (daemon, LISTEN + safety poll) sees OFF ──► STOP_POLICIES['hard']
   1. requeue this worker's in-flight job(s)  (resume-style; node_queue.requeue_running_for_worker)
   2. clear the GPU busy-ghost                (node_queue.clear_worker_current_model)
   3. os._exit(79)                            (frees RAM/VRAM — OS tears down the CUDA context)
                                                                                                              │
supervisor (docker restart: on-failure) restarts the container ◄─────────────────────────────────────────────┘
   on boot: claim_worker._park_until_enabled() reads the row → still OFF → PARK
            (does NOT claim, does NOT heartbeat → ages out of the capacity gauge)

operator sets desired_state='on' (+ NOTIFY) ──► the parked loop resumes IN PLACE (no restart needed)
```

### Why a process exit, not a cooperative unload

A claim worker runs the node body **inline on its main thread** — no thread or
subprocess wraps it. A watcher thread therefore cannot preempt in-flight work,
and a wedged CUDA kernel won't honour a cooperative `cancel_event`. Terminating
the **process** is the only thing that reliably stops the work and reclaims
RAM/VRAM (the OS tears down the CUDA context on exit). This is the same lever
every in-engine watchdog already uses (`os._exit` codes 75 budget / 76 stall /
77 reassigned / 78 gpu-health); a control hard-stop is **79**.

The hard stop reuses the same auto-restart contract the watchdogs rely on
(docker `restart: on-failure`). On restart the worker re-reads `worker_controls`
and PARKS instead of claiming — so it comes back idle, not back to work.

### Why requeue, not cancel

Turning a machine off **redistributes** its in-flight work to a healthy peer (or
back to itself when turned ON) — it does not fail the workflow.
`requeue_running_for_worker` is resume-style: it flips the row `running → queued`,
clears the lease, and bumps priority to the front **without** incrementing
`watchdog_retries` (this is operational redistribution, not a node failure).
Clearing `claimed_by` also trips any surviving worker's `JobStatusWatcher`, so the
job is never double-run across the hand-off.

## Stop-policy seam (extensibility)

`worker_control.STOP_POLICIES` maps a policy name → `handler(worker, *, on_exit)`.
Only `"hard"` is wired today. Future policies slot in as new handlers with **no
schema or API change** (`stop_policy` is free-form TEXT in the DB, validated in
Python against this registry):

- **`drain`** — stop claiming, finish the current job, then park. Maps onto the
  existing cooperative `ClaimWorker._stop` (checked between jobs).
- **`pause`** — stop claiming but keep the warm model loaded for an instant
  resume (no RAM free).

## API

```python
from queue_workflows import worker_control

worker_control.disable_worker("host-a", "gpu")            # hard stop + stay off
worker_control.enable_worker("host-a", "gpu")             # resume
worker_control.set_worker_control("host-a", "gpu",
    desired_state="off", stop_policy="hard", requested_by="ops")
worker_control.get_worker_control("host-a", "gpu")        # row or None
worker_control.desired_state_for("host-a", "gpu")         # 'on' | 'off' (None/absent ⇒ 'on')
```

CLI (console script): `queue-worker-control --queue gpu --off [--host H] [--policy hard]` / `--on`.

Direct SQL — any consumer sharing the DB can write the row; the trigger handles
the wake. ai_leads' Rails toggles workers this way (raw `INSERT … ON CONFLICT`),
no Python on the request path.

`get_worker_control` swallows `UndefinedTable`, so the engine runs unchanged on a
DB that predates migration 0012 (treated as ON, default-on).

## Env knobs

- `AI_LEADS_WORKER_CONTROL_POLL_S` — safety-poll cadence behind the LISTEN wake
  (default 5.0 s).
- `AI_LEADS_DISABLE_WORKER_CONTROL` — keep the watcher inert (tests).

## Residual gap

A hang that **holds the GIL** freezes the watcher thread too, so the in-process
`os._exit` can't fire. The ultimate backstop is a host-local agent that
`docker kill`s the container when it sees `worker_controls.desired_state='off'`
unacked — the engine cannot cross-host kill (it has no docker socket on a remote
host). That host-local kill agent is future hardening, not part of this milestone.

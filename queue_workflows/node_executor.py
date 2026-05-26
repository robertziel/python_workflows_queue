"""Shared node-execution body.

``execute_node(job, *, model_cache=None, cancel_event=None)`` is the
implementation the Postgres-as-queue claim worker (``claim_worker``) calls to
actually run one ``workflow_node_jobs`` row. It's kept as a standalone
host-agnostic execution body (no queue-transport coupling) so it stays
trivially reusable + unit-testable.

It owns everything between "the row is claimed (status='running')" and
"the row is terminal + the dispatch-event outbox row is written":

  * late ``$from`` input resolution at execute time
    (``dispatcher.resolve_inputs_for_job``) + the ``resolved_inputs``
    snapshot;
  * the per-node out_dir;
  * optional warm-model load via the injected :class:`ModelCache`
    (GPU jobs that declare a ``required_model``);
  * the node-module invocation (``_invoke``), threading the model handle
    + ``cancel_event`` + ``model_load_seconds``;
  * the terminal ``mark_completed_in_txn`` / ``mark_failed_in_txn`` +
    ``enqueue_dispatch_event_in_txn`` in ONE transaction (the outbox
    atomicity contract);
  * the best-effort run-card thumbnail refresh on completion.

The node-module resolver is configurable (plan §2b-2): ``_invoke`` no longer
hardcodes ``importlib.import_module(f"workflows.nodes.{name}")`` — it asks the
engine config (``config.resolve_node_module``), which by default builds from
the host-set ``node_module_package`` (e.g. ``"workflows.nodes"``) or imports
the stored ``node_module`` value as a fully-qualified module when no package is
set.
"""

from __future__ import annotations

import contextlib
import inspect
import logging
import os
import time
import traceback
import typing
from pathlib import Path
from typing import Any

from queue_workflows import node_queue, run_store
from queue_workflows.config import get_config
from queue_workflows.db import connection as _db_connection

log = logging.getLogger(__name__)


# ── Node invoker (single source of truth) ─────────────────────────────────


def _rss_mb() -> int | None:
    try:
        with open(f"/proc/{os.getpid()}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) // 1024
    except Exception:
        return None
    return None


def _invoke(
    module_name: str,
    inputs: dict,
    out: Path | None,
    handle: Any,
    cancel_event: Any = None,
    model_load_seconds: float | None = None,
    status_callback: Any = None,
):
    """Import the node module for ``module_name`` (via the engine's
    configurable resolver) and dispatch to its ``run(...)`` with the inputs +
    optional model handle. Supports both the new keyword-only contract and the
    legacy positional per-arg contract — maps ``inputs[<name>]`` to signature
    params by name.

    ``cancel_event`` is the run cancel-watcher's ``threading.Event``. Node
    modules that opt into observing it accept a ``cancel_event`` kwarg in their
    ``run(...)`` signature; this helper threads it through. Modules that don't
    accept the kwarg are unchanged.
    """
    mod = get_config().resolve_node_module(module_name)
    run = getattr(mod, "run")
    sig = inspect.signature(run)
    # Resolve forward-ref annotations (modules use ``from __future__ import
    # annotations`` so ``param.annotation`` is a string like 'Path').
    try:
        hints = typing.get_type_hints(run)
    except Exception:
        hints = {}
    kwargs: dict[str, Any] = {}
    for name, param in sig.parameters.items():
        if param.kind in (inspect.Parameter.VAR_POSITIONAL,
                          inspect.Parameter.VAR_KEYWORD):
            continue
        if name == "out":
            kwargs["out"] = out
        elif name == "model_handle":
            kwargs["model_handle"] = handle
        elif name == "status_callback":
            # A node that declares ``status_callback`` opts into progress
            # reporting; the claim worker wires its StallWatchdog.beat here so
            # each reported step pushes the no-progress deadline out.
            kwargs["status_callback"] = status_callback
        elif name == "cancel_event":
            kwargs["cancel_event"] = cancel_event
        elif name == "model_load_seconds":
            # Auto-wired when the node declares a ``required_model`` —
            # separates the ModelCache.require_model cold-load latency
            # from the node's own inference time.
            kwargs["model_load_seconds"] = model_load_seconds
        elif name == "inputs":
            kwargs["inputs"] = inputs
        elif name in inputs:
            val = inputs[name]
            # A schema input that resolved to None means the workflow
            # definition didn't supply it — fall through to the
            # signature's own default instead of forcing ``param=None``.
            if val is None and param.default is not inspect.Parameter.empty:
                continue
            # Coerce JSON string → Path when the resolved type says Path.
            ann = hints.get(name, param.annotation)
            if isinstance(val, str):
                targets = (ann,) + tuple(getattr(ann, "__args__", ()))
                if Path in targets:
                    val = Path(val)
            kwargs[name] = val
    result = run(**kwargs)
    if isinstance(result, dict) and "context_delta" in result:
        return result
    if isinstance(result, dict):
        return {"context_delta": result}
    return {"context_delta": {}}


def _out_dir_for(job: dict, run: dict) -> Path | None:
    base = run.get("out_dir")
    if not base:
        return None
    p = Path(base) / job["node_id"]
    p.mkdir(parents=True, exist_ok=True)
    return p


_THUMB_REL_PATH = "thumb.jpg"
_THUMB_PX = 100
_THUMB_IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".webp"})


def _update_run_thumbnail(
    run_id: str, step_id: str, run: dict, context_delta: dict,
) -> None:
    """Refresh ``<run_out_dir>/thumb.jpg`` to a 100x100 center-cropped
    copy of the node's ``primary_file`` and flip ``is_primary`` to it in
    ``workflow_run_files``. Called after every node completion so the
    run-card thumbnail tracks the *latest* image the workflow has
    produced. Best-effort — a failure here must not fail the node body."""
    try:
        primary = (context_delta or {}).get("primary_file")
        out_dir = run.get("out_dir")
        if not primary or not out_dir:
            return
        # ``primary_file`` is conventionally written by nodes as a path
        # relative to either the run's out_dir or to the step's out_dir.
        # The step-relative form is the only one that survives nested
        # workflow-step ids; try several resolutions.
        primary_p = Path(primary)
        candidates = [
            Path(out_dir) / step_id / primary_p,           # step-relative
            Path(out_dir) / primary,                        # legacy run-relative
            primary_p if primary_p.is_absolute() else None, # already absolute
        ]
        src_path = next(
            (c for c in candidates if c is not None and c.exists()),
            None,
        )
        if src_path is None:
            return
        if src_path.suffix.lower() not in _THUMB_IMAGE_SUFFIXES:
            return

        from PIL import Image  # local import; PIL isn't needed by CPU nodes
        img = Image.open(src_path).convert("RGB")
        w, h = img.size
        scale = max(_THUMB_PX / w, _THUMB_PX / h)
        img = img.resize(
            (max(_THUMB_PX, int(w * scale)), max(_THUMB_PX, int(h * scale))),
            Image.LANCZOS,
        )
        nw, nh = img.size
        left = (nw - _THUMB_PX) // 2
        top = (nh - _THUMB_PX) // 2
        img = img.crop((left, top, left + _THUMB_PX, top + _THUMB_PX))

        thumb_abs = Path(out_dir) / _THUMB_REL_PATH
        tmp = thumb_abs.with_suffix(".jpg.tmp")
        img.save(tmp, "JPEG", quality=80, optimize=True)
        os.replace(tmp, thumb_abs)

        size_bytes = thumb_abs.stat().st_size
        with _db_connection() as conn, conn.cursor() as cur:
            cur.execute(
                "UPDATE workflow_run_files SET is_primary=false "
                "WHERE run_id=%s AND rel_path != %s",
                (run_id, _THUMB_REL_PATH),
            )
            cur.execute(
                """
                INSERT INTO workflow_run_files
                    (run_id, step_id, rel_path, kind, size_bytes, is_primary)
                VALUES (%s, %s, %s, 'image', %s, true)
                ON CONFLICT (run_id, rel_path) DO UPDATE
                  SET step_id    = EXCLUDED.step_id,
                      size_bytes = EXCLUDED.size_bytes,
                      is_primary = true
                """,
                (run_id, step_id, _THUMB_REL_PATH, size_bytes),
            )
    except Exception:
        log.exception("[_update_run_thumbnail] failed for run=%s step=%s",
                      run_id, step_id)


# ── Shared execute body ────────────────────────────────────────────────────


def execute_node(
    job: dict,
    *,
    model_cache: Any = None,
    cancel_event: Any = None,
    status_callback: Any = None,
) -> str:
    """Run one already-claimed ``workflow_node_jobs`` row to a terminal
    state. Returns one of ``"completed"`` / ``"failed"`` / ``"skipped"``.

    Contract:

      * The row MUST already be ``running`` (the caller's CAS / claim is
        the queued→running transition). ``execute_node`` doesn't re-claim.
      * On a GPU job with ``required_model``, ``model_cache`` MUST be
        supplied; ``model_cache.require_model(id)`` returns the handle and
        the elapsed load time is threaded to the node as
        ``model_load_seconds``. A model-load failure marks the row failed.
      * ``cancel_event`` (if supplied) is threaded into node ``run(...)``
        signatures that opt in via a ``cancel_event`` kwarg.
      * The terminal mark + the dispatch-event outbox row are written in
        ONE transaction. ``mark_*_in_txn`` returns ``None`` when the row
        is already terminal (a duplicate delivery / claim-race loser) — in
        that case we skip the event and return ``"skipped"``.
    """
    from queue_workflows import dispatcher

    job_id = job["id"]
    run = run_store.get_run(job["run_id"]) or {}
    t0 = time.time()

    required_model = job.get("required_model")
    handle = None
    model_load_seconds: float | None = None
    if required_model:
        # Cache-managed model load — only when the node declared one in
        # its schema. Legacy GPU nodes load their weights internally on
        # each call; they carry no ``required_model`` so they skip the
        # cache and invoke directly.
        if model_cache is None:
            err = (
                f"node {job.get('node_id')!r} declares required_model "
                f"{required_model!r} but no model_cache was supplied"
            )
            log.error("[execute_node] %s: %s", job_id, err)
            return _finalise_failed(job, err, t0)
        try:
            t0_model = time.time()
            handle = model_cache.require_model(required_model)
            model_load_seconds = time.time() - t0_model
            node_queue.record_node_event(
                event_type="model_load_done", elapsed_s=model_load_seconds,
                detail={"model_load_s": round(model_load_seconds, 2)},
                **_event_base(job),
            )
        except Exception as exc:
            err = f"model load failed: {type(exc).__name__}: {exc}"
            log.exception("[execute_node] %s %s", job_id, err)
            return _finalise_failed(job, err, t0)
        # Model is warm — beat the stall watchdog so its no-progress window
        # opens HERE (after the multi-minute cold load), not at claim time. From
        # now on the node's per-step ``status_callback`` beats keep it alive; a
        # post-load inference hang (GPU at 0 %) stops beating and trips it.
        if status_callback is not None:
            try:
                status_callback()
            except Exception:
                log.exception("[execute_node] %s post-load stall beat failed", job_id)

    # Re-resolve $from refs at execution time — picks up any upstream
    # sibling context_delta changes between enqueue and now. Snapshot the
    # result for forensics.
    fresh_inputs = dispatcher.resolve_inputs_for_job(job_id, job=job)
    try:
        node_queue.set_resolved_inputs(job_id, fresh_inputs)
    except Exception:
        log.exception("[execute_node] %s could not record resolved_inputs", job_id)

    # Optional host wrapper around the invoke (config.invoke_context): its
    # __enter__ does host setup (e.g. pin a run-context ContextVar + capture a
    # live mock flag) and yields a finalize(context_delta)->context_delta applied
    # on success; __exit__ tears down on EVERY return path below. Default unset ⇒
    # nullcontext(None) ⇒ identical behavior to running the node directly.
    cfg = get_config()
    _invoke_cm = (
        cfg.invoke_context(job, run)
        if cfg.invoke_context is not None
        else contextlib.nullcontext(None)
    )
    with _invoke_cm as _finalize:
        try:
            result = _invoke(
                module_name=job["node_module"],
                inputs=fresh_inputs,
                out=_out_dir_for(job, run),
                handle=handle,
                cancel_event=cancel_event,
                model_load_seconds=model_load_seconds,
                status_callback=status_callback,
            )
        except Exception as exc:
            err = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
            log.error(err)
            return _finalise_failed(job, err, t0)

        context_delta = result.get("context_delta", {})
        if _finalize is not None:
            context_delta = _finalize(context_delta)
        seconds = time.time() - t0
        with _db_connection() as conn, conn.cursor() as cur:
            row = node_queue.mark_completed_in_txn(
                cur, job_id,
                context_delta=context_delta,
                seconds=seconds,
                vm_rss_mb_peak=_rss_mb(),
            )
            if row is None:
                log.warning(
                    "[execute_node] %s already terminal; skipping completion event",
                    job_id,
                )
                return "skipped"
            node_queue.enqueue_dispatch_event_in_txn(
                cur, job["run_id"], job["node_id"], "completed",
            )
        # Forensic node event — best-effort, AFTER the load-bearing terminal txn
        # commits, so an event-write blip can never roll back the completion.
        node_queue.record_node_event(
            event_type="completed", elapsed_s=seconds, **_event_base(job),
        )
        _update_run_thumbnail(
            job["run_id"], job["node_id"], run, context_delta,
        )
        return "completed"


def _event_base(job: dict) -> dict[str, Any]:
    """Common node-event fields pulled from a workflow_node_jobs row, so every
    emit site carries uniform host / attempt / model / queue context.
    ``attempt`` = watchdog_retries at emit time (the cross-attempt key)."""
    return {
        "run_id": job["run_id"],
        "node_id": job["node_id"],
        "job_id": job.get("id"),
        "attempt": int(job.get("watchdog_retries") or 0),
        # claimed_by is the real host identity (the engine leaves host_label NULL
        # in practice and uses claimed_by) — fall back to host_label just in case.
        "host_label": job.get("claimed_by") or job.get("host_label"),
        "queue": job.get("queue"),
        "model": job.get("required_model"),
    }


def _finalise_failed(job: dict, error: str, t0: float) -> str:
    """Write the terminal ``failed`` row + ``failed`` dispatch event in
    one txn. Returns ``"failed"`` (or ``"skipped"`` if the row was
    already terminal)."""
    seconds = time.time() - t0
    with _db_connection() as conn, conn.cursor() as cur:
        row = node_queue.mark_failed_in_txn(
            cur, job["id"], error=error, seconds=seconds,
        )
        if row is None:
            log.warning(
                "[execute_node] %s already terminal; skipping fail event",
                job["id"],
            )
            return "skipped"
        node_queue.enqueue_dispatch_event_in_txn(
            cur, job["run_id"], job["node_id"], "failed",
        )
    # Forensic node event — best-effort, after the terminal txn commits.
    node_queue.record_node_event(
        event_type="failed", elapsed_s=seconds, error=error, **_event_base(job),
    )
    return "failed"


__all__ = ["execute_node", "_invoke", "_out_dir_for", "_rss_mb",
           "_update_run_thumbnail"]

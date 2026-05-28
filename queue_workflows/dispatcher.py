"""DAG dispatcher for the node-per-job engine.

Given a workflow definition (nodes + depends_on), translates a run into
a stream of queued node-jobs:

1. At run start: enqueue all nodes with empty ``depends_on``.
2. After a node completes: find every node whose deps are now fully
   completed (and whose own row doesn't exist yet) and enqueue it.
3. When a node fails: cancel siblings; flip the run to ``failed``.
4. When all nodes are completed: flip the run to ``completed``.
5. Input nodes: when claimed, mark the job + run as ``awaiting_input``.
   Resume via :func:`resume_after_input`.

All DB work goes through :mod:`node_queue` (jobs) + :mod:`run_store` (the run
row). The workflow/pipeline DEFINITION SOURCE is host-injected (plan §1e): the
dispatcher reads workflows + pipeline schemas through ``config.workflow_loader``
/ ``config.pipeline_schema_loader``, and resolves ``$from`` refs through
``config.get_resolve_ref()`` (defaulting to the engine's own
:func:`queue_workflows.refs.resolve_ref`). This module holds the pure DAG-walk
logic so it's unit-testable without a worker pool.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from typing import Any

from queue_workflows import node_queue, run_store
from queue_workflows.config import get_config

log = logging.getLogger(__name__)


def _now():
    return datetime.now(timezone.utc)


# ── injected definition-source accessors ──────────────────────────────────


def _load_workflow(name: str) -> dict:
    loader = get_config().workflow_loader
    if loader is None:
        raise RuntimeError(
            "no workflow loader configured; call "
            "queue_workflows.set_workflow_provider(load_workflow, pipeline_schema)"
        )
    return loader(name)


def _pipeline_schema(name: str) -> dict:
    loader = get_config().pipeline_schema_loader
    if loader is None:
        raise RuntimeError(
            "no pipeline-schema loader configured; call "
            "queue_workflows.set_workflow_provider(load_workflow, pipeline_schema)"
        )
    return loader(name)


def _resolve_ref(value: Any, context: dict) -> Any:
    return get_config().get_resolve_ref()(value, context)


# ── Workflow shape helpers ───────────────────────────────────────────────


def _nodes_of(
    workflow: dict[str, Any],
    run: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Expand a workflow into its node DAG using pipeline schemas.

    Pipeline schemas own the DAG (``nodes: [...]`` with id / depends_on / gpu /
    model / inputs). For each ``kind='pipeline'`` step in the workflow, we
    expand the schema's nodes inline with prefixed ids so multiple pipeline
    steps in the same workflow don't collide.

    Step-level ``depends_on`` is inherited by source nodes so an input step
    properly gates the pipeline.
    """
    out: list[dict[str, Any]] = []
    run_out_dir = (run or {}).get("out_dir") if run else None

    # Step-to-leaf-nodes map — a step-level ``depends_on`` entry like
    # ``"svinfer"`` must expand to "every terminal node of the svinfer
    # pipeline is completed".
    step_leaves: dict[str, list[str]] = {}
    for step in workflow.get("steps", []) or []:
        sid = step["id"]
        if step.get("kind") == "input":
            step_leaves[sid] = [sid]  # input step id == node id
            continue
        if step.get("kind") != "pipeline":
            continue
        schema = _pipeline_schema(step["pipeline"])
        nodes = schema.get("nodes", []) or []
        referenced = {
            d for sn in nodes for d in (sn.get("depends_on") or [])
        }
        leaves = [f"{sid}/{sn['id']}" for sn in nodes if sn["id"] not in referenced]
        step_leaves[sid] = leaves or [f"{sid}/{nodes[-1]['id']}"]

    def _expand_step_deps(deps: list[str]) -> list[str]:
        out_deps: list[str] = []
        for d in deps:
            if d in step_leaves:
                out_deps.extend(step_leaves[d])
            else:
                out_deps.append(d)
        return out_deps

    for step in workflow.get("steps", []) or []:
        kind = step.get("kind")
        step_skip_if = step.get("skip_if")
        if kind == "input":
            out.append({
                "id": step["id"],
                "kind": "input",
                "widget": step.get("widget"),
                "target": step.get("target"),
                "depends_on": _expand_step_deps(step.get("depends_on", [])),
                "skip_if": step_skip_if,
            })
            continue
        if kind != "pipeline":
            continue
        step_id = step["id"]
        step_inputs = step.get("inputs", {}) or {}
        step_deps = _expand_step_deps(list(step.get("depends_on") or []))
        schema = _pipeline_schema(step["pipeline"])
        pipeline_name = step.get("pipeline")
        for sn in schema.get("nodes", []) or []:
            nid = f"{step_id}/{sn['id']}"
            schema_deps = sn.get("depends_on") or []
            deps = [f"{step_id}/{d}" for d in schema_deps]
            if not schema_deps:
                deps.extend(step_deps)
            resolved_inputs: dict[str, Any] = {}
            for inp in sn.get("inputs") or []:
                src = inp.get("from") or ""
                name = inp.get("name")
                if not name:
                    continue
                resolved_inputs[name] = _resolve_input_ref(
                    src, step_id, step_inputs, run_out_dir,
                )
            out.append({
                "id": nid,
                "node": sn.get("node") or sn["id"],
                "depends_on": deps,
                "gpu": bool(sn.get("gpu")),
                "model": sn.get("model"),
                "inputs": resolved_inputs,
                "pipeline_name": pipeline_name,
                "kind": "node",
                # Step-level skip_if propagates to every node in the pipeline so
                # the whole branch goes 'skipped' as one. A schema-level node may
                # also carry its own ``skip_if``. Node-level wins; step-level is
                # the fallback.
                "skip_if": sn.get("skip_if") or step_skip_if,
            })
    return out


_IMPLICIT_RE = re.compile(r"^pipeline\._implicit\((.+)\)$")
_SIBLINGS_RE = re.compile(r"^pipeline\._siblings\((.+)\)$")


def _resolve_input_ref(
    src: str,
    step_id: str,
    step_inputs: dict[str, Any],
    run_out_dir: str | None,
) -> Any:
    """Translate a schema-node input ``from`` ref into a concrete value (or a
    ``$from`` ref that the enqueue path will resolve against ``run.context``)."""
    # Schema shorthand for "inline literal" — ``pipeline._implicit(10)``.
    m = _IMPLICIT_RE.match(src or "")
    if m:
        raw = m.group(1).strip()
        try:
            import ast
            return ast.literal_eval(raw)
        except Exception:
            return raw
    # ``pipeline._siblings([id1, id2, ...])`` — resolve each id to its
    # sibling-node out dir.
    m = _SIBLINGS_RE.match(src or "")
    if m:
        raw = m.group(1).strip()
        try:
            import ast
            ids = ast.literal_eval(raw)
        except Exception:
            return []
        if not isinstance(ids, (list, tuple)):
            return []
        if not run_out_dir:
            return list(ids)
        return [f"{run_out_dir}/{step_id}/{sid}" for sid in ids]
    if src.startswith("pipeline."):
        key = src[len("pipeline."):]
        return step_inputs.get(key)
    # Sibling ref: ``<sibling_id>[.<rest>]``. Resolve to the path under the
    # run's out_dir so the downstream node gets a concrete file to open.
    if run_out_dir:
        sib_id, _, rest = src.partition(".")
        base = f"{run_out_dir}/{step_id}/{sib_id}"
        return f"{base}/{rest}" if rest else base
    # Without a run_out_dir (registry-time expansion) return a template string.
    return src


def _input_node(node: dict[str, Any]) -> bool:
    return node.get("kind") == "input"


def _queue_of(node: dict[str, Any]) -> str:
    # Every ``gpu: true`` node lands on the GPU queue. Nodes without a declared
    # ``model`` aren't cache-managed — the GPU claim worker skips
    # ``require_model`` for them and invokes the node directly.
    return "gpu" if node.get("gpu") else "cpu"


def _required_model(node: dict[str, Any]) -> str | None:
    if node.get("gpu") and node.get("model"):
        return node.get("model")
    return None


# ── Ready-node search ────────────────────────────────────────────────────


def _jobs_by_node_id(run_id: str) -> dict[str, dict[str, Any]]:
    return {j["node_id"]: j for j in node_queue.list_jobs_for_run(run_id)}


def _find_ready_nodes(
    workflow: dict[str, Any],
    existing: dict[str, dict[str, Any]],
    run: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Return every node that has no job row yet AND whose deps have all
    completed (or skipped — a skipped predecessor is a satisfied one for
    branch-gating purposes)."""
    ready: list[dict[str, Any]] = []
    for n in _nodes_of(workflow, run=run):
        nid = n["id"]
        if nid in existing:
            continue
        deps = n.get("depends_on", []) or []
        if all(
            existing.get(d) and existing[d]["status"] in ("completed", "skipped")
            for d in deps
        ):
            ready.append(n)
    return ready


def _eval_skip_context(
    run: dict[str, Any] | None,
    existing: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Build the merged context used for ``skip_if`` evaluation: the run's base
    context plus every completed sibling's context_delta namespaced by node_id.
    Skipped siblings carry empty deltas, so they're harmless to merge in."""
    ctx = dict((run or {}).get("context") or {})
    for nid, j in (existing or {}).items():
        if j.get("status") == "completed" and j.get("context_delta"):
            ctx = {**ctx, nid: j["context_delta"]}
    return ctx


def _should_skip_node(
    node: dict[str, Any],
    run: dict[str, Any] | None,
    existing: dict[str, dict[str, Any]],
) -> bool:
    """Evaluate ``skip_if`` against the merged context. Returns True if the
    dispatcher should insert a status='skipped' row for this node instead of
    enqueueing it.

    Resolver-level errors (KeyError on a missing path, TypeError on a malformed
    expression) fall back to "don't skip" — fail-safe so a misspelled ref
    doesn't silently dead-end the whole branch.
    """
    skip_if = node.get("skip_if")
    if not skip_if:
        return False
    context = _eval_skip_context(run, existing)
    try:
        return bool(_resolve_ref(skip_if, context))
    except (KeyError, TypeError) as exc:
        log.warning(
            "[dispatcher] skip_if eval failed for %s (%s) — not skipping",
            node.get("id"), exc,
        )
        return False


# ── Public API ───────────────────────────────────────────────────────────


def _process_ready(
    run_id: str,
    wf: dict[str, Any],
    run: dict[str, Any],
) -> int:
    """Loop: find ready nodes, dispatch each to _enqueue (queued) or
    insert_skipped_job (skipped per skip_if), repeat. Cascading skips handle
    themselves because a freshly-skipped row is a satisfied predecessor for its
    dependents on the next iteration.

    Returns the total number of new rows inserted (queued + skipped).
    """
    new_rows = 0
    while True:
        existing = _jobs_by_node_id(run_id)
        ready = _find_ready_nodes(wf, existing, run=run)
        if not ready:
            break
        # Batching for affinity routing: when multiple GPU nodes become ready in
        # the same fan-out tick, group them by ``required_model`` so consecutive
        # claims for the same model benefit from warm affinity. Within a model
        # group we keep the original order so the workflow's natural sequencing
        # isn't reshuffled.
        ready_sorted = sorted(
            enumerate(ready),
            key=lambda pair: (
                pair[1].get("model") or "￿",
                pair[0],
            ),
        )
        progress = False
        for _idx, n in ready_sorted:
            if _should_skip_node(n, run, existing):
                node_queue.insert_skipped_job(
                    run_id=run_id,
                    node_id=n["id"],
                    pipeline_name=n.get("pipeline_name"),
                )
            else:
                _enqueue(run_id, n, run)
            new_rows += 1
            progress = True
        if not progress:
            break
    return new_rows


def start_run(run_id: str) -> int:
    """Expand the run's workflow into its initial set of queued node-jobs.

    Idempotent: re-calling doesn't duplicate rows (UNIQUE on (run_id, node_id)).
    Returns how many NEW rows it enqueued.
    """
    run = run_store.get_run(run_id)
    if not run:
        raise KeyError(run_id)
    wf = _load_workflow(run["workflow_name"])
    return _process_ready(run_id, wf, run)


def on_node_completed(run_id: str, node_id: str) -> int:
    """Called by workers after a node-job flips to ``completed``.

    Finds newly-satisfied downstream nodes and enqueues (or skips) them. If
    every node in the DAG has reached a terminal status, flips the run row to
    ``completed``. Returns the number of new node-jobs inserted.
    """
    run = run_store.get_run(run_id)
    if not run:
        return 0
    # Early-return on terminal run. Belt-and-braces with the SQL-level claim
    # guard: if the run was cancelled or failed between this node's CAS win and
    # the dispatch-event drain firing this callback, we must NOT enqueue
    # downstream nodes.
    if run.get("status") in ("cancelled", "failed"):
        return 0
    wf = _load_workflow(run["workflow_name"])
    new_rows = _process_ready(run_id, wf, run)
    # Terminal check: every node in the DAG has either completed or been
    # skipped. Skipped branches count as terminal.
    existing = _jobs_by_node_id(run_id)
    all_nodes = [n["id"] for n in _nodes_of(wf, run=run)]
    if all_nodes and all(
        existing.get(nid) and existing[nid]["status"] in ("completed", "skipped")
        for nid in all_nodes
    ):
        run_store.update_run(run_id, status="completed", finished_at=_now())
    return new_rows


def on_node_failed(run_id: str, node_id: str) -> None:
    """Run-level failure. Cancels queued siblings, marks run failed.

    Short-circuit when the run is already in a terminal cancelled/failed state:
    a separate worker may have already finalised it. Avoid re-cancelling
    siblings that were already flipped, and avoid flipping the run-level error
    message away from whatever the earlier finaliser wrote.
    """
    run = run_store.get_run(run_id) or {}
    if run.get("status") in ("cancelled", "failed"):
        return
    node_queue.cancel_siblings_after_failure(run_id)
    run_store.update_run(
        run_id,
        status="failed",
        finished_at=_now(),
        error=f"node {node_id!r} failed (see workflow_node_jobs.error)",
    )


def on_node_awaiting_input(run_id: str, node_id: str) -> None:
    """Build + persist a per-job ``input_spec`` so the frontend can render the
    widget for this awaiting node.

    Conceptual model: input nodes are normal DAG nodes; they sit in an "input
    queue" the same way CPU/GPU nodes sit in their queues. The run-level status
    stays ``running`` regardless of how many input nodes are parked.
    """
    run = run_store.get_run(run_id) or {}
    # Don't bother building a spec for a cancelled run.
    if run.get("status") in ("cancelled", "failed"):
        return
    wf = _load_workflow(run.get("workflow_name", "")) if run else None
    input_step = None
    if wf:
        input_step = next(
            (s for s in (wf.get("steps") or [])
             if s.get("kind") == "input" and s.get("id") == node_id),
            None,
        )
    spec = _build_input_spec(input_step, run) if input_step else None
    if spec is not None:
        try:
            node_queue.set_input_spec(run_id, node_id, spec)
        except Exception:
            log.exception(
                "[dispatcher] couldn't persist per-job input_spec for %s/%s",
                run_id, node_id,
            )


def _build_input_spec(step: dict[str, Any], run: dict[str, Any]) -> dict[str, Any]:
    """Assemble the payload the frontend reads off ``run.input_spec``.

    Widget-specific fields plus a resolved source list for ``choose_one``.
    Source resolution digs an on-the-fly context that includes a
    ``<step_id>.files`` list for every pipeline step the run has already
    produced artefacts for — see :func:`_run_context_for_refs`.
    """
    spec: dict[str, Any] = {
        "step_id": step["id"],
        "widget": step["widget"],
        "prompt": step.get("prompt", ""),
        "target": step.get("target"),
    }
    widget = step["widget"]
    if widget == "choose_one":
        context = _run_context_for_refs(run)
        source = step.get("source")
        try:
            options = _resolve_ref(source, context) if source else []
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "[dispatcher] failed to resolve choose_one source for "
                "%s: %s", step["id"], exc,
            )
            options = []
        spec["options"] = options or []
        spec["multiple"] = bool(step.get("multiple", False))
    elif widget == "file_upload":
        spec["accept"] = step.get("accept", "*/*")
    elif widget == "pick_or_upload":
        spec["library"] = step.get("library")
        spec["accept"] = step.get("accept", "*/*")
    elif widget == "turnout":
        # Branch picker: a list of {label, value} options. The user's chosen
        # ``value`` is stored in ``context.<step_id>.<target>`` (via
        # ``resume_after_input``) so downstream ``skip_if`` refs can gate which
        # branch executes.
        options = step.get("options") or []
        spec["turnout_options"] = [
            {"label": str(o.get("label", o.get("value", ""))),
             "value": str(o.get("value", ""))}
            for o in options
            if isinstance(o, dict) and o.get("value") is not None
        ]
    elif widget == "pick_perspective":
        # Resolves the list of fisheye/equirect files for the user to pick AND
        # frame in the dock. Same context machinery as choose_one.
        context = _run_context_for_refs(run)
        source = step.get("source")
        try:
            options = _resolve_ref(source, context) if source else []
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "[dispatcher] failed to resolve pick_perspective source for "
                "%s: %s", step["id"], exc,
            )
            options = []
        if isinstance(options, dict):
            options = [options]
        elif not isinstance(options, list):
            options = []
        spec["fisheye_options"] = options
        # Backwards-compat: keep first match under the legacy single-pano keys.
        if options:
            spec["fisheye_rel_path"] = options[0].get("rel_path")
            spec["fisheye_abs_path"] = options[0].get("abs_path")
        spec["pano_meta"] = _resolve_pano_meta(run, options)
    elif widget == "pick_fence":
        # Per-detection fence picker — operator clicks individual detections
        # (their own fence, ignore neighbour's) before the paint refinement
        # step. Spec carries (a) the source image to display as background and
        # (b) the detections.json index the host widget fetches to render each
        # detection as a colored overlay. Same $from/$filter resolution shape
        # as paint_mask + choose_one — no new mini-language.
        context = _run_context_for_refs(run)
        source = step.get("source")
        try:
            src_options = _resolve_ref(source, context) if source else []
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "[dispatcher] failed to resolve pick_fence source for "
                "%s: %s", step["id"], exc,
            )
            src_options = []
        if isinstance(src_options, dict):
            src_options = [src_options]
        elif not isinstance(src_options, list):
            src_options = []
        spec["source_options"] = src_options
        if src_options:
            spec["source_rel_path"] = src_options[0].get("rel_path")
            spec["source_abs_path"] = src_options[0].get("abs_path")
        # Detections JSON — same ref shape; the widget fetches the file via
        # /workflow/:id/file?path=... and parses the list of detection entries.
        detections = step.get("detections")
        try:
            det_options = _resolve_ref(detections, context) if detections else []
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "[dispatcher] failed to resolve pick_fence detections for "
                "%s: %s", step["id"], exc,
            )
            det_options = []
        if isinstance(det_options, dict):
            det_options = [det_options]
        elif not isinstance(det_options, list):
            det_options = []
        spec["detections_options"] = det_options
        if det_options:
            spec["detections_rel_path"] = det_options[0].get("rel_path")
            spec["detections_abs_path"] = det_options[0].get("abs_path")
    elif widget == "paint_mask":
        # Resolves the source image the user will paint a mask on. The widget
        # uploads a binary mask PNG back through the standard multipart pipe;
        # the spec just needs to tell it which image to display.
        context = _run_context_for_refs(run)
        source = step.get("source")
        try:
            options = _resolve_ref(source, context) if source else []
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "[dispatcher] failed to resolve paint_mask source for "
                "%s: %s", step["id"], exc,
            )
            options = []
        if isinstance(options, dict):
            options = [options]
        elif not isinstance(options, list):
            options = []
        spec["source_options"] = options
        if options:
            spec["source_rel_path"] = options[0].get("rel_path")
            spec["source_abs_path"] = options[0].get("abs_path")
        # Optional ``initial_mask`` ref — when set, the widget pre-paints the
        # resolved PNG onto the overlay so the operator starts from an
        # auto-detected mask instead of a blank canvas (e.g. the
        # multi-erase fence experiment seeds this from a GroundingDINO+SAM2
        # pre-pass). Same shape as ``source`` — a $from/$filter ref against
        # the run context — to keep the resolver symmetrical.
        initial_mask = step.get("initial_mask")
        if initial_mask is not None:
            try:
                im_options = _resolve_ref(initial_mask, context)
            except Exception as exc:  # noqa: BLE001
                log.warning(
                    "[dispatcher] failed to resolve paint_mask initial_mask "
                    "for %s: %s", step["id"], exc,
                )
                im_options = []
            if isinstance(im_options, dict):
                im_options = [im_options]
            elif not isinstance(im_options, list):
                im_options = []
            spec["initial_mask_options"] = im_options
            if im_options:
                spec["initial_mask_rel_path"] = im_options[0].get("rel_path")
                spec["initial_mask_abs_path"] = im_options[0].get("abs_path")
    elif widget == "assign_walls":
        # Operator-facing wall-assignment widget for the facade-merge workflow.
        import json
        from pathlib import Path
        out_dir = run.get("out_dir")
        spec["parser_lanes"] = []
        spec["all_proposals"] = []
        spec["plate_candidates"] = []
        if out_dir:
            root = Path(out_dir)
            parse_root = root / "parse_facade"
            lane_dirs: list[Path] = []
            if parse_root.exists():
                lane_dirs = sorted(
                    p for p in parse_root.iterdir()
                    if p.is_dir() and (p / "walls.json").exists()
                )
            for lane_dir in lane_dirs:
                lane = lane_dir.name
                manifest_path = lane_dir / "walls.json"
                walls_out: list[dict[str, Any]] = []
                n_walls = 0
                try:
                    data = json.loads(manifest_path.read_text())
                    for w in data.get("walls", []):
                        rel = w.get("image_path", "")
                        abs_path = (
                            manifest_path.parent / rel
                        ).as_posix() if rel else ""
                        entry = {
                            "id":       w.get("id"),
                            "label":    w.get("label", ""),
                            "abs_path": abs_path,
                        }
                        walls_out.append(entry)
                        spec["all_proposals"].append({
                            **entry,
                            "parser": lane,
                            "global_id": f"{lane}:{entry['id']}",
                        })
                    n_walls = data.get("n_walls", len(walls_out))
                except (OSError, ValueError, KeyError):
                    log.exception(
                        "[dispatcher] couldn't parse %s", manifest_path,
                    )
                spec["parser_lanes"].append({
                    "parser":  lane,
                    "n_walls": n_walls,
                    "walls":   walls_out,
                })
            for sr_lane in ("sr_realesrgan", "sr_swinir", "sr_osediff"):
                plate = root / "diffuse_3d" / sr_lane / "pano_0_upscaled.png"
                if plate.exists():
                    spec["plate_candidates"].append({
                        "lane":     sr_lane,
                        "abs_path": plate.as_posix(),
                    })
    elif widget == "capture_3d":
        # Google Photorealistic 3D Tiles viewer. Surface the parcel lat/lon so
        # the viewer can drop the camera at the right place.
        parcel = (run.get("context") or {}).get("parcel") or {}
        spec["lat"] = parcel.get("lat") if parcel else None
        spec["lon"] = parcel.get("lon") if parcel else None
        label = parcel.get("label") if isinstance(parcel, dict) else None
        if label:
            spec["parcel_label"] = label
    return spec


def _resolve_pano_meta(
    run: dict[str, Any], options: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Map each fisheye option to its ``views.json`` entry by pano index.

    A rel_path like ``pano/pano_fetch/pano_2/sv_equirect.jpg`` resolves to entry
    index 2 in the corresponding ``views.json``. We lift only the keys the
    widget actually needs to keep the spec payload small.
    """
    import json
    import re
    from pathlib import Path

    out_dir = run.get("out_dir")
    if not out_dir or not options:
        return [{} for _ in options]

    root = Path(out_dir)
    cache: dict[Path, list[dict[str, Any]]] = {}
    pano_idx_re = re.compile(r"pano_(\d+)/")

    def _views_for(rel_path: str) -> list[dict[str, Any]] | None:
        p = (root / rel_path).resolve().parent
        for ancestor in [p, *p.parents]:
            try:
                ancestor.relative_to(root)
            except ValueError:
                return None
            candidate = ancestor / "views.json"
            if candidate.exists():
                if candidate not in cache:
                    try:
                        cache[candidate] = json.loads(candidate.read_text())
                    except Exception:  # noqa: BLE001
                        cache[candidate] = []
                return cache[candidate]
        return None

    out: list[dict[str, Any]] = []
    keep = ("pano_id", "parcel_heading_deg", "date", "camera_height_m", "distance_m")
    for opt in options:
        rel = opt.get("rel_path") or ""
        m = pano_idx_re.search(rel)
        if not m:
            out.append({})
            continue
        idx = int(m.group(1))
        views = _views_for(rel) or []
        entry = views[idx] if 0 <= idx < len(views) else {}
        out.append({k: entry.get(k) for k in keep if k in entry})
    return out


def _run_context_for_refs(run: dict[str, Any]) -> dict[str, Any]:
    """Build the context dict ``$from`` refs resolve against when paused at an
    input step.

    Starts with ``run.context`` and augments it with a ``<step_id>: {"files":
    [...]}`` entry for each pipeline step whose output directory exists on disk.
    Each file record has ``rel_path`` + ``kind`` so ``$filter`` clauses work.
    """
    from pathlib import Path
    ctx: dict[str, Any] = dict(run.get("context") or {})
    out_dir = run.get("out_dir")
    if not out_dir:
        return ctx
    root = Path(out_dir)
    if not root.is_dir():
        return ctx
    ext_kind = {
        ".png": "image", ".jpg": "image", ".jpeg": "image",
        ".webp": "image", ".gif": "image",
        ".json": "json", ".geojson": "json",
        ".csv": "csv", ".txt": "text", ".md": "text",
        ".ply": "pointcloud", ".obj": "mesh", ".glb": "mesh",
    }
    # ``rel_path`` is relative to ``out_dir`` (NOT to ``step_dir``) because the
    # frontend feeds it straight into ``/api/workflow/:id/file?path=<rel_path>``.
    for step_dir in root.iterdir():
        if not step_dir.is_dir():
            continue
        files: list[dict[str, Any]] = []
        for f in sorted(step_dir.rglob("*")):
            if not f.is_file():
                continue
            rel = f.relative_to(root).as_posix()
            files.append({
                "rel_path": rel,
                "kind": ext_kind.get(f.suffix.lower(), "file"),
                "abs_path": str(f),
                "size_bytes": f.stat().st_size,
            })
        ctx[step_dir.name] = {
            **(ctx.get(step_dir.name) or {}),
            "files": files,
        }
    return ctx


def resume_after_input(run_id: str, node_id: str, value: Any = None) -> int:
    """Called when the user submits input for an awaiting_input node.

    Writes ``{target: value}`` into the input node's ``context_delta`` so
    downstream nodes can resolve ``$from: <input_node>.<target>`` refs against
    it. ``target`` is read from the input node's ``inputs`` dict (populated at
    enqueue time in :func:`_enqueue`).

    Then marks the job completed and schedules dependents via
    :func:`on_node_completed`. Returns how many downstream jobs got enqueued.
    """
    jobs = _jobs_by_node_id(run_id)
    job = jobs.get(node_id)
    if not job:
        raise KeyError(f"no job for {run_id}:{node_id}")
    if job["status"] not in ("awaiting_input", "queued"):
        return 0
    target = (job.get("inputs") or {}).get("target")
    context_delta: dict[str, Any] = {}
    if target and value is not None:
        context_delta[target] = value
    node_queue.mark_completed(
        job["id"], context_delta=context_delta, seconds=0.0,
    )
    # Input nodes behave like normal nodes — once they complete the only thing
    # left is to cascade their dependents.
    return on_node_completed(run_id, node_id)


# ── Late input resolution ─────────────────────────────────────────────────


def resolve_inputs_for_job(
    job_id: str,
    *,
    job: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Re-resolve a job's input refs against the LIVE run context + completed
    siblings' context_deltas.

    The pre-existing ``workflow_node_jobs.inputs`` column holds an enqueue-time
    snapshot. If an upstream sibling's ``context_delta`` is mutated between
    enqueue and execute, the snapshot is stale by the time the worker runs.

    This helper resolves refs at execution time so the worker sees the current
    state of the world. Returns a dict suitable for passing to ``_invoke``.

    Special case: input-node jobs (``node_module='__input__*'``) hold metadata
    in ``inputs``, not refs. We pass those through unchanged.
    """
    if job is None:
        job = node_queue.get_node_job(job_id)
    if not job:
        return {}
    raw = job.get("inputs") or {}
    if (job.get("node_module") or "").startswith("__input__"):
        return dict(raw)
    run = run_store.get_run(job["run_id"]) or {}
    context = dict(run.get("context") or {})
    for sibling in node_queue.list_jobs_for_run(job["run_id"]):
        if sibling.get("status") == "completed" and sibling.get("context_delta"):
            context = {**context, sibling["node_id"]: sibling["context_delta"]}
    resolved: dict[str, Any] = {}
    for k, v in raw.items():
        try:
            resolved[k] = _resolve_ref(v, context)
        except KeyError:
            # Ref points at a key that's no longer (or never was) present. Fall
            # back to the snapshotted value so the worker still has *something*.
            resolved[k] = v
    return resolved


# ── Internal ─────────────────────────────────────────────────────────────


def _enqueue(run_id: str, node: dict[str, Any], run: dict[str, Any]) -> str:
    """Translate a workflow-JSON node into a queued node-job row.

    INSERT-only (Postgres-as-queue): the row IS the work — the migration-0006
    ``node_job_ready`` trigger fires the NOTIFY inside the insert txn, which is
    all a ``claim_worker`` loop needs to pick it up.
    """
    if _input_node(node):
        # Input nodes park on the CPU queue; a worker picks them up and
        # immediately transitions the run to awaiting_input.
        return node_queue.enqueue_node_job(
            run_id=run_id,
            node_id=node["id"],
            node_module=f"__input__{node['widget']}",
            queue="cpu",
            inputs={"widget": node.get("widget"), "target": node.get("target")},
            priority=int(run.get("priority", 100)),
        )

    # Resolve $from refs in the node's inputs against the run's context *at
    # enqueue time*. Any ref that can't resolve yet raises KeyError; we let it
    # bubble so the caller can mark the run failed with a clear error.
    raw = node.get("inputs", {}) or {}
    context = (run.get("context") or {})
    # Merge in sibling nodes' context_delta so refs like "detect.mask_path"
    # work after 'detect' completes.
    for sibling in node_queue.list_jobs_for_run(run_id):
        if sibling["status"] == "completed" and sibling.get("context_delta"):
            context = {**context, sibling["node_id"]: sibling["context_delta"]}
    resolved = {k: _resolve_ref(v, context) for k, v in raw.items()}

    queue = _queue_of(node)
    return node_queue.enqueue_node_job(
        run_id=run_id,
        node_id=node["id"],
        node_module=node.get("node") or node["id"],
        queue=queue,
        required_model=_required_model(node),
        inputs=resolved,
        priority=int(run.get("priority", 100)),
        pipeline_name=node.get("pipeline_name"),
    )

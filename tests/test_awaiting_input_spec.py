"""``dispatcher.on_node_awaiting_input`` must persist an ``input_spec`` the
frontend can render, and ``_build_input_spec`` builds the widget-specific
payload.

Drives the engine dispatcher + an in-test dict-backed workflow provider (no
filesystem registry) + the engine run_store. The widget builder is generic
engine code (it reads files from the run out_dir), so it ports verbatim; the
``_build_input_spec``-direct tests don't even need a provider.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

import pytest

import queue_workflows
from queue_workflows import dispatcher, run_store
from queue_workflows.db import connection


@pytest.fixture
def provider():
    """Dict-backed workflow provider. Returns a handle exposing
    ``.workflows`` / ``.pipelines`` dicts the test fills in."""
    workflows: dict[str, dict] = {}
    pipelines: dict[str, dict] = {}

    queue_workflows.set_workflow_provider(
        lambda n: workflows[n], lambda n: pipelines[n],
    )

    class _H:
        pass

    h = _H()
    h.workflows = workflows
    h.pipelines = pipelines
    return h


def _insert_run(workflow_name: str, out_dir: Path) -> str:
    run_id = str(uuid.uuid4())
    run_store.insert_run(
        run_id=run_id, workflow_name=workflow_name,
        out_dir=str(out_dir), status="running", mode="node",
    )
    return run_id


def _job_spec(run_id: str, node_id: str) -> dict | None:
    with connection() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT input_spec FROM workflow_node_jobs "
            "WHERE run_id = %s AND node_id = %s",
            (run_id, node_id),
        )
        row = cur.fetchone()
        return row["input_spec"] if row else None


def _job_status(run_id: str, node_id: str) -> str | None:
    with connection() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT status FROM workflow_node_jobs "
            "WHERE run_id = %s AND node_id = %s",
            (run_id, node_id),
        )
        row = cur.fetchone()
        return row["status"] if row else None


def _insert_awaiting_job(run_id: str, node_id: str, target: str = "x") -> None:
    with connection() as conn, conn.cursor() as cur:
        cur.execute(
            "INSERT INTO workflow_node_jobs "
            "(id, run_id, node_id, node_module, queue, status, inputs) "
            "VALUES (%s, %s, %s, 'noop', 'cpu', 'awaiting_input', %s::jsonb)",
            (
                f"{run_id}_{node_id}_{uuid.uuid4().hex[:6]}",
                run_id, node_id, json.dumps({"target": target}),
            ),
        )


# ── on_node_awaiting_input via the dispatcher ─────────────────────────────


def test_on_node_awaiting_input_persists_minimum_spec_for_choose_one(tmp_path, provider):
    name = "_aws_choose_one"
    provider.workflows[name] = {
        "name": name, "mode": "node",
        "steps": [
            {"id": "pick", "kind": "input", "widget": "choose_one",
             "prompt": "Pick one", "target": "chosen",
             "source": {"$value": ["a", "b", "c"]}, "depends_on": []},
        ],
    }
    provider.pipelines[name] = {"name": name, "nodes": []}
    run_id = _insert_run(name, tmp_path / "out")
    _insert_awaiting_job(run_id, "pick", target="chosen")

    dispatcher.on_node_awaiting_input(run_id, "pick")

    spec = _job_spec(run_id, "pick")
    assert spec is not None
    assert spec["step_id"] == "pick"
    assert spec["widget"] == "choose_one"
    assert spec["prompt"] == "Pick one"
    assert spec["target"] == "chosen"
    assert spec["multiple"] is False
    assert spec["options"] == ["a", "b", "c"]


def test_on_node_awaiting_input_resolves_source_files_from_disk(tmp_path, provider):
    name = "_aws_files"
    provider.workflows[name] = {
        "name": name, "mode": "node",
        "steps": [
            {"id": "svinfer", "kind": "pipeline", "pipeline": name, "inputs": {}},
            {"id": "pick_pano", "kind": "input", "widget": "choose_one",
             "prompt": "Pick a pano", "target": "selected_pano",
             "source": {
                 "$from": "svinfer.files",
                 "$filter": {
                     "kind:eq": "image",
                     "rel_path:matches": r"annotated_pano_\d+\.png$",
                 },
             },
             "depends_on": []},
        ],
    }
    provider.pipelines[name] = {
        "name": name,
        "nodes": [{"id": "detect", "node": "fake_detect", "depends_on": [], "gpu": False}],
    }

    out_dir = tmp_path / "out"
    step_dir = out_dir / "svinfer" / "detect" / "views"
    step_dir.mkdir(parents=True)
    (step_dir / "annotated_pano_0.png").write_bytes(b"\x89PNG\r\n\x1a\n fake")
    (step_dir / "annotated_pano_1.png").write_bytes(b"\x89PNG\r\n\x1a\n fake")
    (step_dir / "detections_pano_0.json").write_text("{}")

    run_id = _insert_run(name, out_dir)
    _insert_awaiting_job(run_id, "pick_pano", target="selected_pano")
    dispatcher.on_node_awaiting_input(run_id, "pick_pano")

    spec = _job_spec(run_id, "pick_pano")
    assert spec is not None
    rel_paths = sorted(o["rel_path"] for o in spec["options"])
    assert rel_paths == [
        "svinfer/detect/views/annotated_pano_0.png",
        "svinfer/detect/views/annotated_pano_1.png",
    ]
    for o in spec["options"]:
        assert o["kind"] == "image"
        assert o["abs_path"].startswith(str(step_dir))


def test_input_spec_rel_paths_survive_the_file_endpoint_roundtrip(tmp_path, provider):
    name = "_aws_roundtrip"
    provider.workflows[name] = {
        "name": name, "mode": "node",
        "steps": [
            {"id": "svinfer", "kind": "pipeline", "pipeline": name, "inputs": {}},
            {"id": "pick_pano", "kind": "input", "widget": "choose_one",
             "prompt": "Pick", "target": "selected_pano",
             "source": {"$from": "svinfer.files", "$filter": {"kind:eq": "image"}},
             "depends_on": []},
        ],
    }
    provider.pipelines[name] = {
        "name": name,
        "nodes": [{"id": "detect", "node": "fake_detect", "depends_on": [], "gpu": False}],
    }

    out_dir = tmp_path / "out"
    (out_dir / "svinfer" / "detect" / "views").mkdir(parents=True)
    (out_dir / "svinfer" / "detect" / "views" / "annotated_pano_0.png").write_bytes(
        b"\x89PNG\r\n\x1a\n fake",
    )
    run_id = _insert_run(name, out_dir)
    _insert_awaiting_job(run_id, "pick_pano", target="selected_pano")
    dispatcher.on_node_awaiting_input(run_id, "pick_pano")

    spec = _job_spec(run_id, "pick_pano")
    assert spec and spec["options"]
    for opt in spec["options"]:
        joined = Path(out_dir) / opt["rel_path"]
        assert joined.is_file()


def test_on_node_awaiting_input_passes_through_pick_or_upload_library(tmp_path, provider):
    name = "_aws_pickupload"
    provider.workflows[name] = {
        "name": name, "mode": "node",
        "steps": [
            {"id": "upload_ref", "kind": "input", "widget": "pick_or_upload",
             "prompt": "Pick a reference or upload", "target": "ref_image_path",
             "library": "fence_references",
             "accept": "image/jpeg,image/png,image/webp", "depends_on": []},
        ],
    }
    provider.pipelines[name] = {"name": name, "nodes": []}
    run_id = _insert_run(name, tmp_path / "out")
    _insert_awaiting_job(run_id, "upload_ref", target="ref_image_path")

    dispatcher.on_node_awaiting_input(run_id, "upload_ref")

    spec = _job_spec(run_id, "upload_ref")
    assert spec is not None
    assert spec["widget"] == "pick_or_upload"
    assert spec["library"] == "fence_references"
    assert spec["accept"] == "image/jpeg,image/png,image/webp"
    assert spec["target"] == "ref_image_path"


def test_on_node_awaiting_input_passes_through_file_upload_accept(tmp_path, provider):
    name = "_aws_fileupload"
    provider.workflows[name] = {
        "name": name, "mode": "node",
        "steps": [
            {"id": "upload", "kind": "input", "widget": "file_upload",
             "prompt": "Upload a file", "target": "path",
             "accept": "image/png", "depends_on": []},
        ],
    }
    provider.pipelines[name] = {"name": name, "nodes": []}
    run_id = _insert_run(name, tmp_path / "out")
    _insert_awaiting_job(run_id, "upload", target="path")

    dispatcher.on_node_awaiting_input(run_id, "upload")

    spec = _job_spec(run_id, "upload")
    assert spec is not None
    assert spec["widget"] == "file_upload"
    assert spec["accept"] == "image/png"


def test_on_node_awaiting_input_does_not_touch_run_scalar_fields(tmp_path, provider):
    name = "_aws_noscalar"
    provider.workflows[name] = {
        "name": name, "mode": "node",
        "steps": [
            {"id": "confirm", "kind": "input", "widget": "confirm",
             "prompt": "Proceed?", "target": "ok", "depends_on": []},
        ],
    }
    provider.pipelines[name] = {"name": name, "nodes": []}
    run_id = _insert_run(name, tmp_path / "out")
    _insert_awaiting_job(run_id, "confirm", target="ok")

    dispatcher.on_node_awaiting_input(run_id, "confirm")

    job_spec = _job_spec(run_id, "confirm")
    assert job_spec is not None
    assert job_spec["widget"] == "confirm"
    run = run_store.get_run(run_id)
    assert run["status"] != "awaiting_input"
    assert run["current_step_id"] is None
    assert run["input_spec"] is None


# ── parallel-input resume behaviours ─────────────────────────────────────


def test_resume_after_input_keeps_run_running_when_other_inputs_pending(tmp_path, provider):
    name = "_aws_parallel"
    provider.workflows[name] = {
        "name": name, "mode": "node",
        "steps": [
            {"id": "in_a", "kind": "input", "widget": "file_upload",
             "prompt": "A", "target": "a_path", "accept": "image/png", "depends_on": []},
            {"id": "in_b", "kind": "input", "widget": "file_upload",
             "prompt": "B", "target": "b_path", "accept": "image/png", "depends_on": []},
        ],
    }
    provider.pipelines[name] = {"name": name, "nodes": []}
    run_id = _insert_run(name, tmp_path / "out")
    with connection() as conn, conn.cursor() as cur:
        for nid in ("in_a", "in_b"):
            cur.execute(
                "INSERT INTO workflow_node_jobs "
                "(id, run_id, node_id, node_module, queue, status, inputs, input_spec) "
                "VALUES (%s, %s, %s, 'noop', 'cpu', 'awaiting_input', %s::jsonb, %s::jsonb)",
                (
                    f"{run_id}_{nid}", run_id, nid,
                    json.dumps({"target": f"{nid}_path"}),
                    json.dumps({"widget": "file_upload", "step_id": nid,
                                "target": f"{nid}_path"}),
                ),
            )
    run_store.update_run(run_id, status="awaiting_input", current_step_id="in_a")

    dispatcher.resume_after_input(run_id, "in_a", value="/path/a")

    assert _job_status(run_id, "in_a") == "completed"
    assert _job_status(run_id, "in_b") == "awaiting_input"
    assert _job_spec(run_id, "in_b") is not None


# ── _build_input_spec direct (no provider needed) ────────────────────────


def test_capture_3d_input_spec_surfaces_parcel_lat_lon(tmp_path):
    from queue_workflows.dispatcher import _build_input_spec

    step = {
        "id": "capture_3d", "kind": "input", "widget": "capture_3d",
        "prompt": "Pan to frame the property, then Capture.", "target": "image_path",
    }
    run = {
        "context": {"parcel": {"lat": 52.2297, "lon": 21.0122, "label": "Warsaw"}},
        "out_dir": str(tmp_path / "out"),
    }
    spec = _build_input_spec(step, run)

    assert spec["widget"] == "capture_3d"
    assert spec["target"] == "image_path"
    assert spec["lat"] == pytest.approx(52.2297)
    assert spec["lon"] == pytest.approx(21.0122)
    assert spec.get("parcel_label") == "Warsaw"


def test_capture_3d_input_spec_handles_missing_parcel_context(tmp_path):
    from queue_workflows.dispatcher import _build_input_spec

    step = {
        "id": "capture_3d", "kind": "input", "widget": "capture_3d",
        "prompt": "Frame the view.", "target": "image_path",
    }
    run = {"context": {}, "out_dir": str(tmp_path / "out")}
    spec = _build_input_spec(step, run)
    assert spec["widget"] == "capture_3d"
    assert spec["lat"] is None
    assert spec["lon"] is None


def test_assign_walls_input_spec_surfaces_parser_lanes_and_plates(tmp_path):
    from queue_workflows.dispatcher import _build_input_spec

    out_dir = tmp_path / "run_out"
    out_dir.mkdir()
    parse_root = out_dir / "parse_facade"
    parse_root.mkdir()
    for lane, n_walls in [("parse_pages", 1), ("parse_split", 3), ("parse_vlm", 1)]:
        d = parse_root / lane
        d.mkdir()
        (d / "walls").mkdir()
        manifest = {
            "parser": lane, "n_walls": n_walls,
            "walls": [
                {"id": f"w_{i}", "label": f"Wall {i+1}", "image_path": f"walls/elev_{i}.png"}
                for i in range(n_walls)
            ],
        }
        (d / "walls.json").write_text(json.dumps(manifest))
        for i in range(n_walls):
            (d / "walls" / f"elev_{i}.png").write_bytes(b"\x89PNG\r\n\x1a\n")

    diff_root = out_dir / "diffuse_3d"
    diff_root.mkdir()
    for sr_lane in ("sr_realesrgan", "sr_swinir", "sr_osediff"):
        d = diff_root / sr_lane
        d.mkdir()
        (d / "pano_0_upscaled.png").write_bytes(b"\x89PNG\r\n\x1a\n")

    step = {
        "id": "assign_walls", "kind": "input", "widget": "assign_walls",
        "target": "assignment", "prompt": "Pick parser, plate, and slot the walls.",
    }
    run = {"context": {}, "out_dir": str(out_dir)}
    spec = _build_input_spec(step, run)

    assert spec["widget"] == "assign_walls"
    assert spec["target"] == "assignment"
    by_parser = {p["parser"]: p for p in spec["parser_lanes"]}
    assert set(by_parser) == {"parse_pages", "parse_split", "parse_vlm"}
    assert by_parser["parse_split"]["n_walls"] == 3
    for w in by_parser["parse_split"]["walls"]:
        assert w["abs_path"].endswith(".png")
        assert "label" in w and "id" in w
    plate_lanes = [p["lane"] for p in spec["plate_candidates"]]
    assert sorted(plate_lanes) == ["sr_osediff", "sr_realesrgan", "sr_swinir"]
    for p in spec["plate_candidates"]:
        assert p["abs_path"].endswith("pano_0_upscaled.png")


def test_assign_walls_input_spec_handles_missing_parser_lanes(tmp_path):
    from queue_workflows.dispatcher import _build_input_spec
    out_dir = tmp_path / "run_out"
    out_dir.mkdir()
    parse_root = out_dir / "parse_facade"
    parse_root.mkdir()
    d = parse_root / "parse_pages"
    d.mkdir()
    (d / "walls.json").write_text(json.dumps({
        "parser": "parse_pages", "n_walls": 1,
        "walls": [{"id": "w_0", "label": "Wall 1", "image_path": "walls/elev_0.png"}],
    }))
    (d / "walls").mkdir()
    (d / "walls" / "elev_0.png").write_bytes(b"\x89PNG")

    step = {"id": "assign_walls", "kind": "input", "widget": "assign_walls",
            "target": "assignment"}
    run = {"context": {}, "out_dir": str(out_dir)}
    spec = _build_input_spec(step, run)

    by_parser = {p["parser"]: p for p in spec["parser_lanes"]}
    assert set(by_parser) == {"parse_pages"}
    assert by_parser["parse_pages"]["n_walls"] == 1


def test_assign_walls_input_spec_enumerates_five_lanes_dynamically(tmp_path):
    from queue_workflows.dispatcher import _build_input_spec

    out_dir = tmp_path / "run_out"
    out_dir.mkdir()
    parse_root = out_dir / "parse_facade"
    parse_root.mkdir()
    for lane, n_walls in [
        ("parse_pages", 1), ("parse_split", 3), ("parse_clusters", 4),
        ("parse_captions", 3), ("parse_vlm", 2),
    ]:
        d = parse_root / lane
        d.mkdir()
        (d / "walls").mkdir()
        (d / "walls.json").write_text(json.dumps({
            "parser": lane, "n_walls": n_walls,
            "walls": [
                {"id": f"w_{i}", "label": f"{lane} wall {i + 1}",
                 "image_path": f"walls/elev_{i}.png"}
                for i in range(n_walls)
            ],
        }))
        for i in range(n_walls):
            (d / "walls" / f"elev_{i}.png").write_bytes(b"\x89PNG")

    step = {"id": "assign_walls", "kind": "input", "widget": "assign_walls",
            "target": "assignment"}
    run = {"context": {}, "out_dir": str(out_dir)}
    spec = _build_input_spec(step, run)

    by_parser = {p["parser"]: p for p in spec["parser_lanes"]}
    assert set(by_parser) == {
        "parse_pages", "parse_split", "parse_clusters", "parse_captions", "parse_vlm",
    }
    assert len(spec["all_proposals"]) == 1 + 3 + 4 + 3 + 2
    gids = [p["global_id"] for p in spec["all_proposals"]]
    assert len(set(gids)) == len(gids)
    assert any(g.startswith("parse_clusters:") for g in gids)
    assert any(g.startswith("parse_captions:") for g in gids)


def test_paint_mask_source_wraps_scalar_abs_path_from_upstream_pick(tmp_path, provider):
    """Regression pin for the "ChooseOne pick is ignored by the next
    paint_mask step" bug.

    Workflow shape:
      pick_clean (choose_one, target=clean_plate_path)
        ↓
      pick_install_area (paint_mask, source={"$from": "pick_clean.clean_plate_path"})

    After pick_clean completes with a scalar string in its
    ``context_delta`` (the abs_path the operator picked), the
    dispatcher's paint_mask block must expose that exact path as the
    spec's ``source_abs_path`` — wrapping the scalar into a
    single-element file-info list so the widget renders THAT image,
    not "whichever lane happened to come back first from the
    remove_fence.files scan."

    Without the wrap the previous behaviour fell through into the
    ``elif not isinstance(options, list): options = []`` branch — the
    widget then had ``source_options=[]`` and (via PaintMask's
    ``sourceRelPath ?? null``) rendered nothing or the lane-0 fallback.
    """
    from queue_workflows import dispatcher, node_queue

    name = "_pick_then_paint"
    provider.workflows[name] = {
        "name": name, "mode": "node",
        "steps": [
            {"id": "pick_clean", "kind": "input", "widget": "choose_one",
             "prompt": "pick", "target": "clean_plate_path",
             "depends_on": []},
            {"id": "pick_install_area", "kind": "input", "widget": "paint_mask",
             "prompt": "paint", "target": "install_area_path",
             "source": {"$from": "pick_clean.clean_plate_path"},
             "depends_on": ["pick_clean"]},
        ],
    }

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    # Simulate the lane outputs the operator just picked from.
    lane_dir = out_dir / "remove_fence" / "lane_sd15_inpaint"
    lane_dir.mkdir(parents=True)
    picked = lane_dir / "no_fence.jpg"
    picked.write_bytes(b"\xff\xd8\xff fake")

    run_id = _insert_run(name, out_dir)

    # Complete pick_clean with the operator's scalar pick — exactly
    # what ``resume_after_input`` writes for a single-select ChooseOne.
    pick_job = node_queue.enqueue_node_job(
        run_id=run_id, node_id="pick_clean",
        node_module="__input__choose_one", queue="cpu",
        inputs={"widget": "choose_one", "target": "clean_plate_path"},
        priority=50,
    )
    # Drive the lifecycle directly via SQL — no claim_worker is running.
    # The dispatcher's context-builder only cares about status='completed'
    # + context_delta, so we set just those.
    import json
    with connection() as conn, conn.cursor() as cur:
        cur.execute(
            "UPDATE workflow_node_jobs SET status='completed', "
            "context_delta=%s::jsonb, finished_at=now() WHERE id=%s",
            (json.dumps({"clean_plate_path": str(picked)}), pick_job),
        )
        conn.commit()

    _insert_awaiting_job(run_id, "pick_install_area", target="install_area_path")
    dispatcher.on_node_awaiting_input(run_id, "pick_install_area")

    spec = _job_spec(run_id, "pick_install_area")
    assert spec is not None, "paint_mask spec should be persisted"
    options = spec.get("source_options") or []
    assert len(options) == 1, (
        f"source_options must wrap the scalar pick into ONE option, "
        f"got {options!r}"
    )
    assert options[0]["abs_path"] == str(picked), (
        f"source_abs_path must equal the picked lane file; "
        f"got {options[0].get('abs_path')!r}"
    )
    assert spec.get("source_abs_path") == str(picked)
    # rel_path is the path relative to ``out_dir`` — the same convention
    # the file-endpoint uses when streaming the artefact.
    assert spec.get("source_rel_path") == "remove_fence/lane_sd15_inpaint/no_fence.jpg"

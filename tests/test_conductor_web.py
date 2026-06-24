"""``queue-conductor-web`` — the read-only operator web view.

Asserts the rendered HTML reflects the NEW queueing model (ONE shared cpu/gpu
queue across projects + ingest, NOT per-project queues), that the project filter
scopes every panel, and that the server answers a real HTTP GET with 200 + HTML.
Runs on both backends (pure engine primitives). Visual correctness is covered by
the headless-Chrome audit (see worklog/conductor-web-ui.md).
"""

from __future__ import annotations

import threading
import urllib.request
import uuid
from http.server import ThreadingHTTPServer

import queue_workflows
from queue_workflows import node_queue, run_store
from queue_workflows_conductor import web


def _run(project: str) -> str:
    rid = str(uuid.uuid4())
    run_store.insert_run(run_id=rid, workflow_name="_wf", out_dir="/t",
                         status="running", mode="node", project=project)
    return rid


def _seed() -> None:
    # alpha: a queued cpu job + a queued gpu job + a fresh gpu worker
    node_queue.enqueue_node_job(run_id=_run("alpha"), node_id="a", node_module="m",
                                queue="cpu", project="alpha")
    node_queue.enqueue_node_job(run_id=_run("alpha"), node_id="g", node_module="m",
                                queue="gpu", required_model="sdxl", project="alpha")
    node_queue.upsert_worker_heartbeat(host_label="box1", queue="gpu",
                                       project="alpha", concurrency=2,
                                       current_model="sdxl")
    # beta: one queued cpu job
    node_queue.enqueue_node_job(run_id=_run("beta"), node_id="b", node_module="m",
                                queue="cpu", project="beta")
    # an ingest job
    queue_workflows.register_ingest_task("noop", lambda reason: {})
    node_queue.enqueue_ingest_job(task_name="noop", queue="fetch", project="alpha")


def test_dashboard_shows_shared_queues_and_project_filter():
    _seed()
    html = web.render_dashboard(None)  # broker-wide
    assert html.startswith("<!doctype html>") and "</html>" in html
    # the NEW model: shared cpu/gpu queues, not per-project
    assert "shared queue" in html and "Fleet" in html and "Ingest" in html
    assert "cpu" in html and "gpu" in html
    # both projects appear in the filter bar; broker-wide counts include both
    assert "alpha" in html and "beta" in html
    # broker-wide cpu_queued = 2 (alpha + beta)
    assert ">2<" in html  # the cpu queued stat
    # the fleet worker + its project + warm model render
    assert "box1" in html and "sdxl" in html


def test_project_filter_scopes_the_view():
    _seed()
    alpha = web.render_dashboard("alpha")
    assert "project = alpha" in alpha
    # beta's cpu job is excluded → alpha cpu_queued = 1, and beta's worker absent
    beta = web.render_dashboard("beta")
    assert "project = beta" in beta
    # alpha has a gpu worker; beta has none
    assert "box1" in alpha and "box1" not in beta


def test_render_is_escaped(monkeypatch):
    # a hostile project/model value must not break out of the HTML
    node_queue.upsert_worker_heartbeat(
        host_label="<script>x</script>", queue="gpu", project="p", concurrency=1,
    )
    html = web.render_dashboard(None)
    assert "<script>x</script>" not in html
    assert "&lt;script&gt;" in html


def test_server_answers_http_get():
    _seed()
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), web.ConductorWebHandler)
    port = httpd.server_address[1]
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=5) as r:
            assert r.status == 200
            body = r.read().decode("utf-8")
        assert "broker_parrot" in body and "shared queue" in body
        # unknown path → 404
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/nope", timeout=5)
            assert False, "expected 404"
        except urllib.error.HTTPError as e:
            assert e.code == 404
        # ?project= selects the '' single-tenant sentinel (keep_blank_values),
        # NOT the broker-wide 'all projects' view.
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/?project=", timeout=5) as r:
            empty_body = r.read().decode("utf-8")
        assert "project = (default)" in empty_body
        assert "· all projects" not in empty_body
    finally:
        httpd.shutdown()
        httpd.server_close()

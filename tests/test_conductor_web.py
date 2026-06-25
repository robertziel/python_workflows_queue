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


def test_conductor_backend_flag_selects_store():
    """The standalone conductor scripts (queue-conductor / queue-conductor-web) have
    no host configure(), so they self-select the store via --db-backend. With the
    v1.0.0 sqlite default, a Postgres fleet view needs --db-backend pg, else the pg
    DSN is read as a SQLite path. Lock the shared helper."""
    from queue_workflows_conductor.conductor import _configure_backend
    _configure_backend("sqlite", None)
    assert queue_workflows.get_config().db_backend == "sqlite"
    _configure_backend("pg", None)
    assert queue_workflows.get_config().db_backend == "pg"


def test_recent_jobs_unifies_node_and_ingest():
    """The job-dashboard-style activity feed primitive: one list across BOTH job
    families, project-aware, with the unified lifecycle shape."""
    _seed()
    jobs = node_queue.recent_jobs(limit=50)
    kinds = {j["kind"] for j in jobs}
    assert "node" in kinds and "ingest" in kinds
    names = {j["name"] for j in jobs}
    assert {"a", "b", "g"} <= names      # node ids
    assert "noop" in names               # ingest task name
    for col in ("kind", "name", "queue", "status", "project", "worker", "retries", "recency"):
        assert col in jobs[0]
    # project filter scopes the feed
    beta = node_queue.recent_jobs(project="beta", limit=50)
    assert {j["project"] for j in beta} == {"beta"}
    assert {j["name"] for j in beta} == {"b"}
    # status filter (a 'dead jobs' view would pass status='failed')
    assert node_queue.recent_jobs(status="failed", limit=50) == []


def test_dashboard_has_overview_strip_and_recent_activity():
    """The job-dashboard-inspired additions render: the KPI overview strip + the
    recent-activity feed with status badges."""
    _seed()
    html = web.render_dashboard(None)
    assert "Overview" in html
    for kpi in ("Busy", "Enqueued", "Processed", "Failed", "Workers", "Projects"):
        assert kpi in html
    assert "Recent activity" in html
    assert 'class="kpi' in html      # the KPI strip
    assert 'class="badge' in html    # job-dashboard-style status pills
    assert "noop" in html            # the ingest job appears in the feed


def test_list_node_events_and_job_detail_render():
    """The job-detail timeline: list_node_events returns the per-attempt log and
    render_job shows the metadata + the events (our differentiator over a plain job dashboard)."""
    rid = _run("alpha")
    jid = node_queue.enqueue_node_job(run_id=rid, node_id="nd", node_module="m",
                                      queue="gpu", project="alpha")
    node_queue.record_node_event(run_id=rid, node_id="nd", job_id=jid,
                                 event_type="claimed", host_label="boxA", queue="gpu")
    node_queue.record_node_event(run_id=rid, node_id="nd", job_id=jid,
                                 event_type="completed", host_label="boxA",
                                 queue="gpu", elapsed_s=1.5)
    events = node_queue.list_node_events(jid)
    assert [e["event_type"] for e in events] == ["claimed", "completed"]
    html = web.render_job(node_queue.get_node_job(jid), events, kind="node")
    assert "Event timeline" in html
    assert "claimed" in html and "completed" in html and "boxA" in html
    # an unknown / ingest job has no per-attempt log
    assert node_queue.list_node_events(str(uuid.uuid4())) == []


def test_recent_jobs_retries_filter():
    """min_retries ⇒ a job dashboard's Retries view: node-jobs over the threshold only;
    ingest jobs (no retry counter) drop out."""
    rid = _run("alpha")
    node_queue.enqueue_node_job(run_id=rid, node_id="clean", node_module="m",
                                queue="cpu", project="alpha")
    jid = node_queue.enqueue_node_job(run_id=rid, node_id="flaky", node_module="m",
                                      queue="cpu", project="alpha")
    from queue_workflows.db import connection
    with connection() as c, c.cursor() as cur:
        cur.execute("UPDATE workflow_node_jobs SET watchdog_retries = 2 "
                    "WHERE id = %(id)s", {"id": jid})
        c.commit()
    retried = node_queue.recent_jobs(min_retries=1, limit=50)
    names = {j["name"] for j in retried}
    assert "flaky" in names and "clean" not in names
    assert all(j["kind"] == "node" for j in retried)   # ingest excluded


def test_dashboard_tabs_and_views():
    """The Recent-activity feed gains job-dashboard-style All/Retries/Dead tabs; the view
    param scopes the feed and marks the active tab."""
    _seed()
    html = web.render_dashboard(None, view="all")
    for tab in (">All<", ">Retries<", ">Dead<"):
        assert tab in html
    assert 'class="tab active"' in html
    dead = web.render_dashboard(None, view="dead")
    assert 'href="/?view=dead' in dead   # the tab links preserve the view
    # a bad view falls back to 'all' (no crash)
    assert web.render_dashboard(None, view="bogus").count("Recent activity") == 1


def test_writes_opt_in_renders_toggles():
    """Read-only by default; --enable-writes (writes_enabled) adds the per-worker
    ON/OFF toggle column."""
    _seed()
    assert "<th>control</th>" not in web.render_dashboard(None)
    on = web.render_dashboard(None, writes_enabled=True)
    assert "<th>control</th>" in on and 'action="/control"' in on


def test_write_actions_post_path():
    """do_POST is gated: 403 when disabled; with writes enabled, POST /control
    writes worker_controls (park/resume)."""
    import threading
    import urllib.error
    import urllib.parse
    import urllib.request
    from http.server import ThreadingHTTPServer

    from queue_workflows import worker_control

    data = urllib.parse.urlencode(
        {"host": "h1", "queue": "cpu", "desired_state": "off"}).encode()
    web.ConductorWebHandler.writes_enabled = False
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), web.ConductorWebHandler)
    port = httpd.server_address[1]
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        try:  # disabled → 403
            urllib.request.urlopen(f"http://127.0.0.1:{port}/control", data=data, timeout=5)
            assert False, "expected 403 when writes disabled"
        except urllib.error.HTTPError as e:
            assert e.code == 403
        web.ConductorWebHandler.writes_enabled = True  # opt in
        # CSRF guard: a cross-origin POST is rejected even with writes enabled
        try:
            req = urllib.request.Request(
                f"http://127.0.0.1:{port}/control", data=data,
                headers={"Origin": "http://evil.example"}, method="POST")
            urllib.request.urlopen(req, timeout=5)
            assert False, "expected 403 for cross-origin POST"
        except urllib.error.HTTPError as e:
            assert e.code == 403
        # same-origin (no Origin header) POST → the write lands
        urllib.request.urlopen(f"http://127.0.0.1:{port}/control", data=data, timeout=5)
        ctl = worker_control.get_worker_control("h1", "cpu")
        assert ctl is not None and ctl["desired_state"] == "off"
    finally:
        web.ConductorWebHandler.writes_enabled = False  # don't leak to other tests
        httpd.shutdown()
        httpd.server_close()


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

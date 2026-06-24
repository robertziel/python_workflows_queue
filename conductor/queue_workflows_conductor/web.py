"""``queue-conductor-web`` — a read-only operator WEB view of the new
shared-queue + project model.

The control plane historically rendered the fleet as a terminal table
(:mod:`queue_workflows_conductor.conductor`). This is the same READ-ONLY,
SINGLE-DB capacity view served as one HTML page, reflecting the **new version of
queueing**: ONE shared ``cpu`` + ONE shared ``gpu`` (+ host-defined ingest)
queue across ALL projects — partitioned by *resource*, tagged by *project* — NOT
a queue per project. Every panel takes a **project filter** (the headline change:
"keep the project name in the queue record for easy filtering").

Design / scope, on purpose (mirrors the conductor's existing constraints):

  * **READ-ONLY.** No park/resume/retry/cancel write controls — those, and the
    multi-DB networked aggregator across many app DBs, are the human-gated build
    noted in ``conductor.py``. This view is the conductor's existing single-DB
    read scope, as HTML.
  * **Zero new runtime deps.** Pure stdlib ``http.server`` + server-rendered
    HTML (no web framework, no JS) — consistent with the library's "Postgres is
    the only hard dependency" ethos, and no JS means no client console errors.
  * **SINGLE-DB.** Reads whatever DB the client's ``db_url_env`` points at via
    the engine primitives (:func:`node_queue.snapshot`,
    :func:`node_queue.ingest_snapshot`, :func:`node_queue.fleet_snapshot`,
    :func:`node_queue.list_projects`) — all project-aware (migration 0017).

Usage::

    queue-conductor-web                       # http://127.0.0.1:8787
    queue-conductor-web --host 0.0.0.0 --port 9000
"""

from __future__ import annotations

import html
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from queue_workflows import node_queue

REFRESH_S = 5  # client-side meta refresh (no JS)


# ── pure render (testable without a server) ─────────────────────────────────


def _esc(v: Any) -> str:
    return html.escape("" if v is None else str(v))


def _proj_label(p: str) -> str:
    return "(default)" if p == "" else p


def _project_filter_bar(active: str | None) -> str:
    """Row of project filters. ``active is None`` ⇒ the broker-wide 'All' view."""
    projects = node_queue.list_projects()
    parts = []
    cls = "f active" if active is None else "f"
    parts.append(f'<a class="{cls}" href="/">All projects</a>')
    for p in projects:
        sel = active is not None and active == p
        href = "/?project=" + urllib.parse.quote(p)
        parts.append(
            f'<a class="{"f active" if sel else "f"}" href="{_esc(href)}">'
            f"{_esc(_proj_label(p))}</a>"
        )
    return '<div class="filters">' + "".join(parts) + "</div>"


def _queue_card(name: str, counts: dict[str, int]) -> str:
    cells = "".join(
        f'<div class="stat {s}"><b>{counts.get(f"{name}_{s}", 0)}</b>'
        f"<span>{s}</span></div>"
        for s in ("queued", "running", "completed", "failed", "cancelled")
    )
    return (
        f'<div class="card"><h3>{_esc(name)} <em>shared queue</em></h3>'
        f'<div class="stats">{cells}</div></div>'
    )


def _ingest_cards(ing: dict[str, Any]) -> str:
    # ingest_snapshot groups worker_heartbeats by queue, so the cpu/gpu DAG
    # resource queues leak in here — they belong to the shared-queue section
    # above, not 'ingest'. Show only genuine (non-cpu/gpu) ingest queues.
    queues = {
        q: st for q, st in (ing or {}).get("queues", {}).items()
        if q not in ("cpu", "gpu")
    }
    if not queues:
        return '<div class="card muted">no ingest queues</div>'
    out = []
    for q, st in sorted(queues.items()):
        cells = "".join(
            f'<div class="stat {s}"><b>{st.get(s, 0)}</b><span>{s}</span></div>'
            for s in ("queued", "running", "completed", "failed")
        )
        cells += f'<div class="stat workers"><b>{st.get("workers", 0)}</b><span>workers</span></div>'
        out.append(
            f'<div class="card"><h3>{_esc(q)} <em>ingest</em></h3>'
            f'<div class="stats">{cells}</div></div>'
        )
    return "".join(out)


def _fleet_table(fleet: list[dict[str, Any]]) -> str:
    if not fleet:
        return '<p class="muted">no workers reporting</p>'
    rows = []
    for w in fleet:
        if w.get("flagged_dead"):
            status, scls = "DEAD", "dead"
        elif not w.get("fresh"):
            status, scls = "stale", "stale"
        else:
            status, scls = "ok", "ok"
        rows.append(
            "<tr>"
            f"<td>{_esc(w.get('host_label'))}</td>"
            f"<td>{_esc(w.get('queue'))}</td>"
            f"<td>{_esc(_proj_label(w.get('project') or ''))}</td>"
            f"<td>{_esc(w.get('concurrency'))}</td>"
            f"<td>{_esc(w.get('current_model') or '—')}</td>"
            f'<td class="st {scls}">{status}</td>'
            "</tr>"
        )
    return (
        '<table class="fleet"><thead><tr>'
        "<th>host</th><th>queue</th><th>project</th><th>conc</th>"
        "<th>model</th><th>status</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


_CSS = """
:root{color-scheme:light}
*{box-sizing:border-box}
body{margin:0;font:14px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",
 Helvetica,Arial,sans-serif;background:#f6f8fa;color:#1f2328}
header{padding:18px 22px;border-bottom:1px solid #d0d7de;background:#fff}
h1{margin:0;font-size:17px;letter-spacing:.2px}
h1 small{color:#656d76;font-weight:400;font-size:12px;margin-left:8px}
main{padding:22px;max-width:1100px;margin:0 auto}
h2{font-size:12px;text-transform:uppercase;letter-spacing:1.2px;color:#656d76;
 margin:26px 0 10px;border-bottom:1px solid #d8dee4;padding-bottom:6px}
.filters{margin-top:12px;display:flex;flex-wrap:wrap;gap:8px}
a.f{padding:4px 11px;border:1px solid #d0d7de;border-radius:14px;color:#424a53;
 text-decoration:none;font-size:12px;background:#fff}
a.f.active{background:#2da44e;border-color:#2da44e;color:#fff}
.cards{display:flex;flex-wrap:wrap;gap:14px}
.card{background:#fff;border:1px solid #d0d7de;border-radius:9px;padding:14px 16px;
 min-width:240px;box-shadow:0 1px 0 rgba(27,31,36,.04)}
.card.muted{color:#656d76}
.card h3{margin:0 0 10px;font-size:14px}
.card h3 em{color:#656d76;font-style:normal;font-weight:400;font-size:11px}
.stats{display:flex;gap:14px;flex-wrap:wrap}
.stat{display:flex;flex-direction:column;align-items:flex-start}
.stat b{font-size:20px;line-height:1;color:#1f2328}
.stat span{font-size:10px;text-transform:uppercase;color:#656d76;margin-top:3px}
.stat.running b{color:#1a7f37}.stat.completed b{color:#1a7f37}
.stat.queued b{color:#9a6700}.stat.failed b{color:#d1242f}.stat.workers b{color:#0969da}
table.fleet{border-collapse:collapse;width:100%;font-size:13px;background:#fff;
 border:1px solid #d0d7de;border-radius:9px;overflow:hidden}
.fleet th{text-align:left;color:#656d76;font-weight:600;padding:8px 12px;
 border-bottom:1px solid #d0d7de;background:#f6f8fa}
.fleet td{padding:8px 12px;border-bottom:1px solid #eaeef2}
.st.ok{color:#1a7f37}.st.stale{color:#9a6700}.st.dead{color:#d1242f;font-weight:700}
.muted{color:#656d76}
footer{color:#848d97;font-size:11px;padding:18px 22px;border-top:1px solid #d8dee4}
"""


def render_dashboard(project: str | None = None, *, stale_after_s: float = 30.0) -> str:
    """Render the full dashboard HTML for ``project`` (``None`` ⇒ all projects).
    Pure — fetches the engine snapshots and returns a complete HTML document."""
    snap = node_queue.snapshot(project=project)
    ingest = node_queue.ingest_snapshot(project=project)
    fleet = node_queue.fleet_snapshot(stale_after_s=stale_after_s, project=project)
    counts = snap.get("counts", {})
    scope = "all projects" if project is None else f"project = {_proj_label(project)}"
    body = f"""<!doctype html><html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="refresh" content="{REFRESH_S}">
<title>broker_parrot — fleet + queue</title><style>{_CSS}</style></head><body>
<header>
  <h1>broker_parrot <small>fleet + queue · {_esc(scope)}</small></h1>
  <div class="muted" style="margin-top:6px;font-size:12px">
    one shared <b>cpu</b> + <b>gpu</b> queue across all projects — tagged by project, filter below
  </div>
  {_project_filter_bar(project)}
</header>
<main>
  <h2>Queues — shared cpu / gpu</h2>
  <div class="cards">{_queue_card('cpu', counts)}{_queue_card('gpu', counts)}</div>
  <h2>Ingest queues</h2>
  <div class="cards">{_ingest_cards(ingest)}</div>
  <h2>Fleet — workers</h2>
  {_fleet_table(fleet)}
</main>
<footer>read-only · single-DB · auto-refresh {REFRESH_S}s · queue-conductor-web</footer>
</body></html>"""
    return body


def render_error(exc: BaseException) -> str:
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        f"<title>conductor error</title><style>{_CSS}</style></head><body>"
        "<header><h1>broker_parrot <small>conductor — error</small></h1></header>"
        f"<main><p class='st dead'>could not read the queue database:</p>"
        f"<pre class='muted'>{_esc(exc)}</pre></main></body></html>"
    )


# ── server ──────────────────────────────────────────────────────────────────


class ConductorWebHandler(BaseHTTPRequestHandler):
    server_version = "queue-conductor-web/0.1"
    stale_after_s = 30.0

    def _send(self, code: int, body: str) -> None:
        payload = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path not in ("/", "/index.html"):
            self._send(404, "<h1>404</h1>")
            return
        # keep_blank_values so the single-tenant sentinel project '' is a real,
        # selectable filter (?project=) and not silently dropped to the all view.
        q = urllib.parse.parse_qs(parsed.query, keep_blank_values=True)
        project = q["project"][0] if "project" in q else None
        try:
            self._send(200, render_dashboard(project, stale_after_s=self.stale_after_s))
        except Exception as exc:  # noqa: BLE001 — a DB blip must not kill the server
            self._send(500, render_error(exc))

    def log_message(self, *args: object) -> None:  # silence default stderr spam
        pass


def serve(host: str = "127.0.0.1", port: int = 8787, *, stale_after_s: float = 30.0) -> None:
    ConductorWebHandler.stale_after_s = stale_after_s
    httpd = ThreadingHTTPServer((host, port), ConductorWebHandler)
    print(f"queue-conductor-web serving on http://{host}:{port}  (Ctrl-C to stop)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()


def main(argv: list[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser(
        prog="queue-conductor-web",
        description="Read-only web view of the shared-queue + project fleet.",
    )
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8787)
    p.add_argument("--stale-after", type=float, default=30.0,
                   help="seconds before a worker heartbeat is 'stale' (default 30)")
    args = p.parse_args(argv)
    serve(args.host, args.port, stale_after_s=args.stale_after)
    return 0

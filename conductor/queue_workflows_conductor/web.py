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

  * **READ-ONLY by default; opt-in writes via ``--enable-writes``.** The default is
    the pure single-DB read view. With ``--enable-writes`` it gains two gated
    operator actions — worker ON/OFF (``worker_control.set_worker_control``) and
    re-queue a running node-job (``node_queue.requeue_job_for_retry``) — over a
    POST/redirect/GET path, and nothing else (no cancel/delete). The multi-DB
    networked aggregator across many app DBs remains the human-gated build noted in
    ``conductor.py``.
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

import datetime
import html
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from queue_workflows import node_queue

REFRESH_S = 5  # client-side meta refresh (no JS)

# One process-wide background hw feed (started in serve() for a pg broker). The
# conductor is request-response, but hw_metrics is a persistent LISTEN stream — so a
# single daemon HwFeed holds the latest sample per host and each request reads it.
_HW_FEED: Any = None


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


def _control_cell(host: str, queue: str, control: dict[str, Any] | None) -> str:
    """An ON/OFF toggle form (Sidekiq's quiet/stop) — only rendered when writes are
    enabled. A worker absent from worker_controls is treated as ON."""
    off = bool(control) and control.get("desired_state") == "off"
    nxt = "on" if off else "off"
    label = "OFF" if off else "ON"
    btn = "turn on" if off else "turn off"
    cls = "toggle off" if off else "toggle on"
    return (
        f'<form method="post" action="/control" class="ctl">'
        f'<input type="hidden" name="host" value="{_esc(host)}">'
        f'<input type="hidden" name="queue" value="{_esc(queue)}">'
        f'<input type="hidden" name="desired_state" value="{nxt}">'
        f'<span class="state {cls}">{label}</span>'
        f'<button type="submit">{btn}</button></form>'
    )


def _fleet_table(fleet: list[dict[str, Any]], *, controls: dict | None = None,
                 writes_enabled: bool = False) -> str:
    if not fleet:
        return '<p class="muted">no workers reporting</p>'
    controls = controls or {}
    rows = []
    for w in fleet:
        if w.get("flagged_dead"):
            status, scls = "DEAD", "dead"
        elif not w.get("fresh"):
            status, scls = "stale", "stale"
        else:
            status, scls = "ok", "ok"
        host, queue = w.get("host_label"), w.get("queue")
        ctl = (f"<td>{_control_cell(host, queue, controls.get((host, queue)))}</td>"
               if writes_enabled else "")
        rows.append(
            "<tr>"
            f"<td>{_esc(host)}</td>"
            f"<td>{_esc(queue)}</td>"
            f"<td>{_esc(_proj_label(w.get('project') or ''))}</td>"
            f"<td>{_esc(w.get('concurrency'))}</td>"
            f"<td>{_esc(w.get('current_model') or '—')}</td>"
            f'<td class="st {scls}">{status}</td>'
            f"{ctl}"
            "</tr>"
        )
    ctlth = "<th>control</th>" if writes_enabled else ""
    return (
        '<table class="fleet"><thead><tr>'
        "<th>host</th><th>queue</th><th>project</th><th>conc</th>"
        f"<th>model</th><th>status</th>{ctlth}</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


# ── live hardware panel — moving time-series sparklines (cpu/gpu/ram) ──────────

def _fmt_pct(p: Any) -> str:
    try:
        return f"{float(p):.0f}%"
    except (TypeError, ValueError):
        return "—"


def _gb(mb: Any) -> str:
    try:
        return f"{float(mb) / 1024:.0f}"
    except (TypeError, ValueError):
        return "—"


def _clampf(v: Any) -> float:
    try:
        return max(0.0, min(100.0, float(v)))
    except (TypeError, ValueError):
        return 0.0


def _spark(values: list[Any], *, color: str = "#0969da") -> str:
    """A no-JS inline-SVG **dotted** moving graph for a 0–100 series — one dot per
    sample, NO connecting line (the project-dashboard style), oldest on the LEFT /
    newest on the RIGHT, so the dotted band scrolls left as the page meta-refreshes.
    ``None`` ⇒ 0. Fixed 120×28 viewBox so the dots stay round (no aspect stretch)."""
    n = len(values)
    if n == 0:
        return '<svg class="spark" viewBox="0 0 120 28" width="120" height="28"></svg>'
    W, H = 120.0, 28.0
    step = (W / (n - 1)) if n > 1 else 0.0
    pts = [((i * step) if n > 1 else W, H - (_clampf(v) / 100.0) * H)
           for i, v in enumerate(values)]
    dots = "".join(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="1.1"/>' for x, y in pts)
    lx, ly = pts[-1]
    return (f'<svg class="spark" viewBox="0 0 120 28" width="120" height="28" '
            f'style="color:{color}">'
            f'<g fill="currentColor" opacity=".7">{dots}</g>'
            f'<circle cx="{lx:.1f}" cy="{ly:.1f}" r="2.2" fill="currentColor"/></svg>')


def _ram_pct(s: dict[str, Any]) -> Any:
    u, t = s.get("ram_used_mb"), s.get("ram_total_mb")
    return (100.0 * float(u) / float(t)) if (u and t) else None


def _gpu_max(s: dict[str, Any]) -> Any:
    uses = [g.get("use_pct") for g in (s.get("gpus") or [])
            if g.get("use_pct") is not None]
    return max(uses) if uses else None


def _hw_row(label: str, series: list[Any], cur: Any, sub: str, color: str) -> str:
    return (f'<div class="hwrow"><span class="hwk">{_esc(label)}</span>'
            f"{_spark(series, color=color)}"
            f'<span class="hwv">{_fmt_pct(cur)}'
            f'{(" <em>" + _esc(sub) + "</em>") if sub else ""}</span></div>')


def _hw_panel(history: dict[str, list[dict[str, Any]]] | None) -> str:
    """Per-host hardware cards with **moving time-series sparklines** (CPU% / GPU% /
    RAM%) from the broker's ``hw_metrics`` stream via :meth:`HwFeed.history_by_host`.
    Pure — given ``{host: [sample, …]}`` (oldest→newest); the newest sample is the
    rightmost point, so each meta-refresh scrolls the dotted line left over time."""
    if not history:
        return ('<p class="muted">no hardware telemetry yet — the conductor reads the '
                "broker’s <code>hw_metrics</code> stream (a Postgres broker whose "
                "gpu workers publish per-host cpu/gpu/ram).</p>")
    cards = []
    for host in sorted(history):
        samples = history[host] or []
        if not samples:
            continue
        latest = samples[-1] or {}
        stale = bool(latest.get("stale"))
        gpus = latest.get("gpus") or []
        vu = sum(int(g.get("vram_used_mb") or 0) for g in gpus) or None
        vt = sum(int(g.get("vram_total_mb") or 0) for g in gpus) or None
        gpu_sub = (f"{_gb(vu)}/{_gb(vt)} GB" if (vu and vt)
                   else ("none" if not gpus else ""))
        glabel = f"GPU×{len(gpus)}" if len(gpus) > 1 else "GPU"
        ram_u, ram_t = latest.get("ram_used_mb"), latest.get("ram_total_mb")
        rows = [
            _hw_row("CPU", [s.get("cpu_percent") for s in samples],
                    latest.get("cpu_percent"), "", "#0969da"),
            _hw_row(glabel, [_gpu_max(s) for s in samples],
                    _gpu_max(latest), gpu_sub, "#1a7f37"),
            _hw_row("RAM", [_ram_pct(s) for s in samples],
                    _ram_pct(latest), f"{_gb(ram_u)}/{_gb(ram_t)} GB", "#9a6700"),
        ]
        tag = ('<em class="stale">stale</em>' if stale
               else '<em class="live">live</em>')
        cards.append(f'<div class="card hw{" stale" if stale else ""}">'
                     f"<h3>{_esc(host)} {tag}</h3>{''.join(rows)}</div>")
    return '<div class="cards">' + "".join(cards) + "</div>"


# ── Sidekiq-inspired pieces (a KPI strip + status badges + a recent-activity feed)

def _ago(dt: Any) -> str:
    """Compact relative age (Sidekiq shows 'x ago'). Tolerant of naive/aware/None."""
    if dt is None:
        return "—"
    try:
        now = datetime.datetime.now(datetime.timezone.utc)
        if getattr(dt, "tzinfo", None) is None:
            dt = dt.replace(tzinfo=datetime.timezone.utc)
        s = (now - dt).total_seconds()
    except Exception:
        return _esc(dt)
    if s < 1:
        return "now"
    if s < 60:
        return f"{int(s)}s ago"
    if s < 3600:
        return f"{int(s // 60)}m ago"
    if s < 86400:
        return f"{int(s // 3600)}h ago"
    return f"{int(s // 86400)}d ago"


_BADGE = {"queued": "q", "running": "r", "completed": "c", "failed": "x",
          "cancelled": "n", "awaiting_input": "a"}


def _badge(status: Any) -> str:
    return f'<span class="badge b-{_BADGE.get(str(status), "n")}">{_esc(status)}</span>'


def _stat_strip(counts: dict[str, int], ingest: dict[str, Any],
                fleet: list[dict[str, Any]], projects: list[str]) -> str:
    """Sidekiq-style overview: big-number KPIs summed across node + ingest queues."""
    tot = {"running": 0, "queued": 0, "completed": 0, "failed": 0}
    for k, n in (counts or {}).items():            # node-job counts: 'cpu_completed' …
        for s in tot:
            if k.endswith("_" + s):
                tot[s] += int(n)
    for q, st in (ingest or {}).get("queues", {}).items():
        if q in ("cpu", "gpu"):
            continue
        for s in tot:
            tot[s] += int(st.get(s, 0))
    items = [("Busy", tot["running"], "running"), ("Enqueued", tot["queued"], "queued"),
             ("Processed", tot["completed"], "completed"), ("Failed", tot["failed"], "failed"),
             ("Workers", len(fleet), "workers"), ("Projects", len(projects), "proj")]
    cells = "".join(f'<div class="kpi {cls}"><b>{v}</b><span>{label}</span></div>'
                    for label, v, cls in items)
    return f'<div class="kpis">{cells}</div>'


def _recent_table(jobs: list[dict[str, Any]]) -> str:
    """Sidekiq-style recent-activity feed across both job families (node + ingest)."""
    if not jobs:
        return '<p class="muted">no recent jobs</p>'
    rows = []
    for j in jobs:
        retries = int(j.get("retries") or 0)
        kind = str(j.get("kind") or "")
        rows.append(
            "<tr>"
            f'<td><span class="kind k-{_esc(kind)}">{_esc(kind)}</span></td>'
            f'<td class="mono"><a href="/job/{_esc(j.get("id"))}?kind={_esc(kind)}">'
            f'{_esc(j.get("name") or "—")}</a></td>'
            f"<td>{_esc(j.get('queue'))}</td>"
            f"<td>{_esc(_proj_label(j.get('project') or ''))}</td>"
            f"<td>{_badge(j.get('status'))}</td>"
            f'<td class="{"warn" if retries else ""}">{retries or "—"}</td>'
            f'<td class="mono">{_esc(j.get("worker") or "—")}</td>'
            f'<td class="muted">{_ago(j.get("recency"))}</td>'
            "</tr>"
        )
    return (
        '<table class="fleet"><thead><tr>'
        "<th>kind</th><th>job</th><th>queue</th><th>project</th>"
        "<th>status</th><th>retries</th><th>worker</th><th>when</th>"
        f"</tr></thead><tbody>{''.join(rows)}</tbody></table>"
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
/* Sidekiq-style KPI strip */
.kpis{display:flex;flex-wrap:wrap;gap:12px;margin:0 0 8px}
.kpi{background:#fff;border:1px solid #d0d7de;border-radius:9px;padding:12px 20px;
 min-width:104px;box-shadow:0 1px 0 rgba(27,31,36,.04)}
.kpi b{display:block;font-size:27px;line-height:1;font-weight:700;color:#1f2328}
.kpi span{display:block;font-size:10px;text-transform:uppercase;letter-spacing:.7px;
 color:#656d76;margin-top:6px}
.kpi.running b{color:#0969da}.kpi.queued b{color:#9a6700}
.kpi.completed b{color:#1a7f37}.kpi.failed b{color:#d1242f}.kpi.workers b{color:#0969da}
/* status badges + recent feed */
.badge{display:inline-block;padding:2px 9px;border-radius:11px;font-size:11px;
 font-weight:600;line-height:1.5}
.b-q{background:#fff8c5;color:#7d4e00}.b-r{background:#ddf4ff;color:#0550ae}
.b-c{background:#dafbe1;color:#0a5a2a}.b-x{background:#ffebe9;color:#a40e26}
.b-n{background:#eaeef2;color:#57606a}.b-a{background:#fbefff;color:#6639ba}
.kind{font-size:10px;text-transform:uppercase;letter-spacing:.5px;padding:2px 7px;
 border-radius:5px;font-weight:700}
.k-node{background:#eef2ff;color:#3538cd}.k-ingest{background:#eafff1;color:#15803d}
.mono{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px}
td.warn{color:#9a6700;font-weight:700}
table.fleet a{color:#0969da;text-decoration:none}
table.fleet a:hover{text-decoration:underline}
/* Sidekiq-style All/Retries/Dead tabs on the activity feed */
.tabs{font-size:12px;font-weight:400;text-transform:none;letter-spacing:0;margin-left:10px}
.tabs .tab{padding:3px 11px;border-radius:12px;color:#57606a;text-decoration:none}
.tabs .tab.active{background:#0969da;color:#fff}
/* job-detail metadata grid + error block */
.metas{display:flex;flex-wrap:wrap;gap:10px}
.meta{background:#fff;border:1px solid #d0d7de;border-radius:8px;padding:9px 13px;min-width:120px}
.meta span{display:block;font-size:10px;text-transform:uppercase;letter-spacing:.5px;color:#656d76}
.meta b{display:block;font-size:13px;margin-top:3px;word-break:break-all;font-weight:600}
pre.err{background:#fff8f8;border:1px solid #ffcecb;border-radius:8px;padding:12px;
 color:#a40e26;font-size:12px;overflow:auto;white-space:pre-wrap}
/* write-action controls (opt-in via --enable-writes) */
form.ctl{display:inline-flex;align-items:center;gap:8px;margin:0}
form.ctl button{font-size:11px;padding:3px 11px;border:1px solid #d0d7de;border-radius:6px;
 background:#f6f8fa;color:#24292f;cursor:pointer}
form.ctl button:hover{background:#eef1f4}
.state{font-size:10px;font-weight:700;padding:2px 8px;border-radius:10px}
.state.on{background:#dafbe1;color:#0a5a2a}.state.off{background:#ffebe9;color:#a40e26}
/* live hardware panel (cpu/gpu/ram bars, no JS) */
.card.hw{min-width:300px}
.card.hw.stale{opacity:.55}
.card.hw h3 em{font-style:normal;font-weight:600;font-size:11px}
.card.hw h3 em.live{color:#1a7f37}.card.hw h3 em.stale{color:#9a6700}
.hwrow{display:flex;align-items:center;gap:10px;margin:8px 0}
.hwk{width:46px;font-size:11px;color:#656d76;font-weight:600;flex:none}
.spark{width:120px;height:28px;flex:none;display:block}
.hwv{flex:1;font-size:12px;text-align:right;font-variant-numeric:tabular-nums}
.hwv em{color:#848d97;font-style:normal;font-size:11px;margin-left:4px}
"""


def render_dashboard(project: str | None = None, *, stale_after_s: float = 30.0,
                     view: str = "all", writes_enabled: bool = False,
                     hw_history: dict[str, Any] | None = None) -> str:
    """Render the full dashboard HTML for ``project`` (``None`` ⇒ all projects).
    ``view`` selects the Recent-activity tab: ``all`` | ``retries`` (Sidekiq's
    Retries — node-jobs with ``watchdog_retries>0``) | ``dead`` (failed jobs).
    ``writes_enabled`` (opt-in) renders the per-worker ON/OFF toggles.
    Pure — fetches the engine snapshots and returns a complete HTML document."""
    snap = node_queue.snapshot(project=project)
    ingest = node_queue.ingest_snapshot(project=project)
    fleet = node_queue.fleet_snapshot(stale_after_s=stale_after_s, project=project)
    if view == "retries":
        recent = node_queue.recent_jobs(project=project, min_retries=1, limit=50)
    elif view == "dead":
        recent = node_queue.recent_jobs(project=project, status="failed", limit=50)
    else:
        view, recent = "all", node_queue.recent_jobs(project=project, limit=25)
    projects = node_queue.list_projects()
    controls: dict = {}
    if writes_enabled:
        from queue_workflows import worker_control
        controls = {(w.get("host_label"), w.get("queue")):
                    worker_control.get_worker_control(w.get("host_label"), w.get("queue"))
                    for w in fleet}
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
  <h2>Overview</h2>
  {_stat_strip(counts, ingest, fleet, projects)}
  <h2>Hardware — live fleet (cpu · gpu · ram)</h2>
  {_hw_panel(hw_history)}
  <h2>Queues — shared cpu / gpu</h2>
  <div class="cards">{_queue_card('cpu', counts)}{_queue_card('gpu', counts)}</div>
  <h2>Ingest queues</h2>
  <div class="cards">{_ingest_cards(ingest)}</div>
  <h2>Recent activity {_recent_tabs(view, project)}</h2>
  {_recent_table(recent)}
  <h2>Fleet — workers</h2>
  {_fleet_table(fleet, controls=controls, writes_enabled=writes_enabled)}
</main>
<footer>read-only · single-DB · auto-refresh {REFRESH_S}s · queue-conductor-web</footer>
</body></html>"""
    return body


def _recent_tabs(view: str, project: str | None) -> str:
    """Sidekiq-style All / Retries / Dead tabs for the activity feed (preserves
    the active project filter)."""
    suffix = "" if project is None else "&project=" + urllib.parse.quote(project)
    out = []
    for key, label in (("all", "All"), ("retries", "Retries"), ("dead", "Dead")):
        cls = "tab active" if view == key else "tab"
        out.append(f'<a class="{cls}" href="{_esc("/?view=" + key + suffix)}">{label}</a>')
    return '<span class="tabs">' + "".join(out) + "</span>"


def _job_meta(job: dict[str, Any]) -> str:
    fields = [
        ("status", _badge(job.get("status"))),
        ("queue", _esc(job.get("queue"))),
        ("project", _esc(_proj_label(job.get("project") or ""))),
        ("worker", _esc(job.get("claimed_by") or "—")),
        ("run", _esc(job.get("run_id") or "—")),
        ("retries", _esc(job.get("watchdog_retries", "—"))),
        ("seconds", _esc(job.get("seconds") if job.get("seconds") is not None else "—")),
        ("created", _esc(job.get("created_at"))),
        ("finished", _esc(job.get("finished_at") or "—")),
    ]
    cells = "".join(f'<div class="meta"><span>{k}</span><b>{v}</b></div>' for k, v in fields)
    return f'<div class="metas">{cells}</div>'


# event_type → badge colour (terminal/trip = red-ish, claimed/run = blue, requeue = amber)
_EV_CLS = {"claimed": "r", "completed": "c", "failed": "x", "cancelled": "n",
           "requeued": "q", "reassigned": "q", "gpu_health_trip": "x",
           "stall_trip": "x", "budget_trip": "x"}


def render_job(job: dict[str, Any], events: list[dict[str, Any]],
               *, kind: str = "node", writes_enabled: bool = False) -> str:
    """Job-detail page: the job's metadata + (for a node-job) its per-attempt
    ``workflow_node_events`` timeline — broker_parrot's forensic history, richer
    than Sidekiq's per-job retry list. ``writes_enabled`` shows a re-queue control
    for a running node-job (Sidekiq's retry)."""
    name = job.get("node_id") or job.get("task_name") or job.get("id")
    err = (f'<h2>Error</h2><pre class="err">{_esc(job.get("error"))}</pre>'
           if job.get("error") else "")
    requeue = ""
    if writes_enabled and kind == "node" and job.get("status") == "running":
        requeue = (
            '<form method="post" action="/requeue" class="ctl" style="margin-top:10px">'
            f'<input type="hidden" name="job_id" value="{_esc(job.get("id"))}">'
            '<button type="submit">re-queue (retry on a fresh worker)</button></form>'
        )
    if events:
        rows = "".join(
            "<tr>"
            f'<td>{_esc(e.get("attempt"))}</td>'
            f'<td><span class="badge b-{_EV_CLS.get(str(e.get("event_type")), "n")}">'
            f'{_esc(e.get("event_type"))}</span></td>'
            f'<td class="mono">{_esc(e.get("host_label") or "—")}</td>'
            f'<td>{_esc(e.get("elapsed_s") if e.get("elapsed_s") is not None else "—")}</td>'
            f'<td class="muted">{_esc(e.get("detail") or e.get("error") or "")}</td>'
            f'<td class="muted">{_ago(e.get("created_at"))}</td>'
            "</tr>"
            for e in events
        )
        timeline = ('<table class="fleet"><thead><tr><th>try</th><th>event</th>'
                    "<th>host</th><th>elapsed</th><th>detail</th><th>when</th>"
                    f"</tr></thead><tbody>{rows}</tbody></table>")
    elif kind == "node":
        timeline = '<p class="muted">no events recorded for this node-job yet</p>'
    else:
        timeline = '<p class="muted">ingest jobs have no per-attempt event log</p>'
    return f"""<!doctype html><html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>broker_parrot — job</title><style>{_CSS}</style></head><body>
<header><h1>broker_parrot <small>job · {_esc(name)} · {_esc(kind)}</small></h1>
<div style="margin-top:8px"><a class="f" href="/">← dashboard</a></div></header>
<main>
  <h2>Job <em style="text-transform:none;font-weight:400;color:#848d97">· {_esc(job.get("id"))}</em></h2>
  {_job_meta(job)}
  {requeue}
  {err}
  <h2>Event timeline <em style="text-transform:none;font-weight:400;color:#848d97">· workflow_node_events (per attempt)</em></h2>
  {timeline}
</main>
<footer>read-only · single-DB · queue-conductor-web</footer>
</body></html>"""


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
    writes_enabled = False  # opt-in (--enable-writes); default keeps the view read-only

    def _send(self, code: int, body: str) -> None:
        payload = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        # Never cache: this is a live dashboard (5s meta-refresh) — a cached copy in
        # the browser or an upstream proxy would show stale fleet/queue/hw state.
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urllib.parse.urlparse(self.path)
        # keep_blank_values so the single-tenant sentinel project '' is a real,
        # selectable filter (?project=) and not silently dropped to the all view.
        q = urllib.parse.parse_qs(parsed.query, keep_blank_values=True)
        # /job/<id> — job detail + per-attempt event timeline
        if parsed.path.startswith("/job/"):
            job_id = urllib.parse.unquote(parsed.path[len("/job/"):])
            kind = q.get("kind", ["node"])[0]
            try:
                if kind == "ingest":
                    job, events = node_queue.get_ingest_job(job_id), []
                else:
                    job = node_queue.get_node_job(job_id)
                    events = node_queue.list_node_events(job_id) if job else []
                if job is None:
                    self._send(404, "<h1>404 — no such job</h1>")
                    return
                self._send(200, render_job(job, events, kind=kind,
                                           writes_enabled=self.writes_enabled))
            except Exception as exc:  # noqa: BLE001 — a DB blip must not kill the server
                self._send(500, render_error(exc))
            return
        if parsed.path not in ("/", "/index.html"):
            self._send(404, "<h1>404</h1>")
            return
        project = q["project"][0] if "project" in q else None
        view = q.get("view", ["all"])[0]
        hw_history = _HW_FEED.history_by_host() if _HW_FEED is not None else None
        try:
            self._send(200, render_dashboard(project, stale_after_s=self.stale_after_s,
                                             view=view, writes_enabled=self.writes_enabled,
                                             hw_history=hw_history))
        except Exception as exc:  # noqa: BLE001 — a DB blip must not kill the server
            self._send(500, render_error(exc))

    def do_POST(self) -> None:  # noqa: N802
        """Gated write path (opt-in via --enable-writes). Two operator actions —
        worker ON/OFF (worker_control) + re-queue a running node-job — then a
        303 redirect back (POST/redirect/GET). 403 when writes are disabled."""
        if not self.writes_enabled:
            self._send(403, "<h1>403 — writes disabled (start with --enable-writes)</h1>")
            return
        # CSRF guard: a browser cross-site form POST carries an Origin that won't
        # match our Host — reject it. A request with no Origin (curl / same-origin
        # form) is allowed. Cheap defence so enabling writes + binding 0.0.0.0 can't
        # be driven by another tab the operator has open.
        origin = self.headers.get("Origin")
        if origin and urllib.parse.urlparse(origin).netloc != self.headers.get("Host", ""):
            self._send(403, "<h1>403 — cross-origin POST rejected</h1>")
            return
        parsed = urllib.parse.urlparse(self.path)
        length = int(self.headers.get("Content-Length", 0) or 0)
        form = urllib.parse.parse_qs(
            self.rfile.read(length).decode("utf-8") if length else "",
            keep_blank_values=True,
        )

        def f(k: str, d: str = "") -> str:
            return form.get(k, [d])[0]

        try:
            if parsed.path == "/control":
                from queue_workflows import worker_control
                worker_control.set_worker_control(
                    f("host"), f("queue"), desired_state=f("desired_state"),
                    requested_by="conductor-web",
                )
            elif parsed.path == "/requeue":
                node_queue.requeue_job_for_retry(f("job_id"))
            else:
                self._send(404, "<h1>404</h1>")
                return
        except Exception as exc:  # noqa: BLE001 — a bad write must not kill the server
            self._send(500, render_error(exc))
            return
        # redirect back within THIS origin only (no open-redirect): keep just the
        # Referer's path+query, never its scheme/host.
        ref = urllib.parse.urlparse(self.headers.get("Referer") or "/")
        dest = ref.path + (("?" + ref.query) if ref.query else "")
        if not dest.startswith("/"):
            dest = "/"
        self.send_response(303)
        self.send_header("Location", dest)
        self.end_headers()

    def log_message(self, *args: object) -> None:  # silence default stderr spam
        pass


def _start_hw_feed(stale_after_s: float) -> Any:
    """Start ONE background HwFeed against the broker's ``hw_metrics`` stream — but only
    for a Postgres metrics DSN (``LISTEN``/``NOTIFY`` is pg-only; a sqlite/redis/mongo
    conductor has no hw stream). Never fatal: any failure ⇒ ``None`` ⇒ the panel shows a
    graceful 'no telemetry' note."""
    try:
        from queue_workflows.hw_feed import HwFeed
        from queue_workflows.hw_metrics import metrics_dsn
        dsn = metrics_dsn()
        if not dsn or not dsn.startswith(("postgres://", "postgresql://")):
            return None
        return HwFeed(stale_after_s=stale_after_s).start()
    except Exception:
        return None


def serve(host: str = "127.0.0.1", port: int = 8787, *, stale_after_s: float = 30.0,
          writes_enabled: bool = False, enable_hw: bool = True) -> None:
    ConductorWebHandler.stale_after_s = stale_after_s
    ConductorWebHandler.writes_enabled = writes_enabled
    global _HW_FEED
    if enable_hw:
        _HW_FEED = _start_hw_feed(stale_after_s)
    httpd = ThreadingHTTPServer((host, port), ConductorWebHandler)
    mode = "READ+WRITE" if writes_enabled else "read-only"
    print(f"queue-conductor-web serving on http://{host}:{port} ({mode})  (Ctrl-C to stop)")
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
    p.add_argument("--db-backend", default=None,
                   help="store: pg | sqlite | redis | mongodb (default from "
                   "QUEUE_WORKFLOWS_DB_BACKEND). A Postgres fleet needs "
                   "--db-backend pg — the library default is now sqlite (v1.0.0).")
    p.add_argument("--db-url-env", default=None,
                   help="env var holding the DSN / SQLite path (default: configured)")
    p.add_argument("--enable-writes", action="store_true",
                   help="opt in to operator write actions (worker ON/OFF + re-queue). "
                   "Default OFF — the view is read-only.")
    p.add_argument("--no-hw", action="store_true",
                   help="disable the live Hardware panel (don't start the hw_metrics "
                   "feed). Default ON for a Postgres broker.")
    args = p.parse_args(argv)
    from queue_workflows_conductor.conductor import _configure_backend
    _configure_backend(args.db_backend, args.db_url_env)
    serve(args.host, args.port, stale_after_s=args.stale_after,
          writes_enabled=args.enable_writes, enable_hw=not args.no_hw)
    return 0

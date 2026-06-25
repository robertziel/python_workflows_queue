"""queue-conductor-web live Hardware panel — moving time-series sparklines (cpu/gpu/ram)
from the broker's hw_metrics HISTORY. The panel render is pure (given the per-host history
dict ``{host: [sample, …]}`` oldest→newest), so most of this needs no server and no DB.
"""

from __future__ import annotations

from queue_workflows_conductor import web


def test_hw_panel_renders_sparklines_and_current_values():
    history = {
        "host-a": [
            {"cpu_percent": 20.0, "ram_used_mb": 20000, "ram_total_mb": 128000,
             "gpus": [{"id": 0, "use_pct": 40, "vram_used_mb": 8000, "vram_total_mb": 24576}],
             "stale": False},
            {"cpu_percent": 37.4, "ram_used_mb": 25600, "ram_total_mb": 128000,
             "gpus": [{"id": 0, "use_pct": 88, "vram_used_mb": 18432, "vram_total_mb": 24576}],
             "stale": False},
        ],
        "host-b": [
            {"cpu_percent": 5.0, "ram_used_mb": 4096, "ram_total_mb": 64000,
             "gpus": [], "stale": True},
        ],
    }
    out = web._hw_panel(history)
    assert "host-a" in out and "host-b" in out
    # MOVING sparkline = inline SVG with a dotted polyline (no JS)
    assert "<svg" in out and "<polyline" in out and "<circle" in out
    # 2-point series spans the left (x=0.0) and right (x=120.0) edges
    assert "0.0," in out and "120.0," in out
    # CURRENT values come from the LATEST sample: cpu 37%, gpu 88%, summed VRAM 18/24 GB
    assert "37%" in out and "88%" in out and "18/24 GB" in out
    # stale host flagged; no-gpu host shows 'none'
    assert "stale" in out and "none" in out


def test_hw_panel_empty_is_graceful():
    assert "no hardware telemetry" in web._hw_panel(None)
    assert "no hardware telemetry" in web._hw_panel({})


def test_hw_panel_tolerates_missing_fields_and_single_point():
    out = web._hw_panel({"h": [{"cpu_percent": None, "ram_used_mb": None,
                                "ram_total_mb": None, "gpus": None, "stale": False}]})
    assert "h" in out and "<svg" in out and "—" in out  # graceful dashes, no crash


def test_spark_oldest_left_newest_right():
    svg = web._spark([10, 50, 90])
    assert "<polyline" in svg and "120.0," in svg  # newest point at the right edge
    assert 'r="2.6"' in svg                        # the emphasised newest dot
    # empty series → an empty svg with no polyline
    assert "<svg" in web._spark([]) and "polyline" not in web._spark([])


def test_dashboard_includes_hw_section_and_sparkline():
    out = web.render_dashboard(None, hw_history={
        "host-z": [
            {"cpu_percent": 12.0, "ram_used_mb": 1024, "ram_total_mb": 8192,
             "gpus": [{"id": 0, "use_pct": 73, "vram_used_mb": 1000, "vram_total_mb": 2000}],
             "stale": False},
        ],
    })
    assert "Hardware — live fleet" in out
    assert "host-z" in out and "73%" in out and "<svg" in out

    out2 = web.render_dashboard(None, hw_history=None)
    assert "Hardware — live fleet" in out2 and "no hardware telemetry" in out2

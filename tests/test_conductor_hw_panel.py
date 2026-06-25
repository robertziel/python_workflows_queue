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
    # btop-style DOTTED column graph = inline SVG of dots, NO connecting polyline (no JS)
    assert "<svg" in out and "<circle" in out and "polyline" not in out
    # newest column is right-anchored at x=120.0
    assert 'cx="120.0"' in out
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


def test_spark_btop_columns():
    # btop encoding: value → column height (lit dots = round(value*rows/100), rows=4),
    # newest column anchored at the right (x=120.0), row 0 green via the gradient.
    svg = web._spark([90])
    assert "<circle" in svg and 'cx="120.0"' in svg
    assert "polyline" not in svg                  # dots, not a connecting line
    assert "#32d74b" in svg                        # bottom row green (the value gradient)
    assert svg.count("<circle") >= 3               # 90% → ~4 lit dots
    assert web._spark([1]).count("<circle") == 1   # tiny positive → at least 1 dot
    assert web._spark([0]).count("<circle") == 0   # zero → empty column
    assert "<svg" in web._spark([]) and "circle" not in web._spark([])


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

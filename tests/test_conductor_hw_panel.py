"""queue-conductor-web live Hardware panel — per-host cpu/gpu/ram from the broker's
``hw_metrics`` stream. The panel render is pure (given the latest-sample-per-host dict),
so most of this needs no server and no DB.
"""

from __future__ import annotations

from queue_workflows_conductor import web


def test_hw_panel_renders_per_host_cpu_gpu_ram():
    hw = {
        "host-a": {
            "cpu_percent": 37.4, "ram_used_mb": 25600, "ram_total_mb": 128000,
            "gpus": [{"id": 0, "use_pct": 88, "vram_used_mb": 18432, "vram_total_mb": 24576}],
            "stale": False,
        },
        "host-b": {
            "cpu_percent": 5.0, "ram_used_mb": 4096, "ram_total_mb": 64000,
            "gpus": [], "stale": True,
        },
    }
    out = web._hw_panel(hw)
    # both hosts present
    assert "host-a" in out and "host-b" in out
    # GPU usage rendered (the whole point of this panel) + the no-JS CSS bar
    assert "GPU0" in out and "88%" in out
    assert 'class="bar"' in out and "width:88%" in out
    # CPU% rendered (rounded), VRAM shown in GB (18432/24576 MB -> 18/24 GB)
    assert "37%" in out
    assert "18/24 GB" in out
    # stale host flagged; a no-gpu host shows 'none'
    assert "stale" in out and "none" in out


def test_hw_panel_empty_is_graceful():
    assert "no hardware telemetry" in web._hw_panel(None)
    assert "no hardware telemetry" in web._hw_panel({})


def test_hw_panel_tolerates_missing_fields():
    # None/absent fields must not raise (a sampler with no GPU probe, partial RAM, etc.)
    out = web._hw_panel({"h": {"cpu_percent": None, "ram_used_mb": None,
                               "ram_total_mb": None, "gpus": None, "stale": False}})
    assert "h" in out and "—" in out  # graceful dashes, no crash


def test_bar_clamps_and_formats():
    assert "width:0%" in web._bar(None)
    assert "width:0%" in web._bar(-5)
    assert "width:100%" in web._bar(140)
    assert "width:50%" in web._bar(50)


def test_dashboard_includes_hw_section_and_panel():
    # render_dashboard does DB I/O for the other sections (conftest's test DB), but the
    # Hardware section + panel are driven purely by the hw= arg.
    out = web.render_dashboard(None, hw={
        "host-z": {"cpu_percent": 12.0, "ram_used_mb": 1024, "ram_total_mb": 8192,
                   "gpus": [{"id": 0, "use_pct": 73, "vram_used_mb": 1000, "vram_total_mb": 2000}],
                   "stale": False},
    })
    assert "Hardware — live fleet" in out
    assert "host-z" in out and "73%" in out

    # with no hw, the section still renders with the graceful empty note
    out2 = web.render_dashboard(None, hw=None)
    assert "Hardware — live fleet" in out2 and "no hardware telemetry" in out2

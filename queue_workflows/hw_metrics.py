"""Hardware-metrics sampler.

A tiny daemon thread the gpu claim worker starts at boot. Every few seconds it
collects:

- system CPU utilisation (psutil, across all cores)
- RAM used + total (psutil)
- Swap used (psutil)
- Per-GPU utilisation + VRAM used (nvidia-smi or rocm-smi)

And fires a Postgres ``NOTIFY hw_metrics, <json>``. Rails reads that and
attaches it to the queue-snapshot payload, so the frontend pill can show live
CPU/GPU load.

This is generic telemetry plumbing that any GPU-worker fleet wants; the channel
name + payload shape are part of the engine's monitoring contract (Rails is the
consumer, not a Python importer). ``psutil`` is an OPTIONAL dependency — the
``import psutil`` is guarded; without it CPU/RAM stay ``None`` and only GPU
telemetry flows.
"""

from __future__ import annotations

import json
import logging
import os
import re
import socket
import subprocess
import threading
import time
import typing
from typing import Any

try:
    import psutil
except Exception:  # pragma: no cover - optional
    psutil = None  # type: ignore

from queue_workflows.cgroup_attribution import CgroupAttribution
from queue_workflows.config import get_config
from queue_workflows.db import connection

log = logging.getLogger(__name__)


# Push-only: every SAMPLE_INTERVAL_S seconds we fire a Postgres
# ``NOTIFY hw_metrics, <json>`` and move on — no UPSERT, no row retained.
# Rails' SSE controller has a matching ``LISTEN`` on a dedicated connection and
# forwards each payload to browser clients.
SAMPLE_INTERVAL_S = 5.0
NOTIFY_CHANNEL = "hw_metrics"


def metrics_dsn() -> str | None:
    """Resolve the DSN hw-metrics is published to + read from: ``config.
    metrics_db_url_env`` if set (the shared broker), else ``config.db_url_env``
    (a project whose queue DB already IS the broker). Returns the DSN string, or
    ``None`` if the chosen env var is unset. hw-metrics is Postgres-only (NOTIFY),
    so this is always a pg DSN. Shared by the publisher (:func:`_broadcast`) and
    the reader (:class:`queue_workflows.hw_feed.HwFeed`) so both agree on target."""
    cfg = get_config()
    return os.environ.get(cfg.metrics_db_url_env or cfg.db_url_env)


def _uses_dedicated_metrics_dsn() -> bool:
    """True when hw-metrics targets a DSN distinct from the engine queue pool —
    i.e. a non-broker queue DB plus an explicit ``metrics_db_url_env`` pointing at
    the broker. Then the publisher opens its own connection instead of the pool."""
    cfg = get_config()
    return bool(cfg.metrics_db_url_env) and cfg.metrics_db_url_env != cfg.db_url_env


def _host_label() -> str:
    """Stable identifier for the box this sampler is running on. Set the
    configured host-label env (default ``AI_LEADS_HOST_LABEL``) on remote
    worker boxes so the Rails SSE indicator can group samples by source.
    Defaults to ``socket.gethostname()``."""
    from queue_workflows.config import get_config
    return os.environ.get(get_config().host_label_env, "").strip() or socket.gethostname()


# ── GPU probe (vendor-aware) ─────────────────────────────────────────────
#
# AMD/ROCm and NVIDIA/CUDA boxes both run this sampler. The first call probes
# for ``nvidia-smi`` then ``rocm-smi`` and pins the choice for the process
# lifetime so we don't shell out to ``which`` every tick.


_GPU_PROBE: typing.Callable[[], list[dict[str, Any]]] | None = None


def _gpu_probe() -> list[dict[str, Any]]:
    """Dispatch to the right vendor probe. Picks once on first call."""
    global _GPU_PROBE
    if _GPU_PROBE is None:
        _GPU_PROBE = _select_gpu_probe()
    return _GPU_PROBE()


# Below this, a probed "total VRAM" is treated as UNKNOWN, not as a real cap.
# Unified-memory GPUs make the smi "dedicated VRAM total" meaningless: an AMD
# APU's rocm-smi reports a tiny carveout (observed 512 MB on box-b) while the
# real model memory comes from shared system RAM, and a Grace-Blackwell box-c
# reports none at all. A reading this small can only be a carveout/parse
# artifact — never a GPU that actually runs multi-GB diffusion models — so we
# FAIL OPEN (return None ⇒ "capacity unknown" ⇒ the claim gate falls back to
# claim-any) rather than gate the worker to "fits nothing". Env-overridable.
MIN_PLAUSIBLE_VRAM_MB = 2048

#: Operator override for total GPU VRAM (MB). The RELIABLE source on this fleet's
#: unified-memory hardware, where the smi probe can't report a meaningful total.
#: Set per host (e.g. in compose) to ACTIVATE capacity-aware assignment; unset ⇒
#: the probe is used, and an implausible/absent probe ⇒ unknown ⇒ claim-any.
_VRAM_TOTAL_ENV = "AI_LEADS_GPU_VRAM_TOTAL_MB"


def total_vram_mb() -> int | None:
    """The machine's TOTAL GPU VRAM in MB — the capacity a single model load is
    measured against.

    Resolution order:
      1. ``AI_LEADS_GPU_VRAM_TOTAL_MB`` if set (operator-declared truth — the
         reliable source on unified-memory GPUs where the smi total is bogus).
      2. else the LARGEST single GPU's probed ``vram_total_mb`` (a model loads on
         ONE device, so capacity is the max single card, not the sum) — but only
         if it is at least :data:`MIN_PLAUSIBLE_VRAM_MB`.
      3. else ``None`` — "capacity unknown". Callers FAIL OPEN on ``None`` (claim
         any model), so a missing/bogus probe never wedges a worker to "fits
         nothing". Best-effort: never raises.
    """
    raw = (os.environ.get(_VRAM_TOTAL_ENV, "") or "").strip()
    if raw:
        try:
            v = int(float(raw))
            return v if v > 0 else None
        except (TypeError, ValueError):
            log.warning("[hw_metrics] %s=%r is not an int; ignoring", _VRAM_TOTAL_ENV, raw)
    try:
        gpus = _gpu_probe()
    except Exception:
        log.exception("[hw_metrics] total_vram_mb probe failed; treating as unknown")
        return None
    totals = [int(g.get("vram_total_mb") or 0) for g in (gpus or [])]
    best = max(totals) if totals else 0
    if best < MIN_PLAUSIBLE_VRAM_MB:
        # Implausibly small (unified-memory carveout) or no GPU ⇒ unknown.
        return None
    return best


def _select_gpu_probe() -> typing.Callable[[], list[dict[str, Any]]]:
    """Pick nvidia-smi if present, else rocm-smi, else a no-op."""
    if _which("nvidia-smi"):
        log.info("[hw_metrics] using nvidia-smi for GPU telemetry")
        return _nvidia_smi
    if _which("rocm-smi"):
        log.info("[hw_metrics] using rocm-smi for GPU telemetry")
        return _rocm_smi
    log.info("[hw_metrics] no GPU CLI found; GPU column will be empty")
    return lambda: []


def _which(cmd: str) -> bool:
    try:
        subprocess.check_call(
            ["which", cmd], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        return True
    except Exception:
        return False


def _nvidia_smi() -> list[dict[str, Any]]:
    """Per-GPU dicts via ``nvidia-smi --query-gpu``. CSV (one row per GPU) with
    units stripped. Empty list if the CLI fails. Unified-memory parts (Grace
    Blackwell box-c / Jetson) report ``[N/A]`` for VRAM — per-field permissive
    parse turns that into 0 instead of dropping the row."""
    try:
        raw = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            stderr=subprocess.DEVNULL, timeout=2,
        ).decode()
    except Exception:
        return []

    def _i(v: str) -> int:
        try:
            return int(v)
        except ValueError:
            return 0

    out: list[dict[str, Any]] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 4:
            continue
        gpu_id, use_pct, vram_used_mb, vram_total_mb = (_i(p) for p in parts)
        out.append({
            "id": gpu_id,
            "use_pct": use_pct,
            "vram_used_mb": vram_used_mb,
            "vram_total_mb": vram_total_mb,
        })
    return out


def _rocm_smi() -> list[dict[str, Any]]:
    """Return a list of per-GPU dicts with ``id, use_pct, vram_used_mb,
    vram_total_mb``. Empty list if rocm-smi isn't available or fails."""
    try:
        raw = subprocess.check_output(
            ["rocm-smi", "--json", "--showuse", "--showmeminfo", "vram"],
            stderr=subprocess.DEVNULL, timeout=2,
        ).decode()
    except Exception:
        return []
    try:
        data = json.loads(raw)
    except Exception:
        return []
    out: list[dict[str, Any]] = []
    for key, val in sorted(data.items()):
        m = re.match(r"card(\d+)", key)
        if not m:
            continue
        gpu_id = int(m.group(1))
        use = _as_int(val.get("GPU use (%)"))
        vram_used = _as_int(val.get("VRAM Total Used Memory (B)"))
        vram_total = _as_int(val.get("VRAM Total Memory (B)"))
        out.append({
            "id": gpu_id,
            "use_pct": use if use is not None else 0,
            "vram_used_mb": (vram_used or 0) // (1024 * 1024),
            "vram_total_mb": (vram_total or 0) // (1024 * 1024),
        })
    return out


def _as_int(v: Any) -> int | None:
    if v is None:
        return None
    if isinstance(v, int):
        return v
    try:
        return int(float(str(v).strip().rstrip("%")))
    except (TypeError, ValueError):
        return None


# ── Sample builder ───────────────────────────────────────────────────────


def _build_sample(attrib: CgroupAttribution | None = None) -> dict[str, Any]:
    cpu = mem = swap = None
    if psutil is not None:
        try:
            cpu = psutil.cpu_percent(interval=None)
            vm = psutil.virtual_memory()
            sm = psutil.swap_memory()
            mem = {
                "percent": float(vm.percent),
                "used_mb": int(vm.used // (1024 * 1024)),
                "total_mb": int(vm.total // (1024 * 1024)),
            }
            swap = {
                "used_mb": int(sm.used // (1024 * 1024)),
            }
        except Exception:
            log.exception("[hw_metrics] psutil probe failed")

    # Per-our-containers slice (cgroup v2). ``None`` when the host cgroup mount
    # or docker socket isn't available — the frontend falls back to the
    # single-tone chart in that case.
    cpu_ours_pct: float | None = None
    ram_ours_used_mb: int | None = None
    ram_ours_pct: float | None = None
    if attrib is not None:
        try:
            slice_ = attrib.sample()
        except Exception:
            log.exception("[hw_metrics] cgroup attribution failed")
            slice_ = None
        if slice_ is not None:
            cpu_ours_pct = slice_.get("cpu_percent")
            ram_ours_used_mb = slice_.get("ram_used_mb")
            if mem and ram_ours_used_mb is not None and mem["total_mb"]:
                ram_ours_pct = (ram_ours_used_mb / mem["total_mb"]) * 100.0
                # cgroup memory.current accounts for page-cache that psutil's
                # ``vm.used`` excludes — clamp to host total so the "ours" slice
                # never exceeds the chart's 100 %.
                ram_ours_pct = max(0.0, min(100.0, ram_ours_pct))

    # GPU attribution intentionally omitted — per-PID GPU split isn't reliable.
    # The frontend mirrors the total into ``gpu_ours`` so the GPU column renders
    # single-tone.
    return {
        "cpu_percent": float(cpu) if cpu is not None else None,
        "cpu_ours_percent": cpu_ours_pct,
        "ram_percent": float(mem["percent"]) if mem else None,
        "ram_used_mb": mem["used_mb"] if mem else None,
        "ram_total_mb": mem["total_mb"] if mem else None,
        "ram_ours_used_mb": ram_ours_used_mb,
        "ram_ours_percent": ram_ours_pct,
        "swap_used_mb": swap["used_mb"] if swap else None,
        "gpus": _gpu_probe(),
    }


# ── Persist ──────────────────────────────────────────────────────────────


def _broadcast(sample: dict[str, Any]) -> None:
    """Fire a Postgres NOTIFY with the JSON-encoded sample. No row stored.
    Published to the METRICS DSN (the shared broker — see :func:`metrics_dsn`), so
    every project shows the SAME fleet-wide hardware view; a consumer LISTENs via
    :class:`queue_workflows.hw_feed.HwFeed`. Carries a ``host`` field so the feed
    attributes samples by source."""
    payload = json.dumps({
        "sampled_at": _iso_now(),
        "host": _host_label(),
        **sample,
    })
    if _uses_dedicated_metrics_dsn():
        # queue DB is NOT the broker → open a short autocommit connection to the
        # broker metrics DSN (the 5 s cadence makes connect-per-broadcast fine).
        dsn = metrics_dsn()
        if not dsn:
            log.warning("[hw_metrics] metrics_db_url_env set but its env var is "
                        "empty — hw sample not published")
            return
        import psycopg
        with psycopg.connect(dsn, autocommit=True) as conn, conn.cursor() as cur:
            cur.execute("SELECT pg_notify(%s, %s)", (NOTIFY_CHANNEL, payload))
    else:
        # queue DB IS the metrics target (a broker-consolidated or single-DB
        # project) → reuse the engine pool, unchanged.
        with connection() as conn, conn.cursor() as cur:
            cur.execute("SELECT pg_notify(%s, %s)", (NOTIFY_CHANNEL, payload))


def _iso_now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


# ── Threaded sampler ─────────────────────────────────────────────────────


class HwMetricsSampler(threading.Thread):
    """Thread that samples + broadcasts. Daemon — dies with the worker."""

    def __init__(self, interval_s: float = SAMPLE_INTERVAL_S):
        super().__init__(daemon=True, name="hw-metrics")
        self.interval_s = interval_s
        self._stop_evt = threading.Event()
        # Per-container attribution carries cross-tick state (last
        # ``usage_usec`` per container) for CPU% deltas, so the instance lives
        # with the sampler thread. Prefix comes from config.container_prefix.
        self._attrib = CgroupAttribution()

    def stop(self) -> None:
        self._stop_evt.set()

    def run(self) -> None:
        log.info("[hw_metrics] sampler running (NOTIFY %s, interval=%ss)",
                 NOTIFY_CHANNEL, self.interval_s)
        while not self._stop_evt.is_set():
            try:
                _broadcast(_build_sample(self._attrib))
            except Exception:
                log.exception("[hw_metrics] broadcast failed")
            if self._stop_evt.wait(self.interval_s):
                return


# ── hw-metrics sampler starter (one per host, + flock backstop) ──────────────
#
# Exactly ONE sampler per host broadcasts its CPU/GPU/RAM telemetry — not one
# per worker. That one-per-host guarantee comes from the CALL SITE: only the
# gpu claim worker calls this starter, and there is exactly one gpu-worker
# container per host.
#
# The flock below is a cheap SECONDARY guard, not the primary mechanism: if two
# processes on one box ever race this starter, whichever grabs the exclusive
# flock first owns the sampler and the other skips. The lock is held for the
# lifetime of the winning process. NB: ``/tmp`` is per-container, so the flock
# only coordinates WITHIN a process tree / a shared mount.

_HW_METRICS_LOCK_PATH = "/tmp/queue_workflows_hw_metrics.lock"
_hw_metrics_lock_fd: int | None = None  # held to keep flock for process lifetime
_hw_metrics_thread: HwMetricsSampler | None = None


def start_hw_metrics_sampler_flocked() -> HwMetricsSampler | None:
    """Start the host's hw-metrics sampler. Called by the gpu claim worker only
    (one gpu container per host ⇒ one sampler per host); the flock is a
    secondary guard against an accidental double-start within a host. Returns
    the started sampler, or ``None`` when this process lost the flock contest,
    when telemetry is disabled (``AI_LEADS_DISABLE_HW_METRICS``), or when the
    sampler failed to construct/start (non-fatal — boot must not depend on
    telemetry).

    Idempotent within a process: if we already hold the lock + a live sampler,
    returns it rather than starting a second.
    """
    global _hw_metrics_lock_fd, _hw_metrics_thread

    if os.environ.get("AI_LEADS_DISABLE_HW_METRICS"):
        # Tests opt out — they shouldn't fan out NOTIFYs to a real DB.
        return None
    if _hw_metrics_lock_fd is not None and _hw_metrics_thread is not None:
        # This process already owns the host sampler — don't double-start.
        return _hw_metrics_thread

    import fcntl

    try:
        fd = os.open(_HW_METRICS_LOCK_PATH, os.O_CREAT | os.O_RDWR, 0o644)
    except OSError:
        return None
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except (BlockingIOError, OSError):
        # Another process on this box already owns the sampler.
        os.close(fd)
        return None
    _hw_metrics_lock_fd = fd  # keep alive — closing would release the flock

    try:
        sampler = HwMetricsSampler()
        sampler.start()
    except Exception:
        log.exception("[hw_metrics] failed to start sampler")
        return None
    _hw_metrics_thread = sampler
    return sampler

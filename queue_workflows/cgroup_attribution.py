"""Per-container CPU + RAM attribution from the host cgroup v2 tree.

The hardware sampler reports system-wide totals from psutil. This module
supplements it with the slice owned by *our* containers — so the queue-pill
chart can colour the lower portion (work caused by the host's containers) and
grey the upper portion (everything else on the host).

The container-name prefix is CONFIGURABLE (plan §1f / §2c): the default comes
from ``config.container_prefix`` (``ai_leads-`` unless a host overrides it via
``queue_workflows.configure(container_prefix=...)``) so other projects attribute
their own containers. The ctor still takes an explicit ``name_prefix`` for
tests.

Scope: CPU and RAM only. GPU attribution is intentionally skipped (ROCm/HIP
doesn't update the per-PID kernel counters that would be needed to split GPU
compute accurately).

Wiring
------
The host's cgroup root + docker socket are mounted read-only into the worker
container::

    /sys/fs/cgroup       -> /host/sys/fs/cgroup  (ro)
    /var/run/docker.sock -> /var/run/docker.sock (ro)

Without those mounts the sampler returns ``None`` and the frontend falls back
to the original single-tone chart.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


DEFAULT_CGROUP_ROOT = "/host/sys/fs/cgroup"


def _default_name_prefix() -> str:
    """The configured container-name prefix (``config.container_prefix``)."""
    from queue_workflows.config import get_config
    return get_config().container_prefix


class CgroupAttribution:
    """Reads per-container CPU/RAM from a cgroup v2 mount.

    Stateful — holds the previous ``usage_usec`` reading per container to
    compute deltas. Not thread-safe; the hw_metrics sampler is single-threaded.
    """

    def __init__(
        self,
        cgroup_root: str = DEFAULT_CGROUP_ROOT,
        name_prefix: str | None = None,
        ncpu: int | None = None,
    ) -> None:
        self._root = Path(cgroup_root)
        self._prefix = name_prefix if name_prefix is not None else _default_name_prefix()
        self._ncpu = ncpu or (os.cpu_count() or 1)
        # container_id -> (wall_time, usage_usec)
        self._last: dict[str, tuple[float, int]] = {}
        self._docker = None  # lazily instantiated; None where docker SDK / socket isn't usable
        self._docker_unavailable_logged = False
        self._lock = threading.Lock()

    # ── public API ───────────────────────────────────────────────────

    def sample(self) -> dict[str, Any] | None:
        """Return the per-our-containers slice as ``{cpu_percent, ram_used_mb}``.

        ``cpu_percent`` is ``None`` on the first call (no baseline) or if the
        cgroup mount isn't readable. ``ram_used_mb`` is always an int when the
        mount is readable.

        Returns ``None`` if the host cgroup root or docker socket is
        unavailable — caller should treat that as "no attribution data this
        tick" and emit only system-wide totals.
        """
        if not self._root.exists():
            return None

        ids = self._our_container_ids()
        if ids is None:
            return None

        now = time.monotonic()
        cpu_delta_usec = 0
        ram_bytes = 0
        had_cpu_baseline = True
        oldest_prev_t: float | None = None

        with self._lock:
            seen: set[str] = set()
            for cid in ids:
                scope = self._root / "system.slice" / f"docker-{cid}.scope"
                usage = _read_usage_usec(scope)
                rss = _read_memory_current(scope)
                if usage is None and rss is None:
                    continue
                if rss is not None:
                    ram_bytes += rss
                if usage is not None:
                    seen.add(cid)
                    prev = self._last.get(cid)
                    if prev is None:
                        # First reading for this container — establish
                        # baseline; nothing to add to the delta yet.
                        had_cpu_baseline = False
                        self._last[cid] = (now, usage)
                        continue
                    prev_t, prev_u = prev
                    self._last[cid] = (now, usage)
                    delta_usec = usage - prev_u
                    if delta_usec < 0:
                        # cgroup reset (container restarted) — skip
                        # this container's contribution this tick.
                        continue
                    cpu_delta_usec += delta_usec
                    if oldest_prev_t is None or prev_t < oldest_prev_t:
                        oldest_prev_t = prev_t
            # forget containers that disappeared so memory doesn't grow
            for stale in set(self._last) - seen:
                self._last.pop(stale, None)

        cpu_pct: float | None
        if not had_cpu_baseline or oldest_prev_t is None:
            cpu_pct = None
        else:
            elapsed = now - oldest_prev_t
            if elapsed <= 0:
                cpu_pct = None
            else:
                pct = (cpu_delta_usec / (elapsed * self._ncpu * 1_000_000)) * 100.0
                cpu_pct = max(0.0, min(100.0, pct))

        return {
            "cpu_percent": cpu_pct,
            "ram_used_mb": ram_bytes // (1024 * 1024),
        }

    # ── internals ────────────────────────────────────────────────────

    def _our_container_ids(self) -> list[str] | None:
        """List the full IDs of containers whose names start with our prefix.

        Returns ``None`` if the docker socket isn't mountable — the caller
        treats that as "skip attribution this tick". The unavailability is
        logged once, then suppressed.
        """
        client = self._get_docker()
        if client is None:
            return None
        try:
            containers = client.containers.list(
                filters={"name": self._prefix},
                ignore_removed=True,
            )
        except Exception as exc:  # docker daemon hiccup or socket gone
            log.debug("[cgroup_attribution] docker.list failed: %s", exc)
            return None
        # ``filters={"name": ...}`` is a substring match on Docker's side;
        # double-check the prefix here.
        return [c.id for c in containers if c.name.startswith(self._prefix)]

    def _get_docker(self):
        if self._docker is not None:
            return self._docker
        try:
            import docker  # type: ignore
        except ImportError:
            if not self._docker_unavailable_logged:
                log.warning("[cgroup_attribution] docker SDK not installed — skipping per-container attribution")
                self._docker_unavailable_logged = True
            return None
        try:
            self._docker = docker.from_env()
            # Touch the daemon to fail fast if the socket isn't there.
            self._docker.ping()
        except Exception as exc:
            if not self._docker_unavailable_logged:
                log.warning("[cgroup_attribution] docker socket unavailable (%s) — skipping per-container attribution", exc)
                self._docker_unavailable_logged = True
            self._docker = None
        return self._docker


# ── filesystem helpers (kept module-level for ease of unit-testing) ──


def _read_usage_usec(scope_dir: Path) -> int | None:
    """Parse ``usage_usec`` from a cgroup v2 ``cpu.stat`` file."""
    try:
        with (scope_dir / "cpu.stat").open() as f:
            for line in f:
                if line.startswith("usage_usec "):
                    return int(line.split()[1])
    except FileNotFoundError:
        return None
    except (OSError, ValueError):
        return None
    return None


def _read_memory_current(scope_dir: Path) -> int | None:
    """Parse ``memory.current`` (bytes) from a cgroup v2 scope dir."""
    try:
        with (scope_dir / "memory.current").open() as f:
            return int(f.read().strip())
    except FileNotFoundError:
        return None
    except (OSError, ValueError):
        return None

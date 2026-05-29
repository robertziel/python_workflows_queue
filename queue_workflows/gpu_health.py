"""Per-container GPU + RAM health samplers for the GPU health watchdog.

The health watchdog (``claim_worker.GpuHealthWatchdog``) needs two *per-this-
container* liveness signals, sampled cheaply from INSIDE the gpu worker
container (which has the GPU device passed through but does NOT mount the host
cgroup tree or the docker socket — see ``docker-compose.yml`` ``workers-gpu``):

  1. GPU utilization attributable to THIS container's processes, and
  2. THIS container's resident memory.

Both are deliberately read from the container's own namespaced views so they
need no extra mounts and — critically — EXCLUDE the ollama VLM sidecar that
shares the box-c GPU on host-a/host-b (the box-level util would false-negative a
real render hang because ollama keeps the box GPU > 0 %).


GPU utilization — ``gpu_util_pct()``
------------------------------------
NVIDIA (box-c/Blackwell): ``nvidia-smi pmon -c 1 -s u`` reports a per-PROCESS
``sm%`` column. The NVIDIA driver renders the PID column in the *caller's* PID
namespace and lists ONLY the GPU processes visible in that namespace, so a call
from inside the gpu worker container returns rows for this container's own
processes only — the ollama sidecar (a different container / PID namespace) is
not listed (verified on host-b: ollama's busy python is invisible from the
render container's pmon, and vice-versa). We take the MAX ``sm%`` across the
returned rows as the per-container GPU-busy signal.

LIMITATION (documented, by design): on the box-c the per-PID ``sm%`` column is
frequently ``-`` (N/A) — it only populates when a process is actively issuing SM
work, and even then can be sparse. We treat ``-`` as ``0`` (no *measurable* GPU
work). That is the safe reading for a watchdog whose whole job is to detect "no
GPU work": a genuinely busy kernel reports a high ``sm%`` (observed 95 on a live
render), while a wedged process reports ``-``→0. The residual risk — a process
doing real GPU work that pmon can't measure — is covered by the watchdog's
SECOND arm: it only trips when GPU looks idle AND container RAM is static, so a
working-but-unmeasurable job whose RAM moves is never killed.

ROCm (host-c overflow worker) + any non-NVIDIA / no-pmon box: pmon isn't
available, so ``gpu_util_pct()`` falls back to the box-level
``nvidia-smi``/``rocm-smi`` utilization (via ``hw_metrics._gpu_probe``). On
host-c there is no ollama GPU sidecar, so the box-level signal is already
per-host-clean there; and host-c runs the qwen pipeline fine (the stall is
box-c-specific), so the fallback's coarser attribution is acceptable for the
overflow host.


Container RAM — ``container_ram_mb()``
--------------------------------------
cgroup v2 ``memory.current`` for THIS container, read from the container's own
namespaced cgroup root (``/sys/fs/cgroup/memory.current``). The cgroup namespace
makes the container's own scope appear at the root inside the container, so this
is the worker container's own RSS without needing the host cgroup mount that
``hw_metrics``'s ``CgroupAttribution`` uses (workers-gpu doesn't have it). Falls
back to the process RSS (``/proc/self/status`` VmRSS) if the cgroup file is
absent.
"""

from __future__ import annotations

import logging
import os
import subprocess

log = logging.getLogger(__name__)


# Where a cgroup-v2 container sees its OWN memory accounting (namespaced root).
_CGROUP_MEMORY_CURRENT = "/sys/fs/cgroup/memory.current"

# pmon single-sample call is cheap (~0.06 s on box-c); cap it hard so a stuck
# nvidia-smi can never block the watchdog's poll thread.
_PMON_TIMEOUT_S = 4


def gpu_util_pct() -> int:
    """Best per-this-container GPU utilization %, 0..100.

    Tries the NVIDIA per-process ``pmon`` path first (namespaced to this
    container, excludes the ollama sidecar); falls back to the box-level probe
    when pmon yields nothing usable (ROCm / no nvidia-smi / pmon empty). Returns
    ``0`` if no signal is obtainable at all — the watchdog's RAM arm is then the
    sole guard, so a no-signal box can't be killed on the GPU arm alone.
    """
    v = _nvidia_pmon_container_sm_pct()
    if v is not None:
        return v
    return _box_gpu_util_pct()


def _nvidia_pmon_container_sm_pct() -> int | None:
    """MAX per-process ``sm%`` from ``nvidia-smi pmon -c 1 -s u``.

    The rows are already scoped to this container's PID namespace (the driver
    lists only the GPU processes visible to the caller), so the max ``sm%``
    across them is THIS container's GPU-busy level — the ollama sidecar in its
    own container is not included.

    Returns ``None`` (→ caller falls back to box level) when pmon is absent /
    errors / prints only its header. A row whose ``sm%`` is ``-`` (N/A — the
    common box-c case for a process not actively issuing SM work) counts as 0.
    """
    try:
        raw = subprocess.check_output(
            ["nvidia-smi", "pmon", "-c", "1", "-s", "u"],
            stderr=subprocess.DEVNULL, timeout=_PMON_TIMEOUT_S,
        ).decode()
    except Exception:
        return None
    return _parse_pmon_sm_pct(raw)


def _parse_pmon_sm_pct(raw: str) -> int | None:
    """Parse the MAX ``sm%`` from ``nvidia-smi pmon -s u`` text output.

    pmon layout (``-s u``)::

        # gpu        pid  type    sm   mem   enc   dec   command
        # Idx          #   C/G     %     %     %     %    name
            0         123    C    95     0     -     -    python
            0         456    G     -     -     -     -    Xorg

    Column 0 = gpu idx, 1 = pid, 2 = type, 3 = sm%. We scan data rows (skip
    ``#`` headers), read column 3, treat ``-`` as 0, and return the max. Returns
    ``None`` when there are no data rows (pmon printed only its header, or the
    one all-``-`` placeholder row an idle namespace emits) so the caller falls
    back to the box-level probe rather than reading a hard 0.
    """
    best: int | None = None
    saw_real_pid = False
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 4:
            continue
        pid_tok, sm_tok = parts[1], parts[3]
        # The placeholder row an empty namespace prints is all dashes
        # ("0  -  -  - ... -"); pid_tok == "-" ⇒ no process ⇒ not a data row.
        if pid_tok == "-":
            continue
        saw_real_pid = True
        try:
            sm = 0 if sm_tok == "-" else int(sm_tok)
        except ValueError:
            sm = 0
        best = sm if best is None else max(best, sm)
    if not saw_real_pid:
        # No process rows at all → pmon gave us nothing about THIS container;
        # let the caller use the box-level probe.
        return None
    return best if best is not None else 0


def _box_gpu_util_pct() -> int:
    """Box-level max GPU utilization % via the shared vendor probe.

    Fallback for ROCm / no-pmon hosts. NOTE: this is BOX-level — on a box-c with
    the ollama sidecar it can read > 0 even when this container's render is
    wedged (false-negative), which is exactly why the NVIDIA path above is
    preferred. Used only where the per-container path is unavailable (host-c
    ROCm overflow worker, where there is no GPU sidecar anyway). Returns 0 on
    any failure."""
    try:
        from queue_workflows import hw_metrics
        gpus = hw_metrics._gpu_probe()
    except Exception:
        return 0
    if not gpus:
        return 0
    try:
        return max(int(g.get("use_pct") or 0) for g in gpus)
    except Exception:
        return 0


def container_ram_mb() -> int | None:
    """THIS container's resident memory in MB (cgroup v2 ``memory.current``).

    Read from the container's own namespaced cgroup root so it needs no host
    cgroup mount (workers-gpu doesn't have one). Falls back to the process RSS
    (``/proc/self/status`` VmRSS) when the cgroup file is absent, and returns
    ``None`` only when neither is readable — the watchdog treats ``None`` as "no
    RAM signal this checkpoint" and does NOT trip on the RAM arm."""
    try:
        with open(_CGROUP_MEMORY_CURRENT) as f:
            return int(f.read().strip()) // (1024 * 1024)
    except (FileNotFoundError, OSError, ValueError):
        pass
    return _proc_self_rss_mb()


def _proc_self_rss_mb() -> int | None:
    try:
        with open(f"/proc/{os.getpid()}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) // 1024
    except Exception:
        return None
    return None


__all__ = ["gpu_util_pct", "container_ram_mb"]

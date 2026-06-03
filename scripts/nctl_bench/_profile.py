"""Lightweight hotspot profiler for NCTL Split-MNIST runs.

Designed for the 30.9 Phase 5H-1 instrumentation pass: pinpoint where the
remaining 379s/seed budget is spent so 5H-2/5H-3/5H-4 can target the actual
hotspots instead of guessing.

Disabled-by-default contract:

* When the profiler is OFF (the default), the ``prof()`` context manager is
  a one-line ``yield`` -- zero allocations, zero ``time.monotonic()`` calls,
  zero ``torch.cuda.synchronize()`` calls.  Hot paths therefore pay nothing
  in production runs.
* When ON, every entry calls ``time.monotonic()`` plus (optionally) a
  ``torch.cuda.synchronize()`` so GPU sections are timed end-to-end rather
  than dispatch-to-dispatch.  ``sync=False`` lets caller skip the sync for
  pure-CPU sections (e.g. provenance JSON build, FIFO eviction).

The profiler is a module-global singleton so instrumentation can be sprinkled
deep into the layer code without threading a ProfileTimer reference through
every method.  ``configure(...)`` enables it and selects whether to sync CUDA;
``snapshot()`` returns a JSON-serialisable view; ``reset()`` clears counters.
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path


class ProfileTimer:
    """Accumulating per-section timer.  Thread-unsafe by design (NCTL is
    single-threaded inside ``train_chunk``); concurrent use would skew totals.
    """

    __slots__ = (
        "enabled",
        "sync_cuda",
        "_totals_s",
        "_calls",
        "_min_s",
        "_max_s",
    )

    def __init__(self) -> None:
        self.enabled: bool = False
        self.sync_cuda: bool = False
        self._totals_s: dict[str, float] = defaultdict(float)
        self._calls: dict[str, int] = defaultdict(int)
        self._min_s: dict[str, float] = {}
        self._max_s: dict[str, float] = defaultdict(float)

    def configure(self, *, enabled: bool, sync_cuda: bool = False) -> None:
        self.enabled = bool(enabled)
        self.sync_cuda = bool(sync_cuda)

    def reset(self) -> None:
        self._totals_s.clear()
        self._calls.clear()
        self._min_s.clear()
        self._max_s.clear()

    def add(self, section: str, dt_s: float) -> None:
        self._totals_s[section] += dt_s
        self._calls[section] += 1
        prev_min = self._min_s.get(section)
        if prev_min is None or dt_s < prev_min:
            self._min_s[section] = dt_s
        if dt_s > self._max_s[section]:
            self._max_s[section] = dt_s

    def snapshot(self) -> dict[str, dict[str, float | int]]:
        """Return a JSON-serialisable summary keyed by section name."""
        out: dict[str, dict[str, float | int]] = {}
        for section, total in self._totals_s.items():
            calls = self._calls[section]
            out[section] = {
                "total_s": float(total),
                "calls": int(calls),
                "mean_s": float(total / calls) if calls else 0.0,
                "min_s": float(self._min_s.get(section, 0.0)),
                "max_s": float(self._max_s[section]),
            }
        return out

    def write_json(self, path: Path) -> None:
        payload = {
            "enabled": self.enabled,
            "sync_cuda": self.sync_cuda,
            "sections": self.snapshot(),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True))

    def format_table(self, top: int | None = None) -> str:
        """Render a sorted human-readable table (descending total time).

        Used by ``run_split_mnist.py`` to print a summary to stderr at the
        end of every profiled run.
        """
        snap = self.snapshot()
        rows = sorted(snap.items(), key=lambda kv: -kv[1]["total_s"])
        if top is not None:
            rows = rows[:top]
        if not rows:
            return "(profiler enabled but no sections were recorded)"
        # Compute name width for alignment, capped at 48 chars.
        name_w = min(48, max(len("section"), max(len(name) for name, _ in rows)))
        header = (
            f"{'section':<{name_w}} {'total_s':>10} {'calls':>10} "
            f"{'mean_s':>12} {'max_s':>12}"
        )
        lines = [header, "-" * len(header)]
        for name, stats in rows:
            lines.append(
                f"{name:<{name_w}} "
                f"{stats['total_s']:>10.3f} "
                f"{stats['calls']:>10d} "
                f"{stats['mean_s']:>12.6f} "
                f"{stats['max_s']:>12.6f}"
            )
        return "\n".join(lines)


_PROFILER = ProfileTimer()


def get_profiler() -> ProfileTimer:
    """Return the module-global ProfileTimer singleton.

    Test code MUST use this rather than instantiating its own to ensure the
    instrumentation calls scattered through nctl_network.py share a single
    accumulator.
    """
    return _PROFILER


def configure_profiler(*, enabled: bool, sync_cuda: bool = False) -> None:
    """Convenience wrapper that mutates the singleton."""
    _PROFILER.configure(enabled=enabled, sync_cuda=sync_cuda)


def reset_profiler() -> None:
    _PROFILER.reset()


@contextmanager
def prof(section: str, *, sync: bool = True):
    """Time a ``with`` block under the module-global profiler.

    Optimised disabled-path: zero allocations + zero clock calls so production
    runs pay nothing for the instrumentation.  When enabled and ``sync=True``,
    ``torch.cuda.synchronize()`` is called BEFORE the timer starts AND BEFORE
    the elapsed read so a GPU section's dispatch latency does NOT leak into
    the next section's total.  Callers in pure-CPU paths should pass
    ``sync=False`` to skip the syncs.
    """
    if not _PROFILER.enabled:
        yield
        return
    if sync and _PROFILER.sync_cuda:
        # Lazy import keeps the disabled path import-light.
        import torch  # noqa: PLC0415
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    t0 = time.monotonic()
    try:
        yield
    finally:
        if sync and _PROFILER.sync_cuda:
            import torch  # noqa: PLC0415
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        _PROFILER.add(section, time.monotonic() - t0)


def write_profile_json(path: Path | str) -> None:
    """Convenience: dump the singleton's snapshot to a JSON file."""
    _PROFILER.write_json(Path(path))


def format_profile_table(top: int | None = None) -> str:
    return _PROFILER.format_table(top=top)


__all__ = [
    "ProfileTimer",
    "configure_profiler",
    "format_profile_table",
    "get_profiler",
    "prof",
    "reset_profiler",
    "write_profile_json",
]

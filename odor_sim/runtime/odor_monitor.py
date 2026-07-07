"""Live terminal log + matplotlib plot of per-VOC ppm at the EE (Phase 4.5f).

Debug/operator UX only — not part of the Phase 5 dataset schema. Call
:meth:`OdorMonitor.record` after each GADEN ppm query (from ``OdorCosimSession``
or teleop). Updates are single-threaded from the control loop.
"""

from __future__ import annotations

import os
from collections import deque

import numpy as np

# Rolling history cap (samples) — avoids unbounded growth if history_s is huge.
_MAX_SAMPLES = 50_000


def _color_for_gas(gas_name: str) -> tuple[float, float, float]:
    """RGB in 0..1 for a GADEN gas name string."""
    from odor_sim.config.gas_types import parse_gas_type
    from odor_sim.envs.scene_builder import _gas_color

    try:
        gt = parse_gas_type(gas_name)
        rgb = _gas_color(int(gt))
    except (ValueError, KeyError):
        rgb = [0.7, 0.7, 0.7]
    return (float(rgb[0]), float(rgb[1]), float(rgb[2]))


def _display_available() -> bool:
    """Best-effort check that an interactive matplotlib window can open."""
    if os.name != "nt" and not os.environ.get("DISPLAY"):
        return False
    try:
        import matplotlib

        backend = matplotlib.get_backend().lower()
        if backend in ("agg", "svg", "pdf", "ps", "cairo"):
            return False
    except Exception:  # noqa: BLE001
        return False
    return True


def parse_odor_monitor_spec(spec) -> dict:
    """Normalize ``odor_monitor`` make() kwarg into constructor kwargs.

    Returns ``{"enabled": False}`` when off; otherwise ``{"enabled": True, ...}``.
    """
    if not spec:
        return {"enabled": False}
    if spec is True:
        return {"enabled": True, "log": True, "plot": True}
    if spec == "log":
        return {"enabled": True, "log": True, "plot": False}
    if spec == "plot":
        return {"enabled": True, "log": False, "plot": True}
    if isinstance(spec, dict):
        out = {"enabled": True, "log": bool(spec.get("log", True)), "plot": bool(spec.get("plot", True))}
        for key in ("history_s", "log_every_n"):
            if key in spec:
                out[key] = spec[key]
        return out
    raise ValueError(
        f"odor_monitor must be False, True, 'log', 'plot', or a dict; got {spec!r}"
    )


class OdorMonitor:
    """Terminal live log + optional matplotlib rolling ppm plot.

    Args:
        gas_types: ordered list of unique gas names (e.g. ``['ethanol', 'acetone']``).
        log: print one line per step to stdout.
        plot: open a matplotlib window with one line per gas.
        history_s: rolling window length in GADEN sim_time seconds.
        log_every_n: print every N-th record (1 = every step).
    """

    def __init__(
        self,
        gas_types: list[str],
        *,
        log: bool = True,
        plot: bool = True,
        history_s: float = 120.0,
        log_every_n: int = 1,
    ):
        self.gas_types = list(gas_types)
        self.log = bool(log)
        self.plot_enabled = bool(plot)
        self.history_s = float(history_s)
        self.log_every_n = max(1, int(log_every_n))

        self._sim_times: deque[float] = deque(maxlen=_MAX_SAMPLES)
        self._ppm: dict[str, deque[float]] = {g: deque(maxlen=_MAX_SAMPLES) for g in self.gas_types}
        self._step_count = 0

        self._fig = None
        self._ax = None
        self._lines: dict[str, object] = {}
        self._plot_warned = False

        if self.plot_enabled:
            if _display_available():
                self._init_plot()
            else:
                self._warn_plot_fallback()

    def _warn_plot_fallback(self) -> None:
        if not self._plot_warned:
            print("[odor_monitor] no display; plot disabled (log-only)")
            self._plot_warned = True
        self.plot_enabled = False

    def _init_plot(self) -> None:
        import matplotlib.pyplot as plt

        plt.ion()
        self._fig, self._ax = plt.subplots(figsize=(8, 4))
        self._ax.set_title("EE ppm — OdorSim")
        self._ax.set_xlabel("sim time (s)")
        self._ax.set_ylabel("ppm")
        self._ax.grid(True, alpha=0.3)
        for gas in self.gas_types:
            (line,) = self._ax.plot([], [], label=gas, color=_color_for_gas(gas), linewidth=1.5)
            self._lines[gas] = line
        self._ax.legend(loc="upper right", fontsize=8)
        self._fig.tight_layout()
        self._fig.canvas.draw_idle()
        plt.pause(0.001)

    # ------------------------------------------------------------------ #
    @property
    def history(self) -> dict:
        """``{sim_time: ndarray, <gas>: ndarray, ...}`` for read-back."""
        t = np.asarray(self._sim_times, dtype=float)
        out: dict = {"sim_time": t}
        for gas in self.gas_types:
            out[gas] = np.asarray(self._ppm[gas], dtype=float)
        return out

    def reset(self) -> None:
        """Clear buffers (call on env / episode reset)."""
        self._sim_times.clear()
        for q in self._ppm.values():
            q.clear()
        self._step_count = 0
        if self.plot_enabled and self._ax is not None:
            for line in self._lines.values():
                line.set_data([], [])
            self._ax.relim()
            self._ax.autoscale_view()
            self._refresh_plot()

    def record(self, sim_time: float, ppm: dict) -> None:
        """Append one sample and update log/plot."""
        t = float(sim_time)
        self._sim_times.append(t)
        for gas in self.gas_types:
            self._ppm[gas].append(float(ppm.get(gas, 0.0)))

        self._step_count += 1
        self._trim_history(t)

        if self.log and (self._step_count % self.log_every_n == 0):
            parts = "  ".join(f"{g}={ppm.get(g, 0.0):.2f}" for g in self.gas_types)
            print(f"[odor] t={t:.2f}s  {parts}")

        if self.plot_enabled:
            self._update_plot()

    def _trim_history(self, t_now: float) -> None:
        if self.history_s <= 0:
            return
        t_min = t_now - self.history_s
        while self._sim_times and self._sim_times[0] < t_min:
            self._sim_times.popleft()
            for q in self._ppm.values():
                if q:
                    q.popleft()

    def _update_plot(self) -> None:
        if self._ax is None:
            return
        t = np.asarray(self._sim_times, dtype=float)
        if t.size == 0:
            return
        for gas, line in self._lines.items():
            y = np.asarray(self._ppm[gas], dtype=float)
            if y.size == t.size:
                line.set_data(t, y)
        self._ax.relim()
        self._ax.autoscale_view()
        self._refresh_plot()

    def _refresh_plot(self) -> None:
        if self._fig is None:
            return
        import matplotlib.pyplot as plt

        self._fig.canvas.draw_idle()
        plt.pause(0.001)

    def close(self) -> None:
        """Close the matplotlib figure if open."""
        if self._fig is not None:
            import matplotlib.pyplot as plt

            plt.close(self._fig)
            self._fig = None
            self._ax = None
            self._lines.clear()
            self.plot_enabled = False

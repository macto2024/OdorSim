"""Live terminal log + matplotlib plot of MOX e-nose voltage (live sensor UX).

Debug/operator UX only — not part of the dataset schema. Call
:meth:`SensorMonitor.record` after each live enose update (from
``OdorCosimSession``). Spec parsing mirrors :func:`parse_odor_monitor_spec`.
"""

from __future__ import annotations

import os
from collections import deque

import numpy as np

_MAX_SAMPLES = 50_000

# Distinct colors for the five stock MOX models (and any extras).
_MODEL_COLORS = {
    "TGS2620": (0.85, 0.25, 0.25),
    "TGS2600": (0.20, 0.55, 0.85),
    "TGS2611": (0.30, 0.70, 0.35),
    "TGS2610": (0.90, 0.55, 0.15),
    "TGS2612": (0.60, 0.35, 0.75),
}


def _color_for_model(name: str) -> tuple[float, float, float]:
    if name in _MODEL_COLORS:
        return _MODEL_COLORS[name]
    # Stable hash → RGB for unexpected model ids.
    h = abs(hash(name)) % (256**3)
    return ((h & 255) / 255.0, ((h >> 8) & 255) / 255.0, ((h >> 16) & 255) / 255.0)


def _display_available() -> bool:
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


def parse_sensor_monitor_spec(spec) -> dict:
    """Normalize ``sensor_monitor`` make() kwarg into constructor kwargs.

    Returns ``{"enabled": False}`` when off; otherwise ``{"enabled": True, ...}``.
    Same shapes as :func:`~odor_sim.runtime.odor_monitor.parse_odor_monitor_spec`.
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
        out = {
            "enabled": True,
            "log": bool(spec.get("log", True)),
            "plot": bool(spec.get("plot", True)),
        }
        for key in ("history_s", "log_every_n"):
            if key in spec:
                out[key] = spec[key]
        return out
    raise ValueError(
        f"sensor_monitor must be False, True, 'log', 'plot', or a dict; got {spec!r}"
    )


class SensorMonitor:
    """Terminal live log + optional matplotlib rolling MOX voltage plot.

    Args:
        mox_models: ordered list of MOX model ids (e.g. ``['TGS2620', ...]``).
        log: print one line per step to stdout.
        plot: open a matplotlib window with one line per model.
        history_s: rolling window length in sim-time seconds.
        log_every_n: print every N-th record (1 = every step).
        primary_mox_model: optional model highlighted in the log line.
    """

    def __init__(
        self,
        mox_models: list[str],
        *,
        log: bool = True,
        plot: bool = True,
        history_s: float = 120.0,
        log_every_n: int = 1,
        primary_mox_model: str | None = None,
    ):
        self.mox_models = list(mox_models)
        if not self.mox_models:
            raise ValueError("SensorMonitor requires a non-empty mox_models list")
        self.log = bool(log)
        self.plot_enabled = bool(plot)
        self.history_s = float(history_s)
        self.log_every_n = max(1, int(log_every_n))
        self.primary_mox_model = primary_mox_model or self.mox_models[0]

        self._sim_times: deque[float] = deque(maxlen=_MAX_SAMPLES)
        self._voltage: dict[str, deque[float]] = {
            m: deque(maxlen=_MAX_SAMPLES) for m in self.mox_models
        }
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
            print("[sensor_monitor] no display; plot disabled (log-only)")
            self._plot_warned = True
        self.plot_enabled = False

    def _init_plot(self) -> None:
        import matplotlib.pyplot as plt

        plt.ion()
        self._fig, self._ax = plt.subplots(figsize=(8, 4))
        self._ax.set_title("EE MOX voltage — OdorSim")
        self._ax.set_xlabel("sim time (s)")
        self._ax.set_ylabel("voltage (V)")
        self._ax.grid(True, alpha=0.3)
        for model in self.mox_models:
            (line,) = self._ax.plot(
                [],
                [],
                label=model,
                color=_color_for_model(model),
                linewidth=1.5,
            )
            self._lines[model] = line
        self._ax.legend(loc="upper right", fontsize=8)
        self._fig.tight_layout()
        self._fig.canvas.draw_idle()
        plt.pause(0.001)

    @property
    def history(self) -> dict:
        """``{sim_time: ndarray, <model>: ndarray, ...}`` for read-back."""
        t = np.asarray(self._sim_times, dtype=float)
        out: dict = {"sim_time": t}
        for model in self.mox_models:
            out[model] = np.asarray(self._voltage[model], dtype=float)
        return out

    def reset(self) -> None:
        """Clear buffers (call on env / episode reset)."""
        self._sim_times.clear()
        for q in self._voltage.values():
            q.clear()
        self._step_count = 0
        if self.plot_enabled and self._ax is not None:
            for line in self._lines.values():
                line.set_data([], [])
            self._ax.relim()
            self._ax.autoscale_view()
            self._refresh_plot()

    def record(
        self,
        sim_time: float,
        voltages: dict,
        *,
        enose_state: int | None = None,
        sampling_active: bool | None = None,
    ) -> None:
        """Append one multi-sensor sample and update log/plot.

        Args:
            sim_time: episode-relative time (s).
            voltages: ``{model_id: voltage_V}``.
            enose_state: optional ternary token for the log line.
            sampling_active: optional valve mask for the log line.
        """
        t = float(sim_time)
        self._sim_times.append(t)
        for model in self.mox_models:
            self._voltage[model].append(float(voltages.get(model, 0.0)))

        self._step_count += 1
        self._trim_history(t)

        if self.log and (self._step_count % self.log_every_n == 0):
            parts = "  ".join(f"{m}={voltages.get(m, 0.0):.3f}V" for m in self.mox_models)
            extra = ""
            if enose_state is not None:
                token = {1: "SAMPLE", 0: "idle", -1: "FILTER"}.get(int(enose_state), str(enose_state))
                samp = ""
                if sampling_active is not None:
                    samp = " ON" if sampling_active else " off"
                extra = f"  enose={token}{samp}"
            print(f"[sensor] t={t:.2f}s  {parts}{extra}")

        if self.plot_enabled:
            self._update_plot()

    def _trim_history(self, t_now: float) -> None:
        if self.history_s <= 0:
            return
        t_min = t_now - self.history_s
        while self._sim_times and self._sim_times[0] < t_min:
            self._sim_times.popleft()
            for q in self._voltage.values():
                if q:
                    q.popleft()

    def _update_plot(self) -> None:
        if self._ax is None:
            return
        t = np.asarray(self._sim_times, dtype=float)
        if t.size == 0:
            return
        for model, line in self._lines.items():
            y = np.asarray(self._voltage[model], dtype=float)
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

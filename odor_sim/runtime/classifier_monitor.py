"""Live terminal log + matplotlib panel for ScentFormer odor classification."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from odor_sim.runtime.odor_classifier import MultiWindowOdorPrediction, OdorPrediction

from odor_sim.runtime.odor_classifier import PREFIX_SECONDS

_GAS_LABELS = {
    "ethanol": "ethanol",
    "methanol": "methanol",
    "acetone": "acetone",
    "acetic_acid": "acetic acid",
    "ethyl_acetate": "ethyl acetate",
    "toluene": "toluene",
}

_GT_EDGE = (0.15, 0.65, 0.15)
_BAR_COLOR = (0.25, 0.45, 0.85)
_WAIT_COLOR = (0.75, 0.75, 0.75)


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


def parse_classifier_monitor_spec(spec) -> dict:
    """Normalize ``classifier_monitor`` make() kwarg into constructor kwargs."""
    if not spec:
        return {"enabled": False}
    if spec is True:
        return {"enabled": True, "log": True, "plot": True}
    if spec == "log":
        return {"enabled": True, "log": True, "plot": False}
    if spec == "plot":
        return {"enabled": True, "log": False, "plot": True}
    if isinstance(spec, dict):
        return {
            "enabled": True,
            "log": bool(spec.get("log", True)),
            "plot": bool(spec.get("plot", True)),
            **{k: spec[k] for k in ("log_every_n",) if k in spec},
        }
    raise ValueError(
        f"classifier_monitor must be False, True, 'log', 'plot', or a dict; got {spec!r}"
    )


class ClassifierMonitor:
    """Terminal log + optional matplotlib multi-window confidence panel."""

    def __init__(
        self,
        gas_names: list[str] | tuple[str, ...],
        *,
        log: bool = True,
        plot: bool = True,
        log_every_n: int = 1,
        prefix_seconds: tuple[int, ...] = PREFIX_SECONDS,
    ):
        self.gas_names = list(gas_names)
        self.prefix_seconds = tuple(prefix_seconds)
        self.log = bool(log)
        self.plot_enabled = bool(plot)
        self.log_every_n = max(1, int(log_every_n))
        self._ground_truth: dict[str, str] = {}
        self._record_count = 0
        self._last_prediction: MultiWindowOdorPrediction | None = None

        self._fig = None
        self._ax = None
        self._bars = None
        self._gt_text = None
        self._row_labels: list[str] = []
        self._plot_warned = False

        if self.plot_enabled:
            if _display_available():
                self._init_plot()
            else:
                self._warn_plot_fallback()

    def _warn_plot_fallback(self) -> None:
        if not self._plot_warned:
            print("[classifier_monitor] no display; plot disabled (log-only)")
            self._plot_warned = True
        self.plot_enabled = False

    def _init_plot(self) -> None:
        import matplotlib.pyplot as plt

        plt.ion()
        n_rows = len(self.prefix_seconds)
        self._fig, self._ax = plt.subplots(figsize=(8, 0.55 * n_rows + 1.5))
        self._ax.set_title("ScentFormer — fixed window classifications")
        self._ax.set_xlabel("top-gas confidence")
        self._ax.set_xlim(0.0, 1.0)
        y_pos = np.arange(n_rows)
        self._bars = self._ax.barh(
            y_pos,
            np.zeros(n_rows),
            color=_WAIT_COLOR,
            edgecolor="none",
            height=0.55,
        )
        self._row_labels = [f"{s}s  waiting" for s in self.prefix_seconds]
        self._ax.set_yticks(y_pos)
        self._ax.set_yticklabels(self._row_labels, fontsize=9, family="monospace")
        self._ax.invert_yaxis()
        self._ax.grid(True, axis="x", alpha=0.3)
        self._gt_text = self._ax.text(
            0.02,
            0.98,
            "",
            transform=self._ax.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            family="monospace",
        )
        self._fig.tight_layout()
        self._fig.canvas.draw_idle()
        plt.pause(0.001)

    def set_ground_truth(self, cups: dict[str, str]) -> None:
        """Set cup slot → liquid name mapping (e.g. ``{'left': 'ethanol'}``)."""
        self._ground_truth = dict(cups)
        self._update_gt_text()

    def reset(self) -> None:
        """Clear displayed state (call on env / episode reset)."""
        self._last_prediction = None
        self._record_count = 0
        if self.plot_enabled and self._bars is not None and self._ax is not None:
            for bar in self._bars:
                bar.set_width(0.0)
                bar.set_color(_WAIT_COLOR)
                bar.set_edgecolor("none")
            self._row_labels = [f"{s}s  waiting" for s in self.prefix_seconds]
            self._ax.set_yticklabels(self._row_labels, fontsize=9, family="monospace")
            self._ax.set_title("ScentFormer — fixed window classifications")
            self._refresh_plot()

    def _gt_gases(self) -> set[str]:
        return set(self._ground_truth.values())

    def _update_gt_text(self) -> None:
        if self._gt_text is None:
            return
        if not self._ground_truth:
            self._gt_text.set_text("")
            return
        parts = [f"{slot}={liquid}" for slot, liquid in sorted(self._ground_truth.items())]
        self._gt_text.set_text("GT: " + "  ".join(parts))

    def record(self, sim_time: float, prediction: "MultiWindowOdorPrediction") -> None:
        """Update log/plot from a batched multi-window prediction."""
        self._last_prediction = prediction
        self._record_count += 1

        if self.log and (self._record_count % self.log_every_n == 0):
            gt = ""
            if self._ground_truth:
                gt = " | GT " + " ".join(
                    f"{slot}={liquid}" for slot, liquid in sorted(self._ground_truth.items())
                )
            lines = [f"[classifier] t={sim_time:.1f}s device={prediction.device}{gt}"]
            ready = {int(round(p.window_s)): p for p in prediction.predictions}
            for window_s in self.prefix_seconds:
                pred = ready.get(window_s)
                if pred is None:
                    lines.append(f"  {window_s:2d}s  — waiting")
                else:
                    label = _GAS_LABELS.get(pred.top_gas, pred.top_gas)
                    lines.append(
                        f"  {window_s:2d}s  {label} {pred.top_confidence:.2f} "
                        f"(clean={pred.clean_air:.2f})"
                    )
            print("\n".join(lines))

        if self.plot_enabled:
            self._update_plot(prediction)

    def _update_plot(self, prediction: "MultiWindowOdorPrediction") -> None:
        if self._ax is None or self._bars is None:
            return
        gt_gases = self._gt_gases()
        ready = {int(round(p.window_s)): p for p in prediction.predictions}
        primary = prediction.primary
        for i, window_s in enumerate(self.prefix_seconds):
            bar = self._bars[i]
            pred = ready.get(window_s)
            if pred is None:
                bar.set_width(0.0)
                bar.set_color(_WAIT_COLOR)
                bar.set_edgecolor("none")
                self._row_labels[i] = f"{window_s}s  waiting"
                continue
            width = float(pred.top_confidence)
            bar.set_width(width)
            bar.set_color(_BAR_COLOR)
            label = _GAS_LABELS.get(pred.top_gas, pred.top_gas)
            self._row_labels[i] = f"{window_s}s  {label} {width:.0%}"
            if pred.top_gas in gt_gases:
                bar.set_edgecolor(_GT_EDGE)
                bar.set_linewidth(2.5)
            else:
                bar.set_edgecolor("none")
        self._ax.set_yticklabels(self._row_labels, fontsize=9, family="monospace")
        title = (
            f"primary@20s={primary.top_gas} ({primary.top_confidence:.0%})  "
            f"device={prediction.device}"
        )
        self._ax.set_title(title)
        self._update_gt_text()
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
            self._bars = None
            self._gt_text = None
            self.plot_enabled = False

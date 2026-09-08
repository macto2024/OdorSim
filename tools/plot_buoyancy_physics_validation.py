#!/usr/bin/env python3
"""Plot fixed-cohort, zero-noise GADEN buoyancy validation runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

from odor_sim.odor_encoder.dataset import GAS_ORDER


RUNS = (
    ("tau = 0 s", "tau_0", "#222222", ":"),
    ("tau = 0.5 s", "tau_0p5", "#009E73", "-"),
    ("tau = 1.0 s", "tau_1p0", "#0072B2", "-"),
    ("tau = 2.0 s", "tau_2p0", "#CC79A7", "-"),
)


def _load(root: Path) -> dict[str, dict]:
    return {
        key: json.loads((root / key / "zero_wind_filament_fall.json").read_text())
        for _, key, _, _ in RUNS
    }


def _cohort_series(data: dict, gas: str) -> tuple[np.ndarray, np.ndarray]:
    rows = data["frames"][gas]
    return (
        np.asarray([row["time_s"] for row in rows], float),
        np.asarray([row["oldest_cohort_z_median"] for row in rows], float),
    )


def run(args: argparse.Namespace) -> Path:
    root = Path(args.input_dir)
    datasets = _load(root)
    baseline = datasets["tau_0"]
    source_z = float(baseline["source_position_gaden_m"][2])
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    for key, data in datasets.items():
        if float(data["filament_noise_std_configured"]) != 0.0:
            raise ValueError(f"{key} is not a zero-noise validation run")

    with PdfPages(output) as pdf:
        fig, axes = plt.subplots(2, 3, figsize=(13, 8), sharex=True, sharey=True)
        for axis, gas in zip(axes.flat, GAS_ORDER):
            for label, key, color, style in RUNS:
                time, height = _cohort_series(datasets[key], gas)
                axis.plot(
                    time, 1000.0 * (height - source_z),
                    color=color, linestyle=style, linewidth=2, label=label,
                )
            axis.axhline(0.0, color="black", linewidth=0.8)
            axis.set_title(gas.replace("_", " "))
            axis.grid(alpha=0.25)
        axes[0, 0].legend(fontsize=8)
        for axis in axes[-1]:
            axis.set_xlabel("cohort age (s)")
        for axis in axes[:, 0]:
            axis.set_ylabel("height relative to birth (mm)")
        fig.suptitle("GADEN buoyancy validation: oldest cohort, zero wind and zero noise")
        fig.tight_layout()
        pdf.savefig(fig)
        fig.savefig(output.with_suffix(".png"), dpi=180)
        plt.close(fig)

        fig, axes = plt.subplots(2, 3, figsize=(13, 8), sharex=True)
        for axis, gas in zip(axes.flat, GAS_ORDER):
            base_time, base_height = _cohort_series(baseline, gas)
            for label, key, color, style in RUNS[1:]:
                time, height = _cohort_series(datasets[key], gas)
                if not np.array_equal(time, base_time):
                    raise ValueError(f"time grid mismatch for {gas} in {key}")
                downward_difference_mm = 1000.0 * (base_height - height)
                axis.plot(
                    time, downward_difference_mm,
                    color=color, linestyle=style, linewidth=2, label=label,
                )
            axis.axhline(0.0, color="black", linewidth=0.8)
            axis.set_title(gas.replace("_", " "))
            axis.grid(alpha=0.25)
        axes[0, 0].legend(fontsize=8)
        for axis in axes[-1]:
            axis.set_xlabel("cohort age (s)")
        for axis in axes[:, 0]:
            axis.set_ylabel("additional downward displacement vs tau=0 (mm)")
        fig.suptitle("Paired buoyancy effect with stochastic motion removed")
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    print(f"wrote {output}")
    return output


def build_parser() -> argparse.ArgumentParser:
    root = Path("analysis/odor_encoder/buoyancy_physics_validation")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=root)
    parser.add_argument("--output", type=Path, default=root / "buoyancy_physics_validation.pdf")
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())

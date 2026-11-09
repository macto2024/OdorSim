#!/usr/bin/env python3
"""Compare original, disabled, and concentration-dependent buoyancy models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from odor_sim.odor_encoder.dataset import GAS_ORDER


def _series(rows: list[dict], key: str) -> np.ndarray:
    return np.asarray([row[key] for row in rows], dtype=float)


def run(args: argparse.Namespace) -> Path:
    datasets = [
        ("original: terminal enabled", args.original, "#D55E00", "--"),
        ("buoyancy disabled", args.disabled, "#222222", ":"),
        ("new: tau 0.5 s", args.tau_0p5, "#009E73", "-"),
        ("new: tau 1.0 s", args.tau_1p0, "#0072B2", "-"),
        ("new: tau 2.0 s", args.tau_2p0, "#CC79A7", "-"),
        ("new: tau 5.0 s", args.tau_5p0, "#E69F00", "-"),
    ]
    if args.without_original:
        datasets = datasets[1:]
    loaded = [
        (label, json.loads(Path(path).read_text()), color, style)
        for label, path, color, style in datasets
    ]
    disabled = next(data for label, data, _, _ in loaded if label == "buoyancy disabled")
    source_z = float(disabled["source_position_gaden_m"][2])

    fig, axes = plt.subplots(2, 3, figsize=(13, 8), sharex=True, sharey=True)
    for axis, gas in zip(axes.flat, GAS_ORDER):
        for label, data, color, style in loaded:
            rows = data["frames"][gas]
            is_disabled = label == "buoyancy disabled"
            axis.plot(
                _series(rows, "time_s"), _series(rows, "z_median"),
                color=color,
                linestyle=style,
                linewidth=1.5 if is_disabled else 1.1,
                zorder=10 if is_disabled else 3,
                label=label,
            )
        axis.axhline(source_z, color="black", linestyle=":", linewidth=1, label="source z")
        axis.set_title(gas.replace("_", " "))
        axis.grid(alpha=0.22)
    axes[0, 0].legend(fontsize=8, loc="lower left")
    for axis in axes[-1]:
        axis.set_xlabel("simulation time (s)")
    for axis in axes[:, 0]:
        axis.set_ylabel("active-filament center z (m)")
    comparison_name = (
        "Concentration-dependent buoyancy comparison"
        if args.without_original else "Zero-wind filament height comparison"
    )
    fig.suptitle(
        f"{comparison_name}\n"
        "lines: median active-filament height; stochastic motion enabled"
    )
    fig.tight_layout()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    fig.savefig(output.with_suffix(".png"), dpi=180)
    plt.close(fig)

    comparison = {
        "description": "Median active-filament height histories used in the comparison plot.",
        "source_z_m": source_z,
        "stochastic_motion_enabled": True,
        "curves": {},
    }
    for (label, path, _, _), (_, data, _, _) in zip(datasets, loaded):
        comparison["curves"][label] = {
            "input_json": str(path),
            "buoyancy_tau_s": data.get("buoyancy_tau_s"),
            "time_s": [row["time_s"] for row in data["frames"][GAS_ORDER[0]]],
            "median_z_m_by_gas": {
                gas: [row["z_median"] for row in data["frames"][gas]]
                for gas in GAS_ORDER
            },
        }
    json_output = output.with_suffix(".json")
    json_output.write_text(json.dumps(comparison, indent=2) + "\n")
    print(f"wrote {output}")
    print(f"wrote {json_output}")
    return output


def build_parser() -> argparse.ArgumentParser:
    root = Path("analysis/odor_encoder/zero_wind_filament_fall")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--original",
        default=root / "terminal_velocity_enabled/zero_wind_filament_fall.json",
        type=Path,
    )
    parser.add_argument(
        "--disabled", default=root / "zero_wind_filament_fall.json", type=Path,
    )
    parser.add_argument(
        "--tau-0p5", default=root / "tau_0p5/zero_wind_filament_fall.json", type=Path,
    )
    parser.add_argument(
        "--tau-1p0", default=root / "tau_1p0/zero_wind_filament_fall.json", type=Path,
    )
    parser.add_argument(
        "--tau-2p0", default=root / "tau_2p0/zero_wind_filament_fall.json", type=Path,
    )
    parser.add_argument(
        "--tau-5p0", default=root / "tau_5p0/zero_wind_filament_fall.json", type=Path,
    )
    parser.add_argument(
        "--output", default=root / "terminal_velocity_comparison.pdf", type=Path,
    )
    parser.add_argument(
        "--without-original", action="store_true",
        help="omit the archived terminal-velocity formula from the comparison",
    )
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())

#!/usr/bin/env python3
"""Visualize representative Phase B paths and their measured gas concentrations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

from odor_sim.odor_encoder.dataset import GAS_ORDER


PATHS = (
    "approach_descend_dwell_retreat",
    "approach_dwell_retreat",
    "crosswind_sweep",
    "stationary_downwind",
)
PATH_TITLES = {
    "approach_descend_dwell_retreat": "Approach, descend, dwell, retreat",
    "approach_dwell_retreat": "Approach, dwell, retreat",
    "crosswind_sweep": "Crosswind sweep, recenter, dwell",
    "stationary_downwind": "Stationary downwind",
}
GAS_COLORS = {
    "ethanol": "#0072B2",
    "methanol": "#E69F00",
    "acetone": "#009E73",
    "acetic_acid": "#D55E00",
    "ethyl_acetate": "#CC79A7",
    "toluene": "#6A3D9A",
}


def _target_gas(metadata: dict) -> tuple[str, float]:
    source = np.asarray(metadata["source_concentrations_ppm"][0], dtype=float)
    index = int(np.argmax(source))
    return GAS_ORDER[index], float(source[index])


def select_examples(
    root: Path, split: str, gas: str, source_ppm: float,
) -> dict[str, tuple[Path, dict]]:
    selected: dict[str, tuple[float, Path, dict]] = {}
    for metadata_path in sorted((root / split).glob("*.json")):
        metadata = json.loads(metadata_path.read_text())
        path_type = metadata.get("path_type")
        if path_type not in PATHS:
            continue
        target_gas, target_source_ppm = _target_gas(metadata)
        if target_gas != gas:
            continue
        score = abs(np.log10(target_source_ppm) - np.log10(source_ppm))
        current = selected.get(path_type)
        if current is None or score < current[0]:
            selected[path_type] = (score, metadata_path.with_suffix(".npz"), metadata)
    missing = set(PATHS) - set(selected)
    if missing:
        raise RuntimeError(
            f"no {gas} examples in split={split!r} for paths: {sorted(missing)}"
        )
    return {path: (selected[path][1], selected[path][2]) for path in PATHS}


def load_example(npz_path: Path, metadata: dict) -> dict:
    with np.load(npz_path) as raw:
        item = {
            "time": np.asarray(raw["time_s"], dtype=float),
            "position": np.asarray(raw["position"], dtype=float),
            "ppm": np.asarray(raw["ppm"], dtype=float),
            "phase": np.asarray(raw["phase"], dtype=str),
            "valid": np.asarray(raw["label_valid"], dtype=bool),
            "target_fraction": np.asarray(raw["target_fraction"], dtype=float),
        }
    item["metadata"] = metadata
    item["sources"] = np.asarray(metadata["source_positions"], dtype=float)
    item["source_concentrations"] = np.asarray(
        metadata["source_concentrations_ppm"], dtype=float
    )
    item["target_gas"], item["target_source_ppm"] = _target_gas(metadata)
    item["target_index"] = GAS_ORDER.index(item["target_gas"])
    item["distance"] = np.linalg.norm(
        item["position"] - item["sources"][0][None, :], axis=1
    )
    return item


def _set_equal_3d(ax, points: np.ndarray) -> None:
    minimum = points.min(axis=0)
    maximum = points.max(axis=0)
    center = 0.5 * (minimum + maximum)
    radius = max(float((maximum - minimum).max()) * 0.56, 0.04)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def _plot_trajectory(ax, item: dict, *, compact: bool = False):
    position = item["position"]
    progress = np.linspace(0.0, 1.0, len(position))
    ax.plot(position[:, 0], position[:, 1], position[:, 2], color="0.7", linewidth=1)
    scatter = ax.scatter(
        position[:, 0], position[:, 1], position[:, 2], c=progress,
        cmap="viridis", s=7 if compact else 10, linewidths=0,
    )
    ax.scatter(*position[0], marker="o", color="#0072B2", s=45, label="Start")
    ax.scatter(*position[-1], marker="s", color="#D55E00", s=42, label="End")
    ax.scatter(*item["sources"][0], marker="*", color="#C00000", s=140, label="Target source")
    for index, source in enumerate(item["sources"][1:], start=1):
        gases = [GAS_ORDER[i].replace("_", " ") for i in np.flatnonzero(item["source_concentrations"][index])]
        ax.scatter(*source, marker="X", color="#5A5A5A", s=65, label="Background source" if index == 1 else None)
        if not compact:
            ax.text(*source, ", ".join(gases), fontsize=7)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    _set_equal_3d(ax, np.vstack((position, item["sources"])))
    return scatter


def _shade_phases(ax, time: np.ndarray, phases: np.ndarray) -> None:
    changes = np.flatnonzero(phases[1:] != phases[:-1]) + 1
    boundaries = np.r_[0, changes, len(phases)]
    colors = ("#E8F1F8", "#F7EFE0")
    for index, (start, stop) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        ax.axvspan(time[start], time[stop - 1], color=colors[index % 2], alpha=0.35, linewidth=0)
        if stop - start > 8:
            middle = 0.5 * (time[start] + time[stop - 1])
            ax.text(middle, 0.98, phases[start].replace("_", "\n"), transform=ax.get_xaxis_transform(),
                    ha="center", va="top", fontsize=6, color="0.35")


def detail_figure(path_type: str, item: dict):
    metadata = item["metadata"]
    fig = plt.figure(figsize=(13, 8.5), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, width_ratios=(1.0, 1.35))
    path_ax = fig.add_subplot(grid[:, 0], projection="3d")
    distance_ax = fig.add_subplot(grid[0, 1])
    ppm_ax = fig.add_subplot(grid[1, 1])

    scatter = _plot_trajectory(path_ax, item)
    path_ax.set_title("E-nose path and gas sources", fontsize=10)
    path_ax.legend(loc="upper left", fontsize=7, frameon=False)
    fig.colorbar(scatter, ax=path_ax, label="Episode progress", shrink=0.56, pad=0.08)

    _shade_phases(distance_ax, item["time"], item["phase"])
    distance_ax.plot(item["time"], 100.0 * item["distance"], color="#0072B2", linewidth=1.5)
    distance_ax.axhspan(1.0, 5.0, color="#009E73", alpha=0.12, label="1-5 cm sniff band")
    distance_ax.set_ylabel("Target-source distance (cm)")
    distance_ax.set_xlabel("Time (s)")
    distance_ax.grid(True, alpha=0.22)
    distance_ax.legend(loc="upper right", fontsize=8, frameon=False)

    _shade_phases(ppm_ax, item["time"], item["phase"])
    for index, gas in enumerate(GAS_ORDER):
        is_target = index == item["target_index"]
        ppm_ax.plot(
            item["time"], item["ppm"][:, index], color=GAS_COLORS[gas],
            linewidth=2.0 if is_target else 1.0, alpha=1.0 if is_target else 0.78,
            label=f"{gas.replace('_', ' ')}{' (target)' if is_target else ''}",
        )
    valid_times = item["time"][item["valid"]]
    if len(valid_times):
        ppm_ax.scatter(valid_times, np.full(len(valid_times), 0.006), marker="|", s=18,
                       color="black", alpha=0.3, label="Valid endpoints")
    ppm_ax.axhline(0.1, color="0.25", linestyle="--", linewidth=0.8, label="0.1 ppm")
    ppm_ax.axhline(1.0, color="0.25", linestyle=":", linewidth=0.8, label="1 ppm")
    ppm_ax.set_yscale("symlog", linthresh=0.01, linscale=0.8)
    ppm_ax.set_xlabel("Time (s)")
    ppm_ax.set_ylabel("Concentration at e-nose (ppm)")
    ppm_ax.grid(True, which="both", alpha=0.22)
    ppm_ax.legend(loc="upper left", ncols=2, fontsize=7, frameon=False)

    wind = metadata.get("wind_regime", "unknown").replace("_", " ")
    fig.suptitle(
        f"{PATH_TITLES[path_type]} | {metadata['session_id']}\n"
        f"target={item['target_gas'].replace('_', ' ')} "
        f"({item['target_source_ppm']:.2f} ppm source setting), "
        f"wind={wind}, mean speed={metadata.get('mean_wind_speed_m_s', float('nan')):.3f} m/s",
        fontsize=13,
    )
    return fig


def overview_figure(examples: dict[str, dict]):
    fig = plt.figure(figsize=(13, 9), constrained_layout=True)
    for index, path_type in enumerate(PATHS, start=1):
        ax = fig.add_subplot(2, 2, index, projection="3d")
        _plot_trajectory(ax, examples[path_type], compact=True)
        item = examples[path_type]
        ax.set_title(
            f"{PATH_TITLES[path_type]}\n"
            f"{item['metadata']['session_id']} | min distance={100 * item['distance'].min():.1f} cm",
            fontsize=9,
        )
    fig.suptitle("Representative Phase B e-nose trajectories", fontsize=14)
    return fig


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("datasets/odor_encoder/virtual_single_v6"))
    parser.add_argument("--split", choices=("train", "validation", "test"), default="validation")
    parser.add_argument("--gas", choices=GAS_ORDER, default="ethyl_acetate")
    parser.add_argument("--source-ppm", type=float, default=10.0)
    parser.add_argument("--out-dir", type=Path, default=Path("analysis/odor_encoder/phase_b_trajectories"))
    args = parser.parse_args()
    if args.source_ppm <= 0:
        parser.error("--source-ppm must be positive")

    chosen = select_examples(args.root, args.split, args.gas, args.source_ppm)
    examples = {path: load_example(*chosen[path]) for path in PATHS}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"phase_b_trajectories_{args.gas}_{args.split}"
    pdf_path = args.out_dir / f"{stem}.pdf"
    overview_path = args.out_dir / f"{stem}_overview.png"

    overview = overview_figure(examples)
    overview.savefig(overview_path, dpi=180)
    with PdfPages(pdf_path) as pdf:
        pdf.savefig(overview)
        for path_type in PATHS:
            detail = detail_figure(path_type, examples[path_type])
            pdf.savefig(detail)
            plt.close(detail)
    plt.close(overview)

    for path_type in PATHS:
        item = examples[path_type]
        maxima = ", ".join(
            f"{gas}={item['ppm'][:, index].max():.3g}"
            for index, gas in enumerate(GAS_ORDER)
        )
        print(
            f"{path_type}: {item['metadata']['session_id']}, "
            f"distance={100 * item['distance'].min():.2f}-{100 * item['distance'].max():.2f} cm, "
            f"max ppm [{maxima}]"
        )
    print(f"wrote {pdf_path}")
    print(f"wrote {overview_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

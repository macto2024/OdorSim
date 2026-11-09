#!/usr/bin/env python3
"""Plot representative Phase B source distance against measured target ppm."""

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
    "stationary_downwind": "Stationary hover",
}


def select_examples(root: Path, gas: str, target_source_ppm: float) -> dict[str, tuple[Path, dict]]:
    gas_index = GAS_ORDER.index(gas)
    selected: dict[str, tuple[float, Path, dict]] = {}
    for metadata_path in root.glob("*/*.json"):
        metadata = json.loads(metadata_path.read_text())
        path_type = metadata.get("path_type")
        if path_type not in PATHS:
            continue
        source = np.asarray(metadata["source_concentrations_ppm"][0], dtype=float)
        source_ppm = float(source[gas_index])
        if source_ppm <= 0.0:
            continue
        score = abs(np.log10(source_ppm) - np.log10(target_source_ppm))
        current = selected.get(path_type)
        if current is None or score < current[0]:
            selected[path_type] = (score, metadata_path.with_suffix(".npz"), metadata)
    missing = set(PATHS) - set(selected)
    if missing:
        raise RuntimeError(f"no {gas} examples for paths: {sorted(missing)}")
    return {path: (selected[path][1], selected[path][2]) for path in PATHS}


def load_example(npz_path: Path, metadata: dict, gas: str) -> dict:
    with np.load(npz_path) as raw:
        position = np.asarray(raw["position"], dtype=float)
        time_s = np.asarray(raw["time_s"], dtype=float)
        target_ppm = np.asarray(raw["target_ppm"], dtype=float)
        valid = np.asarray(raw["label_valid"], dtype=bool)
    source_position = np.asarray(metadata["source_positions"][0], dtype=float)
    distance = np.linalg.norm(position - source_position[None, :], axis=1)
    gas_index = GAS_ORDER.index(gas)
    source_ppm = float(metadata["source_concentrations_ppm"][0][gas_index])
    return {
        "distance": distance,
        "time_s": time_s,
        "target_ppm": target_ppm,
        "valid": valid,
        "source_ppm": source_ppm,
        "session_id": metadata["session_id"],
        "split": metadata["split"],
    }


def distance_ppm_figure(examples: dict[str, dict], gas: str):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5), constrained_layout=True)
    scatter = None
    for ax, path in zip(axes.flat, PATHS):
        item = examples[path]
        progress = item["time_s"] / max(float(item["time_s"][-1]), 1e-6)
        ax.plot(item["distance"], item["target_ppm"], color="0.75", linewidth=0.7, zorder=1)
        scatter = ax.scatter(
            item["distance"], item["target_ppm"], c=progress, cmap="viridis",
            s=9, alpha=0.9, linewidths=0, zorder=2,
        )
        invalid = ~item["valid"]
        ax.scatter(
            item["distance"][invalid], item["target_ppm"][invalid],
            color="0.35", marker="x", s=12,
            linewidths=0.5, alpha=0.55, zorder=3,
        )
        ax.axhline(0.1, color="#D55E00", linestyle="--", linewidth=1, label="0.1 ppm collection threshold")
        ax.axhline(1.0, color="#0072B2", linestyle=":", linewidth=1.2, label="1 ppm calibration floor")
        ax.set_yscale("symlog", linthresh=0.01, linscale=0.8)
        ax.set_xlabel("E-nose distance from primary source (m)")
        ax.set_ylabel(f"Measured {gas.replace('_', ' ')} (ppm)")
        ax.set_title(
            f"{PATH_TITLES[path]}\n{item['session_id']} | source setting {item['source_ppm']:.2f} ppm",
            fontsize=10,
        )
        ax.grid(True, which="both", alpha=0.22)
    axes.flat[0].legend(loc="upper right", fontsize=8, frameon=False)
    if scatter is not None:
        fig.colorbar(scatter, ax=axes, label="Episode progress (0=start, 1=end)", shrink=0.8)
    fig.suptitle("Phase B: distance from source versus target-gas concentration", fontsize=14)
    return fig


def time_figure(examples: dict[str, dict], gas: str):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5), constrained_layout=True)
    for ax, path in zip(axes.flat, PATHS):
        item = examples[path]
        ppm_ax = ax.twinx()
        distance_line = ax.plot(item["time_s"], item["distance"], color="#0072B2", label="Distance")[0]
        ppm_line = ppm_ax.plot(item["time_s"], item["target_ppm"], color="#D55E00", label="Target ppm")[0]
        ppm_ax.axhline(0.1, color="#D55E00", linestyle="--", linewidth=0.8, alpha=0.7)
        ppm_ax.axhline(1.0, color="#0072B2", linestyle=":", linewidth=0.8, alpha=0.7)
        ppm_ax.set_yscale("symlog", linthresh=0.01, linscale=0.8)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Distance (m)", color="#0072B2")
        ppm_ax.set_ylabel(f"Measured {gas.replace('_', ' ')} (ppm)", color="#D55E00")
        ax.set_title(PATH_TITLES[path], fontsize=10)
        ax.grid(True, alpha=0.22)
        ax.legend([distance_line, ppm_line], ["Distance", "Target ppm"], loc="upper right", frameon=False)
    fig.suptitle("Phase B examples over time", fontsize=14)
    return fig


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("datasets/odor_encoder/virtual_single_v6"))
    parser.add_argument("--gas", choices=GAS_ORDER, default="ethyl_acetate")
    parser.add_argument("--source-ppm", type=float, default=10.0)
    parser.add_argument("--out-dir", type=Path, default=Path("analysis/odor_encoder"))
    args = parser.parse_args()
    if args.source_ppm <= 0.0:
        parser.error("--source-ppm must be positive")

    chosen = select_examples(args.root, args.gas, args.source_ppm)
    examples = {path: load_example(*chosen[path], args.gas) for path in PATHS}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"phase_b_distance_ppm_{args.gas}"
    pdf_path = args.out_dir / f"{stem}.pdf"
    png_path = args.out_dir / f"{stem}.png"
    distance_figure = distance_ppm_figure(examples, args.gas)
    time_plot = time_figure(examples, args.gas)
    with PdfPages(pdf_path) as pdf:
        pdf.savefig(distance_figure)
        pdf.savefig(time_plot)
    distance_figure.savefig(png_path, dpi=180)
    plt.close(distance_figure)
    plt.close(time_plot)
    for path in PATHS:
        item = examples[path]
        print(
            f"{path}: {item['session_id']} ({item['split']}), "
            f"source={item['source_ppm']:.3f} ppm, "
            f"distance={item['distance'].min():.3f}-{item['distance'].max():.3f} m, "
            f"measured_max={item['target_ppm'].max():.3f} ppm"
        )
    print(f"wrote {pdf_path}")
    print(f"wrote {png_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

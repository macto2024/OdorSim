#!/usr/bin/env python3
"""Run zero-wind filter-position analysis for every ordered two-gas pair."""

from __future__ import annotations

import argparse
from itertools import combinations
import json
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

from odor_sim.config.frame_map import FrameMap
from odor_sim.envs.registry import resolve_scenario
from odor_sim.odor_encoder.dataset import GAS_ORDER
from odor_sim.odor_encoder.phase_b import _run_gaden
from tools.analyze_filter_position_ppm import _layout, _path, _pure, _wind_profile


def _key(cup1_gas: str, cup2_gas: str) -> str:
    return f"{cup1_gas}__{cup2_gas}"


def _plot(result: dict, output: Path, rate: float) -> None:
    cup1_color = "#D55E00"
    cup2_color = "#0072B2"
    with PdfPages(output) as pdf:
        for gas_a, gas_b in combinations(GAS_ORDER, 2):
            fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))
            for row_index, (cup1_gas, cup2_gas) in enumerate(
                ((gas_a, gas_b), (gas_b, gas_a))
            ):
                trials = result["ordered_pairs"][_key(cup1_gas, cup2_gas)]
                filter_axis, trajectory_axis = axes[row_index]

                filter_cup1 = np.asarray([trial["filter_cup1_ppm"] for trial in trials])
                filter_cup2 = np.asarray([trial["filter_cup2_ppm"] for trial in trials])
                filter_time = np.arange(filter_cup1.shape[1]) / rate
                for gas, values, color in (
                    (cup1_gas, filter_cup1, cup1_color),
                    (cup2_gas, filter_cup2, cup2_color),
                ):
                    low, high = np.percentile(values, [10, 90], axis=0)
                    filter_axis.fill_between(filter_time, low, high, color=color, alpha=0.16)
                    filter_axis.plot(
                        filter_time, np.median(values, axis=0),
                        color=color, linewidth=1.6, label=gas.replace("_", " "),
                    )
                filter_axis.set_title(
                    f"At filter pose | cup 1: {cup1_gas.replace('_', ' ')}; "
                    f"cup 2: {cup2_gas.replace('_', ' ')}"
                )
                filter_axis.set_xlabel("seconds at filter pose")
                filter_axis.set_ylabel("ppm")
                filter_axis.grid(alpha=0.25)
                filter_axis.legend(fontsize=8)

                example = trials[0]
                full_cup1 = np.asarray(example["full_cup1_ppm"])
                full_cup2 = np.asarray(example["full_cup2_ppm"])
                full_time = np.arange(len(full_cup1)) / rate
                trajectory_axis.plot(
                    full_time, full_cup1, color=cup1_color, linewidth=1.3,
                    label=cup1_gas.replace("_", " "),
                )
                trajectory_axis.plot(
                    full_time, full_cup2, color=cup2_color, linewidth=1.3,
                    label=cup2_gas.replace("_", " "),
                )
                f0, f1 = example["bounds"]["filter"]
                trajectory_axis.axvspan(
                    f0 / rate, f1 / rate, color="black", alpha=0.10,
                    label="filter hold",
                )
                trajectory_axis.set_yscale("symlog", linthresh=0.01)
                trajectory_axis.set_title("Representative complete trajectory")
                trajectory_axis.set_xlabel("trajectory time (s)")
                trajectory_axis.set_ylabel("physical ppm at e-nose")
                trajectory_axis.grid(alpha=0.25)
                trajectory_axis.legend(fontsize=8)

            fig.suptitle(
                f"Zero wind | buoyancy tau = {result['buoyancy_tau_s']:g} s | "
                f"{gas_a.replace('_', ' ')} / {gas_b.replace('_', ' ')} forward and reverse"
            )
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
    print(f"wrote {output}")


def run(args: argparse.Namespace) -> dict:
    scenario_dir = resolve_scenario(args.scenario, args.scenario_config)
    frame_map = FrameMap.from_scenario(scenario_dir)
    ordered_pairs: dict[str, list[dict]] = {
        _key(first, second): []
        for first in GAS_ORDER for second in GAS_ORDER if first != second
    }

    for trial in range(args.trials):
        seed = args.seed + trial
        rng = np.random.default_rng(seed)
        cup1_robot, cup2_robot, filter_robot = _layout(rng)
        path_robot, bounds = _path(
            cup1_robot, cup2_robot, filter_robot,
            args.rate, args.sniff_s, args.travel_s, args.filter_s,
        )
        source_positions = {
            "cup1": frame_map.robosuite_to_gaden(cup1_robot),
            "cup2": frame_map.robosuite_to_gaden(cup2_robot),
        }
        path_gaden = np.stack([
            frame_map.robosuite_to_gaden(point) for point in path_robot
        ])
        duration_s = args.warmup_s + len(path_gaden) / args.rate
        wind_profile = _wind_profile(rng, duration_s, "zero")
        run_args = SimpleNamespace(
            scenario=args.scenario,
            scenario_config=args.scenario_config,
            runtime_config_dir=args.runtime_config_dir,
            server_log_dir=args.server_log_dir,
            wait_timeout=args.wait_timeout,
            warmup_s=args.warmup_s,
            rate=args.rate,
            seed=seed,
            buoyancy_tau_s=args.buoyancy_tau_s,
        )
        signals: dict[str, dict[str, np.ndarray]] = {"cup1": {}, "cup2": {}}
        for location, source_position in source_positions.items():
            for gas in GAS_ORDER:
                ppm = _run_gaden(
                    run_args,
                    f"all_pairs_tau_{args.buoyancy_tau_s:g}_{trial:03d}_{location}_{gas}",
                    [_pure(gas, args.source_ppm)],
                    source_position[None],
                    path_gaden,
                    wind_profile,
                )
                signals[location][gas] = ppm[:, GAS_ORDER.index(gas)]
                print(
                    f"[all-pairs] trial={trial + 1}/{args.trials} "
                    f"{location}={gas} tau={args.buoyancy_tau_s:g}s"
                )

        f0, f1 = bounds["filter"]
        for cup1_gas in GAS_ORDER:
            for cup2_gas in GAS_ORDER:
                if cup1_gas == cup2_gas:
                    continue
                cup1_values = signals["cup1"][cup1_gas]
                cup2_values = signals["cup2"][cup2_gas]
                ordered_pairs[_key(cup1_gas, cup2_gas)].append({
                    "trial": trial,
                    "seed": seed,
                    "cup1_robot": cup1_robot.tolist(),
                    "cup2_robot": cup2_robot.tolist(),
                    "filter_robot": filter_robot.tolist(),
                    "bounds": {name: list(value) for name, value in bounds.items()},
                    "filter_cup1_ppm": cup1_values[f0:f1].tolist(),
                    "filter_cup2_ppm": cup2_values[f0:f1].tolist(),
                    "full_cup1_ppm": cup1_values.tolist(),
                    "full_cup2_ppm": cup2_values.tolist(),
                })

    result = {
        "description": (
            "All 15 distinct gas pairs in both cup assignments under exactly zero wind. "
            "Each gas/location plume was simulated independently and ordered pairs were "
            "composed by exact superposition because GADEN gas simulations do not interact."
        ),
        "gas_order": list(GAS_ORDER),
        "ordered_pair_count": len(ordered_pairs),
        "trials_per_ordered_pair": args.trials,
        "source_setting_ppm": args.source_ppm,
        "rate_hz": args.rate,
        "warmup_s": args.warmup_s,
        "sniff_s": args.sniff_s,
        "travel_s": args.travel_s,
        "filter_s": args.filter_s,
        "wind_regime": "zero",
        "wind_speed_m_s": 0.0,
        "buoyancy_tau_s": args.buoyancy_tau_s,
        "ordered_pairs": ordered_pairs,
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{args.artifact_stem}.json"
    pdf_path = output_dir / f"{args.artifact_stem}.pdf"
    json_path.write_text(json.dumps(result, indent=2) + "\n")
    _plot(result, pdf_path, args.rate)
    print(f"wrote {json_path}")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--buoyancy-tau-s", type=float, required=True)
    parser.add_argument("--source-ppm", type=float, default=100.0)
    parser.add_argument("--trials", type=int, default=6)
    parser.add_argument("--seed", type=int, default=4280)
    parser.add_argument("--rate", type=float, default=20.0)
    parser.add_argument("--warmup-s", type=float, default=15.0)
    parser.add_argument("--sniff-s", type=float, default=10.0)
    parser.add_argument("--travel-s", type=float, default=2.0)
    parser.add_argument("--filter-s", type=float, default=15.0)
    parser.add_argument("--scenario", default="10x6_uniform")
    parser.add_argument("--scenario-config", default="config1")
    parser.add_argument("--wait-timeout", type=float, default=60.0)
    parser.add_argument("--runtime-config-dir", default=".runtime/filter_position_all_pairs")
    parser.add_argument("--server-log-dir", default=".roslog/filter_position_all_pairs")
    parser.add_argument("--output-dir", default="analysis/odor_encoder/filter_position_test")
    parser.add_argument("--artifact-stem", required=True)
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())

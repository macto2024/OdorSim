#!/usr/bin/env python3
"""Measure physical plume concentration at the OVLA two-cup filter pose."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

from odor_sim.config.frame_map import FrameMap
from odor_sim.envs.registry import resolve_scenario
from odor_sim.odor_encoder.dataset import GAS_ORDER, MixtureSpec
from odor_sim.odor_encoder.phase_b import _run_gaden
from odor_sim.odor_encoder.wind import WindProfile, WIND_REGIMES, sample_wind_profile


# Merged ClassifyLiquidPlaceOVLA placement ranges, in robosuite coordinates.
LEFT_X = (0.12, 0.20)
LEFT_Y = (-0.16, -0.08)
RIGHT_X = (0.12, 0.20)
RIGHT_Y = (0.08, 0.16)
PLATE_X = (-0.06, 0.06)
PLATE_Y = (-0.06, 0.06)
CUP_BODY_Z = 0.87
SNIFF_Z = CUP_BODY_Z + 0.12
# Measured after reset with the merged Panda task (seed 42).
FILTER_EEF_Z = 1.0046438593615088


def _pure(gas: str, ppm: float) -> MixtureSpec:
    values = np.zeros(len(GAS_ORDER), np.float32)
    values[GAS_ORDER.index(gas)] = ppm
    return MixtureSpec(values, "filter_position_test")


def _repeat(point: np.ndarray, steps: int) -> np.ndarray:
    return np.repeat(np.asarray(point, np.float32)[None], steps, axis=0)


def _path(
    cup1: np.ndarray,
    cup2: np.ndarray,
    filter_position: np.ndarray,
    rate: float,
    sniff_s: float,
    travel_s: float,
    filter_s: float,
) -> tuple[np.ndarray, dict[str, tuple[int, int]]]:
    n_sniff = int(round(sniff_s * rate))
    n_travel = max(2, int(round(travel_s * rate)))
    n_filter = int(round(filter_s * rate))
    sniff1 = cup1.copy()
    sniff1[2] = SNIFF_Z
    sniff2 = cup2.copy()
    sniff2[2] = SNIFF_Z
    segments = [
        _repeat(sniff1, n_sniff),
        np.linspace(sniff1, filter_position, n_travel, endpoint=False, dtype=np.float32),
        _repeat(filter_position, n_filter),
        np.linspace(filter_position, sniff2, n_travel, endpoint=False, dtype=np.float32),
        _repeat(sniff2, n_sniff),
    ]
    bounds: dict[str, tuple[int, int]] = {}
    cursor = 0
    for name, segment in zip(
        ("sniff1", "to_filter", "filter", "to_sniff2", "sniff2"), segments
    ):
        bounds[name] = (cursor, cursor + len(segment))
        cursor += len(segment)
    return np.concatenate(segments), bounds


def _layout(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cup1 = np.asarray([
        rng.uniform(*LEFT_X), rng.uniform(*LEFT_Y), CUP_BODY_Z,
    ])
    cup2 = np.asarray([
        rng.uniform(*RIGHT_X), rng.uniform(*RIGHT_Y), CUP_BODY_Z,
    ])
    filter_position = np.asarray([
        rng.uniform(*PLATE_X), rng.uniform(*PLATE_Y), FILTER_EEF_Z,
    ])
    return cup1, cup2, filter_position


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    return None if denominator <= 1e-9 else float(numerator / denominator)


def _summarize(rows: list[dict], regime: str) -> dict:
    selected = [row for row in rows if row["regime"] == regime]
    cup1 = np.concatenate([np.asarray(row["filter_cup1_ppm"]) for row in selected])
    cup2 = np.concatenate([np.asarray(row["filter_cup2_ppm"]) for row in selected])
    total = cup1 + cup2
    fraction = np.divide(cup1, total, out=np.zeros_like(cup1), where=total > 0)
    ratios = [
        row["filter_cup1_p90_over_sniff1_p90"]
        for row in selected
        if row["filter_cup1_p90_over_sniff1_p90"] is not None
    ]
    return {
        "trials": len(selected),
        "filter_cup1_ppm_median": float(np.median(cup1)),
        "filter_cup1_ppm_p90": float(np.percentile(cup1, 90)),
        "filter_cup1_ppm_p99": float(np.percentile(cup1, 99)),
        "filter_cup1_ppm_max": float(np.max(cup1)),
        "filter_cup2_ppm_median": float(np.median(cup2)),
        "filter_cup2_ppm_p90": float(np.percentile(cup2, 90)),
        "cup1_fraction_of_two_gases_median": float(np.median(fraction)),
        "filter_samples_cup1_ge_0_1ppm": float(np.mean(cup1 >= 0.1)),
        "filter_samples_cup1_ge_1ppm": float(np.mean(cup1 >= 1.0)),
        "median_trial_filter_p90_over_sniff1_p90": (
            float(np.median(ratios)) if ratios else None
        ),
    }


def _selected_regimes(args: argparse.Namespace) -> tuple[str, ...]:
    if args.wind_regime == "all":
        return tuple(WIND_REGIMES)
    return (args.wind_regime,)


def _wind_profile(
    rng: np.random.Generator, duration_s: float, regime: str,
) -> WindProfile:
    if regime != "zero":
        return sample_wind_profile(rng, duration_s, regime)
    samples = max(2, int(np.ceil(duration_s)) + 2)
    return WindProfile(
        regime="zero",
        vectors_m_s=np.zeros((samples, 3), np.float32),
        nominal_direction=np.asarray([1.0, 0.0, 0.0], np.float32),
    )


def _plot(rows: list[dict], summaries: dict, args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / f"{args.artifact_stem}.pdf"
    colors = {args.cup1_gas: "#D55E00", args.cup2_gas: "#0072B2"}
    regimes = _selected_regimes(args)

    with PdfPages(pdf_path) as pdf:
        fig, axes = plt.subplots(len(regimes), 1, figsize=(11, 4 + 2.5 * len(regimes)), sharex=True)
        axes = np.atleast_1d(axes)
        for axis, regime in zip(axes, regimes):
            regime_rows = [row for row in rows if row["regime"] == regime]
            time = np.arange(len(regime_rows[0]["filter_cup1_ppm"])) / args.rate
            for gas, key in (
                (args.cup1_gas, "filter_cup1_ppm"),
                (args.cup2_gas, "filter_cup2_ppm"),
            ):
                values = np.asarray([row[key] for row in regime_rows], float)
                median = np.median(values, axis=0)
                low, high = np.percentile(values, [10, 90], axis=0)
                axis.fill_between(time, low, high, color=colors[gas], alpha=0.16)
                axis.plot(time, median, color=colors[gas], lw=1.8, label=gas)
            summary = summaries[regime]
            axis.set_title(
                f"{regime.replace('_', ' ')} | cup-1 filter median/p90 "
                f"{summary['filter_cup1_ppm_median']:.3g}/"
                f"{summary['filter_cup1_ppm_p90']:.3g} ppm"
            )
            axis.set_ylabel("ppm at filter")
            axis.grid(alpha=0.25)
            axis.legend(loc="upper right")
        axes[-1].set_xlabel("seconds at plate/filter pose")
        fig.suptitle(
            "Physical gas at OVLA filter position\n"
            "lines: trial median; bands: 10th-90th percentiles"
        )
        fig.tight_layout()
        pdf.savefig(fig)
        fig.savefig(output_dir / f"{args.artifact_stem}.png", dpi=180)
        plt.close(fig)

        for regime in regimes:
            regime_rows = [row for row in rows if row["regime"] == regime]
            example = regime_rows[0]
            ppm = np.asarray(example["full_ppm"], float)
            time = np.arange(len(ppm)) / args.rate
            bounds = example["bounds"]
            fig, (map_axis, ppm_axis) = plt.subplots(1, 2, figsize=(13, 5.5))
            cup1 = np.asarray(example["cup1_robot"])
            cup2 = np.asarray(example["cup2_robot"])
            filt = np.asarray(example["filter_robot"])
            wind = np.asarray(example["mean_wind_robot"])
            map_axis.scatter(cup1[0], cup1[1], s=100, color=colors[args.cup1_gas], label=f"cup 1: {args.cup1_gas}")
            map_axis.scatter(cup2[0], cup2[1], s=100, color=colors[args.cup2_gas], label=f"cup 2: {args.cup2_gas}")
            map_axis.scatter(filt[0], filt[1], s=120, marker="X", color="black", label="filter over plate")
            speed = np.linalg.norm(wind[:2])
            if speed > 0:
                arrow = wind[:2] / speed * 0.12
                map_axis.arrow(filt[0], filt[1], arrow[0], arrow[1], width=0.002, color="#009E73")
            map_axis.set_aspect("equal")
            map_axis.set_xlabel("robosuite x (m)")
            map_axis.set_ylabel("robosuite y (m)")
            map_axis.set_title(f"Representative layout; mean wind {speed:.3f} m/s")
            map_axis.grid(alpha=0.25)
            map_axis.legend(fontsize=8)

            for gas, index in ((args.cup1_gas, GAS_ORDER.index(args.cup1_gas)), (args.cup2_gas, GAS_ORDER.index(args.cup2_gas))):
                ppm_axis.plot(time, ppm[:, index], color=colors[gas], label=gas)
            f0, f1 = bounds["filter"]
            ppm_axis.axvspan(f0 / args.rate, f1 / args.rate, color="black", alpha=0.10, label="filter hold")
            ppm_axis.set_yscale("symlog", linthresh=0.01)
            ppm_axis.set_xlabel("trajectory time (s)")
            ppm_axis.set_ylabel("physical ppm at e-nose")
            ppm_axis.set_title(regime.replace("_", " "))
            ppm_axis.grid(alpha=0.25)
            ppm_axis.legend()
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
    print(f"wrote {pdf_path}")


def run(args: argparse.Namespace) -> dict:
    scenario_dir = resolve_scenario(args.scenario, args.scenario_config)
    frame_map = FrameMap.from_scenario(scenario_dir)
    rows: list[dict] = []
    regimes = _selected_regimes(args)
    for regime_index, regime in enumerate(regimes):
        for trial in range(args.trials):
            seed = args.seed + regime_index * 10000 + trial
            rng = np.random.default_rng(seed)
            cup1_robot, cup2_robot, filter_robot = _layout(rng)
            path_robot, bounds = _path(
                cup1_robot, cup2_robot, filter_robot,
                args.rate, args.sniff_s, args.travel_s, args.filter_s,
            )
            sources_gaden = np.stack([
                frame_map.robosuite_to_gaden(cup1_robot),
                frame_map.robosuite_to_gaden(cup2_robot),
            ])
            path_gaden = np.stack([
                frame_map.robosuite_to_gaden(point) for point in path_robot
            ])
            wind_profile = _wind_profile(
                rng, args.warmup_s + len(path_gaden) / args.rate, regime,
            )
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
            ppm = _run_gaden(
                run_args,
                f"filter_{regime}_{trial:03d}",
                [_pure(args.cup1_gas, args.source_ppm), _pure(args.cup2_gas, args.source_ppm)],
                sources_gaden,
                path_gaden,
                wind_profile,
            )
            gas1 = GAS_ORDER.index(args.cup1_gas)
            gas2 = GAS_ORDER.index(args.cup2_gas)
            s0, s1 = bounds["sniff1"]
            f0, f1 = bounds["filter"]
            sniff1_p90 = float(np.percentile(ppm[s0:s1, gas1], 90))
            filter1_p90 = float(np.percentile(ppm[f0:f1, gas1], 90))
            mean_wind_robot = frame_map.rotation.T @ wind_profile.mean_vector_m_s
            row = {
                "regime": regime,
                "trial": trial,
                "seed": seed,
                "cup1_robot": cup1_robot.tolist(),
                "cup2_robot": cup2_robot.tolist(),
                "filter_robot": filter_robot.tolist(),
                "cup1_filter_distance_m": float(np.linalg.norm(filter_robot - cup1_robot)),
                "cup2_filter_distance_m": float(np.linalg.norm(filter_robot - cup2_robot)),
                "mean_wind_robot": mean_wind_robot.tolist(),
                "mean_wind_speed_m_s": float(np.linalg.norm(wind_profile.mean_vector_m_s)),
                "bounds": {name: list(value) for name, value in bounds.items()},
                "sniff1_cup1_ppm_p90": sniff1_p90,
                "filter_cup1_ppm_p90": filter1_p90,
                "filter_cup1_p90_over_sniff1_p90": _safe_ratio(filter1_p90, sniff1_p90),
                "filter_cup1_ppm": ppm[f0:f1, gas1].tolist(),
                "filter_cup2_ppm": ppm[f0:f1, gas2].tolist(),
                "full_ppm": ppm.tolist(),
            }
            rows.append(row)
            print(
                f"[{regime}] {trial + 1}/{args.trials} "
                f"distance1={row['cup1_filter_distance_m']:.3f}m "
                f"cup1 filter p90={filter1_p90:.3g}ppm"
            )

    summaries = {regime: _summarize(rows, regime) for regime in regimes}
    result = {
        "description": "Two strength-1 cup sources remain physically active during filtering.",
        "important": "The current discrete FILTER path feeds clean air to MOX even when physical GADEN ppm is nonzero.",
        "gas_order": list(GAS_ORDER),
        "cup1_gas": args.cup1_gas,
        "cup2_gas": args.cup2_gas,
        "source_setting_ppm": args.source_ppm,
        "rate_hz": args.rate,
        "warmup_s": args.warmup_s,
        "sniff_s": args.sniff_s,
        "travel_s": args.travel_s,
        "filter_s": args.filter_s,
        "buoyancy_tau_s": args.buoyancy_tau_s,
        "wind_regime": args.wind_regime,
        "summaries": summaries,
        "trials": rows,
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{args.artifact_stem}.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    _plot(rows, summaries, args)
    print(json.dumps(summaries, indent=2))
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cup1-gas", choices=GAS_ORDER, default="ethanol")
    parser.add_argument("--cup2-gas", choices=GAS_ORDER, default="ethyl_acetate")
    parser.add_argument("--source-ppm", type=float, default=100.0, help="100 maps to strength 1")
    parser.add_argument("--trials", type=int, default=6)
    parser.add_argument("--seed", type=int, default=4280)
    parser.add_argument("--rate", type=float, default=20.0)
    parser.add_argument("--warmup-s", type=float, default=15.0)
    parser.add_argument("--buoyancy-tau-s", type=float, default=1.0)
    parser.add_argument(
        "--wind-regime", choices=("all", "zero", *WIND_REGIMES), default="all",
        help="all runs the standard three regimes; zero uses an exactly zero wind field",
    )
    parser.add_argument("--sniff-s", type=float, default=10.0)
    parser.add_argument("--travel-s", type=float, default=2.0)
    parser.add_argument("--filter-s", type=float, default=15.0)
    parser.add_argument("--scenario", default="10x6_uniform")
    parser.add_argument("--scenario-config", default="config1")
    parser.add_argument("--wait-timeout", type=float, default=60.0)
    parser.add_argument("--runtime-config-dir", default=".runtime/filter_position_test")
    parser.add_argument("--server-log-dir", default=".roslog/filter_position_test")
    parser.add_argument("--output-dir", default="analysis/odor_encoder/filter_position_test")
    parser.add_argument("--artifact-stem", default="filter_position_concentration")
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())

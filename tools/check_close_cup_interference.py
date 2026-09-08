#!/usr/bin/env python3
"""Measure a neighboring cup's plume at an e-nose fixed above a target cup."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from odor_sim.odor_encoder.dataset import GAS_ORDER, MixtureSpec
from odor_sim.odor_encoder.phase_b import _run_gaden


ORIENTATIONS = {
    "upwind": np.asarray([-1.0, 0.0, 0.0]),
    "crosswind": np.asarray([0.0, 1.0, 0.0]),
    "downwind": np.asarray([1.0, 0.0, 0.0]),
}


def _pure(gas_index: int, ppm: float) -> MixtureSpec:
    concentration = np.zeros(len(GAS_ORDER), np.float32)
    concentration[gas_index] = ppm
    return MixtureSpec(concentration, "controlled_close_cup")


def run(args: argparse.Namespace) -> dict:
    target_position = np.asarray([2.0, 3.0, 0.5])
    steps = int(round(args.duration_s * args.rate))
    tail_steps = max(1, int(round(args.summary_s * args.rate)))
    rows = []

    for orientation_name in args.orientations:
        direction = ORIENTATIONS[orientation_name]
        noise_position = target_position + args.cup_distance * direction
        sensor_positions = np.stack((
            target_position + [0.0, 0.0, args.sensor_height],
            noise_position + [0.0, 0.0, args.sensor_height],
        ))
        path = np.repeat(sensor_positions[None], steps, axis=0)
        target_index = GAS_ORDER.index(args.target_gas)
        for noise_gas in args.noise_gases:
            noise_index = GAS_ORDER.index(noise_gas)
            if noise_index == target_index:
                continue
            ppm = _run_gaden(
                args,
                f"closecup_{orientation_name}_{args.target_gas}_{noise_gas}",
                [_pure(target_index, args.source_ppm), _pure(noise_index, args.source_ppm)],
                np.stack((target_position, noise_position)),
                path,
            )
            target_enose_noise = ppm[-tail_steps:, 0, noise_index]
            target_enose_target = ppm[-tail_steps:, 0, target_index]
            noise_enose_noise = ppm[-tail_steps:, 1, noise_index]
            noise_enose_target = ppm[-tail_steps:, 1, target_index]
            total = target_enose_noise + target_enose_target
            noise_fraction = np.divide(
                target_enose_noise, total,
                out=np.zeros_like(target_enose_noise), where=total > 0,
            )
            row = {
                "orientation": orientation_name,
                "noise_gas": noise_gas,
                "target_gas": GAS_ORDER[target_index],
                "target_enose_noise_ppm_median": float(np.median(target_enose_noise)),
                "target_enose_noise_ppm_p90": float(np.percentile(target_enose_noise, 90)),
                "target_enose_target_ppm_median": float(np.median(target_enose_target)),
                "target_enose_noise_fraction_median": float(np.median(noise_fraction)),
                "noise_enose_noise_ppm_median": float(np.median(noise_enose_noise)),
                "noise_enose_noise_ppm_p90": float(np.percentile(noise_enose_noise, 90)),
                "noise_enose_target_ppm_median": float(np.median(noise_enose_target)),
            }
            rows.append(row)
            print(
                f"{orientation_name:9s} {noise_gas:13s} "
                f"noise@target={row['target_enose_noise_ppm_median']:.4g} ppm "
                f"noise@noise={row['noise_enose_noise_ppm_median']:.4g} ppm"
            )

    result = {
        "wind_m_s": [0.3, 0.0, 0.0],
        "target_gas": args.target_gas,
        "noise_gases": [gas for gas in args.noise_gases if gas != args.target_gas],
        "source_setting_ppm": args.source_ppm,
        "cup_distance_m": args.cup_distance,
        "sensor_offset_above_each_source_m": [0.0, 0.0, args.sensor_height],
        "duration_s": args.duration_s,
        "warmup_s": args.warmup_s,
        "summary_s": args.summary_s,
        "rows": rows,
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote {destination}")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-ppm", type=float, default=10.0)
    parser.add_argument("--target-gas", choices=GAS_ORDER, default="acetone")
    parser.add_argument("--noise-gases", nargs="+", choices=GAS_ORDER, default=list(GAS_ORDER))
    parser.add_argument("--cup-distance", type=float, default=0.20)
    parser.add_argument("--sensor-height", type=float, default=0.03)
    parser.add_argument(
        "--orientations", nargs="+", choices=tuple(ORIENTATIONS),
        default=list(ORIENTATIONS),
    )
    parser.add_argument("--duration-s", type=float, default=20.0)
    parser.add_argument("--summary-s", type=float, default=5.0)
    parser.add_argument("--rate", type=float, default=20.0)
    parser.add_argument("--warmup-s", type=float, default=8.0)
    parser.add_argument("--scenario", default="10x6_uniform")
    parser.add_argument("--scenario-config", default="config1")
    parser.add_argument("--server-log-dir", default=".roslog/close_cup_interference")
    parser.add_argument("--wait-timeout", type=float, default=60.0)
    parser.add_argument(
        "--output",
        default="analysis/odor_encoder/close_cup_interference.json",
    )
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())

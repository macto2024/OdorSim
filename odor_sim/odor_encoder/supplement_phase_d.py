"""Collect only missing Phase-D pair/offset supervision reported by validation.

The utility preserves physical gas-label masking. It adds unique Phase-D episodes
to deficient validation/test pair groups until each second-sniff offset reaches
the requested supervised count.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess
import sys

import numpy as np

from odor_sim.odor_encoder.dataset import GAS_ORDER
from odor_sim.odor_encoder.validate_collection import SECOND_OFFSETS, _load_root


INDEX_PATTERN = re.compile(r"physical_d_(\d+)_")


def _phase_d_report(document: dict) -> dict:
    for report in document.get("datasets", []):
        if str(report.get("label", "")).upper() == "D":
            return report
    raise ValueError("validation report does not contain a Phase-D dataset")


def _next_index(root: Path) -> int:
    maximum = -1
    for path in root.glob("*/*.json"):
        match = INDEX_PATTERN.search(path.name)
        if match:
            maximum = max(maximum, int(match.group(1)))
    return maximum + 1


def _base_seed(metadata: dict) -> int:
    match = INDEX_PATTERN.search(str(metadata["episode_id"]))
    if not match:
        raise ValueError("cannot recover episode index from existing Phase-D metadata")
    index = int(match.group(1))
    return int(metadata["seed"]) - 30_000_000 - index * 1009


def _reference_metadata(root: Path) -> dict:
    try:
        path = next(root.glob("*/*.json"))
    except StopIteration as exc:
        raise ValueError(f"no Phase-D sidecars found under {root}") from exc
    metadata = json.loads(path.read_text())
    if metadata.get("pipeline") != "physical_two_cup_v1":
        raise ValueError(f"{path} is not a physical two-cup dataset")
    return metadata


def _new_supervised_counts(paths: list[Path], threshold_ppm: float) -> dict[str, int]:
    counts = {offset: 0 for offset in SECOND_OFFSETS}
    for path in paths:
        with np.load(path) as data:
            eligible = np.asarray(data["endpoint_eligible"], bool)
            second = np.asarray(data["sniff_role"]) == "second"
            buckets = np.asarray(data["endpoint_bucket"])
            supervised = np.asarray(data["gas_loss_mask"], bool)
            meaningful = np.asarray(data["local_total_ppm"], float) >= threshold_ppm
            valid = eligible & second & supervised & meaningful
            for offset in SECOND_OFFSETS:
                counts[offset] += int(np.sum(valid & (buckets == f"second_sniff_{offset}")))
    return counts


def _collector_command(
    args: argparse.Namespace,
    metadata: dict,
    root: Path,
    split: str,
    gas_a: str,
    gas_b: str,
    start_index: int,
    count: int,
) -> list[str]:
    command = [
        sys.executable, "-m", "odor_sim.odor_encoder.physical_pipeline", "D",
        "--episodes", str(count),
        "--start-index", str(start_index),
        "--out-dir", str(root),
        "--split", split,
        "--gas-a", gas_a,
        "--gas-b", gas_b,
        "--seed", str(_base_seed(metadata)),
        "--rate", str(metadata["rate_hz"]),
        "--emission-concentration-exponent", str(metadata["emission_concentration_exponent"]),
        "--emission-scale", str(metadata["emission_scale"]),
        "--filaments-per-second", str(metadata["filaments_per_second"]),
        "--wind-regime", str(metadata["wind_regime"]),
        "--warmup-s", str(metadata["warmup_s"]),
        "--buoyancy-tau-s", str(metadata["buoyancy_tau_s"]),
        "--source-z", str(metadata["cup_a_position_gaden_m"][2]),
        "--filter-z", str(metadata["filter_position_gaden_m"][2]),
        "--presence-threshold-ppm", str(metadata["presence_threshold_ppm"]),
        "--gas-class-min-ppm", str(metadata["gas_class_min_ppm"]),
        "--dominant-threshold", str(metadata["dominant_threshold"]),
        "--slope-window-s", str(metadata["slope_window_s"]),
        "--rising-rate-per-s", str(metadata["rising_rate_per_s"]),
        "--falling-rate-per-s", str(metadata["falling_rate_per_s"]),
        "--recovery-residual-threshold", str(metadata["recovery_residual_threshold"]),
        "--ros-domain-id", str(args.ros_domain_id),
        "--max-episode-attempts", str(args.max_episode_attempts),
        "--log-every", str(max(1, min(count, args.batch_size))),
    ]
    if metadata.get("original_terminal_velocity_enabled", False):
        command.append("--original-terminal-velocity")
    return command


def run(args: argparse.Namespace) -> int:
    document = json.loads(args.validation_report.read_text())
    report = _phase_d_report(document)
    root = Path(args.dataset_root or report["root"])
    reported_coverage = report.get("second_gas_pair_offset_coverage") or {}
    required = args.required or int(reported_coverage.get("minimum_supervised_required", 0))
    if required < 1:
        raise SystemExit("set --required or validate with a positive coverage requirement")

    metadata = _reference_metadata(root)
    # Rescan rather than trusting a possibly stale report, making an interrupted
    # supplementation command safe to rerun without recollecting repaired jobs.
    live_report = _load_root(
        "D", root, float(metadata["presence_threshold_ppm"]), required,
    )
    coverage = live_report["second_gas_pair_offset_coverage"]
    supervised_by_split = coverage["supervised_by_split"]
    next_index = _next_index(root)
    jobs: list[tuple[str, str, str, dict[str, int]]] = []
    for split in args.splits:
        counts = supervised_by_split.get(split, {})
        for gas_a in GAS_ORDER:
            for gas_b in GAS_ORDER:
                if gas_a == gas_b:
                    continue
                current = {
                    offset: int(counts.get(f"{gas_a}->{gas_b}|{offset}", 0))
                    for offset in SECOND_OFFSETS
                }
                if min(current.values()) < required:
                    jobs.append((split, gas_a, gas_b, current))

    if not jobs:
        print(f"[supplement-D] all requested cells already have >= {required} supervised endpoints")
        return 0
    print(
        f"[supplement-D] root={root} jobs={len(jobs)} required={required} "
        f"start_index={next_index}", flush=True,
    )
    for split, gas_a, gas_b, current in jobs:
        added = 0
        print(f"[supplement-D] {split}:{gas_a}->{gas_b} initial={current}", flush=True)
        while min(current.values()) < required:
            if added >= args.max_episodes_per_pair:
                missing = {key: value for key, value in current.items() if value < required}
                raise RuntimeError(
                    f"coverage still insufficient for {split}:{gas_a}->{gas_b} "
                    f"after {added} new episodes: {missing}"
                )
            count = min(args.batch_size, args.max_episodes_per_pair - added)
            start = next_index
            command = _collector_command(
                args, metadata, root, split, gas_a, gas_b, start, count,
            )
            print("[supplement-D] " + " ".join(command), flush=True)
            subprocess.run(command, check=True)
            paths = [
                root / split / f"physical_d_{index:06d}_{gas_a}_to_{gas_b}.npz"
                for index in range(start, start + count)
            ]
            missing_paths = [str(path) for path in paths if not path.is_file()]
            if missing_paths:
                raise RuntimeError(f"collector did not create expected files: {missing_paths}")
            increments = _new_supervised_counts(
                paths, float(metadata["presence_threshold_ppm"]),
            )
            for offset, increment in increments.items():
                current[offset] += increment
            next_index += count
            added += count
            print(
                f"[supplement-D] {split}:{gas_a}->{gas_b} "
                f"added={added} supervised={current}", flush=True,
            )
    print(
        "[supplement-D] coverage targets reached in added files; rerun "
        "validate_collection to verify the complete dataset",
        flush=True,
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-report", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--required", type=int)
    parser.add_argument(
        "--splits", nargs="+", choices=("train", "validation", "test"),
        default=("validation", "test"),
    )
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--max-episodes-per-pair", type=int, default=150)
    parser.add_argument("--ros-domain-id", type=int, default=84)
    parser.add_argument("--max-episode-attempts", type=int, default=3)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.batch_size < 1 or args.max_episodes_per_pair < args.batch_size:
        raise SystemExit("batch size must be positive and no larger than the per-pair limit")
    if not 0 <= args.ros_domain_id <= 232:
        raise SystemExit("--ros-domain-id must be between 0 and 232")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())

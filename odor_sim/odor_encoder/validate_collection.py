"""Validate deployment-consistent physical two-cup odor-encoder datasets."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np

from odor_sim.odor_encoder.dataset import GAS_ORDER

SECOND_OFFSETS = ("0.5s", "1s", "2s", "3s", "5s", "10s")

REQUIRED = {
    "time_s", "timestamp", "position", "enose_position_xyz", "ppm", "local_total_ppm", "dominant_gas_index",
    "dominant_fraction", "raw_mox_8ch", "processed_mox_8ch",
    "mox_internal_rs_r0", "phase", "exposure_state", "gas_mask_reason",
    "label_gas", "gas_loss_mask", "label_presence", "label_intensity",
    "label_recovery", "endpoint_eligible", "endpoint_bucket", "sniff_elapsed_s",
    "time_since_current_sniff_onset_s", "time_since_last_gas_transition_s",
    "time_since_departure_from_previous_cup_s", "cup_a_position_xyz",
    "cup_b_position_xyz", "gas_a", "gas_b", "nominal_source_condition_a",
    "nominal_source_condition_b", "local_ppm_from_cup_a", "local_ppm_from_cup_b", "episode_id",
    "trajectory_id", "split_id", "random_seed", "source_realization_id",
}


def _dominance_bin(value: float) -> str:
    if value >= 0.95:
        return ">=95%"
    if value >= 0.90:
        return "90-95%"
    if value >= 0.80:
        return "80-90%"
    if value >= 0.70:
        return "70-80%"
    return "<70%"


def _load_root(
    label: str,
    root: Path,
    meaningful_presence_threshold_ppm: float = 5.0,
    min_second_offset_endpoints_per_pair: int = 0,
) -> dict:
    failures = []
    identities = {split: {"episode": set(), "seed": set(), "source": set()} for split in ("train", "validation", "test")}
    pair_counts = Counter()
    pair_counts_by_split = {name: Counter() for name in ("train", "validation", "test")}
    gas_slots = defaultdict(Counter)
    gas_local = defaultdict(list)
    gas_nominal = defaultdict(list)
    mask_reasons = Counter()
    dominance_bins = Counter()
    dominance_by_phase = defaultdict(Counter)
    meaningful_presence_positive = meaningful_presence_negative = 0
    second_offset_eligible = {
        split: Counter() for split in ("train", "validation", "test")
    }
    second_offset_supervised = {
        split: Counter() for split in ("train", "validation", "test")
    }
    endpoint_count = mature_count = early_count = recovery_positive = 0
    second_trajectory_count = 0
    nonzero_background = 0
    inspected = 0

    for split in identities:
        for path in sorted((root / split).glob("*.npz")):
            sidecar_path = path.with_suffix(".json")
            if not sidecar_path.is_file():
                failures.append(f"{path}: missing JSON sidecar")
                continue
            metadata = json.loads(sidecar_path.read_text())
            inspected += 1
            if metadata.get("pipeline") != "physical_two_cup_v1":
                failures.append(f"{path}: not physical_two_cup_v1")
            for flag in (
                "both_sources_continuously_active", "physical_gaden_motion",
                "no_sensor_reset_during_sequence",
            ):
                if not metadata.get(flag, False):
                    failures.append(f"{path}: {flag} is false")
            if metadata.get("filter_forces_zero_ppm", True):
                failures.append(f"{path}: filter forcibly sets ppm to zero")

            episode = str(metadata.get("episode_id"))
            seed = int(metadata.get("seed", -1))
            source = str(metadata.get("source_realization_id"))
            identities[split]["episode"].add(episode)
            identities[split]["seed"].add(seed)
            identities[split]["source"].add(source)
            gas_a, gas_b = metadata.get("gas_a"), metadata.get("gas_b")
            if gas_a != gas_b:
                pair_counts[f"{gas_a}->{gas_b}"] += 1
                pair_counts_by_split[split][f"{gas_a}->{gas_b}"] += 1
            assignment = metadata.get("cup_assignment", "")
            gas_slots[gas_a]["left" if assignment.startswith("A_left") else "right"] += 1
            gas_slots[gas_b]["right" if assignment.startswith("A_left") else "left"] += 1
            nominal_a = np.asarray(metadata.get("nominal_source_condition_a", []), float)
            nominal_b = np.asarray(metadata.get("nominal_source_condition_b", []), float)
            if nominal_a.size:
                gas_nominal[gas_a].append(float(nominal_a.max()))
            if nominal_b.size:
                gas_nominal[gas_b].append(float(nominal_b.max()))

            with np.load(path) as data:
                missing = REQUIRED.difference(data.files)
                if missing:
                    failures.append(f"{path}: missing arrays {sorted(missing)}")
                    continue
                length = len(data["time_s"])
                if label in ("C", "D") and np.max(np.ptp(data["position"], axis=0)) < 1e-3:
                    failures.append(f"{path}: Phase {label} has no physical e-nose motion")
                for key in REQUIRED:
                    if key in ("gas_a", "gas_b"):
                        continue
                    if data[key].ndim and data[key].shape[0] not in (length, len(GAS_ORDER)):
                        failures.append(f"{path}: {key} has incompatible first dimension")
                ppm = np.asarray(data["ppm"], float)
                total = ppm.sum(1)
                if not np.allclose(total, data["local_total_ppm"], rtol=1e-5, atol=1e-6):
                    failures.append(f"{path}: local_total_ppm is not summed local ppm")
                threshold = float(metadata["presence_threshold_ppm"])
                expected_presence = total >= threshold
                if not np.array_equal(expected_presence, data["label_presence"].astype(bool)):
                    failures.append(f"{path}: presence label ignores endpoint local ppm")
                falling = np.asarray(data["exposure_state"]) == "FALLING_DOMINANT"
                mixed = np.asarray(data["exposure_state"]) == "MIXED_OR_AMBIGUOUS"
                gas_mask = np.asarray(data["gas_loss_mask"]).astype(bool)
                if np.any(gas_mask & falling):
                    failures.append(f"{path}: falling endpoint has gas supervision")
                if np.any(gas_mask & mixed):
                    failures.append(f"{path}: mixed endpoint has gas supervision")

                eligible = np.asarray(data["endpoint_eligible"]).astype(bool)
                endpoint_count += int(eligible.sum())
                meaningful_positive = eligible & (total >= meaningful_presence_threshold_ppm)
                meaningful_presence_positive += int(meaningful_positive.sum())
                meaningful_presence_negative += int((eligible & ~meaningful_positive).sum())
                phases = np.asarray(data["phase"])
                dominance_values = np.asarray(data["dominant_fraction"], float)
                for index in np.flatnonzero(eligible):
                    bucket = _dominance_bin(float(dominance_values[index]))
                    dominance_bins[bucket] += 1
                    dominance_by_phase[str(phases[index])][bucket] += 1
                elapsed = np.asarray(data["sniff_elapsed_s"], float)
                sniff_endpoints = eligible & np.isfinite(elapsed)
                mature_count += int(np.sum(sniff_endpoints & (elapsed > 10)))
                early_count += int(np.sum(sniff_endpoints & (elapsed < 1)))
                recovery_positive += int(np.sum(eligible & (data["label_recovery"] > 0.5)))
                mask_reasons.update(np.asarray(data["gas_mask_reason"])[eligible].tolist())
                role = np.asarray(data["sniff_role"])
                if np.any(eligible & (role == "second")):
                    second_trajectory_count += 1
                if label == "D":
                    buckets = np.asarray(data["endpoint_bucket"])
                    effective_gas_mask = gas_mask & (total >= meaningful_presence_threshold_ppm)
                    pair = f"{gas_a}->{gas_b}"
                    for offset in SECOND_OFFSETS:
                        selected = eligible & (role == "second") & (
                            buckets == f"second_sniff_{offset}"
                        )
                        key = f"{pair}|{offset}"
                        second_offset_eligible[split][key] += int(selected.sum())
                        second_offset_supervised[split][key] += int(
                            (selected & effective_gas_mask).sum()
                        )

                for gas_index, gas in enumerate(GAS_ORDER):
                    selected = eligible & gas_mask & (data["label_gas"] == gas_index)
                    gas_local[gas].extend(total[selected].tolist())
                other = ppm.copy()
                dominant = np.asarray(data["dominant_gas_index"], int)
                other[np.arange(length), dominant] = 0
                nonzero_background += int(np.sum(eligible & (other.sum(1) > 0.01)))

    if inspected and any(not identities[split]["episode"] for split in identities):
        empty = [split for split in identities if not identities[split]["episode"]]
        failures.append(f"empty required splits: {empty}")

    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        for identity in ("episode", "seed", "source"):
            overlap = identities[left][identity] & identities[right][identity]
            if overlap:
                failures.append(f"{identity} overlap between {left}/{right}: {len(overlap)}")

    if label == "D":
        missing_pairs = [f"{a}->{b}" for a in GAS_ORDER for b in GAS_ORDER if a != b and pair_counts[f"{a}->{b}"] == 0]
        if missing_pairs:
            failures.append(f"missing ordered gas pairs: {missing_pairs}")
        for split, counts_by_pair in pair_counts_by_split.items():
            missing_in_split = [
                f"{a}->{b}" for a in GAS_ORDER for b in GAS_ORDER
                if a != b and counts_by_pair[f"{a}->{b}"] == 0
            ]
            if missing_in_split:
                failures.append(f"{split} missing ordered gas pairs: {missing_in_split}")
        if second_trajectory_count == 0:
            failures.append("no second-gas trajectories")
        if min_second_offset_endpoints_per_pair > 0:
            insufficient = []
            for split in ("train", "validation", "test"):
                for gas_a in GAS_ORDER:
                    for gas_b in GAS_ORDER:
                        if gas_a == gas_b:
                            continue
                        for offset in SECOND_OFFSETS:
                            key = f"{gas_a}->{gas_b}|{offset}"
                            count = second_offset_supervised[split][key]
                            if count < min_second_offset_endpoints_per_pair:
                                insufficient.append(f"{split}:{key}={count}")
            if insufficient:
                failures.append(
                    "insufficient supervised second-gas pair/offset coverage "
                    f"(required>={min_second_offset_endpoints_per_pair}): {insufficient}"
                )
    if label in ("C", "D") and recovery_positive == 0:
        failures.append("no recovery-positive endpoints")
    if endpoint_count and mature_count / endpoint_count > 0.80:
        failures.append("more than 80% of endpoints are >10 s mature exposure")
    if early_count == 0:
        failures.append("no early-sniff endpoints")
    if nonzero_background == 0:
        failures.append("background gas from opposite cup is always zero")
    if meaningful_presence_negative == 0:
        failures.append(
            "no meaningful-presence-negative endpoints at "
            f"{meaningful_presence_threshold_ppm:g} ppm"
        )

    for gas, slots in gas_slots.items():
        count = sum(slots.values())
        if count >= 10:
            left_fraction = slots["left"] / count
            if not 0.2 <= left_fraction <= 0.8:
                failures.append(f"{gas} is tied to a cup side ({left_fraction:.3f} left)")
    local_medians = {
        gas: float(np.median(values)) for gas, values in gas_local.items() if values
    }
    local_median_ratio = None
    if len(local_medians) == len(GAS_ORDER):
        values = np.maximum(list(local_medians.values()), 1e-6)
        local_median_ratio = float(np.max(values) / np.min(values))
        if local_median_ratio > 3.0:
            failures.append(
                "gas classes have systematically different local-ppm medians "
                f"(max/min={local_median_ratio:.3f}, allowed<=3.000)"
            )
    nominal_medians = {
        gas: float(np.median(values)) for gas, values in gas_nominal.items() if values
    }
    nominal_median_ratio = None
    if len(nominal_medians) == len(GAS_ORDER):
        values = np.maximum(list(nominal_medians.values()), 1e-9)
        nominal_median_ratio = float(np.max(values) / np.min(values))
        if nominal_median_ratio > 3.0:
            failures.append(
                "gas classes have systematically different source-condition medians "
                f"(max/min={nominal_median_ratio:.3f}, allowed<=3.000)"
            )

    return {
        "label": label, "root": str(root), "valid": not failures,
        "failures": failures, "trajectory_count": inspected,
        "split_counts": {
            split: len(identities[split]["episode"]) for split in identities
        },
        "ordered_pair_counts": dict(sorted(pair_counts.items())),
        "ordered_pair_counts_by_split": {
            split: dict(sorted(counts.items()))
            for split, counts in pair_counts_by_split.items()
        },
        "endpoint_count": endpoint_count,
        "early_sniff_endpoint_count": early_count,
        "mature_over_10s_endpoint_fraction": mature_count / max(endpoint_count, 1),
        "recovery_positive_endpoint_count": recovery_positive,
        "second_gas_trajectory_count": second_trajectory_count,
        "nonzero_other_cup_endpoint_count": nonzero_background,
        "meaningful_presence": {
            "threshold_ppm": meaningful_presence_threshold_ppm,
            "positive": meaningful_presence_positive,
            "negative": meaningful_presence_negative,
            "negative_fraction": meaningful_presence_negative / max(endpoint_count, 1),
        },
        "dominance_histogram": dict(dominance_bins),
        "dominance_histogram_by_trajectory_phase": {
            phase: dict(counts) for phase, counts in sorted(dominance_by_phase.items())
        },
        "second_gas_pair_offset_coverage": {
            "minimum_supervised_required": min_second_offset_endpoints_per_pair,
            "eligible_by_split": {
                split: dict(sorted(counts.items()))
                for split, counts in second_offset_eligible.items()
            },
            "supervised_by_split": {
                split: dict(sorted(counts.items()))
                for split, counts in second_offset_supervised.items()
            },
        } if label == "D" else None,
        "gas_mask_coverage": {
            "supervised": mask_reasons.get("SUPERVISED", 0),
            "masked": endpoint_count - mask_reasons.get("SUPERVISED", 0),
            "by_reason": dict(sorted(mask_reasons.items())),
        },
        "local_ppm_median_by_gas": local_medians,
        "local_ppm_median_max_min_ratio": local_median_ratio,
        "nominal_source_condition_median_by_gas": nominal_medians,
        "nominal_source_condition_median_max_min_ratio": nominal_median_ratio,
        "cup_side_counts_by_gas": {
            gas: dict(counts) for gas, counts in sorted(gas_slots.items())
        },
    }


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", action="append", nargs=2, metavar=("PHASE", "ROOT"),
        help="repeat for A/B/C/D; defaults to physical pilot roots",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--meaningful-presence-threshold-ppm", type=float, default=5.0)
    parser.add_argument(
        "--min-second-offset-endpoints-per-pair", type=int, default=0,
        help="fail Phase D unless every ordered pair has this many supervised endpoints at every offset",
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    pairs = args.dataset or [
        ("A", "datasets/odor_encoder/physical_phase_a_pilot_v1"),
        ("B", "datasets/odor_encoder/physical_phase_b_pilot_v1"),
        ("C", "datasets/odor_encoder/physical_phase_c_pilot_v1"),
        ("D", "datasets/odor_encoder/physical_phase_d_pilot_v1"),
    ]
    reports = [
        _load_root(
            label.upper(), Path(root), args.meaningful_presence_threshold_ppm,
            args.min_second_offset_endpoints_per_pair,
        )
        for label, root in pairs
    ]

    aggregate = {split: {kind: set() for kind in ("episode", "seed", "source")}
                 for split in ("train", "validation", "test")}
    for _, root_value in pairs:
        root = Path(root_value)
        for split in aggregate:
            for sidecar in (root / split).glob("*.json"):
                metadata = json.loads(sidecar.read_text())
                aggregate[split]["episode"].add(str(metadata.get("episode_id")))
                aggregate[split]["seed"].add(int(metadata.get("seed", -1)))
                aggregate[split]["source"].add(str(metadata.get("source_realization_id")))
    cross_failures = []
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        for kind in ("episode", "seed", "source"):
            overlap = aggregate[left][kind] & aggregate[right][kind]
            if overlap:
                cross_failures.append(f"{kind} overlap across all phases in {left}/{right}: {len(overlap)}")
    document = {
        "valid": all(report["valid"] for report in reports) and not cross_failures,
        "cross_phase_split_failures": cross_failures,
        "datasets": reports,
    }
    rendered = json.dumps(document, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    return 0 if document["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

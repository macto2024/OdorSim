"""Report physical and sensor statistics for Phase B emission-calibration pilots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from odor_sim.odor_encoder.dataset import GAS_ORDER, SENSOR_ORDER

NOMINAL_BINS = (
    ("1_to_3", 1.0, 3.0, False),
    ("3_to_10", 3.0, 10.0, False),
    ("10_to_30", 10.0, 30.0, False),
    ("30_to_60", 30.0, 60.0, False),
    ("60_to_100", 60.0, 100.0, True),
)
LOCAL_CENTROID_BINS = (
    ("1_to_10", 1.0, 10.0, False),
    ("10_to_30", 10.0, 30.0, False),
    ("30_to_60", 30.0, 60.0, False),
    ("60_to_100", 60.0, 100.0, True),
)


def _quantiles(values: np.ndarray) -> dict[str, float | None]:
    if not len(values):
        return {
            name: None for name in ("min", "median", "p90", "p95", "p99", "max")
        }
    quantiles = np.quantile(values, (0.0, 0.5, 0.9, 0.95, 0.99, 1.0))
    return {
        name: float(value)
        for name, value in zip(
            ("min", "median", "p90", "p95", "p99", "max"), quantiles,
        )
    }


def _nominal_bin(value: float) -> str | None:
    for name, low, high, include_high in NOMINAL_BINS:
        if low <= value < high or (include_high and low <= value <= high):
            return name
    return None


def _correlation(x: np.ndarray, y: np.ndarray) -> float | None:
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def _ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    return ranks


def _sample_rows(
    chunks: list[np.ndarray], labels: list[np.ndarray], limit: int,
) -> tuple[np.ndarray, np.ndarray]:
    if not chunks:
        return np.empty((0, len(SENSOR_ORDER)), np.float32), np.empty(0, np.int64)
    x = np.concatenate(chunks)
    y = np.concatenate(labels)
    if len(y) > limit:
        indices = np.linspace(0, len(y) - 1, limit, dtype=np.int64)
        x, y = x[indices], y[indices]
    return x, y


def _centroid_metrics(
    train_x: np.ndarray, train_y: np.ndarray,
    validation_x: np.ndarray, validation_y: np.ndarray,
) -> dict:
    train_present = {gas for gas in range(len(GAS_ORDER)) if np.any(train_y == gas)}
    validation_present = {
        gas for gas in range(len(GAS_ORDER)) if np.any(validation_y == gas)
    }
    missing_train = [GAS_ORDER[gas] for gas in range(len(GAS_ORDER)) if gas not in train_present]
    missing_evaluation = [
        GAS_ORDER[gas] for gas in range(len(GAS_ORDER)) if gas not in validation_present
    ]
    coverage = {
        "all_six_classes_present": not missing_train and not missing_evaluation,
        "missing_train_classes": missing_train,
        "missing_evaluation_classes": missing_evaluation,
    }
    if missing_train or missing_evaluation:
        return {
            **coverage, "accuracy": None, "macro_accuracy": None,
            "per_gas_accuracy": {gas: None for gas in GAS_ORDER},
        }
    available = list(range(len(GAS_ORDER)))
    mean = train_x.mean(0)
    std = np.maximum(train_x.std(0), 1e-6)
    train_z = (train_x - mean) / std
    validation_z = (validation_x - mean) / std
    centroids = np.stack([train_z[train_y == gas].mean(0) for gas in available])
    distances = ((validation_z[:, None] - centroids[None]) ** 2).sum(2)
    prediction = np.asarray(available)[distances.argmin(1)]
    per_gas = {
        GAS_ORDER[gas]: float(np.mean(prediction[validation_y == gas] == gas))
        for gas in available
    }
    return {
        **coverage,
        "accuracy": float(np.mean(prediction == validation_y)),
        "macro_accuracy": float(np.mean(list(per_gas.values()))),
        "per_gas_accuracy": per_gas,
    }


def analyze_dataset(
    root: Path, *, sniff_radius_m: float, saturation_voltage: float,
    max_rows_per_split: int, centroid_evaluation_split: str,
) -> dict:
    sniff_ppm: list[np.ndarray] = []
    sniff_target_ppm: list[np.ndarray] = []
    sniff_dominance: list[np.ndarray] = []
    sniff_voltage: list[np.ndarray] = []
    nominal_bins = {
        name: {"total": [], "target": [], "trajectories": 0}
        for name, *_ in NOMINAL_BINS
    }
    trajectory_nominal: list[float] = []
    trajectory_local_median: list[float] = []
    near_source_by_split: dict[str, tuple[list[np.ndarray], list[np.ndarray]]] = {
        split: ([], []) for split in ("train", "validation", "test")
    }
    centroid_bins_by_split = {
        name: {
            split: ([], []) for split in ("train", "validation", "test")
        }
        for name, *_ in LOCAL_CENTROID_BINS
    }
    trajectories = 0
    nonclean_trajectories = 0
    criteria_evaluated = 0
    criteria_satisfied = 0
    sniff_samples = 0
    valid_samples = 0

    for split in ("train", "validation", "test"):
        voltage_chunks, label_chunks = near_source_by_split[split]
        for path in sorted((root / split).glob("*.npz")):
            trajectories += 1
            sidecar_path = path.with_suffix(".json")
            metadata = (
                json.loads(sidecar_path.read_text()) if sidecar_path.is_file() else {}
            )
            with np.load(path) as data:
                presence = np.asarray(data["label_presence"], np.float32)
                source_presence = presence.max(0)
                if source_presence.sum() <= 0:
                    continue
                nonclean_trajectories += 1
                if "trajectory_satisfies_training_criteria" in metadata:
                    criteria_evaluated += 1
                    criteria_satisfied += int(
                        metadata["trajectory_satisfies_training_criteria"]
                    )
                gas_index = int(source_presence.argmax())
                ppm = np.asarray(data["ppm"], np.float32)
                position = np.asarray(data["position"], np.float32)
                voltage = np.asarray(data["voltage"], np.float32)
                source_positions = metadata.get("source_positions", [])
                if not source_positions:
                    continue
                source = np.asarray(source_positions[0], np.float32)
                distance = np.linalg.norm(position - source, axis=1)
                near = np.isfinite(distance) & (distance <= sniff_radius_m)
                if np.any(near):
                    target = ppm[near, gas_index]
                    total = ppm[near].sum(1)
                    source_concentrations = metadata.get("source_concentrations_ppm", [])
                    nominal = (
                        float(source_concentrations[0][gas_index])
                        if source_concentrations else None
                    )
                    dominance = np.divide(
                        target, total, out=np.zeros_like(target), where=total > 0,
                    )
                    sniff_ppm.append(total)
                    sniff_target_ppm.append(target)
                    sniff_dominance.append(dominance)
                    sniff_voltage.append(voltage[near])
                    voltage_chunks.append(voltage[near])
                    label_chunks.append(np.full(int(near.sum()), gas_index, np.int64))
                    near_voltage = voltage[near]
                    for bin_name, low, high, include_high in LOCAL_CENTROID_BINS:
                        in_bin = (total >= low) & (
                            (total <= high) if include_high else (total < high)
                        )
                        if np.any(in_bin):
                            bin_voltage, bin_labels = centroid_bins_by_split[bin_name][split]
                            bin_voltage.append(near_voltage[in_bin])
                            bin_labels.append(
                                np.full(int(in_bin.sum()), gas_index, np.int64)
                            )
                    sniff_samples += int(near.sum())
                    if nominal is not None:
                        bin_name = _nominal_bin(nominal)
                        if bin_name is not None:
                            nominal_bins[bin_name]["total"].append(total)
                            nominal_bins[bin_name]["target"].append(target)
                            nominal_bins[bin_name]["trajectories"] += 1
                        trajectory_nominal.append(nominal)
                        trajectory_local_median.append(float(np.median(total)))

                valid_key = (
                    "label_class_valid"
                    if "label_class_valid" in data else "label_valid"
                )
                valid = np.asarray(data[valid_key], bool)
                if "endpoint_kind" in data:
                    valid &= np.asarray(data["endpoint_kind"]) == "dominant"
                if np.any(valid):
                    valid_samples += int(valid.sum())

    total_ppm = np.concatenate(sniff_ppm) if sniff_ppm else np.empty(0)
    target_ppm = np.concatenate(sniff_target_ppm) if sniff_target_ppm else np.empty(0)
    dominance = np.concatenate(sniff_dominance) if sniff_dominance else np.empty(0)
    voltage = (
        np.concatenate(sniff_voltage)
        if sniff_voltage else np.empty((0, len(SENSOR_ORDER)))
    )
    sampled = {
        split: _sample_rows(*near_source_by_split[split], max_rows_per_split)
        for split in near_source_by_split
    }
    centroid = _centroid_metrics(
        *sampled["train"], *sampled[centroid_evaluation_split],
    )
    centroid["evaluation_split"] = centroid_evaluation_split
    centroid_by_local_bin = {}
    for name, *_ in LOCAL_CENTROID_BINS:
        bin_samples = {
            split: _sample_rows(
                *centroid_bins_by_split[name][split], max_rows_per_split,
            )
            for split in ("train", "validation", "test")
        }
        metrics = _centroid_metrics(
            *bin_samples["train"], *bin_samples[centroid_evaluation_split],
        )
        metrics["evaluation_split"] = centroid_evaluation_split
        centroid_by_local_bin[name] = metrics
    nominal_bin_report = {}
    bin_medians = []
    missing_bins = []
    for name, *_ in NOMINAL_BINS:
        entry = nominal_bins[name]
        total = np.concatenate(entry["total"]) if entry["total"] else np.empty(0)
        target = np.concatenate(entry["target"]) if entry["target"] else np.empty(0)
        nominal_bin_report[name] = {
            "trajectory_count": entry["trajectories"],
            "sample_count": int(len(total)),
            "local_total_ppm": _quantiles(total),
            "local_target_ppm": _quantiles(target),
        }
        if len(total):
            bin_medians.append(float(np.median(total)))
        else:
            missing_bins.append(name)
    nominal_values = np.asarray(trajectory_nominal, np.float64)
    local_medians = np.asarray(trajectory_local_median, np.float64)
    monotonic = {
        "all_bins_present": not missing_bins,
        "missing_bins": missing_bins,
        "bin_medians_non_decreasing": (
            bool(np.all(np.diff(bin_medians) >= 0)) if len(bin_medians) >= 2 else None
        ),
        "trajectory_median_pearson": _correlation(nominal_values, local_medians),
        "trajectory_median_spearman": _correlation(
            _ranks(nominal_values), _ranks(local_medians),
        ),
    }
    return {
        "root": str(root),
        "trajectory_count": trajectories,
        "nonclean_trajectory_count": nonclean_trajectories,
        "training_criteria": {
            "evaluated_trajectory_count": criteria_evaluated,
            "satisfied_trajectory_count": criteria_satisfied,
            "satisfaction_fraction": (
                criteria_satisfied / criteria_evaluated if criteria_evaluated else None
            ),
        },
        "sniff_radius_m": sniff_radius_m,
        "sniff_sample_count": sniff_samples,
        "valid_dominant_sample_count": valid_samples,
        "local_total_ppm": {
            **_quantiles(total_ppm),
            "fraction_above_100": float(np.mean(total_ppm > 100)) if len(total_ppm) else None,
            "fraction_above_200": float(np.mean(total_ppm > 200)) if len(total_ppm) else None,
        },
        "local_target_ppm": _quantiles(target_ppm),
        "local_ppm_by_nominal_bin": nominal_bin_report,
        "nominal_to_local_monotonicity": monotonic,
        "target_dominance": _quantiles(dominance),
        "voltage": {
            "minimum_by_sensor": voltage.min(0).tolist() if len(voltage) else None,
            "maximum_by_sensor": voltage.max(0).tolist() if len(voltage) else None,
            "fraction_near_saturation_by_sensor": (
                np.mean(voltage >= saturation_voltage, axis=0).tolist()
                if len(voltage) else None
            ),
            "fraction_samples_with_any_saturated_channel": (
                float(np.mean(np.any(voltage >= saturation_voltage, axis=1)))
                if len(voltage) else None
            ),
            "saturation_threshold_v": saturation_voltage,
        },
        "nearest_centroid_near_source": centroid,
        "nearest_centroid_by_local_ppm": centroid_by_local_bin,
    }


def _parse_dataset(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("dataset must use LABEL=PATH")
    label, raw_path = value.split("=", 1)
    if not label or not raw_path:
        raise argparse.ArgumentTypeError("dataset must use LABEL=PATH")
    return label, Path(raw_path)


def run(args: argparse.Namespace) -> int:
    report = {
        label: analyze_dataset(
            path,
            sniff_radius_m=args.sniff_radius_m,
            saturation_voltage=args.saturation_voltage,
            max_rows_per_split=args.max_rows_per_split,
            centroid_evaluation_split=args.centroid_evaluation_split,
        )
        for label, path in args.dataset
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n")
    print(
        "label\tmedian_ppm\tp90\tp95\tp99\t>100\t>200\tdominance\t"
        "saturation\tcriteria_pass\tmonotonic\tcentroid_macro\tmissing_classes"
    )
    for label, result in report.items():
        ppm = result["local_total_ppm"]
        dominance = result["target_dominance"]["median"]
        saturation = result["voltage"]["fraction_samples_with_any_saturated_channel"]
        criteria = result["training_criteria"]["satisfaction_fraction"]
        monotonic = result["nominal_to_local_monotonicity"]["bin_medians_non_decreasing"]
        centroid = result["nearest_centroid_near_source"]["macro_accuracy"]
        missing = sorted(set(
            result["nearest_centroid_near_source"]["missing_train_classes"]
            + result["nearest_centroid_near_source"]["missing_evaluation_classes"]
        ))
        fields = (
            label, ppm["median"], ppm["p90"], ppm["p95"], ppm["p99"],
            ppm["fraction_above_100"], ppm["fraction_above_200"],
            dominance, saturation, criteria, monotonic, centroid,
            ",".join(missing) if missing else "none",
        )
        print("\t".join(
            value if isinstance(value, str) else (
                "n/a" if value is None else f"{value:.4g}"
            )
            for value in fields
        ))
    print(f"report: {destination.resolve()}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", action="append", required=True, type=_parse_dataset,
        metavar="LABEL=PATH",
    )
    parser.add_argument(
        "--output",
        default="analysis/odor_encoder/phase_b_emission_calibration.json",
    )
    parser.add_argument("--sniff-radius-m", type=float, default=0.08)
    parser.add_argument("--saturation-voltage", type=float, default=4.90)
    parser.add_argument("--max-rows-per-split", type=int, default=100_000)
    parser.add_argument(
        "--centroid-evaluation-split", choices=("validation", "test"),
        default="validation",
    )
    return parser


if __name__ == "__main__":
    raise SystemExit(run(build_parser().parse_args()))

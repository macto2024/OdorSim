"""Fit an inverse nominal-command to local-ppm calibration for Phase B."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def fit(root: Path, sniff_radius_m: float) -> dict:
    nominal_values = []
    local_values = []
    configurations = []
    for path in sorted(root.glob("*/*.npz")):
        metadata_path = path.with_suffix(".json")
        if not metadata_path.is_file():
            continue
        metadata = json.loads(metadata_path.read_text())
        configurations.append({
            "emission_concentration_exponent": metadata.get(
                "emission_concentration_exponent", 1.0
            ),
            "emission_scale": metadata.get("emission_scale"),
            "filaments_per_second": metadata.get("filaments_per_second"),
            "warmup_s": metadata.get("warmup_s", 8.0),
            "wind_regime": metadata.get("wind_regime"),
            "buoyancy_tau_s": metadata.get("buoyancy_tau_s"),
        })
        sources = metadata.get("source_concentrations_ppm", [])
        positions = metadata.get("source_positions", [])
        if not sources or not positions:
            continue
        nominal = float(max(sources[0]))
        if nominal <= 0:
            continue
        with np.load(path) as data:
            near = np.linalg.norm(
                np.asarray(data["position"]) - np.asarray(positions[0]), axis=1,
            ) <= sniff_radius_m
            if not np.any(near):
                continue
            local = float(np.median(np.asarray(data["ppm"])[near].sum(1)))
        if local > 0:
            nominal_values.append(nominal)
            local_values.append(local)
    if len(nominal_values) < 10:
        raise RuntimeError("at least 10 non-clean trajectories are required")
    unique_configurations = {json.dumps(value, sort_keys=True) for value in configurations}
    if len(unique_configurations) != 1:
        raise RuntimeError("dataset contains multiple emission configurations")
    x = np.log(np.asarray(nominal_values))
    y = np.log(np.asarray(local_values))
    slope, intercept = np.polyfit(x, y, 1)
    predicted = intercept + slope * x
    ss_res = float(np.sum((y - predicted) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return {
        "schema_version": 1,
        "model": "log_linear_nominal_to_local_median",
        "source_dataset": str(root),
        "sniff_radius_m": sniff_radius_m,
        "trajectory_count": len(x),
        "emission_configuration": json.loads(unique_configurations.pop()),
        "log_intercept": float(intercept),
        "log_slope": float(slope),
        "log_space_r_squared": 1.0 - ss_res / ss_tot if ss_tot > 0 else None,
        "nominal_support_ppm": [float(np.min(np.exp(x))), float(np.max(np.exp(x)))],
        "local_median_support_ppm": [float(np.min(np.exp(y))), float(np.max(np.exp(y)))],
        "inverse": "nominal = exp((log(desired_local_ppm) - log_intercept) / log_slope)",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--sniff-radius-m", type=float, default=0.08)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = fit(args.dataset, args.sniff_radius_m)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

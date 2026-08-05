#!/usr/bin/env python3
"""Compare three CH aggregation methods in the worksheet Ra/Rg domain."""

from __future__ import annotations

import argparse
import csv
import math
import tempfile
from collections import defaultdict
from pathlib import Path
from statistics import fmean, median

import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.backends.backend_pdf import PdfPages

from build_mox_registry import CONCS, GASES, extract

COLORS = {
    "methanol": "#0072B2",
    "ethanol": "#D55E00",
    "acetic_acid": "#009E73",
    "ethyl_acetate": "#CC79A7",
    "toluene": "#E69F00",
    "acetone": "#5B4B8A",
}


def fit_power_law(points: dict[float, float]) -> dict | None:
    """Fit Ra/Rg = A * concentration^B in natural-log space."""
    valid = sorted(
        (float(concentration), float(response))
        for concentration, response in points.items()
        if concentration > 0 and response > 0 and math.isfinite(response)
    )
    if len(valid) < 3:
        return None
    x = np.log(np.asarray([point[0] for point in valid], dtype=float))
    y = np.log(np.asarray([point[1] for point in valid], dtype=float))
    b, log_a = np.polyfit(x, y, 1)
    predicted = log_a + b * x
    if b <= 0 or not np.isfinite([b, log_a]).all():
        return None
    return {
        "a": float(np.exp(log_a)),
        "b": float(b),
        "rmse_log_to_aggregate_points": float(
            np.sqrt(np.mean((predicted - y) ** 2))
        ),
        "n_points": len(valid),
    }


def response_points(channel: dict, gas_name: str) -> dict[float, float]:
    """Return one physical channel's unfitted worksheet Ra/Rg points."""
    return {
        float(concentration): float(response)
        for concentration, response in channel["measurements_ra_rg"].get(
            gas_name, {}
        ).items()
        if float(response) > 0
    }


def pooled_rmse_log(
    a: float,
    b: float,
    channel_points: list[dict[float, float]],
) -> float:
    """RMSE of one curve against every valid physical CH measurement."""
    residuals = []
    for points in channel_points:
        for concentration, response in points.items():
            predicted = math.log(a) + b * math.log(concentration)
            residuals.append(predicted - math.log(response))
    return float(np.sqrt(np.mean(np.square(residuals)))) if residuals else math.nan


def build_method_comparison(replicates: dict[str, dict]) -> dict[str, dict]:
    """Build per-CH-fit/mean-first/median-first response-domain parameters."""
    grouped: dict[str, list[tuple[str, dict]]] = defaultdict(list)
    for profile_id, profile in replicates.items():
        grouped[profile["family"]].append((profile_id, profile))

    models = {}
    for family, profiles in sorted(grouped.items()):
        gases = {}
        for gas_name in GASES:
            channel_fits = []
            channel_points = []
            for profile_id, profile in profiles:
                points = response_points(profile, gas_name)
                curve = fit_power_law(points)
                if curve is None:
                    continue
                channel_points.append(points)
                channel_fits.append(
                    {
                        "profile_id": profile_id,
                        "a": curve["a"],
                        "b": curve["b"],
                        "rmse_log": curve["rmse_log_to_aggregate_points"],
                    }
                )
            if not channel_fits:
                continue

            mean_points = {}
            median_points = {}
            counts = {}
            for concentration in CONCS:
                responses = [
                    points[concentration]
                    for points in channel_points
                    if concentration in points
                ]
                if responses:
                    key = str(int(concentration))
                    mean_points[concentration] = fmean(responses)
                    median_points[concentration] = median(responses)
                    counts[key] = len(responses)

            parameter_mean = {
                "a": fmean(curve["a"] for curve in channel_fits),
                "b": fmean(curve["b"] for curve in channel_fits),
                "channel_fit_count": len(channel_fits),
                "channel_fits": channel_fits,
            }
            parameter_mean["pooled_rmse_log"] = pooled_rmse_log(
                parameter_mean["a"],
                parameter_mean["b"],
                channel_points,
            )

            mean_first = fit_power_law(mean_points)
            median_first = fit_power_law(median_points)
            for method in (mean_first, median_first):
                method["pooled_rmse_log"] = pooled_rmse_log(
                    method["a"],
                    method["b"],
                    channel_points,
                )

            mean_first["aggregate_points"] = {
                str(int(concentration)): float(value)
                for concentration, value in sorted(mean_points.items())
            }
            median_first["aggregate_points"] = {
                str(int(concentration)): float(value)
                for concentration, value in sorted(median_points.items())
            }
            gases[gas_name] = {
                "contributing_counts": counts,
                "parameter_mean_after_channel_fits": parameter_mean,
                "mean_ratio_first_then_fit": mean_first,
                "median_ratio_first_then_fit": median_first,
            }

        models[family] = {
            "family": family,
            "response_definition": "Ra/Rg = A * concentration_ppm^B",
            "physical_replicate_count": len(profiles),
            "gases": gases,
        }
    return models


def curve_values(method: dict, concentrations: np.ndarray) -> np.ndarray:
    return float(method["a"]) * concentrations ** float(method["b"])


def plot_model_comparison(model_name: str, model: dict, output_dir: Path):
    """Plot all odors for the three aggregation methods on identical axes."""
    figure, axes = plt.subplots(
        1,
        3,
        figsize=(20.0, 6.5),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    log_concentration = np.linspace(0.0, 2.0, 300)
    concentrations = 10.0**log_concentration
    method_specs = (
        (
            "Fit each CH, then mean(A, B)",
            "parameter_mean_after_channel_fits",
            None,
        ),
        (
            "Mean Ra/Rg, then fit",
            "mean_ratio_first_then_fit",
            "aggregate_points",
        ),
        (
            "Median Ra/Rg, then fit",
            "median_ratio_first_then_fit",
            "aggregate_points",
        ),
    )

    maximum = 1.0
    for gas in model["gases"].values():
        for _, method_key, point_key in method_specs:
            method = gas[method_key]
            maximum = max(maximum, float(np.max(curve_values(method, concentrations))))
            if point_key:
                maximum = max(
                    maximum,
                    max(float(value) for value in method[point_key].values()),
                )
    y_max = maximum * 1.08

    for axis, (title, method_key, point_key) in zip(axes, method_specs):
        for gas_name in GASES:
            gas = model["gases"].get(gas_name)
            if gas is None:
                continue
            method = gas[method_key]
            color = COLORS[gas_name]

            if method_key == "parameter_mean_after_channel_fits":
                for channel_fit in method["channel_fits"]:
                    axis.plot(
                        log_concentration,
                        curve_values(channel_fit, concentrations),
                        color=color,
                        linewidth=0.65,
                        alpha=0.13,
                    )

            axis.plot(
                log_concentration,
                curve_values(method, concentrations),
                color=color,
                linewidth=2.1,
                label=f"{gas_name.replace('_', ' ')}  (B={method['b']:.3f})",
            )
            if point_key:
                points = method[point_key]
                axis.scatter(
                    np.log10(np.asarray([float(value) for value in points])),
                    np.asarray([float(value) for value in points.values()]),
                    color=color,
                    edgecolor="white",
                    linewidth=0.7,
                    s=40,
                    zorder=3,
                )

        axis.set_xlim(0.0, 2.0)
        axis.set_ylim(0.0, y_max)
        axis.set_xticks(np.log10(np.asarray(CONCS)))
        axis.set_xticklabels(["0", "0.699", "1", "1.699", "2"])
        axis.set_xlabel("log10(concentration / ppm)")
        axis.set_title(title)
        axis.grid(True, color="#D9DDE2", linewidth=0.7)
        axis.legend(loc="upper left", fontsize=7.8, frameon=True)

    axes[0].set_ylabel("Ra/Rg")
    figure.suptitle(f"{model_name}: CH aggregation comparison", fontsize=15)
    output_path = output_dir / f"{model_name}.png"
    figure.savefig(output_path, dpi=170)
    return figure


def write_csv(path: Path, models: dict[str, dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "model",
                "gas",
                "channel_count",
                "channel_fit_mean_a",
                "channel_fit_mean_b",
                "channel_fit_mean_pooled_rmse_log",
                "mean_ratio_first_a",
                "mean_ratio_first_b",
                "mean_ratio_first_pooled_rmse_log",
                "median_ratio_first_a",
                "median_ratio_first_b",
                "median_ratio_first_pooled_rmse_log",
            ]
        )
        for model_name, model in models.items():
            for gas_name in GASES:
                gas = model["gases"][gas_name]
                parameter_mean = gas["parameter_mean_after_channel_fits"]
                mean_first = gas["mean_ratio_first_then_fit"]
                median_first = gas["median_ratio_first_then_fit"]
                writer.writerow(
                    [
                        model_name,
                        gas_name,
                        parameter_mean["channel_fit_count"],
                        parameter_mean["a"],
                        parameter_mean["b"],
                        parameter_mean["pooled_rmse_log"],
                        mean_first["a"],
                        mean_first["b"],
                        mean_first["pooled_rmse_log"],
                        median_first["a"],
                        median_first["b"],
                        median_first["pooled_rmse_log"],
                    ]
                )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("analysis/mox_comparison/ra_rg_method_comparison.yaml"),
    )
    parser.add_argument(
        "--comparison-csv",
        type=Path,
        default=Path("analysis/mox_comparison/ra_rg_method_comparison.csv"),
    )
    parser.add_argument(
        "--plot-dir",
        type=Path,
        default=Path("analysis/mox_curves_comparison"),
    )
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(prefix="odorsim_mox_compare_") as temp:
        replicates, exclusions = extract(args.source_dir, Path(temp))
    models = build_method_comparison(replicates)
    output = {
        "schema_version": 2,
        "response_definition": "Ra/Rg = A * concentration_ppm^B",
        "red_channel_policy": "same raw-sheet red-font exclusions as runtime registry",
        "methods": {
            "parameter_mean_after_channel_fits": (
                "Fit A,B for every physical CH, then arithmetic-mean A and B."
            ),
            "mean_ratio_first_then_fit": (
                "Arithmetic-mean Ra/Rg across CH at each ppm, then fit A,B."
            ),
            "median_ratio_first_then_fit": (
                "Median Ra/Rg across CH at each ppm, then fit A,B."
            ),
        },
        "source_exclusion_count": len(exclusions),
        "models": models,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        yaml.safe_dump(output, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    write_csv(args.comparison_csv, models)

    args.plot_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = args.plot_dir / "all_method_comparisons.pdf"
    with PdfPages(pdf_path) as pdf:
        for model_name in models:
            figure = plot_model_comparison(model_name, models[model_name], args.plot_dir)
            pdf.savefig(figure)
            plt.close(figure)

    print(f"Wrote three-method parameters for {len(models)} models to {args.output}")
    print(f"Wrote side-by-side values to {args.comparison_csv}")
    print(f"Wrote {len(models)} PNG plots and one PDF to {args.plot_dir}")


if __name__ == "__main__":
    main()

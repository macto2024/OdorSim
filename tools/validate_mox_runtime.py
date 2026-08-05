#!/usr/bin/env python3
"""Validate calibrated MOX steady-state and transient runtime behavior."""

from __future__ import annotations

import argparse
import csv
import math
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib import font_manager
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D

from odor_sim.sensors.mox_pid import MOX_MODELS, MOXSensor, get_mox_profile

CONCENTRATIONS = (1.0, 5.0, 10.0, 50.0, 100.0)
LOG_9 = math.log(9.0)
FIT_PERCENT_TOLERANCE = 30.0
TAU_RELATIVE_TOLERANCE = 0.01
SOURCE_FILENAMES = {
    "CROSS2024": "100-500交叉物质.xlsx",
    "PRECOAT2024": "104和303涂覆分子筛前.xlsx",
    "MULTI2023": "601和701和43#多物质.xls",
}
def _curve_rs_rair(curve: dict, concentration):
    concentration = np.asarray(concentration, dtype=float)
    coefficient = float(curve["coefficient_k"])
    exponent = float(curve["exponent_n"])
    return 1.0 / (1.0 + coefficient * concentration**exponent)


def _interpolated_crossing(
    previous_time: float,
    previous_progress: float,
    time: float,
    progress: float,
    level: float,
) -> float | None:
    if previous_progress < level <= progress:
        difference = progress - previous_progress
        if difference <= 0.0:
            return time
        fraction = (level - previous_progress) / difference
        return previous_time + fraction * (time - previous_time)
    return None


def _simulate_phase(
    sensor: MOXSensor,
    ppm: dict,
    target_rs_r0: float,
    configured_tau: float,
    phase: str,
    trace_stride: int,
) -> tuple[dict, list[dict]]:
    start_rs_r0 = sensor.rs_r0
    start_voltage = sensor.voltage
    amplitude = target_rs_r0 - start_rs_r0
    if amplitude == 0.0:
        raise ValueError(f"{phase} has zero response amplitude")

    dt = 1.0 / sensor.rate
    maximum_steps = int(math.ceil(max(10.0, 8.0 * configured_tau) * sensor.rate))
    previous_time = 0.0
    previous_progress = 0.0
    previous_rs_r0 = start_rs_r0
    previous_voltage = start_voltage
    t10 = None
    t90 = None
    monotonic_resistance = True
    monotonic_voltage = True
    trace = [
        {
            "phase": phase,
            "time_s": 0.0,
            "progress": 0.0,
            "rs_r0": start_rs_r0,
            "resistance_ohm": sensor.resistance,
            "voltage_v": start_voltage,
            "ideal_rs_r0": target_rs_r0,
        }
    ]

    for step in range(1, maximum_steps + 1):
        sensor.update(ppm)
        time = step * dt
        progress = (sensor.rs_r0 - start_rs_r0) / amplitude
        if amplitude < 0.0:
            monotonic_resistance &= sensor.rs_r0 <= previous_rs_r0 + 1e-12
            monotonic_voltage &= sensor.voltage >= previous_voltage - 1e-12
        else:
            monotonic_resistance &= sensor.rs_r0 >= previous_rs_r0 - 1e-12
            monotonic_voltage &= sensor.voltage <= previous_voltage + 1e-12

        if t10 is None:
            t10 = _interpolated_crossing(
                previous_time, previous_progress, time, progress, 0.1
            )
        if t90 is None:
            t90 = _interpolated_crossing(
                previous_time, previous_progress, time, progress, 0.9
            )
        if step % trace_stride == 0 or progress >= 0.999:
            trace.append(
                {
                    "phase": phase,
                    "time_s": time,
                    "progress": progress,
                    "rs_r0": sensor.rs_r0,
                    "resistance_ohm": sensor.resistance,
                    "voltage_v": sensor.voltage,
                    "ideal_rs_r0": target_rs_r0,
                }
            )
        previous_time = time
        previous_progress = progress
        previous_rs_r0 = sensor.rs_r0
        previous_voltage = sensor.voltage
        if progress >= 0.999:
            break

    if t10 is None or t90 is None:
        measured_tau = None
    else:
        measured_tau = (t90 - t10) / LOG_9
    discrete_tau = dt / math.log1p(dt / configured_tau)
    tau_relative_error = (
        abs(measured_tau / discrete_tau - 1.0) if measured_tau is not None else math.inf
    )
    return (
        {
            "t10_s": t10,
            "t90_s": t90,
            "tau_measured_s": measured_tau,
            "tau_configured_s": configured_tau,
            "tau_discrete_expected_s": discrete_tau,
            "tau_relative_error": tau_relative_error,
            "tau_pass": tau_relative_error <= TAU_RELATIVE_TOLERANCE,
            "resistance_monotonic": monotonic_resistance,
            "voltage_monotonic": monotonic_voltage,
            "completed": previous_progress >= 0.999,
        },
        trace,
    )


def validate_case(
    model: str,
    gas: str,
    concentration: float,
    rate: float = 20.0,
    retain_trace: bool = False,
) -> tuple[dict, list[dict]]:
    profile = get_mox_profile(model)
    curve = profile["sensitivity"][gas]
    sensor = MOXSensor(model, rate=rate)
    sensor.update({})

    measured_point = float(curve["measured_points"][str(int(concentration))])
    fitted_point = float(_curve_rs_rair(curve, concentration))
    runtime_ideal = sensor._ideal_rs_r0({gas: concentration})
    runtime_ideal_error = abs(runtime_ideal - fitted_point)
    fit_percent_error = 100.0 * (fitted_point / measured_point - 1.0)
    baseline_voltage = sensor.baseline_voltage()
    stride = max(1, int(round(rate * 0.5)))
    response_detectable = runtime_ideal < sensor.sensitivity_air - 1e-12

    if response_detectable:
        rise, rise_trace = _simulate_phase(
            sensor,
            {gas: concentration},
            runtime_ideal,
            float(curve["tau_rise_s"]),
            "rise",
            stride,
        )
        gas_voltage = sensor.voltage
        recovery_start = sensor.rs_r0
        decay, decay_trace = _simulate_phase(
            sensor,
            {},
            sensor.sensitivity_air,
            float(curve["tau_decay_s"]),
            "recovery",
            stride,
        )
    else:
        not_applicable = {
            "t10_s": None,
            "t90_s": None,
            "tau_measured_s": None,
            "tau_configured_s": None,
            "tau_discrete_expected_s": None,
            "tau_relative_error": 0.0,
            "tau_pass": True,
            "resistance_monotonic": True,
            "voltage_monotonic": True,
            "completed": True,
        }
        rise = dict(not_applicable)
        decay = dict(not_applicable)
        rise_trace = []
        decay_trace = []
        gas_voltage = baseline_voltage
        recovery_start = sensor.rs_r0

    checks = {
        "runtime_curve_pass": runtime_ideal_error <= 1e-12,
        "excel_fit_pass": abs(fit_percent_error) <= FIT_PERCENT_TOLERANCE,
        "baseline_voltage_pass": abs(baseline_voltage - 1.0) <= 1e-12,
        "gas_voltage_pass": (
            gas_voltage > baseline_voltage
            if response_detectable
            else abs(gas_voltage - baseline_voltage) <= 1e-12
        ),
        "rise_tau_pass": rise["tau_pass"],
        "recovery_tau_pass": decay["tau_pass"],
        "rise_monotonic_pass": rise["resistance_monotonic"]
        and rise["voltage_monotonic"],
        "recovery_monotonic_pass": decay["resistance_monotonic"]
        and decay["voltage_monotonic"],
        "rise_completed_pass": rise["completed"],
        "recovery_completed_pass": decay["completed"],
    }
    row = {
        "model": model,
        "gas": gas,
        "concentration_ppm": concentration,
        "excel_rs_rair": measured_point,
        "fitted_rs_rair": fitted_point,
        "runtime_ideal_rs_rair": runtime_ideal,
        "runtime_ideal_absolute_error": runtime_ideal_error,
        "fit_percent_error": fit_percent_error,
        "response_detectable": response_detectable,
        "baseline_voltage_v": baseline_voltage,
        "gas_voltage_v": gas_voltage,
        "rise_tau_configured_s": rise["tau_configured_s"],
        "rise_tau_measured_s": rise["tau_measured_s"],
        "rise_t10_s": rise["t10_s"],
        "rise_t90_s": rise["t90_s"],
        "rise_tau_relative_error": rise["tau_relative_error"],
        "recovery_start_rs_rair": recovery_start,
        "recovery_tau_configured_s": decay["tau_configured_s"],
        "recovery_tau_measured_s": decay["tau_measured_s"],
        "recovery_t10_s": decay["t10_s"],
        "recovery_t90_s": decay["t90_s"],
        "recovery_tau_relative_error": decay["tau_relative_error"],
        **checks,
    }
    row["all_pass"] = all(checks.values())
    trace = []
    if retain_trace:
        for sample in rise_trace + decay_trace:
            trace.append(
                {
                    "model": model,
                    "gas": gas,
                    "concentration_ppm": concentration,
                    **sample,
                }
            )
    return row, trace


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _read_excel_comparisons(path: Path) -> list[dict]:
    comparisons = []
    with path.open("r", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            concentration = float(row["concentration_ppm"])
            curve = get_mox_profile(row["family"])["sensitivity"][row["gas"]]
            runtime_rs_rair = float(_curve_rs_rair(curve, concentration))
            odorsim_ra_rg = 1.0 / max(runtime_rs_rair, 0.01)
            excel_ra_rg = float(row["excel_ratio_ra_rg"])
            window_ra_rg = float(row["window_ratio_ra_rg"])
            comparisons.append(
                {
                    "file": SOURCE_FILENAMES[row["source"]],
                    "source": row["source"],
                    "sheet": row["sheet"],
                    "sensor": row["family"],
                    "physical_channel": row["channel"],
                    "gas": row["gas"],
                    "concentration_ppm": concentration,
                    "excel_formula_ra_rg": excel_ra_rg,
                    "adaptive_window_ra_rg": window_ra_rg,
                    "odorsim_fitted_ra_rg": odorsim_ra_rg,
                    "odorsim_error_vs_excel_percent": 100.0
                    * (odorsim_ra_rg / excel_ra_rg - 1.0),
                    "odorsim_error_vs_window_percent": 100.0
                    * (odorsim_ra_rg / window_ra_rg - 1.0),
                }
            )
    return comparisons


def _read_excel_transients(path: Path) -> list[dict]:
    points = []
    with path.open("r", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            concentration = float(row["concentration_ppm"])
            if concentration not in {1.0, 10.0, 100.0}:
                continue
            profile = get_mox_profile(row["family"])
            if row["source"] not in profile["source_ids"]:
                continue
            points.append(
                {
                    "source": row["source"],
                    "sheet": row["sheet"],
                    "model": row["family"],
                    "physical_channel": row["channel"],
                    "gas": row["gas"],
                    "concentration_ppm": concentration,
                    "phase": row["phase"],
                    "time_s": float(row["time_s"]),
                    "rg_ra": float(row["resistance_ratio_rg_ra"]),
                }
            )
    return points


def _write_excel_sheet_plots(path: Path, comparisons: list[dict]) -> list[dict]:
    cjk_font = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    if cjk_font.exists():
        font_manager.fontManager.addfont(cjk_font)
        plt.rcParams["font.family"] = font_manager.FontProperties(
            fname=cjk_font
        ).get_name()
    sheet_groups = {}
    for row in comparisons:
        sheet_groups.setdefault(
            (row["file"], row["source"], row["sheet"]), []
        ).append(row)

    page_index = []
    with PdfPages(path) as pdf:
        for page, ((filename, source, sheet), rows) in enumerate(
            sheet_groups.items(), start=1
        ):
            gases = sorted({row["gas"] for row in rows})
            sensors = sorted({row["sensor"] for row in rows})
            color_map = {
                sensor: plt.colormaps["tab10"](index % 10)
                for index, sensor in enumerate(sensors)
            }
            figure, axes = plt.subplots(
                1,
                len(gases),
                figsize=(max(8.5, 5.2 * len(gases)), 6.2),
                squeeze=False,
            )
            x_fit = np.logspace(0, 2, 200)
            for axis, gas in zip(axes[0], gases):
                gas_rows = [row for row in rows if row["gas"] == gas]
                for sensor in sensors:
                    sensor_rows = [
                        row for row in gas_rows if row["sensor"] == sensor
                    ]
                    if not sensor_rows:
                        continue
                    curve = get_mox_profile(sensor)["sensitivity"][gas]
                    fitted_rs_rair = _curve_rs_rair(curve, x_fit)
                    axis.plot(
                        x_fit,
                        1.0 / np.maximum(fitted_rs_rair, 0.01),
                        color=color_map[sensor],
                        linewidth=1.8,
                        label=sensor,
                    )
                    axis.scatter(
                        [row["concentration_ppm"] for row in sensor_rows],
                        [row["excel_formula_ra_rg"] for row in sensor_rows],
                        color=color_map[sensor],
                        marker="x",
                        s=24,
                        alpha=0.65,
                    )
                    axis.scatter(
                        [row["concentration_ppm"] for row in sensor_rows],
                        [row["adaptive_window_ra_rg"] for row in sensor_rows],
                        facecolors="none",
                        edgecolors=color_map[sensor],
                        marker="o",
                        s=25,
                        alpha=0.65,
                    )
                    medians = {}
                    for concentration in CONCENTRATIONS:
                        values = [
                            row["adaptive_window_ra_rg"]
                            for row in sensor_rows
                            if row["concentration_ppm"] == concentration
                        ]
                        if values:
                            medians[concentration] = float(np.median(values))
                    axis.scatter(
                        list(medians),
                        list(medians.values()),
                        facecolors=color_map[sensor],
                        edgecolors="black",
                        linewidths=0.45,
                        marker="D",
                        s=30,
                        zorder=4,
                    )
                axis.set_xscale("log")
                axis.set_yscale("log")
                axis.set(
                    title=gas,
                    xlabel="Concentration (ppm, log scale)",
                    ylabel="Ra/Rg (log scale)",
                )
                axis.grid(True, which="both", alpha=0.22)
                axis.legend(title="Sensor", fontsize=7, title_fontsize=8)
            marker_legend = [
                Line2D([0], [0], color="black", linewidth=1.8, label="OdorSim fitted curve"),
                Line2D([0], [0], color="black", marker="x", linestyle="None", label="Original Excel formula"),
                Line2D([0], [0], color="black", marker="o", markerfacecolor="none", linestyle="None", label="Adaptive-window value"),
                Line2D([0], [0], color="black", marker="D", linestyle="None", label="Sheet sensor median"),
            ]
            figure.legend(
                handles=marker_legend,
                loc="lower center",
                bbox_to_anchor=(0.5, 0.01),
                ncol=4,
                fontsize=8,
            )
            figure.suptitle(f"{filename} | {sheet} ({source})")
            figure.tight_layout(rect=(0.0, 0.09, 1.0, 0.94))
            pdf.savefig(figure)
            plt.close(figure)
            page_index.append(
                {
                    "page": page,
                    "file": filename,
                    "sheet": sheet,
                    "gases": ", ".join(gases),
                    "sensors": ", ".join(sensors),
                }
            )
    return page_index


def _write_plots(
    path: Path,
    cases: list[dict],
    traces: list[dict],
    excel_transients: list[dict],
) -> None:
    trace_groups = {}
    for row in traces:
        trace_groups.setdefault(
            (
                row["model"],
                row["gas"],
                row["concentration_ppm"],
                row["phase"],
            ),
            [],
        ).append(row)
    excel_groups = {}
    for row in excel_transients:
        excel_groups.setdefault(
            (
                row["model"],
                row["gas"],
                row["concentration_ppm"],
                row["phase"],
            ),
            [],
        ).append(row)

    with PdfPages(path) as pdf:
        for model in MOX_MODELS:
            profile = get_mox_profile(model)
            gases = list(profile["sensitivity"])
            colors = {
                gas: plt.colormaps["tab10"](index)
                for index, gas in enumerate(gases)
            }
            source_markers = {
                "CROSS2024": "o",
                "PRECOAT2024": "^",
                "MULTI2023": "s",
            }
            selected_source_markers = {
                source: source_markers[source]
                for source in profile["source_ids"]
            }
            x_fit = np.logspace(0, 2, 200)

            figure, axis = plt.subplots(figsize=(9, 6), constrained_layout=True)
            for gas, curve in profile["sensitivity"].items():
                measured = [
                    float(curve["measured_points"][str(int(value))])
                    for value in CONCENTRATIONS
                ]
                axis.plot(
                    x_fit,
                    _curve_rs_rair(curve, x_fit),
                    color=colors[gas],
                    label=gas,
                )
                axis.scatter(
                    CONCENTRATIONS, measured, color=colors[gas], s=22
                )
            axis.set_xscale("log")
            axis.set(
                title=f"{model}: Excel medians and runtime sensitivity fits",
                xlabel="Concentration (ppm, log scale)",
                ylabel="Rg/Ra",
            )
            axis.grid(True, which="both", alpha=0.25)
            axis.legend(ncol=2, fontsize=8)
            pdf.savefig(figure)
            plt.close(figure)

            figure, axis = plt.subplots(figsize=(9, 6), constrained_layout=True)
            log_x_fit = np.log10(x_fit)
            log_concentrations = np.log10(CONCENTRATIONS)
            for gas, curve in profile["sensitivity"].items():
                measured_ra_rg = [
                    1.0 / float(curve["measured_points"][str(int(value))])
                    for value in CONCENTRATIONS
                ]
                axis.plot(
                    log_x_fit,
                    1.0 / _curve_rs_rair(curve, x_fit),
                    color=colors[gas],
                    label=gas,
                )
                axis.scatter(
                    log_concentrations,
                    measured_ra_rg,
                    color=colors[gas],
                    s=22,
                )
            axis.set(
                title=f"{model}: Excel medians and runtime conductance fits",
                xlabel="log10 concentration (ppm)",
                ylabel="Ra/Rg",
            )
            axis.grid(True, alpha=0.25)
            axis.legend(ncol=2, fontsize=8)
            pdf.savefig(figure)
            plt.close(figure)

            figure, axes = plt.subplots(
                2, 3, figsize=(15, 8.5), sharey="row"
            )
            for column, concentration in enumerate((1.0, 10.0, 100.0)):
                for row_index, phase in enumerate(("rise", "recovery")):
                    axis = axes[row_index, column]
                    for gas in gases:
                        simulation = trace_groups[
                            (model, gas, concentration, phase)
                        ]
                        axis.plot(
                            [sample["time_s"] for sample in simulation],
                            [sample["rs_r0"] for sample in simulation],
                            color=colors[gas],
                            linewidth=1.8,
                        )
                        workbook = excel_groups.get(
                            (model, gas, concentration, phase), []
                        )
                        workbook = [
                            sample
                            for sample in workbook
                            if sample["time_s"] >= 0.0
                            and (
                                phase == "rise"
                                or sample["time_s"]
                                <= simulation[-1]["time_s"]
                            )
                        ]
                        for source, marker in selected_source_markers.items():
                            source_points = [
                                sample
                                for sample in workbook
                                if sample["source"] == source
                            ]
                            axis.scatter(
                                [
                                    sample["time_s"]
                                    for sample in source_points
                                ],
                                [sample["rg_ra"] for sample in source_points],
                                color=colors[gas],
                                marker=marker,
                                s=11,
                                alpha=0.3,
                                edgecolors="none",
                            )
                    axis.axhline(
                        1.0, color="black", linewidth=0.7, linestyle="--"
                    )
                    axis.set_title(f"{concentration:g} ppm {phase}")
                    axis.set_xlabel("Time from onset/purge (s)")
                    if column == 0:
                        axis.set_ylabel("Rg/Ra")
                    axis.set_xlim(left=0.0)
                    axis.grid(True, alpha=0.22)
            legend_handles = [
                Line2D([0], [0], color=colors[gas], linewidth=2.0, label=gas)
                for gas in gases
            ] + [
                Line2D(
                    [0], [0], color="black", linewidth=1.8, label="OdorSim"
                ),
            ] + [
                Line2D(
                    [0],
                    [0],
                    color="black",
                    marker=marker,
                    linestyle="None",
                    markersize=5,
                    label=f"Excel {source.removesuffix('2024').removesuffix('2023')}",
                )
                for source, marker in selected_source_markers.items()
            ]
            figure.legend(
                handles=legend_handles,
                loc="lower center",
                bbox_to_anchor=(0.5, 0.01),
                ncol=5,
                fontsize=8,
            )
            figure.suptitle(
                f"{model}: transient comparison in normalized resistance"
            )
            figure.tight_layout(rect=(0.0, 0.1, 1.0, 0.95))
            pdf.savefig(figure)
            plt.close(figure)


def _write_report(
    path: Path,
    cases: list[dict],
    rate: float,
    comparison_count: int,
    transient_point_count: int,
    sheet_pages: list[dict],
) -> dict:
    failures = [row for row in cases if not row["all_pass"]]
    check_names = [key for key in cases[0] if key.endswith("_pass") and key != "all_pass"]
    check_failures = Counter(
        key for row in cases for key in check_names if not row[key]
    )
    maximum_fit_error = max(abs(row["fit_percent_error"]) for row in cases)
    maximum_rise_tau_error = max(row["rise_tau_relative_error"] for row in cases)
    maximum_decay_tau_error = max(row["recovery_tau_relative_error"] for row in cases)
    summary = {
        "case_count": len(cases),
        "detectable_step_count": sum(row["response_detectable"] for row in cases),
        "zero_step_count": sum(not row["response_detectable"] for row in cases),
        "passed": len(cases) - len(failures),
        "failed": len(failures),
        "rate_hz": rate,
        "maximum_absolute_fit_percent_error": maximum_fit_error,
        "maximum_rise_tau_relative_error": maximum_rise_tau_error,
        "maximum_recovery_tau_relative_error": maximum_decay_tau_error,
        "failure_counts": dict(check_failures),
        "excel_physical_channel_points": comparison_count,
        "excel_transient_points": transient_point_count,
        "excel_sheet_plot_pages": len(sheet_pages),
    }
    lines = [
        "# MOX Runtime Validation",
        "",
        f"- Cases: {len(cases)} (17 sensors x 6 gases x 5 concentrations)",
        f"- Passed: {summary['passed']}",
        f"- Failed: {summary['failed']}",
        f"- Detectable concentration steps: {summary['detectable_step_count']}",
        f"- Zero-amplitude fitted steps: {summary['zero_step_count']} (timing not applicable)",
        f"- Runtime rate: {rate:g} Hz",
        f"- Maximum Excel-point fit error: {maximum_fit_error:.3f}% (limit {FIT_PERCENT_TOLERANCE:g}%)",
        f"- Maximum rise-tau relative error: {100 * maximum_rise_tau_error:.4f}%",
        f"- Maximum recovery-tau relative error: {100 * maximum_decay_tau_error:.4f}%",
        f"- Excel transient points plotted: {transient_point_count}",
        "- Temperature and humidity are not inputs to this validation.",
        "",
        "## Checks",
        "",
        "- Runtime ideal output exactly reproduces the fitted conductance curve.",
        "- Fitted values stay within 30% of each Excel median point.",
        "- Rise and recovery reproduce the configured discrete-filter time constants within 1%.",
        "- Resistance and voltage remain monotonic for a clean concentration step.",
        "- Clean-air divider voltage is 1 V and gas exposure increases voltage.",
        "",
        "## Runtime PDF Page Index",
        "",
        "Each sensor has three pages: Rg/Ra sensitivity, Ra/Rg versus log10 concentration, and Excel/OdorSim Rg/Ra transients at 1, 10, and 100 ppm.",
        "",
    ]
    for index, model in enumerate(MOX_MODELS):
        lines.append(
            f"- Pages {3 * index + 1}-{3 * index + 3}: {model}"
        )
    lines += [
        "",
        "## Excel Sheet Comparison PDF",
        "",
        "Solid line: OdorSim nominal fit converted to Ra/Rg. X: original Excel formula. Open circle: adaptive-window value. Filled diamond: median across that sheet's physical channels.",
        "",
    ]
    for page in sheet_pages:
        lines.append(
            f"- Page {page['page']}: {page['file']} / {page['sheet']} / {page['gases']}"
        )
    if failures:
        lines += ["", "## Failures", ""]
        for row in failures:
            failed = [key for key in check_names if not row[key]]
            lines.append(
                f"- {row['model']} / {row['gas']} / {row['concentration_ppm']:g} ppm: "
                + ", ".join(failed)
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rate", type=float, default=20.0)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("analysis/mox_runtime_validation")
    )
    parser.add_argument(
        "--excel-audit",
        type=Path,
        default=Path("analysis/mox_window_audit/ratio_comparison.csv"),
    )
    parser.add_argument(
        "--excel-transients",
        type=Path,
        default=Path("analysis/mox_window_audit/transient_points.csv"),
    )
    args = parser.parse_args()
    if args.rate <= 0.0:
        parser.error("--rate must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cases = []
    traces = []
    for model in MOX_MODELS:
        for gas in get_mox_profile(model)["sensitivity"]:
            for concentration in CONCENTRATIONS:
                row, trace = validate_case(
                    model,
                    gas,
                    concentration,
                    rate=args.rate,
                    retain_trace=concentration in {1.0, 10.0, 100.0},
                )
                cases.append(row)
                traces.extend(trace)

    _write_csv(args.output_dir / "validation_cases.csv", cases)
    _write_csv(
        args.output_dir / "reference_timeseries_1_10_100ppm.csv", traces
    )
    excel_transients = _read_excel_transients(args.excel_transients)
    _write_plots(
        args.output_dir / "mox_runtime_validation.pdf",
        cases,
        traces,
        excel_transients,
    )
    comparisons = _read_excel_comparisons(args.excel_audit)
    _write_csv(args.output_dir / "excel_sheet_comparison.csv", comparisons)
    sheet_pages = _write_excel_sheet_plots(
        args.output_dir / "excel_sheet_comparison_ra_rg.pdf", comparisons
    )
    summary = _write_report(
        args.output_dir / "README.md",
        cases,
        args.rate,
        len(comparisons),
        len(excel_transients),
        sheet_pages,
    )
    (args.output_dir / "summary.yaml").write_text(
        yaml.safe_dump(summary, sort_keys=False), encoding="utf-8"
    )
    print(yaml.safe_dump(summary, sort_keys=False).strip())
    raise SystemExit(1 if summary["failed"] else 0)


if __name__ == "__main__":
    main()

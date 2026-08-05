#!/usr/bin/env python3
"""Compare workbook point ratios with adaptive air/gas window medians.

Workbook formulas provide one air-row and one gas-row anchor. This tool uses
the surrounding raw trace to detect gas arrival and the following air purge,
then calculates:

* Rair from the latest stable 7-sample suffix (falling back to 5 or 3);
* Rgas from 3-7 samples around the smoothed gas-phase minimum.

It is an audit only and does not modify the runtime MOX registry.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import shutil
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path
from statistics import median
from xml.etree import ElementTree as ET

import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.backends.backend_pdf import PdfPages

from build_mox_registry import (
    NS,
    TABLES,
    Book,
    col_number,
    label,
    norm,
    red_channel_gases,
    response_rows,
)

FORMULA_RE = re.compile(
    r"^\$?([A-Z]+)\$?(\d+)/\$?([A-Z]+)\$?(\d+)$"
)
WINDOW_SIZES = (7, 5, 3)
LOG_9 = math.log(9.0)


def _column_letters(number: int) -> str:
    letters = ""
    while number:
        number, remainder = divmod(number - 1, 26)
        letters = chr(65 + remainder) + letters
    return letters


def _hhmmss_seconds(value) -> float | None:
    if value is None:
        return None
    try:
        text = f"{int(float(value)):06d}"
        hour, minute, second = int(text[:2]), int(text[2:4]), int(text[4:])
    except (TypeError, ValueError):
        return None
    if hour > 23 or minute > 59 or second > 59:
        return None
    return float(hour * 3600 + minute * 60 + second)


def _normalized_workbook(source: Path, output_dir: Path) -> Path:
    """Re-save through LibreOffice so shared formulas are expanded per cell."""
    executable = shutil.which("libreoffice") or shutil.which("soffice")
    if not executable:
        raise RuntimeError("LibreOffice is required to expand workbook formulas")
    subprocess.run(
        [
            executable,
            "--headless",
            "--convert-to",
            "xlsx",
            "--outdir",
            str(output_dir),
            str(source),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    result = output_dir / f"{source.stem}.xlsx"
    if not result.exists():
        raise RuntimeError(f"LibreOffice did not produce {result}")
    return result


def _channel_families(books: dict[str, Book]) -> dict[tuple[str, str], str]:
    result = {}
    for table in TABLES:
        book = books[table.source]
        if table.sheet not in book.targets:
            continue
        cells = book.sheet(table.sheet)
        for column in range(table.first_col, table.last_col + 1):
            physical_channel = cells.get((3, column))
            channel = (
                str(physical_channel).strip()
                if re.fullmatch(r"CH\d+\s*", str(physical_channel or ""), re.IGNORECASE)
                else label(cells, column)
            )
            result[(table.source, norm(channel))] = table.family
    return result


def _median_filter(values: list[float]) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if len(array) < 3:
        return array
    padded = np.pad(array, (1, 1), mode="edge")
    return np.asarray(
        [median(padded[index : index + 3]) for index in range(len(array))],
        dtype=float,
    )


def _window_stats(rows: list[int], cells: dict, column: int) -> dict:
    values = np.asarray([float(cells[(row, column)]) for row in rows], dtype=float)
    center = float(np.median(values))
    mad = float(np.median(np.abs(values - center)))
    times = [_hhmmss_seconds(cells.get((row, 1))) for row in rows]
    if all(value is not None for value in times):
        unwrapped = np.asarray(times, dtype=float)
        for index in range(1, len(unwrapped)):
            if unwrapped[index] < unwrapped[index - 1]:
                unwrapped[index:] += 86400.0
        duration = float(unwrapped[-1] - unwrapped[0])
        slope = (
            float(np.polyfit(unwrapped - unwrapped[0], values, 1)[0])
            if duration > 0 and len(values) >= 3
            else 0.0
        )
    else:
        duration = float(max(len(values) - 1, 0))
        slope = (
            float(np.polyfit(np.arange(len(values)), values, 1)[0])
            if len(values) >= 3
            else 0.0
        )
    return {
        "median": center,
        "mad": mad,
        "slope_ohm_s": slope,
        "duration_s": duration,
        "total_fitted_drift": abs(slope) * duration,
    }


def _timed_values(cells: dict, column: int, rows) -> tuple[np.ndarray, np.ndarray]:
    """Return monotonically unwrapped timestamps and resistance values."""
    samples = []
    day_offset = 0.0
    previous = None
    for row in rows:
        timestamp = _hhmmss_seconds(cells.get((row, 1)))
        value = cells.get((row, column))
        if timestamp is None or not isinstance(value, (int, float)):
            continue
        if previous is not None and timestamp + day_offset < previous - 43200.0:
            day_offset += 86400.0
        timestamp += day_offset
        previous = timestamp
        samples.append((timestamp, float(value)))
    if not samples:
        return np.asarray([]), np.asarray([])
    return (
        np.asarray([sample[0] for sample in samples], dtype=float),
        np.asarray([sample[1] for sample in samples], dtype=float),
    )


def _first_crossing(
    timestamps: np.ndarray, progress: np.ndarray, level: float
) -> float | None:
    for index in range(1, len(progress)):
        if progress[index - 1] < level <= progress[index]:
            interval = progress[index] - progress[index - 1]
            if interval <= 0.0:
                return float(timestamps[index])
            fraction = (level - progress[index - 1]) / interval
            return float(
                timestamps[index - 1]
                + fraction * (timestamps[index] - timestamps[index - 1])
            )
    return None


def _estimate_time_constant(
    cells: dict,
    column: int,
    rows,
    initial_resistance: float,
    final_resistance: float,
) -> dict:
    """Estimate an effective first-order tau from interpolated 10-90% time."""
    timestamps, resistance = _timed_values(cells, column, rows)
    amplitude = final_resistance - initial_resistance
    if len(timestamps) < 3 or abs(amplitude) <= 0.0:
        return {"tau_s": None, "quality": "unresolved"}

    progress = (resistance - initial_resistance) / amplitude
    progress = np.maximum.accumulate(np.clip(progress, -0.25, 1.25))
    t10 = _first_crossing(timestamps, progress, 0.1)
    t90 = _first_crossing(timestamps, progress, 0.9)
    if t10 is None or t90 is None or t90 <= t10:
        return {"tau_s": None, "quality": "unresolved"}

    intervals = np.diff(timestamps)
    intervals = intervals[intervals > 0.0]
    if not len(intervals):
        return {"tau_s": None, "quality": "unresolved"}
    sample_interval = float(np.median(intervals))
    transition_10_90 = float(t90 - t10)
    if transition_10_90 >= 2.0 * sample_interval:
        quality = "good"
    elif transition_10_90 >= 0.5 * sample_interval:
        quality = "coarse"
    else:
        quality = "unresolved"
    return {
        "tau_s": transition_10_90 / LOG_9,
        "quality": quality,
        "transition_10_90_s": transition_10_90,
        "sample_interval_s": sample_interval,
    }


def _contiguous_rows(cells: dict, column: int, start: int, end: int) -> list[int]:
    return [
        row
        for row in range(max(start, 1), end + 1)
        if (row, column) in cells
        and isinstance(cells[(row, column)], (int, float))
        and math.isfinite(float(cells[(row, column)]))
    ]


def _strongest_step(
    cells: dict,
    column: int,
    start: int,
    end: int,
    direction: int,
) -> int | None:
    rows = _contiguous_rows(cells, column, start, end)
    if len(rows) < 2:
        return None
    values = [float(cells[(row, column)]) for row in rows]
    smooth = _median_filter(values)
    differences = np.diff(smooth)
    offset = int(np.argmin(differences) if direction < 0 else np.argmax(differences))
    return rows[offset + 1]


def _select_air_window(
    cells: dict,
    column: int,
    air_end: int,
    preliminary_amplitude: float,
) -> tuple[list[int], dict, str]:
    fallback = None
    for size in WINDOW_SIZES:
        rows = _contiguous_rows(cells, column, air_end - size + 1, air_end)
        if len(rows) != size:
            continue
        stats = _window_stats(rows, cells, column)
        tolerance = max(
            4.0 * stats["mad"],
            0.003 * abs(stats["median"]),
            0.03 * max(preliminary_amplitude, 0.0),
        )
        fallback = (rows, stats, f"drifting_{size}")
        if stats["total_fitted_drift"] <= tolerance:
            return rows, stats, f"stable_{size}"
    if fallback is None:
        raise ValueError("fewer than three usable air samples")
    return fallback


def _select_gas_window(
    cells: dict,
    column: int,
    gas_start: int,
    purge_row: int,
    gas_anchor_row: int,
    air_stats: dict,
) -> tuple[list[int], dict, int, str]:
    # The workbook gas point was manually selected inside the intended pulse.
    # Search locally around it so an unrelated late dip or the next pulse cannot
    # become this pulse's "minimum".
    rows = _contiguous_rows(
        cells,
        column,
        max(gas_start, gas_anchor_row - 15),
        min(purge_row - 1, gas_anchor_row + 15),
    )
    if len(rows) < 3:
        raise ValueError("fewer than three usable gas samples")
    values = [float(cells[(row, column)]) for row in rows]
    smooth = _median_filter(values)
    minimum_offset = int(np.argmin(smooth))
    minimum_row = rows[minimum_offset]
    amplitude = max(air_stats["median"] - float(smooth[minimum_offset]), 0.0)
    tolerance = max(3.0 * air_stats["mad"], 0.02 * amplitude)

    for size in WINDOW_SIZES:
        half = size // 2
        start = max(0, min(minimum_offset - half, len(rows) - size))
        selected = rows[start : start + size]
        if len(selected) != size:
            continue
        selected_smooth = smooth[start : start + size]
        if np.all(selected_smooth <= smooth[minimum_offset] + tolerance):
            stats = _window_stats(selected, cells, column)
            return selected, stats, minimum_row, f"minimum_{size}"

    start = max(0, min(minimum_offset - 1, len(rows) - 3))
    selected = rows[start : start + 3]
    stats = _window_stats(selected, cells, column)
    return selected, stats, minimum_row, "forced_minimum_3"


def _extract_formula_records(
    source: str,
    book: Book,
    family_by_channel: dict[tuple[str, str], str],
    red_pairs: set[tuple[str, str]],
) -> list[dict]:
    records = []
    for sheet_name in book.targets:
        cells = book.sheet(sheet_name)
        response_by_row = {
            row: (gas_name, concentration)
            for row, gas_name, concentration in response_rows(cells)
        }
        if not response_by_row:
            continue
        root = ET.fromstring(book.zf.read(book.targets[sheet_name]))
        for cell in root.findall(".//x:sheetData/x:row/x:c", NS):
            formula = cell.find("x:f", NS)
            value = cell.find("x:v", NS)
            match_cell = re.fullmatch(r"([A-Z]+)(\d+)", cell.attrib["r"])
            if formula is None or value is None or match_cell is None:
                continue
            response_row = int(match_cell.group(2))
            if response_row not in response_by_row:
                continue
            match_formula = FORMULA_RE.fullmatch(formula.text or "")
            if match_formula is None:
                continue
            left_col, air_row, right_col, gas_row = match_formula.groups()
            if left_col != right_col:
                continue
            column = col_number(left_col)
            channel = str(cells.get((3, column), "")).strip()
            family = family_by_channel.get((source, norm(channel)))
            gas_name, concentration = response_by_row[response_row]
            if (
                family is None
                or (norm(channel), gas_name) in red_pairs
                or not re.fullmatch(r"CH\d+\s*", channel, re.IGNORECASE)
            ):
                continue
            try:
                excel_ratio = float(value.text)
            except (TypeError, ValueError):
                continue
            records.append(
                {
                    "source": source,
                    "sheet": sheet_name,
                    "family": family,
                    "channel": channel,
                    "gas": gas_name,
                    "concentration_ppm": float(concentration),
                    "column": column,
                    "formula": formula.text,
                    "air_anchor_row": int(air_row),
                    "gas_anchor_row": int(gas_row),
                    "excel_ratio_ra_rg": excel_ratio,
                    "cells": cells,
                }
            )
    return records


def _analyze_records(
    records: list[dict],
) -> tuple[list[dict], list[dict], list[dict]]:
    grouped = defaultdict(list)
    for record in records:
        grouped[(record["source"], record["sheet"], record["column"])].append(record)

    results, failures, transient_points = [], [], []
    for group in grouped.values():
        group.sort(key=lambda item: item["gas_anchor_row"])
        max_raw_row = max(row for row, _ in group[0]["cells"])
        for index, record in enumerate(group):
            cells, column = record["cells"], record["column"]
            try:
                previous_gas = (
                    group[index - 1]["gas_anchor_row"] if index > 0 else 1
                )
                onset = _strongest_step(
                    cells,
                    column,
                    max(record["air_anchor_row"], previous_gas + 1),
                    record["gas_anchor_row"],
                    -1,
                )
                if onset is None:
                    raise ValueError("could not detect gas onset")
                air_end = onset - 2
                preliminary_amplitude = max(
                    float(cells[(record["air_anchor_row"], column)])
                    - float(cells[(record["gas_anchor_row"], column)]),
                    0.0,
                )
                air_rows, air_stats, air_quality = _select_air_window(
                    cells, column, air_end, preliminary_amplitude
                )
                next_gas = (
                    group[index + 1]["gas_anchor_row"]
                    if index + 1 < len(group)
                    else min(record["gas_anchor_row"] + 120, max_raw_row)
                )
                purge = _strongest_step(
                    cells,
                    column,
                    record["gas_anchor_row"],
                    next_gas,
                    1,
                )
                if purge is None or purge <= onset + 2:
                    raise ValueError("could not detect following air purge")
                gas_rows, gas_stats, minimum_row, gas_quality = _select_gas_window(
                    cells,
                    column,
                    onset,
                    purge,
                    record["gas_anchor_row"],
                    air_stats,
                )
                response_amplitude = air_stats["median"] - gas_stats["median"]
                minimum_dynamic_amplitude = max(
                    10.0 * air_stats["mad"],
                    0.005 * abs(air_stats["median"]),
                )
                if response_amplitude > minimum_dynamic_amplitude:
                    rise_dynamics = _estimate_time_constant(
                        cells,
                        column,
                        range(air_rows[-1], minimum_row + 1),
                        air_stats["median"],
                        gas_stats["median"],
                    )
                    decay_dynamics = _estimate_time_constant(
                        cells,
                        column,
                        range(max(purge - 1, 1), next_gas + 1),
                        gas_stats["median"],
                        air_stats["median"],
                    )
                else:
                    rise_dynamics = {
                        "tau_s": None,
                        "quality": "low_signal",
                    }
                    decay_dynamics = {
                        "tau_s": None,
                        "quality": "low_signal",
                    }
                ratio = air_stats["median"] / gas_stats["median"]
                for phase, phase_rows in (
                    ("rise", range(max(onset - 1, 1), minimum_row + 1)),
                    ("recovery", range(max(purge - 1, 1), next_gas + 1)),
                ):
                    timestamps, resistances = _timed_values(
                        cells, column, phase_rows
                    )
                    if not len(timestamps):
                        continue
                    relative_time = timestamps - timestamps[0]
                    stop = next(
                        (
                            index
                            for index in range(1, len(relative_time))
                            if relative_time[index] < relative_time[index - 1]
                        ),
                        len(relative_time),
                    )
                    if phase == "recovery":
                        amplitude = air_stats["median"] - gas_stats["median"]
                        if amplitude > 0.0:
                            progress = (
                                resistances[:stop] - gas_stats["median"]
                            ) / amplitude
                            recovered = np.flatnonzero(progress >= 0.99)
                            if len(recovered):
                                stop = min(stop, int(recovered[0]) + 3)
                            elif len(progress):
                                # If the chamber never reaches 99% clean air,
                                # stop at its best recovery before the next
                                # gas exposure drives resistance down again.
                                stop = min(stop, int(np.argmax(progress)) + 1)
                    for timestamp, resistance in zip(
                        relative_time[:stop], resistances[:stop]
                    ):
                        transient_points.append(
                            {
                                "source": record["source"],
                                "sheet": record["sheet"],
                                "family": record["family"],
                                "channel": record["channel"],
                                "gas": record["gas"],
                                "concentration_ppm": record[
                                    "concentration_ppm"
                                ],
                                "phase": phase,
                                "time_s": float(timestamp),
                                "resistance_ratio_rg_ra": float(
                                    resistance / air_stats["median"]
                                ),
                            }
                        )
                result = {
                    key: value
                    for key, value in record.items()
                    if key not in {"cells", "column"}
                }
                result.update(
                    {
                        "detected_gas_onset_row": onset,
                        "detected_purge_row": purge,
                        "air_window_rows": air_rows,
                        "gas_window_rows": gas_rows,
                        "gas_minimum_row": minimum_row,
                        "air_window_sample_indices": [
                            cells.get((row, 2)) for row in air_rows
                        ],
                        "gas_window_sample_indices": [
                            cells.get((row, 2)) for row in gas_rows
                        ],
                        "detected_onset_sample_index": cells.get((onset, 2)),
                        "detected_purge_sample_index": cells.get((purge, 2)),
                        "air_median_ohm": air_stats["median"],
                        "air_mad_ohm": air_stats["mad"],
                        "air_slope_ohm_s": air_stats["slope_ohm_s"],
                        "air_quality": air_quality,
                        "gas_median_ohm": gas_stats["median"],
                        "gas_mad_ohm": gas_stats["mad"],
                        "gas_quality": gas_quality,
                        "tau_rise_s": rise_dynamics["tau_s"],
                        "tau_rise_quality": rise_dynamics["quality"],
                        "tau_rise_10_90_s": rise_dynamics.get(
                            "transition_10_90_s"
                        ),
                        "tau_decay_s": decay_dynamics["tau_s"],
                        "tau_decay_quality": decay_dynamics["quality"],
                        "tau_decay_10_90_s": decay_dynamics.get(
                            "transition_10_90_s"
                        ),
                        "window_ratio_ra_rg": ratio,
                        "difference": ratio - record["excel_ratio_ra_rg"],
                        "percent_difference": 100.0
                        * (ratio / record["excel_ratio_ra_rg"] - 1.0),
                    }
                )
                results.append(result)
            except (KeyError, ValueError, ZeroDivisionError) as exc:
                failures.append(
                    {
                        key: value
                        for key, value in record.items()
                        if key not in {"cells", "column"}
                    }
                    | {"reason": str(exc)}
                )
    return results, failures, transient_points


def _write_csv(path: Path, rows: list[dict]) -> None:
    columns = [
        "source", "sheet", "family", "channel", "gas", "concentration_ppm",
        "formula", "excel_ratio_ra_rg", "window_ratio_ra_rg", "difference",
        "percent_difference", "air_median_ohm", "gas_median_ohm",
        "air_mad_ohm", "gas_mad_ohm", "air_slope_ohm_s", "air_quality",
        "gas_quality", "detected_onset_sample_index",
        "detected_purge_sample_index", "air_window_sample_indices",
        "gas_window_sample_indices", "tau_rise_s", "tau_rise_quality",
        "tau_rise_10_90_s", "tau_decay_s", "tau_decay_quality",
        "tau_decay_10_90_s",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _summary(results: list[dict], failures: list[dict]) -> dict:
    differences = np.asarray(
        [result["percent_difference"] for result in results], dtype=float
    )
    by_family = {}
    for family in sorted({result["family"] for result in results}):
        values = np.asarray(
            [
                result["percent_difference"]
                for result in results
                if result["family"] == family
            ],
            dtype=float,
        )
        by_family[family] = {
            "count": len(values),
            "median_percent_difference": float(np.median(values)),
            "median_absolute_percent_difference": float(np.median(np.abs(values))),
            "p95_absolute_percent_difference": float(np.percentile(np.abs(values), 95)),
            "maximum_absolute_percent_difference": float(np.max(np.abs(values))),
        }
    return {
        "successful_comparisons": len(results),
        "failed_comparisons": len(failures),
        "median_percent_difference": float(np.median(differences)),
        "median_absolute_percent_difference": float(np.median(np.abs(differences))),
        "p90_absolute_percent_difference": float(np.percentile(np.abs(differences), 90)),
        "p95_absolute_percent_difference": float(np.percentile(np.abs(differences), 95)),
        "maximum_absolute_percent_difference": float(np.max(np.abs(differences))),
        "within_1_percent": int(np.sum(np.abs(differences) <= 1.0)),
        "within_2_percent": int(np.sum(np.abs(differences) <= 2.0)),
        "within_5_percent": int(np.sum(np.abs(differences) <= 5.0)),
        "by_family": by_family,
    }


def _fit_ra_rg(points: dict[float, float]) -> tuple[float, float]:
    ordered = sorted(points.items())
    x = np.log([point[0] for point in ordered])
    y = np.log([point[1] for point in ordered])
    b, log_a = np.polyfit(x, y, 1)
    return float(np.exp(log_a)), float(b)


def _nominal_comparisons(results: list[dict]) -> tuple[list[dict], list[dict]]:
    grouped = defaultdict(list)
    for result in results:
        grouped[
            (result["family"], result["gas"], result["concentration_ppm"])
        ].append(result)
    ratios = []
    for (family, gas_name, concentration), rows in sorted(grouped.items()):
        excel = float(median(row["excel_ratio_ra_rg"] for row in rows))
        window = float(median(row["window_ratio_ra_rg"] for row in rows))
        ratios.append(
            {
                "family": family,
                "gas": gas_name,
                "concentration_ppm": concentration,
                "channel_count": len(rows),
                "excel_median_ra_rg": excel,
                "window_median_ra_rg": window,
                "percent_difference": 100.0 * (window / excel - 1.0),
            }
        )

    fit_groups = defaultdict(list)
    for row in ratios:
        fit_groups[(row["family"], row["gas"])].append(row)
    fits = []
    for (family, gas_name), rows in sorted(fit_groups.items()):
        if len(rows) < 3:
            continue
        excel_a, excel_b = _fit_ra_rg(
            {
                row["concentration_ppm"]: row["excel_median_ra_rg"]
                for row in rows
            }
        )
        window_a, window_b = _fit_ra_rg(
            {
                row["concentration_ppm"]: row["window_median_ra_rg"]
                for row in rows
            }
        )
        fits.append(
            {
                "family": family,
                "gas": gas_name,
                "excel_a_ra_rg": excel_a,
                "excel_b_ra_rg": excel_b,
                "window_a_ra_rg": window_a,
                "window_b_ra_rg": window_b,
                "a_percent_difference": 100.0 * (window_a / excel_a - 1.0),
                "b_difference": window_b - excel_b,
            }
        )
    return ratios, fits


def _write_table(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _transient_trace_audit(
    results: list[dict], transient_points: list[dict]
) -> tuple[list[dict], list[dict]]:
    grouped_points = defaultdict(list)
    for point in transient_points:
        grouped_points[
            (
                point["source"],
                point["sheet"],
                point["family"],
                point["channel"],
                point["gas"],
                point["concentration_ppm"],
                point["phase"],
            )
        ].append(point)

    audit_rows = []
    inventory = {}
    for result in results:
        identity = (
            result["source"],
            result["sheet"],
            result["family"],
            result["channel"],
            result["gas"],
            result["concentration_ppm"],
        )
        rise = grouped_points.get(identity + ("rise",), [])
        rise_ratios = [
            float(point["resistance_ratio_rg_ra"]) for point in rise
        ]
        tail = rise_ratios[-min(5, len(rise_ratios)) :]
        plotted_tail = float(median(tail)) if tail else None
        excel_rg_ra = 1.0 / float(result["excel_ratio_ra_rg"])
        window_rg_ra = 1.0 / float(result["window_ratio_ra_rg"])
        audit_rows.append(
            {
                "source": result["source"],
                "sheet": result["sheet"],
                "family": result["family"],
                "channel": result["channel"],
                "gas": result["gas"],
                "concentration_ppm": result["concentration_ppm"],
                "excel_formula_ra_rg": result["excel_ratio_ra_rg"],
                "excel_formula_rg_ra": excel_rg_ra,
                "adaptive_window_ra_rg": result["window_ratio_ra_rg"],
                "adaptive_window_rg_ra": window_rg_ra,
                "plotted_rise_point_count": len(rise_ratios),
                "plotted_rise_tail_median_rg_ra": plotted_tail,
                "plotted_rise_minimum_rg_ra": (
                    min(rise_ratios) if rise_ratios else None
                ),
                "tail_error_vs_window_percent": (
                    100.0 * (plotted_tail / window_rg_ra - 1.0)
                    if plotted_tail is not None
                    else None
                ),
                "tail_error_vs_excel_percent": (
                    100.0 * (plotted_tail / excel_rg_ra - 1.0)
                    if plotted_tail is not None
                    else None
                ),
            }
        )
        inventory_key = identity[:5]
        record = inventory.setdefault(
            inventory_key,
            {
                "source": result["source"],
                "sheet": result["sheet"],
                "family": result["family"],
                "channel": result["channel"],
                "gas": result["gas"],
                "concentrations": [],
            },
        )
        record["concentrations"].append(result["concentration_ppm"])
    inventory_rows = []
    for record in inventory.values():
        concentrations = sorted(set(record.pop("concentrations")))
        inventory_rows.append(
            {
                **record,
                "concentration_count": len(concentrations),
                "concentrations_ppm": ";".join(
                    f"{value:g}" for value in concentrations
                ),
            }
        )
    return audit_rows, inventory_rows


def _write_plots(
    path: Path,
    results: list[dict],
    nominal_ratios: list[dict],
) -> None:
    excel = np.asarray([row["excel_ratio_ra_rg"] for row in results])
    window = np.asarray([row["window_ratio_ra_rg"] for row in results])
    difference = np.asarray([row["percent_difference"] for row in results])
    with PdfPages(path) as pdf:
        figure, axis = plt.subplots(figsize=(8, 7), constrained_layout=True)
        axis.scatter(excel, window, s=10, alpha=0.35)
        bounds = [min(excel.min(), window.min()), max(excel.max(), window.max())]
        axis.plot(bounds, bounds, color="black", linewidth=1.2)
        axis.set(xlabel="Excel single-point Ra/Rg", ylabel="Window-median Ra/Rg")
        axis.grid(True, alpha=0.25)
        pdf.savefig(figure)
        plt.close(figure)

        nominal_excel = np.asarray(
            [row["excel_median_ra_rg"] for row in nominal_ratios]
        )
        nominal_window = np.asarray(
            [row["window_median_ra_rg"] for row in nominal_ratios]
        )
        figure, axis = plt.subplots(figsize=(8, 7), constrained_layout=True)
        axis.scatter(nominal_excel, nominal_window, s=18, alpha=0.55)
        bounds = [
            min(nominal_excel.min(), nominal_window.min()),
            max(nominal_excel.max(), nominal_window.max()),
        ]
        axis.plot(bounds, bounds, color="black", linewidth=1.2)
        axis.set(
            xlabel="Excel median-across-channel Ra/Rg",
            ylabel="Window median-across-channel Ra/Rg",
        )
        axis.grid(True, alpha=0.25)
        pdf.savefig(figure)
        plt.close(figure)

        figure, axis = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
        clipped = difference[np.abs(difference) <= np.percentile(np.abs(difference), 99)]
        axis.hist(clipped, bins=60, color="#2878B5", alpha=0.85)
        axis.axvline(0.0, color="black", linewidth=1.2)
        axis.set(
            xlabel="Window ratio difference from Excel (%)",
            ylabel="Comparison count",
        )
        axis.grid(True, axis="y", alpha=0.25)
        pdf.savefig(figure)
        plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis/mox_window_audit"),
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="skip PDF plots (useful when called by the registry generator)",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    source_by_file = {table.file: table.source for table in TABLES}
    with tempfile.TemporaryDirectory(prefix="odorsim_ratio_audit_") as temp:
        temp_path = Path(temp)
        books = {
            source: Book(_normalized_workbook(args.source_dir / filename, temp_path))
            for filename, source in source_by_file.items()
        }
        families = _channel_families(books)
        records = []
        for source, book in books.items():
            records.extend(
                _extract_formula_records(
                    source,
                    book,
                    families,
                    red_channel_gases(book),
                )
            )
        results, failures, transient_points = _analyze_records(records)

    results.sort(
        key=lambda row: (
            row["family"], row["channel"], row["gas"], row["concentration_ppm"]
        )
    )
    summary = _summary(results, failures)
    nominal_ratios, nominal_fits = _nominal_comparisons(results)
    transient_audit, transient_inventory = _transient_trace_audit(
        results, transient_points
    )
    nominal_difference = np.asarray(
        [row["percent_difference"] for row in nominal_ratios], dtype=float
    )
    summary["nominal_median_comparisons"] = {
        "count": len(nominal_ratios),
        "median_absolute_percent_difference": float(
            np.median(np.abs(nominal_difference))
        ),
        "p90_absolute_percent_difference": float(
            np.percentile(np.abs(nominal_difference), 90)
        ),
        "p95_absolute_percent_difference": float(
            np.percentile(np.abs(nominal_difference), 95)
        ),
        "maximum_absolute_percent_difference": float(
            np.max(np.abs(nominal_difference))
        ),
        "within_1_percent": int(np.sum(np.abs(nominal_difference) <= 1.0)),
        "within_2_percent": int(np.sum(np.abs(nominal_difference) <= 2.0)),
        "within_5_percent": int(np.sum(np.abs(nominal_difference) <= 5.0)),
    }
    _write_csv(args.output_dir / "ratio_comparison.csv", results)
    _write_table(
        args.output_dir / "transient_points.csv", transient_points
    )
    _write_table(
        args.output_dir / "transient_trace_audit.csv", transient_audit
    )
    _write_table(
        args.output_dir / "transient_source_inventory.csv",
        transient_inventory,
    )
    _write_table(
        args.output_dir / "nominal_ratio_comparison.csv", nominal_ratios
    )
    _write_table(
        args.output_dir / "nominal_ab_comparison.csv", nominal_fits
    )
    (args.output_dir / "ratio_comparison.yaml").write_text(
        yaml.safe_dump(
            {
                "summary": summary,
                "nominal_ratio_comparisons": nominal_ratios,
                "nominal_ab_comparisons": nominal_fits,
                "channel_comparisons": results,
                "failures": failures,
            },
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    if not args.no_plots:
        _write_plots(
            args.output_dir / "ratio_comparison_plots.pdf",
            results,
            nominal_ratios,
        )
    print(yaml.safe_dump(summary, sort_keys=False, allow_unicode=True))


if __name__ == "__main__":
    main()

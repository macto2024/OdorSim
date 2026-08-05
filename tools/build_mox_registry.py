#!/usr/bin/env python3
"""Generate median-first MOX sensor fits from calibration workbooks.

For each sensor, gas, and concentration, the importer takes the median of the
physical-channel ``Ra/Rg`` measurements. It then fits the excess-conductance
power law ``Ra/Rg - 1 = K * concentration_ppm ** n`` for runtime. Individual
channels are retained as unfitted provenance, never as selectable models.
"""
from __future__ import annotations

import argparse
import csv
import math
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from xml.etree import ElementTree as ET

import numpy as np
import yaml
from scipy.optimize import curve_fit

NS = {"x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
RNS = {"r": "http://schemas.openxmlformats.org/package/2006/relationships"}
RID = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"
CONCS = (1.0, 5.0, 10.0, 50.0, 100.0)
GASES = ("methanol", "ethanol", "acetic_acid", "ethyl_acetate", "toluene", "acetone")
RUNTIME_GASES = set(GASES)
ALIASES = {
    "methanol": "methanol", "甲醇": "methanol",
    "ethanol": "ethanol", "乙醇": "ethanol",
    "aceticacid": "acetic_acid", "乙酸": "acetic_acid", "醋酸": "acetic_acid",
    "ethylacetate": "ethyl_acetate", "乙酸乙酯": "ethyl_acetate",
    "toluene": "toluene", "甲苯": "toluene",
    "acetone": "acetone", "丙酮": "acetone",
}


@dataclass(frozen=True)
class Table:
    file: str
    source: str
    sheet: str
    family: str
    first_col: int
    last_col: int


TABLES = (
    Table("100-500交叉物质.xlsx", "CROSS2024", "Sheet3-101", "101", 4, 8),
    Table("100-500交叉物质.xlsx", "CROSS2024", "Sheet4-203", "203", 4, 9),
    Table("100-500交叉物质.xlsx", "CROSS2024", "Sheet5-302", "302", 4, 9),
    Table("100-500交叉物质.xlsx", "CROSS2024", "Sheet6-401", "401", 4, 9),
    Table("100-500交叉物质.xlsx", "CROSS2024", "Sheet7-509", "509", 4, 9),
    Table("100-500交叉物质.xlsx", "CROSS2024", "Sheet8-费加罗", "TGS2610", 4, 5),
    Table("100-500交叉物质.xlsx", "CROSS2024", "Sheet8-费加罗", "TGS2611", 6, 7),
    Table("100-500交叉物质.xlsx", "CROSS2024", "Sheet8-费加罗", "TGS2600", 8, 9),
    Table("104和303涂覆分子筛前.xlsx", "PRECOAT2024", "Sheet4-500", "500-1", 3, 12),
    Table("104和303涂覆分子筛前.xlsx", "PRECOAT2024", "Sheet5-104", "104", 3, 20),
    Table("104和303涂覆分子筛前.xlsx", "PRECOAT2024", "Sheet6-303", "303", 3, 20),
    *(Table("104和303涂覆分子筛前.xlsx", "PRECOAT2024", sheet, family, col, col)
      for family, col in (("TGS2620", 4), ("TGS2600", 5), ("TGS2602", 6))
      for sheet in ("Sheet1-甲醇、乙醇、乙酸", "Sheet2-乙酸乙酯", "Sheet3-甲苯、丙酮")),
    Table("601和701和43#多物质.xls", "MULTI2023", "Sheet6-500-1", "500-1", 4, 13),
    Table("601和701和43#多物质.xls", "MULTI2023", "Sheet7-43#烧", "43_SINTERED", 4, 6),
    Table("601和701和43#多物质.xls", "MULTI2023", "Sheet8-43#未", "43_UNSINTERED", 4, 6),
    Table("601和701和43#多物质.xls", "MULTI2023", "Sheet9-601", "601", 4, 8),
    Table("601和701和43#多物质.xls", "MULTI2023", "Sheet10-701", "701", 4, 8),
    *(Table("601和701和43#多物质.xls", "MULTI2023", sheet, family, col, col)
      for family, col in (("TGS2620", 4), ("TGS2600", 5), ("TGS2602", 6))
      for sheet in ("Sheet1-甲醇、乙醇", "Sheet2-乙酸", "Sheet3-乙酸乙酯", "Sheet4-甲苯", "Sheet5-丙酮")),
)

# Stock GADEN rise/recovery values are retained for reference only:
# TGS2620 (2.96, 15.71), TGS2600 (4.80, 18.75),
# TGS2611 (3.44, 6.35), TGS2610 (3.44, 6.35).
# They are not measurements from the supplied workbooks and are never loaded.
DEFAULT_TAU = (4.0, 15.0)
SOURCE_PRIORITY = ("MULTI2023", "CROSS2024", "PRECOAT2024")
SOURCE_OVERRIDES = {
    "TGS2610": "CROSS2024",
    "TGS2611": "CROSS2024",
    "TGS2600": "CROSS2024",
}


def selected_source(family: str, available_sources) -> str:
    available = set(available_sources)
    override = SOURCE_OVERRIDES.get(family)
    if override is not None:
        if override not in available:
            raise ValueError(
                f"{family} requires {override}, available: {sorted(available)}"
            )
        return override
    for source in SOURCE_PRIORITY:
        if source in available:
            return source
    raise ValueError(f"{family} has no recognized calibration source")


def number(value):
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def norm(value):
    return re.sub(r"[\s_-]+", "", str(value or "")).lower()


def gas(value):
    return ALIASES.get(norm(value).removesuffix("响应"))


def col_number(ref):
    result = 0
    for letter in re.match(r"[A-Z]+", ref).group(0):
        result = result * 26 + ord(letter) - 64
    return result


class Book:
    def __init__(self, path):
        self.zf = zipfile.ZipFile(path)
        try:
            root = ET.fromstring(self.zf.read("xl/sharedStrings.xml"))
            self.strings = ["".join(t.text or "" for t in si.findall(".//x:t", NS))
                            for si in root.findall("x:si", NS)]
        except KeyError:
            self.strings = []
        root = ET.fromstring(self.zf.read("xl/workbook.xml"))
        relroot = ET.fromstring(self.zf.read("xl/_rels/workbook.xml.rels"))
        rels = {r.attrib["Id"]: r.attrib["Target"].lstrip("/")
                for r in relroot.findall("r:Relationship", RNS)}
        self.targets = {}
        for sheet in root.findall("x:sheets/x:sheet", NS):
            target = rels[sheet.attrib[RID]]
            self.targets[sheet.attrib["name"]] = target if target.startswith("xl/") else "xl/" + target
        self.cache = {}
        self.style_cache = {}
        styles = ET.fromstring(self.zf.read("xl/styles.xml"))
        fonts = styles.find("x:fonts", NS)
        cell_xfs = styles.find("x:cellXfs", NS)
        self.red_styles = set()
        for style_id, xf in enumerate(cell_xfs):
            font = fonts[int(xf.attrib.get("fontId", 0))]
            color = font.find("x:color", NS)
            rgb = color.attrib.get("rgb", "").upper() if color is not None else ""
            if rgb in {"FFFF0000", "FFC00000"}:
                self.red_styles.add(style_id)

    def sheet(self, name):
        if name in self.cache:
            return self.cache[name]
        root = ET.fromstring(self.zf.read(self.targets[name]))
        cells = {}
        styles = {}
        for cell in root.findall(".//x:sheetData/x:row/x:c", NS):
            ref = cell.attrib["r"]
            row = int(re.search(r"\d+", ref).group(0))
            col = col_number(ref)
            styles[(row, col)] = int(cell.attrib.get("s", 0))
            value = cell.find("x:v", NS)
            if value is None:
                continue
            raw = value.text or ""
            cells[(row, col)] = self.strings[int(raw)] if cell.attrib.get("t") == "s" else number(raw)
        self.cache[name] = cells
        self.style_cache[name] = styles
        return cells

    def is_red(self, sheet_name, row, col):
        self.sheet(sheet_name)
        return self.style_cache[sheet_name].get((row, col), 0) in self.red_styles


def workbook_path(source_dir, filename, temp_dir):
    path = source_dir / filename
    if path.suffix.lower() != ".xls":
        return path
    executable = shutil.which("libreoffice") or shutil.which("soffice")
    if not executable:
        raise RuntimeError("LibreOffice is required for the legacy .xls workbook")
    subprocess.run([executable, "--headless", "--convert-to", "xlsx", "--outdir", str(temp_dir), str(path)],
                   check=True, capture_output=True, text=True)
    return temp_dir / (path.stem + ".xlsx")


def label(cells, col):
    for row in (2, 3, 1, 4):
        value = cells.get((row, col))
        if value not in (None, ""):
            return str(value).strip()
    return f"COL{col:02d}"


def response_rows(cells):
    current = None
    result = []
    for row in range(1, min(max((r for r, _ in cells), default=0), 40) + 1):
        first = cells.get((row, 1))
        if norm(first) == "电阻":
            break
        current = gas(first) or current
        concentration = number(cells.get((row, 2)))
        if current and concentration in CONCS:
            result.append((row, current, concentration))
    return result


def red_channel_gases(book):
    """Return channel/gas pairs whose raw-sheet CH header uses a red font."""
    excluded = set()
    for sheet_name in book.targets:
        cells = book.sheet(sheet_name)
        gases = {gas_name for _, gas_name, _ in response_rows(cells)}
        if not gases:
            continue
        max_col = max((col for _, col in cells), default=0)
        for col in range(3, max_col + 1):
            channel = cells.get((3, col))
            if re.fullmatch(r"CH\d+\s*", str(channel or ""), re.IGNORECASE) and book.is_red(
                sheet_name, 3, col
            ):
                excluded.update((norm(channel), gas_name) for gas_name in gases)
    return excluded


def raw_r0(book):
    values = defaultdict(list)
    for sheet_name in book.targets:
        cells = book.sheet(sheet_name)
        marker = next((row for row in range(1, max((r for r, _ in cells), default=0) + 1)
                       if norm(cells.get((row, 1))) == "电阻"), None)
        if marker is None:
            continue
        max_col = max((c for _, c in cells), default=0)
        max_row = max((r for r, _ in cells), default=0)
        for col in range(3, max_col + 1):
            if book.is_red(sheet_name, 3, col):
                continue
            series = []
            for row in range(marker + 1, max_row + 1):
                value = number(cells.get((row, col)))
                if value is not None and 10 <= value <= 1e9:
                    series.append(value)
            if len(series) < 10:
                continue
            for header_row in (2, 3):
                header = cells.get((header_row, col))
                if header not in (None, ""):
                    values[norm(header)].extend(series)
    return {key: float(np.percentile(series, 95)) for key, series in values.items() if len(series) >= 10}


def fit(points):
    valid = sorted((c, r) for c, r in points.items() if c > 0 and 0 < r < 1.0)
    if len(valid) < 3:
        return None
    concentrations = np.asarray([point[0] for point in valid], dtype=float)
    rs_rair = np.asarray([point[1] for point in valid], dtype=float)
    excess_conductance = 1.0 / rs_rair - 1.0
    initial_n, initial_log_k = np.polyfit(
        np.log(concentrations), np.log(excess_conductance), 1
    )
    try:
        (coefficient_k, exponent_n), _ = curve_fit(
            lambda concentration, k, n: 1.0 / (1.0 + k * concentration**n),
            concentrations,
            rs_rair,
            p0=(float(np.exp(initial_log_k)), float(initial_n)),
            sigma=rs_rair,
            bounds=((0.0, 0.0), (np.inf, 5.0)),
            maxfev=20_000,
        )
    except (RuntimeError, ValueError):
        return None
    fitted = 1.0 / (1.0 + coefficient_k * concentrations**exponent_n)
    return {
        "coefficient_k": float(coefficient_k),
        "exponent_n": float(exponent_n),
        "rmse_rs_rair": float(np.sqrt(np.mean((fitted - rs_rair) ** 2))),
        "n_points": len(valid),
    }


def extract(source_dir, temp_dir):
    books, r0s, red_pairs, channels, exclusions = {}, {}, {}, {}, []
    for table in TABLES:
        if table.source not in books:
            books[table.source] = Book(workbook_path(source_dir, table.file, temp_dir))
            r0s[table.source] = raw_r0(books[table.source])
            red_pairs[table.source] = red_channel_gases(books[table.source])
            exclusions.extend(
                f"{table.source}/{channel}/{gas_name}: red raw-sheet channel excluded"
                for channel, gas_name in sorted(red_pairs[table.source])
            )
        book = books[table.source]
        if table.sheet not in book.targets:
            exclusions.append(f"{table.source}/{table.sheet}: missing sheet")
            continue
        cells = book.sheet(table.sheet)
        rows = response_rows(cells)
        if not rows:
            exclusions.append(f"{table.source}/{table.sheet}: no processed response table")
            continue
        for col in range(table.first_col, table.last_col + 1):
            channel = label(cells, col)
            channel_id = f"{table.family}_{table.source}_{re.sub('[^A-Za-z0-9]+', '_', channel).strip('_').upper()}"
            points = defaultdict(dict)
            for row, gas_name, concentration in rows:
                if (norm(channel), gas_name) in red_pairs[table.source]:
                    continue
                response = number(cells.get((row, col)))
                location = f"{table.source}/{table.sheet}!R{row}C{col}"
                if response is None or response <= 0 or response > 1000:
                    exclusions.append(f"{location}: invalid Ra/Rg={response!r}")
                elif 1.0 / response > 1.25:
                    exclusions.append(f"{location}: implausible Rs/Rair={1.0/response:.5g}")
                else:
                    points[gas_name][concentration] = response
            measurements = {
                gas_name: {
                    str(int(concentration)): float(response)
                    for concentration, response in sorted(points[gas_name].items())
                }
                for gas_name in GASES
                if points[gas_name]
            }
            if not measurements:
                continue
            resistance = r0s[table.source].get(norm(channel))
            r0_source = "estimated: 95th percentile of workbook raw resistance trace"
            if resistance is None:
                resistance = 10000.0
                r0_source = "fallback: unmatched raw channel; RL=R0 keeps default voltage ratio-based"
            if channel_id in channels:
                channels[channel_id]["source_sheets"].append(table.sheet)
                channels[channel_id]["measurements_ra_rg"].update(measurements)
                continue
            channels[channel_id] = {
                "family": table.family, "source_id": table.source,
                "source_sheets": [table.sheet], "source_channel": channel,
                "r0_ohm": float(resistance), "r0_source": r0_source,
                "measurements_ra_rg": measurements,
            }
    for channel_id, channel_data in channels.items():
        for gas_name in sorted(set(GASES) - set(channel_data["measurements_ra_rg"])):
            if (norm(channel_data["source_channel"]), gas_name) not in red_pairs[channel_data["source_id"]]:
                exclusions.append(
                    f"{channel_id}/{gas_name}: no valid workbook response across source sheets"
                )
    return channels, exclusions


def nominal_profiles(channels):
    grouped = defaultdict(list)
    for channel in channels.values():
        grouped[channel["family"]].append(channel)
    result = {}
    for family, all_family_channels in grouped.items():
        available_sources = sorted(
            {channel["source_id"] for channel in all_family_channels}
        )
        selected_source_id = selected_source(family, available_sources)
        family_channels = [
            channel
            for channel in all_family_channels
            if channel["source_id"] == selected_source_id
        ]
        family_dynamics = {
            direction: [
                channel["dynamics"][gas_name][f"tau_{direction}_s"]
                for channel in family_channels
                for gas_name in channel.get("dynamics", {})
                if channel["dynamics"][gas_name].get(f"tau_{direction}_s") is not None
            ]
            for direction in ("rise", "decay")
        }
        curves = {}
        for gas_name in GASES:
            points = {}
            median_ra_rg_points = {}
            contributing_counts = {}
            for concentration in CONCS:
                key = str(int(concentration))
                values = [
                    channel["measurements_ra_rg"][gas_name][key]
                    for channel in family_channels
                    if gas_name in channel["measurements_ra_rg"]
                    and key in channel["measurements_ra_rg"][gas_name]
                ]
                if values:
                    median_ra_rg = float(median(values))
                    median_ra_rg_points[key] = median_ra_rg
                    points[concentration] = 1.0 / median_ra_rg
                    contributing_counts[key] = len(values)
            curve = fit(points)
            if curve:
                curve["runtime_supported"] = gas_name in RUNTIME_GASES
                curve["measured_points"] = {str(int(c)): float(r) for c, r in sorted(points.items())}
                curve["median_ra_rg_points"] = median_ra_rg_points
                curve["contributing_channel_counts"] = contributing_counts
                for direction, default_tau in zip(("rise", "decay"), DEFAULT_TAU):
                    gas_values = [
                        channel["dynamics"][gas_name][f"tau_{direction}_s"]
                        for channel in family_channels
                        if gas_name in channel.get("dynamics", {})
                        and channel["dynamics"][gas_name].get(
                            f"tau_{direction}_s"
                        )
                        is not None
                    ]
                    if len(gas_values) >= 2:
                        tau = float(median(gas_values))
                        tau_source_description = (
                            "workbook effective 10-90% median across "
                            "physical channels for this gas"
                        )
                        count = len(gas_values)
                    elif len(family_dynamics[direction]) >= 2:
                        tau = float(median(family_dynamics[direction]))
                        tau_source_description = (
                            "workbook effective 10-90% sensor-family median; "
                            "gas-specific transition coverage insufficient"
                        )
                        count = len(family_dynamics[direction])
                    else:
                        tau = default_tau
                        tau_source_description = (
                            "fallback: workbook transition coverage insufficient"
                        )
                        count = 0
                    curve[f"tau_{direction}_s"] = tau
                    curve[f"tau_{direction}_source"] = tau_source_description
                    curve[f"tau_{direction}_channel_count"] = count
                curves[gas_name] = curve
        tau_rise = float(median(
            curve["tau_rise_s"] for curve in curves.values()
        )) if curves else DEFAULT_TAU[0]
        tau_decay = float(median(
            curve["tau_decay_s"] for curve in curves.values()
        )) if curves else DEFAULT_TAU[1]
        result[family] = {
            "family": family, "kind": "nominal",
            "aggregation_method": (
                "source-selected adaptive late-air/gas-minimum ratio per channel; "
                "median Ra/Rg across channels at each concentration; fit once"
            ),
            "source_ids": [selected_source_id],
            "available_source_ids": available_sources,
            "source_selection": (
                "explicit CROSS override"
                if family in SOURCE_OVERRIDES
                else "MULTI, then CROSS, then PRECOAT priority"
            ),
            "replicate_count": len(family_channels),
            "r0_ohm": float(median(channel["r0_ohm"] for channel in family_channels)),
            "r0_source": "median physical-channel late-air reference", "sensitivity_air": 1.0,
            "tau_rise_s": tau_rise, "tau_decay_s": tau_decay,
            "tau_source": (
                "median of gas-specific workbook effective time constants; "
                "see each sensitivity curve for provenance"
            ),
            "concentration_range_ppm": [1.0, 100.0], "sensitivity": curves,
        }
        result[family]["load_resistance_ohm"] = 0.25 * result[family]["r0_ohm"]
        result[family]["load_resistance_source"] = (
            "experiment divider rule: RL=0.25*Rair_reference for 1 V in air at Vcc=5 V"
        )
    return result


def extract_windowed(source_dir: Path, temp_dir: Path):
    """Run the raw-trace window audit and return unfitted channel measurements."""
    audit_dir = temp_dir / "window_audit"
    command = [
        sys.executable,
        str(Path(__file__).with_name("audit_mox_ratio_windows.py")),
        "--source-dir",
        str(source_dir),
        "--output-dir",
        str(audit_dir),
        "--no-plots",
    ]
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(
            "adaptive resistance-window extraction failed:\n"
            + completed.stderr
            + completed.stdout
        )

    channels = {}
    with (audit_dir / "ratio_comparison.csv").open(
        "r", encoding="utf-8", newline=""
    ) as stream:
        for row in csv.DictReader(stream):
            channel = row["channel"].strip()
            channel_id = (
                f"{row['family']}_{row['source']}_"
                f"{re.sub('[^A-Za-z0-9]+', '_', channel).strip('_').upper()}"
            )
            record = channels.setdefault(
                channel_id,
                {
                    "family": row["family"],
                    "source_id": row["source"],
                    "source_sheets": [],
                    "source_channel": channel,
                    "_r_air_values": [],
                    "measurements_ra_rg": defaultdict(dict),
                    "_tau_values": {
                        "rise": defaultdict(list),
                        "decay": defaultdict(list),
                    },
                    "_tau_quality": {
                        "rise": defaultdict(lambda: defaultdict(int)),
                        "decay": defaultdict(lambda: defaultdict(int)),
                    },
                },
            )
            if row["sheet"] not in record["source_sheets"]:
                record["source_sheets"].append(row["sheet"])
            record["_r_air_values"].append(float(row["air_median_ohm"]))
            concentration = str(int(float(row["concentration_ppm"])))
            record["measurements_ra_rg"][row["gas"]][concentration] = float(
                row["window_ratio_ra_rg"]
            )
            for direction in ("rise", "decay"):
                quality = row[f"tau_{direction}_quality"]
                record["_tau_quality"][direction][row["gas"]][quality] += 1
                value = row[f"tau_{direction}_s"]
                if quality in {"good", "coarse"} and value:
                    record["_tau_values"][direction][row["gas"]].append(
                        float(value)
                    )

    for record in channels.values():
        r_air = float(median(record.pop("_r_air_values")))
        record["r0_ohm"] = r_air
        record["r0_source"] = (
            "median of adaptive late-air window medians across valid exposures"
        )
        record["load_resistance_ohm"] = 0.25 * r_air
        record["load_resistance_source"] = (
            "experiment divider rule: RL=0.25*Rair_reference for 1 V in air at Vcc=5 V"
        )
        record["measurements_ra_rg"] = {
            gas_name: dict(points)
            for gas_name, points in record["measurements_ra_rg"].items()
        }
        tau_values = record.pop("_tau_values")
        tau_quality = record.pop("_tau_quality")
        record["dynamics"] = {}
        for gas_name in GASES:
            dynamics = {}
            for direction in ("rise", "decay"):
                values = tau_values[direction][gas_name]
                if values:
                    dynamics[f"tau_{direction}_s"] = float(median(values))
                    dynamics[f"tau_{direction}_q25_s"] = float(
                        np.percentile(values, 25)
                    )
                    dynamics[f"tau_{direction}_q75_s"] = float(
                        np.percentile(values, 75)
                    )
                    dynamics[f"tau_{direction}_transition_count"] = len(values)
                dynamics[f"tau_{direction}_quality_counts"] = dict(
                    tau_quality[direction][gas_name]
                )
            record["dynamics"][gas_name] = dynamics
    return channels


def report(path, registry, exclusions, channel_count):
    models = registry["models"]
    lines = ["# MOX Calibration Import Report", "",
             "Physical-channel `Ra/Rg` values are median-aggregated at each concentration, "
             "then fitted as `Ra/Rg - 1 = K * concentration^n` once per nominal sensor.", "",
             "## Summary", "", f"- Nominal families: {len(registry['nominal_models'])}",
             f"- Unfitted physical-channel records: {channel_count}",
             f"- Stored gases: {', '.join(GASES)}",
             f"- Runtime-enabled gases: {', '.join(sorted(RUNTIME_GASES))}",
             "- Source priority: MULTI, then CROSS, then PRECOAT",
             "- CROSS overrides: TGS2610, TGS2611, TGS2600",
             "- Calibrated range: 1-100 ppm; conductance law approaches clean air below it and caps above it",
             "- Dynamics: effective tau=(t90-t10)/ln(9), median per physical channel then across channels",
             "- Coarse timing warning: workbook cadence is approximately 9-14 seconds",
             "- Absolute resistance: median adaptive late-air baseline",
             "- Load resistor: RL=0.25*Rair_reference (1 V in air at Vcc=5 V)",
             "", "## Nominal Fits", "",
             "| Model | Source | Replicates | R0 ohm | Gas | K | n | Rs/Rair RMSE | Runtime |",
             "|---|---|---:|---:|---|---:|---:|---:|:---:|"]
    for name in registry["nominal_models"]:
        p = models[name]
        for gas_name, curve in p["sensitivity"].items():
            lines.append(f"| {name} | {p['source_ids'][0]} | {p['replicate_count']} | {p['r0_ohm']:.2f} | {gas_name} | {curve['coefficient_k']:.6g} | {curve['exponent_n']:.6g} | {curve['rmse_rs_rair']:.5f} | {'yes' if curve['runtime_supported'] else 'no'} |")
    lines += [
        "",
        "## Effective Dynamics",
        "",
        "These include chamber filling/purging and are not intrinsic sensor-only constants.",
        "",
        "| Model | Gas | Rise tau s | Rise channels | Recovery tau s | Recovery channels |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for name in registry["nominal_models"]:
        for gas_name, curve in models[name]["sensitivity"].items():
            lines.append(
                f"| {name} | {gas_name} | {curve['tau_rise_s']:.3f} | "
                f"{curve['tau_rise_channel_count']} | "
                f"{curve['tau_decay_s']:.3f} | "
                f"{curve['tau_decay_channel_count']} |"
            )
    lines += ["", "## Quality Exclusions", ""] + [f"- {item}" for item in exclusions]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("odor_sim/config/mox_sensors.yaml"))
    parser.add_argument("--report", type=Path, default=Path("docs/mox_calibration_report.md"))
    parser.add_argument(
        "--provenance-output",
        type=Path,
        default=Path("analysis/mox_calibration_provenance.yaml"),
        help="generated physical-channel audit data (not loaded at runtime)",
    )
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="odorsim_mox_") as temp:
        temp_path = Path(temp)
        _, exclusions = extract(args.source_dir, temp_path)
        channels = extract_windowed(args.source_dir, temp_path)
    nominal = nominal_profiles(channels)
    preferred = ("TGS2620", "TGS2600", "TGS2611", "TGS2610", "TGS2602", "101", "203", "302", "401", "509", "104", "303", "500-1", "43_SINTERED", "43_UNSINTERED", "601", "701")
    order = [name for name in preferred if name in nominal] + sorted(set(nominal) - set(preferred))
    models = {name: nominal[name] for name in order}
    registry = {
        "schema_version": 4,
        "response_definition": (
            "Ra/Rg = 1 + K * concentration_ppm^n; "
            "Rs/Rair = 1 / (1 + K * concentration_ppm^n)"
        ),
        "aggregation_domain": "Ra/Rg",
        "aggregation_method": (
            "source-selected adaptive late-air/gas-minimum ratio per channel; "
            "median physical-channel Ra/Rg at each concentration; fit once"
        ),
        "source_selection_policy": {
            "priority": list(SOURCE_PRIORITY),
            "overrides": SOURCE_OVERRIDES,
        },
        "runtime_supported_gases": sorted(RUNTIME_GASES),
        "nominal_models": order,
        "models": models,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "# Generated by tools/build_mox_registry.py; do not edit fitted values by hand.\n"
        + yaml.safe_dump(registry, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    args.provenance_output.parent.mkdir(parents=True, exist_ok=True)
    args.provenance_output.write_text(
        "# Generated calibration provenance; not loaded by OdorSim at runtime.\n"
        + yaml.safe_dump(
            {
                "schema_version": 1,
                "channel_measurements": {
                    name: channels[name] for name in sorted(channels)
                },
            },
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    report(args.report, registry, exclusions, len(channels))
    print(f"Wrote {len(order)} median-first nominal fits and "
          f"{len(channels)} unfitted channel records; "
          f"{len(exclusions)} exclusions")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Track all six strength-1 GADEN filament clouds with exactly zero wind."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import rclpy
from matplotlib.backends.backend_pdf import PdfPages
from visualization_msgs.msg import Marker

from odor_sim.bridge.gaden_bridge import GadenBridge
from odor_sim.config.frame_map import FrameMap
from odor_sim.envs.odor_profile import OdorProfile
from odor_sim.envs.registry import resolve_scenario
from odor_sim.envs.scene_builder import SceneBuilder, _gas_color
from odor_sim.odor_encoder.dataset import GAS_ORDER
from odor_sim.odor_encoder.wind import WindProfile, write_isolated_config
from odor_sim.runtime.gaden_server import GadenServerManager


PLOT_COLORS = {
    "ethanol": "#009E73",
    "methanol": "#0072B2",
    "acetone": "#E69F00",
    "acetic_acid": "#CC79A7",
    "ethyl_acetate": "#D55E00",
    "toluene": "#4D4D4D",
}


def _gas_object(gas: str):
    return SimpleNamespace(
        object_id=f"zero_wind_{gas}",
        odor_profile=OdorProfile.from_spec([{"gas_type": gas, "strength": 1.0}]),
    )


def _color_key(color) -> tuple[float, float, float]:
    return (float(color.r), float(color.g), float(color.b))


def _nearest_gas(color, expected: dict[str, np.ndarray]) -> str:
    rgb = np.asarray(_color_key(color), float)
    return min(expected, key=lambda gas: float(np.linalg.norm(rgb - expected[gas])))


def _percentile_or_nan(values: list[float], q: float) -> float:
    return float(np.percentile(values, q)) if values else float("nan")


def run(args: argparse.Namespace) -> dict:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    source_config = resolve_scenario(args.scenario, args.scenario_config)
    config_dir = output_dir / "runtime_config"
    zero_profile = WindProfile(
        regime="zero",
        vectors_m_s=np.zeros((max(2, int(args.duration_s) + 2), 3), np.float32),
        nominal_direction=np.asarray([1.0, 0.0, 0.0], np.float32),
    )
    write_isolated_config(source_config, config_dir, zero_profile)

    builder = SceneBuilder(frame_map=FrameMap.from_scenario(source_config))
    for gas in GAS_ORDER:
        builder.add_object(_gas_object(gas))
    source_position = np.asarray([args.source_x, args.source_y, args.source_z], float)
    source_positions = np.repeat(source_position[None], len(GAS_ORDER), axis=0)
    builder.export_gaden_scene(
        config_dir,
        scene_id=args.scene_id,
        initial_positions_gaden=source_positions,
        buoyancy_tau_s=args.buoyancy_tau_s,
        filament_noise_std=args.filament_noise_std,
    )

    expected_colors = {
        gas: np.asarray(_gas_color(builder.sources[index].gas_type), float)
        for index, gas in enumerate(GAS_ORDER)
    }
    server = GadenServerManager(
        config_dir,
        scene_id=args.scene_id,
        publish_markers=True,
        log_dir=args.server_log_dir,
    )
    bridge = None
    marker_count = 0
    latest: Marker | None = None

    def on_marker(message: Marker) -> None:
        nonlocal marker_count, latest
        marker_count += 1
        latest = message

    frames: dict[str, list[dict]] = {gas: [] for gas in GAS_ORDER}
    oldest_cohort_size: dict[str, int] = {gas: 0 for gas in GAS_ORDER}
    try:
        server.start(timeout=args.wait_timeout, kill_stale=True)
        bridge = GadenBridge()
        if not bridge.wait_for_server(args.wait_timeout):
            raise RuntimeError("GADEN /odor_value service unavailable")
        bridge.node.create_subscription(
            Marker, "/gaden/filament_visualization", on_marker, 10
        )
        bridge.publish_source_poses(source_positions)

        steps = int(round(args.duration_s / args.delta_time))
        for step in range(steps):
            before = marker_count
            if not bridge.step():
                raise RuntimeError(f"GADEN step {step} was not confirmed")
            if not bridge._spin_until(lambda: marker_count > before, bridge.step_timeout):
                raise RuntimeError(f"filament marker missing after step {step}")
            assert latest is not None
            grouped: dict[str, list[float]] = {gas: [] for gas in GAS_ORDER}
            for point, color in zip(latest.points, latest.colors):
                grouped[_nearest_gas(color, expected_colors)].append(float(point.z))
            time_s = (step + 1) * args.delta_time
            for gas in GAS_ORDER:
                values = grouped[gas]
                if oldest_cohort_size[gas] == 0 and values:
                    oldest_cohort_size[gas] = len(values)
                cohort = values[:oldest_cohort_size[gas]]
                frames[gas].append(
                    {
                        "time_s": time_s,
                        "count": len(values),
                        "z": values,
                        "z_min": min(values) if values else None,
                        "z_p10": _percentile_or_nan(values, 10),
                        "z_median": _percentile_or_nan(values, 50),
                        "z_p90": _percentile_or_nan(values, 90),
                        "z_max": max(values) if values else None,
                        "oldest_cohort_z": cohort,
                        "oldest_cohort_z_median": _percentile_or_nan(cohort, 50),
                    }
                )
            if (step + 1) % max(1, int(round(1.0 / args.delta_time))) == 0:
                print(f"[zero-wind] {time_s:.1f}/{args.duration_s:.1f} s")
    finally:
        if bridge is not None:
            bridge.close()
        server.stop()
        if rclpy.ok():
            rclpy.shutdown()

    summary = {}
    for gas in GAS_ORDER:
        final = frames[gas][-1]
        all_z = [z for frame in frames[gas] for z in frame["z"]]
        summary[gas] = {
            "final_active_filaments": final["count"],
            "final_z_min": final["z_min"],
            "final_z_median": final["z_median"],
            "final_z_p90": final["z_p90"],
            "lowest_observed_z": min(all_z) if all_z else None,
            "ever_above_source_fraction": (
                float(np.mean(np.asarray(all_z) > args.source_z)) if all_z else None
            ),
            "ever_below_source_fraction": (
                float(np.mean(np.asarray(all_z) < args.source_z)) if all_z else None
            ),
        }

    result = {
        "description": "Actual GADEN runtime filament markers under exactly zero wind.",
        "source_position_gaden_m": source_position.tolist(),
        "source_strength": 1.0,
        "filament_ppm_center_at_birth": 40.0,
        "filaments_per_second": 40,
        "delta_time_s": args.delta_time,
        "duration_s": args.duration_s,
        "buoyancy_tau_s": args.buoyancy_tau_s,
        "filament_noise_std_configured": args.filament_noise_std,
        "oldest_cohort_size_by_gas": oldest_cohort_size,
        "wind_m_s": [0.0, 0.0, 0.0],
        "summary": summary,
        "frames": frames,
    }
    (output_dir / "zero_wind_filament_fall.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    _plot(frames, args, output_dir)
    print(json.dumps(summary, indent=2))
    return result


def _plot(frames: dict[str, list[dict]], args, output_dir: Path) -> None:
    pdf_path = output_dir / "zero_wind_filament_fall.pdf"
    with PdfPages(pdf_path) as pdf:
        fig, axes = plt.subplots(2, 3, figsize=(13, 8), sharex=True, sharey=True)
        for axis, gas in zip(axes.flat, GAS_ORDER):
            rows = frames[gas]
            for row in rows:
                if row["z"]:
                    axis.scatter(
                        np.full(len(row["z"]), row["time_s"]), row["z"],
                        s=1.2, color=PLOT_COLORS[gas], alpha=0.035, rasterized=True,
                    )
            time = np.asarray([row["time_s"] for row in rows])
            low = np.asarray([row["z_p10"] for row in rows])
            median = np.asarray([row["z_median"] for row in rows])
            high = np.asarray([row["z_p90"] for row in rows])
            axis.fill_between(time, low, high, color=PLOT_COLORS[gas], alpha=0.20, label="10-90%")
            axis.plot(time, median, color=PLOT_COLORS[gas], lw=2, label="median")
            axis.axhline(args.source_z, color="black", ls="--", lw=1, label="source z")
            axis.set_title(gas.replace("_", " "))
            axis.grid(alpha=0.22)
        axes[0, 0].legend(fontsize=8, loc="lower left")
        for axis in axes[-1]:
            axis.set_xlabel("simulation time (s)")
        for axis in axes[:, 0]:
            axis.set_ylabel("filament-center z (m)")
        fig.suptitle(
            "Actual GADEN filament centers: zero wind, strength 1\n"
            "points are all active filaments; source continuously emits 40/s"
        )
        fig.tight_layout()
        pdf.savefig(fig, dpi=180)
        fig.savefig(output_dir / "zero_wind_filament_fall.png", dpi=180)
        plt.close(fig)

        fig, axis = plt.subplots(figsize=(11, 6))
        for gas in GAS_ORDER:
            rows = frames[gas]
            axis.plot(
                [row["time_s"] for row in rows],
                [row["z_median"] for row in rows],
                color=PLOT_COLORS[gas], lw=2, label=gas.replace("_", " "),
            )
        axis.axhline(args.source_z, color="black", ls="--", lw=1.2, label="source z")
        axis.set_xlabel("simulation time (s)")
        axis.set_ylabel("median active-filament z (m)")
        axis.set_title("Median vertical location by gas")
        axis.grid(alpha=0.25)
        axis.legend(ncol=2)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)
    print(f"wrote {pdf_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration-s", type=float, default=5.0)
    parser.add_argument("--delta-time", type=float, default=0.05)
    parser.add_argument("--buoyancy-tau-s", type=float, default=1.0)
    parser.add_argument("--filament-noise-std", type=float, default=0.02)
    parser.add_argument("--source-x", type=float, default=2.0)
    parser.add_argument("--source-y", type=float, default=3.0)
    parser.add_argument("--source-z", type=float, default=0.57)
    parser.add_argument("--scenario", default="10x6_uniform")
    parser.add_argument("--scenario-config", default="config1")
    parser.add_argument("--scene-id", default="zero_wind_fall")
    parser.add_argument("--wait-timeout", type=float, default=60.0)
    parser.add_argument("--server-log-dir", default=".roslog/zero_wind_filament_fall")
    parser.add_argument(
        "--output-dir", default="analysis/odor_encoder/zero_wind_filament_fall"
    )
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())

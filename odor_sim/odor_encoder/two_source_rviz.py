"""Visualize two continuously emitting GADEN gas sources live in RViz."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np

from odor_sim.config.frame_map import FrameMap
from odor_sim.envs.registry import resolve_scenario
from odor_sim.envs.scene_builder import SceneBuilder
from odor_sim.odor_encoder.dataset import GAS_ORDER, MixtureSpec
from odor_sim.odor_encoder.phase_b import (
    _load_local_ppm_calibration,
    _map_desired_local_to_nominal,
    _object,
)
from odor_sim.odor_encoder.wind import (
    WIND_REGIMES,
    WindProfile,
    sample_wind_profile,
    write_isolated_config,
    zero_wind_profile,
)
from odor_sim.odor_encoder.wind_rviz import _VisualizationServer, _stop_process


_REPO_ROOT = Path(__file__).resolve().parents[2]
_RVIZ_CONFIG = Path(__file__).with_name("two_source_filaments.rviz")
_SCENE_ID = "two_source_demo"
_DEFAULT_CALIBRATION = (
    _REPO_ROOT / "odor_sim/config/phase_b_local_ppm_exp0p5_scale0p06_fps15.json"
)

# Centers of the ClassifyLiquidPlaceOVLA cup slots in robosuite coordinates.
_OVLA_LEFT_CUP = np.asarray([0.16, -0.12, 0.87], dtype=float)
_OVLA_RIGHT_CUP = np.asarray([0.16, 0.12, 0.87], dtype=float)


def _desired_source(gas: str, ppm: float) -> MixtureSpec:
    values = np.zeros(len(GAS_ORDER), np.float32)
    values[GAS_ORDER.index(gas)] = ppm
    return MixtureSpec(values, "two_source_rviz")


def _crosswind_profile(profile: WindProfile, positions: np.ndarray, seed: int) -> WindProfile:
    """Rotate mean horizontal flow perpendicular to the cup-to-cup axis."""
    axis = np.asarray(positions[1, :2] - positions[0, :2], dtype=float)
    axis /= max(np.linalg.norm(axis), 1e-12)
    target = np.asarray([-axis[1], axis[0]], dtype=float)
    if seed % 2:
        target *= -1.0
    current = np.asarray(profile.nominal_direction[:2], dtype=float)
    current /= max(np.linalg.norm(current), 1e-12)
    angle = np.arctan2(target[1], target[0]) - np.arctan2(current[1], current[0])
    rotation = np.asarray([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)],
    ])
    vectors = np.asarray(profile.vectors_m_s, dtype=float).copy()
    vectors[:, :2] = vectors[:, :2] @ rotation.T
    return WindProfile(
        profile.regime, vectors.astype(np.float32),
        np.asarray([target[0], target[1], 0.0], np.float32),
    )


def build_project(args: argparse.Namespace) -> tuple[Path, np.ndarray, dict]:
    """Create a calibrated two-source project in the OVLA cup layout."""
    source_config = resolve_scenario(args.scenario, args.scenario_config)
    output = Path(args.out_dir).expanduser()
    if not output.is_absolute():
        output = _REPO_ROOT / output
    run_name = f"{args.regime}_{args.gas_a}_{args.gas_b}"
    config_dir = output / run_name / "environment_configurations" / "config1"

    frame_map = FrameMap.from_scenario(source_config)
    if args.layout == "ovla":
        positions_robot = np.stack((_OVLA_LEFT_CUP, _OVLA_RIGHT_CUP))
        positions = np.stack([
            frame_map.robosuite_to_gaden(position) for position in positions_robot
        ])
    else:
        center = np.asarray([args.center_x, args.center_y, args.source_z], dtype=float)
        direction = np.asarray([
            np.cos(np.deg2rad(args.axis_degrees)),
            np.sin(np.deg2rad(args.axis_degrees)),
            0.0,
        ])
        positions = np.stack((
            center - 0.5 * args.spacing_m * direction,
            center + 0.5 * args.spacing_m * direction,
        ))
        positions_robot = np.stack([
            frame_map.gaden_to_robosuite(position) for position in positions
        ])

    if args.regime == "zero":
        profile = zero_wind_profile(args.wind_duration_s)
    else:
        profile = sample_wind_profile(
            np.random.default_rng(args.seed), args.wind_duration_s, args.regime,
        )
        if args.wind_orientation == "crosswind":
            profile = _crosswind_profile(profile, positions, args.seed)
        profile = WindProfile(
            profile.regime,
            profile.vectors_m_s * args.wind_speed_scale,
            profile.nominal_direction,
        )
    write_isolated_config(source_config, config_dir, profile)

    calibration = _load_local_ppm_calibration(args.local_ppm_calibration)
    if calibration is None:
        raise ValueError("--local-ppm-calibration is required")
    emission = calibration["emission_configuration"]
    desired = (
        _desired_source(args.gas_a, args.local_ppm_a),
        _desired_source(args.gas_b, args.local_ppm_b),
    )
    nominal = tuple(
        _map_desired_local_to_nominal(item, calibration, allow_extrapolation=True)
        for item in desired
    )

    builder = SceneBuilder(frame_map=frame_map)
    for name, source in zip(("source_a", "source_b"), nominal):
        builder.add_object(_object(
            name, source, model="separated",
            emission_scale=float(emission["emission_scale"]),
            concentration_exponent=float(
                emission["emission_concentration_exponent"]
            ),
            filaments_per_second=int(emission["filaments_per_second"]),
        ))
    builder.export_gaden_scene(
        config_dir, scene_id=_SCENE_ID, initial_positions_gaden=positions,
        buoyancy_tau_s=args.buoyancy_tau_s,
        original_terminal_velocity_enabled=args.original_terminal_velocity,
    )
    details = {
        "layout": args.layout,
        "positions_robosuite_m": positions_robot.tolist(),
        "positions_gaden_m": positions.tolist(),
        "desired_local_ppm": [args.local_ppm_a, args.local_ppm_b],
        "source_nominal_ppm": [
            float(source.concentration_ppm.max()) for source in nominal
        ],
        "emission_configuration": emission,
        "local_ppm_calibration": str(Path(args.local_ppm_calibration).resolve()),
    }
    return config_dir, positions, details


def run(args: argparse.Namespace) -> int:
    args.local_ppm_a = args.local_ppm if args.local_ppm_a is None else args.local_ppm_a
    args.local_ppm_b = args.local_ppm if args.local_ppm_b is None else args.local_ppm_b
    if args.local_ppm_a <= 0 or args.local_ppm_b <= 0:
        raise SystemExit("desired local ppm values must be positive")
    if args.source_start_delay_s < 0 or args.plot_rate_hz <= 0:
        raise SystemExit("startup delay and plot rate are invalid")
    if not 0.05 <= args.spacing_m <= 2.0:
        raise SystemExit("--spacing-m must be between 0.05 and 2.0")
    if args.buoyancy_tau_s < 0:
        raise SystemExit("--buoyancy-tau-s must be non-negative")
    if args.wind_speed_scale <= 0 or args.arrow_scale <= 0:
        raise SystemExit("wind and arrow scales must be positive")
    if args.original_terminal_velocity and args.buoyancy_tau_s > 0:
        raise SystemExit(
            "legacy terminal velocity and buoyancy acceleration are mutually exclusive"
        )

    config_dir, positions, details = build_project(args)
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    server = _VisualizationServer(
        config_dir,
        scene_id=_SCENE_ID,
        step_on_timer=False,
        publish_markers=True,
        log_dir=log_dir / "server",
    )

    marker_process = None
    marker_log = None
    monitor_process = None
    monitor_log = None
    rviz = None
    original_ros_domain = os.environ.get("ROS_DOMAIN_ID")
    try:
        os.environ["ROS_DOMAIN_ID"] = str(args.ros_domain_id)
        server.start(timeout=args.wait_timeout, kill_stale=False)
        details["startup"] = {
            "initial_filament_count": 0,
            "source_activation_delay_s": args.source_start_delay_s,
            "lockstep_rate_hz": args.plot_rate_hz,
        }
        report_path = config_dir.parents[1] / "source_position_ppm.json"
        report_path.write_text(json.dumps(details, indent=2) + "\n")
        marker_log = open(log_dir / f"markers_{args.regime}.log", "w")
        marker_command = [
            os.sys.executable,
            "-m",
            "odor_sim.odor_encoder.wind_markers",
            "--wind-file",
            str(config_dir / "wind" / "wind_0.csv"),
            "--regime",
            args.regime,
            "--tabletop",
            "--table-center-x", "2.0",
            "--table-center-y", "3.0",
            "--tabletop-z", "0.8",
            "--table-size-x", "0.8",
            "--table-size-y", "0.8",
            "--arrow-scale", str(args.arrow_scale),
        ]
        sensor_position = np.asarray(
            positions[0] if args.sensor_cup == "a" else positions[1], dtype=float,
        ).copy()
        if args.sensor_cup == "midpoint":
            sensor_position = positions.mean(axis=0)
        sensor_position[2] += args.sensor_height_m
        for label, gas, position in zip(
            ("A", "B"), (args.gas_a, args.gas_b), positions,
        ):
            marker_command.extend([
                "--source-label",
                f"{label}: {gas}",
                *(f"{coordinate:.6f}" for coordinate in position),
            ])
        marker_command.extend([
            "--sensor-position",
            *(f"{coordinate:.6f}" for coordinate in sensor_position),
        ])
        marker_process = subprocess.Popen(
            marker_command,
            stdout=marker_log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

        child_environment = dict(os.environ)
        for variable in ("QT_QPA_PLATFORM_PLUGIN_PATH", "QT_QPA_FONTDIR"):
            if "site-packages/cv2" in child_environment.get(variable, ""):
                child_environment.pop(variable, None)
        monitor_command = [
            os.sys.executable, "-m", "odor_sim.odor_encoder.two_source_live_monitor",
            "--gas-a", args.gas_a,
            "--gas-b", args.gas_b,
            "--sensor-position", *(f"{value:.6f}" for value in sensor_position),
            "--rate-hz", str(args.plot_rate_hz),
            "--history-s", str(args.plot_history_s),
            "--wait-timeout", str(args.wait_timeout),
            "--startup-delay-s", str(args.source_start_delay_s),
        ]
        for position in positions:
            monitor_command.extend([
                "--source-position", *(f"{value:.6f}" for value in position),
            ])
        if args.headless:
            monitor_command.append("--no-plots")
        monitor_log = open(log_dir / f"live_monitor_{args.regime}.log", "w")
        monitor_process = subprocess.Popen(
            monitor_command,
            env=child_environment,
            stdout=monitor_log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

        vectors = np.loadtxt(
            config_dir / "wind" / "wind_0.csv", delimiter=",", ndmin=2,
        )
        speed = np.linalg.norm(vectors, axis=1)
        print(
            f"[two-source-rviz] A={args.gas_a} B={args.gas_b} "
            f"spacing={np.linalg.norm(positions[1] - positions[0]):.3f} m"
        )
        print(
            f"[two-source-rviz] regime={args.regime} speed min/mean/max="
            f"{speed.min():.4f}/{speed.mean():.4f}/{speed.max():.4f} m/s"
        )
        print(f"[two-source-rviz] source positions GADEN: {positions.tolist()}")
        print(
            f"[two-source-rviz] e-nose={args.sensor_cup} position="
            f"{sensor_position.tolist()}"
        )
        print(
            "[two-source-rviz] desired local ppm: "
            f"A={args.local_ppm_a:g}, B={args.local_ppm_b:g}"
        )
        print(
            "[two-source-rviz] starts with zero filaments; emission begins after "
            f"{args.source_start_delay_s:g} s"
        )
        print(f"[two-source-rviz] ppm report: {report_path}")
        print("[two-source-rviz] colored filaments: /gaden/filament_visualization")
        print("[two-source-rviz] live ppm and eight-channel MOX plots started")
        if args.headless:
            time.sleep(args.headless_seconds)
            return 0

        rviz = subprocess.Popen(
            ["rviz2", "-d", str(_RVIZ_CONFIG)],
            env=child_environment,
            start_new_session=True,
        )
        return rviz.wait()
    except KeyboardInterrupt:
        return 130
    finally:
        _stop_process(rviz)
        _stop_process(monitor_process)
        _stop_process(marker_process)
        if monitor_log is not None:
            monitor_log.close()
        if marker_log is not None:
            marker_log.close()
        server.stop()
        if original_ros_domain is None:
            os.environ.pop("ROS_DOMAIN_ID", None)
        else:
            os.environ["ROS_DOMAIN_ID"] = original_ros_domain


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gas-a", choices=GAS_ORDER, default="ethanol")
    parser.add_argument("--gas-b", choices=GAS_ORDER, default="ethyl_acetate")
    parser.add_argument(
        "--local-ppm", type=float, default=50.0,
        help="desired local calibration condition for both sources (ppm)",
    )
    parser.add_argument("--local-ppm-a", type=float)
    parser.add_argument("--local-ppm-b", type=float)
    parser.add_argument(
        "--local-ppm-calibration", default=str(_DEFAULT_CALIBRATION),
    )
    parser.add_argument(
        "--layout", choices=("ovla", "custom"), default="ovla",
        help="ovla uses the centers of the task left/right cup slots",
    )
    parser.add_argument("--spacing-m", type=float, default=0.25)
    parser.add_argument("--source-z", type=float, default=0.80)
    parser.add_argument("--center-x", type=float, default=5.0)
    parser.add_argument("--center-y", type=float, default=3.0)
    parser.add_argument("--axis-degrees", type=float, default=0.0)
    parser.add_argument(
        "--regime", choices=("zero", *WIND_REGIMES), default="stagnant_correlated",
    )
    parser.add_argument(
        "--wind-speed-scale", type=float, default=1.5,
        help="visualization-only multiplier applied to the sampled wind field",
    )
    parser.add_argument("--arrow-scale", type=float, default=1.35)
    parser.add_argument(
        "--wind-orientation", choices=("crosswind", "random"), default="crosswind",
        help="crosswind makes mean flow perpendicular to the cup-to-cup axis",
    )
    parser.add_argument(
        "--buoyancy-tau-s", type=float, default=1.0,
        help="experimental concentration-dependent buoyancy relaxation time",
    )
    parser.add_argument(
        "--original-terminal-velocity",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "enable the historical GADEN terminal-velocity expression; "
            "default: disabled"
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wind-duration-s", type=float, default=300.0)
    parser.add_argument("--scenario", default="10x6_uniform")
    parser.add_argument("--scenario-config", default="config1")
    parser.add_argument(
        "--out-dir", default="analysis/gaden_two_source_examples",
    )
    parser.add_argument("--log-dir", default=".roslog/two_source_rviz")
    parser.add_argument("--wait-timeout", type=float, default=60.0)
    parser.add_argument("--plot-rate-hz", type=float, default=20.0)
    parser.add_argument("--plot-history-s", type=float, default=120.0)
    parser.add_argument("--sensor-cup", choices=("a", "b", "midpoint"), default="a")
    parser.add_argument("--sensor-height-m", type=float, default=0.03)
    parser.add_argument(
        "--source-start-delay-s", type=float, default=3.0,
        help="wall-clock pause showing the empty scene before lockstep emission starts",
    )
    parser.add_argument("--ros-domain-id", type=int, default=78)
    parser.add_argument("--headless", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--headless-seconds", type=float, default=3.0, help=argparse.SUPPRESS,
    )
    return parser


if __name__ == "__main__":
    raise SystemExit(run(build_parser().parse_args()))

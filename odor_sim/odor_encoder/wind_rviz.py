"""Run one generated GADEN wind example live in RViz."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from odor_sim.odor_encoder.wind import WIND_REGIMES
from odor_sim.odor_encoder.wind_example import build_example, build_parser as example_parser
from odor_sim.runtime.gaden_server import GadenServerManager

_RVIZ_CONFIG = Path(__file__).with_name("wind_filaments.rviz")
_REPO_ROOT = Path(__file__).resolve().parents[2]


class _VisualizationServer(GadenServerManager):
    """Directly launch an alias not matched by legacy global cleanup code."""

    def _command(self) -> list[str]:
        installed = _REPO_ROOT / "ros2_ws/install/odor_gaden_rt/lib/odor_gaden_rt/rt_server"
        if not installed.is_file():
            raise FileNotFoundError(f"odor_gaden_rt executable not found: {installed}")
        alias = _REPO_ROOT / ".runtime/wind_rviz/visualization_server"
        alias.parent.mkdir(parents=True, exist_ok=True)
        if alias.is_symlink() and alias.resolve() != installed.resolve():
            alias.unlink()
        if not alias.exists():
            alias.symlink_to(installed)
        command = [
            str(alias), "--ros-args",
            "-p", f"scenarioPath:={self.scenario_path}",
            "-p", f"sceneID:={self.scene_id}",
            "-p", f"stepOnTimer:={'true' if self.step_on_timer else 'false'}",
            "-p", f"publishMarkers:={'true' if self.publish_markers else 'false'}",
        ]
        command.extend(self.extra_ros_args)
        return command


def _stop_process(process: subprocess.Popen | None) -> None:
    if process is None or process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=5)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=2)


def run(args: argparse.Namespace) -> int:
    example_args = example_parser().parse_args([
        "--regime", args.regime,
        "--duration-s", str(args.wind_duration_s),
        "--seed", str(args.seed),
        "--gas", args.gas,
        "--strength", str(args.strength),
        "--out-dir", args.out_dir,
    ])
    example_args.print_gui_command = False
    project = build_example(example_args)
    config_dir = project.parent / "environment_configurations" / "config1"
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    server = _VisualizationServer(
        config_dir, scene_id="wind_demo", step_on_timer=True,
        publish_markers=True, log_dir=log_dir / "server",
    )
    markers = None
    rviz = None
    marker_log = None
    original_ros_domain = os.environ.get("ROS_DOMAIN_ID")
    try:
        # Keep visualization isolated from an active Phase B collector. All
        # children inherit this domain; the caller's shell is not modified.
        os.environ["ROS_DOMAIN_ID"] = str(args.ros_domain_id)
        server.start(timeout=args.wait_timeout, kill_stale=False)
        marker_log = open(log_dir / f"markers_{args.regime}.log", "w")
        markers = subprocess.Popen(
            [
                sys.executable, "-m", "odor_sim.odor_encoder.wind_markers",
                "--wind-file", str(config_dir / "wind" / "wind_0.csv"),
                "--regime", args.regime,
            ],
            stdout=marker_log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        print(f"[wind-rviz] regime={args.regime} gas={args.gas}")
        print("[wind-rviz] filaments: /gaden/filament_visualization")
        print("[wind-rviz] source:    /gaden/source_visualization")
        print("[wind-rviz] wind:      /odor_sim/wind_visualization")
        if args.headless:
            time.sleep(args.headless_seconds)
            return 0
        rviz_environment = dict(os.environ)
        for variable in ("QT_QPA_PLATFORM_PLUGIN_PATH", "QT_QPA_FONTDIR"):
            if "site-packages/cv2" in rviz_environment.get(variable, ""):
                rviz_environment.pop(variable, None)
        rviz = subprocess.Popen(
            ["rviz2", "-d", str(_RVIZ_CONFIG)],
            env=rviz_environment,
            start_new_session=True,
        )
        return rviz.wait()
    except KeyboardInterrupt:
        return 130
    finally:
        _stop_process(rviz)
        _stop_process(markers)
        if marker_log is not None:
            marker_log.close()
        server.stop()
        if original_ros_domain is None:
            os.environ.pop("ROS_DOMAIN_ID", None)
        else:
            os.environ["ROS_DOMAIN_ID"] = original_ros_domain


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--regime", required=True, choices=WIND_REGIMES)
    parser.add_argument("--gas", default="ethanol")
    parser.add_argument("--strength", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wind-duration-s", type=float, default=300.0)
    parser.add_argument("--out-dir", default="analysis/gaden_wind_examples")
    parser.add_argument("--log-dir", default=".roslog/wind_rviz")
    parser.add_argument("--wait-timeout", type=float, default=60.0)
    parser.add_argument(
        "--ros-domain-id", type=int, default=77,
        help="isolated ROS domain used only by the visualization subprocesses",
    )
    parser.add_argument("--headless", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--headless-seconds", type=float, default=3.0, help=argparse.SUPPRESS)
    return parser


if __name__ == "__main__":
    raise SystemExit(run(build_parser().parse_args()))

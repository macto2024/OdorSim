"""Generate a GADEN GUI project demonstrating one OdorSim wind regime."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from odor_sim.config.frame_map import FrameMap
from odor_sim.envs.odor_profile import OdorProfile
from odor_sim.envs.registry import resolve_scenario
from odor_sim.envs.scene_builder import SceneBuilder
from odor_sim.odor_encoder.wind import WIND_REGIMES, sample_wind_profile, write_isolated_config

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _gaden_gui_command() -> str:
    """Find a GUI binary even when another gaden_common overlay shadows it."""
    for prefix in os.environ.get("AMENT_PREFIX_PATH", "").split(os.pathsep):
        if not prefix:
            continue
        executable = Path(prefix) / "lib" / "gaden_common" / "gaden_gui"
        if executable.is_file() and os.access(executable, os.X_OK):
            return str(executable)
    return "ros2 run gaden_common gaden_gui"


def build_example(args: argparse.Namespace) -> Path:
    output_root = Path(args.out_dir).expanduser()
    if not output_root.is_absolute():
        output_root = _REPO_ROOT / output_root
    root = output_root / args.regime
    config_dir = root / "environment_configurations" / "config1"
    source_config = resolve_scenario(args.scenario, args.scenario_config)
    profile = sample_wind_profile(
        np.random.default_rng(args.seed), args.duration_s, args.regime,
    )
    write_isolated_config(source_config, config_dir, profile)
    component = {"gas_type": args.gas, "strength": args.strength}
    obj = SimpleNamespace(
        object_id=f"wind_demo_{args.gas}",
        odor_profile=OdorProfile.from_spec([component]),
    )
    builder = SceneBuilder(frame_map=FrameMap.from_scenario(source_config))
    builder.add_object(obj)
    builder.export_gaden_scene(
        config_dir, scene_id="wind_demo",
        initial_positions_gaden=np.asarray([[2.0, 3.0, 0.5]]),
    )
    (root / "gaden.gproj").write_text(
        "# Gaden Project\nversionMajor: 3\nversionMinor: 0\n"
        "configurationsDirectory: environment_configurations\n"
    )
    speeds = profile.speed_m_s
    print(f"project: {(root / 'gaden.gproj').resolve()}")
    print(
        f"regime={args.regime} speed min/mean/max="
        f"{speeds.min():.4f}/{speeds.mean():.4f}/{speeds.max():.4f} m/s"
    )
    if getattr(args, "print_gui_command", True):
        print(f"open it with: {_gaden_gui_command()}")
    return root / "gaden.gproj"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--regime", required=True, choices=WIND_REGIMES)
    parser.add_argument("--duration-s", type=float, default=60.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gas", default="ethanol")
    parser.add_argument("--strength", type=float, default=0.75)
    parser.add_argument("--scenario", default="10x6_uniform")
    parser.add_argument("--scenario-config", default="config1")
    parser.add_argument("--out-dir", default="analysis/gaden_wind_examples")
    return parser


if __name__ == "__main__":
    build_example(build_parser().parse_args())

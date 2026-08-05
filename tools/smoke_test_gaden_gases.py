"""Run the real GADEN server with OdorSim's four extended gas types."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from odor_sim.bridge.export_scene import build_scene_builder
from odor_sim.bridge.gaden_bridge import GadenBridge
from odor_sim.config.frame_map import FrameMap
from odor_sim.runtime.gaden_server import GadenServerManager


RECIPES = (
    "methanol_strong",
    "acetic_acid_strong",
    "ethyl_acetate_strong",
    "toluene_strong",
)
GASES = ("methanol", "acetic_acid", "ethyl_acetate", "toluene")


def run_smoke_test(config_dir: Path, *, steps: int = 80) -> dict:
    config_dir = config_dir.resolve()
    scene_id = "odorsim_extended_gases_smoke"
    builder = build_scene_builder(
        list(RECIPES),
        frame_map=FrameMap.from_scenario(config_dir),
    )
    positions = np.asarray(
        [
            [4.0, 1.5, 0.8],
            [4.0, 2.5, 0.8],
            [4.0, 3.5, 0.8],
            [4.0, 4.5, 0.8],
        ],
        dtype=float,
    )
    builder.export_gaden_scene(
        config_dir,
        scene_id=scene_id,
        initial_positions_gaden=positions,
    )

    server = GadenServerManager(config_dir, scene_id=scene_id)
    bridge = None
    try:
        server.start(timeout=60.0)
        bridge = GadenBridge(node_name="odorsim_extended_gases_smoke")
        if not bridge.wait_for_server(timeout=15.0):
            raise RuntimeError("/odor_value service was not available")
        bridge.publish_source_poses(positions)
        for _ in range(steps):
            if not bridge.step():
                raise RuntimeError("GADEN lockstep confirmation timed out")
        readings = bridge.query_ppm(positions)
    finally:
        if bridge is not None:
            bridge.close()
        server.stop()

    for reading in readings:
        if set(reading) != set(GASES):
            raise AssertionError(f"unexpected GADEN gas names: {sorted(reading)}")
        if not all(math.isfinite(value) and value >= 0.0 for value in reading.values()):
            raise AssertionError(f"invalid concentration values: {reading}")
    own_source_ppm = {
        gas: float(readings[index][gas]) for index, gas in enumerate(GASES)
    }
    if not all(value > 0.0 for value in own_source_ppm.values()):
        raise AssertionError(f"new gases did not emit at their sources: {own_source_ppm}")
    return {"steps": steps, "own_source_ppm": own_source_ppm}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=Path(
            "scenarios/10x6_uniform/environment_configurations/config1"
        ),
    )
    parser.add_argument("--steps", type=int, default=80)
    args = parser.parse_args()
    print(json.dumps(run_smoke_test(args.config_dir, steps=args.steps), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

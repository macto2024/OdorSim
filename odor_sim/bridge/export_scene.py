"""Export a GADEN scene whose source list matches an OdorSim env.

The ``odor_gaden_rt`` server loads a fixed list of sources (a "scene") at
startup and expects ``/gaden/source_poses`` to carry exactly one pose per
source, in the same order. This CLI builds that scene from the same Phase 3
SceneBuilder the bridge uses at runtime, so the server's source *i* is the
env's source *i* index-for-index.

Two ways to describe the sources:

  * ``--recipe NAME`` (repeatable): one object per recipe (the common case; a
    single object with a multi-VOC recipe still expands to several sources).
  * default: a single ``ripe_fruit`` object (matches ``OdorLift``).

Example (writes ``scenes/rt_scene.yaml`` + per-source ``simulations/`` under the
scenario config dir, then start the server with ``sceneID:=rt_scene``)::

    python -m odor_sim.bridge.export_scene \
        --config-dir scenarios/10x6_uniform/environment_configurations/config1 \
        --scene-id rt_scene --recipe ripe_fruit
"""

from __future__ import annotations

import argparse
from pathlib import Path

from odor_sim.config.frame_map import FrameMap
from odor_sim.envs.odor_object import OdorObject
from odor_sim.envs.scene_builder import SceneBuilder


def build_scene_builder(recipes: list[str], frame_map: FrameMap | None = None) -> SceneBuilder:
    """Build a SceneBuilder with one OdorObject per recipe name."""
    sb = SceneBuilder(frame_map=frame_map)
    for i, recipe in enumerate(recipes):
        name = f"obj{i}_{recipe}"
        sb.add_object(OdorObject.from_recipe(name, recipe))
    return sb


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config-dir",
        default="scenarios/10x6_uniform/environment_configurations/config1",
        help="GADEN environment configuration directory (contains config.yaml)",
    )
    parser.add_argument("--scene-id", default="rt_scene", help="scene file name to write")
    parser.add_argument(
        "--recipe",
        action="append",
        default=None,
        help="VOC recipe name (repeat for multiple objects); default: ripe_fruit",
    )
    args = parser.parse_args(argv)

    recipes = args.recipe or ["ripe_fruit"]
    frame_map = FrameMap.from_scenario(args.config_dir)
    sb = build_scene_builder(recipes, frame_map=frame_map)

    scene_path = sb.export_gaden_scene(args.config_dir, scene_id=args.scene_id)

    print(f"Wrote scene with {sb.num_sources} source(s) -> {scene_path}")
    for src in sb.sources:
        print(f"  src{src.index:03d}  {src.object_id}  {src.component.gaden_gas_name()}")
    print()
    print("Start the server with this scene:")
    print(
        "  ros2 run odor_gaden_rt rt_server --ros-args \\\n"
        f"    -p scenarioPath:=$PWD/{Path(args.config_dir)} \\\n"
        f"    -p sceneID:={args.scene_id}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

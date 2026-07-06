"""Phase 3 test: robosuite task-authoring layer (OdorObject/OdorProfile, scene
builder + object->source map, EE sensor site, frame map, VOC recipe table).

Runnable two ways:
    python tests/test_phase3.py          # prints PASS/FAIL for each check
    pytest tests/test_phase3.py          # standard test discovery

The robosuite env is created headless (no renderer, no camera obs) so it needs
no GL context.
"""

from __future__ import annotations

import os
import sys

import numpy as np

# Headless MuJoCo GL (safe even though we disable rendering).
os.environ.setdefault("MUJOCO_GL", "egl")

from odor_sim.config.frame_map import FrameMap
from odor_sim.config.gas_types import GasType, MOX_SUPPORTED, parse_gas_type
from odor_sim.config.recipes import get_recipe, list_recipes
from odor_sim.envs.odor_object import OdorObject
from odor_sim.envs.odor_profile import OdorProfile, VOCComponent
from odor_sim.envs.scene_builder import SceneBuilder


def test_gas_types_match_gaden():
    assert int(GasType.ethanol) == 0
    assert int(GasType.acetone) == 6
    assert parse_gas_type("acetone") is GasType.acetone
    assert parse_gas_type(6) is GasType.acetone
    assert parse_gas_type("fluorine") is GasType.fluorine  # tolerate spelling
    assert MOX_SUPPORTED == {
        GasType.ethanol,
        GasType.methane,
        GasType.hydrogen,
        GasType.propanol,
        GasType.chlorine,
        GasType.fluorine,
        GasType.acetone,
    }


def test_recipe_table_loads():
    names = list_recipes()
    assert "ripe_fruit" in names
    profile = get_recipe("ripe_fruit")
    assert isinstance(profile, OdorProfile)
    assert profile.num_sources == 2
    assert profile.gas_types == [GasType.ethanol, GasType.acetone]
    # strength -> emission params are monotone
    weak = VOCComponent(GasType.ethanol, strength=0.2)
    strong = VOCComponent(GasType.ethanol, strength=0.9)
    assert strong.filament_ppm_center > weak.filament_ppm_center
    assert strong.num_filaments_sec > weak.num_filaments_sec


def test_scene_builder_object_to_source_map():
    a = OdorObject.from_recipe("fruit_a", "ripe_fruit")       # 2 VOCs
    b = OdorObject.from_recipe("solvent_b", "solvent_leak")    # 2 VOCs
    c = OdorObject.from_recipe("single_c", "ethanol_strong")   # 1 VOC

    sb = SceneBuilder()
    sb.add_objects([a, b, c])

    assert sb.num_sources == 5
    assert sb.object_to_sources == {
        "fruit_a": [0, 1],
        "solvent_b": [2, 3],
        "single_c": [4],
    }
    # total sources == sum of profile lengths
    assert sb.num_sources == a.num_sources + b.num_sources + c.num_sources


def test_frame_map_roundtrip_and_alignment():
    fm = FrameMap.from_scenario(
        "scenarios/10x6_uniform/environment_configurations/config1"
    )
    # robosuite table center -> Phase 2 test source location in the room
    g = fm.robosuite_to_gaden([0.0, 0.0, 0.8])
    assert np.allclose(g, [2.0, 3.0, 0.5], atol=1e-6)
    # inverse is consistent
    back = fm.gaden_to_robosuite(g)
    assert np.allclose(back, [0.0, 0.0, 0.8], atol=1e-6)
    # +x in robosuite stays +x in GADEN (wind direction)
    gx = fm.robosuite_to_gaden([0.1, 0.0, 0.8])
    assert gx[0] > g[0] and np.isclose(gx[1], g[1])


def test_scene_builder_source_positions_follow_object():
    obj = OdorObject.from_recipe("fruit", "ripe_fruit")  # ethanol + acetone(+z 0.02)
    sb = SceneBuilder(frame_map=FrameMap())
    sb.add_object(obj)

    pos0 = np.array([0.0, 0.0, 0.8])
    src0 = sb.gaden_source_positions({"fruit": (pos0, np.eye(3))})
    assert src0.shape == (2, 3)
    # the acetone component carries a +0.02 z local offset
    assert np.isclose(src0[1][2] - src0[0][2], 0.02, atol=1e-6)

    # move the object; both sources shift by the same delta
    pos1 = pos0 + np.array([0.3, -0.1, 0.0])
    src1 = sb.gaden_source_positions({"fruit": (pos1, np.eye(3))})
    assert np.allclose(src1 - src0, np.array([0.3, -0.1, 0.0]), atol=1e-6)


def test_scene_export_index_aligned(tmp_path=None):
    import tempfile
    import yaml

    a = OdorObject.from_recipe("fruit_a", "ripe_fruit")
    c = OdorObject.from_recipe("single_c", "ethanol_strong")
    sb = SceneBuilder()
    sb.add_objects([a, c])

    d = tempfile.mkdtemp()
    # minimal config.yaml so nothing else is needed
    with open(os.path.join(d, "config.yaml"), "w") as f:
        f.write("cell_size: 0.1\n")
    scene_path = sb.export_gaden_scene(d, scene_id="scene1")
    with open(scene_path) as f:
        scene = yaml.safe_load(f)
    sims = scene["simulations"]
    assert len(sims) == sb.num_sources == 3
    # scene order matches source index order
    assert sims[0]["sim"].startswith("src000_fruit_a_ethanol")
    assert sims[2]["sim"].startswith("src002_single_c_ethanol")


def test_odor_lift_env_loads_and_steps():
    import robosuite  # noqa: F401
    from odor_sim.envs.odor_lift import OdorLift

    env = OdorLift(
        robots="Panda",
        recipe="ripe_fruit",
        scenario_config_dir="scenarios/10x6_uniform/environment_configurations/config1",
        enose_site_offset=(0.0, 0.0, -0.02),
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        control_freq=20,
        horizon=50,
    )
    obs = env.reset()

    # instruction present in observation
    assert "instruction" in obs and isinstance(obs["instruction"], str) and obs["instruction"]

    # proprio present
    assert any(k.startswith("robot0_") for k in obs), "no proprio keys in obs"

    # object->source map: 1 object, 2 VOCs -> 2 sources
    assert env.object_to_sources == {"odor_cube": [0, 1]}
    assert env.scene_builder.num_sources == 2

    # EE sensor pose readable
    ee_pos, ee_mat = env.get_enose_site_pose()
    assert ee_pos.shape == (3,) and ee_mat.shape == (3, 3)

    # object world pose readable
    poses = env.get_object_world_poses()
    assert "odor_cube" in poses

    # GADEN-frame source poses computed; move object -> sources follow
    src_before = env.get_gaden_source_poses()
    assert src_before.shape == (2, 3)

    # step a few times without crashing
    a_dim = env.action_dim
    for _ in range(10):
        obs, reward, done, info = env.step(np.zeros(a_dim))
    assert "instruction" in obs
    assert "odor_cube_pos" in obs and "enose_site_pos" in obs
    env.close()


def _run_all():
    tests = [
        test_gas_types_match_gaden,
        test_recipe_table_loads,
        test_scene_builder_object_to_source_map,
        test_frame_map_roundtrip_and_alignment,
        test_scene_builder_source_positions_follow_object,
        test_scene_export_index_aligned,
        test_odor_lift_env_loads_and_steps,
    ]
    failures = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except Exception as e:  # noqa: BLE001
            failures += 1
            print(f"FAIL  {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failures}/{len(tests)} checks passed")
    return failures


if __name__ == "__main__":
    sys.exit(1 if _run_all() else 0)

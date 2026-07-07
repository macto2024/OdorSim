"""Phase 4.5a test: odor_sim.make() dry run (no ROS server).

Verifies the unified facade builds the env and exports a GADEN scene straight
from the env's SceneBuilder — the single-source-of-truth rule — with the scene's
source list index-aligned to ``env.get_gaden_source_poses()``. No server is
spawned (``auto_start_gaden=False``), so this runs headless with no ROS.

    python tests/test_make_export.py     # prints PASS/FAIL
    pytest tests/test_make_export.py
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("MUJOCO_GL", "egl")

import yaml

import odor_sim as odorsim
from odor_sim.envs.registry import list_tasks, resolve_scenario

SCENARIO = "10x6_uniform"


def test_registry_lists_odorlift():
    assert "OdorLift" in list_tasks()


def test_make_dry_run_builds_env_and_exports_scene():
    with odorsim.make(
        "OdorLift",
        recipe="ripe_fruit",
        scenario=SCENARIO,
        auto_start_gaden=False,
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        control_freq=20,
        horizon=50,
    ) as cosim:
        cosim.reset()

        # no server / bridge in a dry run
        assert cosim.bridge is None and cosim.server is None
        assert cosim.has_gaden is False

        # scene file written under the resolved scenario config dir
        assert cosim.scene_path is not None and cosim.scene_path.exists()
        assert cosim.scene_id == "odorlift_ripe_fruit"

        # source count matches the env; ripe_fruit == ethanol + acetone == 2
        n = cosim.env.scene_builder.num_sources
        assert n == 2
        assert cosim.env.get_gaden_source_poses().shape == (n, 3)

        # exported scene is index-aligned with the env's source list
        scene = yaml.safe_load(cosim.scene_path.read_text())
        sims = scene["simulations"]
        assert len(sims) == n
        env_sources = cosim.env.scene_builder.sources
        for i, entry in enumerate(sims):
            assert entry["sim"].startswith(f"src{i:03d}_")
            assert env_sources[i].component.gaden_gas_name() in entry["sim"]

        # dry-run step: env advances, ppm is present-but-empty (no GADEN)
        obs, reward, done, info = cosim.step(cosim.zero_action())
        assert "instruction" in obs
        assert info["ppm"] == {}


def test_scenario_resolves_to_config_dir():
    cfg = resolve_scenario(SCENARIO)
    assert (cfg / "config.yaml").is_file()


def _run_all():
    tests = [
        test_registry_lists_odorlift,
        test_make_dry_run_builds_env_and_exports_scene,
        test_scenario_resolves_to_config_dir,
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

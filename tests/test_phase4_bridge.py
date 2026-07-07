"""Phase 4a integration test: robosuite <-> odor_gaden_rt bridge.

This test needs the GADEN real-time server running in LOCKSTEP mode with a
scene whose source list matches the env. Set it up in a separate terminal
first (see the printed instructions / the package README):

    # Terminal 1
    source setup/activate.sh
    python -m odor_sim.bridge.export_scene \
        --config-dir scenarios/10x6_uniform/environment_configurations/config1 \
        --scene-id rt_scene --recipe ripe_fruit
    ros2 run odor_gaden_rt rt_server --ros-args \
        -p scenarioPath:=$PWD/scenarios/10x6_uniform/environment_configurations/config1 \
        -p sceneID:=rt_scene

    # Terminal 2
    source setup/activate.sh
    python tests/test_phase4_bridge.py

PASS criteria:
  * bridge connects; env steps N times and /gaden/sim_time advances in lockstep;
  * with the source parked at the EE, ppm at the EE rises over time (plume
    reaches the sensor) and reports the scene's gases (ethanol, acetone);
  * moving the source far upwind of the EE makes the EE ppm drop (peak follows
    the object).
"""

from __future__ import annotations

import os
import sys

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")

RECIPE = "ripe_fruit"
CONFIG_DIR = "scenarios/10x6_uniform/environment_configurations/config1"


def _make_env():
    from odor_sim.envs.odor_lift import OdorLift

    env = OdorLift(
        robots="Panda",
        recipe=RECIPE,
        scenario_config_dir=CONFIG_DIR,
        enose_site_offset=(0.0, 0.0, -0.02),
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        control_freq=20,
        horizon=10000,
    )
    env.reset()
    return env


def main() -> int:
    from odor_sim.bridge import GadenBridge

    env = _make_env()
    bridge = GadenBridge(step_timeout=2.0)

    if not bridge.wait_for_server(timeout=10.0):
        print("FAIL  /odor_value service not available. Is odor_gaden_rt running in lockstep mode?")
        bridge.close()
        env.close()
        return 1

    failures = 0

    def check(name, ok, detail=""):
        nonlocal failures
        if ok:
            print(f"PASS  {name}  {detail}")
        else:
            failures += 1
            print(f"FAIL  {name}  {detail}")

    # Source count must match the running scene, or the server ignores poses.
    n_sources = env.scene_builder.num_sources
    print(f"env has {n_sources} source(s); recipe={RECIPE}")

    # 1) Lockstep: sim_time advances as we step.
    t0 = bridge.sim_time
    for _ in range(20):
        bridge.step(1)
    t1 = bridge.sim_time
    check("lockstep_sim_time_advances", t1 > t0, f"sim_time {t0:.3f} -> {t1:.3f}")

    # 2) Park the source(s) at the EE's GADEN-frame position; the plume should
    #    build up and reach the sensor. This is the analog of the arm arriving
    #    at the odor object.
    ee_world, _ = env.get_enose_site_pose()
    ee_gaden = env.frame_map.robosuite_to_gaden(ee_world)
    at_ee = np.tile(ee_gaden, (n_sources, 1))

    ppm_series = []
    for _ in range(200):
        bridge.publish_source_poses(at_ee)
        bridge.step(1)
        ppm = bridge.query_ppm(ee_gaden)[0]
        ppm_series.append(sum(ppm.values()))
    ppm_near = ppm_series[-1]
    check(
        "ppm_rises_when_source_at_ee",
        ppm_near > 1.0,
        f"total ppm at EE = {ppm_near:.2f} (gases: {list(ppm.keys())})",
    )
    check(
        "scene_gases_reported",
        set(ppm.keys()) >= {"ethanol", "acetone"},
        f"gases = {sorted(ppm.keys())}",
    )
    # rising trend: late-window mean well above early-window mean
    early = float(np.mean(ppm_series[:20]))
    late = float(np.mean(ppm_series[-20:]))
    check("ppm_trend_increases", late > early, f"early {early:.2f} -> late {late:.2f}")

    # 3) Move the source far upwind of the EE; ppm at the EE should fall.
    upwind = ee_gaden + np.array([-3.0, 0.0, 0.0])  # -x is upwind of the sensor
    upwind = np.clip(upwind, [0.3, 0.3, 0.2], None)
    far = np.tile(upwind, (n_sources, 1))
    for _ in range(200):
        bridge.publish_source_poses(far)
        bridge.step(1)
    ppm_far = sum(bridge.query_ppm(ee_gaden)[0].values())
    check("ppm_drops_when_source_moves_away", ppm_far < ppm_near, f"{ppm_near:.2f} -> {ppm_far:.2f}")

    # 4) End-to-end step_env returns a well-formed record.
    rec = bridge.step_env(env)
    ok = (
        "sim_time" in rec
        and rec["source_poses_gaden"].shape == (n_sources, 3)
        and isinstance(rec["ppm"], dict)
        and rec["ee_pos_gaden"].shape == (3,)
    )
    check("step_env_record_shape", ok, f"keys={sorted(rec.keys())}")

    bridge.close()
    env.close()

    total = 6
    print(f"\n{total - failures}/{total} checks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

# OdorSim

Co-simulation of **GADEN** (3D odor/gas dispersion) with **robosuite 1.5.2**
(MuJoCo physics) over **ROS 2 Jazzy**, for collecting odor-aware manipulation
demonstrations and training/evaluating VLA policies that fuse an e-nose sense.

See [docs/gaden_robosuite_cosim_d067d61f.plan.md](docs/gaden_robosuite_cosim_d067d61f.plan.md)
for the full design and phase plan.

## Repository layout

```
OdorSim/
├── setup/                     # reproducible install scripts
│   ├── install_ros_gaden.sh   #   ROS 2 + GADEN (clones & builds gaden/ in place)
│   ├── install_sim_env.sh     #   Python venv: robosuite 1.5.2 + mujoco + lerobot
│   ├── requirements-sim.txt   #   sim-side pip deps
│   └── activate.sh            #   `source` this to enter the full environment
├── ros2_ws/src/               # OUR ROS 2 packages (overlay on gaden)
│   └── odor_gaden_rt/         #   real-time moving-source GADEN server (Phase 2)
├── odor_sim/                  # OUR Python package (pip install -e .)
│   ├── envs/                  #   robosuite tasks, OdorObject/OdorProfile, RM65 (Phase 3)
│   ├── bridge/                #   rclpy client + teleop app (Phase 4)
│   ├── sensors/               #   MOX/PID e-nose model, ppm->voltage (Phase 4)
│   ├── recording/             #   LeRobot dataset writer (Phase 5)
│   ├── policy/                #   PolicyAdapter + eval harness (Phase 6)
│   └── config/                #   VOC recipe table, frame map, scenario ids
├── scenarios/                 # GADEN room configs we author
├── docs/                      # design plan
├── gaden/                     # cloned + built by setup (gitignored, not ours)
└── pyproject.toml
```

`gaden/`, `ros2_ws/build|install|log`, `.venv/`, and the deprecated
`simulation/` playground are gitignored.

## Setup (fresh machine)

Ubuntu 24.04 + system Python 3.12. Run in order:

```bash
# 1. ROS 2 Jazzy + GADEN (clones gaden/, builds it, then our overlay)
bash setup/install_ros_gaden.sh

# 2. robosuite/mujoco/lerobot venv (system-site-packages so rclpy is visible)
bash setup/install_sim_env.sh
```

Then, in any new shell:

```bash
source setup/activate.sh   # ROS 2 + GADEN + overlay + venv, in one step
```

## Odor task authoring (Phase 3)

`odor_sim.envs` is a small task-authoring layer over robosuite 1.5.2:

- **`OdorProfile` / `VOCComponent`** (`envs/odor_profile.py`): an object emits a
  mixture of VOCs; each component has a `gas_type`, an abstract `strength`
  (0..1, mapped to GADEN `filamentPPMcenter` / `numFilaments_sec`), and an
  optional local offset.
- **`OdorObject`** (`envs/odor_object.py`): a robosuite `BoxObject` carrying an
  `OdorProfile`. Build one from the recipe table with
  `OdorObject.from_recipe(name, "ripe_fruit")`.
- **VOC recipe table** (`config/voc_recipes.yaml` + `config/recipes.py`): named
  mixtures using MOX-detectable gases (ethanol, methane, hydrogen, propanol,
  chlorine, fluorine, acetone).
- **`SceneBuilder`** (`envs/scene_builder.py`): flattens every object's VOCs
  into an ordered GADEN source list, keeps the `object_id -> [source indices]`
  map (feeds the Phase 4 bridge / `/gaden/source_poses`), and can export an
  index-aligned GADEN scene (`export_gaden_scene`).
- **`FrameMap`** (`config/frame_map.py`): affine `p_gaden = R @ p_robosuite + t`
  built from a scenario config; the default aligns the robosuite table with the
  `10x6_uniform` room.
- **`OdorLift`** (`envs/odor_lift.py`): a Lift-style Panda task whose cube is an
  `OdorObject`. Exposes the EE odor-sensor pose (`get_enose_site_pose`), object
  world poses, and per-step GADEN source poses. Every observation carries the
  task `instruction` string.

Quick check (headless):

```bash
source setup/activate.sh
python tests/test_phase3.py     # prints PASS/FAIL for 7 checks
```

## Bridge + e-nose sensor model + teleop (Phase 4)

`odor_sim.bridge` couples a robosuite env to the running GADEN server, and
`odor_sim.sensors` is the shared MOX/PID e-nose model.

- **`GadenBridge`** (`bridge/gaden_bridge.py`): per control step it publishes the
  object-expanded source poses to `/gaden/source_poses`, ticks `/gaden/step`
  once (lockstep, confirmed via `/gaden/sim_time`), and queries `/odor_value`
  for the ground-truth per-gas ppm at the EE. `bridge.step_env(env)` returns
  `{sim_time, ee_pos_world, ee_pos_gaden, source_poses_gaden, ppm, gas_types}`.
- **`export_scene`** (`bridge/export_scene.py`): writes a GADEN scene whose
  source list matches an env index-for-index, so `/gaden/source_poses` lines up
  with `env.get_gaden_source_poses()`. Run it before starting the server.
- **MOX/PID model** (`sensors/mox_pid.py`): pure-Python port of GADEN's
  `fake_gas_sensor.cpp` sensitivity + transient, plus a voltage divider.
  `ContinuousEnose` (always exposed) and `SamplingEnose` (valve-gated by the
  ternary `enose_state`) both derive from the same ppm(t); `synthesize_*`
  helpers run them offline over a stored ppm series.
- **Teleop** (`bridge/teleop.py`): `TeleopSession` drives `OdorLift` with a
  keyboard/SpaceMouse while co-stepping the plume, recording action +
  `enose_state` token + auto-hold/sampling mask + raw ppm(t) + instruction per
  episode. E-nose keys: `1` sample (auto-hold ~7 s), `0` idle, `2` filter.

### Test procedures

The e-nose model test is standalone; the bridge and teleop tests need the
server running in lockstep with a matching scene.

```bash
# 4c — sensor model (no ROS, no robosuite)
source setup/activate.sh
python tests/test_phase4_sensor.py        # 7/7 checks

# 4a/4b — start the server once (Terminal 1)
source setup/activate.sh
python -m odor_sim.bridge.export_scene \
    --config-dir scenarios/10x6_uniform/environment_configurations/config1 \
    --scene-id rt_scene --recipe ripe_fruit
ros2 run odor_gaden_rt rt_server --ros-args \
    -p scenarioPath:=$PWD/scenarios/10x6_uniform/environment_configurations/config1 \
    -p sceneID:=rt_scene

# 4a — bridge, 4b — teleop recording (Terminal 2)
source setup/activate.sh
python tests/test_phase4_bridge.py        # 6/6 checks
python tests/test_phase4_teleop.py        # 11/11 checks
```

Interactive collection (needs a display; server running as above):

```bash
python -m odor_sim.bridge.teleop --recipe ripe_fruit --device keyboard
```

## Status

Phases 0-4 complete: repo skeleton + setup scripts (0-1), the real-time
moving-source GADEN server `odor_gaden_rt` (2), the robosuite odor
task-authoring layer in `odor_sim.envs` (3), and the rclpy bridge + shared
MOX/PID e-nose model + teleop collection app (4). Remaining phases build the
LeRobot dataset recording (5) and the training-ready eval hooks (6). Each phase
is delivered with a self-contained test procedure for verification before
moving on.

# OdorSim

Co-simulation of **GADEN** (3D odor/gas dispersion) with **robosuite 1.5.2**
(MuJoCo physics) over **ROS 2 Jazzy**, for collecting odor-aware manipulation
demonstrations and training/evaluating VLA policies that fuse an e-nose sense.

See [docs/gaden_robosuite_cosim_d067d61f.plan.md](docs/gaden_robosuite_cosim_d067d61f.plan.md)
for the full design and phase plan.

## Quick start (unified `make()`)

One call brings up the whole co-simulation — it builds the robosuite task,
exports a matching GADEN scene, spawns the `odor_gaden_rt` server in lockstep,
and connects the bridge. No separate server terminal:

```bash
source setup/activate.sh
python - <<'EOF'
import odor_sim as odorsim

with odorsim.make("OdorLift", recipe="ripe_fruit") as cosim:
    obs = cosim.reset()
    print(obs["instruction"])
    for _ in range(100):
        obs, reward, done, info = cosim.step(cosim.zero_action())
    print(info["ppm"])          # ground-truth per-gas ppm at the EE
EOF
```

`make()` must be run from a shell that has sourced `setup/activate.sh` (so the
server subprocess inherits ROS + the GADEN overlay). It returns an
`OdorCosimSession` (env + bridge + server) whose `step()` adds `info["ppm"]`,
`info["sim_time"]`, and `info["gaden_source_poses"]`. The GADEN scene is always
exported from the env's `SceneBuilder`, so source indices can never desync from
`env.get_gaden_source_poses()`. Useful options: `auto_start_gaden=False` (dry
run: build env + export scene, no ROS), `connect_only=True` (attach to a
server you started yourself), `scenario="..."`, `has_renderer=True`,
`odor_monitor=True` (live terminal ppm log + separate matplotlib plot window).

### Live ppm monitor (teleop / VNC)

With a display (local desktop or [VNC](VNC_SETUP.md)) you can open a second
window that plots per-VOC ppm at the EE vs GADEN sim time:

```bash
export DISPLAY=:1   # if using VNC on the VM
source setup/activate.sh

python -m odor_sim.bridge.teleop \
    --env OdorLift --recipe ripe_fruit --device keyboard \
    --odor-monitor
```

`--odor-monitor log` prints ppm lines only (headless-friendly).
`--odor-monitor plot` opens the graph without terminal spam.

Programmatic equivalent:

```python
with odorsim.make("OdorLift", recipe="ripe_fruit", has_renderer=True, odor_monitor=True) as cosim:
    ...
```

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
- **Teleop** (`bridge/teleop.py`): `TeleopSession` drives a task with a
  keyboard/SpaceMouse while co-stepping the plume, recording action +
  `enose_state` token + auto-hold/sampling mask + raw ppm(t) + instruction per
  episode. E-nose keys: `1` sample (auto-hold ~7 s), `0` idle, `2` filter. Since
  Phase 4.5 it uses `odorsim.make()`, so it spawns its own GADEN server (single
  command, no second terminal).

### Test procedures

Since Phase 4.5 the co-sim tests spawn their own server via `make()`, so most
run from a single terminal. Only the legacy `test_phase4_bridge.py` still
attaches to a server you start by hand.

```bash
source setup/activate.sh

# Standalone — no ROS server needed
python tests/test_phase4_sensor.py        # 4c e-nose model     — 7/7
python tests/test_make_export.py          # 4.5a make() dry run — 3/3
python tests/test_odor_monitor.py       # 4.5f monitor unit  — 6/6

# Auto-start the server (single terminal, via odorsim.make)
python tests/test_make_cosim.py           # 4.5c full co-sim    — 7/7
python tests/test_phase4_teleop.py        # 4b teleop recording — 11/11
```

Legacy Phase 4a bridge test (needs the server started manually):

```bash
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
python tests/test_phase4_bridge.py        # 6/6 checks
```

Interactive collection (needs a display; server auto-started by teleop):

```bash
python -m odor_sim.bridge.teleop --env OdorLift --recipe ripe_fruit --device keyboard --odor-monitor
```

See [VNC_SETUP.md](VNC_SETUP.md) for remote display setup on the VM.

## Status

Phases 0-4.5 complete: repo skeleton + setup scripts (0-1), the real-time
moving-source GADEN server `odor_gaden_rt` (2), the robosuite odor
task-authoring layer in `odor_sim.envs` (3), the rclpy bridge + shared MOX/PID
e-nose model + teleop collection app (4), and the unified `odorsim.make()`
co-simulation facade that spawns the server and wires the bridge from one call
(4.5), including optional live ppm log/plot via `odor_monitor` (4.5f). Remaining
phases build the LeRobot dataset recording (5) and the training-ready eval hooks (6). Each phase is delivered with a self-contained
test procedure for verification before moving on.

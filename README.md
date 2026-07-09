# OdorSim

OdorSim connects **GADEN** odor / gas dispersion with **robosuite** manipulation tasks through ROS 2. It is meant for collecting odor-aware robot demonstrations and converting them into LeRobot datasets.

## What Is In This Repo

```text
setup/                  install and activation scripts
odor_sim/               Python package: tasks, bridge, teleop, recording tools
ros2_ws/src/odor_gaden_rt/
                        ROS 2 real-time GADEN server
scenarios/              GADEN room / environment configs
datasets/teleop/        raw teleop episodes
datasets/lerobot/       converted LeRobot datasets
```

## Setup

Use Ubuntu 24.04. From the repo root, run these in order:

```bash
# 1. Install ROS 2 Jazzy, clone/build GADEN, and build the ROS overlay
bash setup/install_ros_gaden.sh

# 2. Create the Python sim environment and install OdorSim editable
bash setup/install_sim_env.sh
```

Then, in every new shell:

```bash
source setup/activate.sh
```

## Common Workflow

Most of the time, use this order:

```bash
source setup/activate.sh

# 1. Run a quick co-sim script
python - <<'EOF'
import odor_sim as odorsim

with odorsim.make("OdorLift", objects=["mango"]) as cosim:
    obs = cosim.reset()
    print(obs["instruction"])
    for _ in range(500):
        obs, reward, done, info = cosim.step(cosim.zero_action())
    print(info["ppm"])
EOF

# 2. Collect raw demonstrations
python -m odor_sim.bridge.teleop --env OdorLift --objects milk --device keyboard --odor-monitor
```

Teleop records **ground-truth ppm** at the end effector plus your **e-nose mode** each step. It does **not** write voltage yet — that comes from synthesis (step 3).

There are two e-nose behaviors, both derived from the same recorded ppm trace:

- **Continuous** — the sensor always sees the plume. Voltage rises and falls as you move through the odor field. No extra keys needed during teleop.
- **Sniffing (sampling)** — you operate this mode. Press `1` to sniff: the arm auto-holds for ~7 s while the valve is open. Press `0` (idle) or `2` (filter/purge) to close the valve and expose the sensor to clean air so it can decay back toward baseline.

During synthesis, these become two different voltage streams in `features.npz`:

- `enose_voltage_continuous` — always-on MOX response
- `enose_voltage_sampling` — valve-gated response driven by your `1` / `0` / `2` presses during teleop

```bash
# 3. Synthesize e-nose voltage features from recorded ppm
python -m odor_sim.recording.synthesize datasets/teleop

# 4. Convert raw episodes to LeRobot format
#    Do this after creating/activating the LeRobot env shown below.
python -m odor_sim.recording.convert --input datasets/teleop --output odorsim/odorlift
```

## Run The GADEN RT Server

You usually do **not** need to start the server manually. `odor_sim.make(...)` and teleop can export the scene, start `odor_gaden_rt`, connect the bridge, and stop the server for you.

Manual server mode is useful for debugging ROS topics and services.

First export a scene:

```bash
source setup/activate.sh

python -m odor_sim.bridge.export_scene \
  --config-dir scenarios/10x6_uniform/environment_configurations/config1 \
  --scene-id rt_scene \
  --recipe ripe_fruit
```

Start the server in lockstep mode:

```bash
ros2 run odor_gaden_rt rt_server --ros-args \
  -p scenarioPath:=$PWD/scenarios/10x6_uniform/environment_configurations/config1 \
  -p sceneID:=rt_scene \
  -p stepOnTimer:=false \
  -p publishMarkers:=false
```

Server parameters:

- `scenarioPath`: required path to a GADEN environment config directory containing `config.yaml`.
- `sceneID`: scene file to load, default `scene1`.
- `stepOnTimer`: `false` for lockstep control through `/gaden/step`; `true` to free-run on a timer.
- `publishMarkers`: `true` to publish RViz markers; `false` for lighter headless runs.

Useful ROS interfaces:

- Publish source poses to `/gaden/source_poses`.
- Step lockstep simulation with `/gaden/step`.
- Reset the plume with `/gaden/reset`.
- Read sim time from `/gaden/sim_time`.
- Query ppm with `/odor_value`.
- Query wind with `/wind_value`.

## Use OdorSim Co-Simulation

Recommended Python entry point:

```python
import odor_sim as odorsim

with odorsim.make("OdorLift", objects=["mango"]) as cosim:
    obs = cosim.reset()
    action = cosim.zero_action()
    obs, reward, done, info = cosim.step(action)
    print(info["ppm"])
```

Available tasks:

- `OdorLift`: Lift-style manipulation task. Spawns one or more catalog objects; you choose which one is the lift target (default: the first). Extra objects are odor-emitting distractors. Success is when the target rises ~4 cm above its resting height.


# Pick the lift target explicitly (need not be first)
with odorsim.make("OdorLift", objects=["milk", "mango"], target_object="mango") as cosim:
    ...
```

Useful `odor_sim.make(...)` options:

- `objects=["mango"]`: choose catalog object(s) to spawn.
- `target_object="mango"`: which spawned object is the lift target (default: first in `objects`).
- `scenario="10x6_uniform"`: choose a scenario under `scenarios/`, or pass a direct config directory.
- `scenario_config="config1"`: choose the config under `environment_configurations/`.
- `scene_id=None`: override the generated scene name.
- `auto_start_gaden=True`: start the GADEN server automatically.
- `connect_only=False`: connect to a server you already started.
- `bridge=True`: enable ROS bridge and ppm queries.
- `export=True`: export the GADEN scene from the robosuite env before starting.
- `server_log_dir=None`: choose where server logs are written.
- `wait_timeout=60.0`: seconds to wait for server readiness.
- `step_on_timer=False`: advanced; free-runs GADEN instead of bridge lockstep.
- `publish_markers=False`: enable or disable RViz marker publishing.
- `odor_monitor=False`: use `True`, `"log"`, or `"plot"` for live ppm monitoring.
- Any extra robosuite env kwargs, such as `robots="Panda"`, `has_renderer=True`, `control_freq=20`, or `render_camera="agentview"`.

For a dry run that builds the env and exports the scene without ROS:

```python
with odorsim.make("OdorLift", objects=["mango"], auto_start_gaden=False) as cosim:
    print(cosim.scene_path)
```

To connect to a manually started server:

```python
with odorsim.make("OdorLift", objects=["mango"], connect_only=True, scene_id="rt_scene") as cosim:
    obs = cosim.reset()
```

## Teleoperate And Record Data

Teleop drives an `OdorLift` (or other registered task) with a robosuite device while co-stepping the GADEN plume. Each control step records the arm action, e-nose mode, ground-truth ppm at the end effector, proprio, and the task instruction. Episodes are written as one folder per run under `datasets/teleop/`.

### OdorLift teleop

`OdorLift` spawns catalog objects on the table and asks you to **lift the target object** off the surface. The on-screen HUD shows the instruction (e.g. `pick up the mango`). Lifting a distractor does not count — only the target object's height is checked for success (~4 cm above its resting pose). After `--success-hold-steps` consecutive successful steps (default 10), the episode auto-ends.

**Single object** (default target is the only object):

```bash
source setup/activate.sh

python -m odor_sim.bridge.teleop \
  --env OdorLift \
  --objects milk \
  --device keyboard \
  --odor-monitor \
  --out-dir datasets/teleop
```

**Multiple objects** — first spawned object is the lift target by default; extras are odor distractors with their own recipes from `voc_recipes.yaml`:

```bash
python -m odor_sim.bridge.teleop \
  --env OdorLift \
  --objects mango milk \
  --device keyboard \
  --odor-monitor
```

**Explicit lift target** — spawn several objects but reward lifting a specific one (need not be first):

```bash
python -m odor_sim.bridge.teleop \
  --env OdorLift \
  --objects milk mango \
  --target-object mango \
  --device keyboard \
  --odor-monitor
```

**Controls during teleop:**

| Input | Action |
|-------|--------|
| robosuite keyboard / SpaceMouse | move arm and gripper (standard robosuite device controls) |
| `1` | e-nose **sample / sniff** — opens valve, auto-holds arm ~7 s (`--sample-hold-s`) |
| `0` | e-nose **idle** — valve closed, sensor purges toward baseline |
| `2` | e-nose **filter / purge** — same purge behavior as idle |
| device reset key (`Ctrl+Q` on keyboard) | end episode manually |

Continuous odor mode needs no e-nose keys — ppm is always recorded. Sniffing mode only exposes the sensor to the plume while you hold `1`; synthesis uses the recorded `enose_state` timeline to build the gated voltage stream.

**What gets recorded** (`datasets/teleop/episode_YYYYMMDD_HHMMSS/`):

```text
episode.npz          # actions, states, ppm(t), enose_state, proprio
meta.json            # instruction, objects, target_object, gas_types, success, ...
frames/agentview/    # camera PNGs (unless --no-frames)
frames/wrist/
```

`meta.json` includes the catalog `objects` list, the resolved `target_object` name, the natural-language `instruction`, and whether the episode ended in task `success`. Voltage features are **not** written at teleop time — run synthesis afterward (see below).

Important teleop options:

- `--env`: task name, default `OdorLift`.
- `--objects`: catalog object name(s) to spawn; default is the task's own default object (`odor_cube`).
- `--target-object`: which spawned object is the lift target (must be one of `--objects`); default is the first object.
- `--scenario`: scenario name or config directory, default `10x6_uniform`.
- `--device`: `keyboard` or `spacemouse`, default `keyboard`.
- `--robots`: robosuite robot name, default `Panda`.
- `--controller`: robosuite composite controller name or config path.
- `--camera`: on-screen viewer camera, default `agentview`.
- `--pos-sensitivity`: position input scale, default `1.0`.
- `--rot-sensitivity`: rotation input scale, default `1.0`.
- `--goal-update-mode`: `target` or `achieved`, default `target`.
- `--sample-hold-s`: seconds to auto-hold when sampling, default `7.0`.
- `--success-hold-steps`: successful steps before auto-ending, default `10`; use `0` to disable.
- `--out-dir`: raw dataset output directory, default `datasets/teleop`.
- `--no-frames`: disable camera-frame recording.
- `--camera-size`: recorded frame size, default `256`.
- `--no-bridge`: drive robosuite only, without GADEN / ppm.
- `--odor-monitor [MODE]`: live ppm monitor. Use no value for log + plot, or `log` / `plot`.

## Convert Data To LeRobot

The data pipeline is two stages: synthesize odor sensor features, then convert to LeRobot.

Synthesize e-nose voltage features (writes `features.npz` next to each episode):

```bash
source setup/activate.sh

python -m odor_sim.recording.synthesize datasets/teleop
```

This replays each episode's ppm(t) through the MOX model twice:

- **Continuous** → `enose_voltage_continuous` (always exposed to ppm)
- **Sniffing** → `enose_voltage_sampling` (only exposed while `enose_state == 1`)

Both streams land in `features.npz` and are included in the LeRobot export when you convert.

Useful synthesis options:

- `--mox-model`: MOX model, default `TGS2620`.
- `--load-resistance`: voltage-divider load resistance in ohms.
- `--vcc`: supply voltage, default `5.0`.
- `--rate`: sensor update rate, default is the episode control frequency.
- `--no-overwrite`: skip episodes that already have `features.npz`.

Install LeRobot dependencies in a separate environment, then run conversion from that environment:

```bash
python3 -m venv .venv-lerobot
source .venv-lerobot/bin/activate
pip install -U pip wheel
pip install -e .
pip install "lerobot[dataset]>=0.6"
```

Convert all raw episodes:

```bash
python -m odor_sim.recording.convert \
  --input datasets/teleop \
  --output odorsim/odorlift
```

Convert selected episodes:

```bash
python -m odor_sim.recording.convert \
  --input datasets/teleop/episode_20260708_173353 \
  --output odorsim/odorlift
```

Useful conversion options:

- `--input`: one or more episode dirs or parent dirs containing `episode_*/`.
- `--output`: LeRobot repo id / dataset name, for example `odorsim/odorlift`.
- `--root`: parent output directory, default `datasets/lerobot`.
- `--fps`: dataset FPS, default is the episode control frequency.
- `--no-videos`: store frames as image features instead of MP4 videos.
- `--robot-type`: optional robot type metadata tag.

By default the converted dataset is written to:

```text
datasets/lerobot/odorsim/odorlift/
```

## Tips

- Always start commands from the repo root after `source setup/activate.sh`.
- Use `--no-frames` for quick teleop debugging, but keep frames enabled for LeRobot conversion.
- Use `--odor-monitor log` when you only want terminal ppm output.
- Use `auto_start_gaden=False` when debugging scene export without launching ROS.
- Use `connect_only=True` only when you intentionally started `odor_gaden_rt` yourself.

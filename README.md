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

# 1. Run a quick co-sim script (live continuous voltage, no GADEN)
python - <<'EOF'
import odor_sim as odorsim

with odorsim.make(
    "OdorLift", objects=["mango"], auto_start_gaden=False, odor_mode="continuous"
) as cosim:
    obs = cosim.reset()
    print(obs["instruction"], "action_dim=", cosim.action_dim)
    for _ in range(5):
        obs, reward, done, info = cosim.step(cosim.zero_action())
    print("ppm", info.get("ppm"))
    print("enose_voltage", info.get("enose_voltage"))
    print("enose_voltages", info.get("enose_voltages"))
EOF

# 2. Collect demos with live continuous voltage (or --odor-mode discrete)
python -m odor_sim.bridge.teleop \
  --env OdorLift --objects milk --device keyboard \
  --odor-mode continuous --odor-monitor
```

Teleop always records **ground-truth ppm** and your **e-nose mode** each step. With `--odor-mode continuous` or `discrete`, it also records **live MOX voltage** into `episode.npz` (same sensor model as offline synthesis). Offline synth remains optional if you want the *other* stream or other MOX models without re-mining.

Two mutually exclusive live paths (pick one per session):

- **Continuous** (`--odor-mode continuous`) — always-on MOX; voltage every step. No e-nose keys required for the live stream.
- **Discrete / sniffing** (`--odor-mode discrete`) — valve-gated `SamplingEnose`; press `3`/`4`/`5` as before. Live stream is `enose_voltage_sampling`.

You can still synthesize **both** streams offline from ppm:

```bash
# 3. Optional: synthesize both continuous + sampling voltage for all MOX models
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
- `ClassifyLiquid`: Same lift success condition, but you choose **liquids** (odor classes) instead of objects. Each liquid is shown in a randomly chosen cup mesh every episode (`porcelain_mug`, `white_yellow_mug`, or `red_coffee_mug`), so appearance is decorrelated from the smell the policy must classify. Available liquids: `water` (odorless control), `coconut_water`, `alcohol`, `coke`, `grape_juice`, `wine`, plus pure single-VOC classes for every MOX gas at strengths 0.3 / 0.7 / 1.0 (`ethanol0.3`, `methane1.0`, `hydrogen0.7`, `propanol0.3`, `chlorine1.0`, `fluorine0.7`, `acetone0.3`, … — see `odor_sim/config/liquids.yaml`).
- `ClassifyLiquidPlace`: Same liquid/cup setup as `ClassifyLiquid`, but success is placing the target liquid's cup **on** a catalog place target (default `plate`) — contact, cup above the surface, and XY centers within ~3 cm (LIBERO-style `On`). The place target is odorless so smell classification stays on the liquids.

```python
# Pick the lift target explicitly (need not be first)
with odorsim.make("OdorLift", objects=["milk", "mango"], target_object="mango") as cosim:
    ...

# ClassifyLiquid: liquids + random cups; lift the target liquid
with odorsim.make(
    "ClassifyLiquid",
    liquids=["water", "coke", "wine"],
    target_liquid="coke",
) as cosim:
    ...

# ClassifyLiquidPlace: put the target liquid's cup on a plate
with odorsim.make(
    "ClassifyLiquidPlace",
    liquids=["water", "coke", "wine"],
    target_liquid="coke",
    place_target="plate",
) as cosim:
    ...
```

Useful `odor_sim.make(...)` options:

- `objects=["mango"]`: choose catalog object(s) to spawn (`OdorLift`).
- `target_object="mango"`: which spawned object is the lift target (default: first in `objects`).
- `liquids=["coke", "wine"]`: choose liquid name(s) to spawn (`ClassifyLiquid` / `ClassifyLiquidPlace`); each gets a random cup.
- `target_liquid="coke"`: which spawned liquid is the task target (default: first in `liquids`).
- `place_target="plate"`: catalog object to put the target liquid on (`ClassifyLiquidPlace`).
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
- `sensor_monitor=False`: use `True`, `"log"`, or `"plot"` for live MOX voltage monitoring (requires `odor_mode` continuous/discrete).
- `odor_mode="none"`: live MOX voltage path — `"none"` (ppm only), `"continuous"` (always-on), or `"discrete"` (valve-gated by trailing `enose_state` on the action). Modes are mutually exclusive.
- `mox_model=None`: which MOX sensor(s) to stream. Default/`"all"` runs all five in parallel (`TGS2620`, `TGS2600`, `TGS2611`, `TGS2610`, `TGS2612`). Pass a model id, comma list, or Python list to select a subset. `info["enose_voltages"]` has every active reading; `info["enose_voltage"]` is the primary scalar (`TGS2620` when included).
- `load_resistance=None`: voltage-divider RL [ohms]; default is each model's R0.
- `vcc=5.0`: divider supply voltage.
- Any extra robosuite env kwargs, such as `robots="Panda"`, `has_renderer=True`, `control_freq=20`, or `render_camera="agentview"`.

When `odor_mode` is set, each `step` fills `info["enose_voltage"]` / `info["enose_voltages"]` (and for discrete, `info["enose_state"]` / `info["sampling_active"]`). Session `action_dim` is `env.action_dim + 1` in discrete mode.
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

Teleop drives an `OdorLift` (or other registered task) with a robosuite device while co-stepping the GADEN plume. Each control step records the arm action, e-nose mode, ground-truth ppm at the end effector, proprio, and the task instruction. With `--odor-mode continuous|discrete`, live MOX voltage is recorded too. Episodes are written as one folder per run under `datasets/teleop/`.

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

### ClassifyLiquid teleop

`ClassifyLiquid` is also a lift task, but the class is the **liquid** (VOC recipe), not the cup. You pass `--liquids`; each episode puts every liquid in a randomly chosen mug from the shared cup pool, so vision alone cannot tell the liquids apart. The HUD instruction looks like `pick up the cup of coke`. Success is the same lift check as `OdorLift` (~4 cm above rest on the target liquid).

```bash
python -m odor_sim.bridge.teleop \
  --env ClassifyLiquid \
  --liquids water coke wine \
  --target-liquid coke \
  --device keyboard \
  --odor-monitor
```

Pure single-VOC classes at fixed strengths also work for every MOX gas (`ethanol`, `methane`, `hydrogen`, `propanol`, `chlorine`, `fluorine`, `acetone` × `0.3` / `0.7` / `1.0`), e.g. `--liquids ethanol0.3 acetone1.0`.

Liquid names come from `odor_sim/config/liquids.yaml` (`water`, `coconut_water`, `alcohol`, `coke`, `grape_juice`, `wine`, plus the pure-VOC names above). Cup meshes are `porcelain_mug`, `white_yellow_mug`, and `red_coffee_mug`. Objects are placed uniformly in a 0.6 m × 0.6 m region on the table.

### ClassifyLiquidPlace teleop

`ClassifyLiquidPlace` uses the same liquids and random cups as `ClassifyLiquid`, but the goal is to **put the target liquid's cup on a place target** (default `--place-target plate`). The HUD instruction looks like `put the cup of coke on the plate`. Success is an `On` check (cup contacting the plate, above it, and near its center) — not the lift-height check. After `--success-hold-steps` consecutive successful steps (default 10), the episode auto-ends.

```bash
python -m odor_sim.bridge.teleop \
  --env ClassifyLiquidPlace \
  --liquids water coke wine \
  --target-liquid coke \
  --place-target plate \
  --device keyboard \
  --odor-monitor
```

**Controls during teleop:**

| Input | Action |
|-------|--------|
| robosuite keyboard / SpaceMouse | move arm and gripper (standard robosuite device controls) |
| `3` | e-nose **sample / sniff** — SAMPLE ~7 s (arm hold) → FILTER ~10 s → idle |
| `4` | e-nose **idle** now — cancels an in-flight sample/filter sequence |
| `5` | e-nose **filter / purge** now — cancels an in-flight sequence |
| device reset key (`Ctrl+Q` on keyboard) | end episode manually |

E-nose keys use `3`/`4`/`5` so they stay clear of contested `0`/`1`/`2`. Durations are `--sample-hold-s` (default 7) and `--filter-hold-s` (default 10). Teleop prints a live HUD with phase countdown and logs each key press plus every state transition.

Continuous odor mode needs no e-nose keys for the live voltage stream — ppm and continuous voltage are always recorded when `--odor-mode continuous`. Discrete mode only exposes the sensor to the plume while sampling (`3`); the live stream is the gated voltage. Offline synthesize can still rebuild both streams from ppm + `enose_state`.

**What gets recorded** (`datasets/teleop/episode_YYYYMMDD_HHMMSS/`):

```text
episode.npz          # actions, states, ppm(t), enose_state, proprio
                     # + enose_voltage_continuous OR enose_voltage_sampling when --odor-mode set
meta.json            # instruction, objects, target_object, gas_types, odor_mode, success, ...
frames/agentview/    # camera PNGs (unless --no-frames)
frames/wrist/
```

`meta.json` includes the catalog `objects` list, the resolved `target_object` name, the natural-language `instruction`, `odor_mode` / `mox_model`, and whether the episode ended in task `success`. Live voltage is written when `--odor-mode` is not `none`; offline synthesize remains optional for the other stream / other sensors.

Important teleop options:

- `--env`: task name, default `OdorLift` (also `ClassifyLiquid`, `ClassifyLiquidPlace`).
- `--objects`: catalog object name(s) to spawn (`OdorLift`); default is the task's own default object (`odor_cube`).
- `--target-object`: which spawned object is the lift target (must be one of `--objects`); default is the first object.
- `--liquids`: liquid name(s) to spawn (`ClassifyLiquid` / `ClassifyLiquidPlace`); each is shown in a random cup every episode.
- `--target-liquid`: which spawned liquid is the task target (must be one of `--liquids`); default is the first liquid.
- `--place-target`: catalog object to put the target liquid on (`ClassifyLiquidPlace`); default is `plate`.
- `--scenario`: scenario name or config directory, default `10x6_uniform`.
- `--device`: `keyboard` or `spacemouse`, default `keyboard`.
- `--robots`: robosuite robot name, default `Panda`.
- `--controller`: robosuite composite controller name or config path.
- `--camera`: on-screen viewer camera, default `agentview`.
- `--pos-sensitivity`: position input scale, default `1.0`.
- `--rot-sensitivity`: rotation input scale, default `1.0`.
- `--goal-update-mode`: `target` or `achieved`, default `target`.
- `--sample-hold-s`: seconds to auto-hold when sampling, default `7.0`.
- `--filter-hold-s`: seconds of auto FILTER after sample before idle, default `10.0`.
- `--success-hold-steps`: successful steps before auto-ending, default `10`; use `0` to disable.
- `--out-dir`: raw dataset output directory, default `datasets/teleop`.
- `--no-frames`: disable camera-frame recording.
- `--camera-size`: recorded frame size, default `256`.
- `--no-bridge`: drive robosuite only, without GADEN / ppm.
- `--odor-mode`: `none` (default, ppm only), `continuous`, or `discrete` live voltage.
- `--mox-model`: zero or more MOX ids. Omit = all five sensors in parallel. Example: `--mox-model TGS2620 TGS2600`.
- `--vcc`: divider supply voltage, default `5.0`.
- `--odor-monitor [MODE]`: live ppm monitor. Use no value for log + plot, or `log` / `plot`.
- `--sensor-monitor [MODE]`: live MOX voltage monitor (needs `--odor-mode continuous|discrete`). Same MODE shapes as `--odor-monitor`.

## Convert Data To LeRobot

The data pipeline can skip synthesis when teleop already wrote live voltage into
`episode.npz`. Otherwise: synthesize odor sensor features, then convert to LeRobot.

Optional synthesize (writes `features.npz` next to each episode; useful for the
other stream / other MOX models):

```bash
source setup/activate.sh

python -m odor_sim.recording.synthesize datasets/teleop
```

This replays each episode's ppm(t) through the MOX model twice:

- **Continuous** → `enose_voltage_continuous` (always exposed to ppm)
- **Sniffing** → `enose_voltage_sampling` (only exposed while `enose_state == 1`)

By default **every** MOX model is synthesized. Each model writes its own
`features_<MODEL>.npz` / `features_meta_<MODEL>.json` pair, and the primary model
(`TGS2620`) is also mirrored into the canonical `features.npz` / `features_meta.json`
that the LeRobot export consumes. This lets you compare sensors offline without
re-teleoping. Pass `--mox-model X` to synthesize just model `X` into `features.npz`:

```bash
# One episode, single sensor only
python -m odor_sim.recording.synthesize datasets/teleop/episode_<timestamp> \
  --mox-model TGS2600 --no-overwrite
```

Both streams land in `features.npz` and are included in the LeRobot export when you convert.

Useful synthesis options:

- `--mox-model`: MOX model. Omit to synthesize all models (`TGS2620`, `TGS2600`, `TGS2611`, `TGS2610`, `TGS2612`); the default export uses `TGS2620`.
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
- Use `--sensor-monitor log` (with `--odor-mode continuous|discrete`) for terminal MOX voltage output.
- Use `auto_start_gaden=False` when debugging scene export without launching ROS.
- Use `connect_only=True` only when you intentionally started `odor_gaden_rt` yourself.

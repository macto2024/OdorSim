# OdorSim co-simulation

## Recommended entry point

```python
import odor_sim as odorsim

with odorsim.make("OdorLift", objects=["mango"]) as cosim:
    obs = cosim.reset()
    action = cosim.zero_action()
    obs, reward, done, info = cosim.step(action)
    print(info["ppm"])
```

## Tasks

- **`OdorLift`**: Lift-style manipulation. Spawns one or more catalog objects; choose which is the lift target (default: the first). Extra objects are odor-emitting distractors. Success when the target rises ~4 cm above its resting height.
- **`ClassifyLiquid`**: Same lift success, but you choose **liquids** (odor classes) instead of objects. Each liquid is shown in a randomly chosen cup mesh every episode (`porcelain_mug`, `white_yellow_mug`, or `red_coffee_mug`), so appearance is decorrelated from smell. Available liquids: `water`, `coconut_water`, `alcohol`, `coke`, `grape_juice`, `wine`, `methanol`, `acetic_acid`, `ethyl_acetate`, `toluene`, plus pure single-VOC classes for every MOX gas at strengths 0.3 / 0.7 / 1.0 (see `odor_sim/config/liquids.yaml`).
- **`ClassifyLiquidPlace`**: Same liquid/cup setup as `ClassifyLiquid`, but success is placing the target liquid's cup **on** a catalog place target (default `plate`) — contact, cup above the surface, and XY centers within ~3 cm (LIBERO-style `On`). The place target is odorless. Scripted smell→decide→place: [scripted-odor-place-collection.md](scripted-odor-place-collection.md).
- **`OdorPlace`**: Vision-only pick-and-place of a catalog mug onto a plate (no odor classes). Scripted OpenVLA-OFT demos: [scripted-collection.md](scripted-collection.md).


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

## Useful `odor_sim.make(...)` options

- `objects=["mango"]`: catalog object(s) to spawn (`OdorLift`).
- `target_object="mango"`: which spawned object is the lift target (default: first in `objects`).
- `liquids=["coke", "wine"]`: liquid name(s) (`ClassifyLiquid` / `ClassifyLiquidPlace`); each gets a random cup.
- `target_liquid="coke"`: which spawned liquid is the task target (default: first in `liquids`).
- `place_target="plate"`: catalog object to put the target liquid on (`ClassifyLiquidPlace`).
- `scenario="10x6_uniform"`: scenario under `scenarios/`, or a direct config directory.
- `scenario_config="config1"`: config under `environment_configurations/`.
- `scene_id=None`: override the generated scene name.
- `auto_start_gaden=True`: start the GADEN server automatically.
- `connect_only=False`: connect to a server you already started.
- `bridge=True`: enable ROS bridge and ppm queries.
- `export=True`: export the GADEN scene from the robosuite env before starting.
- `server_log_dir=None`: where server logs are written.
- `wait_timeout=60.0`: seconds to wait for server readiness.
- `step_on_timer=False`: free-run GADEN instead of bridge lockstep.
- `publish_markers=False`: RViz marker publishing.
- `odor_monitor=False`: `True`, `"log"`, or `"plot"` for live ppm monitoring.
- `sensor_monitor=False`: `True`, `"log"`, or `"plot"` for live MOX voltage (needs `odor_mode` continuous/discrete).
- `odor_mode="none"`: live MOX path — `"none"` (ppm only), `"continuous"`, or `"discrete"` (valve-gated by trailing `enose_state` on the action). Modes are mutually exclusive.
- `mox_model=None`: which MOX sensor(s) to stream. Default uses the fixed eight-sensor OVLA array; `"all"` runs all 17 Excel-calibrated models. Pass a model id, comma list, or Python list. `info["enose_voltages"]` has every active reading; `info["enose_voltage"]` is the primary scalar (`101` when included).
- `load_resistance=None`: voltage-divider RL [ohms]; default is each model's calibrated divider value.
- `vcc=5.0`: divider supply voltage.
- Extra robosuite kwargs: `robots="Panda"`, `has_renderer=True`, `control_freq=20`, `render_camera="agentview"`, etc.

When `odor_mode` is set, each `step` fills `info["enose_voltage"]` / `info["enose_voltages"]` (and for discrete, `info["enose_state"]` / `info["sampling_active"]`). Session `action_dim` is `env.action_dim + 1` in discrete mode.

### Dry run (env + scene export, no ROS)

```python
with odorsim.make("OdorLift", objects=["mango"], auto_start_gaden=False) as cosim:
    print(cosim.scene_path)
```

### Connect to a manually started server

```python
with odorsim.make("OdorLift", objects=["mango"], connect_only=True, scene_id="rt_scene") as cosim:
    obs = cosim.reset()
```

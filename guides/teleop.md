# Teleoperate and record data

Teleop drives a registered task with a robosuite device while co-stepping the GADEN plume. Each control step records the arm action, e-nose mode, ground-truth ppm at the end effector, proprio, and the task instruction. With `--odor-mode continuous|discrete`, live MOX voltage is recorded too. Episodes land under `datasets/teleop/`.

Always:

```bash
source setup/activate.sh
```

## OdorLift

Spawns catalog objects; lift the target (~4 cm). HUD shows e.g. `pick up the mango`. After `--success-hold-steps` consecutive successes (default 10), the episode auto-ends.

```bash
# Single object
python -m odor_sim.bridge.teleop \
  --env OdorLift --objects milk --device keyboard \
  --odor-monitor --out-dir datasets/teleop

# Multiple objects (first is target by default)
python -m odor_sim.bridge.teleop \
  --env OdorLift --objects mango milk --device keyboard --odor-monitor

# Explicit target
python -m odor_sim.bridge.teleop \
  --env OdorLift --objects milk mango --target-object mango \
  --device keyboard --odor-monitor
```

## ClassifyLiquid

Class is the **liquid**, not the cup. Cups are randomly assigned each episode.

```bash
python -m odor_sim.bridge.teleop \
  --env ClassifyLiquid \
  --liquids water coke wine --target-liquid coke \
  --device keyboard --odor-monitor
```

Pure single-VOC classes work too (`ethanol`, `acetone`, `ethanol0.3`, …). The Excel gases are `ethanol`, `methanol`, `acetone`, `acetic_acid`, `ethyl_acetate`, and `toluene`. Liquid names: `odor_sim/config/liquids.yaml`. Cups: `porcelain_mug`, `white_yellow_mug`, `red_coffee_mug`.

## ClassifyLiquidPlace

Same liquids/cups; put the target cup **on** `--place-target` (default `plate`).

```bash
python -m odor_sim.bridge.teleop \
  --env ClassifyLiquidPlace \
  --liquids water coke wine --target-liquid coke \
  --place-target plate --device keyboard --odor-monitor
```

## ClassifyLiquidPlaceOVLA + live ScentFormer

OVLA layout: matched white cups, smell is the only cue. Liquids must be from the six Excel VOCs when using the classifier: `ethanol`, `methanol`, `acetone`, `acetic_acid`, `ethyl_acetate`, `toluene`.

`--odor-classifier` loads `Smellnet_chkpt/phase_b_best_loss.pt` and runs batched inference on all ready fixed windows (3/5/7/10/15/20 s) from live MOX voltages (training-time preprocessing: `(V - baseline) / baseline`). Requires `--odor-mode continuous` or `discrete`, `torch`, and the default eight-sensor MOX array (omit `--mox-model`).

```bash
python -m odor_sim.bridge.teleop \
  --env ClassifyLiquidPlaceOVLA \
  --liquids ethanol toluene --target-liquid ethanol --place-target plate \
  --device keyboard \
  --odor-mode continuous \
  --odor-classifier \
  --sensor-monitor
```

Opens a second matplotlib window with one row per window length (top gas + confidence). After each reset, the panel shows ground truth (`left` / `right` cup → liquid). The terminal HUD includes e.g. `smell@20s=ethanol 94%` when the 20 s window is ready.

Optional: `--odor-classifier /path/to/phase_b_best_loss.pt` for a custom checkpoint; pass `device` via `odor_sim.make(..., odor_classifier={"device": "cuda"})` when scripting.

## Live odor modes

Teleop always records **ppm** and **e-nose mode**. With `--odor-mode continuous` or `discrete`, it also records live MOX voltage (same model as offline synthesis).

| Mode | Behavior |
|------|----------|
| `continuous` | Always-on MOX; voltage every step. No e-nose keys needed for the live stream. |
| `discrete` | Valve-gated `SamplingEnose`; press `3`/`4`/`5`. Live stream is `enose_voltage_sampling`. |

Offline synth remains optional for the other stream or other MOX models — see [lerobot.md](lerobot.md).

## Controls

| Input | Action |
|-------|--------|
| robosuite keyboard / SpaceMouse | move arm and gripper |
| `3` | e-nose **sample / sniff** — SAMPLE ~7 s → FILTER ~10 s → idle |
| `4` | e-nose **idle** now |
| `5` | e-nose **filter / purge** now |
| device reset (`Ctrl+Q` on keyboard) | end episode manually |

Durations: `--sample-hold-s` (default 7), `--filter-hold-s` (default 10).

## Episode layout

```text
datasets/teleop/episode_YYYYMMDD_HHMMSS/
  episode.npz          # actions, states, ppm(t), enose_state, proprio
                       # + live voltage when --odor-mode set
  meta.json            # instruction, objects/liquids, odor_mode, success, ...
  frames/agentview/
  frames/wrist/
```

## Important CLI options

- `--env`: `OdorLift` (default), `ClassifyLiquid`, `ClassifyLiquidPlace`, `ClassifyLiquidPlaceOVLA`
- `--objects` / `--target-object`: `OdorLift`
- `--liquids` / `--target-liquid` / `--place-target`: liquid tasks
- `--scenario`: default `10x6_uniform`
- `--device`: `keyboard` or `spacemouse`
- `--odor-mode`: `none` | `continuous` | `discrete`
- `--odor-classifier` [`CKPT`]: live ScentFormer panel (needs `--odor-mode continuous|discrete`)
- `--mox-model`: omit = fixed eight; `all` = all 17; e.g. `--mox-model 101 TGS2600`
- `--odor-monitor` / `--sensor-monitor`: `log` | `plot` | both (no value)
- `--out-dir`, `--no-frames`, `--camera-size`, `--success-hold-steps`, `--no-bridge`

## Tips

- Use `--no-frames` for quick debugging; keep frames for LeRobot conversion.
- Use `--odor-monitor log` / `--sensor-monitor log` for terminal-only monitors.

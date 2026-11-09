# Scripted odor pick-place collection

Headless **smell → decide → pick-place** demos on **`ClassifyLiquidPlaceOVLA`**. Two identical porcelain cups sit in randomized left/right slots and both physical gas sources remain active. The arm sniffs cup 1 and, when it is wrong, moves directly to cup 2 without resetting the MOX state.

Use this for odor-conditioned OVLA policies. For **vision-only** mug→plate demos, see [scripted-collection.md](scripted-collection.md) (`collect_place_oft` / `OdorPlace`).

Ground-truth liquid IDs drive the scripted decision so demos succeed reliably. Sensing is continuous at 20 Hz using the fixed eight-channel array. Main collection uses zero wind, disabled original terminal velocity, disabled experimental buoyancy, and the calibrated separated emission model (`exponent=0.5`, `scale=0.06`, `15 filaments/s`).

The two requested local source conditions are paired rather than independently
sampled: `C_A=C_base*r`, `C_B=C_base/r`, with
`r ~ LogUniform(0.8, 1.25)`. Cup slots produce approximately 22–29 cm XY
spacing. A grasp/reject decision is retained only when the final causal second
has median total concentration at least 5 ppm, median intended-gas dominance at
least 70%, and at least 80% of samples pass both evidence thresholds.

## Quick collect

```bash
source setup/activate.sh

# Parallel MuJoCo + GADEN workers (each gets its own ROS_DOMAIN_ID)
python -m odor_sim.bridge.collect_odor_place_oft \
  --out-dir /root/Odorbot/datasets/odor_ovla_oft \
  --num-demos 1500 \
  --num-workers 2 \
  --quiet
```

Resume-safe via `manifest.csv`: re-run the same command; finished plan indices are
skipped. The manifest fixes target-first/target-second order before collection,
so a failed target-second attempt cannot retry as target-first. Stop a serial run
before switching to workers so domain IDs / `kill_all` stay clean.

Single-episode / debug:

```bash
python -m odor_sim.bridge.scripted_odor_place \
  --liquids ethanol methanol \
  --instructed ethanol \
  --num-demos 5 \
  --out-dir datasets/teleop/odor_place_scripted
```

Build RLDS. This keeps sparse sniff HOLD landmarks at approximately
0/0.25/0.5/1/2/3/5/7/9 seconds plus the final decision frame, drops the redundant
20-Hz hover ticks, and attaches the causal odor prefix available at each
retained frame:

```bash
python -m odor_sim.recording.build_rlds_odor_ovla \
  --input /root/Odorbot/datasets/odor_ovla_oft/raw \
  --manifest /root/Odorbot/datasets/odor_ovla_oft/manifest.csv \
  --output /root/Odorbot/datasets/rlds \
  --dataset-name odorsim_ovla_odorplace_sniff10s_hold_v2 \
  --odor-history-s 5
```

Validate a 30-episode pilot before scaling:

```bash
python -m odor_sim.recording.validate_odor_ovla_collection \
  --dataset datasets/odor_ovla_physical_pilot_v3 \
  --expected-episodes 30 \
  --expected-pairs 30
```

The command writes `validation.json` inside the dataset and returns a non-zero
exit status when a required acceptance criterion fails. Dominance >=90% on 70%
of decisions is reported as a preference warning rather than a hard failure.

Old raw demonstrations collected with discrete sampling, the legacy emission
mapping, terminal velocity, or a forced FILTER phase must be recollected.
Rebuilding RLDS alone cannot change their physical sensor history. The field
`odor_padding_mask` is `true` for left-padding and must be passed to the
ScentFormer attention mask. Do not apply another no-op/zero-action filter in
the OVLA input pipeline, because the protected zero-delta actions are the HOLD
supervision.

RLDS stores the raw rolling five-second MOX window. The frozen encoder's
deployment API is `model.encode(...)`; pass only its latent odor embedding to
OVLA. Do not concatenate gas probabilities, presence, intensity, or recovery.

## What each episode does

Default scene: **2 porcelain cups** drawn uniformly from the ScentFormerAB VOC pool
(`ethanol`, `methanol`, `acetone`, `acetic_acid`, `ethyl_acetate`, `toluene`).

1. Hover the e-nose about 4 cm above a randomly chosen first source and HOLD (`--sniff-duration`, default 10 s). The gripper remains 8 cm high; the sensor site is mounted 4 cm below it.
2. If that cup is the instructed liquid → grasp immediately from the sniff hover and place on the plate
3. Else → raise to a collision-safe high staging pose, move toward the other cup
   without a dwell/reset, descend and HOLD → grasp target. MOX state remains
   continuous throughout this direct transition.

Language instruction looks like: `put the cup of ethanol on the plate`.

## Layout

```text
<out-dir>/
  raw/scene_two_cup/episode_*/
    episode.npz          # control-rate stream + phase tags
    meta.json
    frames/{agentview,wrist}/
    odor_events.npz      # (N_sniff, T_sniff, 8) voltages
    odor_events.json
  manifest.csv
```

Raw event traces retain the full 10-second sniff. RLDS uses a rolling 5-second history (`100 x 8` at 20 Hz) at each retained action step, left-padded only near episode startup.

## Important flags

| Flag | Default | Meaning |
|------|---------|---------|
| `--num-demos` | 50 | Successful episodes to collect |
| `--voc-pool` | 6 ScentFormer gases | Uniform draw per episode |
| `--num-cups` | 2 | Cups on the table |
| `--absent-frac` | 0.0 | Fraction where instructed liquid is not on the table |
| `--instructed` | (sampled) | Pin the language target; default samples present/absent |
| `--sniff-duration` | 10 | Seconds hovering above a cup (SAMPLE) |
| `--transition-mode` | `direct` | Direct carryover path; `physical_filter` is an ablation |
| `--target-first-frac` | 0.5 | P(sniff instructed cup first) when present |
| `--mox-models` | 8 Smellnet sensors | Encoder channel order |
| `--odor-mode` | `continuous` | `continuous` is required for deployment collection |
| `--num-workers` | 1 | Parallel processes (each owns ROS domain + GADEN) |
| `--ros-domain-base` | 50 | Worker `i` uses `ROS_DOMAIN_ID=base+i` |

Motion / grasp knobs match the vision collector (`--grasp-dwell`, speed shaping, etc. via `scripted_place` helpers).

## Gas types and MOX curves

Collection ppm IDs come from `odor_sim/config/gas_types.py`. Live voltages use the eight default sensors and Excel curves in `odor_sim/config/mox_sensors.yaml` (`101`, `203`, `302`, `401`, `TGS2600`, `TGS2602`, `TGS2610`, `TGS2611`).

OVLA liquids (`ethanol`, `methanol`, …) declare that six-gas Excel panel. Only the named VOC emits; the rest are strength 0 so GADEN does not spawn extra sources. Cup / instruction IDs are unsuffixed; `ethanol1.0` still works as an alias. `meta.json` / `episode.npz` ppm columns stay `ethanol, methanol, acetone, acetic_acid, ethyl_acetate, toluene`.

Methanol, acetic acid, ethyl acetate, and toluene are GADEN IDs 14–17. `make()` refuses to start `odor_gaden_rt` if the binary was built without `patches/gaden_core_add_excel_gases.patch`. After applying the patch:

```bash
source /opt/ros/jazzy/setup.bash
cd /root/OdorSim/gaden && colcon build --symlink-install --packages-select gaden_common
source /root/OdorSim/gaden/install/setup.bash
cd /root/OdorSim/ros2_ws && colcon build --symlink-install --packages-select odor_gaden_rt
source /root/OdorSim/setup/activate.sh
```

Discard or re-collect episodes whose `odor_events.npz` voltages are all zeros, or that crashed with `Invalid gas type: 14`.

## Related

- Stationary odor-only sniffs (no pick-place): [odor-sniff-collection.md](odor-sniff-collection.md)
- Vision pick-place: [scripted-collection.md](scripted-collection.md)
- Raw / RLDS field contract: [../docs/SCHEMA.md](../docs/SCHEMA.md) (`ovla-odor-0.1`)
- Task API: [co-simulation.md](co-simulation.md) (`ClassifyLiquidPlaceOVLA`)

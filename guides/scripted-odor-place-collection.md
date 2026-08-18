# Scripted odor pick-place collection

Headless **smell → decide → pick-place** demos on **`ClassifyLiquidPlaceOVLA`**. Two identical porcelain (white) cups sit in tight left/right slots with a plate; the arm sniffs a random first cup (SAMPLE), purges between cups (FILTER), then either places the instructed liquid on the plate or returns home if it is absent.

Use this for odor-conditioned OVLA policies. For **vision-only** mug→plate demos, see [scripted-collection.md](scripted-collection.md) (`collect_place_oft` / `OdorPlace`).

Ground-truth liquid IDs drive the scripted decision so demos succeed reliably; recorded SAMPLE voltages are what a learned policy must use. Default **`odor_mode=discrete`** (GADEN + MOX array). Default sensors and VOC names match the Smellnet / ScentFormerAB checkpoint.

## Quick collect

```bash
source setup/activate.sh

# Parallel MuJoCo + GADEN workers (each gets its own ROS_DOMAIN_ID)
python -m odor_sim.bridge.collect_odor_place_oft \
  --out-dir /root/Odorbot/datasets/odor_ovla_oft \
  --num-demos 50 \
  --num-workers 4 \
  --quiet
```

Resume-safe via `manifest.csv`: re-run the same command; finished episodes are skipped. Stop a serial run before switching to workers so domain IDs / `kill_all` stay clean.

Single-episode / debug:

```bash
python -m odor_sim.bridge.scripted_odor_place \
  --liquids ethanol methanol \
  --instructed ethanol \
  --num-demos 5 \
  --out-dir datasets/teleop/odor_place_scripted
```

Build RLDS (collapses sniff/filter hovers; attaches `observation.odor`):

```bash
python -m odor_sim.recording.build_rlds_odor_ovla \
  --input /root/Odorbot/datasets/odor_ovla_oft/raw \
  --manifest /root/Odorbot/datasets/odor_ovla_oft/manifest.csv \
  --output /root/Odorbot/datasets/rlds \
  --dataset-name odorsim_ovla_odorplace_sniff10s
```

## What each episode does

Default scene: **2 porcelain cups** drawn uniformly from the ScentFormerAB VOC pool
(`ethanol`, `methanol`, `acetone`, `acetic_acid`, `ethyl_acetate`, `toluene`).

1. Hover above a randomly chosen first cup → **SAMPLE** hold (`--sniff-duration`, default 10 s)
2. If that cup is the instructed liquid → grasp immediately from the sniff hover and place on the plate
3. Else → move aside → **FILTER** hold (default `1.5 × sniff`) → try the next cup
4. If no cup matches → return to the post-reset home pose (instructed liquid **absent**)

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

`T_sniff = round(odor_sampling_rate * sniff_duration_s)` (200 at the 10 s / 20 Hz default). Changing sniff duration is a new collection run (put it in a separate `--out-dir` / RLDS name).

## Important flags

| Flag | Default | Meaning |
|------|---------|---------|
| `--num-demos` | 50 | Successful episodes to collect |
| `--voc-pool` | 6 ScentFormer gases | Uniform draw per episode |
| `--num-cups` | 2 | Cups on the table |
| `--absent-frac` | 0.2 | Fraction where instructed liquid is not on the table |
| `--instructed` | (sampled) | Pin the language target; default samples present/absent |
| `--sniff-duration` | 10 | Seconds hovering above a cup (SAMPLE) |
| `--filter-hold-s` | 1.5× sniff | FILTER nose-rest after a wrong cup |
| `--target-first-frac` | 0.5 | P(sniff instructed cup first) when present |
| `--mox-models` | 8 Smellnet sensors | Encoder channel order |
| `--odor-mode` | `discrete` | `discrete` / `continuous` / `none` |
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

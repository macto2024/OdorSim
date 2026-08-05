# Scripted odor pick-place collection

Headless **smell → decide → pick-place** demos on **`ClassifyLiquidPlace`**. The arm sniffs each cup (SAMPLE), purges between cups (FILTER), then either places the instructed liquid on the plate or returns home if it is absent.

Use this for odor-conditioned VLA / multimodal policies. For **vision-only** mug→plate demos, see [scripted-collection.md](scripted-collection.md) (`collect_place_oft` / `OdorPlace`).

Ground-truth liquid IDs drive the scripted decision so demos succeed reliably; recorded SAMPLE/FILTER voltages are what a learned policy must use. Default **`odor_mode=discrete`** (GADEN + FakeGasSensor).

## Quick collect

```bash
source setup/activate.sh

# Parallel MuJoCo + GADEN workers (each gets its own ROS_DOMAIN_ID)
python -m odor_sim.bridge.collect_odor_place_oft \
  --out-dir /path/to/Odorbot/datasets/odor_place_oft \
  --num-demos 50 \
  --num-workers 4 \
  --quiet
```

Resume-safe via `manifest.csv`: re-run the same command; finished episodes are skipped. Stop a serial run before switching to workers so domain IDs / `kill_all` stay clean.

Single-episode / debug:

```bash
python -m odor_sim.bridge.scripted_odor_place \
  --liquids ethanol1.0 propanol1.0 \
  --instructed ethanol1.0 \
  --num-demos 5 \
  --out-dir datasets/teleop/odor_place_scripted
```

## What each episode does

Default scene: **2 cups** drawn uniformly from a 6-VOC pool at strength 1.0
(`ethanol1.0`, `propanol1.0`, `acetone1.0`, `methane1.0`, `hydrogen1.0`, `chlorine1.0`).

1. Hover above a cup → **SAMPLE** hold (`--sample-hold-s`, default 10 s)
2. If that cup is the instructed liquid → grasp and place on the plate
3. Else → move aside → **FILTER** hold (`--filter-hold-s`, default 15 s) → try the next cup
4. If no cup matches → return to the post-reset home pose (instructed liquid **absent**)

Language instruction looks like: `put the cup of ethanol on the plate` (strength suffix dropped in text).

## Layout

```text
<out-dir>/
  raw/episode_*/
    episode.npz
    meta.json
    frames/...
  manifest.csv
```

## Important flags

| Flag | Default | Meaning |
|------|---------|---------|
| `--num-demos` | 50 | Successful episodes to collect |
| `--voc-pool` | 6× `*1.0` VOCs | Uniform draw per episode |
| `--num-cups` | 2 | Cups on the table |
| `--absent-frac` | 0.2 | Fraction where instructed liquid is not on the table |
| `--instructed` | (sampled) | Pin the language target; default samples present/absent |
| `--sample-hold-s` / `--filter-hold-s` | 10 / 15 | Sniff and purge durations |
| `--target-first-frac` | 0.4 | P(sniff target cup first) when present |
| `--odor-mode` | `discrete` | `discrete` / `continuous` / `none` |
| `--num-workers` | 1 | Parallel processes (each owns ROS domain + GADEN) |
| `--ros-domain-base` | 50 | Worker `i` uses `ROS_DOMAIN_ID=base+i` |

Motion / grasp knobs match the vision collector (`--grasp-dwell`, speed shaping, etc. via `scripted_place` helpers).

## Related

- Stationary odor-only sniffs (no pick-place): [odor-sniff-collection.md](odor-sniff-collection.md)
- Vision pick-place: [scripted-collection.md](scripted-collection.md)
- Task API: [co-simulation.md](co-simulation.md) (`ClassifyLiquidPlace`)

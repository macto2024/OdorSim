# Odor-only sniff collection

Collect **labeled MOX voltage time series** without robot trajectories or camera frames. MuJoCo/robosuite is only used as a pose + odor-source provider; the arm parks (optionally approaches the source) and holds while GADEN + the FakeGasSensor MOX models stream voltages.

Use this for ScentFormer-style sim classifiers (pure VOCs or multi-VOC objects).

## Quick start

Preferred for pure VOC classification — collects strengths **0.3 / 0.7 / 1.0** but labels only the VOC name (`ethanol`, not `ethanol1.0`):

```bash
source setup/activate.sh

python -m odor_sim.bridge.collect_odor \
  --vocs ethanol methane acetone hydrogen propanol chlorine \
  --duration-s 20 --odor-mode continuous \
  --num-per-label 50 --num-workers 4 \
  --window-s 10 --stride-s 5 \
  --out-dir datasets/odor_sniffs/voc6_continuous
```

`--num-workers N` runs **N parallel processes** (like `collect_odor_place_oft`). Each worker gets its own `ROS_DOMAIN_ID` (base `--ros-domain-base`, default 40) and GADEN server so topics do not collide. Resume via `manifest.csv` still works — **stop any serial run first**, then relaunch with workers; finished sniffs are skipped.

With `--vocs`, sniff indices cycle strengths (`sniff_000`→0.3, `001`→0.7, `002`→1.0, `003`→0.3, …). All land under `ethanol/` with `label=ethanol`; meta/manifest keep `spawn=ethanol0.3` and `strength=0.3`.

Exact catalog liquids (label = liquid name):

```bash
python -m odor_sim.bridge.collect_odor \
  --liquids ethanol1.0 methane1.0 acetone1.0 \
  --duration-s 20 --odor-mode continuous \
  --num-per-label 20 \
  --out-dir datasets/odor_sniffs/voc3_s1
```

Discrete (valve SAMPLE → FILTER → IDLE cycle):

```bash
python -m odor_sim.bridge.collect_odor \
  --vocs ethanol methane acetone \
  --duration-s 20 --odor-mode discrete \
  --sample-hold-s 7 --filter-hold-s 10 \
  --num-per-label 20 \
  --out-dir datasets/odor_sniffs/voc3_discrete
```

Multi-VOC catalog objects:

```bash
python -m odor_sim.bridge.collect_odor \
  --objects mango milk wine_bottle \
  --duration-s 20 --odor-mode continuous \
  --num-per-label 30 \
  --out-dir datasets/odor_sniffs/objects_continuous
```

Dry-run (print plan, no GADEN):

```bash
python -m odor_sim.bridge.collect_odor \
  --vocs ethanol --odor-mode continuous --num-per-label 6 --dry-run
```

## What is recorded

Each sniff is stored under `--out-dir/<label>/`:

```text
datasets/odor_sniffs/voc6_continuous/
  manifest.csv
  ethanol1.0/
    sniff_000.npz
    sniff_000_meta.json
    windows/sniff_000_windows.npz   # if --window-s > 0
  ...
  windows/
    dataset.npz    # consolidated X (N, Tw, C), y, sniff_id, label_names
    classes.json
```

`sniff_*.npz` fields:

| Array | Shape | Notes |
|-------|-------|--------|
| `voltages` | `(T, C)` | MOX voltages; columns = `mox_models` (default all 5 TGS) |
| `ppm` | `(T, G)` | Ground-truth ppm at EE |
| `enose_state` | `(T,)` | `1` SAMPLE / `0` IDLE / `-1` FILTER |
| `sim_time` | `(T,)` | GADEN sim time |
| `source_distance` | `(T,)` | EE site → target object (m) |
| `label` | scalar | Class name string |

`T = duration_s * control_freq` (default 20 × 20 = 400). Default window export: **10 s** windows (`Tw=200`) with **5 s** stride.

## Important flags

| Flag | Default | Meaning |
|------|---------|---------|
| `--vocs` | | Base VOC names; expands to 0.3/0.7/1.0, labels as VOC only |
| `--liquids` / `--objects` | (one of vocs/liquids/objects) | Exact liquids or catalog objects |
| `--strengths` | 0.3 0.7 1.0 | Strengths for `--vocs` |
| `--duration-s` | 20 | Sniff length |
| `--odor-mode` | required | `continuous` or `discrete` |
| `--num-per-label` | 50 | Sniffs per class label (strengths cycled with `--vocs`) |
| `--num-workers` | 1 | Parallel processes (each owns a ROS domain + GADEN) |
| `--ros-domain-base` | 40 | Worker i uses `ROS_DOMAIN_ID=base+i` |
| `--control-freq` | 20 | Sensor / control rate (Hz) |
| `--mox-model` | all five | Subset of TGS models if desired |
| `--approach-target` | on | OSC move above source, then hold |
| `--no-approach-target` | | Hold at reset EE pose |
| `--window-s` / `--stride-s` | 10 / 5 | Set `--window-s 0` to skip window export |
| `--seed` | 0 | Sniff `i` uses `seed + i` |

Resume: re-run the same command; completed rows in `manifest.csv` and existing `sniff_XXX.npz` files are skipped.

## Train ScentFormer on collected windows

SmellNet's `models/run.py` expects CSV folders, so use the OdorSim helper that loads `windows/dataset.npz` directly (ScentFormer code from `/root/SmellNet`):

```bash
source setup/activate.sh
# needs torch (SmellNet venv if OdorSim venv lacks it)

PYTHONPATH=/root/OdorSim /root/SmellNet/.venv/bin/python -m odor_sim.bridge.train_scentformer \
  --data datasets/odor_sniffs/voc6_continuous/windows/dataset.npz \
  --epochs 40 --batch-size 32 --lr 1e-3 \
  --val-frac 0.2
```

This:

1. Loads `X (N, 200, 5)`, `y`, `sniff_id`, `label_names`
2. Splits **train/val by sniff id** (no window leakage)
3. Standardizes channels on the train split
4. Trains SmellNet's Transformer (ScentFormer) classifier
5. Writes `windows/scentformer_run/scentformer.pt` + `metrics.json`

Optional: `--baseline-subtract` (subtract first timestep per window).

## Training tips

- Split **train/val by sniff id** (see `sniff_id` in `windows/dataset.npz`), not by random windows from the same sniff.
- Keep **odor_mode** the same at train and eval.
- Baseline subtract / standardize / temporal differencing belong in the **training** pipeline (SmellNet-style), not in this collector.
- Pure VOC liquids and multi-VOC objects are different tasks — use separate `--out-dir` trees.
- This is **sim TGS FakeGasSensor** data, not real MQ/Grove SmellNet hardware.

## Considerations

- **Geometry:** distance and plume intermittency dominate the voltage shape. Random object placement plus `--approach-target` give diversity.
- **Transients:** MOX `tau_rise` is ~3–5 s; a 20 s sniff with 10 s windows is a reasonable starting point.
- **Cost:** each sniff starts a GADEN session. Use `--num-workers 4` (etc.) for parallel collection; keep renderer/cameras off (the script already does). Stop any serial run before switching to workers so `kill_all` / domain IDs stay clean.
- **Labels:** liquid names from `odor_sim/config/liquids.yaml`; objects from `odor_sim/config/objects.yaml`.

# Staged Odor Encoder Data (Legacy)

> This document describes the pre-schema-v4 synthetic/single-source pipeline.
> For new collection and training use `guides/odor-encoder-abcd-collection.md`.


Run from the OdorSim repository after `source setup/activate.sh`.

Install the optional trainer dependency once with
`python -m pip install -e '.[training]'`. Confirm CUDA is visible with
`python -c "import torch; print(torch.cuda.is_available())"` before a full run.

The fixed gas order is `ethanol, methanol, acetone, acetic_acid,
ethyl_acetate, toluene`. The fixed channel order is `101, 203, 302, 401,
TGS2600, TGS2602, TGS2610, TGS2611`.

## Pilot collection

```bash
python -m odor_sim.odor_encoder.phase_a --sessions 5000
python -m odor_sim.odor_encoder.phase_b --trajectories 1200
```

The default output directories are `datasets/odor_encoder/static_single_v3` and
`datasets/odor_encoder/virtual_single_v6`. Phase B V6 must be recollected because
it adds randomized airflow while sampling the e-nose 0.01-0.05 m above the target.
The new path preserves earlier trajectory geometries for comparison.
Every `.npz` is one complete static
session or virtual trajectory. Files are
split before causal windows are sampled; do not move individual windows across
the `train`, `validation`, and `test` directories.

V6 intentionally uses uncontained point sources; its metadata records
`containment_model: uncontained_point_sources`. It does not claim to model cup
walls or table collision geometry.

Each non-clean episode has one labeled primary gas. Phase B places one or two
ordinary gas-emitting neighbors at a true center-to-center distance of
0.20-0.30 m. They are placed upwind (25%), crosswind (50%), or downwind (25%)
relative to mean episode flow. The realized distances and orientations are
stored in episode metadata. A timestep is a valid supervised
endpoint only when the primary gas is at least 0.1 ppm and at least 75% of
the local measured mixture. A three-second clean recovery tail remains valid
to represent MOX lag. Virtual trajectories last 24-32 seconds so the slowest
MOX channels can supply valid 20-second histories. Phase A uses step,
constant, ramp-up, ramp-down, and incomplete-recovery profiles. All exposed
profiles have small correlated concentration fluctuations; short pulses are
not generated.

Every complete 400-trajectory Phase B block contains 40 clean trajectories and
60 trajectories for each gas. Within each gas, path counts are exactly 27
approach-descend, 21 approach, 6 stationary, and 6 crosswind. Rejected GADEN
runs retry the same scheduled target, so plume failures cannot silently skew
the accepted class distribution. Up to 50 independently seeded plume/layout
attempts are allowed because rare background-dominant cases can exhaust a
smaller retry budget; the target observability thresholds are not relaxed.
The stationary path hovers 0.01-0.05 m above
the source. Crosswind paths sweep laterally across the cup, return to the plume
center, and dwell. Fixed points farther downwind can miss GADEN's narrow
stochastic filament plume entirely.

Phase B samples three spatially uniform airflow regimes per episode: 40%
stagnant air with slowly correlated three-dimensional drift, 45% random weak
indoor flow at 0.03-0.12 m/s, and 15% vent-like flow at 0.15-0.30 m/s. The
stagnant profile changes once per second; the other two are fixed within an
episode. These are uniform-wind approximations, not CFD recirculation. The full
wind sequence and summary are stored in episode metadata. To force one regime,
pass `--wind-regime stagnant_correlated`, `--wind-regime
uniform_weak_indoor`, or `--wind-regime vent_like`.

Visualize one live filament plume in RViz with:

```bash
python -m odor_sim.odor_encoder.wind_rviz --regime stagnant_correlated
python -m odor_sim.odor_encoder.wind_rviz --regime uniform_weak_indoor
python -m odor_sim.odor_encoder.wind_rviz --regime vent_like
```

Run one command at a time and close RViz before starting the next regime. The
launcher starts the live GADEN server, room CAD marker publisher, and RViz, and
stops the background processes when RViz closes. No offline preprocessing or
GADEN desktop GUI is required.

## Staged training

```bash
python -m odor_sim.odor_encoder.train_ab \
  --static datasets/odor_encoder/static_single_v3 \
  --virtual datasets/odor_encoder/virtual_single_v6 \
  --out-dir checkpoints/odor_encoder_ab_v3 \
  --device cuda
```

The A/B trainer uses A: `100% static` and B: `20% static + 80% virtual`.
It keeps one model, saves `phase_a.pt` and `phase_b.pt`, and evaluates both
static and virtual data at 3/5/7/10/15/20-second prefixes. One learned
`[SCENT]` token pools the temporal sequence.

Training writes `checkpoints/odor_encoder_ab_v3/history.json` after every
epoch. Each epoch contains training and per-domain validation values for the
total loss, all five component losses, primary-gas accuracy, exact presence,
clean-air and recovery balanced accuracy, primary-gas probability MAE, and intensity MAE. The validation sample is
fixed across epochs, so its trend is comparable. `metrics.json` contains the
more expensive 3/5/7/10/15/20-second validation results saved after each
phase. Training never reads the test split; run it only after selecting a
checkpoint from validation.

Phase B writes `phase_b_best_loss.pt` and `phase_b_best_accuracy.pt` as well
as the final `phase_b.pt`. “Best accuracy” now means macro gas accuracy, giving
each gas equal influence. Compare both checkpoints on validation, select one,
and run the test split only once after selection:

```bash
python -m odor_sim.odor_encoder.train_ab \
  --evaluate validation \
  --checkpoint checkpoints/odor_encoder_ab_v3/phase_b_best_accuracy.pt \
  --eval-output checkpoints/odor_encoder_ab_v3/validation_best_accuracy.json \
  --device cuda

python -m odor_sim.odor_encoder.train_ab \
  --evaluate validation \
  --checkpoint checkpoints/odor_encoder_ab_v3/phase_b_best_loss.pt \
  --eval-output checkpoints/odor_encoder_ab_v3/validation_best_loss.json \
  --device cuda

python -m odor_sim.odor_encoder.train_ab \
  --evaluate test \
  --checkpoint checkpoints/odor_encoder_ab_v3/phase_b_best_loss.pt \
  --eval-output checkpoints/odor_encoder_ab_v3/test_selected.json \
  --device cuda
```

These commands write the explicitly named validation and test JSON files with
confusion matrices, per-gas accuracy, clean-air precision/recall/balanced
accuracy, recovery metrics, and gas accuracy grouped by exposure state, path,
and total-ppm range. All prefix durations reuse the same episodes and endpoint;
only the amount of causal history changes.

## Modified ScentFormer architecture

```text
voltage window [B, T, 8]
  -> clean-air voltage normalization
  -> linear projection 8 -> 256 + learned position embeddings
  -> append learned [SCENT] token
  -> 4 Transformer encoder layers, 8 heads
  -> normalized [SCENT] output [B, 256]
  -> presence head [B, 6]
  -> primary-gas distribution head [B, 6]
  -> log-total-ppm head [B, 1]
  -> clean-air head [B, 1]
  -> post-exposure recovery head [B, 1]
```

Shorter histories are left-padded to 400 samples and use an attention padding
mask. The appended `[SCENT]` token is always valid and pools all non-padding
sensor timesteps into the representation that will later be projected into
OVLA scent tokens.

Full initial targets are 20,000-50,000 static sessions, 5,000-10,000 virtual
trajectories. Generate the pilot first and grow
the dataset only while held-out metrics continue improving.

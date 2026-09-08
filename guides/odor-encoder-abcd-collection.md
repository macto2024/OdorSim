# Physical Two-Cup Odor Encoder Pipeline

Run all commands from the repository root after:

```bash
source setup/activate.sh
```

The new schema-v4 pipeline uses the `ClassifyLiquidPlaceOVLA` cup-slot ranges,
two continuously active GADEN sources, one uninterrupted eight-channel MOX
state, and causal event-relative endpoints. Gas labels come from endpoint-local
GADEN composition. Filter motion queries GADEN normally; it never forces ppm to
zero.

The main defaults are:

```text
sample rate                  20 Hz
concentration exponent       0.5
emission scale               0.06
filaments per second         15
wind                         zero
original terminal velocity   disabled
experimental buoyancy        disabled
sensor height above source   0.01-0.05 m
meaningful presence          local total ppm >= 5
```

The source calibration file is
`odor_sim/config/phase_b_local_ppm_exp0p5_scale0p06_fps15.json`. A physics
mismatch fails collection unless `--allow-calibration-mismatch` is explicitly
used for an ablation.

## Isolated Two-Cup Live Visualization

This visualization can run while dataset collection is active because it uses
its own ROS domain and never invokes global GADEN cleanup. It shows the OVLA
table and two cup locations, live filaments, correlated wind arrows, a virtual
e-nose marker, a rolling per-gas ppm plot, and all eight rolling MOX voltages.
Its experimental buoyancy and nonzero wind are visualization settings only.
The GADEN server starts with zero filaments and waits three seconds before
lockstep emission begins. Mean wind is perpendicular to the cup-to-cup axis,
and the e-nose is the red point above the selected cup.

```bash
python -m odor_sim.odor_encoder.two_source_rviz \
  --gas-a ethanol \
  --gas-b acetone \
  --local-ppm 50 \
  --sensor-cup a \
  --buoyancy-tau-s 1 \
  --ros-domain-id 78
```

`--local-ppm` applies the same desired-local calibration condition to both
sources. Use `--local-ppm-a` and `--local-ppm-b` to override them separately.
The default correlated wind is scaled by 1.5 for visibility but remains weak
(typically around 0.01-0.03 m/s). `--sensor-cup` may be `a`, `b`, or
`midpoint`. Close RViz to stop its GADEN server and both live plots.

## Pilot Collection

Use complete multiples of 30 where practical. Each 30-episode block covers all
ordered unequal gas pairs once. The per-pair split schedule gives the
300-episode pilot an exact 70/15/15 aggregate split and keeps every ordered
pair represented in every split.

```bash
python -m odor_sim.odor_encoder.physical_pipeline A \
  --episodes 300 \
  --out-dir datasets/odor_encoder/physical_phase_a_pilot_v2

python -m odor_sim.odor_encoder.physical_pipeline B \
  --episodes 510 \
  --out-dir datasets/odor_encoder/physical_phase_b_pilot_v1

python -m odor_sim.odor_encoder.physical_pipeline C \
  --episodes 300 \
  --out-dir datasets/odor_encoder/physical_phase_c_pilot_v1

python -m odor_sim.odor_encoder.physical_pipeline D \
  --episodes 510 \
  --out-dir datasets/odor_encoder/physical_phase_d_pilot_v1
```

Rerunning the same command resumes collection by skipping existing episode IDs.
Pass `--overwrite` only when intentionally replacing records.

The physical collector supports one concurrent process per phase. Defaults use
isolated ROS domains A=81, B=82, C=83 and D=84, plus private GADEN executable
aliases that cannot be killed by legacy global cleanup. Start each phase in a
separate terminal. Do not start two workers for the same phase/output directory.
An A process started before this isolation change may continue on its original
domain; the new B/C/D workers remain isolated from it.

Phase A uses stationary near-cup, between-cup, offset, and jitter positions.
Phase B approaches either cup, descends, dwells, and retreats. Phase C includes
physical onset, gas-to-filter recovery, filter-to-gas, and same-gas return
paths. Phase D visits A then B, mixing direct transfers with physical filter
holds. Neither C nor D resets MOX state.

## Validate Collection

Run this before training. It exits nonzero on split leakage, missing D pairs,
missing early/recovery/second-gas coverage, invalid local labels, forced-clean
filter behavior, or systematic cup/source shortcuts.

```bash
python -m odor_sim.odor_encoder.validate_collection \
  --dataset A datasets/odor_encoder/physical_phase_a_pilot_v2 \
  --dataset B datasets/odor_encoder/physical_phase_b_pilot_v1 \
  --dataset C datasets/odor_encoder/physical_phase_c_pilot_v1 \
  --dataset D datasets/odor_encoder/physical_phase_d_pilot_v1 \
  --output analysis/odor_encoder/physical_pilot_validation.json
```

Do not scale collection until this report is valid and its ppm, dominance, mask,
early-sniff, and recovery distributions look reasonable.

## Train Baseline

The baseline uses a continuous rolling 20-second history. Later stages replay
earlier data; the final batch mix is A/B/C/D = 20/25/20/35.

Training defines *meaningful presence* as local total concentration >= 5 ppm.
This is configurable with `--presence-threshold-ppm`; it does not alter or
clamp the stored physical ppm. Gas loss is also masked below this threshold so
presence and gas-class targets cannot conflict. The training sampler reserves
20% of draws for presence-negative endpoints and 35% for class-balanced,
trusted gas endpoints; the remaining 45% retain event-bucket sampling. Phase A
has no recovery-positive state, so its recovery loss is explicitly masked.
Evaluation remains an unbalanced census of stored endpoints rather than using
the training sampler.

```bash
python -m odor_sim.odor_encoder.train_abcd \
  --model baseline \
  --phase-a datasets/odor_encoder/physical_phase_a_pilot_v2 \
  --phase-b datasets/odor_encoder/physical_phase_b_pilot_v1 \
  --phase-c datasets/odor_encoder/physical_phase_c_pilot_v1 \
  --phase-d datasets/odor_encoder/physical_phase_d_pilot_v1 \
  --out-dir checkpoints/physical_scentformer_baseline_v3 \
  --device cuda
```

## Train Dual-Timescale

The dual model uses the latest 10 seconds at 20 Hz and a low-pass block-averaged
40-second history at 5 Hz.

```bash
python -m odor_sim.odor_encoder.train_abcd \
  --model dual_timescale \
  --phase-a datasets/odor_encoder/physical_phase_a_pilot_v2 \
  --phase-b datasets/odor_encoder/physical_phase_b_pilot_v1 \
  --phase-c datasets/odor_encoder/physical_phase_c_pilot_v1 \
  --phase-d datasets/odor_encoder/physical_phase_d_pilot_v1 \
  --out-dir checkpoints/physical_scentformer_dual_v3 \
  --batch-size 8 \
  --device cuda
```

The model exports its latent embedding and exactly four predictions: six-class
gas logits, gas presence, local `log1p(ppm)` intensity, and recovery/carryover.
There is no clean-air softmax class and no separate clean head.

## Validation And Test

Choose the checkpoint using validation only. Per-epoch checkpoint selection uses
a 5-second causal history by default (`--selection-prefix-s 5`). `best_score`
weights gas macro accuracy 50%, second-sniff gas accuracy 20%,
presence balanced accuracy 15%, and recovery balanced accuracy 15%. A term is
excluded and the remaining weights are renormalized when the validation subset
does not contain both binary classes (for example Phase-A recovery).

```bash
python -m odor_sim.odor_encoder.train_abcd \
  --evaluate validation \
  --checkpoint checkpoints/physical_scentformer_baseline_v3/phase_d_best_score.pt \
  --eval-output checkpoints/physical_scentformer_baseline_v3/validation_best_score.json \
  --phase-a datasets/odor_encoder/physical_phase_a_pilot_v2 \
  --phase-b datasets/odor_encoder/physical_phase_b_pilot_v1 \
  --phase-c datasets/odor_encoder/physical_phase_c_pilot_v1 \
  --phase-d datasets/odor_encoder/physical_phase_d_pilot_v1 \
  --device cuda

python -m odor_sim.odor_encoder.train_abcd \
  --evaluate test \
  --checkpoint checkpoints/physical_scentformer_baseline_v3/phase_d_best_score.pt \
  --eval-output checkpoints/physical_scentformer_baseline_v3/test_selected.json \
  --phase-a datasets/odor_encoder/physical_phase_a_pilot_v2 \
  --phase-b datasets/odor_encoder/physical_phase_b_pilot_v1 \
  --phase-c datasets/odor_encoder/physical_phase_c_pilot_v1 \
  --phase-d datasets/odor_encoder/physical_phase_d_pilot_v1 \
  --device cuda
```

Use the same commands with the dual checkpoint/output paths to evaluate the
dual-timescale model. Evaluation reports gas accuracy by time, ppm, dominance,
phase, exposure state, and first/second sniff; ordered-pair second-gas results;
presence/recovery metrics; intensity error; NLL/margin/ECE; and all gas-mask
coverage reasons.

Compare the `1s`, `3s`, `5s`, `10s`, and `20s` Phase-D
`gas.first_vs_second.second` entries in validation output before choosing the
deployment history. The current model accepts each duration from one checkpoint;
longer history is not automatically preferable.

Phase-D evaluation also reports
`gas.second_gas_accuracy_by_offset` for 0.5/1/2/3/5/10 seconds. Every entry
contains eligible, supervised and masked sample counts, aggregate accuracy, and
a 95% Wilson confidence interval. Ordered-pair cells include their own Wilson
intervals.

## Full Collection And Coverage Gate

After the pilot, collect complete multiples of 30. A practical first full run is
A=3000, B=5100, C=3000, D=9000. D=9000 provides 300 trajectories per ordered
pair before splitting; masking can still reduce the usable count.

```bash
python -m odor_sim.odor_encoder.physical_pipeline A --episodes 3000 \
  --out-dir datasets/odor_encoder/physical_phase_a_full_v1
python -m odor_sim.odor_encoder.physical_pipeline B --episodes 5100 \
  --out-dir datasets/odor_encoder/physical_phase_b_full_v1
python -m odor_sim.odor_encoder.physical_pipeline C --episodes 3000 \
  --out-dir datasets/odor_encoder/physical_phase_c_full_v1
python -m odor_sim.odor_encoder.physical_pipeline D --episodes 9000 \
  --out-dir datasets/odor_encoder/physical_phase_d_full_v1
```

Enforce at least ten supervised examples for every ordered pair and second-gas
offset in each split:

```bash
python -m odor_sim.odor_encoder.validate_collection \
  --dataset A datasets/odor_encoder/physical_phase_a_full_v1 \
  --dataset B datasets/odor_encoder/physical_phase_b_full_v1 \
  --dataset C datasets/odor_encoder/physical_phase_c_full_v1 \
  --dataset D datasets/odor_encoder/physical_phase_d_full_v1 \
  --meaningful-presence-threshold-ppm 5 \
  --min-second-offset-endpoints-per-pair 10 \
  --output analysis/odor_encoder/physical_full_validation_v1.json
```

If only the D coverage gate fails, add only the deficient ordered-pair/split
groups. The supplement command rescans the live dataset, assigns globally unique
episode indices, and collects in small batches until every requested second-gas
offset has ten physically supervised endpoints:

```bash
python -m odor_sim.odor_encoder.supplement_phase_d \
  --validation-report analysis/odor_encoder/physical_full_validation_v1.json \
  --dataset-root datasets/odor_encoder/physical_phase_d_full_v1 \
  --required 10
```

It is safe to rerun after an interruption. Do not run another Phase-D collector
against the same output directory concurrently, since both processes could
choose the same next episode index. A/B/C collection may continue on their
separate ROS domains. Rerun `validate_collection` afterward. Do not reduce the
dominance or physical masking thresholds merely to pass this gate.

## Acceptance Trajectory

Generate and inspect a completely held-out ethanol-to-acetone Phase D trajectory.
The separate seed prevents source-realization overlap with the pilot:

```bash
python -m odor_sim.odor_encoder.physical_pipeline D \
  --episodes 1 --gas-a ethanol --gas-b acetone --split test \
  --seed 99001 --recovery-durations-s 5 --direct-transfer-fraction 0 \
  --out-dir datasets/odor_encoder/physical_acceptance_ethanol_acetone

python -m odor_sim.odor_encoder.inspect_acceptance \
  --phase-d datasets/odor_encoder/physical_acceptance_ethanol_acetone \
  --split test

python -m odor_sim.odor_encoder.inspect_acceptance \
  --phase-d datasets/odor_encoder/physical_acceptance_ethanol_acetone \
  --split test \
  --checkpoint checkpoints/physical_scentformer_baseline_v3/phase_d_best_score.pt \
  --device cuda
```

The tool writes a PDF of physical ppm, all MOX channels, presence/recovery/mask
targets, plus JSON with causal predictions at first-sniff, departure/recovery,
and second-sniff endpoints.

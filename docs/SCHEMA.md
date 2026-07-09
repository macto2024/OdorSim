# OdorSim dataset schema (Phase 5 capture contract)

Status: **draft** — frozen before Phase 5a implementation. Phase 6 adapters
target this contract; extend only with explicit version bumps.

Control frequency: **20 Hz** (default). All per-step arrays are length `T`
(timesteps in the episode).

## Design rules

1. **Mine raw ppm(t), not voltage.** MOX/PID voltage is synthesized offline (5b).
2. **No baked normalization.** Store raw physical units; per-episode stats in meta.
3. **VLA-agnostic superset.** Adapters pick image set, odor channel, action encoding at train time.
4. **Two-stage pipeline.** Raw episodes on disk first; LeRobot v3.0 conversion is a separate step (5c).

## Raw episode layout (5a output)

```
datasets/teleop/episode_<timestamp>/
  episode.npz          # per-step arrays (see below)
  meta.json            # episode metadata
  frames/              # optional PNG sequence (or stacked in npz)
    agentview/
    wrist/
  features.npz         # 5b offline synthesis (optional until 5b runs)
  features_meta.json   # sensor model id/params (5b)
```

## Per-timestep fields (`episode.npz`)

| Key | Shape | Dtype | Units / notes |
|-----|-------|-------|---------------|
| `sim_time` | `(T,)` | float32 | GADEN simulation time (s) |
| `ppm` | `(T, G)` | float32 | Ground-truth per-gas concentration at EE; `G = len(gas_types)` |
| `action` | `(T, A+1)` | float32 | `[robosuite_action..., enose_state]`; `enose_state` ∈ {-1, 0, 1} |
| `enose_state` | `(T,)` | int8 | Redundant copy of last action dim for convenience |
| `sampling_active` | `(T,)` | bool | True during auto-hold sample window |
| `state` | `(T, S)` | float32 | **Legacy flat proprio** (sorted `robot0_*` keys); kept for compatibility |
| `proprio` | `(T, ...)` | — | **5a structured dict** (see Proprio section); stored as separate arrays or nested |

### Images (5a)

| Key | Shape | Dtype | Camera |
|-----|-------|-------|--------|
| `observation.images.agentview` | `(T, 256, 256, 3)` | uint8 | Table overview |
| `observation.images.wrist` | `(T, 256, 256, 3)` | uint8 | `robot0_eye_in_hand` |

RGB channel order: HWC, uint8 0–255. No JPEG compression in npz (converter may encode to MP4 in LeRobot).

### Structured proprio (5a — replaces flat `state` long-term)

Stored as named arrays in `episode.npz` or a `proprio/` group:

| Key | Shape | Dtype | Notes |
|-----|-------|-------|-------|
| `joint_pos` | `(T, J)` | float32 | rad |
| `joint_vel` | `(T, J)` | float32 | rad/s |
| `eef_pos` | `(T, 3)` | float32 | m, world frame |
| `eef_quat` | `(T, 4)` | float32 | xyzw, world frame |
| `gripper_qpos` | `(T, Gg)` | float32 | gripper joint positions |

`J` and `Gg` depend on robot (`robots` in meta).

## Episode metadata (`meta.json`)

| Key | Type | Notes |
|-----|------|-------|
| `instruction` | str | Task language label |
| `recipe` | str | VOC recipe name |
| `robots` | str | robosuite robot name (e.g. `Panda`) |
| `gas_types` | list[str] | Column order for `ppm` |
| `control_freq` | int | Hz (default 20) |
| `sample_hold_steps` | int | Auto-hold duration in steps |
| `success_hold_steps` | int | Sustained-success auto-end threshold |
| `success` | bool | Task success at episode end |
| `num_steps` | int | `T` |
| `action_layout` | str | `"[robosuite_action..., enose_state]"` |
| `controller_type` | str | e.g. `OSC_POSE` delta (5a) |
| `scenario` | str | GADEN scenario name or path (5a) |
| `scene_id` | str | Exported GADEN scene id (5a) |

## Offline-synthesized features (5b — `features.npz`)

Not recorded at mine time. Re-runnable from `ppm` + `enose_state`.

| Key | Shape | Dtype | Notes |
|-----|-------|-------|-------|
| `enose_voltage_continuous` | `(T,)` | float32 | V; always-exposed MOX model |
| `enose_voltage_sampling` | `(T,)` | float32 | V; gated by `enose_state` |
| `sample_windows` | list | — | Completed `SampleWindow` records (trigger_step, duration, traces, `odor_class`) |

`features_meta.json` stores sensor model id (e.g. `TGS2620`), `R0`, `RL`, `Vcc`, `rate`.

## LeRobot v3.0 mapping (5c)

Target: **LeRobotDataset v3.0** via **lerobot >= 0.6** (chunked parquet + MP4).
Implemented in [`odor_sim/recording/convert.py`](../odor_sim/recording/convert.py)
(`convert_episodes`). Cameras are MP4-encoded by default (`use_videos=True`).

| LeRobot feature | dtype | Source |
|-----------------|-------|--------|
| `observation.images.agentview` | video (RGB) | `frames/agentview/*.png` |
| `observation.images.wrist` | video (RGB) | `frames/wrist/*.png` |
| `observation.state` | float32 | structured proprio concatenated (`joint_pos`, `joint_vel`, `eef_pos`, `eef_quat`, `gripper_qpos`); flat `state` fallback |
| `action` | float32 | `action` array (robosuite composite + enose dim) |
| `observation.ppm` | float32 | raw per-gas ppm at the EE |
| `observation.enose_state` | float32 | ternary sampling token (last action dim) |
| `observation.enose_voltage_continuous` | float32 | 5b `features.npz` (only if present) |
| `observation.enose_voltage_sampling` | float32 | 5b `features.npz` (only if present) |
| `task` / language | str | `meta.instruction` |

Odor voltage is kept as its own observation channel (not folded into
`observation.state`) so Phase 6 adapters can ablate it. Default local layout:
`datasets/lerobot/<repo_id>/`.

## Action encoding notes

- **Recorded:** robosuite composite action vector + `enose_state` as last dim.
- **Controller:** default `OSC_POSE` with `input_type=delta`, `input_ref_frame=base`.
- **Re-derivable at train time:** delta-EE, absolute-EE+gripper, joint deltas (from recorded state + controller meta).

## Odor adapter channels (Phase 6 — not stored separately)

Adapters derive from 5b features:

| Mode | Observation odor channel |
|------|--------------------------|
| `no-odor` | omitted |
| `continuous` | last-K stacked `enose_voltage_continuous` |
| `sampling` | completed window trace when `enose_state==1` window closes |

## Version history

| Version | Date | Change |
|---------|------|--------|
| 0.1-draft | 2026-07-08 | Initial schema; cameras agentview+wrist; two-stage pipeline; LeRobot v3.0 |
| 0.1 | 2026-07-08 | Phase 5 implemented (5a mining superset, 5b synthesis, 5c LeRobot v3.0 converter); mapping finalized against lerobot 0.6 |

"""Build an odor-aware TFDS/RLDS dataset from OVLA ClassifyLiquidPlace demos.

Unlike :mod:`odor_sim.recording.build_rlds_odorplace` (vision-only), this
builder:

* sparsely retains sniff HOLD landmarks and the final decision frame while
  dropping redundant 20-Hz hover ticks and filter holds
* attaches causal, left-padded raw ``observation.odor`` prefixes during each sniff
* records episode IDs so later no-odor / shuffled-odor views can swap traces

OpenVLA-OFT treats every remaining RLDS step as a training sample, so collapsing
most holds avoids a stationary-hover prior while retained landmarks teach the
policy to wait for olfactory evidence.

The RLDS odor tensor is input to the frozen ScentFormer. The deployment adapter
must call ``model.encode(...)`` and pass only that latent embedding to OVLA;
gas/presence/intensity/recovery heads remain training and evaluation outputs.

Example::

    python -m odor_sim.recording.build_rlds_odor_ovla \\
        --input /root/Odorbot/datasets/odor_ovla_oft/raw \\
        --manifest /root/Odorbot/datasets/odor_ovla_oft/manifest.csv \\
        --output /root/Odorbot/datasets/rlds \\
        --dataset-name odorsim_ovla_odorplace_sniff10s
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np

from odor_sim.recording.build_rlds_odorplace import (
    _build_proprio_state,
    _load_frames,
    discover_episode_dirs,
    is_noop,
)
from odor_sim.recording.odor_config import DEFAULT_ODOR_CONFIG
from odor_sim.recording.odor_events import (
    DEFAULT_SNIFF_LANDMARKS_S,
    PHASE_SNIFF_HOLD,
    collapse_keep_mask,
    load_odor_events,
    sniff_landmark_steps,
)

DEFAULT_DATASET_NAME = "odorsim_ovla_odorplace_sniff10s"
DEFAULT_OUTPUT = "/root/Odorbot/datasets/rlds"
DEFAULT_ODOR_HISTORY_S = 5.0


def _ovla_stratum(ep_dir: Path) -> tuple[str, str, bool]:
    """Return target, distractor, and actual target-first order for splitting."""
    meta = json.loads((ep_dir / "meta.json").read_text(encoding="utf-8"))
    target = str(meta["instructed_liquid"])
    liquids = [str(value) for value in meta.get("liquids", [])]
    distractors = [value for value in liquids if value != target]
    if len(distractors) != 1:
        raise ValueError(f"{ep_dir}: expected exactly one target and one distractor")
    event_doc = json.loads((ep_dir / "odor_events.json").read_text(encoding="utf-8"))
    events = list(event_doc.get("events") or [])
    if not events:
        raise ValueError(f"{ep_dir}: no odor events")
    target_first = str(events[0]["object_id"]) == target
    return target, distractors[0], target_first


def _split_episodes_stratified(
    eps: list[Path], val_frac: float, seed: int,
) -> dict[str, list[Path]]:
    """Split by ordered gas pair and sniff order with exact global val size."""
    if not 0.0 <= float(val_frac) < 1.0:
        raise ValueError("val_frac must be in [0, 1)")
    if len(eps) <= 1 or val_frac == 0:
        return {"train": list(eps), "val": []}

    rng = np.random.default_rng(seed)
    groups: dict[tuple[str, str, bool], list[Path]] = defaultdict(list)
    for ep_dir in eps:
        groups[_ovla_stratum(ep_dir)].append(ep_dir)
    keys = sorted(groups)
    rng.shuffle(keys)
    for key in keys:
        rng.shuffle(groups[key])

    target_total = max(1, int(round(len(eps) * float(val_frac))))
    pair_keys = sorted({key[:2] for key in keys})
    rng.shuffle(pair_keys)
    pair_sizes = {
        pair: sum(len(groups.get((*pair, order), [])) for order in (False, True))
        for pair in pair_keys
    }
    pair_allocations = {
        pair: min(int(np.floor(size * float(val_frac))), size - 1)
        for pair, size in pair_sizes.items()
    }
    remaining = target_total - sum(pair_allocations.values())
    ranked_pairs = sorted(
        pair_keys,
        key=lambda pair: (
            pair_sizes[pair] * float(val_frac) - pair_allocations[pair]
        ),
        reverse=True,
    )
    while remaining > 0:
        progressed = False
        for pair in ranked_pairs:
            if pair_allocations[pair] < pair_sizes[pair] - 1:
                pair_allocations[pair] += 1
                remaining -= 1
                progressed = True
                if remaining == 0:
                    break
        if not progressed:
            raise ValueError("cannot allocate requested validation fraction")

    allocations = {key: 0 for key in keys}
    order_totals = {False: 0, True: 0}
    for pair in pair_keys:
        pair_val = pair_allocations[pair]
        pair_size = pair_sizes[pair]
        order_keys = [(*pair, order) for order in (False, True) if (*pair, order) in groups]
        for key in order_keys:
            proportional = pair_val * len(groups[key]) / pair_size
            allocations[key] = min(int(np.floor(proportional)), len(groups[key]) - 1)
        pair_remaining = pair_val - sum(allocations[key] for key in order_keys)
        while pair_remaining > 0:
            eligible = [
                key for key in order_keys if allocations[key] < len(groups[key]) - 1
            ]
            if not eligible:
                raise ValueError(f"cannot allocate validation examples for pair {pair}")
            # Prefer the largest proportional remainder; equal remainders go to
            # the currently less represented sniff order globally.
            key = max(
                eligible,
                key=lambda candidate: (
                    pair_val * len(groups[candidate]) / pair_size
                    - allocations[candidate],
                    -order_totals[candidate[2]],
                ),
            )
            allocations[key] += 1
            pair_remaining -= 1
        for key in order_keys:
            order_totals[key[2]] += allocations[key]

    train: list[Path] = []
    val: list[Path] = []
    for key in keys:
        n_val = allocations[key]
        val.extend(groups[key][:n_val])
        train.extend(groups[key][n_val:])
    rng.shuffle(train)
    rng.shuffle(val)
    return {"train": train, "val": val}


def validate_deployment_protocol(meta: dict, ep_dir: Path) -> None:
    """Reject raw episodes generated under the obsolete odor protocol."""
    expected = {
        "schema_version": "ovla-odor-0.3",
        "odor_mode": "continuous",
        "transition_mode": "direct",
        "wind_regime": "zero",
        "source_emission_model": "separated_power_law",
        "source_condition_sampling": "paired_symmetric_log_uniform",
        "original_terminal_velocity_enabled": False,
        "ovla_odor_interface": "scentformer_embedding_only",
        "mox_reset_between_sniffs": False,
    }
    mismatches = [
        f"{key}={meta.get(key)!r} (expected {value!r})"
        for key, value in expected.items() if meta.get(key) != value
    ]
    numeric = {
        "emission_scale": 0.06,
        "concentration_exponent": 0.5,
        "filaments_per_second": 15.0,
        "buoyancy_tau_s": 0.0,
        "odor_sampling_rate": 20.0,
        "decision_window_s": 1.0,
        "decision_min_total_ppm": 5.0,
        "decision_dominance_threshold": 0.70,
        "decision_min_pass_fraction": 0.80,
    }
    for key, value in numeric.items():
        try:
            matches = np.isclose(float(meta.get(key)), value, atol=1e-8)
        except (TypeError, ValueError):
            matches = False
        if not matches:
            mismatches.append(f"{key}={meta.get(key)!r} (expected {value})")
    if list(meta.get("mox_models") or []) != list(DEFAULT_ODOR_CONFIG.mox_models):
        mismatches.append("mox_models do not match the fixed encoder channel order")
    if len(meta.get("liquids") or []) != 2:
        mismatches.append("episode does not contain exactly two gas cups")
    event_path = ep_dir / "odor_events.json"
    if not event_path.is_file():
        mismatches.append("odor_events.json is missing")
    else:
        events = json.loads(event_path.read_text(encoding="utf-8")).get("events", [])
        if not events:
            mismatches.append("episode has no sniff decision events")
        elif any(event.get("evidence_accepted") is not True for event in events):
            mismatches.append("one or more sniff decisions lack accepted physical evidence")
    if mismatches:
        raise ValueError(
            f"{ep_dir}: incompatible OVLA odor protocol: " + "; ".join(mismatches)
        )


def attach_continuous_odor(
    keep_indices: np.ndarray,
    *,
    voltage: np.ndarray,
    history_steps: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Attach a causal rolling history from the continuous episode stream."""
    voltage = np.asarray(voltage, dtype=np.float32)
    if voltage.ndim != 2:
        raise ValueError(f"continuous voltage must be (T,C), got {voltage.shape}")
    history_steps = int(history_steps)
    if history_steps <= 0:
        raise ValueError("history_steps must be positive")
    odor = np.zeros((len(keep_indices), history_steps, voltage.shape[1]), np.float32)
    valid = np.zeros(len(keep_indices), np.bool_)
    padding = np.ones((len(keep_indices), history_steps), np.bool_)
    for row, endpoint_value in enumerate(keep_indices):
        endpoint = min(int(endpoint_value), len(voltage) - 1)
        if endpoint < 0:
            continue
        start = max(0, endpoint - history_steps + 1)
        window = voltage[start:endpoint + 1]
        odor[row, -len(window):] = window
        padding[row, -len(window):] = False
        valid[row] = True
    return odor, valid, padding


def _latest_event_index(end_steps: np.ndarray, step: int) -> int:
    """Index of the most recently completed sniff at ``step``, or -1."""
    done = np.where(end_steps <= int(step))[0]
    if done.size == 0:
        return -1
    return int(done[-1])


def attach_odor(
    keep_indices: np.ndarray,
    *,
    voltage: np.ndarray,
    start_steps: "np.ndarray | None" = None,
    end_steps: np.ndarray,
    t_sniff: int,
    num_channels: int,
    return_padding_mask: bool = False,
) -> "tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray]":
    """Attach the active causal sniff prefix, else the latest completed sniff.

    Active prefixes are left-padded to ``t_sniff``. Consequently a retained
    HOLD frame can never observe sensor samples from later in its sniff event.
    """
    n = int(keep_indices.shape[0])
    odor = np.zeros((n, t_sniff, num_channels), dtype=np.float32)
    valid = np.zeros((n,), dtype=np.bool_)
    padding_mask = np.ones((n, t_sniff), dtype=np.bool_)
    if voltage.size == 0 or end_steps.size == 0:
        return (odor, valid, padding_mask) if return_padding_mask else (odor, valid)
    causal_events = start_steps is not None
    starts = (
        np.asarray(start_steps, dtype=np.int64)
        if causal_events
        else np.asarray(end_steps, dtype=np.int64)
    )
    if starts.shape != end_steps.shape:
        raise ValueError("start_steps and end_steps must have the same shape")
    for i, step_value in enumerate(keep_indices):
        step = int(step_value)
        active = (
            np.where((starts <= step) & (step <= end_steps))[0]
            if causal_events else np.empty(0, dtype=np.int64)
        )
        ev = int(active[-1]) if active.size else _latest_event_index(end_steps, step)
        if ev < 0:
            continue
        if causal_events and step <= int(end_steps[ev]):
            available = min(t_sniff, step - int(starts[ev]) + 1)
            if available <= 0:
                continue
            odor[i, -available:] = voltage[ev, :available]
            padding_mask[i, -available:] = False
        else:
            odor[i] = voltage[ev]
            padding_mask[i] = False
        valid[i] = True
    return (odor, valid, padding_mask) if return_padding_mask else (odor, valid)


def collapse_episode_steps(
    *,
    phase: np.ndarray,
    action7: np.ndarray,
    sniff_end_steps: list[int],
    sniff_landmarks: "list[int] | tuple[int, ...]" = (),
) -> tuple[np.ndarray, int, int]:
    """Return original indices to keep after phase collapse + no-op filter.

    Returns ``(keep_indices, n_hold_dropped, n_noop)``.
    """
    keep_mask = collapse_keep_mask(phase, sniff_end_steps, sniff_landmarks)
    n_hold = int((~keep_mask).sum())
    idxs = np.flatnonzero(keep_mask)
    protected_holds = set(int(index) for index in sniff_landmarks)
    protected_holds.update(int(index) for index in sniff_end_steps)
    kept = []
    prev = None
    n_noop = 0
    for i in idxs:
        a = action7[int(i)]
        if int(i) in protected_holds and int(phase[int(i)]) == PHASE_SNIFF_HOLD:
            kept.append(int(i))
            prev = a
            continue
        if is_noop(a, prev):
            n_noop += 1
            prev = a
            continue
        kept.append(int(i))
        prev = a
    if not kept:
        raise ValueError("all steps filtered (holds + no-ops)")
    return np.asarray(kept, dtype=np.int64), n_hold, n_noop


def parse_episode(
    ep_dir: Path,
    *,
    load_images: bool = True,
    sniff_landmarks_s: "tuple[float, ...] | list[float]" = DEFAULT_SNIFF_LANDMARKS_S,
    sniff_landmark_jitter_s: float = 0.1,
    landmark_seed: int = 0,
    odor_history_s: float = DEFAULT_ODOR_HISTORY_S,
) -> dict:
    """Load one OVLA raw episode, collapse holds, attach odor traces."""
    meta = json.loads((ep_dir / "meta.json").read_text(encoding="utf-8"))
    validate_deployment_protocol(meta, ep_dir)
    data = np.load(ep_dir / "episode.npz", allow_pickle=False)
    action = np.asarray(data["action"], dtype=np.float32)
    if action.shape[1] < 7:
        raise ValueError(f"{ep_dir}: action dim {action.shape[1]} < 7")
    action7 = action[:, :7].copy()

    if "phase" not in data.files:
        raise ValueError(f"{ep_dir}: missing phase array (not an OVLA odor episode)")
    phase = np.asarray(data["phase"], dtype=np.int8)

    odor_pack = load_odor_events(ep_dir)
    events = list(odor_pack["meta"].get("events") or [])
    voltage = np.asarray(odor_pack["voltage"], dtype=np.float32)
    start_steps = np.asarray([int(ev["start_step"]) for ev in events], dtype=np.int64)
    end_steps = np.asarray([int(ev["end_step"]) for ev in events], dtype=np.int64)
    t_sniff = int(round(float(odor_history_s) * float(
        meta.get("odor_sampling_rate") or DEFAULT_ODOR_CONFIG.odor_sampling_rate
    )))
    num_channels = int(
        voltage.shape[2]
        if voltage.ndim == 3
        else len(meta.get("mox_models") or DEFAULT_ODOR_CONFIG.mox_models)
    )
    if t_sniff <= 0:
        raise ValueError("odor history must contain at least one sample")

    sampling_rate = float(
        meta.get("odor_sampling_rate") or DEFAULT_ODOR_CONFIG.odor_sampling_rate
    )
    episode_id = str(meta.get("episode_id") or ep_dir.name)
    stable_id = int.from_bytes(
        hashlib.sha256(episode_id.encode("utf-8")).digest()[:8], "little"
    )
    landmark_indices = sniff_landmark_steps(
        events,
        rate_hz=sampling_rate,
        landmarks_s=sniff_landmarks_s,
        jitter_s=sniff_landmark_jitter_s,
        seed=int(landmark_seed) + stable_id,
    )
    keep_idx, n_hold, n_noop = collapse_episode_steps(
        phase=phase,
        action7=action7,
        sniff_end_steps=[int(x) for x in end_steps],
        sniff_landmarks=landmark_indices,
    )

    state_all = _build_proprio_state({k: data[k] for k in data.files})
    t = min(action7.shape[0], state_all.shape[0], phase.shape[0])
    keep_idx = keep_idx[keep_idx < t]
    action_kept = action7[keep_idx]
    state_kept = state_all[keep_idx]

    if load_images:
        images = _load_frames(ep_dir, "agentview")
        wrist = _load_frames(ep_dir, "wrist")
        t_img = min(t, images.shape[0], wrist.shape[0])
        keep_idx = keep_idx[keep_idx < t_img]
        action_kept = action7[keep_idx]
        state_kept = state_all[keep_idx]
        images = images[keep_idx]
        wrist = wrist[keep_idx]
    else:
        images = None
        wrist = None

    if "enose_voltages" not in data.files:
        raise ValueError(f"{ep_dir}: missing continuous enose_voltages")
    continuous_voltage = np.asarray(data["enose_voltages"], dtype=np.float32)
    if continuous_voltage.ndim == 1:
        continuous_voltage = continuous_voltage[:, None]
    if continuous_voltage.shape[1] != num_channels:
        raise ValueError(
            f"{ep_dir}: continuous MOX channels {continuous_voltage.shape[1]} "
            f"!= event channels {num_channels}"
        )
    odor, odor_valid, odor_padding_mask = attach_continuous_odor(
        keep_idx,
        voltage=continuous_voltage,
        history_steps=t_sniff,
    )

    instruction = str(meta.get("instruction", "")).strip()
    if not instruction:
        raise ValueError(f"{ep_dir}: empty instruction")

    return {
        "images": images,
        "wrist": wrist,
        "state": state_kept,
        "action": action_kept,
        "odor": odor,
        "odor_valid": odor_valid,
        "odor_padding_mask": odor_padding_mask,
        "instruction": instruction,
        "file_path": str(ep_dir.resolve()),
        "n_hold": n_hold,
        "n_noop": n_noop,
        "n_sniff_landmarks": len(landmark_indices),
        "t_sniff": t_sniff,
        "num_channels": num_channels,
        "meta": meta,
        "episode_id": episode_id,
        "scene_id": str(meta.get("scene_id") or ""),
        "seed": meta.get("seed"),
        "sniff_duration_s": float(meta.get("sniff_duration_s") or 0.0),
        "mox_models": [str(x) for x in (meta.get("mox_models") or [])],
        "liquids": [str(x) for x in (meta.get("liquids") or meta.get("objects") or [])],
        "instructed_liquid": str(meta.get("instructed_liquid") or ""),
    }


def _infer_odor_shape(
    episode_dirs: list[Path], odor_history_s: float,
) -> tuple[int, int]:
    first = parse_episode(
        episode_dirs[0], load_images=False, odor_history_s=odor_history_s,
    )
    return int(first["t_sniff"]), int(first["num_channels"])


def build_tfds_dataset(
    episode_dirs: list[Path],
    *,
    data_dir: Path,
    dataset_name: str,
    val_frac: float = 0.05,
    seed: int = 0,
    image_size: int = 256,
    sniff_landmarks_s: "tuple[float, ...] | list[float]" = DEFAULT_SNIFF_LANDMARKS_S,
    sniff_landmark_jitter_s: float = 0.1,
    odor_history_s: float = DEFAULT_ODOR_HISTORY_S,
) -> Path:
    """Create a TFDS builder and run ``download_and_prepare``."""
    import tensorflow as tf
    import tensorflow_datasets as tfds

    t_sniff, num_channels = _infer_odor_shape(episode_dirs, odor_history_s)
    splits = _split_episodes_stratified(episode_dirs, val_frac, seed)
    print(
        f"[build_rlds_ovla] episodes={len(episode_dirs)} "
        f"train={len(splits['train'])} val={len(splits['val'])} "
        f"odor=({t_sniff}, {num_channels})"
    )

    def _get(path: Path) -> dict:
        # TFDS consumes every episode once. Retaining all decoded image arrays
        # would exhaust memory during a full 1,500-episode conversion.
        ep = parse_episode(
            path,
            load_images=True,
            sniff_landmarks_s=sniff_landmarks_s,
            sniff_landmark_jitter_s=sniff_landmark_jitter_s,
            landmark_seed=seed,
            odor_history_s=odor_history_s,
        )
        if int(ep["t_sniff"]) != t_sniff or int(ep["num_channels"]) != num_channels:
            raise ValueError(
                f"{path}: odor shape {(ep['t_sniff'], ep['num_channels'])} "
                f"!= dataset {(t_sniff, num_channels)}; collect one sniff duration per RLDS build"
            )
        print(
            f"  parsed {path.name}: T={ep['action'].shape[0]} "
            f"(kept {ep['n_sniff_landmarks']} sniff landmarks; "
            f"dropped {ep['n_hold']} holds, {ep['n_noop']} no-ops)"
        )
        return ep

    split_paths = {k: [str(p.resolve()) for p in v] for k, v in splits.items() if v}

    class _Builder(tfds.core.GeneratorBasedBuilder):
        VERSION = tfds.core.Version("1.1.0")
        RELEASE_NOTES = {
            "1.0.0": "OVLA odor pick-place demos with collapsed sniff holds.",
            "1.1.0": "Sparse HOLD supervision with causal active-sniff odor prefixes.",
        }

        def _info(self) -> tfds.core.DatasetInfo:
            return tfds.core.DatasetInfo(
                builder=self,
                description="OdorSim ClassifyLiquidPlaceOVLA demos (scripted OSC + odor).",
                features=tfds.features.FeaturesDict(
                    {
                        "steps": tfds.features.Dataset(
                            {
                                "observation": tfds.features.FeaturesDict(
                                    {
                                        "image": tfds.features.Image(
                                            shape=(image_size, image_size, 3),
                                            dtype=np.uint8,
                                            encoding_format="jpeg",
                                        ),
                                        "wrist_image": tfds.features.Image(
                                            shape=(image_size, image_size, 3),
                                            dtype=np.uint8,
                                            encoding_format="jpeg",
                                        ),
                                        "state": tfds.features.Tensor(
                                            shape=(8,),
                                            dtype=np.float32,
                                        ),
                                        "odor": tfds.features.Tensor(
                                            shape=(t_sniff, num_channels),
                                            dtype=np.float32,
                                        ),
                                        "odor_valid": tfds.features.Scalar(dtype=np.bool_),
                                        "odor_padding_mask": tfds.features.Tensor(
                                            shape=(t_sniff,), dtype=np.bool_,
                                        ),
                                    }
                                ),
                                "action": tfds.features.Tensor(shape=(7,), dtype=np.float32),
                                "discount": tfds.features.Scalar(dtype=np.float32),
                                "reward": tfds.features.Scalar(dtype=np.float32),
                                "is_first": tfds.features.Scalar(dtype=np.bool_),
                                "is_last": tfds.features.Scalar(dtype=np.bool_),
                                "is_terminal": tfds.features.Scalar(dtype=np.bool_),
                                "language_instruction": tfds.features.Text(),
                            }
                        ),
                        "episode_metadata": tfds.features.FeaturesDict(
                            {
                                "file_path": tfds.features.Text(),
                                "episode_id": tfds.features.Text(),
                                "scene_id": tfds.features.Text(),
                                "instructed_liquid": tfds.features.Text(),
                                "sniff_duration_s": tfds.features.Scalar(dtype=np.float32),
                                "odor_interface": tfds.features.Text(),
                            }
                        ),
                    }
                ),
                supervised_keys=None,
            )

        def _split_generators(self, dl_manager):
            return {
                split_name: self._generate_examples(paths=paths)
                for split_name, paths in split_paths.items()
            }

        def _generate_examples(self, paths: list):
            for path_str in paths:
                ep_dir = Path(path_str)
                try:
                    ep = _get(ep_dir)
                except Exception as exc:  # noqa: BLE001
                    print(f"[build_rlds_ovla] skip {ep_dir}: {exc}")
                    continue
                imgs = ep["images"]
                wrist = ep["wrist"]
                state = ep["state"]
                acts = ep["action"]
                odor = ep["odor"]
                odor_valid = ep["odor_valid"]
                odor_padding_mask = ep["odor_padding_mask"]
                if imgs.shape[1] != image_size or imgs.shape[2] != image_size:
                    imgs = np.stack(
                        [
                            np.asarray(
                                tf.image.resize(im, (image_size, image_size)).numpy(),
                                dtype=np.uint8,
                            )
                            for im in imgs
                        ]
                    )
                    wrist = np.stack(
                        [
                            np.asarray(
                                tf.image.resize(im, (image_size, image_size)).numpy(),
                                dtype=np.uint8,
                            )
                            for im in wrist
                        ]
                    )
                t = int(acts.shape[0])
                steps = []
                for i in range(t):
                    steps.append(
                        {
                            "observation": {
                                "image": imgs[i],
                                "wrist_image": wrist[i],
                                "state": state[i].astype(np.float32),
                                "odor": odor[i].astype(np.float32),
                                "odor_valid": bool(odor_valid[i]),
                                "odor_padding_mask": odor_padding_mask[i],
                            },
                            "action": acts[i].astype(np.float32),
                            "discount": np.float32(1.0),
                            "reward": np.float32(1.0 if i == t - 1 else 0.0),
                            "is_first": i == 0,
                            "is_last": i == t - 1,
                            "is_terminal": i == t - 1,
                            "language_instruction": ep["instruction"],
                        }
                    )
                yield ep["file_path"], {
                    "steps": steps,
                    "episode_metadata": {
                        "file_path": ep["file_path"],
                        "episode_id": ep["episode_id"],
                        "scene_id": ep["scene_id"],
                        "instructed_liquid": ep["instructed_liquid"],
                        "sniff_duration_s": np.float32(ep["sniff_duration_s"]),
                        "odor_interface": "scentformer_embedding_only",
                    },
                }

    Builder = type(dataset_name, (_Builder,), {})

    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    dest = data_dir / dataset_name
    if dest.exists():
        print(f"[build_rlds_ovla] removing existing {dest}")
        shutil.rmtree(dest)

    from tensorflow_datasets.core.utils import gcs_utils

    gcs_utils._is_gcs_disabled = True
    builder = Builder(data_dir=str(data_dir))
    download_config = tfds.download.DownloadConfig(try_download_gcs=False)
    builder.download_and_prepare(download_config=download_config)
    print(f"[build_rlds_ovla] done → {data_dir / dataset_name}")
    return data_dir / dataset_name


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Build odor-aware RLDS from ClassifyLiquidPlaceOVLA raw episodes"
    )
    p.add_argument(
        "--input",
        type=str,
        required=True,
        help="Raw episodes root (…/odor_ovla_oft/raw)",
    )
    p.add_argument(
        "--manifest",
        type=str,
        default=None,
        help="Optional manifest.csv (preferred episode list)",
    )
    p.add_argument("--output", type=str, default=DEFAULT_OUTPUT)
    p.add_argument("--dataset-name", type=str, default=DEFAULT_DATASET_NAME)
    p.add_argument("--val-frac", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument(
        "--odor-history-s", type=float, default=DEFAULT_ODOR_HISTORY_S,
        help="continuous causal MOX history passed to ScentFormer (default 5 s)",
    )
    p.add_argument(
        "--sniff-landmarks-s", type=float, nargs="+",
        default=DEFAULT_SNIFF_LANDMARKS_S,
        help="seconds after sniff onset retained as HOLD actions; decision frame is always kept",
    )
    p.add_argument(
        "--sniff-landmark-jitter-s", type=float, default=0.1,
        help="deterministic per-episode timing jitter applied to non-final HOLD landmarks",
    )
    return p


def main(argv=None) -> int:
    args = build_argparser().parse_args(argv)
    raw_root = Path(args.input)
    manifest = Path(args.manifest) if args.manifest else None
    if not raw_root.is_dir():
        raise SystemExit(f"input not found: {raw_root}")
    episode_dirs = discover_episode_dirs(raw_root, manifest)
    if not episode_dirs:
        raise SystemExit(f"no episodes found under {raw_root}")
    for ep_dir in episode_dirs:
        meta = json.loads((ep_dir / "meta.json").read_text(encoding="utf-8"))
        validate_deployment_protocol(meta, ep_dir)
    print(f"[build_rlds_ovla] found {len(episode_dirs)} episodes")
    build_tfds_dataset(
        episode_dirs,
        data_dir=Path(args.output),
        dataset_name=args.dataset_name,
        val_frac=float(args.val_frac),
        seed=int(args.seed),
        image_size=int(args.image_size),
        sniff_landmarks_s=tuple(float(value) for value in args.sniff_landmarks_s),
        sniff_landmark_jitter_s=float(args.sniff_landmark_jitter_s),
        odor_history_s=float(args.odor_history_s),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

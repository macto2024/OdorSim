"""Build an odor-aware TFDS/RLDS dataset from OVLA ClassifyLiquidPlace demos.

Unlike :mod:`odor_sim.recording.build_rlds_odorplace` (vision-only), this
builder:

* drops sniff/filter hover ticks using ``episode.npz['phase']``, keeping only
  the last SAMPLE step of each sniff as a decision frame
* attaches ``observation.odor`` ``(T_sniff, C)`` from ``odor_events.npz``
* records episode IDs so later no-odor / shuffled-odor views can swap traces

OpenVLA-OFT treats every remaining RLDS step as a training sample, so collapsing
holds is required to avoid teaching a stationary-hover prior.

Example::

    python -m odor_sim.recording.build_rlds_odor_ovla \\
        --input /root/Odorbot/datasets/odor_ovla_oft/raw \\
        --manifest /root/Odorbot/datasets/odor_ovla_oft/manifest.csv \\
        --output /root/Odorbot/datasets/rlds \\
        --dataset-name odorsim_ovla_odorplace_sniff10s
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from odor_sim.recording.build_rlds_odorplace import (
    _build_proprio_state,
    _load_frames,
    _split_episodes,
    discover_episode_dirs,
    is_noop,
)
from odor_sim.recording.odor_config import DEFAULT_ODOR_CONFIG
from odor_sim.recording.odor_events import collapse_keep_mask, load_odor_events

DEFAULT_DATASET_NAME = "odorsim_ovla_odorplace_sniff10s"
DEFAULT_OUTPUT = "/root/Odorbot/datasets/rlds"


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
    end_steps: np.ndarray,
    t_sniff: int,
    num_channels: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Per kept step: latest completed sniff trace, or zeros + ``odor_valid=0``."""
    n = int(keep_indices.shape[0])
    odor = np.zeros((n, t_sniff, num_channels), dtype=np.float32)
    valid = np.zeros((n,), dtype=np.bool_)
    if voltage.size == 0 or end_steps.size == 0:
        return odor, valid
    for i, step in enumerate(keep_indices):
        ev = _latest_event_index(end_steps, int(step))
        if ev < 0:
            continue
        odor[i] = voltage[ev]
        valid[i] = True
    return odor, valid


def collapse_episode_steps(
    *,
    phase: np.ndarray,
    action7: np.ndarray,
    sniff_end_steps: list[int],
) -> tuple[np.ndarray, int, int]:
    """Return original indices to keep after phase collapse + no-op filter.

    Returns ``(keep_indices, n_hold_dropped, n_noop)``.
    """
    keep_mask = collapse_keep_mask(phase, sniff_end_steps)
    n_hold = int((~keep_mask).sum())
    idxs = np.flatnonzero(keep_mask)
    kept = []
    prev = None
    n_noop = 0
    for i in idxs:
        a = action7[int(i)]
        if is_noop(a, prev):
            n_noop += 1
            prev = a
            continue
        kept.append(int(i))
        prev = a
    if not kept:
        raise ValueError("all steps filtered (holds + no-ops)")
    return np.asarray(kept, dtype=np.int64), n_hold, n_noop


def parse_episode(ep_dir: Path, *, load_images: bool = True) -> dict:
    """Load one OVLA raw episode, collapse holds, attach odor traces."""
    meta = json.loads((ep_dir / "meta.json").read_text(encoding="utf-8"))
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
    end_steps = np.asarray([int(ev["end_step"]) for ev in events], dtype=np.int64)
    t_sniff = int(odor_pack["meta"].get("sniff_steps") or (voltage.shape[1] if voltage.ndim == 3 else 0))
    num_channels = int(
        voltage.shape[2]
        if voltage.ndim == 3
        else len(meta.get("mox_models") or DEFAULT_ODOR_CONFIG.mox_models)
    )
    if t_sniff <= 0:
        t_sniff = int(meta.get("sniff_steps") or DEFAULT_ODOR_CONFIG.sniff_steps)

    keep_idx, n_hold, n_noop = collapse_episode_steps(
        phase=phase,
        action7=action7,
        sniff_end_steps=[int(x) for x in end_steps],
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

    odor, odor_valid = attach_odor(
        keep_idx,
        voltage=voltage,
        end_steps=end_steps,
        t_sniff=t_sniff,
        num_channels=num_channels,
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
        "instruction": instruction,
        "file_path": str(ep_dir.resolve()),
        "n_hold": n_hold,
        "n_noop": n_noop,
        "t_sniff": t_sniff,
        "num_channels": num_channels,
        "meta": meta,
        "episode_id": str(meta.get("episode_id") or ep_dir.name),
        "scene_id": str(meta.get("scene_id") or ""),
        "seed": meta.get("seed"),
        "sniff_duration_s": float(meta.get("sniff_duration_s") or 0.0),
        "mox_models": [str(x) for x in (meta.get("mox_models") or [])],
        "liquids": [str(x) for x in (meta.get("liquids") or meta.get("objects") or [])],
        "instructed_liquid": str(meta.get("instructed_liquid") or ""),
    }


def _infer_odor_shape(episode_dirs: list[Path]) -> tuple[int, int]:
    first = parse_episode(episode_dirs[0], load_images=False)
    return int(first["t_sniff"]), int(first["num_channels"])


def build_tfds_dataset(
    episode_dirs: list[Path],
    *,
    data_dir: Path,
    dataset_name: str,
    val_frac: float = 0.05,
    seed: int = 0,
    image_size: int = 256,
) -> Path:
    """Create a TFDS builder and run ``download_and_prepare``."""
    import tensorflow as tf
    import tensorflow_datasets as tfds

    t_sniff, num_channels = _infer_odor_shape(episode_dirs)
    splits = _split_episodes(episode_dirs, val_frac, seed)
    print(
        f"[build_rlds_ovla] episodes={len(episode_dirs)} "
        f"train={len(splits['train'])} val={len(splits['val'])} "
        f"odor=({t_sniff}, {num_channels})"
    )

    parsed_cache: dict[str, dict] = {}

    def _get(path: Path) -> dict:
        key = str(path.resolve())
        if key not in parsed_cache:
            parsed_cache[key] = parse_episode(path, load_images=True)
            ep = parsed_cache[key]
            if int(ep["t_sniff"]) != t_sniff or int(ep["num_channels"]) != num_channels:
                raise ValueError(
                    f"{path}: odor shape {(ep['t_sniff'], ep['num_channels'])} "
                    f"!= dataset {(t_sniff, num_channels)}; collect one sniff duration per RLDS build"
                )
            print(
                f"  parsed {path.name}: T={ep['action'].shape[0]} "
                f"(dropped {ep['n_hold']} holds, {ep['n_noop']} no-ops)"
            )
        return parsed_cache[key]

    split_paths = {k: [str(p.resolve()) for p in v] for k, v in splits.items() if v}

    class _Builder(tfds.core.GeneratorBasedBuilder):
        VERSION = tfds.core.Version("1.0.0")
        RELEASE_NOTES = {
            "1.0.0": "OVLA odor pick-place demos with collapsed sniff holds."
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
    print(f"[build_rlds_ovla] found {len(episode_dirs)} episodes")
    build_tfds_dataset(
        episode_dirs,
        data_dir=Path(args.output),
        dataset_name=args.dataset_name,
        val_frac=float(args.val_frac),
        seed=int(args.seed),
        image_size=int(args.image_size),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

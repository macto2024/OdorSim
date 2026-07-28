"""Build a TFDS/RLDS dataset from OdorPlace raw episodes for OpenVLA-OFT.

Reads ``episode.npz`` + ``frames/{agentview,wrist}`` (+ optional manifest.csv)
and writes a TensorFlow Datasets RLDS builder under::

    <output>/<dataset-name>/

Matching LIBERO-style fields expected by openvla-oft ``finetune.py``:

    observation/image, observation/wrist_image, observation/state (8,),
    action (7,), language_instruction

Near-zero (no-op) actions are filtered like LIBERO ``*_no_noops``.

Example::

    python -m odor_sim.recording.build_rlds_odorplace \\
        --input /root/Odorbot/datasets/odorplace_oft/raw \\
        --manifest /root/Odorbot/datasets/odorplace_oft/manifest.csv \\
        --output /root/Odorbot/datasets/rlds \\
        --dataset-name odorsim_odorplace_no_noops
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from pathlib import Path

import numpy as np

DEFAULT_DATASET_NAME = "odorsim_odorplace_no_noops"
DEFAULT_OUTPUT = "/root/Odorbot/datasets/rlds"
NOOP_THRESHOLD = 1e-4


def quat2axisangle(quat: np.ndarray) -> np.ndarray:
    """Quaternion (x, y, z, w) → axis-angle (LIBERO / robosuite convention)."""
    quat = np.asarray(quat, dtype=np.float64).copy()
    quat[3] = float(np.clip(quat[3], -1.0, 1.0))
    den = math.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3, dtype=np.float64)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def is_noop(action: np.ndarray, prev_action: "np.ndarray | None", threshold: float = NOOP_THRESHOLD) -> bool:
    """LIBERO-style no-op filter (near-zero arm delta + unchanged gripper)."""
    action = np.asarray(action, dtype=np.float64).ravel()
    if prev_action is None:
        return float(np.linalg.norm(action[:-1])) < threshold
    prev_action = np.asarray(prev_action, dtype=np.float64).ravel()
    return (
        float(np.linalg.norm(action[:-1])) < threshold
        and float(action[-1]) == float(prev_action[-1])
    )


def _load_frames(ep_dir: Path, subdir: str) -> np.ndarray:
    from PIL import Image

    frame_dir = ep_dir / "frames" / subdir
    paths = sorted(frame_dir.glob("*.png"))
    if not paths:
        raise FileNotFoundError(f"no PNG frames in {frame_dir}")
    return np.stack(
        [np.asarray(Image.open(p).convert("RGB"), dtype=np.uint8) for p in paths],
        axis=0,
    )


def _build_proprio_state(episode: dict) -> np.ndarray:
    """(T, 8) = eef_pos(3) + axis_angle(3) + gripper_qpos(2)."""
    eef_pos = np.asarray(episode["eef_pos"], dtype=np.float32)
    eef_quat = np.asarray(episode["eef_quat"], dtype=np.float32)
    grip = np.asarray(episode["gripper_qpos"], dtype=np.float32)
    t = eef_pos.shape[0]
    aa = np.stack([quat2axisangle(eef_quat[i]).astype(np.float32) for i in range(t)], axis=0)
    if grip.ndim == 1:
        grip = grip.reshape(-1, 1)
    if grip.shape[1] == 1:
        grip = np.concatenate([grip, grip], axis=1)
    return np.concatenate([eef_pos[:, :3], aa[:, :3], grip[:, :2]], axis=1).astype(np.float32)


def _filter_noops(
    actions: np.ndarray,
    images: np.ndarray,
    wrist: np.ndarray,
    state: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    keep_img, keep_wrist, keep_state, keep_act = [], [], [], []
    prev = None
    n_noop = 0
    for i in range(actions.shape[0]):
        a = actions[i]
        if is_noop(a, prev):
            n_noop += 1
            prev = a
            continue
        keep_img.append(images[i])
        keep_wrist.append(wrist[i])
        keep_state.append(state[i])
        keep_act.append(a)
        prev = a
    if not keep_act:
        raise ValueError("all steps filtered as no-ops")
    return (
        np.stack(keep_img),
        np.stack(keep_wrist),
        np.stack(keep_state),
        np.stack(keep_act).astype(np.float32),
        n_noop,
    )


def discover_episode_dirs(raw_root: Path, manifest: "Path | None") -> list[Path]:
    """Episode dirs from manifest paths, else recursive search under raw_root."""
    if manifest is not None and manifest.is_file():
        dirs = []
        with manifest.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                p = Path(row["episode_path"])
                if p.is_dir():
                    dirs.append(p)
                else:
                    # Fallback: relative under raw_root
                    alt = raw_root / row.get("scene", "") / row["episode_id"]
                    # scene column is A_single etc; on-disk is scene_a_single
                    candidates = list(raw_root.rglob(row["episode_id"]))
                    if candidates:
                        dirs.append(candidates[0])
                    elif alt.is_dir():
                        dirs.append(alt)
        return dirs

    dirs = []
    for meta in sorted(raw_root.rglob("meta.json")):
        ep = meta.parent
        if (ep / "episode.npz").is_file() and (ep / "frames" / "agentview").is_dir():
            dirs.append(ep)
    return dirs


def parse_episode(ep_dir: Path) -> dict:
    """Load one raw episode into arrays ready for RLDS steps."""
    meta = json.loads((ep_dir / "meta.json").read_text(encoding="utf-8"))
    data = np.load(ep_dir / "episode.npz", allow_pickle=False)
    action = np.asarray(data["action"], dtype=np.float32)
    if action.shape[1] < 7:
        raise ValueError(f"{ep_dir}: action dim {action.shape[1]} < 7")
    action7 = action[:, :7].copy()

    images = _load_frames(ep_dir, "agentview")
    wrist = _load_frames(ep_dir, "wrist")
    state = _build_proprio_state({k: data[k] for k in data.files})

    t = min(action7.shape[0], images.shape[0], wrist.shape[0], state.shape[0])
    action7, images, wrist, state = action7[:t], images[:t], wrist[:t], state[:t]

    images, wrist, state, action7, n_noop = _filter_noops(action7, images, wrist, state)
    instruction = str(meta.get("instruction", "")).strip()
    if not instruction:
        raise ValueError(f"{ep_dir}: empty instruction")

    return {
        "images": images,
        "wrist": wrist,
        "state": state,
        "action": action7,
        "instruction": instruction,
        "file_path": str(ep_dir.resolve()),
        "n_noop": n_noop,
        "meta": meta,
    }


def _split_episodes(eps: list[Path], val_frac: float, seed: int) -> dict[str, list[Path]]:
    rng = np.random.default_rng(seed)
    order = list(eps)
    rng.shuffle(order)
    n_val = max(1, int(round(len(order) * val_frac))) if len(order) >= 20 else max(0, len(order) // 20)
    if len(order) <= 1:
        n_val = 0
    val = order[:n_val]
    train = order[n_val:] or order
    return {"train": train, "val": val}


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

    splits = _split_episodes(episode_dirs, val_frac, seed)
    print(
        f"[build_rlds] episodes={len(episode_dirs)} "
        f"train={len(splits['train'])} val={len(splits['val'])}"
    )

    parsed_cache: dict[str, dict] = {}

    def _get(path: Path) -> dict:
        key = str(path.resolve())
        if key not in parsed_cache:
            parsed_cache[key] = parse_episode(path)
            print(
                f"  parsed {path.name}: T={parsed_cache[key]['action'].shape[0]} "
                f"(filtered {parsed_cache[key]['n_noop']} no-ops)"
            )
        return parsed_cache[key]

    split_paths = {k: [str(p.resolve()) for p in v] for k, v in splits.items() if v}

    class _Builder(tfds.core.GeneratorBasedBuilder):
        VERSION = tfds.core.Version("1.0.0")
        RELEASE_NOTES = {"1.0.0": "OdorPlace scripted demos for OpenVLA-OFT."}

        def _info(self) -> tfds.core.DatasetInfo:
            return tfds.core.DatasetInfo(
                builder=self,
                description="OdorSim OdorPlace pick-place demos (scripted OSC).",
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
                            }
                        ),
                    }
                ),
                supervised_keys=None,
            )

        def _split_generators(self, dl_manager):
            # Must return generators (or beam), not kwargs dicts — a bare dict is
            # Iterable and TFDS will iterate its keys ("paths") and crash.
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
                    print(f"[build_rlds] skip {ep_dir}: {exc}")
                    continue
                imgs = ep["images"]
                wrist = ep["wrist"]
                state = ep["state"]
                acts = ep["action"]
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
                t = acts.shape[0]
                steps = []
                for i in range(t):
                    steps.append(
                        {
                            "observation": {
                                "image": imgs[i],
                                "wrist_image": wrist[i],
                                "state": state[i].astype(np.float32),
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
                    "episode_metadata": {"file_path": ep["file_path"]},
                }

    # Class name == TFDS dataset name on disk.
    Builder = type(dataset_name, (_Builder,), {})

    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    dest = data_dir / dataset_name
    if dest.exists():
        print(f"[build_rlds] removing existing {dest}")
        shutil.rmtree(dest)

    # Fully disable TFDS GCS (gs://tfds-data) BEFORE Builder() — constructor calls
    # info.initialize_from_bucket() which probes dataset_info/<name> via path.exists()
    # and hangs ~60s/retry on offline hosts. try_download_gcs=False alone is not enough.
    from tensorflow_datasets.core.utils import gcs_utils

    gcs_utils._is_gcs_disabled = True
    builder = Builder(data_dir=str(data_dir))
    download_config = tfds.download.DownloadConfig(try_download_gcs=False)
    builder.download_and_prepare(download_config=download_config)
    print(f"[build_rlds] done → {data_dir / dataset_name}")
    return data_dir / dataset_name


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build RLDS dataset from OdorPlace raw episodes")
    p.add_argument(
        "--input",
        type=str,
        required=True,
        help="Raw episodes root (…/odorplace_oft/raw)",
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
    print(f"[build_rlds] found {len(episode_dirs)} episodes")
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

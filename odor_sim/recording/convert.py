"""Phase 5c: convert raw OdorSim episode dirs to a LeRobotDataset v3.0.

Second stage of the two-stage pipeline: raw episodes on disk
(from teleop mining, Phase 5a) are mapped into a loadable LeRobot dataset via
``lerobot >= 0.6`` (``LeRobotDataset`` codebase version ``v3.0``: chunked
parquet + MP4-encoded camera streams).

Feature mapping:

  * ``observation.images.<cam>``  <- ``frames/<cam>/*.png`` (agentview, wrist)
  * ``observation.state``         <- structured proprio, concatenated
                                     (joint_pos, joint_vel, eef_pos, eef_quat,
                                      gripper_qpos); falls back to flat ``state``
  * ``action``                    <- recorded ``action`` (robosuite + enose dim)
  * ``observation.ppm``           <- raw per-gas ppm at the EE
  * ``observation.enose_state``   <- ternary sampling token
  * ``observation.enose_voltage_continuous`` / ``...sampling`` <- live
                                     ``episode.npz`` keys and/or Phase 5b
                                     ``features.npz`` (only if present)
  * ``observation.source_identity`` / ``observation.source_distance``
                                     <- ground-truth odor labels (episode.npz
                                     or features.npz; only if present)
  * ``task``                      <- ``meta.instruction``

Odor voltage is kept as its own observation channel (not folded into
``observation.state``) so Phase 6 adapters can ablate it freely.

Usage::

    source setup/activate.sh
    python -m odor_sim.recording.convert --input datasets/teleop --output odorsim/odorlift
    # dataset written under datasets/lerobot/odorsim/odorlift by default
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from odor_sim.recording.synthesize import _find_episode_dirs

# Structured proprio concatenation order for observation.state.
_STATE_KEYS = ("joint_pos", "joint_vel", "eef_pos", "eef_quat", "gripper_qpos")
DEFAULT_LEROBOT_ROOT = "datasets/lerobot"


def _read_frames(ep_dir: Path, subdir: str) -> np.ndarray:
    """Load a camera's PNG sequence as (T, H, W, 3) uint8."""
    from PIL import Image

    frame_dir = ep_dir / "frames" / subdir
    paths = sorted(frame_dir.glob("*.png"))
    if not paths:
        raise FileNotFoundError(f"no PNG frames in {frame_dir}")
    return np.stack([np.asarray(Image.open(p).convert("RGB"), dtype=np.uint8) for p in paths])


def _episode_state(episode) -> "tuple[np.ndarray, list[str]]":
    """Concatenated structured proprio (or flat state fallback) + dim names."""
    keys = [k for k in _STATE_KEYS if k in episode.files]
    if not keys:
        state = np.asarray(episode["state"], dtype=np.float32)
        names = [f"state_{i}" for i in range(state.shape[1])]
        return state, names
    cols = [np.asarray(episode[k], dtype=np.float32) for k in keys]
    state = np.concatenate(cols, axis=1)
    names = [f"{k}_{i}" for k, col in zip(keys, cols) for i in range(col.shape[1])]
    return state, names


def _build_features(sample: dict, cameras: dict, use_videos: bool) -> dict:
    """LeRobot feature spec from one episode's array shapes."""
    img_dtype = "video" if use_videos else "image"
    features: dict = {}
    for cam in cameras:
        h, w, c = sample["images"][cam].shape[1:]
        features[f"observation.images.{cam}"] = {
            "dtype": img_dtype,
            "shape": [int(h), int(w), int(c)],
            "names": ["height", "width", "channels"],
        }
    features["observation.state"] = {
        "dtype": "float32",
        "shape": [len(sample["state_names"])],
        "names": sample["state_names"],
    }
    features["action"] = {
        "dtype": "float32",
        "shape": [int(sample["action"].shape[1])],
        "names": [f"action_{i}" for i in range(sample["action"].shape[1])],
    }
    features["observation.ppm"] = {
        "dtype": "float32",
        "shape": [len(sample["gas_types"])],
        "names": list(sample["gas_types"]),
    }
    features["observation.enose_state"] = {
        "dtype": "float32",
        "shape": [1],
        "names": ["enose_state"],
    }
    if sample["has_voltage"]:
        for name in ("enose_voltage_continuous", "enose_voltage_sampling"):
            if name in sample["voltage"]:
                features[f"observation.{name}"] = {
                    "dtype": "float32",
                    "shape": [1],
                    "names": [name],
                }
    if sample.get("enose_voltages") is not None:
        n = int(sample["enose_voltages"].shape[1])
        names = list(sample.get("mox_models") or [f"mox_{i}" for i in range(n)])
        features["observation.enose_voltages"] = {
            "dtype": "float32",
            "shape": [n],
            "names": names,
        }
    if sample["source_identity"] is not None:
        features["observation.source_identity"] = {
            "dtype": "float32",
            "shape": [1],
            "names": ["source_identity"],
        }
    if sample["source_distance"] is not None:
        features["observation.source_distance"] = {
            "dtype": "float32",
            "shape": [1],
            "names": ["source_distance"],
        }
    return features


def _assert_synced(name: str, length: int, T: int) -> None:
    if length != T:
        raise ValueError(f"sync check failed: {name} has length {length}, expected T={T}")


def _load_episode(ep_dir: Path, cameras_meta: dict):
    """Load one raw episode into per-step arrays for LeRobot ingestion."""
    episode = np.load(ep_dir / "episode.npz")
    meta = json.loads((ep_dir / "meta.json").read_text())

    images = {cam: _read_frames(ep_dir, subdir) for cam, subdir in cameras_meta.items()}
    state, state_names = _episode_state(episode)
    action = np.asarray(episode["action"], dtype=np.float32)
    ppm = np.asarray(episode["ppm"], dtype=np.float32)
    enose_state = np.asarray(episode["enose_state"], dtype=np.float32).reshape(-1, 1)

    T = action.shape[0]
    gas_types = list(meta.get("gas_types", []))

    voltage = {}
    # Prefer live-mined voltage in episode.npz; fill gaps from features.npz.
    for name in ("enose_voltage_continuous", "enose_voltage_sampling"):
        if name in episode.files:
            voltage[name] = np.asarray(episode[name], dtype=np.float32).reshape(-1, 1)
    if not voltage and "enose_voltage" in episode.files:
        odor_mode = str(meta.get("odor_mode") or "none")
        if odor_mode == "discrete":
            key = "enose_voltage_sampling"
        elif odor_mode == "continuous":
            key = "enose_voltage_continuous"
        else:
            key = None
        if key is not None:
            voltage[key] = np.asarray(episode["enose_voltage"], dtype=np.float32).reshape(-1, 1)

    # Multi-sensor live array (T, n_models) + per-model columns when present.
    enose_voltages_multi = None
    mox_models = list(meta.get("mox_models") or [])
    if "enose_voltages" in episode.files:
        enose_voltages_multi = np.asarray(episode["enose_voltages"], dtype=np.float32)
        if enose_voltages_multi.ndim == 1:
            enose_voltages_multi = enose_voltages_multi.reshape(-1, 1)

    features_path = ep_dir / "features.npz"
    feats = None
    if features_path.is_file():
        feats = np.load(features_path, allow_pickle=True)
        for name in ("enose_voltage_continuous", "enose_voltage_sampling"):
            if name not in voltage and name in feats.files:
                voltage[name] = np.asarray(feats[name], dtype=np.float32).reshape(-1, 1)

    source_identity = None
    source_distance = None
    if "source_identity" in episode.files:
        source_identity = np.asarray(episode["source_identity"], dtype=np.float32).reshape(-1, 1)
    elif feats is not None and "source_identity" in feats.files:
        source_identity = np.asarray(feats["source_identity"], dtype=np.float32).reshape(-1, 1)
    if "source_distance" in episode.files:
        source_distance = np.asarray(episode["source_distance"], dtype=np.float32).reshape(-1, 1)
    elif feats is not None and "source_distance" in feats.files:
        source_distance = np.asarray(feats["source_distance"], dtype=np.float32).reshape(-1, 1)

    # Synchronization: every modality must match action length T.
    _assert_synced("ppm", ppm.shape[0], T)
    _assert_synced("enose_state", enose_state.shape[0], T)
    _assert_synced("state", state.shape[0], T)
    if gas_types and ppm.ndim == 2 and ppm.shape[1] != len(gas_types):
        raise ValueError(
            f"sync check failed: ppm columns {ppm.shape[1]} != len(gas_types)={len(gas_types)}"
        )
    for cam, frames in images.items():
        _assert_synced(f"images.{cam}", frames.shape[0], T)
    for name, arr in voltage.items():
        _assert_synced(name, arr.shape[0], T)
    if enose_voltages_multi is not None:
        _assert_synced("enose_voltages", enose_voltages_multi.shape[0], T)
        if mox_models and enose_voltages_multi.shape[1] != len(mox_models):
            raise ValueError(
                f"sync check failed: enose_voltages cols "
                f"{enose_voltages_multi.shape[1]} != len(mox_models)={len(mox_models)}"
            )
    if source_identity is not None:
        _assert_synced("source_identity", source_identity.shape[0], T)
    if source_distance is not None:
        _assert_synced("source_distance", source_distance.shape[0], T)

    return {
        "num_steps": T,
        "instruction": meta.get("instruction", ""),
        "gas_types": gas_types,
        "class_names": list(meta.get("class_names") or meta.get("objects") or []),
        "num_classes": meta.get("num_classes"),
        "images": images,
        "state": state,
        "state_names": state_names,
        "action": action,
        "ppm": ppm,
        "enose_state": enose_state,
        "voltage": voltage,
        "has_voltage": bool(voltage) or enose_voltages_multi is not None,
        "enose_voltages": enose_voltages_multi,
        "mox_models": mox_models,
        "source_identity": source_identity,
        "source_distance": source_distance,
    }


def convert_episodes(
    inputs,
    repo_id: str,
    *,
    root: "str | Path | None" = None,
    fps: "int | None" = None,
    use_videos: bool = True,
    robot_type: "str | None" = None,
) -> "Path":
    """Convert raw episode dir(s) into a LeRobotDataset v3.0 on disk.

    Args:
        inputs: one path or list of paths; each may be a single episode dir or a
            parent dir containing ``episode_*/`` subdirs.
        repo_id: LeRobot repo id (also the on-disk dataset name).
        root: parent dir for the local dataset (default ``datasets/lerobot``);
            the dataset is written to ``root/repo_id``.
        fps: control frequency; defaults to the first episode's ``control_freq``.
        use_videos: encode camera streams as MP4 (v3.0 default). False stores
            frames as PNG image features.
        robot_type: optional robot type tag for the dataset metadata.

    Returns:
        The dataset root :class:`~pathlib.Path`.
    """
    # Keep the converter fully local/offline (no Hugging Face Hub round-trips).
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    if isinstance(inputs, (str, Path)):
        inputs = [inputs]
    episode_dirs: list[Path] = []
    for p in inputs:
        episode_dirs.extend(_find_episode_dirs(Path(p)))
    if not episode_dirs:
        raise FileNotFoundError(f"no episodes found under {inputs}")

    # Only episodes with camera frames can become a VLA dataset.
    usable: list[tuple[Path, dict]] = []
    for ep in episode_dirs:
        meta = json.loads((ep / "meta.json").read_text())
        cameras = meta.get("cameras")
        if not cameras:
            print(f"[convert] skip {ep.name}: no camera frames in meta")
            continue
        cameras_meta = {cam: Path(rel).name for cam, rel in cameras.items()}
        usable.append((ep, cameras_meta))
    if not usable:
        raise ValueError("no episodes with camera frames to convert")

    first_ep, first_cams = usable[0]
    first = _load_episode(first_ep, first_cams)
    if fps is None:
        fps = int(json.loads((first_ep / "meta.json").read_text()).get("control_freq", 20))

    features = _build_features(first, first_cams, use_videos)

    root_dir = Path(root) if root is not None else Path(DEFAULT_LEROBOT_ROOT)
    dataset_root = root_dir / repo_id
    if dataset_root.exists():
        import shutil

        shutil.rmtree(dataset_root)

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        features=features,
        root=dataset_root,
        robot_type=robot_type,
        use_videos=use_videos,
    )

    total_frames = 0
    for ep_dir, cameras_meta in usable:
        data = first if ep_dir == first_ep else _load_episode(ep_dir, cameras_meta)
        T = data["num_steps"]
        for t in range(T):
            frame = {
                "observation.state": data["state"][t],
                "action": data["action"][t],
                "observation.ppm": data["ppm"][t],
                "observation.enose_state": data["enose_state"][t],
                "task": data["instruction"],
            }
            for cam in cameras_meta:
                frame[f"observation.images.{cam}"] = data["images"][cam][t]
            for name, arr in data["voltage"].items():
                frame[f"observation.{name}"] = arr[t]
            if data.get("enose_voltages") is not None:
                frame["observation.enose_voltages"] = data["enose_voltages"][t]
            if data["source_identity"] is not None:
                frame["observation.source_identity"] = data["source_identity"][t]
            if data["source_distance"] is not None:
                frame["observation.source_distance"] = data["source_distance"][t]
            dataset.add_frame(frame)
        dataset.save_episode()
        total_frames += T
        print(f"[convert] {ep_dir.name}: {T} steps")

    print(
        f"[convert] wrote {len(usable)} episode(s), {total_frames} frames "
        f"-> {dataset_root} (repo_id={repo_id}, fps={fps})"
    )
    return dataset_root


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--input",
        nargs="+",
        required=True,
        help="raw episode dir(s) or parent dir(s) with episode_*/ subdirs",
    )
    parser.add_argument("--output", required=True, help="LeRobot repo id / dataset name")
    parser.add_argument(
        "--root",
        default=DEFAULT_LEROBOT_ROOT,
        help=f"parent dir for the local dataset (default {DEFAULT_LEROBOT_ROOT})",
    )
    parser.add_argument("--fps", type=int, default=None, help="control frequency (default: episode control_freq)")
    parser.add_argument(
        "--no-videos",
        action="store_true",
        help="store camera frames as image features instead of MP4",
    )
    parser.add_argument("--robot-type", default=None, help="robot type tag for dataset metadata")
    args = parser.parse_args(argv)

    convert_episodes(
        args.input,
        repo_id=args.output,
        root=args.root,
        fps=args.fps,
        use_videos=not args.no_videos,
        robot_type=args.robot_type,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

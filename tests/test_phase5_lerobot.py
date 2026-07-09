"""Phase 5c test: raw episode -> LeRobotDataset v3.0 -> load-back (no ROS).

Builds a synthetic raw episode fixture (structured proprio + action + ppm +
enose_state, plus agentview/wrist PNG frame sequences), synthesizes 5b features,
converts to a LeRobotDataset v3.0, then re-opens it and asserts the frame count,
episode count and feature shapes round-trip. GADEN/ROS is not needed: the
converter only reads files on disk.

    source setup/activate.sh
    python tests/test_phase5_lerobot.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

# LeRobot/HF datasets need a writable cache + no network. Set before import.
_HF_HOME = Path(tempfile.mkdtemp(prefix="hf_home_"))
os.environ["HF_HOME"] = str(_HF_HOME)
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

T = 12
H = W = 96
GAS_TYPES = ["ethanol", "acetone"]
ACTION_DIM = 8


def _make_fixture(ep_dir: Path) -> None:
    from PIL import Image

    (ep_dir / "frames" / "agentview").mkdir(parents=True)
    (ep_dir / "frames" / "wrist").mkdir(parents=True)

    ppm = np.zeros((T, 2), dtype=np.float32)
    ppm[4:9, 0] = 40.0
    enose = np.zeros(T, dtype=np.int8)
    enose[5:8] = 1

    np.savez(
        ep_dir / "episode.npz",
        sim_time=np.arange(T, dtype=np.float32) * 0.05,
        state=np.random.rand(T, 10).astype(np.float32),
        action=np.random.rand(T, ACTION_DIM).astype(np.float32),
        enose_state=enose,
        sampling_active=(enose == 1),
        ppm=ppm,
        joint_pos=np.random.rand(T, 7).astype(np.float32),
        joint_vel=np.random.rand(T, 7).astype(np.float32),
        eef_pos=np.random.rand(T, 3).astype(np.float32),
        eef_quat=np.random.rand(T, 4).astype(np.float32),
        gripper_qpos=np.random.rand(T, 2).astype(np.float32),
    )
    meta = {
        "instruction": "pick up the ripe fruit object",
        "gas_types": GAS_TYPES,
        "control_freq": 20,
        "num_steps": T,
        "robots": "Panda",
        "controller_type": "OSC_POSE",
        "cameras": {"agentview": "frames/agentview", "wrist": "frames/wrist"},
        "image_size": [H, W],
        "proprio_keys": ["eef_pos", "eef_quat", "gripper_qpos", "joint_pos", "joint_vel"],
    }
    (ep_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    rng = np.random.default_rng(0)
    for cam in ("agentview", "wrist"):
        for t in range(T):
            img = rng.integers(0, 256, size=(H, W, 3), dtype=np.uint8)
            Image.fromarray(img).save(ep_dir / "frames" / cam / f"{t:06d}.png")


def main() -> int:
    from odor_sim.recording.convert import convert_episodes
    from odor_sim.recording.synthesize import synthesize_episode

    failures = 0

    def check(name, ok, detail=""):
        nonlocal failures
        if ok:
            print(f"PASS  {name}  {detail}")
        else:
            failures += 1
            print(f"FAIL  {name}  {detail}")

    root = Path(tempfile.mkdtemp(prefix="lerobot_test_"))
    ep_dir = root / "teleop" / "episode_fixture"
    _make_fixture(ep_dir)

    synthesize_episode(ep_dir)  # write features.npz for the odor voltage channels

    ds_root = convert_episodes(
        root / "teleop", repo_id="odorsim/test", root=str(root / "lerobot"), use_videos=True
    )
    check("dataset_created", (ds_root / "meta" / "info.json").exists(), str(ds_root))

    from lerobot.datasets.lerobot_dataset import CODEBASE_VERSION, LeRobotDataset

    ds = LeRobotDataset(repo_id="odorsim/test", root=ds_root)
    check("codebase_v3", CODEBASE_VERSION == "v3.0", CODEBASE_VERSION)
    check("num_frames", ds.num_frames == T, f"{ds.num_frames} == {T}")
    check("num_episodes", ds.num_episodes == 1, f"{ds.num_episodes}")
    check("fps", ds.fps == 20, f"{ds.fps}")

    feats = set(ds.features.keys())
    expected = {
        "observation.images.agentview",
        "observation.images.wrist",
        "observation.state",
        "action",
        "observation.ppm",
        "observation.enose_state",
        "observation.enose_voltage_continuous",
        "observation.enose_voltage_sampling",
    }
    check("features_present", expected.issubset(feats), str(sorted(expected - feats)))

    item = ds[0]
    check("action_shape", tuple(item["action"].shape) == (ACTION_DIM,), str(tuple(item["action"].shape)))
    # structured proprio concatenated: 7+7+3+4+2 = 23
    check("state_shape", tuple(item["observation.state"].shape) == (23,), str(tuple(item["observation.state"].shape)))
    check("ppm_shape", tuple(item["observation.ppm"].shape) == (len(GAS_TYPES),), str(tuple(item["observation.ppm"].shape)))

    ag = item["observation.images.agentview"]
    # v3.0 decodes video frames to CHW float tensors.
    shape = tuple(int(x) for x in ag.shape)
    check("image_chw", len(shape) == 3 and shape[0] == 3 and shape[1:] == (H, W), str(shape))

    check("task_language", ds[0]["task"] == "pick up the ripe fruit object", str(ds[0].get("task")))

    total = 12
    print(f"\n{total - failures}/{total} checks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

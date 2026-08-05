"""Stationary odor-sniff collector (no trajectory / camera frames).

Records labeled multi-MOX voltages from a parked (optionally approached) EE
for ScentFormer-style training. Reuses ``odor_sim.make`` + GADEN lockstep;
MuJoCo is only a pose/source provider.

Example::

    source setup/activate.sh
    python -m odor_sim.bridge.collect_odor \\
      --vocs ethanol methane acetone hydrogen propanol chlorine \\
      --duration-s 20 --odor-mode continuous \\
      --num-per-label 50 --window-s 10 --stride-s 5 \\
      --out-dir datasets/odor_sniffs/voc6_continuous
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import json
import multiprocessing as mp
import os
import re
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from odor_sim.runtime.session import FILTER, IDLE, SAMPLE, resolve_mox_models
from odor_sim.sensors.mox_pid import DEFAULT_MOX_MODELS

# OSC_POSE maps action in [-1, 1] -> EE delta in meters (BASIC default ±0.05 m).
_OSC_POS_SCALE = 0.05
_GRIPPER_OPEN = -1.0

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_OUT = "datasets/odor_sniffs"
_DEFAULT_NUM_WORKERS = 1
_DEFAULT_ROS_DOMAIN_BASE = 40
_MANIFEST_FIELDS = [
    "label",
    "sniff_idx",
    "spawn",
    "strength",
    "path",
    "seed",
    "odor_mode",
    "duration_s",
    "num_steps",
    "status",
    "timestamp",
    "worker_id",
]

# Pure-VOC liquid catalog uses names like ethanol0.3 / ethanol1.0
_DEFAULT_STRENGTHS = (0.3, 0.7, 1.0)


def _format_strength(strength: float) -> str:
    """Match liquids.yaml naming: 0.3 -> '0.3', 1.0 -> '1.0'."""
    return f"{float(strength):.1f}"


def voc_liquid_name(voc: str, strength: float) -> str:
    """Catalog liquid name for a pure VOC at a given strength."""
    return f"{voc}{_format_strength(strength)}"


def resolve_collection_jobs(
    *,
    liquids: list[str] | None,
    objects: list[str] | None,
    vocs: list[str] | None,
    strengths: list[float],
    num_per_label: int,
) -> tuple[str, list[str], list[dict]]:
    """Build per-sniff jobs.

    Returns ``(kind, label_names, jobs)`` where each job is
    ``{label, spawn, strength, sniff_idx, seed_offset}``.

    With ``--vocs``, every strength is cycled under the base VOC label
    (e.g. spawn ethanol0.3 / 0.7 / 1.0, label ``ethanol``).
    """
    if vocs:
        kind = "liquid"
        if not strengths:
            raise ValueError("--strengths must be non-empty when using --vocs")
        label_names = list(vocs)
        jobs: list[dict] = []
        for label in label_names:
            for i in range(num_per_label):
                strength = float(strengths[i % len(strengths)])
                jobs.append(
                    {
                        "label": label,
                        "spawn": voc_liquid_name(label, strength),
                        "strength": strength,
                        "sniff_idx": i,
                        "seed_offset": i,
                    }
                )
        return kind, label_names, jobs

    if liquids:
        kind = "liquid"
        label_names = list(liquids)
        jobs = [
            {
                "label": lab,
                "spawn": lab,
                "strength": None,
                "sniff_idx": i,
                "seed_offset": i,
            }
            for lab in label_names
            for i in range(num_per_label)
        ]
        return kind, label_names, jobs

    if objects:
        kind = "object"
        label_names = list(objects)
        jobs = [
            {
                "label": lab,
                "spawn": lab,
                "strength": None,
                "sniff_idx": i,
                "seed_offset": i,
            }
            for lab in label_names
            for i in range(num_per_label)
        ]
        return kind, label_names, jobs

    raise ValueError("one of --vocs, --liquids, or --objects is required")


# --------------------------------------------------------------------------- #
# Discrete SAMPLE → FILTER → IDLE cycle
# --------------------------------------------------------------------------- #
class DiscreteEnoseScheduler:
    """Cycle SAMPLE → FILTER → IDLE for the full sniff duration."""

    def __init__(self, sample_steps: int, filter_steps: int, idle_steps: int = 1):
        self.sample_steps = max(1, int(sample_steps))
        self.filter_steps = max(1, int(filter_steps))
        self.idle_steps = max(1, int(idle_steps))
        self._phase = "sample"
        self._left = self.sample_steps

    def next(self) -> int:
        if self._phase == "sample":
            state = SAMPLE
            self._left -= 1
            if self._left <= 0:
                self._phase = "filter"
                self._left = self.filter_steps
            return state
        if self._phase == "filter":
            state = FILTER
            self._left -= 1
            if self._left <= 0:
                self._phase = "idle"
                self._left = self.idle_steps
            return state
        # Idle gap, then restart SAMPLE so long sniffs keep cycling.
        state = IDLE
        self._left -= 1
        if self._left <= 0:
            self._phase = "sample"
            self._left = self.sample_steps
        return state


# --------------------------------------------------------------------------- #
# Lightweight approach helpers
# --------------------------------------------------------------------------- #
def _eef_pos(env, obs) -> np.ndarray:
    for key in ("robot0_eef_pos", "robot0_right_eef_pos"):
        if key in obs:
            return np.asarray(obs[key], dtype=float).ravel()[:3]
    return np.asarray(env.get_enose_site_pose()[0], dtype=float).ravel()[:3]


def _gripper_dof(env) -> int:
    try:
        robot = env.robots[0]
        arm = robot.arms[0]
        return int(robot.gripper[arm].dof)
    except Exception:  # noqa: BLE001
        return 1


def _target_object_id(env) -> str:
    if hasattr(env, "target_object_name"):
        return str(env.target_object_name)
    if hasattr(env, "target_liquid_name"):
        return str(env.target_liquid_name)
    poses = env.get_object_world_poses()
    if not poses:
        raise RuntimeError("env has no odor objects to approach")
    return next(iter(poses.keys()))


def _sniff_target_pos(env, *, z_clearance: float = 0.12) -> np.ndarray:
    """Point above the target object center for a stationary sniff."""
    oid = _target_object_id(env)
    pos = np.asarray(env.get_object_world_poses()[oid][0], dtype=float).ravel()[:3]
    out = pos.copy()
    out[2] = float(pos[2]) + float(z_clearance)
    return out


def _pose_delta_action(env, obs, target_pos: np.ndarray, *, pos_gain: float = 4.0) -> np.ndarray:
    cur = _eef_pos(env, obs)
    err = np.asarray(target_pos, dtype=float).ravel()[:3] - cur
    unit = np.clip((pos_gain * err) / _OSC_POS_SCALE, -1.0, 1.0)
    action = np.zeros(int(env.action_dim), dtype=float)
    n_pose = min(3, action.size)
    action[:n_pose] = unit[:n_pose]
    gdof = _gripper_dof(env)
    if gdof > 0 and action.size >= gdof:
        action[-gdof:] = _GRIPPER_OPEN
    return action


def _at_target(env, obs, target_pos: np.ndarray, *, xy_tol: float = 0.04, z_tol: float = 0.05) -> bool:
    cur = _eef_pos(env, obs)
    xy_ok = np.linalg.norm(cur[:2] - target_pos[:2]) < xy_tol
    z_ok = abs(float(cur[2] - target_pos[2])) < z_tol
    return bool(xy_ok and z_ok)


def _pack_action(cosim, env_action: np.ndarray, enose_state: int = IDLE) -> np.ndarray:
    env_action = np.asarray(env_action, dtype=float).ravel()
    if cosim.odor_mode == "discrete":
        return np.concatenate([env_action, [float(enose_state)]])
    return env_action


def approach_target(cosim, obs, *, max_steps: int = 120, z_clearance: float = 0.12):
    """Move EE above the odor source (IDLE enose); returns final obs."""
    env = cosim.env
    target = _sniff_target_pos(env, z_clearance=z_clearance)
    for _ in range(max_steps):
        if _at_target(env, obs, target):
            break
        env_action = _pose_delta_action(env, obs, target)
        obs, _, _, _ = cosim.step(_pack_action(cosim, env_action, IDLE))
    # brief settle hold
    zero = np.zeros(int(env.action_dim), dtype=float)
    gdof = _gripper_dof(env)
    if gdof > 0:
        zero[-gdof:] = _GRIPPER_OPEN
    for _ in range(10):
        obs, _, _, _ = cosim.step(_pack_action(cosim, zero, IDLE))
    return obs


# --------------------------------------------------------------------------- #
# Recording / IO
# --------------------------------------------------------------------------- #
def _ppm_row(ppm: dict, gas_types: list[str]) -> np.ndarray:
    return np.asarray([float((ppm or {}).get(g, 0.0)) for g in gas_types], dtype=np.float32)


def _voltage_row(voltages: dict, mox_models: list[str]) -> np.ndarray:
    return np.asarray([float((voltages or {}).get(m, 0.0)) for m in mox_models], dtype=np.float32)


def _load_manifest_done(manifest_path: Path) -> set[tuple[str, int]]:
    done: set[tuple[str, int]] = set()
    if not manifest_path.is_file():
        return done
    with manifest_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("status") != "ok":
                continue
            try:
                done.add((row["label"], int(row["sniff_idx"])))
            except (KeyError, ValueError):
                continue
    return done


def _append_manifest(manifest_path: Path, row: dict) -> None:
    """Append one manifest row with an exclusive file lock (parallel-safe)."""
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("a+", newline="") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            f.seek(0)
            empty = f.read(1) == ""
            f.seek(0, os.SEEK_END)
            writer = csv.DictWriter(f, fieldnames=_MANIFEST_FIELDS)
            if empty:
                writer.writeheader()
            writer.writerow({k: row.get(k, "") for k in _MANIFEST_FIELDS})
            f.flush()
            os.fsync(f.fileno())
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def _safe_scene_id(*parts: object) -> str:
    raw = "_".join(str(p) for p in parts if p is not None and str(p) != "")
    return re.sub(r"[^0-9A-Za-z_]+", "_", raw).strip("_").lower() or "collect_odor"

def save_sniff(
    *,
    out_path: Path,
    voltages: np.ndarray,
    ppm: np.ndarray,
    enose_state: np.ndarray,
    sim_time: np.ndarray,
    source_distance: np.ndarray,
    label: str,
    meta: dict,
) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        voltages=np.asarray(voltages, dtype=np.float32),
        ppm=np.asarray(ppm, dtype=np.float32),
        enose_state=np.asarray(enose_state, dtype=np.int8),
        sim_time=np.asarray(sim_time, dtype=np.float32),
        source_distance=np.asarray(source_distance, dtype=np.float32),
        label=np.asarray(label),
        gas_types=np.asarray(meta["gas_types"]),
        mox_models=np.asarray(meta["mox_models"]),
    )
    meta_path = out_path.with_name(out_path.stem + "_meta.json")
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
    return out_path


def sliding_windows(
    voltages: np.ndarray,
    *,
    window_steps: int,
    stride_steps: int,
) -> np.ndarray:
    """Return ``(N, Tw, C)`` windows from ``(T, C)`` voltages."""
    x = np.asarray(voltages, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError(f"voltages must be (T, C), got {x.shape}")
    t, _c = x.shape
    if window_steps <= 0 or t < window_steps:
        return np.zeros((0, max(window_steps, 0), x.shape[1]), dtype=np.float32)
    stride = max(1, int(stride_steps))
    starts = range(0, t - window_steps + 1, stride)
    return np.stack([x[s : s + window_steps] for s in starts], axis=0).astype(np.float32)


def export_window_dataset(
    out_dir: Path,
    *,
    labels: list[str],
    window_s: float,
    stride_s: float,
    rate: float,
) -> Path | None:
    """Consolidate all sniff_*.npz under ``out_dir`` into windows/dataset.npz."""
    window_steps = max(1, int(round(window_s * rate)))
    stride_steps = max(1, int(round(stride_s * rate)))
    xs: list[np.ndarray] = []
    ys: list[int] = []
    sniff_ids: list[str] = []
    label_names = list(labels)

    for li, label in enumerate(label_names):
        label_dir = out_dir / label
        if not label_dir.is_dir():
            continue
        for sniff_path in sorted(label_dir.glob("sniff_*.npz")):
            data = np.load(sniff_path, allow_pickle=True)
            volts = np.asarray(data["voltages"], dtype=np.float32)
            wins = sliding_windows(volts, window_steps=window_steps, stride_steps=stride_steps)
            if wins.shape[0] == 0:
                continue
            xs.append(wins)
            ys.extend([li] * int(wins.shape[0]))
            sniff_ids.extend([f"{label}/{sniff_path.stem}"] * int(wins.shape[0]))

    win_dir = out_dir / "windows"
    win_dir.mkdir(parents=True, exist_ok=True)
    classes_path = win_dir / "classes.json"
    classes_path.write_text(
        json.dumps(
            {
                "label_names": label_names,
                "window_s": float(window_s),
                "stride_s": float(stride_s),
                "rate": float(rate),
                "window_steps": window_steps,
                "stride_steps": stride_steps,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    if not xs:
        print(f"[collect_odor] no windows written under {win_dir}")
        return None

    X = np.concatenate(xs, axis=0)
    y = np.asarray(ys, dtype=np.int64)
    sniff_id_arr = np.asarray(sniff_ids)
    dataset_path = win_dir / "dataset.npz"
    np.savez_compressed(
        dataset_path,
        X=X,
        y=y,
        sniff_id=sniff_id_arr,
        label_names=np.asarray(label_names),
    )
    print(
        f"[collect_odor] windows → {dataset_path}  "
        f"X={tuple(X.shape)} classes={len(label_names)}"
    )
    return dataset_path


# --------------------------------------------------------------------------- #
# One sniff
# --------------------------------------------------------------------------- #
def collect_one_sniff(
    *,
    label: str,
    spawn: str,
    kind: str,  # "liquid" | "object"
    odor_mode: str,
    duration_s: float,
    control_freq: float,
    mox_models: list[str],
    approach: bool,
    sample_hold_s: float,
    filter_hold_s: float,
    seed: int,
    scenario: str,
    scenario_config: str,
    z_clearance: float,
    approach_max_steps: int,
    strength: float | None = None,
    scene_id: str | None = None,
    kill_stale_gaden: bool = True,
    server_log_dir: str | Path | None = None,
    worker_id: int = 0,
) -> tuple[dict, dict]:
    """Run one sniff; return ``(arrays, meta)``.

    ``spawn`` is the catalog liquid/object name passed to the env.
    ``label`` is the class name written into the dataset (may collapse strengths).
    """
    import odor_sim as odorsim

    num_steps = max(1, int(round(duration_s * control_freq)))
    env_name = "ClassifyLiquid" if kind == "liquid" else "OdorLift"
    make_kwargs: dict = dict(
        odor_mode=odor_mode,
        mox_model=mox_models,
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        control_freq=control_freq,
        scenario=scenario,
        scenario_config=scenario_config,
        seed=seed,
        kill_stale_gaden=kill_stale_gaden,
        server_log_dir=server_log_dir,
    )
    if scene_id:
        make_kwargs["scene_id"] = scene_id
    if kind == "liquid":
        make_kwargs["liquids"] = [spawn]
        make_kwargs["target_liquid"] = spawn
    else:
        make_kwargs["objects"] = [spawn]
        make_kwargs["target_object"] = spawn

    scheduler = None
    if odor_mode == "discrete":
        scheduler = DiscreteEnoseScheduler(
            sample_steps=max(1, int(round(sample_hold_s * control_freq))),
            filter_steps=max(1, int(round(filter_hold_s * control_freq))),
        )

    with odorsim.make(env_name, **make_kwargs) as cosim:
        obs = cosim.reset()
        env = cosim.env
        gas_types = list(env.gas_types)
        if approach:
            obs = approach_target(
                cosim, obs, max_steps=approach_max_steps, z_clearance=z_clearance
            )

        volts_rows = []
        ppm_rows = []
        state_rows = []
        time_rows = []
        dist_rows = []

        zero = np.zeros(int(env.action_dim), dtype=float)
        gdof = _gripper_dof(env)
        if gdof > 0:
            zero[-gdof:] = _GRIPPER_OPEN

        target_id = _target_object_id(env)
        for _ in range(num_steps):
            enose_state = scheduler.next() if scheduler is not None else IDLE
            obs, _, _, info = cosim.step(_pack_action(cosim, zero, enose_state))
            voltages = info.get("enose_voltages") or {}
            ppm = info.get("ppm") or {}
            volts_rows.append(_voltage_row(voltages, mox_models))
            ppm_rows.append(_ppm_row(ppm, gas_types))
            state_rows.append(int(info.get("enose_state", enose_state)))
            time_rows.append(float(info.get("sim_time", 0.0)))
            enose_pos = env.get_enose_site_pose()[0]
            target_pos = env.get_object_world_poses()[target_id][0]
            dist_rows.append(
                float(
                    np.linalg.norm(
                        np.asarray(enose_pos, dtype=float) - np.asarray(target_pos, dtype=float)
                    )
                )
            )

        arrays = {
            "voltages": np.stack(volts_rows, axis=0),
            "ppm": np.stack(ppm_rows, axis=0),
            "enose_state": np.asarray(state_rows, dtype=np.int8),
            "sim_time": np.asarray(time_rows, dtype=np.float32),
            "source_distance": np.asarray(dist_rows, dtype=np.float32),
        }
        meta = {
            "label": label,
            "spawn": spawn,
            "strength": float(strength) if strength is not None else None,
            "kind": kind,
            "odor_mode": odor_mode,
            "duration_s": float(duration_s),
            "control_freq": float(control_freq),
            "num_steps": int(num_steps),
            "mox_models": list(mox_models),
            "gas_types": list(gas_types),
            "seed": int(seed),
            "approach": bool(approach),
            "z_clearance": float(z_clearance),
            "sample_hold_s": float(sample_hold_s) if odor_mode == "discrete" else None,
            "filter_hold_s": float(filter_hold_s) if odor_mode == "discrete" else None,
            "scene_id": getattr(cosim, "scene_id", None),
            "env": env_name,
            "worker_id": int(worker_id),
            "note": "odor-only sniff (no trajectory / frames)",
        }
        return arrays, meta


def _execute_job(payload: dict) -> dict:
    """Run one sniff job end-to-end (used by serial and ProcessPool workers)."""
    out_dir = Path(payload["out_dir"])
    label = payload["label"]
    spawn = payload["spawn"]
    sniff_idx = int(payload["sniff_idx"])
    strength = payload.get("strength")
    seed = int(payload["seed"])
    worker_id = int(payload.get("worker_id", 0))
    label_dir = out_dir / label
    label_dir.mkdir(parents=True, exist_ok=True)
    sniff_path = label_dir / f"sniff_{sniff_idx:03d}.npz"
    manifest_path = out_dir / "manifest.csv"

    if sniff_path.exists():
        row = {
            "label": label,
            "sniff_idx": sniff_idx,
            "spawn": spawn,
            "strength": "" if strength is None else strength,
            "path": str(sniff_path.relative_to(out_dir)),
            "seed": seed,
            "odor_mode": payload["odor_mode"],
            "duration_s": payload["duration_s"],
            "num_steps": int(round(payload["duration_s"] * payload["control_freq"])),
            "status": "ok",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "worker_id": worker_id,
        }
        _append_manifest(manifest_path, row)
        return row

    scene_id = _safe_scene_id("collect", spawn, f"w{worker_id}", sniff_idx)
    server_log_dir = _REPO_ROOT / ".roslog" / f"collect_odor_w{worker_id}"
    print(
        f"[collect_odor:w{worker_id}] {label} sniff_{sniff_idx:03d} "
        f"spawn={spawn} strength={strength} seed={seed} ...",
        flush=True,
    )
    try:
        arrays, meta = collect_one_sniff(
            label=label,
            spawn=spawn,
            kind=payload["kind"],
            odor_mode=payload["odor_mode"],
            duration_s=payload["duration_s"],
            control_freq=payload["control_freq"],
            mox_models=payload["mox_models"],
            approach=payload["approach"],
            sample_hold_s=payload["sample_hold_s"],
            filter_hold_s=payload["filter_hold_s"],
            seed=seed,
            scenario=payload["scenario"],
            scenario_config=payload["scenario_config"],
            z_clearance=payload["z_clearance"],
            approach_max_steps=payload["approach_max_steps"],
            strength=strength,
            scene_id=scene_id,
            kill_stale_gaden=payload.get("kill_stale_gaden", True),
            server_log_dir=server_log_dir,
            worker_id=worker_id,
        )
        meta["sniff_idx"] = sniff_idx
        save_sniff(
            out_path=sniff_path,
            voltages=arrays["voltages"],
            ppm=arrays["ppm"],
            enose_state=arrays["enose_state"],
            sim_time=arrays["sim_time"],
            source_distance=arrays["source_distance"],
            label=label,
            meta=meta,
        )
        if payload.get("window_s", 0) and payload["window_s"] > 0:
            w_steps = max(1, int(round(payload["window_s"] * payload["control_freq"])))
            s_steps = max(1, int(round(payload["stride_s"] * payload["control_freq"])))
            wins = sliding_windows(
                arrays["voltages"], window_steps=w_steps, stride_steps=s_steps
            )
            win_dir = label_dir / "windows"
            win_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                win_dir / f"sniff_{sniff_idx:03d}_windows.npz",
                X=wins,
                label=np.asarray(label),
                spawn=np.asarray(spawn),
                window_steps=w_steps,
                stride_steps=s_steps,
            )
        row = {
            "label": label,
            "sniff_idx": sniff_idx,
            "spawn": spawn,
            "strength": "" if strength is None else strength,
            "path": str(sniff_path.relative_to(out_dir)),
            "seed": seed,
            "odor_mode": payload["odor_mode"],
            "duration_s": payload["duration_s"],
            "num_steps": meta["num_steps"],
            "status": "ok",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "worker_id": worker_id,
        }
        _append_manifest(manifest_path, row)
        print(
            f"[collect_odor:w{worker_id}] wrote {sniff_path}  "
            f"voltages={tuple(arrays['voltages'].shape)}",
            flush=True,
        )
        return row
    except Exception as exc:  # noqa: BLE001
        row = {
            "label": label,
            "sniff_idx": sniff_idx,
            "spawn": spawn,
            "strength": "" if strength is None else strength,
            "path": "",
            "seed": seed,
            "odor_mode": payload["odor_mode"],
            "duration_s": payload["duration_s"],
            "num_steps": "",
            "status": f"fail:{type(exc).__name__}",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "worker_id": worker_id,
        }
        _append_manifest(manifest_path, row)
        print(
            f"[collect_odor:w{worker_id}] FAIL {label} sniff_{sniff_idx:03d}: {exc}",
            flush=True,
        )
        return row


def _worker_run_job(payload: dict) -> dict:
    """ProcessPool entry: isolate ROS domain, then run one sniff."""
    os.environ.setdefault("MUJOCO_GL", "egl")
    domain = int(payload.get("ros_domain_id", _DEFAULT_ROS_DOMAIN_BASE))
    os.environ["ROS_DOMAIN_ID"] = str(domain)
    # Prefer per-worker ROS logs so parallel rclpy inits don't collide.
    log_root = _REPO_ROOT / ".roslog" / f"collect_odor_w{payload.get('worker_id', 0)}"
    log_root.mkdir(parents=True, exist_ok=True)
    os.environ["ROS_LOG_DIR"] = str(log_root / "rcl")
    return _execute_job(payload)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Collect stationary labeled MOX sniffs (no robot trajectories)."
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--vocs",
        nargs="+",
        default=None,
        help=(
            "Base VOC names (ClassifyLiquid). Expands each to strengths "
            f"{list(_DEFAULT_STRENGTHS)} (override with --strengths) but labels "
            "as the VOC name only, e.g. ethanol0.3/0.7/1.0 → label 'ethanol'."
        ),
    )
    src.add_argument(
        "--liquids",
        nargs="+",
        default=None,
        help="Exact liquid catalog names (ClassifyLiquid), e.g. ethanol1.0 coke",
    )
    src.add_argument(
        "--objects",
        nargs="+",
        default=None,
        help="Catalog object labels with multi-VOC recipes (OdorLift), e.g. mango milk",
    )
    p.add_argument(
        "--strengths",
        nargs="+",
        type=float,
        default=list(_DEFAULT_STRENGTHS),
        help="Strengths used with --vocs (default: 0.3 0.7 1.0).",
    )
    p.add_argument("--duration-s", type=float, default=20.0, help="Sniff length in seconds.")
    p.add_argument(
        "--odor-mode",
        choices=["continuous", "discrete"],
        required=True,
        help="Live MOX path (required).",
    )
    p.add_argument(
        "--num-per-label",
        type=int,
        default=50,
        help="Sniffs per class label (with --vocs, strengths are cycled across these).",
    )
    p.add_argument("--control-freq", type=float, default=20.0, help="Control / sensor rate (Hz).")
    p.add_argument(
        "--mox-model",
        nargs="+",
        default=None,
        help=f"MOX models to record (default: {list(DEFAULT_MOX_MODELS)}).",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default=_DEFAULT_OUT,
        help="Output root (default: datasets/odor_sniffs).",
    )
    p.add_argument("--seed", type=int, default=0, help="Base seed (sniff i uses seed+i).")
    p.add_argument("--sample-hold-s", type=float, default=7.0, help="Discrete SAMPLE hold (s).")
    p.add_argument("--filter-hold-s", type=float, default=10.0, help="Discrete FILTER hold (s).")
    p.add_argument(
        "--approach-target",
        dest="approach",
        action="store_true",
        default=True,
        help="Move EE above the source before sniffing (default: on).",
    )
    p.add_argument(
        "--no-approach-target",
        dest="approach",
        action="store_false",
        help="Hold at reset EE pose (skip approach).",
    )
    p.add_argument("--z-clearance", type=float, default=0.12, help="Approach height above object (m).")
    p.add_argument("--approach-max-steps", type=int, default=120, help="Max OSC steps for approach.")
    p.add_argument("--window-s", type=float, default=10.0, help="Sliding window length (s). 0 disables.")
    p.add_argument("--stride-s", type=float, default=5.0, help="Sliding window stride (s).")
    p.add_argument(
        "--num-workers",
        type=int,
        default=_DEFAULT_NUM_WORKERS,
        help=(
            "Parallel worker processes (default 1 = serial). Each worker uses "
            "its own ROS_DOMAIN_ID + GADEN server. Prefer 2–6 depending on CPU/RAM."
        ),
    )
    p.add_argument(
        "--ros-domain-base",
        type=int,
        default=_DEFAULT_ROS_DOMAIN_BASE,
        help=(
            f"Base ROS_DOMAIN_ID for workers (default {_DEFAULT_ROS_DOMAIN_BASE}); "
            "worker i uses base+i. Avoid colliding with other ROS sessions."
        ),
    )
    p.add_argument("--scenario", type=str, default="10x6_uniform")
    p.add_argument("--scenario-config", type=str, default="config1")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print plan and exit without starting GADEN.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.duration_s <= 0:
        raise SystemExit("--duration-s must be > 0")
    if args.num_per_label <= 0:
        raise SystemExit("--num-per-label must be > 0")
    if args.control_freq <= 0:
        raise SystemExit("--control-freq must be > 0")
    if args.num_workers <= 0:
        raise SystemExit("--num-workers must be > 0")
    if args.vocs and not args.strengths:
        raise SystemExit("--strengths must be non-empty when using --vocs")

    kind, label_names, jobs = resolve_collection_jobs(
        liquids=args.liquids,
        objects=args.objects,
        vocs=args.vocs,
        strengths=list(args.strengths),
        num_per_label=args.num_per_label,
    )

    mox_models = resolve_mox_models(args.mox_model)
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = _REPO_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    for lab in label_names:
        (out_dir / lab).mkdir(parents=True, exist_ok=True)

    manifest_path = out_dir / "manifest.csv"
    done = _load_manifest_done(manifest_path)
    pending = [j for j in jobs if (j["label"], j["sniff_idx"]) not in done]

    print(
        f"[collect_odor] kind={kind} labels={label_names} mode={args.odor_mode} "
        f"duration={args.duration_s}s rate={args.control_freq}Hz "
        f"n={args.num_per_label} jobs={len(jobs)} pending={len(pending)} "
        f"workers={args.num_workers} mox={mox_models} out={out_dir}"
    )
    if args.vocs:
        print(f"[collect_odor] strengths={args.strengths} (cycled; label=VOC name)")
    if args.dry_run:
        for j in pending[:12]:
            print(
                f"  plan: label={j['label']} spawn={j['spawn']} "
                f"strength={j['strength']} sniff_{j['sniff_idx']:03d}"
            )
        if len(pending) > 12:
            print(f"  ... ({len(pending) - 12} more pending)")
        print(
            f"[collect_odor] dry-run: {len(jobs)} sniffs planned, "
            f"{len(jobs) - len(pending)} done, {len(pending)} pending"
        )
        return 0

    if not pending:
        print("[collect_odor] nothing pending")
        if args.window_s and args.window_s > 0:
            export_window_dataset(
                out_dir,
                labels=label_names,
                window_s=args.window_s,
                stride_s=args.stride_s,
                rate=args.control_freq,
            )
        return 0

    payloads = []
    for i, job in enumerate(pending):
        label = job["label"]
        sniff_idx = int(job["sniff_idx"])
        seed = int(args.seed) + label_names.index(label) * int(args.num_per_label) + sniff_idx
        worker_id = i % int(args.num_workers)
        payloads.append(
            {
                "out_dir": str(out_dir),
                "label": label,
                "spawn": job["spawn"],
                "strength": job["strength"],
                "sniff_idx": sniff_idx,
                "seed": seed,
                "kind": kind,
                "odor_mode": args.odor_mode,
                "duration_s": float(args.duration_s),
                "control_freq": float(args.control_freq),
                "mox_models": list(mox_models),
                "approach": bool(args.approach),
                "sample_hold_s": float(args.sample_hold_s),
                "filter_hold_s": float(args.filter_hold_s),
                "scenario": args.scenario,
                "scenario_config": args.scenario_config,
                "z_clearance": float(args.z_clearance),
                "approach_max_steps": int(args.approach_max_steps),
                "window_s": float(args.window_s),
                "stride_s": float(args.stride_s),
                "worker_id": worker_id,
                "ros_domain_id": int(args.ros_domain_base) + worker_id,
                # Parallel: never kill sibling GADEN servers.
                "kill_stale_gaden": args.num_workers == 1,
            }
        )

    ok = 0
    fail = 0
    if args.num_workers == 1:
        for i, payload in enumerate(payloads):
            print(
                f"[collect_odor] ({i + 1}/{len(payloads)}) "
                f"{payload['label']} sniff_{payload['sniff_idx']:03d}",
                flush=True,
            )
            row = _execute_job(payload)
            if str(row.get("status", "")).startswith("ok"):
                ok += 1
            else:
                fail += 1
    else:
        print(
            f"[collect_odor] launching {args.num_workers} workers on "
            f"{len(payloads)} sniffs (ROS_DOMAIN_ID "
            f"{args.ros_domain_base}..{args.ros_domain_base + args.num_workers - 1}, "
            f"MUJOCO_GL=egl)",
            flush=True,
        )
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=int(args.num_workers), mp_context=ctx
        ) as pool:
            futures = {pool.submit(_worker_run_job, p): p for p in payloads}
            for fut in as_completed(futures):
                p = futures[fut]
                try:
                    row = fut.result()
                except Exception as exc:  # noqa: BLE001
                    fail += 1
                    print(
                        f"[collect_odor] worker error "
                        f"{p['label']} sniff_{p['sniff_idx']:03d}: {exc}",
                        flush=True,
                    )
                    continue
                if str(row.get("status", "")).startswith("ok"):
                    ok += 1
                else:
                    fail += 1
                print(
                    f"[collect_odor] progress ok={ok} fail={fail} "
                    f"done={ok + fail}/{len(payloads)}",
                    flush=True,
                )

    if args.window_s and args.window_s > 0:
        export_window_dataset(
            out_dir,
            labels=label_names,
            window_s=args.window_s,
            stride_s=args.stride_s,
            rate=args.control_freq,
        )

    print(
        f"[collect_odor] done ok={ok} fail={fail}. manifest → {manifest_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

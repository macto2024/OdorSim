"""Batch collection for the odor-conditioned ClassifyLiquidPlaceOVLA task.

Each episode uses two identical cups whose physical GADEN sources remain
active throughout. The continuous eight-channel MOX state is never reset
between cup sniffs. By default a rejected first cup transitions directly to
the second cup, matching the deployment policy.

Parallelism (each worker gets its own ROS_DOMAIN_ID + GADEN server)::

    python -m odor_sim.bridge.collect_odor_place_oft \\
        --out-dir /root/Odorbot/datasets/odor_ovla_oft \\
        --num-demos 50 --num-workers 4 --quiet
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
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from odor_sim.bridge.scripted_odor_place import (
    DEFAULT_DECISION_DOMINANCE,
    DEFAULT_DECISION_MIN_PASS_FRACTION,
    DEFAULT_DECISION_MIN_TOTAL_PPM,
    DEFAULT_DECISION_WINDOW_S,
    DEFAULT_FILTER_HOLD_S,
    DEFAULT_OVLA_ACTION_NOISE,
    DEFAULT_OVLA_ACTION_NOISE_FLOOR,
    DEFAULT_SAMPLE_HOLD_S,
    DEFAULT_TARGET_FIRST_FRAC,
    ScriptedOdorPlaceCollector,
)
from odor_sim.bridge.scripted_place import (
    add_motion_shaping_args,
    motion_shaping_kwargs,
)
from odor_sim.recording.odor_config import (
    DEFAULT_ODOR_CONFIG,
    DEFAULT_VOC_POOL,
    FILTER_DURATION_FACTOR,
    odor_config_from_args,
)

DEFAULT_OUT_DIR = "/root/Odorbot/datasets/odor_ovla_oft"
_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_ROS_DOMAIN_BASE = 50
DEFAULT_NUM_WORKERS = 1

DEFAULT_NUM_CUPS = 2
DEFAULT_LOCAL_PPM_CALIBRATION = (
    _REPO_ROOT / "odor_sim/config/phase_b_local_ppm_exp0p5_scale0p06_fps15.json"
)

MANIFEST_FIELDS = [
    "plan_index",
    "planned_target_first",
    "attempt_index",
    "episode_id",
    "episode_path",
    "liquids",
    "instructed_liquid",
    "target_present",
    "instruction",
    "num_steps",
    "seed",
    "success",
    "worker_id",
    "scene_id",
    "sniff_duration_s",
    "filter_duration_s",
    "odor_sampling_rate",
    "mox_models",
    "voc_pool",
    "transition_mode",
    "wind_regime",
    "emission_scale",
    "concentration_exponent",
    "filaments_per_second",
    "original_terminal_velocity_enabled",
    "buoyancy_tau_s",
    "source_base_local_ppm",
    "source_pair_ratio",
    "source_condition_sampling",
    "decision_window_s",
    "decision_min_total_ppm",
    "decision_dominance_threshold",
    "decision_min_pass_fraction",
]


def _safe_scene_id(*parts) -> str:
    raw = "_".join(str(p) for p in parts if p is not None and str(p) != "")
    return re.sub(r"[^a-zA-Z0-9_]+", "_", raw).strip("_").lower() or "scene"


def _read_meta(ep_dir: Path) -> dict:
    meta_path = ep_dir / "meta.json"
    if not meta_path.is_file():
        return {}
    return json.loads(meta_path.read_text(encoding="utf-8"))


def _manifest_row(
    ep_dir: Path,
    *,
    liquids,
    instructed,
    seed: int,
    worker_id: int = 0,
    plan_index: "int | None" = None,
    planned_target_first: "bool | None" = None,
    attempt_index: "int | None" = None,
) -> dict:
    meta = _read_meta(ep_dir)
    mox = meta.get("mox_models") or []
    if isinstance(mox, list):
        mox_s = ",".join(str(x) for x in mox)
    else:
        mox_s = str(mox)
    voc = meta.get("voc_pool") or liquids
    if isinstance(voc, list):
        voc_s = ";".join(str(x) for x in voc)
    else:
        voc_s = str(voc)
    return {
        "plan_index": "" if plan_index is None else int(plan_index),
        "planned_target_first": (
            "" if planned_target_first is None else bool(planned_target_first)
        ),
        "attempt_index": "" if attempt_index is None else int(attempt_index),
        "episode_id": meta.get("episode_id") or ep_dir.name,
        "episode_path": str(ep_dir.resolve()),
        "liquids": ";".join(liquids),
        "instructed_liquid": instructed,
        "target_present": bool(meta.get("target_present", instructed in liquids)),
        "instruction": meta.get("instruction", ""),
        "num_steps": int(meta.get("num_steps", 0)),
        "seed": int(seed),
        "success": True,
        "worker_id": int(worker_id),
        "scene_id": meta.get("scene_id", ""),
        "sniff_duration_s": meta.get("sniff_duration_s", ""),
        "filter_duration_s": meta.get("filter_duration_s", ""),
        "odor_sampling_rate": meta.get("odor_sampling_rate", ""),
        "mox_models": mox_s,
        "voc_pool": voc_s,
        "transition_mode": meta.get("transition_mode", ""),
        "wind_regime": meta.get("wind_regime", ""),
        "emission_scale": meta.get("emission_scale", ""),
        "concentration_exponent": meta.get("concentration_exponent", ""),
        "filaments_per_second": meta.get("filaments_per_second", ""),
        "original_terminal_velocity_enabled": meta.get(
            "original_terminal_velocity_enabled", ""
        ),
        "buoyancy_tau_s": meta.get("buoyancy_tau_s", ""),
        "source_base_local_ppm": meta.get("source_base_local_ppm", ""),
        "source_pair_ratio": meta.get("source_pair_ratio", ""),
        "source_condition_sampling": meta.get("source_condition_sampling", ""),
        "decision_window_s": meta.get("decision_window_s", ""),
        "decision_min_total_ppm": meta.get("decision_min_total_ppm", ""),
        "decision_dominance_threshold": meta.get(
            "decision_dominance_threshold", ""
        ),
        "decision_min_pass_fraction": meta.get("decision_min_pass_fraction", ""),
    }


def _load_local_ppm_calibration(path: str | Path) -> dict:
    calibration = json.loads(Path(path).read_text(encoding="utf-8"))
    if calibration.get("model") != "log_linear_nominal_to_local_median":
        raise ValueError(f"unsupported local-ppm calibration in {path}")
    return calibration


def _local_to_nominal(local_ppm: float, calibration: dict) -> float:
    return float(np.exp(
        (np.log(float(local_ppm)) - float(calibration["log_intercept"]))
        / float(calibration["log_slope"])
    ))


def _plan_target_first(
    plan_index: int,
    target_first_frac: float,
    rng: np.random.Generator,
    *,
    balance_cycle: int = 30,
) -> bool:
    """Choose sniff order once so retries cannot change episode semantics."""
    fraction = float(np.clip(target_first_frac, 0.0, 1.0))
    if fraction <= 0.0:
        return False
    if fraction >= 1.0:
        return True
    if np.isclose(fraction, 0.5):
        index = int(plan_index)
        cycle = max(1, int(balance_cycle))
        # Alternate globally during the first cycle, then flip every pair's
        # order on each repetition. With 30 ordered pairs this gives every
        # pair equal target-first and target-second coverage over two cycles.
        return (index + index // cycle) % 2 == 0
    return bool(rng.random() < fraction)


def _sample_paired_local_ppm(
    rng: np.random.Generator,
    cup_ids: list[str],
    *,
    local_ppm_min: float,
    local_ppm_max: float,
    ratio_min: float,
    ratio_max: float,
) -> tuple[dict[str, float], float, float]:
    """Sample C_A=C_base*r and C_B=C_base/r without favoring either cup."""
    if len(cup_ids) != 2:
        raise ValueError("paired source calibration requires exactly two cups")
    ratio = float(np.exp(rng.uniform(np.log(ratio_min), np.log(ratio_max))))
    ratio_extent = max(ratio, 1.0 / ratio)
    base_min = float(local_ppm_min) * ratio_extent
    base_max = float(local_ppm_max) / ratio_extent
    if base_max < base_min:
        raise ValueError(
            "local ppm range is too narrow for the requested source-ratio range"
        )
    base = float(np.exp(rng.uniform(np.log(base_min), np.log(base_max))))
    concentrations = {
        str(cup_ids[0]): base * ratio,
        str(cup_ids[1]): base / ratio,
    }
    return concentrations, base, ratio


def _append_manifest_row(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", newline="", encoding="utf-8") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            f.seek(0)
            empty = f.read(1) == ""
            f.seek(0, os.SEEK_END)
            writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
            if empty:
                writer.writeheader()
            writer.writerow(row)
            f.flush()
            os.fsync(f.fileno())
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def _load_manifest(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    rows = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if not row.get("episode_id") or not Path(row.get("episode_path", "")).is_dir():
                continue
            # Drop restkey leftovers (None) from older/mismatched headers and
            # keep only known columns so DictWriter rewrite cannot fail.
            clean = {k: (row.get(k) or "") for k in MANIFEST_FIELDS}
            if not clean.get("worker_id"):
                clean["worker_id"] = "0"
            rows.append(clean)
    return rows


def _write_manifest(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".csv.tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in MANIFEST_FIELDS})
    tmp.replace(path)


def sample_episode_liquids(
    rng: np.random.Generator,
    voc_pool: list[str],
    *,
    num_cups: int,
    target_present: bool,
    instructed: "str | None" = None,
) -> tuple[list[str], str]:
    """Uniformly sample table liquids + instruction from ``voc_pool``."""
    pool = list(dict.fromkeys(str(x) for x in voc_pool))
    if len(pool) < num_cups:
        raise ValueError(
            f"voc_pool has {len(pool)} entries but num_cups={num_cups}"
        )
    if target_present and instructed is not None and instructed not in pool:
        raise ValueError(
            f"pinned instructed {instructed!r} not in voc_pool {pool!r}"
        )
    if (not target_present) and len(pool) <= num_cups and instructed is None:
        raise ValueError(
            "absent scenes need voc_pool larger than num_cups so an off-table "
            f"instruction exists (pool={len(pool)}, num_cups={num_cups})"
        )

    if target_present:
        if instructed is not None:
            others = [v for v in pool if v != instructed]
            need = num_cups - 1
            if need > len(others):
                raise ValueError("not enough VOCs left to fill the table")
            rest = (
                [str(x) for x in rng.choice(others, size=need, replace=False)]
                if need > 0
                else []
            )
            liquids = [instructed] + rest
            rng.shuffle(liquids)
            return liquids, instructed
        liquids = [str(x) for x in rng.choice(pool, size=num_cups, replace=False)]
        return liquids, str(rng.choice(liquids))

    if instructed is not None:
        others = [v for v in pool if v != instructed]
        if len(others) < num_cups:
            raise ValueError(
                f"cannot build absent scene for {instructed!r}: only "
                f"{len(others)} other VOCs in pool"
            )
        liquids = [str(x) for x in rng.choice(others, size=num_cups, replace=False)]
        return liquids, instructed

    liquids = [str(x) for x in rng.choice(pool, size=num_cups, replace=False)]
    off_table = [v for v in pool if v not in liquids]
    if not off_table:
        raise ValueError("absent scene needs at least one VOC left off the table")
    return liquids, str(rng.choice(off_table))


def _collect_one(payload: dict) -> "dict | None":
    """Run one planned episode (serial or ProcessPool worker)."""
    os.environ.setdefault("MUJOCO_GL", "egl")
    worker_id = int(payload.get("worker_id", 0))
    seed = int(payload["seed"])
    attempt_offset = int(payload.get("attempt_offset", 0))
    liquids = list(payload["liquids"])
    instructed = str(payload["instructed"])
    out_dir = Path(payload["out_dir"])
    raw_dir = out_dir / "raw" / "scene_two_cup"
    raw_dir.mkdir(parents=True, exist_ok=True)
    motion = dict(payload.get("motion") or {})
    max_attempts = int(payload.get("max_attempts", 15))
    tag = f"w{worker_id}"

    scene_id = _safe_scene_id(
        "odorplace",
        f"w{worker_id}",
        *liquids,
        "plate",
        payload.get("plan_index", seed),
    )
    server_log_dir = _REPO_ROOT / ".roslog" / f"collect_odor_place_w{worker_id}"

    for attempt in range(max_attempts):
        t0 = time.time()
        try:
            cfg = odor_config_from_args(
                sniff_duration_s=float(payload["sample_hold_s"]),
                filter_duration_s=(
                    None
                    if payload.get("filter_hold_s") is None
                    else float(payload["filter_hold_s"])
                ),
                mox_models=payload.get("mox_models"),
                voc_pool=payload.get("voc_pool") or liquids,
            )
            with ScriptedOdorPlaceCollector(
                out_dir=raw_dir,
                liquids=liquids,
                instructed_liquid=instructed,
                seed=seed + attempt_offset + attempt,
                record_frames=True,
                has_renderer=False,
                odor_mode=str(payload["odor_mode"]),
                odor_config=cfg,
                target_first_frac=float(
                    payload.get("target_first_frac", DEFAULT_TARGET_FIRST_FRAC)
                ),
                verbose=not bool(payload.get("quiet", False)),
                kill_stale_gaden=bool(payload.get("kill_stale_gaden", True)),
                scene_id=scene_id,
                server_log_dir=server_log_dir,
                worker_id=worker_id,
                transition_mode=str(payload["transition_mode"]),
                desired_local_ppm_by_object=payload["desired_local_ppm_by_object"],
                source_nominal_ppm=payload["source_nominal_ppm"],
                source_base_local_ppm=payload["source_base_local_ppm"],
                source_pair_ratio=payload["source_pair_ratio"],
                decision_window_s=float(payload["decision_window_s"]),
                decision_min_total_ppm=float(payload["decision_min_total_ppm"]),
                decision_dominance_threshold=float(
                    payload["decision_dominance_threshold"]
                ),
                decision_min_pass_fraction=float(
                    payload["decision_min_pass_fraction"]
                ),
                emission_scale=float(payload["emission_scale"]),
                concentration_exponent=float(payload["concentration_exponent"]),
                filaments_per_second=int(payload["filaments_per_second"]),
                wind_regime=str(payload["wind_regime"]),
                buoyancy_tau_s=float(payload["buoyancy_tau_s"]),
                original_terminal_velocity_enabled=bool(
                    payload["original_terminal_velocity_enabled"]
                ),
                **motion,
            ) as collector:
                paths = collector.collect(1, max_attempts=1)
        except Exception as exc:  # noqa: BLE001
            print(
                f"[collect_odor_place_oft:{tag}] seed={seed + attempt} error: {exc}",
                flush=True,
            )
            paths = []
        dt = time.time() - t0
        if paths:
            row = _manifest_row(
                paths[0],
                liquids=liquids,
                instructed=instructed,
                seed=seed + attempt_offset + attempt,
                worker_id=worker_id,
                plan_index=payload.get("plan_index"),
                planned_target_first=payload.get("planned_target_first"),
                attempt_index=attempt_offset + attempt,
            )
            _append_manifest_row(out_dir / "manifest.csv", row)
            print(
                f"[collect_odor_place_oft:{tag}] ok kind={payload['kind']} "
                f"liquids={liquids} instructed={instructed!r} "
                f"({dt:.1f}s) → {paths[0].name}",
                flush=True,
            )
            return row
        print(
            f"[collect_odor_place_oft:{tag}] fail kind={payload['kind']} "
                f"attempt={attempt_offset + attempt + 1} "
                f"(pass {attempt + 1}/{max_attempts}) ({dt:.1f}s)",
            flush=True,
        )
    return None


def _worker_run(payload: dict) -> "dict | None":
    """ProcessPool entry: isolate ROS domain, then collect one episode."""
    domain = int(payload.get("ros_domain_id", _DEFAULT_ROS_DOMAIN_BASE))
    os.environ["ROS_DOMAIN_ID"] = str(domain)
    os.environ.setdefault("MUJOCO_GL", "egl")
    log_root = _REPO_ROOT / ".roslog" / f"collect_odor_place_w{payload.get('worker_id', 0)}"
    log_root.mkdir(parents=True, exist_ok=True)
    return _collect_one(payload)


def collect_batch(
    *,
    out_dir: Path,
    num_demos: int,
    voc_pool: list[str],
    num_cups: int,
    instructed: "str | None",
    absent_frac: float,
    base_seed: int,
    max_attempts_per_ep: int,
    quiet: bool,
    sample_hold_s: float,
    filter_hold_s: "float | None",
    target_first_frac: float,
    odor_mode: str,
    motion: dict,
    num_workers: int = 1,
    ros_domain_base: int = _DEFAULT_ROS_DOMAIN_BASE,
    mox_models: "list[str] | None" = None,
    transition_mode: str = "direct",
    local_ppm_min: float = 5.0,
    local_ppm_max: float = 100.0,
    local_ppm_calibration: str | Path = DEFAULT_LOCAL_PPM_CALIBRATION,
    emission_scale: float = 0.06,
    concentration_exponent: float = 0.5,
    filaments_per_second: int = 15,
    wind_regime: str = "zero",
    original_terminal_velocity_enabled: bool = False,
    buoyancy_tau_s: float = 0.0,
    source_ratio_min: float = 0.8,
    source_ratio_max: float = 1.25,
    decision_window_s: float = DEFAULT_DECISION_WINDOW_S,
    decision_min_total_ppm: float = DEFAULT_DECISION_MIN_TOTAL_PPM,
    decision_dominance_threshold: float = DEFAULT_DECISION_DOMINANCE,
    decision_min_pass_fraction: float = DEFAULT_DECISION_MIN_PASS_FRACTION,
    attempt_offset: int = 0,
) -> list[dict]:
    """Collect ``num_demos`` episodes with uniform VOC draws from ``voc_pool``."""
    if num_cups != 2:
        raise ValueError("deployment-consistent OVLA collection requires exactly two cups")
    if local_ppm_min <= 0 or local_ppm_max < local_ppm_min:
        raise ValueError("require 0 < local_ppm_min <= local_ppm_max")
    if wind_regime != "zero":
        raise ValueError("main OVLA collection requires wind_regime='zero'")
    if buoyancy_tau_s != 0 or original_terminal_velocity_enabled:
        raise ValueError(
            "main OVLA collection requires both terminal and experimental buoyancy disabled"
        )
    if source_ratio_min <= 0 or source_ratio_max < source_ratio_min:
        raise ValueError("require 0 < source_ratio_min <= source_ratio_max")
    if not np.isclose(source_ratio_min * source_ratio_max, 1.0, rtol=1e-6):
        raise ValueError(
            "source ratio bounds must be reciprocal for a symmetric log-uniform distribution"
        )
    if int(attempt_offset) < 0:
        raise ValueError("attempt_offset must be >= 0")
    out_dir = Path(out_dir)
    (out_dir / "raw" / "scene_two_cup").mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.csv"
    rows = _load_manifest(manifest_path)
    # Older manifests predate explicit plan IDs. Preserve their rows and map
    # them sequentially once; all new collections retain exact IDs so a
    # parallel/interrupted run can resume the missing plans rather than merely
    # appending a potentially duplicated prefix.
    upgraded_manifest = False
    for fallback_index, row in enumerate(rows):
        if row.get("plan_index", "") == "":
            row["plan_index"] = str(fallback_index)
            upgraded_manifest = True
    if upgraded_manifest:
        _write_manifest(manifest_path, rows)
    completed_plan_indices = {
        int(row["plan_index"])
        for row in rows
        if str(row.get("plan_index", "")).strip() != ""
    }
    requested_plan_indices = list(range(int(num_demos)))
    pending_plan_indices = [
        index for index in requested_plan_indices
        if index not in completed_plan_indices
    ]
    need = len(pending_plan_indices)
    num_workers = max(1, int(num_workers))
    print(
        f"[collect_odor_place_oft] resumed {len(rows)} rows; need {need} more "
        f"(voc_pool={voc_pool}, num_cups={num_cups}, absent_frac={absent_frac}, "
        f"num_workers={num_workers})"
    )
    if need <= 0:
        return rows

    n_absent = int(round(int(num_demos) * float(absent_frac)))
    n_present = max(0, int(num_demos) - n_absent)
    calibration = _load_local_ppm_calibration(local_ppm_calibration)
    ordered_pairs = [(a, b) for a in voc_pool for b in voc_pool if a != b]

    payloads = []
    for i, global_plan_index in enumerate(pending_plan_indices):
        kind = "present" if global_plan_index < n_present else "absent"
        rng = np.random.default_rng(int(base_seed) + global_plan_index)
        if kind == "present" and instructed is None and num_cups == 2:
            ep_instructed, distractor = ordered_pairs[global_plan_index % len(ordered_pairs)]
            ep_liquids = [ep_instructed, distractor]
            rng.shuffle(ep_liquids)
        else:
            ep_liquids, ep_instructed = sample_episode_liquids(
                rng, voc_pool, num_cups=num_cups,
                target_present=(kind == "present"), instructed=instructed,
            )
        desired_local, source_base, source_ratio = _sample_paired_local_ppm(
            rng,
            ep_liquids,
            local_ppm_min=local_ppm_min,
            local_ppm_max=local_ppm_max,
            ratio_min=source_ratio_min,
            ratio_max=source_ratio_max,
        )
        nominal_source = {
            gas: _local_to_nominal(value, calibration)
            for gas, value in desired_local.items()
        }
        worker_id = i % num_workers
        planned_target_first = (
            _plan_target_first(
                global_plan_index,
                target_first_frac,
                rng,
                balance_cycle=len(ordered_pairs),
            )
            if kind == "present"
            else None
        )
        payloads.append(
            {
                "out_dir": str(out_dir),
                "kind": kind,
                "liquids": ep_liquids,
                "instructed": ep_instructed,
                "seed": int(base_seed) + global_plan_index * 1000,
                "plan_index": global_plan_index,
                "max_attempts": max_attempts_per_ep,
                "attempt_offset": int(attempt_offset),
                "quiet": quiet,
                "sample_hold_s": sample_hold_s,
                "filter_hold_s": filter_hold_s,
                # Force the already-planned order on every retry. Previously
                # seed changes could turn a failed target-second episode into
                # a saved target-first episode.
                "target_first_frac": (
                    float(planned_target_first)
                    if planned_target_first is not None
                    else target_first_frac
                ),
                "planned_target_first": planned_target_first,
                "odor_mode": odor_mode,
                "motion": motion,
                "mox_models": list(mox_models) if mox_models is not None else None,
                "voc_pool": list(voc_pool),
                "worker_id": worker_id,
                "ros_domain_id": int(ros_domain_base) + worker_id,
                "kill_stale_gaden": num_workers == 1,
                "transition_mode": transition_mode,
                "desired_local_ppm_by_object": desired_local,
                "source_nominal_ppm": nominal_source,
                "source_base_local_ppm": source_base,
                "source_pair_ratio": source_ratio,
                "decision_window_s": decision_window_s,
                "decision_min_total_ppm": decision_min_total_ppm,
                "decision_dominance_threshold": decision_dominance_threshold,
                "decision_min_pass_fraction": decision_min_pass_fraction,
                "emission_scale": emission_scale,
                "concentration_exponent": concentration_exponent,
                "filaments_per_second": filaments_per_second,
                "wind_regime": wind_regime,
                "original_terminal_velocity_enabled": original_terminal_velocity_enabled,
                "buoyancy_tau_s": buoyancy_tau_s,
            }
        )

    saved_rows = list(rows)
    ok = 0
    fail = 0

    if num_workers == 1:
        for i, payload in enumerate(payloads):
            print(
                f"[collect_odor_place_oft] ({i + 1}/{len(payloads)}) "
                f"{payload['kind']} {payload['liquids']} → {payload['instructed']!r}",
                flush=True,
            )
            row = _collect_one(payload)
            if row is None:
                fail += 1
            else:
                ok += 1
                saved_rows.append(row)
    else:
        print(
            f"[collect_odor_place_oft] launching {num_workers} workers on "
            f"{len(payloads)} episodes (ROS_DOMAIN_ID "
            f"{ros_domain_base}..{ros_domain_base + num_workers - 1}, "
            f"MUJOCO_GL=egl, kill_stale_gaden=False)",
            flush=True,
        )
        os.environ.setdefault("MUJOCO_GL", "egl")
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=num_workers, mp_context=ctx
        ) as pool:
            futures = {pool.submit(_worker_run, p): p for p in payloads}
            for fut in as_completed(futures):
                p = futures[fut]
                try:
                    row = fut.result()
                except Exception as exc:  # noqa: BLE001
                    fail += 1
                    print(
                        f"[collect_odor_place_oft] worker error "
                        f"seed={p['seed']}: {exc}",
                        flush=True,
                    )
                    continue
                if row is None:
                    fail += 1
                else:
                    ok += 1
                    saved_rows.append(row)
                print(
                    f"[collect_odor_place_oft] progress "
                    f"ok={ok} fail={fail} done={ok + fail}/{len(payloads)}",
                    flush=True,
                )

        # Reload + dedupe after concurrent appends.
        saved_rows = _load_manifest(manifest_path)
        seen = set()
        deduped = []
        for r in saved_rows:
            if r["episode_path"] in seen:
                continue
            seen.add(r["episode_path"])
            deduped.append(r)
        saved_rows = deduped
        _write_manifest(manifest_path, saved_rows)

    by_present = Counter(str(r.get("target_present")) for r in saved_rows)
    by_instr = Counter(r.get("instructed_liquid") for r in saved_rows)
    print(
        f"[collect_odor_place_oft] summary total={len(saved_rows)} "
        f"ok={ok} fail={fail} by_present={dict(by_present)} "
        f"by_instructed={dict(by_instr)}"
    )
    print(f"[collect_odor_place_oft] manifest → {manifest_path}")
    return saved_rows


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Batch odor pick-place collection")
    p.add_argument("--out-dir", type=str, default=DEFAULT_OUT_DIR)
    p.add_argument("--num-demos", type=int, default=50)
    p.add_argument(
        "--voc-pool",
        nargs="+",
        default=list(DEFAULT_VOC_POOL),
        help=(
            "VOC catalog names to sample from uniformly per episode "
            f"(default: {' '.join(DEFAULT_VOC_POOL)})"
        ),
    )
    p.add_argument(
        "--liquids",
        nargs="+",
        default=None,
        help="Alias for --voc-pool (deprecated name; same uniform sampling)",
    )
    p.add_argument(
        "--num-cups",
        type=int,
        default=DEFAULT_NUM_CUPS,
        help=f"Cups spawned per scene (default {DEFAULT_NUM_CUPS})",
    )
    p.add_argument(
        "--instructed",
        default=None,
        help=(
            "Pin the language-instruction liquid. Default: sample uniformly "
            "(among on-table VOCs for present scenes, off-table for absent)."
        ),
    )
    p.add_argument(
        "--absent-frac",
        type=float,
        default=0.0,
        help="Fraction of demos where instructed liquid is not on the table",
    )
    p.add_argument("--base-seed", type=int, default=0)
    p.add_argument("--max-attempts-per-ep", type=int, default=15)
    p.add_argument(
        "--attempt-offset",
        type=int,
        default=0,
        help=(
            "continue deterministic retry seeds after an earlier exhausted pass; "
            "for example, use 15 after a pass with 15 attempts per episode"
        ),
    )
    p.add_argument(
        "--sniff-duration",
        type=float,
        default=None,
        help=(
            f"Seconds hovering above a cup while SAMPLE-ing "
            f"(default {DEFAULT_SAMPLE_HOLD_S})"
        ),
    )
    p.add_argument(
        "--sample-hold-s",
        type=float,
        default=None,
        help="Legacy alias for --sniff-duration",
    )
    p.add_argument(
        "--filter-hold-s",
        type=float,
        default=None,
        help=(
            "FILTER nose-rest seconds after a wrong cup "
            f"(default {FILTER_DURATION_FACTOR}× sniff)"
        ),
    )
    p.add_argument(
        "--mox-models",
        nargs="+",
        default=None,
        help=(
            "MOX sensor ids in encoder order "
            f"(default: {' '.join(DEFAULT_ODOR_CONFIG.mox_models)})"
        ),
    )
    p.add_argument(
        "--target-first-frac",
        type=float,
        default=DEFAULT_TARGET_FIRST_FRAC,
        help=(
            "P(sniff instructed cup first) when it is on the table "
            f"(default {DEFAULT_TARGET_FIRST_FRAC}); remainder sniff a "
            "distractor first then transition directly to the target."
        ),
    )
    p.add_argument(
        "--odor-mode",
        default="continuous",
        choices=("discrete", "continuous", "none"),
    )
    p.add_argument("--transition-mode", default="direct", choices=("direct", "physical_filter"))
    p.add_argument("--local-ppm-min", type=float, default=5.0)
    p.add_argument("--local-ppm-max", type=float, default=100.0)
    p.add_argument("--source-ratio-min", type=float, default=0.8)
    p.add_argument("--source-ratio-max", type=float, default=1.25)
    p.add_argument("--decision-window-s", type=float, default=DEFAULT_DECISION_WINDOW_S)
    p.add_argument(
        "--decision-min-total-ppm", type=float, default=DEFAULT_DECISION_MIN_TOTAL_PPM
    )
    p.add_argument(
        "--decision-dominance-threshold", type=float, default=DEFAULT_DECISION_DOMINANCE
    )
    p.add_argument(
        "--decision-min-pass-fraction",
        type=float,
        default=DEFAULT_DECISION_MIN_PASS_FRACTION,
    )
    p.add_argument("--local-ppm-calibration", default=str(DEFAULT_LOCAL_PPM_CALIBRATION))
    p.add_argument("--emission-scale", type=float, default=0.06)
    p.add_argument("--concentration-exponent", type=float, default=0.5)
    p.add_argument("--filaments-per-second", type=int, default=15)
    p.add_argument("--wind-regime", choices=("zero",), default="zero")
    p.add_argument("--buoyancy-tau-s", type=float, default=0.0)
    p.add_argument("--original-terminal-velocity", action="store_true")
    p.add_argument(
        "--num-workers",
        type=int,
        default=DEFAULT_NUM_WORKERS,
        help=(
            "Parallel MuJoCo + GADEN worker processes (default 1). Each worker "
            "uses its own ROS_DOMAIN_ID. Prefer 2–4 depending on CPU/RAM."
        ),
    )
    p.add_argument(
        "--ros-domain-base",
        type=int,
        default=_DEFAULT_ROS_DOMAIN_BASE,
        help=(
            f"Base ROS_DOMAIN_ID for workers (default {_DEFAULT_ROS_DOMAIN_BASE}); "
            "worker i uses base+i."
        ),
    )
    p.add_argument("--quiet", action="store_true")
    add_motion_shaping_args(p)
    p.set_defaults(
        action_noise=DEFAULT_OVLA_ACTION_NOISE,
        action_noise_floor=DEFAULT_OVLA_ACTION_NOISE_FLOOR,
    )
    return p


def main(argv=None) -> int:
    args = build_argparser().parse_args(argv)
    if int(args.num_workers) <= 0:
        raise SystemExit("--num-workers must be > 0")
    if int(args.num_workers) > 1:
        os.environ.setdefault("MUJOCO_GL", "egl")
    voc_pool = list(args.liquids) if args.liquids is not None else list(args.voc_pool)
    cfg = odor_config_from_args(
        sniff_duration_s=args.sniff_duration,
        sample_hold_s=args.sample_hold_s,
        filter_duration_s=args.filter_hold_s,
        mox_models=args.mox_models,
        voc_pool=voc_pool,
    )
    rows = collect_batch(
        out_dir=Path(args.out_dir),
        num_demos=int(args.num_demos),
        voc_pool=voc_pool,
        num_cups=int(args.num_cups),
        instructed=None if args.instructed in (None, "") else str(args.instructed),
        absent_frac=float(args.absent_frac),
        base_seed=int(args.base_seed),
        max_attempts_per_ep=int(args.max_attempts_per_ep),
        quiet=bool(args.quiet),
        sample_hold_s=float(cfg.sniff_duration_s),
        filter_hold_s=float(cfg.filter_s),
        target_first_frac=float(args.target_first_frac),
        odor_mode=str(args.odor_mode),
        motion=motion_shaping_kwargs(args),
        num_workers=int(args.num_workers),
        ros_domain_base=int(args.ros_domain_base),
        mox_models=list(cfg.mox_models),
        transition_mode=str(args.transition_mode),
        local_ppm_min=float(args.local_ppm_min),
        local_ppm_max=float(args.local_ppm_max),
        local_ppm_calibration=args.local_ppm_calibration,
        emission_scale=float(args.emission_scale),
        concentration_exponent=float(args.concentration_exponent),
        filaments_per_second=int(args.filaments_per_second),
        wind_regime=str(args.wind_regime),
        original_terminal_velocity_enabled=bool(args.original_terminal_velocity),
        buoyancy_tau_s=float(args.buoyancy_tau_s),
        source_ratio_min=float(args.source_ratio_min),
        source_ratio_max=float(args.source_ratio_max),
        decision_window_s=float(args.decision_window_s),
        decision_min_total_ppm=float(args.decision_min_total_ppm),
        decision_dominance_threshold=float(args.decision_dominance_threshold),
        decision_min_pass_fraction=float(args.decision_min_pass_fraction),
        attempt_offset=int(args.attempt_offset),
    )
    return 0 if len(rows) >= int(args.num_demos) else 1


if __name__ == "__main__":
    raise SystemExit(main())

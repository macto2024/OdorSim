"""Batch collection for the odor-conditioned ClassifyLiquidPlace task.

Each episode draws cups uniformly from a VOC pool (default 6 pure VOCs at
strength 1.0). Default scene: 2 cups on the table. Scripted policy sniffs each
cup (SAMPLE 10 s), filters between cups (FILTER 15 s aside), then either
pick-places the instructed liquid or returns home if it is absent.

Parallelism (each worker gets its own ROS_DOMAIN_ID + GADEN server)::

    python -m odor_sim.bridge.collect_odor_place_oft \\
        --out-dir /root/Odorbot/datasets/odor_place_oft \\
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
    DEFAULT_FILTER_HOLD_S,
    DEFAULT_SAMPLE_HOLD_S,
    DEFAULT_TARGET_FIRST_FRAC,
    ScriptedOdorPlaceCollector,
)
from odor_sim.bridge.scripted_place import (
    add_motion_shaping_args,
    motion_shaping_kwargs,
)

DEFAULT_OUT_DIR = "/root/Odorbot/datasets/odor_place_oft"
_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_ROS_DOMAIN_BASE = 50
DEFAULT_NUM_WORKERS = 1

# Six pure VOC classes (strength 1.0) — uniform draw per scene.
DEFAULT_VOC_POOL = (
    "ethanol1.0",
    "propanol1.0",
    "acetone1.0",
    "methane1.0",
    "hydrogen1.0",
    "chlorine1.0",
)
DEFAULT_NUM_CUPS = 2

MANIFEST_FIELDS = [
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
    ep_dir: Path, *, liquids, instructed, seed: int, worker_id: int = 0
) -> dict:
    meta = _read_meta(ep_dir)
    return {
        "episode_id": ep_dir.name,
        "episode_path": str(ep_dir.resolve()),
        "liquids": ";".join(liquids),
        "instructed_liquid": instructed,
        "target_present": bool(meta.get("target_present", instructed in liquids)),
        "instruction": meta.get("instruction", ""),
        "num_steps": int(meta.get("num_steps", 0)),
        "seed": int(seed),
        "success": True,
        "worker_id": int(worker_id),
    }


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
            with ScriptedOdorPlaceCollector(
                out_dir=raw_dir,
                liquids=liquids,
                instructed_liquid=instructed,
                seed=seed + attempt,
                record_frames=True,
                has_renderer=False,
                odor_mode=str(payload["odor_mode"]),
                sample_hold_s=float(payload["sample_hold_s"]),
                filter_hold_s=float(payload["filter_hold_s"]),
                target_first_frac=float(
                    payload.get("target_first_frac", DEFAULT_TARGET_FIRST_FRAC)
                ),
                verbose=not bool(payload.get("quiet", False)),
                kill_stale_gaden=bool(payload.get("kill_stale_gaden", True)),
                scene_id=scene_id,
                server_log_dir=server_log_dir,
                worker_id=worker_id,
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
                seed=seed + attempt,
                worker_id=worker_id,
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
            f"attempt={attempt + 1}/{max_attempts} ({dt:.1f}s)",
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
    filter_hold_s: float,
    target_first_frac: float,
    odor_mode: str,
    motion: dict,
    num_workers: int = 1,
    ros_domain_base: int = _DEFAULT_ROS_DOMAIN_BASE,
) -> list[dict]:
    """Collect ``num_demos`` episodes with uniform VOC draws from ``voc_pool``."""
    out_dir = Path(out_dir)
    (out_dir / "raw" / "scene_two_cup").mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.csv"
    rows = _load_manifest(manifest_path)
    need = max(0, int(num_demos) - len(rows))
    num_workers = max(1, int(num_workers))
    print(
        f"[collect_odor_place_oft] resumed {len(rows)} rows; need {need} more "
        f"(voc_pool={voc_pool}, num_cups={num_cups}, absent_frac={absent_frac}, "
        f"num_workers={num_workers})"
    )
    if need <= 0:
        return rows

    n_absent = int(round(need * float(absent_frac)))
    n_present = max(0, need - n_absent)
    plan = [("present", i) for i in range(n_present)] + [
        ("absent", i) for i in range(n_absent)
    ]
    rng = np.random.default_rng(int(base_seed) + len(rows))

    payloads = []
    for i, (kind, _) in enumerate(plan):
        ep_liquids, ep_instructed = sample_episode_liquids(
            rng,
            voc_pool,
            num_cups=num_cups,
            target_present=(kind == "present"),
            instructed=instructed,
        )
        worker_id = i % num_workers
        payloads.append(
            {
                "out_dir": str(out_dir),
                "kind": kind,
                "liquids": ep_liquids,
                "instructed": ep_instructed,
                "seed": int(base_seed) + len(rows) + i * 1000,
                "plan_index": i,
                "max_attempts": max_attempts_per_ep,
                "quiet": quiet,
                "sample_hold_s": sample_hold_s,
                "filter_hold_s": filter_hold_s,
                "target_first_frac": target_first_frac,
                "odor_mode": odor_mode,
                "motion": motion,
                "worker_id": worker_id,
                "ros_domain_id": int(ros_domain_base) + worker_id,
                "kill_stale_gaden": num_workers == 1,
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
        default=0.2,
        help="Fraction of demos where instructed liquid is not on the table",
    )
    p.add_argument("--base-seed", type=int, default=0)
    p.add_argument("--max-attempts-per-ep", type=int, default=15)
    p.add_argument("--sample-hold-s", type=float, default=DEFAULT_SAMPLE_HOLD_S)
    p.add_argument("--filter-hold-s", type=float, default=DEFAULT_FILTER_HOLD_S)
    p.add_argument(
        "--target-first-frac",
        type=float,
        default=DEFAULT_TARGET_FIRST_FRAC,
        help=(
            "P(sniff instructed cup first) when it is on the table "
            f"(default {DEFAULT_TARGET_FIRST_FRAC}); remainder sniff a "
            "distractor first then FILTER then target."
        ),
    )
    p.add_argument(
        "--odor-mode",
        default="discrete",
        choices=("discrete", "continuous", "none"),
    )
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
    return p


def main(argv=None) -> int:
    args = build_argparser().parse_args(argv)
    if int(args.num_workers) <= 0:
        raise SystemExit("--num-workers must be > 0")
    if int(args.num_workers) > 1:
        os.environ.setdefault("MUJOCO_GL", "egl")
    voc_pool = list(args.liquids) if args.liquids is not None else list(args.voc_pool)
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
        sample_hold_s=float(args.sample_hold_s),
        filter_hold_s=float(args.filter_hold_s),
        target_first_frac=float(args.target_first_frac),
        odor_mode=str(args.odor_mode),
        motion=motion_shaping_kwargs(args),
        num_workers=int(args.num_workers),
        ros_domain_base=int(args.ros_domain_base),
    )
    return 0 if len(rows) >= int(args.num_demos) else 1


if __name__ == "__main__":
    raise SystemExit(main())

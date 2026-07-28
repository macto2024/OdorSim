"""Batch OdorPlace collection for OpenVLA-OFT fine-tuning.

Collects 450 successful scripted demos across three scenes (default)::

    Scene A (single cup): 50 × 3 mugs = 150
        30× generic  "pick up the cup and put it on the plate"
        20× named    "pick up the {nice_name} and put it on the plate"
    Scene B (multi + named): 150  (100 with 2 mugs, 50 with 3)
    Scene C (multi + any):   150  (100 with 2 mugs, 50 with 3)

Writes::

    <out-dir>/
      raw/scene_a_single|scene_b_named|scene_c_any/episode_*/
      manifest.csv
      README.md

Parallelism::

    # 6 MuJoCo workers (separate processes; default = min(6, CPU count))
    python -m odor_sim.bridge.collect_odorplace_oft \\
        --out-dir /root/Odorbot/datasets/odorplace_oft \\
        --num-per-scene 150 --num-workers 6 --quiet
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import json
import os
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from odor_sim.envs.odor_place import MUG_POOL, nice_mug_name

DEFAULT_OUT_DIR = "/root/Odorbot/datasets/odorplace_oft"
DEFAULT_NUM_WORKERS = min(6, max(1, (os.cpu_count() or 4)))

GENERIC_INSTRUCTION = "pick up the cup and put it on the plate"
ANY_INSTRUCTION = "pick up any of the cups and put it on the plate"

SCENE_DIRS = {
    "A_single": "scene_a_single",
    "B_named": "scene_b_named",
    "C_any": "scene_c_any",
}

MANIFEST_FIELDS = [
    "episode_id",
    "episode_path",
    "scene",
    "num_cups",
    "cups",
    "target_catalog",
    "instruction",
    "instruction_type",
    "num_steps",
    "seed",
    "success",
]


@dataclass(frozen=True)
class EpisodeSpec:
    scene: str  # A_single | B_named | C_any
    cups: tuple[str, ...]
    target_index: int
    instruction: str
    instruction_type: str  # generic | named | any
    seed: int

    @property
    def target_catalog(self) -> str:
        return self.cups[self.target_index]

    @property
    def out_subdir(self) -> str:
        return SCENE_DIRS[self.scene]

    def key(self) -> tuple:
        return (
            self.scene,
            self.instruction,
            ";".join(self.cups),
            self.target_catalog,
            int(self.seed),
        )


def named_instruction(catalog: str) -> str:
    return f"pick up the {nice_mug_name(catalog)} and put it on the plate"


def _scale_counts(n: int, base: int = 150) -> dict[str, int]:
    """Scale scene-A per-mug quotas and B/C 2-cup vs 3-cup mix with ``n``."""
    g = max(1, int(round(30 * n / base))) if n > 0 else 0
    named = max(0, int(round(20 * n / base))) if n > 0 else 0
    per_mug = max(1, n // 3) if n >= 3 else n
    if g + named != per_mug and per_mug > 0:
        named = max(0, per_mug - g)
        if g + named < per_mug:
            g = per_mug - named
    two = max(0, int(round(100 * n / base))) if n > 0 else 0
    three = max(0, n - two)
    return {
        "generic_per_mug": g,
        "named_per_mug": named,
        "two_cup": two,
        "three_cup": three,
    }


def build_specs(num_per_scene: int, *, base_seed: int = 0) -> list[EpisodeSpec]:
    """Build the full episode plan for scenes A/B/C."""
    rng = np.random.default_rng(base_seed)
    q = _scale_counts(int(num_per_scene))
    specs: list[EpisodeSpec] = []
    seed = int(base_seed)

    for mug in MUG_POOL:
        for _ in range(q["generic_per_mug"]):
            specs.append(
                EpisodeSpec(
                    scene="A_single",
                    cups=(mug,),
                    target_index=0,
                    instruction=GENERIC_INSTRUCTION,
                    instruction_type="generic",
                    seed=seed,
                )
            )
            seed += 1
        for _ in range(q["named_per_mug"]):
            specs.append(
                EpisodeSpec(
                    scene="A_single",
                    cups=(mug,),
                    target_index=0,
                    instruction=named_instruction(mug),
                    instruction_type="named",
                    seed=seed,
                )
            )
            seed += 1

    a_specs = [s for s in specs if s.scene == "A_single"]
    while len(a_specs) < num_per_scene:
        mug = str(rng.choice(MUG_POOL))
        a_specs.append(
            EpisodeSpec(
                scene="A_single",
                cups=(mug,),
                target_index=0,
                instruction=GENERIC_INSTRUCTION,
                instruction_type="generic",
                seed=seed,
            )
        )
        seed += 1
    a_specs = a_specs[:num_per_scene]
    specs = list(a_specs)

    def append_multi(scene: str, instruction_type: str, count_two: int, count_three: int):
        nonlocal seed
        for num_cups, count in ((2, count_two), (3, count_three)):
            for _ in range(count):
                cups = tuple(
                    str(x) for x in rng.choice(list(MUG_POOL), size=num_cups, replace=False)
                )
                target_index = int(rng.integers(0, num_cups))
                if instruction_type == "named":
                    instr = named_instruction(cups[target_index])
                else:
                    instr = ANY_INSTRUCTION
                specs.append(
                    EpisodeSpec(
                        scene=scene,
                        cups=cups,
                        target_index=target_index,
                        instruction=instr,
                        instruction_type=instruction_type,
                        seed=seed,
                    )
                )
                seed += 1

    append_multi("B_named", "named", q["two_cup"], q["three_cup"])
    append_multi("C_any", "any", q["two_cup"], q["three_cup"])
    return specs


def _read_meta(ep_dir: Path) -> dict:
    meta_path = ep_dir / "meta.json"
    if not meta_path.is_file():
        return {}
    return json.loads(meta_path.read_text(encoding="utf-8"))


def _manifest_row(spec: EpisodeSpec, ep_dir: Path) -> dict:
    meta = _read_meta(ep_dir)
    return {
        "episode_id": ep_dir.name,
        "episode_path": str(ep_dir.resolve()),
        "scene": spec.scene,
        "num_cups": len(spec.cups),
        "cups": ";".join(spec.cups),
        "target_catalog": meta.get("target_catalog", spec.target_catalog),
        "instruction": meta.get("instruction", spec.instruction),
        "instruction_type": spec.instruction_type,
        "num_steps": int(meta.get("num_steps", 0)),
        "seed": int(spec.seed),
        "success": True,
    }


def _load_manifest(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    rows = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if not row.get("episode_id"):
                continue
            if Path(row["episode_path"]).is_dir():
                rows.append(row)
    return rows


def _write_manifest(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".csv.tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    tmp.replace(path)


def _append_manifest_row(path: Path, row: dict) -> None:
    """Append one row under an exclusive flock (safe across workers)."""
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


def _print_summary(rows: list[dict]) -> None:
    print("\n[collect_odorplace_oft] === summary ===")
    print(f"  total successful: {len(rows)}")
    by_scene = Counter(r["scene"] for r in rows)
    by_itype = Counter(r["instruction_type"] for r in rows)
    by_target = Counter(r["target_catalog"] for r in rows)
    by_n = Counter(int(r["num_cups"]) for r in rows)
    print(f"  by scene: {dict(by_scene)}")
    print(f"  by instruction_type: {dict(by_itype)}")
    print(f"  by target_catalog: {dict(by_target)}")
    print(f"  by num_cups: {dict(by_n)}")


def collect_one(
    spec: EpisodeSpec,
    *,
    out_dir: Path,
    max_attempts: int,
    quiet: bool,
    robots: str,
    camera_size: int,
    max_episode_steps: int,
    worker_id: int = 0,
) -> "Path | None":
    """Try to collect one successful episode for ``spec`` (retry with new seeds)."""
    # Import inside worker so spawn children load robosuite only when needed.
    from odor_sim.bridge.scripted_place import ScriptedPlaceCollector

    scene_dir = out_dir / "raw" / spec.out_subdir
    scene_dir.mkdir(parents=True, exist_ok=True)
    # Offset retry seeds by worker so parallel retries don't collide.
    seed = int(spec.seed) + int(worker_id) * 1_000_000
    tag = f"w{worker_id}"
    for attempt in range(max_attempts):
        t0 = time.time()
        try:
            with ScriptedPlaceCollector(
                out_dir=scene_dir,
                cups=list(spec.cups),
                target_cup=int(spec.target_index),
                place_target="plate",
                instruction=spec.instruction,
                robots=robots,
                camera_size=camera_size,
                record_frames=True,
                has_renderer=False,
                seed=seed,
                max_episode_steps=max_episode_steps,
                verbose=not quiet,
            ) as collector:
                saved = collector.collect(1, max_attempts=1)
        except Exception as exc:  # noqa: BLE001
            print(
                f"[collect_odorplace_oft:{tag}] seed={seed} attempt={attempt + 1} "
                f"error: {exc}",
                flush=True,
            )
            saved = []
        dt = time.time() - t0
        if saved:
            print(
                f"[collect_odorplace_oft:{tag}] ok scene={spec.scene} "
                f"cups={list(spec.cups)} target={spec.target_catalog} "
                f"itype={spec.instruction_type} seed={seed} ({dt:.1f}s) "
                f"→ {saved[0].name}",
                flush=True,
            )
            return saved[0]
        print(
            f"[collect_odorplace_oft:{tag}] fail scene={spec.scene} seed={seed} "
            f"attempt={attempt + 1}/{max_attempts} ({dt:.1f}s); retrying",
            flush=True,
        )
        seed += 10_000 + attempt
    return None


def _worker_run_spec(payload: dict) -> "dict | None":
    """ProcessPool entry: run one EpisodeSpec, append manifest, return row."""
    # Prefer EGL for parallel offscreen MuJoCo (set before mujoco import).
    os.environ.setdefault("MUJOCO_GL", "egl")

    spec = EpisodeSpec(
        scene=payload["scene"],
        cups=tuple(payload["cups"]),
        target_index=int(payload["target_index"]),
        instruction=payload["instruction"],
        instruction_type=payload["instruction_type"],
        seed=int(payload["seed"]),
    )
    out_dir = Path(payload["out_dir"])
    ep = collect_one(
        spec,
        out_dir=out_dir,
        max_attempts=int(payload["max_attempts"]),
        quiet=bool(payload["quiet"]),
        robots=payload["robots"],
        camera_size=int(payload["camera_size"]),
        max_episode_steps=int(payload["max_episode_steps"]),
        worker_id=int(payload["worker_id"]),
    )
    if ep is None:
        return None
    row = _manifest_row(spec, ep)
    _append_manifest_row(out_dir / "manifest.csv", row)
    return row


def run_collection(
    *,
    out_dir: Path,
    num_per_scene: int,
    base_seed: int,
    max_attempts_per_ep: int,
    quiet: bool,
    robots: str,
    camera_size: int,
    max_episode_steps: int,
    num_workers: int = 1,
    dry_run: bool = False,
) -> list[dict]:
    specs = build_specs(num_per_scene, base_seed=base_seed)
    num_workers = max(1, int(num_workers))
    print(
        f"[collect_odorplace_oft] planned {len(specs)} episodes "
        f"(num_per_scene={num_per_scene}, base_seed={base_seed}, "
        f"num_workers={num_workers})"
    )
    for scene, sub in SCENE_DIRS.items():
        n = sum(1 for s in specs if s.scene == scene)
        print(f"  {scene}: {n} → raw/{sub}/")

    if dry_run:
        for i, s in enumerate(specs[:12]):
            print(
                f"  [{i}] {s.scene} cups={list(s.cups)} target={s.target_index} "
                f"itype={s.instruction_type} instr={s.instruction!r} seed={s.seed}"
            )
        if len(specs) > 12:
            print(f"  ... ({len(specs) - 12} more)")
        return []

    out_dir.mkdir(parents=True, exist_ok=True)
    for sub in SCENE_DIRS.values():
        (out_dir / "raw" / sub).mkdir(parents=True, exist_ok=True)

    manifest_path = out_dir / "manifest.csv"
    rows = _load_manifest(manifest_path)
    # Deduplicate by episode_path (parallel appends / resume).
    seen_paths: set[str] = set()
    deduped: list[dict] = []
    for r in rows:
        if r["episode_path"] in seen_paths:
            continue
        seen_paths.add(r["episode_path"])
        deduped.append(r)
    rows = deduped
    _write_manifest(manifest_path, rows)
    print(f"[collect_odorplace_oft] resumed {len(rows)} rows from manifest")

    done_keys = {
        (r["scene"], r["instruction"], r["cups"], r["target_catalog"], int(r["seed"]))
        for r in rows
    }
    pending = [s for s in specs if s.key() not in done_keys]
    print(f"[collect_odorplace_oft] pending {len(pending)} / {len(specs)}")

    if not pending:
        _print_summary(rows)
        return rows

    if num_workers == 1:
        for i, spec in enumerate(pending):
            print(
                f"[collect_odorplace_oft] ({i + 1}/{len(pending)}) "
                f"{spec.scene} {list(spec.cups)} → {spec.instruction!r}",
                flush=True,
            )
            ep = collect_one(
                spec,
                out_dir=out_dir,
                max_attempts=max_attempts_per_ep,
                quiet=quiet,
                robots=robots,
                camera_size=camera_size,
                max_episode_steps=max_episode_steps,
                worker_id=0,
            )
            if ep is None:
                print(
                    f"[collect_odorplace_oft] GIVING UP on spec seed={spec.seed} "
                    f"after {max_attempts_per_ep} attempts",
                    flush=True,
                )
                continue
            row = _manifest_row(spec, ep)
            rows.append(row)
            _write_manifest(manifest_path, rows)
    else:
        payloads = []
        for i, spec in enumerate(pending):
            payloads.append(
                {
                    **asdict(spec),
                    "cups": list(spec.cups),
                    "out_dir": str(out_dir),
                    "max_attempts": max_attempts_per_ep,
                    "quiet": quiet,
                    "robots": robots,
                    "camera_size": camera_size,
                    "max_episode_steps": max_episode_steps,
                    "worker_id": i % num_workers,
                    "spec_index": i,
                }
            )
        # spawn avoids forking a process that already imported mujoco/OpenGL
        print(
            f"[collect_odorplace_oft] launching {num_workers} workers "
            f"on {len(payloads)} specs (MUJOCO_GL=egl)",
            flush=True,
        )
        import multiprocessing as mp

        ctx = mp.get_context("spawn")
        ok = 0
        fail = 0
        with ProcessPoolExecutor(
            max_workers=num_workers, mp_context=ctx
        ) as pool:
            futures = {pool.submit(_worker_run_spec, p): p for p in payloads}
            for fut in as_completed(futures):
                p = futures[fut]
                try:
                    row = fut.result()
                except Exception as exc:  # noqa: BLE001
                    fail += 1
                    print(
                        f"[collect_odorplace_oft] worker error "
                        f"seed={p['seed']}: {exc}",
                        flush=True,
                    )
                    continue
                if row is None:
                    fail += 1
                    print(
                        f"[collect_odorplace_oft] GIVING UP seed={p['seed']} "
                        f"after {max_attempts_per_ep} attempts",
                        flush=True,
                    )
                else:
                    ok += 1
                    rows.append(row)
                    print(
                        f"[collect_odorplace_oft] progress "
                        f"ok={ok} fail={fail} "
                        f"done={ok + fail}/{len(payloads)}",
                        flush=True,
                    )

        # Reload + dedupe manifest (workers appended concurrently).
        rows = _load_manifest(manifest_path)
        seen = set()
        deduped = []
        for r in rows:
            if r["episode_path"] in seen:
                continue
            seen.add(r["episode_path"])
            deduped.append(r)
        rows = deduped
        _write_manifest(manifest_path, rows)

    _print_summary(rows)
    print(f"[collect_odorplace_oft] manifest → {manifest_path}")
    return rows


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Batch OdorPlace collection for OpenVLA-OFT")
    p.add_argument("--out-dir", type=str, default=DEFAULT_OUT_DIR)
    p.add_argument(
        "--num-per-scene",
        type=int,
        default=150,
        help="Successful episodes per scene A/B/C (default 150 → 450 total)",
    )
    p.add_argument("--base-seed", type=int, default=0)
    p.add_argument(
        "--max-attempts-per-ep",
        type=int,
        default=25,
        help="Retries (new seeds) per planned episode before giving up",
    )
    p.add_argument(
        "--num-workers",
        type=int,
        default=DEFAULT_NUM_WORKERS,
        help=(
            "Parallel MuJoCo worker processes (default "
            f"{DEFAULT_NUM_WORKERS}). Use 1 for serial."
        ),
    )
    p.add_argument("--robots", default="Panda")
    p.add_argument("--camera-size", type=int, default=256)
    p.add_argument("--max-episode-steps", type=int, default=800)
    p.add_argument("--quiet", action="store_true")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the episode plan without running the sim",
    )
    return p


def main(argv=None) -> int:
    args = build_argparser().parse_args(argv)
    # Parent also prefers EGL when workers will spawn.
    if int(args.num_workers) > 1:
        os.environ.setdefault("MUJOCO_GL", "egl")
    rows = run_collection(
        out_dir=Path(args.out_dir),
        num_per_scene=int(args.num_per_scene),
        base_seed=int(args.base_seed),
        max_attempts_per_ep=int(args.max_attempts_per_ep),
        quiet=bool(args.quiet),
        robots=args.robots,
        camera_size=int(args.camera_size),
        max_episode_steps=int(args.max_episode_steps),
        num_workers=int(args.num_workers),
        dry_run=bool(args.dry_run),
    )
    if args.dry_run:
        return 0
    expected = 3 * int(args.num_per_scene)
    return 0 if len(rows) >= expected else 1


if __name__ == "__main__":
    # Required for ProcessPoolExecutor spawn on some platforms.
    raise SystemExit(main())

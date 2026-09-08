"""Quarantine OVLA episodes whose decision sniff pose is too far from its cup."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from datetime import datetime
from pathlib import Path


def _read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def _max_sniff_distance(ep_dir: Path) -> float:
    document = json.loads((ep_dir / "odor_events.json").read_text(encoding="utf-8"))
    distances = [
        float(event["sensor_object_distance"])
        for event in document.get("events", [])
    ]
    return max(distances, default=0.0)


def prepare(args: argparse.Namespace) -> dict:
    root = Path(args.dataset).resolve()
    manifest = root / "manifest.csv"
    if not manifest.is_file():
        raise FileNotFoundError(f"manifest not found: {manifest}")
    fieldnames, rows = _read_rows(manifest)

    affected: list[tuple[dict[str, str], float]] = []
    for row in rows:
        ep_dir = Path(row["episode_path"]).resolve()
        distance = _max_sniff_distance(ep_dir)
        if distance > float(args.max_sniff_distance_m):
            affected.append((row, distance))

    max_attempt = max(
        (int(row.get("attempt_index") or 0) + 1 for row in rows), default=0
    )
    attempt_block = max(1, int(args.attempt_block))
    recommended_offset = ((max_attempt + attempt_block - 1) // attempt_block) * attempt_block
    summary = {
        "dataset": str(root),
        "applied": bool(args.apply),
        "threshold_m": float(args.max_sniff_distance_m),
        "episode_count_before": len(rows),
        "affected_episode_count": len(affected),
        "affected_plan_indices": [int(row["plan_index"]) for row, _ in affected],
        "affected": [
            {
                "plan_index": int(row["plan_index"]),
                "episode": Path(row["episode_path"]).name,
                "max_sniff_distance_m": distance,
            }
            for row, distance in affected
        ],
        "recommended_attempt_offset": recommended_offset,
    }

    if not args.apply or not affected:
        print(json.dumps(summary, indent=2))
        return summary

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    quarantine = root / "quarantine" / f"sniff_distance_{stamp}"
    quarantine.mkdir(parents=True, exist_ok=False)
    backup = quarantine / "manifest.before_recollection.csv"
    shutil.copy2(manifest, backup)

    affected_paths = {str(Path(row["episode_path"]).resolve()) for row, _ in affected}
    for row, _ in affected:
        source = Path(row["episode_path"]).resolve()
        try:
            source.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"refusing to move episode outside dataset: {source}") from exc
        destination = quarantine / source.name
        if destination.exists():
            raise FileExistsError(f"quarantine destination already exists: {destination}")
        shutil.move(str(source), str(destination))

    retained = [
        row for row in rows if str(Path(row["episode_path"]).resolve()) not in affected_paths
    ]
    temporary = manifest.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(retained)
    temporary.replace(manifest)

    summary["episode_count_after"] = len(retained)
    summary["quarantine"] = str(quarantine)
    summary["manifest_backup"] = str(backup)
    (quarantine / "recollection.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return summary


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--max-sniff-distance-m", type=float, default=0.05)
    parser.add_argument(
        "--attempt-block",
        type=int,
        default=15,
        help="round the recommended retry offset up to this block size",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="move affected episodes and rewrite the manifest; default is dry-run",
    )
    return parser


def main(argv=None) -> int:
    prepare(build_argparser().parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

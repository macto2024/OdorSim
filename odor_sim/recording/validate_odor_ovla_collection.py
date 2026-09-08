"""Validate deployment-consistent two-cup OVLA odor demonstrations."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

from odor_sim.recording.build_rlds_odor_ovla import validate_deployment_protocol


def _quantiles(values: list[float]) -> dict[str, float]:
    data = np.asarray(values, dtype=float)
    if data.size == 0:
        return {}
    return {
        "min": float(np.min(data)),
        "median": float(np.median(data)),
        "p90": float(np.quantile(data, 0.90)),
        "p95": float(np.quantile(data, 0.95)),
        "max": float(np.max(data)),
    }


def validate(args: argparse.Namespace) -> dict:
    root = Path(args.dataset)
    manifest = root / "manifest.csv"
    if not manifest.is_file():
        raise FileNotFoundError(f"manifest not found: {manifest}")
    with manifest.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    failures: list[str] = []
    warnings: list[str] = []
    planned = Counter()
    actual = Counter()
    pairs = Counter()
    targets = Counter()
    event_patterns = Counter()
    dominance: list[float] = []
    presence: list[float] = []
    dominance_pass_fraction: list[float] = []
    sniff_distances: list[float] = []
    excessive_sniff_distance_episodes: set[str] = set()
    cup_spacing: list[float] = []
    voltages: list[np.ndarray] = []
    attempt_count = 0
    event_count = 0

    expected = int(args.expected_episodes)
    if len(rows) != expected:
        failures.append(f"episode count is {len(rows)}, expected {expected}")
    plan_ids = [int(row["plan_index"]) for row in rows if row.get("plan_index", "") != ""]
    if len(plan_ids) != len(rows) or len(set(plan_ids)) != len(rows):
        failures.append("plan_index values are missing or duplicated")

    for row in rows:
        ep_dir = Path(row["episode_path"])
        try:
            meta = json.loads((ep_dir / "meta.json").read_text(encoding="utf-8"))
            event_doc = json.loads(
                (ep_dir / "odor_events.json").read_text(encoding="utf-8")
            )
            arrays = np.load(ep_dir / "episode.npz")
            validate_deployment_protocol(meta, ep_dir)
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{ep_dir.name}: invalid episode: {exc}")
            continue

        events = event_doc.get("events", [])
        target = str(row["instructed_liquid"])
        liquids = str(row["liquids"]).split(";")
        distractors = [gas for gas in liquids if gas != target]
        if len(distractors) != 1:
            failures.append(f"{ep_dir.name}: expected one target and one distractor")
            continue
        pairs[(target, distractors[0])] += 1
        targets[target] += 1

        planned_first = str(row.get("planned_target_first", "")).lower() == "true"
        actual_first = bool(events and events[0].get("object_id") == target)
        planned[planned_first] += 1
        actual[actual_first] += 1
        if planned_first != actual_first:
            failures.append(f"{ep_dir.name}: planned/actual sniff order mismatch")
        pattern = tuple(str(event.get("decision")) for event in events)
        event_patterns[pattern] += 1
        expected_pattern = ("grasp",) if actual_first else ("reject", "grasp")
        if pattern != expected_pattern:
            failures.append(
                f"{ep_dir.name}: decision sequence {pattern}, expected {expected_pattern}"
            )
        event_count += len(events)
        attempt_count += int(row.get("attempt_index") or 0) + 1

        ppm = np.asarray(arrays["ppm"], dtype=float)
        voltage = np.asarray(arrays["enose_voltages"], dtype=float)
        sim_time = np.asarray(arrays["sim_time"], dtype=float)
        if ppm.shape[0] != voltage.shape[0] or ppm.shape[0] != sim_time.shape[0]:
            failures.append(f"{ep_dir.name}: stream lengths are not synchronized")
        if not np.isfinite(ppm).all() or not np.isfinite(voltage).all():
            failures.append(f"{ep_dir.name}: non-finite ppm or voltage")
        voltages.append(voltage)

        layout = meta["spawn_layout"]
        positions = [np.asarray(layout[gas]["pos"], dtype=float) for gas in liquids]
        cup_spacing.append(float(np.linalg.norm(positions[0][:2] - positions[1][:2])))

        desired = meta.get("desired_local_ppm_by_object") or {}
        base = float(meta.get("source_base_local_ppm"))
        ratio = float(meta.get("source_pair_ratio"))
        if not np.isclose(
            float(desired[liquids[0]]) * float(desired[liquids[1]]), base**2,
            rtol=1e-6,
        ):
            failures.append(f"{ep_dir.name}: paired source product invariant failed")
        if not float(args.source_ratio_min) <= ratio <= float(args.source_ratio_max):
            failures.append(f"{ep_dir.name}: source ratio {ratio:.4f} outside bounds")

        gas_types = list(meta["gas_types"])
        window_steps = max(1, int(round(args.window_s * float(meta["control_freq"]))))
        for event in events:
            gas_index = gas_types.index(str(event["object_id"]))
            end = int(event["end_step"])
            first = max(int(event["start_step"]), end - window_steps + 1)
            recent = ppm[first : end + 1]
            total = np.sum(recent, axis=1)
            intended = recent[:, gas_index]
            dom = np.divide(
                intended, total, out=np.zeros_like(intended), where=total > 1e-9
            )
            med_total = float(np.median(total))
            med_dom = float(np.median(dom))
            pass_fraction = float(np.mean(dom >= args.dominance_threshold))
            dominance.append(med_dom)
            presence.append(med_total)
            dominance_pass_fraction.append(pass_fraction)
            sniff_distance = float(event["sensor_object_distance"])
            sniff_distances.append(sniff_distance)
            if sniff_distance > float(args.max_sniff_distance_m):
                excessive_sniff_distance_episodes.add(ep_dir.name)
            if med_total < args.min_total_ppm:
                failures.append(f"{ep_dir.name}: decision presence {med_total:.3f} ppm < threshold")
            if med_dom < args.dominance_threshold:
                failures.append(f"{ep_dir.name}: decision dominance {med_dom:.3f} < threshold")
            if pass_fraction < args.min_pass_fraction:
                failures.append(
                    f"{ep_dir.name}: dominance pass fraction {pass_fraction:.3f} < threshold"
                )

        for camera in ("agentview", "wrist"):
            frame_count = len(list((ep_dir / "frames" / camera).glob("*.png")))
            if frame_count != len(sim_time):
                failures.append(
                    f"{ep_dir.name}: {camera} has {frame_count} frames for {len(sim_time)} steps"
                )

    if len(pairs) != int(args.expected_pairs):
        failures.append(f"ordered-pair coverage is {len(pairs)}, expected {args.expected_pairs}")
    if excessive_sniff_distance_episodes:
        failures.append(
            f"{len(excessive_sniff_distance_episodes)} episodes contain a decision "
            f"sniff distance > {args.max_sniff_distance_m:.3f} m"
        )
    if abs(actual[True] - actual[False]) > 1:
        failures.append(f"target-first/second imbalance: {dict(actual)}")
    high_fraction = float(np.mean(np.asarray(dominance) >= args.preferred_dominance)) if dominance else 0.0
    if high_fraction < args.preferred_fraction:
        warnings.append(
            f"only {high_fraction:.1%} of decisions reach dominance "
            f">={args.preferred_dominance:.2f}; preferred {args.preferred_fraction:.1%}"
        )
    success_rate = len(rows) / attempt_count if attempt_count else 0.0
    if success_rate < args.min_success_rate:
        failures.append(
            f"collection success rate {success_rate:.1%} < {args.min_success_rate:.1%}"
        )

    voltage = np.concatenate(voltages) if voltages else np.empty((0, 8))
    saturation_fraction = float(np.mean(voltage >= args.saturation_voltage)) if voltage.size else 0.0
    if saturation_fraction > 0:
        failures.append(f"MOX saturation fraction is {saturation_fraction:.6f}")

    result = {
        "valid": not failures,
        "failures": failures,
        "warnings": warnings,
        "episode_count": len(rows),
        "ordered_pair_count": len(pairs),
        "target_counts": dict(targets),
        "planned_target_first": {str(key): value for key, value in planned.items()},
        "actual_target_first": {str(key): value for key, value in actual.items()},
        "event_patterns": {"->".join(key): value for key, value in event_patterns.items()},
        "event_count": event_count,
        "collection_success_rate": success_rate,
        "dominance": _quantiles(dominance),
        "fraction_dominance_ge_0_70": float(np.mean(np.asarray(dominance) >= 0.70)) if dominance else 0.0,
        "fraction_dominance_ge_0_90": float(np.mean(np.asarray(dominance) >= 0.90)) if dominance else 0.0,
        "recent_total_ppm": _quantiles(presence),
        "dominance_pass_fraction": _quantiles(dominance_pass_fraction),
        "sniff_distance_m": _quantiles(sniff_distances),
        "excessive_sniff_distance_episode_count": len(
            excessive_sniff_distance_episodes
        ),
        "excessive_sniff_distance_episode_names": sorted(
            excessive_sniff_distance_episodes
        ),
        "cup_spacing_m": _quantiles(cup_spacing),
        "mox_voltage_max": float(np.max(voltage)) if voltage.size else None,
        "mox_saturation_fraction": saturation_fraction,
    }
    destination = Path(args.output) if args.output else root / "validation.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"[validate_odor_ovla] report -> {destination}")
    return result


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--expected-episodes", type=int, default=30)
    parser.add_argument("--expected-pairs", type=int, default=30)
    parser.add_argument("--window-s", type=float, default=1.0)
    parser.add_argument("--min-total-ppm", type=float, default=5.0)
    parser.add_argument("--dominance-threshold", type=float, default=0.70)
    parser.add_argument("--min-pass-fraction", type=float, default=0.80)
    parser.add_argument("--preferred-dominance", type=float, default=0.90)
    parser.add_argument("--preferred-fraction", type=float, default=0.70)
    parser.add_argument("--source-ratio-min", type=float, default=0.8)
    parser.add_argument("--source-ratio-max", type=float, default=1.25)
    parser.add_argument("--saturation-voltage", type=float, default=4.95)
    parser.add_argument("--max-sniff-distance-m", type=float, default=0.05)
    parser.add_argument("--min-success-rate", type=float, default=0.70)
    return parser


def main(argv=None) -> int:
    result = validate(build_argparser().parse_args(argv))
    return 0 if result["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

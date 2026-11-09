"""Phase C: continuous clean-air onset and natural single-gas recovery."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from odor_sim.odor_encoder.dataset import (
    GAS_ORDER, current_class_labels, select_split, simulate_mox, write_episode,
)

RECOVERY_DURATIONS_S = (2.0, 5.0, 10.0, 15.0, 20.0, 30.0)


def _fluctuations(steps: int, rng: np.random.Generator, scale: float = 0.02) -> np.ndarray:
    white = rng.normal(0.0, scale, steps)
    width = max(3, min(21, steps // 15))
    return np.convolve(white, np.ones(width) / width, mode="same").astype(np.float32)


def build_sequence(
    *, gas_index: int, concentration_ppm: float, rate: float,
    baseline_s: float, exposure_s: float, recovery_s: float,
    onset_loss_delay_s: float, recovery_loss_delay_s: float,
    rng: np.random.Generator,
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, int],
]:
    """Create one air -> gas -> air sequence and causal class masks."""
    counts = {
        "baseline": max(2, int(round(baseline_s * rate))),
        "exposure": max(2, int(round(exposure_s * rate))),
        "recovery": max(2, int(round(recovery_s * rate))),
    }
    total = sum(counts.values())
    ppm = np.zeros((total, len(GAS_ORDER)), np.float32)
    class_index = np.zeros(total, np.int64)
    phase = np.full(total, "baseline_clean", dtype="U48")
    valid = np.ones(total, bool)

    gas_start = counts["baseline"]
    recovery_start = gas_start + counts["exposure"]
    exposure = np.clip(
        1.0 + _fluctuations(counts["exposure"], rng), 0.90, 1.10,
    )
    ppm[gas_start:recovery_start, gas_index] = concentration_ppm * exposure
    class_index[gas_start:recovery_start] = gas_index + 1
    phase[gas_start:recovery_start] = "single_gas_exposure"
    phase[gas_start:min(recovery_start, gas_start + int(round(rate)))] = "gas_onset"
    phase[recovery_start:] = "clean_air_recovery"

    valid[:min(counts["baseline"], int(round(rate)))] = False
    onset_delay = max(0, int(round(onset_loss_delay_s * rate)))
    valid[gas_start:min(recovery_start, gas_start + onset_delay)] = False
    recovery_delay = max(0, int(round(recovery_loss_delay_s * rate)))
    valid[recovery_start:min(total, recovery_start + recovery_delay)] = False
    recovery = np.zeros(total, bool)
    recovery[recovery_start:] = True
    boundaries = {
        "gas_start": gas_start,
        "recovery_start": recovery_start,
        "sequence_end": total,
    }
    return ppm, class_index, phase, valid, recovery, boundaries


def collect(args: argparse.Namespace) -> None:
    out = Path(args.out_dir)
    for index in range(args.sequences):
        gas_index = index % len(GAS_ORDER)
        gas = GAS_ORDER[gas_index]
        session_id = f"transient_{gas}_{index:06d}"
        split = select_split(session_id, args.seed, args.split)
        path = out / split / f"{session_id}.npz"
        if path.is_file() and not args.overwrite:
            continue
        rng = np.random.default_rng(args.seed + index)
        concentration_ppm = float(10.0 ** rng.uniform(0.0, 2.0))
        baseline_s = float(rng.uniform(args.min_baseline_s, args.max_baseline_s))
        exposure_s = float(rng.uniform(args.min_exposure_s, args.max_exposure_s))
        recovery_s = RECOVERY_DURATIONS_S[(index // len(GAS_ORDER)) % len(RECOVERY_DURATIONS_S)]
        ppm, class_index, phase, valid, recovery, boundaries = build_sequence(
            gas_index=gas_index,
            concentration_ppm=concentration_ppm,
            rate=args.rate,
            baseline_s=baseline_s,
            exposure_s=exposure_s,
            recovery_s=recovery_s,
            onset_loss_delay_s=args.onset_loss_delay_s,
            recovery_loss_delay_s=args.recovery_loss_delay_s,
            rng=rng,
        )
        noise_std_v = float(rng.uniform(args.min_noise_std_v, args.max_noise_std_v))
        voltage, rs_r0 = simulate_mox(
            ppm, args.rate, measurement_noise_std_v=noise_std_v, rng=rng,
        )
        labels = current_class_labels(class_index, ppm, valid, recovery)
        write_episode(
            out,
            session_id=session_id,
            split=split,
            domain="onset_recovery",
            rate=args.rate,
            voltage=voltage,
            rs_r0=rs_r0,
            ppm=ppm,
            time_s=np.arange(len(ppm), dtype=np.float32) / args.rate,
            position=np.full((len(ppm), 3), np.nan, np.float32),
            phase=phase,
            labels=labels,
            metadata={
                "seed": args.seed + index,
                "gas": gas,
                "gas_class_index": gas_index + 1,
                "concentration_ppm": concentration_ppm,
                "baseline_s": baseline_s,
                "exposure_s": exposure_s,
                "recovery_s": recovery_s,
                "onset_loss_delay_s": args.onset_loss_delay_s,
                "recovery_loss_delay_s": args.recovery_loss_delay_s,
                "measurement_noise_std_v": noise_std_v,
                "no_sensor_reset_during_sequence": True,
                "genuine_clean_start": True,
                "transport_mode": "controlled_local_ppm",
                "transition_indices": boundaries,
            },
        )
        if (index + 1) % args.log_every == 0 or index + 1 == args.sequences:
            print(
                f"[phase-c] {index + 1}/{args.sequences} gas={gas} "
                f"exposure={exposure_s:.1f}s recovery={recovery_s:g}s"
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequences", type=int, default=4000)
    parser.add_argument("--out-dir", default="datasets/odor_encoder/onset_recovery_v1")
    parser.add_argument("--rate", type=float, default=20.0)
    parser.add_argument("--min-baseline-s", type=float, default=2.0)
    parser.add_argument("--max-baseline-s", type=float, default=5.0)
    parser.add_argument("--min-exposure-s", type=float, default=5.0)
    parser.add_argument("--max-exposure-s", type=float, default=12.0)
    parser.add_argument("--onset-loss-delay-s", type=float, default=0.5)
    parser.add_argument("--recovery-loss-delay-s", type=float, default=1.0)
    parser.add_argument("--min-noise-std-v", type=float, default=0.0)
    parser.add_argument("--max-noise-std-v", type=float, default=0.01)
    parser.add_argument("--split", choices=("auto", "train", "validation", "test"), default="auto")
    parser.add_argument("--seed", type=int, default=20000)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.sequences < 1 or args.rate <= 0:
        raise SystemExit("--sequences and --rate must be positive")
    if not 0 < args.min_baseline_s <= args.max_baseline_s:
        raise SystemExit("baseline durations must satisfy 0 < min <= max")
    if not 0 < args.min_exposure_s <= args.max_exposure_s:
        raise SystemExit("exposure durations must satisfy 0 < min <= max")
    if args.onset_loss_delay_s < 0 or args.recovery_loss_delay_s < 0:
        raise SystemExit("classification loss delays must be non-negative")
    if not 0 <= args.min_noise_std_v <= args.max_noise_std_v:
        raise SystemExit("noise values must satisfy 0 <= min <= max")
    collect(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

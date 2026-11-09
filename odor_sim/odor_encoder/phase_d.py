"""Collect continuous gas-A -> transit -> gas-B MOX carryover sequences."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from odor_sim.odor_encoder.dataset import (
    GAS_ORDER,
    SAMPLE_RATE_HZ,
    current_class_labels,
    select_split,
    simulate_mox,
    write_episode,
)


def _segment_concentration(
    rng: np.random.Generator,
    count: int,
    nominal_ppm: float,
) -> np.ndarray:
    """Return a positive concentration trace with small correlated fluctuations."""
    if count <= 0:
        return np.empty(0, dtype=np.float64)
    white = rng.normal(0.0, 1.0, size=count)
    correlated = np.empty(count, dtype=np.float64)
    correlated[0] = white[0]
    for i in range(1, count):
        correlated[i] = 0.94 * correlated[i - 1] + 0.06 * white[i]
    scale = 1.0 + 0.04 * correlated
    return np.maximum(nominal_ppm * scale, 0.0)


def build_sequence(
    rng: np.random.Generator,
    gas_a_index: int,
    gas_b_index: int,
    rate_hz: float = SAMPLE_RATE_HZ,
    gas_b_loss_delay_s: float = 1.0,
    transit_loss_delay_s: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, int | float]]:
    """Build one causal baseline -> A -> air transit -> B concentration trace."""
    baseline_s = float(rng.uniform(2.0, 5.0))
    exposure_a_s = float(rng.uniform(7.0, 12.0))
    transit_s = float(rng.uniform(1.0, 7.0))
    exposure_b_s = float(rng.uniform(7.0, 12.0))
    concentration_a = float(10.0 ** rng.uniform(0.0, 2.0))
    concentration_b = float(10.0 ** rng.uniform(0.0, 2.0))

    lengths = [max(1, int(round(s * rate_hz))) for s in (baseline_s, exposure_a_s, transit_s, exposure_b_s)]
    n0, na, nt, nb = lengths
    i_a = n0
    i_transit = i_a + na
    i_b = i_transit + nt
    total = i_b + nb

    ppm = np.zeros((total, len(GAS_ORDER)), dtype=np.float64)
    ppm[i_a:i_transit, gas_a_index] = _segment_concentration(rng, na, concentration_a)
    ppm[i_b:, gas_b_index] = _segment_concentration(rng, nb, concentration_b)

    class_index = np.zeros(total, dtype=np.int64)
    class_index[i_a:i_transit] = gas_a_index + 1
    class_index[i_b:] = gas_b_index + 1

    phase = np.full(total, "baseline_clean", dtype="U24")
    phase[i_a:i_transit] = "gas_a_exposure"
    phase[i_transit:i_b] = "transit_recovery"
    phase[i_b:] = "gas_b_exposure"

    valid = np.ones(total, dtype=bool)
    valid[: min(total, int(round(rate_hz)))] = False
    onset_a = min(na, int(round(0.5 * rate_hz)))
    valid[i_a : i_a + onset_a] = False
    transit_delay = min(nt, int(round(transit_loss_delay_s * rate_hz)))
    valid[i_transit : i_transit + transit_delay] = False
    onset_b = min(nb, int(round(gas_b_loss_delay_s * rate_hz)))
    valid[i_b : i_b + onset_b] = False
    recovery = np.zeros(total, dtype=bool)
    recovery[i_transit:i_b] = True

    boundaries: dict[str, int | float] = {
        "gas_a_start_index": i_a,
        "transit_start_index": i_transit,
        "gas_b_start_index": i_b,
        "end_index": total,
        "baseline_s": baseline_s,
        "gas_a_exposure_s": exposure_a_s,
        "transit_s": transit_s,
        "gas_b_exposure_s": exposure_b_s,
        "gas_a_concentration_ppm": concentration_a,
        "gas_b_concentration_ppm": concentration_b,
    }
    return ppm, class_index, phase, valid, recovery, boundaries


def event_timing_arrays(
    length: int, boundaries: dict[str, int | float], rate: float,
) -> dict[str, np.ndarray]:
    """Describe event boundaries without altering the continuous MOX history."""
    gas_a = int(boundaries["gas_a_start_index"])
    transit = int(boundaries["transit_start_index"])
    gas_b = int(boundaries["gas_b_start_index"])
    event_id = np.zeros(length, np.int8)
    event_id[gas_a:transit] = 1
    event_id[transit:gas_b] = 2
    event_id[gas_b:] = 3
    event_elapsed_s = np.empty(length, np.float32)
    for event in range(4):
        rows = np.flatnonzero(event_id == event)
        event_elapsed_s[rows] = np.arange(len(rows), dtype=np.float32) / rate
    sniff_elapsed_s = np.full(length, np.nan, np.float32)
    sniff_rows = (event_id == 1) | (event_id == 3)
    sniff_elapsed_s[sniff_rows] = event_elapsed_s[sniff_rows]
    return {
        "event_id": event_id,
        "event_elapsed_s": event_elapsed_s,
        "sniff_elapsed_s": sniff_elapsed_s,
    }


def collect(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    total = len(GAS_ORDER) ** 2 * args.sequences_per_pair
    completed = 0

    for gas_a_index, gas_a in enumerate(GAS_ORDER):
        for gas_b_index, gas_b in enumerate(GAS_ORDER):
            for replicate in range(args.sequences_per_pair):
                pair_index = gas_a_index * len(GAS_ORDER) + gas_b_index
                seed = args.seed + pair_index * 100_000 + replicate
                rng = np.random.default_rng(seed)
                session_id = f"carry_{gas_a}_to_{gas_b}_{replicate:04d}"
                split = select_split(session_id, seed, args.split)
                path = out_dir / split / f"{session_id}.npz"
                if path.is_file() and not args.overwrite:
                    completed += 1
                    continue
                ppm, class_index, phase, valid, recovery, boundaries = build_sequence(
                    rng,
                    gas_a_index,
                    gas_b_index,
                    args.rate,
                    args.gas_b_loss_delay_s,
                    args.transit_loss_delay_s,
                )
                noise_std_v = float(rng.uniform(args.min_noise_std_v, args.max_noise_std_v))
                voltage, rs_r0 = simulate_mox(
                    ppm,
                    args.rate,
                    measurement_noise_std_v=noise_std_v,
                    rng=rng,
                )
                labels = current_class_labels(class_index, ppm, valid, recovery)
                labels.update(event_timing_arrays(len(ppm), boundaries, args.rate))
                time_s = np.arange(ppm.shape[0], dtype=np.float64) / args.rate
                position = np.full((ppm.shape[0], 3), np.nan, dtype=np.float64)
                metadata = {
                    "session_id": session_id,
                    "domain": "sequential_carryover",
                    "split": split,
                    "seed": seed,
                    "sample_rate_hz": args.rate,
                    "gas_a": gas_a,
                    "gas_b": gas_b,
                    "same_gas_control": gas_a == gas_b,
                    "ordered_pair": f"{gas_a}->{gas_b}",
                    "event_id_semantics": {
                        "0": "initial_clean_air", "1": "gas_a_sniff",
                        "2": "clean_air_transit", "3": "gas_b_sniff",
                    },
                    "measurement_noise_std_v": noise_std_v,
                    "gas_b_loss_delay_s": args.gas_b_loss_delay_s,
                    "transit_loss_delay_s": args.transit_loss_delay_s,
                    "no_sensor_reset_during_sequence": True,
                    "wind_regime": "zero",
                    "buoyancy_tau_s": 0.0,
                    "original_terminal_velocity_enabled": False,
                    **boundaries,
                }
                write_episode(
                    out_dir,
                    split=split,
                    session_id=session_id,
                    domain="sequential_carryover",
                    rate=args.rate,
                    voltage=voltage,
                    rs_r0=rs_r0,
                    ppm=ppm,
                    time_s=time_s,
                    position=position,
                    phase=phase,
                    labels=labels,
                    metadata=metadata,
                )
                completed += 1
                if completed % 100 == 0 or completed == total:
                    print(f"[phase-d] {completed}/{total} sequences")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequences-per-pair", type=int, default=120)
    parser.add_argument("--out-dir", default="datasets/odor_encoder/carryover_v2")
    parser.add_argument("--rate", type=float, default=SAMPLE_RATE_HZ)
    parser.add_argument("--seed", type=int, default=41000)
    parser.add_argument("--split", choices=("auto", "train", "validation", "test"), default="auto")
    parser.add_argument("--gas-b-loss-delay-s", type=float, default=1.0)
    parser.add_argument("--transit-loss-delay-s", type=float, default=1.0)
    parser.add_argument("--min-noise-std-v", type=float, default=0.0)
    parser.add_argument("--max-noise-std-v", type=float, default=0.01)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.sequences_per_pair <= 0:
        raise ValueError("--sequences-per-pair must be positive")
    if args.rate <= 0:
        raise ValueError("--rate must be positive")
    if not 0 <= args.min_noise_std_v <= args.max_noise_std_v:
        raise ValueError("noise standard-deviation range is invalid")
    if args.gas_b_loss_delay_s < 0 or args.transit_loss_delay_s < 0:
        raise ValueError("classification loss delays must be non-negative")
    collect(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

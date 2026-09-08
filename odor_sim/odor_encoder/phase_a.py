"""Phase A: controlled static single-gas sequence generation."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from odor_sim.odor_encoder.dataset import (
    GAS_ORDER, sample_background, sample_single_source, select_split, simulate_mox,
    source_labels, write_episode,
)

PATTERNS = ("step", "constant", "ramp_up", "ramp_down", "incomplete_recovery")


def _smooth_fluctuations(steps: int, rng: np.random.Generator, scale: float) -> np.ndarray:
    """Small correlated fluctuations, not independent step-wise white noise."""
    white = rng.normal(0.0, scale, steps)
    width = max(3, min(21, steps // 20))
    kernel = np.ones(width, np.float64) / width
    return np.convolve(white, kernel, mode="same").astype(np.float32)


def concentration_envelope(pattern: str, steps: int, rate: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Return concentration multiplier and composition-supervision mask."""
    x = np.zeros(steps, np.float32)
    baseline = min(steps // 4, int(round(rng.uniform(2.0, 4.0) * rate)))
    remain = max(1, steps - baseline)
    if pattern == "step":
        gas_end = baseline + int(0.65 * remain)
        x[baseline:gas_end] = 1.0
    elif pattern == "constant":
        x[baseline:] = 1.0
    elif pattern == "ramp_up":
        gas_end = baseline + int(0.75 * remain)
        x[baseline:gas_end] = np.linspace(0.03, 1.0, max(0, gas_end - baseline))
    elif pattern == "ramp_down":
        gas_end = baseline + int(0.75 * remain)
        x[baseline:gas_end] = np.linspace(1.0, 0.03, max(0, gas_end - baseline))
    elif pattern == "incomplete_recovery":
        first = min(steps, baseline + int(0.32 * remain))
        second = min(steps, first + int(0.18 * remain))
        x[baseline:first] = 1.0
        x[first:second] = np.linspace(0.0, 0.15, max(0, second - first))
        x[second:min(steps, second + int(0.34 * remain))] = rng.uniform(0.45, 1.0)
    else:
        raise ValueError(f"unknown pattern {pattern!r}")
    exposed = x > 0
    x[exposed] *= np.clip(1.0 + _smooth_fluctuations(steps, rng, 0.025)[exposed], 0.9, 1.1)
    # Recovery is useful input history, but source-fraction supervision begins
    # only once the controlled exposure has started.
    valid = np.arange(steps) >= baseline
    return x, valid


def collect(args: argparse.Namespace) -> None:
    out = Path(args.out_dir)
    saturation = 0
    for index in range(args.sessions):
        session_id = f"static_{index:06d}"
        split = select_split(session_id, args.seed, args.split)
        if (out / split / f"{session_id}.npz").is_file():
            continue
        rng = np.random.default_rng(args.seed + index)
        spec = sample_single_source(rng, clean_probability=args.clean_probability)
        background = sample_background(
            rng, spec, max_gases=args.background_max_gases,
            min_fraction=args.background_min_fraction,
            max_fraction=args.background_max_fraction,
        )
        pattern = PATTERNS[int(rng.integers(len(PATTERNS)))]
        duration = rng.uniform(args.min_duration_s, args.max_duration_s)
        steps = max(2, int(round(duration * args.rate)))
        envelope, valid = concentration_envelope(pattern, steps, args.rate, rng)
        ppm = envelope[:, None] * spec.concentration_ppm[None, :]
        if background.gases:
            background_level = np.zeros(steps, np.float32)
            background_level[valid] = np.clip(
                0.65 + _smooth_fluctuations(steps, rng, 0.12)[valid], 0.35, 1.0
            )
            ppm += background_level[:, None] * background.concentration_ppm[None, :]
        precondition_ppm = None
        precondition_gas = None
        if spec.gases and rng.random() < args.precondition_probability:
            precondition_steps = int(round(rng.uniform(2.0, 8.0) * args.rate))
            precondition_gas = str(rng.choice(GAS_ORDER))
            precondition_ppm = np.zeros((precondition_steps, len(GAS_ORDER)), np.float32)
            precondition_ppm[:, GAS_ORDER.index(precondition_gas)] = 10.0 ** rng.uniform(0.0, 1.5)
        noise_std_v = float(rng.uniform(args.min_noise_std_v, args.max_noise_std_v))
        voltage, rs_r0 = simulate_mox(
            ppm, args.rate, measurement_noise_std_v=noise_std_v,
            rng=rng, precondition_ppm=precondition_ppm,
        )
        saturation += int(np.any(voltage >= args.saturation_v))
        exposed = (envelope > 0) & bool(spec.gases)
        recovery = np.maximum.accumulate(exposed) & ~exposed & valid
        labels = source_labels(spec, ppm, valid, recovery)
        write_episode(
            out, session_id=session_id, split=split, domain="static", rate=args.rate,
            voltage=voltage, rs_r0=rs_r0, ppm=ppm,
            time_s=np.arange(steps) / args.rate,
            position=np.full((steps, 3), np.nan, np.float32),
            phase=np.full(steps, pattern), labels=labels,
            metadata={
                "pattern": pattern, "seed": int(args.seed + index),
                "source_concentration_ppm": spec.concentration_ppm.tolist(),
                "background_concentration_ppm": background.concentration_ppm.tolist(),
                "evaluation_group": spec.evaluation_group,
                "measurement_noise_std_v": noise_std_v,
                "precondition_gas": precondition_gas,
                "precondition_steps": 0 if precondition_ppm is None else len(precondition_ppm),
            },
        )
        if (index + 1) % args.log_every == 0 or index + 1 == args.sessions:
            print(f"[phase-a] {index + 1}/{args.sessions} sessions")
    print(f"[phase-a] voltage >= {args.saturation_v:g} V in {saturation}/{args.sessions} sessions")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sessions", type=int, default=5000, help="Pilot target: 5000")
    parser.add_argument("--out-dir", default="datasets/odor_encoder/static_single_v3")
    parser.add_argument("--rate", type=float, default=20.0)
    parser.add_argument("--min-duration-s", type=float, default=12.0)
    parser.add_argument("--max-duration-s", type=float, default=25.0)
    parser.add_argument("--saturation-v", type=float, default=4.95)
    parser.add_argument("--clean-probability", type=float, default=0.10)
    parser.add_argument("--background-max-gases", type=int, default=2)
    parser.add_argument("--background-min-fraction", type=float, default=0.002)
    parser.add_argument("--background-max-fraction", type=float, default=0.03)
    parser.add_argument("--min-noise-std-v", type=float, default=0.0)
    parser.add_argument("--max-noise-std-v", type=float, default=0.01)
    parser.add_argument("--precondition-probability", type=float, default=0.15)
    parser.add_argument("--split", choices=("auto", "train", "validation", "test"), default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=100)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.sessions < 1 or args.rate <= 0 or args.min_duration_s <= 0 or args.max_duration_s < args.min_duration_s:
        raise SystemExit("invalid session/rate/duration arguments")
    if not 0.0 <= args.clean_probability <= 1.0:
        raise SystemExit("--clean-probability must be in [0, 1]")
    if not 0 < args.background_min_fraction <= args.background_max_fraction:
        raise SystemExit("background fractions must satisfy 0 < min <= max")
    if not 0 <= args.min_noise_std_v <= args.max_noise_std_v:
        raise SystemExit("noise values must satisfy 0 <= min <= max")
    if not 0 <= args.precondition_probability <= 1:
        raise SystemExit("--precondition-probability must be in [0, 1]")
    collect(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

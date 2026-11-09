"""Deployment-consistent two-cup GADEN data collection for ScentFormer A/B/C/D.

Every phase uses two continuously active physical GADEN sources, one uninterrupted
MOX state, endpoint-local physical labels, and event-relative endpoint buckets.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import time

import numpy as np

from odor_sim.config.frame_map import FrameMap
from odor_sim.envs.registry import resolve_scenario
from odor_sim.odor_encoder.dataset import (
    GAS_ORDER,
    SENSOR_ORDER,
    MixtureSpec,
    simulate_mox,
    write_episode,
)
from odor_sim.odor_encoder.phase_b import (
    _load_local_ppm_calibration,
    _map_desired_local_to_nominal,
    _object,
    _run_gaden,
)
from odor_sim.odor_encoder.wind import WIND_REGIMES, sample_wind_profile, zero_wind_profile
from odor_sim.sensors.mox_pid import MOXSensor

LEFT_X = (0.12, 0.20)
LEFT_Y = (-0.16, -0.08)
RIGHT_X = (0.12, 0.20)
RIGHT_Y = (0.08, 0.16)
PLATE_X = (-0.06, 0.06)
PLATE_Y = (-0.06, 0.06)
DEFAULT_SOURCE_Z = 0.87
DEFAULT_FILTER_Z = 1.0046438593615088
PHASE_ROS_DOMAINS = {"A": 81, "B": 82, "C": 83, "D": 84}
DEFAULT_CALIBRATION = (
    Path(__file__).resolve().parents[1]
    / "config/phase_b_local_ppm_exp0p5_scale0p06_fps15.json"
)
CROSS_PAIRS = tuple((a, b) for a in range(len(GAS_ORDER)) for b in range(len(GAS_ORDER)) if a != b)
PHASE_SEED_OFFSET = {"A": 0, "B": 10_000_000, "C": 20_000_000, "D": 30_000_000}
PAIR_SPLIT_CYCLE = (
    "train", "validation", "train", "test", "train", "train",
    "train", "validation", "train", "train", "test", "train",
    "train", "validation", "train", "train", "train", "test", "train", "train",
)
PAIR_SPLIT_CYCLE_SWAPPED = tuple(
    "test" if split == "validation" else "validation" if split == "test" else split
    for split in PAIR_SPLIT_CYCLE
)
SNIFF_OFFSETS_S = (-0.25, 0.0, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0)
DEPARTURE_OFFSETS_S = (-0.05, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0)


@dataclass
class SniffEvent:
    role: str
    gas_index: int
    search_start: int
    search_end: int
    dwell_end: int


@dataclass
class TrajectoryPlan:
    position: np.ndarray
    phase: np.ndarray
    sniff_events: list[SniffEvent] = field(default_factory=list)
    departure_indices: list[int] = field(default_factory=list)
    trajectory_type: str = ""
    memory_start_index: int | None = None


class PathBuilder:
    def __init__(self, rate: float, initial: np.ndarray):
        self.rate = float(rate)
        self.current = np.asarray(initial, np.float32)
        self.rows: list[np.ndarray] = []
        self.phases: list[str] = []

    @property
    def index(self) -> int:
        return len(self.rows)

    def hold(self, seconds: float, phase: str, *, jitter_m: float = 0.0, rng=None) -> tuple[int, int]:
        count = max(1, int(round(seconds * self.rate)))
        start = self.index
        for _ in range(count):
            point = self.current.copy()
            if jitter_m and rng is not None:
                point += rng.normal(0.0, jitter_m, 3).astype(np.float32)
            self.rows.append(point)
            self.phases.append(phase)
        return start, self.index

    def move(self, destination: np.ndarray, seconds: float, phase: str) -> tuple[int, int]:
        destination = np.asarray(destination, np.float32)
        count = max(2, int(round(seconds * self.rate)))
        start = self.index
        for point in np.linspace(self.current, destination, count, endpoint=False):
            self.rows.append(np.asarray(point, np.float32))
            self.phases.append(phase)
        self.current = destination
        return start, self.index

    def finish(self) -> tuple[np.ndarray, np.ndarray]:
        return np.asarray(self.rows, np.float32), np.asarray(self.phases, dtype="U48")


def scheduled_pair(index: int, seed: int, same_gas_fraction: float = 0.0) -> tuple[int, int]:
    """Balance all 30 cross-gas directions in each complete 30-episode block."""
    block, slot = divmod(index, len(CROSS_PAIRS))
    schedule = list(CROSS_PAIRS)
    rng = np.random.default_rng(seed + block * 104729)
    rng.shuffle(schedule)
    pair = schedule[slot]
    control_rng = np.random.default_rng(seed + block * 104729 + slot * 15485863)
    if same_gas_fraction > 0 and control_rng.random() < same_gas_fraction:
        gas = int(control_rng.integers(len(GAS_ORDER)))
        return gas, gas
    return pair


def _pair_schedule(index: int, seed: int) -> tuple[int, int, list[tuple[int, int]]]:
    """Return block, slot, and the deterministic cross-pair schedule."""
    block, slot = divmod(index, len(CROSS_PAIRS))
    schedule = list(CROSS_PAIRS)
    rng = np.random.default_rng(seed + block * 104729)
    rng.shuffle(schedule)
    return block, slot, schedule


def _balanced_desired_ppm(
    index: int,
    seed: int,
    gas_index: int,
    role_index: int,
    low: float,
    high: float,
) -> float:
    """Log-stratify source conditions identically for every gas and cup role.

    Each gas occurs five times as cup A and five times as cup B in a complete
    30-pair block. Each log-concentration stratum is therefore used once per
    gas and role. Gas-specific permutations break any fixed association with
    the opposite-cup gas while preserving identical marginal distributions.
    """
    block, slot, schedule = _pair_schedule(index, seed)
    occurrence_rank = sum(
        pair[role_index] == gas_index for pair in schedule[:slot]
    )
    if schedule[slot][role_index] != gas_index or occurrence_rank >= 5:
        # Optional same-gas controls replace the scheduled cross pair, so they
        # cannot participate in the exact cross-pair stratification.
        rng = np.random.default_rng(
            seed + block * 1_000_003 + index * 1009 + role_index * 97 + gas_index,
        )
        return _sample_desired_ppm(rng, low, high)

    value_rng = np.random.default_rng(seed + block * 1_000_003 + role_index * 10007)
    log_quantiles = (np.arange(5) + value_rng.uniform(size=5)) / 5.0
    values = np.exp(np.log(low) + log_quantiles * (np.log(high) - np.log(low)))
    permutation_rng = np.random.default_rng(
        seed + block * 1_000_003 + role_index * 10007 + gas_index * 15485863,
    )
    return float(values[permutation_rng.permutation(5)[occurrence_rank]])


def _phase_a_mode(index: int, gas_a_index: int, gas_b_index: int) -> str:
    """Spread all six Phase-A position modes across every 30-pair block."""
    modes = ("near_a", "near_b", "between", "offset_a", "offset_b", "jitter_a")
    block = index // len(CROSS_PAIRS)
    pair_code = gas_a_index * (len(GAS_ORDER) - 1)
    pair_code += gas_b_index if gas_b_index < gas_a_index else gas_b_index - 1
    return modes[(pair_code + block) % len(modes)]


def _scheduled_split(index: int, gas_a_index: int, gas_b_index: int) -> str:
    """Maintain 70/15/15 globally while retaining every pair in every split."""
    block = index // len(CROSS_PAIRS)
    pair_code = gas_a_index * (len(GAS_ORDER) - 1)
    pair_code += gas_b_index if gas_b_index < gas_a_index else gas_b_index - 1
    cycle = PAIR_SPLIT_CYCLE if pair_code % 2 == 0 else PAIR_SPLIT_CYCLE_SWAPPED
    return cycle[block % len(cycle)]


def _phase_a_role_is_target(mode: str, role_index: int) -> bool:
    return (role_index == 0 and mode in ("near_a", "jitter_a")) or (
        role_index == 1 and mode == "near_b"
    )


def _phase_a_desired_ppm(
    index: int,
    seed: int,
    gas_index: int,
    role_index: int,
    low: float,
    high: float,
) -> float:
    """Balance concentrations within targeted and background Phase-A roles.

    Balancing only the overall source distribution is insufficient: a gas can
    still receive mostly weak conditions whenever the e-nose is near its cup.
    Every gas therefore walks the same stratified sequence separately for
    targeted-near-cup and non-target/background occurrences.
    """
    gas_a, gas_b = scheduled_pair(index, seed)
    current_mode = _phase_a_mode(index, gas_a, gas_b)
    targeted = _phase_a_role_is_target(current_mode, role_index)
    occurrence_rank = 0
    for previous_index in range(index):
        previous_a, previous_b = scheduled_pair(previous_index, seed)
        previous_pair = (previous_a, previous_b)
        previous_mode = _phase_a_mode(previous_index, previous_a, previous_b)
        for previous_role, previous_gas in enumerate(previous_pair):
            if previous_gas == gas_index and (
                _phase_a_role_is_target(previous_mode, previous_role) == targeted
            ):
                occurrence_rank += 1

    strata = 10
    cycle, offset = divmod(occurrence_rank, strata)
    rng = np.random.default_rng(seed + cycle * 1_000_003 + int(targeted) * 15485863)
    log_quantiles = (np.arange(strata) + rng.uniform(size=strata)) / strata
    values = np.exp(np.log(low) + log_quantiles * (np.log(high) - np.log(low)))
    return float(values[rng.permutation(strata)[offset]])


def _pure(index: int, ppm: float) -> MixtureSpec:
    values = np.zeros(len(GAS_ORDER), np.float32)
    values[index] = ppm
    return MixtureSpec(values, "two_cup")


def _sample_desired_ppm(rng: np.random.Generator, low: float, high: float) -> float:
    return float(np.exp(rng.uniform(np.log(low), np.log(high))))


def _layout(rng: np.random.Generator, frame_map: FrameMap, source_z: float, filter_z: float):
    left = np.asarray([rng.uniform(*LEFT_X), rng.uniform(*LEFT_Y), source_z])
    right = np.asarray([rng.uniform(*RIGHT_X), rng.uniform(*RIGHT_Y), source_z])
    if rng.random() < 0.5:
        cup_a_robot, cup_b_robot = left, right
        assignment = "A_left_B_right"
    else:
        cup_a_robot, cup_b_robot = right, left
        assignment = "A_right_B_left"
    filter_robot = np.asarray([rng.uniform(*PLATE_X), rng.uniform(*PLATE_Y), filter_z])
    cup_a = frame_map.robosuite_to_gaden(cup_a_robot)
    cup_b = frame_map.robosuite_to_gaden(cup_b_robot)
    filter_position = frame_map.robosuite_to_gaden(filter_robot)
    return cup_a, cup_b, filter_position, cup_a_robot, cup_b_robot, filter_robot, assignment


def _sniff_point(source: np.ndarray, rng: np.random.Generator, args) -> np.ndarray:
    radius = rng.uniform(0.0, args.max_lateral_offset_m)
    theta = rng.uniform(-np.pi, np.pi)
    return np.asarray(source, np.float32) + np.asarray([
        radius * np.cos(theta), radius * np.sin(theta),
        rng.uniform(args.min_sniff_height_m, args.max_sniff_height_m),
    ], np.float32)


def _add_sniff(
    builder: PathBuilder,
    source: np.ndarray,
    gas_index: int,
    role: str,
    rng: np.random.Generator,
    args,
    *,
    dwell_s: float,
) -> SniffEvent:
    sniff = _sniff_point(source, rng, args)
    hover = sniff.copy()
    hover[2] = max(hover[2], float(source[2]) + rng.uniform(0.10, 0.20))
    approach_start, _ = builder.move(
        hover, rng.uniform(args.min_approach_s, args.max_approach_s),
        "second_cup_approach" if role == "second" else "approach",
    )
    builder.move(
        sniff, rng.uniform(args.min_descend_s, args.max_descend_s),
        "second_cup_onset" if role == "second" else "first_contact",
    )
    _, dwell_end = builder.hold(
        dwell_s, "second_cup_dwell" if role == "second" else "dwell",
        jitter_m=args.dwell_jitter_m, rng=rng,
    )
    return SniffEvent(role, gas_index, approach_start, dwell_end, dwell_end)


def build_trajectory(
    phase_name: str,
    index: int,
    gas_a_index: int,
    gas_b_index: int,
    cup_a: np.ndarray,
    cup_b: np.ndarray,
    filter_position: np.ndarray,
    rng: np.random.Generator,
    args,
) -> TrajectoryPlan:
    """Build an embodied path. C/D never synthesize ppm or reset MOX state."""
    builder = PathBuilder(args.rate, filter_position)
    events: list[SniffEvent] = []
    departures: list[int] = []
    memory_start = None

    if phase_name == "A":
        mode = _phase_a_mode(index, gas_a_index, gas_b_index)
        if mode in ("near_a", "near_b", "jitter_a"):
            source = cup_a if mode != "near_b" else cup_b
            gas_index = gas_a_index if mode != "near_b" else gas_b_index
            builder.current = _sniff_point(source, rng, args)
            start, end = builder.hold(
                rng.uniform(args.min_static_s, args.max_static_s), "dwell",
                jitter_m=args.dwell_jitter_m if mode == "jitter_a" else 0.0, rng=rng,
            )
            events.append(SniffEvent("first", gas_index, start, end, end))
        elif mode == "between":
            builder.current = 0.5 * (cup_a + cup_b) + np.asarray([0, 0, 0.05])
            builder.hold(rng.uniform(args.min_static_s, args.max_static_s), "background")
        else:
            source = cup_a if mode == "offset_a" else cup_b
            direction = filter_position - source
            direction /= max(float(np.linalg.norm(direction[:2])), 1e-6)
            builder.current = source + direction * rng.uniform(0.08, 0.18)
            builder.current[2] = source[2] + rng.uniform(0.03, 0.08)
            builder.hold(rng.uniform(args.min_static_s, args.max_static_s), "stationary")
        trajectory_type = mode

    elif phase_name == "B":
        source_is_a = index % 2 == 0
        source = cup_a if source_is_a else cup_b
        gas_index = gas_a_index if source_is_a else gas_b_index
        builder.hold(rng.uniform(1.0, 3.0), "background")
        event = _add_sniff(
            builder, source, gas_index, "first", rng, args,
            dwell_s=rng.uniform(args.min_sniff_s, args.max_sniff_s),
        )
        events.append(event)
        departures.append(builder.index)
        memory_start = builder.index
        builder.move(
            filter_position, rng.uniform(args.min_transit_s, args.max_transit_s), "retreat",
        )
        builder.hold(rng.uniform(1.0, 4.0), "recovery_filter")
        trajectory_type = "approach_dwell_retreat_a" if source_is_a else "approach_dwell_retreat_b"

    elif phase_name == "C":
        mode = ("onset_a", "onset_b", "gas_filter_a", "gas_filter_b", "filter_to_a", "same_gas_return")[index % 6]
        source = cup_b if mode in ("onset_b", "gas_filter_b") else cup_a
        gas_index = gas_b_index if source is cup_b else gas_a_index
        if mode.startswith("onset") or mode.startswith("filter_to"):
            builder.hold(rng.uniform(1.0, 4.0), "background")
        event = _add_sniff(
            builder, source, gas_index, "first", rng, args,
            dwell_s=rng.uniform(args.min_sniff_s, args.max_sniff_s),
        )
        events.append(event)
        if mode.startswith("gas_filter") or mode == "same_gas_return":
            departures.append(builder.index)
            memory_start = builder.index
            builder.move(filter_position, rng.uniform(1.0, 4.0), "retreat")
            builder.hold(float(rng.choice(args.recovery_durations_s)), "recovery_filter")
        if mode == "same_gas_return":
            events.append(_add_sniff(
                builder, source, gas_index, "second", rng, args,
                dwell_s=rng.uniform(args.min_sniff_s, args.max_sniff_s),
            ))
        trajectory_type = mode

    elif phase_name == "D":
        builder.hold(rng.uniform(1.0, 4.0), "background")
        events.append(_add_sniff(
            builder, cup_a, gas_a_index, "first", rng, args,
            dwell_s=rng.uniform(args.min_sniff_s, args.max_sniff_s),
        ))
        departures.append(builder.index)
        memory_start = builder.index
        recovery_s = float(rng.choice(args.recovery_durations_s))
        direct = recovery_s == 0.0 or rng.random() < args.direct_transfer_fraction
        if direct:
            builder.move(cup_b + np.asarray([0, 0, 0.15]), rng.uniform(1.0, 7.0), "transit")
        else:
            builder.move(filter_position, rng.uniform(1.0, 4.0), "recovery_filter")
            builder.hold(recovery_s, "recovery_filter")
            builder.move(cup_b + np.asarray([0, 0, 0.15]), rng.uniform(1.0, 7.0), "transit")
        events.append(_add_sniff(
            builder, cup_b, gas_b_index, "second", rng, args,
            dwell_s=rng.uniform(args.min_sniff_s, args.max_sniff_s),
        ))
        trajectory_type = "direct_transfer" if direct else "filter_transfer"
    else:
        raise ValueError(f"unknown phase {phase_name!r}")

    position, phases = builder.finish()
    return TrajectoryPlan(position, phases, events, departures, trajectory_type, memory_start)


def _causal_mean(values: np.ndarray, width: int) -> np.ndarray:
    cumulative = np.concatenate(([0.0], np.cumsum(values, dtype=np.float64)))
    result = np.empty(len(values), np.float64)
    for i in range(len(values)):
        start = max(0, i - width + 1)
        result[i] = (cumulative[i + 1] - cumulative[start]) / (i - start + 1)
    return result


def physical_labels(
    ppm: np.ndarray,
    rs_r0: np.ndarray,
    *,
    rate: float,
    presence_threshold_ppm: float,
    gas_class_min_ppm: float,
    dominant_threshold: float,
    slope_window_s: float,
    rising_rate_per_s: float,
    falling_rate_per_s: float,
    memory_start_index: int | None,
    recovery_residual_threshold: float,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Create endpoint-local labels using only present and past physical state."""
    ppm = np.asarray(ppm, np.float32)
    total = ppm.sum(1)
    dominant_index = ppm.argmax(1).astype(np.int64)
    dominant_ppm = ppm[np.arange(len(ppm)), dominant_index]
    dominant_fraction = np.divide(
        dominant_ppm, total, out=np.zeros_like(total), where=total > 0,
    )
    presence = total >= presence_threshold_ppm

    lag = max(1, int(round(slope_window_s * rate)))
    smooth_width = max(1, int(round(min(0.25, slope_window_s) * rate)))
    slopes = np.zeros(len(ppm), np.float32)
    for gas_index in range(len(GAS_ORDER)):
        trace = _causal_mean(ppm[:, gas_index], smooth_width)
        rows = np.flatnonzero(dominant_index == gas_index)
        for i in rows:
            past = max(0, i - lag)
            slopes[i] = (trace[i] - trace[past]) / max((i - past) / rate, 1.0 / rate)
    relative_slope = slopes / np.maximum(dominant_ppm, presence_threshold_ppm)

    exposure_state = np.full(len(ppm), "CLEAN_OR_NO_VOC", dtype="U24")
    mixed = presence & (dominant_fraction < dominant_threshold)
    falling = presence & ~mixed & (relative_slope <= -falling_rate_per_s)
    rising = presence & ~mixed & (relative_slope >= rising_rate_per_s)
    stable = presence & ~mixed & ~falling & ~rising
    exposure_state[mixed] = "MIXED_OR_AMBIGUOUS"
    exposure_state[falling] = "FALLING_DOMINANT"
    exposure_state[rising] = "RISING_DOMINANT"
    exposure_state[stable] = "STABLE_DOMINANT"

    gas_mask = (
        presence
        & (total >= gas_class_min_ppm)
        & (dominant_fraction >= dominant_threshold)
        & (rising | stable)
    )
    mask_reason = np.full(len(ppm), "SUPERVISED", dtype="U16")
    mask_reason[~presence] = "NO_GAS"
    mask_reason[mixed] = "MIXED"
    mask_reason[falling] = "FALLING"
    weak = presence & (total < gas_class_min_ppm)
    mask_reason[weak] = "WEAK"

    recovery = np.zeros(len(ppm), np.float32)
    residual_score = np.zeros(len(ppm), np.float32)
    if memory_start_index is not None and memory_start_index < len(ppm):
        _, reference = simulate_mox(ppm[memory_start_index:], rate)
        actual = np.asarray(rs_r0[memory_start_index:], np.float32)
        scales = np.asarray(
            [MOXSensor(model=model, rate=rate).sensitivity_air for model in SENSOR_ORDER],
            np.float32,
        )
        scores = np.mean(np.abs(actual - reference) / np.maximum(scales, 1e-4), axis=1)
        residual_score[memory_start_index:] = scores
        recovery[memory_start_index:] = scores >= recovery_residual_threshold

    labels = {
        "label_gas": dominant_index,
        "label_class": dominant_index,
        "gas_loss_mask": gas_mask.astype(np.float32),
        "label_class_valid": gas_mask,
        "label_valid": np.ones(len(ppm), bool),
        "label_presence": presence.astype(np.float32),
        "label_intensity": np.log1p(total).astype(np.float32),
        "label_recovery": recovery,
    }
    physical = {
        "local_total_ppm": total.astype(np.float32),
        "dominant_gas_index": dominant_index,
        "dominant_fraction": dominant_fraction.astype(np.float32),
        "concentration_slope_ppm_s": slopes,
        "relative_concentration_slope_s": relative_slope,
        "exposure_state": exposure_state,
        "gas_mask_reason": mask_reason,
        "recovery_residual_score": residual_score,
    }
    return labels, physical


def _detect_event_onsets(
    events: list[SniffEvent], ppm: np.ndarray, total: np.ndarray,
    presence_threshold: float, dominance_threshold: float,
) -> list[int]:
    onsets = []
    for event in events:
        rows = np.arange(event.search_start, min(event.search_end, len(ppm)))
        fraction = np.divide(
            ppm[rows, event.gas_index], total[rows],
            out=np.zeros(len(rows), np.float32), where=total[rows] > 0,
        )
        detected = rows[
            (total[rows] >= presence_threshold)
            & (fraction >= min(0.70, dominance_threshold))
        ]
        onsets.append(int(detected[0]) if len(detected) else event.search_start)
    return onsets


def timing_and_endpoints(
    plan: TrajectoryPlan,
    ppm: np.ndarray,
    labels: dict[str, np.ndarray],
    physical: dict[str, np.ndarray],
    *,
    rate: float,
    rng: np.random.Generator,
    jitter_s: float,
    presence_threshold: float,
    dominance_threshold: float,
) -> dict[str, np.ndarray]:
    length = len(ppm)
    total = physical["local_total_ppm"]
    onsets = _detect_event_onsets(
        plan.sniff_events, ppm, total, presence_threshold, dominance_threshold,
    )
    sniff_elapsed = np.full(length, np.nan, np.float32)
    since_onset = np.full(length, np.nan, np.float32)
    sniff_role = np.full(length, "none", dtype="U8")
    for event, onset in zip(plan.sniff_events, onsets):
        stop = min(event.dwell_end, length)
        if onset < stop:
            values = np.arange(stop - onset, dtype=np.float32) / rate
            sniff_elapsed[onset:stop] = values
            since_onset[onset:stop] = values
            sniff_role[onset:stop] = event.role

    since_departure = np.full(length, np.nan, np.float32)
    if plan.departure_indices:
        departure = plan.departure_indices[0]
        since_departure[departure:] = np.arange(length - departure, dtype=np.float32) / rate

    since_transition = np.zeros(length, np.float32)
    current = np.where(labels["label_presence"] > 0.5, physical["dominant_gas_index"], -1)
    last_change = 0
    for i in range(1, length):
        if current[i] != current[i - 1]:
            last_change = i
        since_transition[i] = (i - last_change) / rate

    endpoint_bucket = np.full(length, "", dtype="U48")
    endpoint_eligible = np.zeros(length, bool)

    def mark(index: int, name: str):
        index = int(index)
        if not 0 <= index < length:
            return
        if not endpoint_eligible[index]:
            endpoint_eligible[index] = True
            endpoint_bucket[index] = name

    for fraction, name in ((0.0, "trajectory_start"), (0.25, "trajectory_early"),
                           (0.5, "trajectory_middle"), (0.75, "trajectory_late"),
                           (1.0, "trajectory_end")):
        mark(round(fraction * (length - 1)), name)

    for event, onset in zip(plan.sniff_events, onsets):
        for offset in SNIFF_OFFSETS_S:
            jitter = rng.uniform(-jitter_s, jitter_s) if offset >= 0 else 0.0
            suffix = "pre" if offset < 0 else f"{offset:g}s"
            endpoint = onset + round((offset + jitter) * rate)
            if endpoint < event.dwell_end:
                mark(endpoint, f"{event.role}_sniff_{suffix}")
        mark(event.dwell_end - 1, f"{event.role}_sniff_mature")

    for departure_number, departure in enumerate(plan.departure_indices, start=1):
        for offset in DEPARTURE_OFFSETS_S:
            jitter = rng.uniform(-jitter_s, jitter_s) if offset > 0 else 0.0
            suffix = "before" if offset < 0 else f"{offset:g}s"
            mark(departure + round((offset + jitter) * rate), f"departure{departure_number}_{suffix}")

    return {
        "sniff_elapsed_s": sniff_elapsed,
        "time_since_current_sniff_onset_s": since_onset,
        "time_since_last_gas_transition_s": since_transition,
        "time_since_departure_from_previous_cup_s": since_departure,
        "sniff_role": sniff_role,
        "endpoint_bucket": endpoint_bucket,
        "endpoint_eligible": endpoint_eligible,
    }


def _run_physical_gaden(args, session_id, desired_specs, positions, path, rng):
    calibration = _load_local_ppm_calibration(args.local_ppm_calibration)
    if calibration is None:
        raise ValueError("--local-ppm-calibration is required")
    expected = calibration["emission_configuration"]
    active = {
        "emission_concentration_exponent": args.emission_concentration_exponent,
        "emission_scale": args.emission_scale,
        "filaments_per_second": args.filaments_per_second,
        "warmup_s": args.warmup_s,
        "wind_regime": args.wind_regime,
        "buoyancy_tau_s": args.buoyancy_tau_s,
    }
    mismatch = [key for key, value in active.items() if expected.get(key) != value]
    if mismatch and not args.allow_calibration_mismatch:
        raise ValueError("local-ppm calibration mismatch: " + ", ".join(mismatch))

    nominal = [
        _map_desired_local_to_nominal(spec, calibration, allow_extrapolation=True)
        for spec in desired_specs
    ]
    profile = (
        zero_wind_profile(args.warmup_s + len(path) / args.rate + 2)
        if args.wind_regime == "zero"
        else sample_wind_profile(
            rng, args.warmup_s + len(path) / args.rate + 2, args.wind_regime,
        )
    )
    runner_args = argparse.Namespace(
        scenario=args.scenario,
        scenario_config=args.scenario_config,
        runtime_config_dir=args.runtime_config_dir,
        server_log_dir=args.server_log_dir,
        wait_timeout=args.wait_timeout,
        warmup_s=args.warmup_s,
        rate=args.rate,
        emission_model="separated",
        emission_scale=args.emission_scale,
        emission_concentration_exponent=args.emission_concentration_exponent,
        filaments_per_second=args.filaments_per_second,
        background_emission_ratio=1.0,
        buoyancy_tau_s=args.buoyancy_tau_s,
        original_terminal_velocity=args.original_terminal_velocity,
        seed=args.seed,
        parallel_gaden=True,
    )
    ppm = _run_gaden(runner_args, session_id, nominal, positions, path, profile)
    return ppm, nominal, profile


def collect(args: argparse.Namespace) -> None:
    phase_name = args.phase.upper()
    out = Path(args.out_dir or f"datasets/odor_encoder/physical_phase_{phase_name.lower()}_pilot_v1")
    source_config = resolve_scenario(args.scenario, args.scenario_config)
    frame_map = FrameMap.from_scenario(source_config)
    baseline = np.asarray([
        MOXSensor(model=model, rate=args.rate).baseline_voltage() for model in SENSOR_ORDER
    ], np.float32)

    start_index = int(args.start_index)
    stop_index = start_index + int(args.episodes)
    for completed, index in enumerate(range(start_index, stop_index), start=1):
        if args.gas_a is not None:
            gas_a_index = GAS_ORDER.index(args.gas_a)
            gas_b_index = GAS_ORDER.index(args.gas_b)
        else:
            gas_a_index, gas_b_index = scheduled_pair(
                index, args.seed, args.same_gas_fraction,
            )
        gas_a, gas_b = GAS_ORDER[gas_a_index], GAS_ORDER[gas_b_index]
        session_id = f"physical_{phase_name.lower()}_{index:06d}_{gas_a}_to_{gas_b}"
        split = (
            _scheduled_split(index, gas_a_index, gas_b_index)
            if args.split == "auto" else args.split
        )
        destination = out / split / f"{session_id}.npz"
        if destination.is_file() and not args.overwrite:
            continue
        episode_seed = args.seed + PHASE_SEED_OFFSET[phase_name] + index * 1009
        rng = np.random.default_rng(episode_seed)
        (
            cup_a, cup_b, filter_position,
            cup_a_robot, cup_b_robot, filter_robot, assignment,
        ) = _layout(rng, frame_map, args.source_z, args.filter_z)
        source_sampler = (
            _phase_a_desired_ppm if phase_name == "A" else _balanced_desired_ppm
        )
        desired_a = source_sampler(
            index, args.seed, gas_a_index, 0,
            args.concentration_min_ppm, args.concentration_max_ppm,
        )
        desired_b = source_sampler(
            index, args.seed, gas_b_index, 1,
            args.concentration_min_ppm, args.concentration_max_ppm,
        )
        desired_specs = (_pure(gas_a_index, desired_a), _pure(gas_b_index, desired_b))
        plan = build_trajectory(
            phase_name, index, gas_a_index, gas_b_index,
            cup_a, cup_b, filter_position, rng, args,
        )
        # A ROS service timeout or a crashed native server must not discard an
        # otherwise resumable multi-hour collection. Replaying the RNG state
        # keeps every retry physically identical to the original attempt.
        gaden_rng_state = copy.deepcopy(rng.bit_generator.state)
        last_error: Exception | None = None
        for attempt in range(1, args.max_episode_attempts + 1):
            rng.bit_generator.state = copy.deepcopy(gaden_rng_state)
            try:
                ppm, nominal_specs, wind_profile = _run_physical_gaden(
                    args, session_id, desired_specs, np.stack((cup_a, cup_b)),
                    plan.position, rng,
                )
                break
            except (RuntimeError, TimeoutError) as exc:
                last_error = exc
                if attempt == args.max_episode_attempts:
                    raise RuntimeError(
                        f"physical-{phase_name} episode {index} failed after "
                        f"{attempt} attempts; completed episode files remain resumable"
                    ) from exc
                print(
                    f"[physical-{phase_name}] retry episode={index} "
                    f"attempt={attempt}/{args.max_episode_attempts}: {exc}",
                    flush=True,
                )
                time.sleep(args.retry_delay_s)
        else:  # pragma: no cover - the final attempt raises above
            raise RuntimeError("unreachable collection retry state") from last_error
        noise_std = float(rng.uniform(args.min_noise_std_v, args.max_noise_std_v))
        voltage, rs_r0 = simulate_mox(
            ppm, args.rate, measurement_noise_std_v=noise_std, rng=rng,
        )
        labels, physical = physical_labels(
            ppm, rs_r0, rate=args.rate,
            presence_threshold_ppm=args.presence_threshold_ppm,
            gas_class_min_ppm=args.gas_class_min_ppm,
            dominant_threshold=args.dominant_threshold,
            slope_window_s=args.slope_window_s,
            rising_rate_per_s=args.rising_rate_per_s,
            falling_rate_per_s=args.falling_rate_per_s,
            memory_start_index=plan.memory_start_index,
            recovery_residual_threshold=args.recovery_residual_threshold,
        )
        timing = timing_and_endpoints(
            plan, ppm, labels, physical, rate=args.rate, rng=rng,
            jitter_s=args.endpoint_jitter_s,
            presence_threshold=args.presence_threshold_ppm,
            dominance_threshold=args.dominant_threshold,
        )
        length = len(ppm)
        contribution_a = np.zeros_like(ppm)
        contribution_b = np.zeros_like(ppm)
        if gas_a_index != gas_b_index:
            contribution_a[:, gas_a_index] = ppm[:, gas_a_index]
            contribution_b[:, gas_b_index] = ppm[:, gas_b_index]
        else:
            contribution_a[:] = np.nan
            contribution_b[:] = np.nan
        processed = (voltage - baseline) / np.maximum(baseline, 1e-4)
        repeated = lambda value, dtype=None: np.full(length, value, dtype=dtype)
        extra = {
            **physical,
            **timing,
            "timestamp": np.arange(length, dtype=np.float32) / args.rate,
            "enose_position_xyz": plan.position.astype(np.float32),
            "episode_id": repeated(session_id, "U96"),
            "trajectory_id": repeated(session_id, "U96"),
            "split_id": repeated(split, "U12"),
            "random_seed": repeated(episode_seed, np.int64),
            "cup_a_position_xyz": np.repeat(cup_a[None], length, axis=0).astype(np.float32),
            "cup_b_position_xyz": np.repeat(cup_b[None], length, axis=0).astype(np.float32),
            "gas_a": repeated(gas_a, "U24"),
            "gas_b": repeated(gas_b, "U24"),
            "nominal_source_condition_a": np.repeat(
                nominal_specs[0].concentration_ppm[None], length, axis=0,
            ).astype(np.float32),
            "nominal_source_condition_b": np.repeat(
                nominal_specs[1].concentration_ppm[None], length, axis=0,
            ).astype(np.float32),
            "local_ppm_from_cup_a": contribution_a,
            "local_ppm_from_cup_b": contribution_b,
            "raw_mox_8ch": voltage.astype(np.float32),
            "processed_mox_8ch": processed.astype(np.float32),
            "mox_internal_rs_r0": rs_r0.astype(np.float32),
            "source_realization_id": repeated(f"{session_id}:{episode_seed}", "U128"),
            "filament_count": repeated(-1, np.int64),
        }
        for gas_index, gas in enumerate(GAS_ORDER):
            extra[f"local_ppm_{gas}"] = ppm[:, gas_index].astype(np.float32)

        metadata = {
            "schema_version": 4,
            "pipeline": "physical_two_cup_v1",
            "episode_id": session_id,
            "trajectory_id": session_id,
            "seed": episode_seed,
            "gas_a": gas_a,
            "gas_b": gas_b,
            "ordered_pair": f"{gas_a}->{gas_b}",
            "same_gas_control": gas_a == gas_b,
            "cup_assignment": assignment,
            "cup_a_position_gaden_m": cup_a.tolist(),
            "cup_b_position_gaden_m": cup_b.tolist(),
            "filter_position_gaden_m": filter_position.tolist(),
            "cup_a_position_robosuite_m": cup_a_robot.tolist(),
            "cup_b_position_robosuite_m": cup_b_robot.tolist(),
            "filter_position_robosuite_m": filter_robot.tolist(),
            "desired_local_ppm_a": desired_a,
            "desired_local_ppm_b": desired_b,
            "nominal_source_condition_a": nominal_specs[0].concentration_ppm.tolist(),
            "nominal_source_condition_b": nominal_specs[1].concentration_ppm.tolist(),
            "trajectory_type": plan.trajectory_type,
            "source_realization_id": f"{session_id}:{episode_seed}",
            "both_sources_continuously_active": True,
            "physical_gaden_motion": True,
            "no_sensor_reset_during_sequence": True,
            "filter_forces_zero_ppm": False,
            "wind_regime": args.wind_regime,
            "mean_wind_speed_m_s": float(np.linalg.norm(wind_profile.mean_vector_m_s)),
            "emission_scale": args.emission_scale,
            "emission_concentration_exponent": args.emission_concentration_exponent,
            "filaments_per_second": args.filaments_per_second,
            "warmup_s": args.warmup_s,
            "buoyancy_tau_s": args.buoyancy_tau_s,
            "original_terminal_velocity_enabled": args.original_terminal_velocity,
            "ros_domain_id": args.ros_domain_id,
            "presence_threshold_ppm": args.presence_threshold_ppm,
            "gas_class_min_ppm": args.gas_class_min_ppm,
            "dominant_threshold": args.dominant_threshold,
            "slope_window_s": args.slope_window_s,
            "rising_rate_per_s": args.rising_rate_per_s,
            "falling_rate_per_s": args.falling_rate_per_s,
            "recovery_residual_threshold": args.recovery_residual_threshold,
            "measurement_noise_std_v": noise_std,
            "initial_mox_rs_r0": rs_r0[0].tolist(),
            "sniff_events": [
                {
                    "role": event.role,
                    "gas": GAS_ORDER[event.gas_index],
                    "search_start_index": event.search_start,
                    "dwell_end_index": event.dwell_end,
                    "duration_s": (event.dwell_end - event.search_start) / args.rate,
                }
                for event in plan.sniff_events
            ],
            "endpoint_count": int(timing["endpoint_eligible"].sum()),
            "gas_supervised_fraction": float(labels["gas_loss_mask"].mean()),
        }
        write_episode(
            out, session_id=session_id, split=split,
            domain=f"physical_phase_{phase_name.lower()}", rate=args.rate,
            voltage=voltage, rs_r0=rs_r0, ppm=ppm,
            time_s=np.arange(length, dtype=np.float32) / args.rate,
            position=plan.position, phase=plan.phase, labels=labels,
            metadata=metadata, extra_arrays=extra, class_order=GAS_ORDER,
        )
        if completed % args.log_every == 0 or completed == args.episodes:
            print(
                f"[physical-{phase_name}] {completed}/{args.episodes} "
                f"index={index} "
                f"{gas_a}->{gas_b} type={plan.trajectory_type} "
                f"endpoints={int(timing['endpoint_eligible'].sum())}"
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("A", "B", "C", "D", "a", "b", "c", "d"))
    parser.add_argument("--episodes", type=int, required=True)
    parser.add_argument(
        "--start-index", type=int, default=0,
        help=(
            "first episode index; --episodes is the number generated from this "
            "index (default: start at zero, preserving resumable collection)"
        ),
    )
    parser.add_argument("--out-dir")
    parser.add_argument("--rate", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=51000)
    parser.add_argument("--split", choices=("auto", "train", "validation", "test"), default="auto")
    parser.add_argument("--same-gas-fraction", type=float, default=0.0)
    parser.add_argument("--gas-a", choices=GAS_ORDER)
    parser.add_argument("--gas-b", choices=GAS_ORDER)
    parser.add_argument("--concentration-min-ppm", type=float, default=1.0)
    parser.add_argument("--concentration-max-ppm", type=float, default=100.0)
    parser.add_argument("--local-ppm-calibration", default=str(DEFAULT_CALIBRATION))
    parser.add_argument("--allow-calibration-mismatch", action="store_true")
    parser.add_argument("--emission-concentration-exponent", type=float, default=0.5)
    parser.add_argument("--emission-scale", type=float, default=0.06)
    parser.add_argument("--filaments-per-second", type=int, default=15)
    parser.add_argument("--wind-regime", choices=("zero", *WIND_REGIMES), default="zero")
    parser.add_argument("--warmup-s", type=float, default=8.0)
    parser.add_argument("--buoyancy-tau-s", type=float, default=0.0)
    parser.add_argument("--original-terminal-velocity", action="store_true")
    parser.add_argument("--scenario", default="10x6_uniform")
    parser.add_argument("--scenario-config", default="config1")
    parser.add_argument("--runtime-config-dir", default=".runtime/odor_encoder_physical")
    parser.add_argument("--server-log-dir", default=".roslog/odor_encoder_physical")
    parser.add_argument(
        "--ros-domain-id", type=int,
        help="isolated ROS domain; defaults to A=81, B=82, C=83, D=84",
    )
    parser.add_argument("--wait-timeout", type=float, default=60.0)
    parser.add_argument("--max-episode-attempts", type=int, default=3)
    parser.add_argument("--retry-delay-s", type=float, default=2.0)
    parser.add_argument("--source-z", type=float, default=DEFAULT_SOURCE_Z)
    parser.add_argument("--filter-z", type=float, default=DEFAULT_FILTER_Z)
    parser.add_argument("--min-sniff-height-m", type=float, default=0.01)
    parser.add_argument("--max-sniff-height-m", type=float, default=0.05)
    parser.add_argument("--max-lateral-offset-m", type=float, default=0.02)
    parser.add_argument("--min-approach-s", type=float, default=1.0)
    parser.add_argument("--max-approach-s", type=float, default=4.0)
    parser.add_argument("--min-descend-s", type=float, default=0.5)
    parser.add_argument("--max-descend-s", type=float, default=2.0)
    parser.add_argument("--min-transit-s", type=float, default=1.0)
    parser.add_argument("--max-transit-s", type=float, default=7.0)
    parser.add_argument("--min-sniff-s", type=float, default=5.0)
    parser.add_argument("--max-sniff-s", type=float, default=12.0)
    parser.add_argument("--min-static-s", type=float, default=12.0)
    parser.add_argument("--max-static-s", type=float, default=25.0)
    parser.add_argument(
        "--recovery-durations-s", type=float, nargs="+",
        default=(0.0, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0),
    )
    parser.add_argument("--direct-transfer-fraction", type=float, default=0.25)
    parser.add_argument("--dwell-jitter-m", type=float, default=0.002)
    parser.add_argument("--min-noise-std-v", type=float, default=0.0)
    parser.add_argument("--max-noise-std-v", type=float, default=0.01)
    parser.add_argument(
        "--presence-threshold-ppm", type=float, default=5.0,
        help="minimum local total ppm treated as meaningful VOC exposure",
    )
    parser.add_argument("--gas-class-min-ppm", type=float, default=5.0)
    parser.add_argument("--dominant-threshold", type=float, default=0.90)
    parser.add_argument("--slope-window-s", type=float, default=0.5)
    parser.add_argument("--rising-rate-per-s", type=float, default=0.03)
    parser.add_argument("--falling-rate-per-s", type=float, default=0.08)
    parser.add_argument("--recovery-residual-threshold", type=float, default=0.02)
    parser.add_argument("--endpoint-jitter-s", type=float, default=0.10)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--log-every", type=int, default=10)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.phase = args.phase.upper()
    if args.ros_domain_id is None:
        args.ros_domain_id = PHASE_ROS_DOMAINS[args.phase]
    if not 0 <= args.ros_domain_id <= 232:
        raise SystemExit("--ros-domain-id must be between 0 and 232")
    os.environ["ROS_DOMAIN_ID"] = str(args.ros_domain_id)
    if args.episodes < 1 or args.start_index < 0 or args.rate <= 0:
        raise SystemExit("episodes and rate must be positive")
    if args.max_episode_attempts < 1 or args.retry_delay_s < 0:
        raise SystemExit("retry attempts must be positive and delay must be non-negative")
    if (args.gas_a is None) != (args.gas_b is None):
        raise SystemExit("--gas-a and --gas-b must be supplied together")
    if not 0 <= args.same_gas_fraction < 1:
        raise SystemExit("--same-gas-fraction must be in [0, 1)")
    if not 0 < args.concentration_min_ppm < args.concentration_max_ppm:
        raise SystemExit("invalid concentration range")
    if not 0 < args.dominant_threshold <= 1:
        raise SystemExit("--dominant-threshold must be in (0, 1]")
    if args.original_terminal_velocity and args.buoyancy_tau_s > 0:
        raise SystemExit("terminal velocity and experimental buoyancy are mutually exclusive")
    if not 0 <= args.direct_transfer_fraction <= 1:
        raise SystemExit("--direct-transfer-fraction must be in [0, 1]")
    if not 0 <= args.min_noise_std_v <= args.max_noise_std_v:
        raise SystemExit("invalid measurement-noise range")
    collect(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

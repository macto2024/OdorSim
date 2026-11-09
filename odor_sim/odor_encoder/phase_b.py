"""Phase B: robot-free scripted virtual e-nose trajectories through GADEN."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace

import numpy as np

from odor_sim.config.frame_map import FrameMap
from odor_sim.envs.odor_profile import OdorProfile
from odor_sim.envs.registry import resolve_scenario
from odor_sim.envs.scene_builder import SceneBuilder
from odor_sim.odor_encoder.dataset import (
    GAS_ORDER, MixtureSpec, select_split, simulate_mox,
    source_labels, write_episode,
)
from odor_sim.odor_encoder.wind import (
    WIND_REGIMES, WIND_REGIME_WEIGHTS, sample_wind_profile,
    write_isolated_config, zero_wind_profile,
)

PATH_NAMES = (
    "approach_descend_dwell_retreat", "approach_dwell_retreat",
    "stationary_downwind", "crosswind_sweep",
)
PATH_WEIGHTS = np.asarray([0.45, 0.35, 0.10, 0.10])
SCHEDULE_BLOCK_SIZE = 400
WIND_SCHEDULE_BLOCK_SIZE = 20


def scheduled_target(index: int, seed: int) -> tuple[int | None, str]:
    """Return a stable, balanced clean/gas/path target for one episode index.

    Every complete 400-episode block has 10% clean sessions. Each gas has 60
    non-clean sessions split 27/21/6/6 across the four path types.
    """
    block, slot = divmod(index, SCHEDULE_BLOCK_SIZE)
    schedule: list[tuple[int | None, str]] = []
    path_counts = (27, 21, 6, 6)
    for gas_index in range(len(GAS_ORDER)):
        for path, count in zip(PATH_NAMES, path_counts):
            schedule.extend([(gas_index, path)] * count)
    clean_path_counts = (18, 14, 4, 4)
    for path, count in zip(PATH_NAMES, clean_path_counts):
        schedule.extend([(None, path)] * count)
    rng = np.random.default_rng(seed + block * 104729)
    rng.shuffle(schedule)
    return schedule[slot]


def scheduled_wind_regime(index: int, seed: int) -> str:
    """Return an exact 40/45/15 regime schedule in every 20 episodes."""
    block, slot = divmod(index, WIND_SCHEDULE_BLOCK_SIZE)
    schedule = [WIND_REGIMES[0]] * 8 + [WIND_REGIMES[1]] * 9 + [WIND_REGIMES[2]] * 3
    rng = np.random.default_rng(seed + 7919 + block * 65537)
    rng.shuffle(schedule)
    return schedule[slot]


def _scheduled_source(
    gas_index: int | None, rng: np.random.Generator,
    concentration_sampling: str = "log_uniform",
    concentration_stratum: int | None = None,
    concentration_min_ppm: float = 1.0,
    concentration_max_ppm: float = 100.0,
) -> MixtureSpec:
    vector = np.zeros(len(GAS_ORDER), np.float32)
    if gas_index is None:
        return MixtureSpec(vector, "clean")
    if concentration_sampling == "log_uniform":
        vector[gas_index] = np.exp(rng.uniform(
            np.log(concentration_min_ppm), np.log(concentration_max_ppm),
        ))
    elif concentration_sampling == "linear_uniform":
        vector[gas_index] = rng.uniform(concentration_min_ppm, concentration_max_ppm)
    elif concentration_sampling == "stratified_log":
        boundaries = np.asarray((1.0, 10.0, 30.0, 60.0, 100.0))
        boundaries = np.clip(boundaries, concentration_min_ppm, concentration_max_ppm)
        intervals = [
            (low, high) for low, high in zip(boundaries[:-1], boundaries[1:])
            if high > low
        ]
        if not intervals:
            raise ValueError("concentration range does not contain a sampling interval")
        low, high = intervals[(concentration_stratum or 0) % len(intervals)]
        vector[gas_index] = np.exp(rng.uniform(np.log(low), np.log(high)))
    else:
        raise ValueError(f"unknown concentration sampling {concentration_sampling!r}")
    return MixtureSpec(vector, "single_gas")


def _load_local_ppm_calibration(path: str | None) -> dict | None:
    if not path:
        return None
    calibration = json.loads(Path(path).read_text())
    if calibration.get("model") != "log_linear_nominal_to_local_median":
        raise ValueError("unsupported local-ppm calibration model")
    return calibration


def _map_desired_local_to_nominal(
    desired: MixtureSpec, calibration: dict, *, allow_extrapolation: bool = False,
) -> MixtureSpec:
    vector = np.zeros_like(desired.concentration_ppm)
    for index in np.flatnonzero(desired.concentration_ppm):
        local_ppm = float(desired.concentration_ppm[index])
        nominal = np.exp(
            (np.log(local_ppm) - float(calibration["log_intercept"]))
            / float(calibration["log_slope"])
        )
        if allow_extrapolation:
            vector[index] = nominal
        else:
            low, high = calibration["nominal_support_ppm"]
            vector[index] = np.clip(nominal, low, high)
    return MixtureSpec(vector, desired.evaluation_group)


def _segments(points: list[np.ndarray], counts: list[int]) -> np.ndarray:
    rows = []
    for a, b, count in zip(points[:-1], points[1:], counts):
        rows.append(np.linspace(a, b, max(1, count), endpoint=False))
    rows.append(points[-1][None])
    return np.concatenate(rows, axis=0)


def scripted_path(
    name: str, steps: int, source: np.ndarray, sniff_distance_m: float = 0.03,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a bounded path and its semantic phase per timestep."""
    s = np.asarray(source, float)
    yaw = 0.0 if rng is None else float(rng.uniform(-np.pi, np.pi))
    rotation = np.asarray([
        [np.cos(yaw), -np.sin(yaw), 0.0],
        [np.sin(yaw), np.cos(yaw), 0.0],
        [0.0, 0.0, 1.0],
    ])
    horizontal_sniff = np.zeros(3) if rng is None else np.asarray([
        rng.uniform(0.0, 0.02), 0.0, 0.0,
    ]) @ rotation.T
    sniff_point = s + horizontal_sniff + [0.0, 0.0, sniff_distance_m]
    if name == "stationary_downwind":
        path = np.repeat(sniff_point[None], steps, axis=0)
        phases = np.full(steps, "stationary")
    elif name == "crosswind_sweep":
        span = 0.30 if rng is None else float(rng.uniform(0.18, 0.42))
        start = s + np.asarray([0.0, -span, sniff_distance_m]) @ rotation.T
        end = s + np.asarray([0.0, span, sniff_distance_m]) @ rotation.T
        center = sniff_point
        sweep = int(0.55 * steps)
        recenter = int(0.20 * steps)
        path = _segments(
            [start, end, center, center],
            [sweep, recenter, max(1, steps - sweep - recenter - 1)],
        )
        phases = np.concatenate((
            np.full(sweep, "crosswind_outbound"),
            np.full(recenter, "crosswind_return"),
            np.full(max(1, steps - sweep - recenter), "dwell"),
        ))
    elif name in ("approach_dwell_retreat", "approach_descend_dwell_retreat"):
        approach_fraction = 0.30 if rng is None else float(rng.uniform(0.20, 0.38))
        dwell_fraction = 0.45 if rng is None else float(rng.uniform(0.35, 0.55))
        n1, n2 = int(approach_fraction * steps), int(dwell_fraction * steps)
        distance = 2.0 if rng is None else float(rng.uniform(0.45, 1.20))
        lateral = 0.35 if rng is None else float(rng.uniform(-0.35, 0.35))
        start_height = 0.45 if rng is None else float(rng.uniform(0.20, 0.50))
        start = s + np.asarray([distance, lateral, start_height]) @ rotation.T
        hover = s + np.asarray([0.0, 0.0, rng.uniform(0.10, 0.22) if rng is not None else 0.15])
        sniff = sniff_point
        if name == "approach_descend_dwell_retreat":
            descend = max(1, int(0.12 * steps))
            retreat = max(1, steps - n1 - n2 - descend - 1)
            points = [start, hover, sniff, sniff, start]
            counts = [n1, descend, n2, retreat]
            phases = np.concatenate((
                np.full(n1, "approach"), np.full(descend, "descend"),
                np.full(n2, "dwell"), np.full(retreat + 1, "retreat"),
            ))
        else:
            retreat = max(1, steps - n1 - n2 - 1)
            points = [start, sniff, sniff, start]
            counts = [n1, n2, retreat]
            phases = np.concatenate((
                np.full(n1, "approach"), np.full(n2, "dwell"),
                np.full(retreat + 1, "retreat"),
            ))
        path = _segments(points, counts)
        if rng is not None and n1 > 2:
            curve = float(rng.uniform(-0.10, 0.10)) * np.sin(np.linspace(0.0, np.pi, min(n1, len(path))))
            lateral_axis = np.asarray([-np.sin(yaw), np.cos(yaw), 0.0])
            path[:len(curve)] += curve[:, None] * lateral_axis
    else:
        raise ValueError(name)
    path = path[:steps] if len(path) >= steps else np.vstack([path, np.repeat(path[-1:], steps - len(path), 0)])
    phases = phases[:steps] if len(phases) >= steps else np.concatenate((
        phases, np.repeat(phases[-1:], steps - len(phases)),
    ))
    return path.astype(np.float32), phases


def _emission_strength(ppm: float) -> float:
    """Map the 1-100 ppm source setting smoothly onto the usable source range."""
    return float(np.clip(0.15 + 0.85 * np.log10(max(ppm, 1.0)) / 2.0, 0.15, 1.0))


def _emission_parameters(
    ppm: float, *, model: str, emission_scale: float,
    concentration_exponent: float, filaments_per_second: int,
    background_ratio: float,
) -> dict[str, float | int]:
    if model == "legacy_coupled":
        strength = _emission_strength(ppm)
        return {
            "strength": strength,
            "filament_ppm_center": 5.0 + 35.0 * strength,
            "num_filaments_sec": int(round(5.0 + 35.0 * strength)),
        }
    return {
        # Scale only filament concentration. Lambda stays independent, making
        # total emission respond approximately linearly to emission_scale.
        "strength": 0.5,
        "filament_ppm_center": (
            emission_scale * 10.0 * (ppm / 10.0) ** concentration_exponent
            * background_ratio
        ),
        "num_filaments_sec": filaments_per_second,
    }


def _object(
    name: str, spec: MixtureSpec, *, model: str = "legacy_coupled",
    emission_scale: float = 1.0, filaments_per_second: int = 20,
    concentration_exponent: float = 1.0, background_ratio: float = 1.0,
):
    entries = []
    for gas, ppm in zip(GAS_ORDER, spec.concentration_ppm):
        if ppm > 0:
            parameters = _emission_parameters(
                float(ppm), model=model, emission_scale=emission_scale,
                concentration_exponent=concentration_exponent,
                filaments_per_second=filaments_per_second,
                background_ratio=background_ratio,
            )
            entries.append({
                "gas_type": gas,
                "strength": parameters["strength"],
                "filament_ppm_center": parameters["filament_ppm_center"],
                "num_filaments_sec": parameters["num_filaments_sec"],
            })
    return SimpleNamespace(object_id=name, odor_profile=OdorProfile.from_spec(entries))


def _background_sources(
    rng: np.random.Generator, primary: MixtureSpec, max_gases: int,
) -> list[MixtureSpec]:
    """Create ordinary neighboring cup sources; local dominance defines nuisance."""
    if not primary.gases or max_gases < 1:
        return []
    primary_index = int(np.flatnonzero(primary.concentration_ppm)[0])
    candidates = np.delete(np.arange(len(GAS_ORDER)), primary_index)
    count = int(rng.integers(1, min(max_gases, len(candidates)) + 1))
    sources = []
    for gas_index in rng.choice(candidates, count, replace=False):
        vector = np.zeros(len(GAS_ORDER), np.float32)
        vector[gas_index] = 10.0 ** rng.uniform(0.0, 2.0)
        sources.append(MixtureSpec(vector, "background_source"))
    return sources


def sample_background_offset(
    rng: np.random.Generator, min_distance: float, max_distance: float,
    wind_direction: np.ndarray | None = None, orientation: str | None = None,
) -> np.ndarray:
    """Place a neighbor at an exact separation relative to mean wind."""
    distance = float(rng.uniform(min_distance, max_distance))
    if wind_direction is None:
        angle = float(rng.uniform(-np.pi, np.pi))
        return np.asarray([distance * np.cos(angle), distance * np.sin(angle), 0.0])
    direction = np.asarray(wind_direction, float).copy()
    direction[2] = 0.0
    direction /= np.linalg.norm(direction)
    orientation = orientation or str(rng.choice(
        ("upwind", "crosswind", "downwind"), p=(0.25, 0.50, 0.25),
    ))
    if orientation == "upwind":
        offset_direction = -direction
    elif orientation == "downwind":
        offset_direction = direction
    elif orientation == "crosswind":
        side = -1.0 if rng.random() < 0.5 else 1.0
        offset_direction = side * np.asarray([-direction[1], direction[0], 0.0])
    else:
        raise ValueError(f"unknown orientation {orientation!r}")
    return distance * offset_direction


def _primary_evidence(
    spec: MixtureSpec, ppm: np.ndarray, *, min_target_ppm: float,
    max_target_ppm: float, max_total_ppm: float,
    min_target_fraction: float, rate: float, evidence_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    ppm = np.asarray(ppm, np.float32)
    if not spec.gases:
        return np.ones(len(ppm), bool), np.zeros(len(ppm), np.float32)
    target_index = int(np.flatnonzero(spec.concentration_ppm)[0])
    target = ppm[:, target_index]
    total = ppm.sum(1)
    fraction = np.divide(target, total, out=np.zeros_like(target), where=total > 0)
    evidence_steps = max(1, int(round(evidence_s * rate)))
    if evidence_steps == 1:
        mean_target = target
        mean_total = total
    else:
        kernel = np.ones(evidence_steps, np.float64)
        mean_target = np.convolve(target, kernel, mode="full")[:len(target)] / evidence_steps
        mean_total = np.convolve(total, kernel, mode="full")[:len(total)] / evidence_steps
    mean_fraction = np.divide(
        mean_target, mean_total, out=np.zeros_like(mean_target), where=mean_total > 0,
    )
    evidence = (
        (np.arange(len(ppm)) >= evidence_steps - 1)
        & (mean_target >= min_target_ppm)
        & (mean_target <= max_target_ppm)
        & (mean_total <= max_total_ppm)
        & (mean_fraction >= min_target_fraction)
    )
    return evidence, fraction


def _rising_or_stable_mask(
    target_ppm: np.ndarray, *, rate: float, trend_s: float,
    max_decline_fraction: float,
) -> np.ndarray:
    """Mask sustained concentration decline while tolerating filament noise."""
    target = np.asarray(target_ppm, np.float64)
    if trend_s <= 0:
        return np.ones(len(target), bool)
    steps = max(1, int(round(trend_s * rate)))
    smooth_steps = max(1, int(round(min(0.5, trend_s) * rate)))
    kernel = np.ones(smooth_steps, np.float64) / smooth_steps
    smooth = np.convolve(target, kernel, mode="full")[:len(target)]
    mask = np.ones(len(target), bool)
    if len(target) > steps:
        previous = smooth[:-steps]
        current = smooth[steps:]
        minimum_allowed = previous * (1.0 - max_decline_fraction)
        mask[steps:] = current >= minimum_allowed
    return mask


def _elapsed_while_true(mask: np.ndarray, rate: float) -> np.ndarray:
    elapsed = np.full(len(mask), np.nan, np.float32)
    count = 0
    for index, active in enumerate(mask):
        if active:
            elapsed[index] = count / rate
            count += 1
        else:
            count = 0
    return elapsed


def observable_primary_mask(
    spec: MixtureSpec, ppm: np.ndarray, *, min_target_ppm: float,
    min_target_fraction: float, rate: float, recovery_s: float,
    max_target_ppm: float = float("inf"),
    max_total_ppm: float = float("inf"),
    evidence_s: float = 0.0,
    trend_s: float = 0.0,
    max_decline_fraction: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return endpoints with sustained, calibrated primary-gas evidence."""
    evidence, fraction = _primary_evidence(
        spec, ppm, min_target_ppm=min_target_ppm,
        max_target_ppm=max_target_ppm, max_total_ppm=max_total_ppm,
        min_target_fraction=min_target_fraction, rate=rate, evidence_s=evidence_s,
    )
    total = np.asarray(ppm, np.float32).sum(1)
    if spec.gases:
        target_index = int(np.flatnonzero(spec.concentration_ppm)[0])
        evidence &= _rising_or_stable_mask(
            ppm[:, target_index], rate=rate, trend_s=trend_s,
            max_decline_fraction=max_decline_fraction,
        )
    valid = evidence.copy()

    # Preserve a short clean early-recovery interval because MOX response lags
    # behind plume disappearance. Competing-plume intervals remain invalid.
    recovery_steps = max(0, int(round(recovery_s * rate)))
    last_evidence = -recovery_steps - 1
    for index in range(len(valid)):
        if evidence[index]:
            last_evidence = index
        elif index - last_evidence <= recovery_steps and total[index] < min_target_ppm:
            valid[index] = True
    return valid, fraction


def _run_gaden(
    args, session_id: str, specs: list[MixtureSpec], positions: np.ndarray,
    path: np.ndarray, wind_profile=None,
) -> np.ndarray:
    from odor_sim.bridge.gaden_bridge import GadenBridge
    from odor_sim.runtime.gaden_server import (
        GadenServerManager, IsolatedGadenServerManager,
    )

    source_config_dir = resolve_scenario(args.scenario, args.scenario_config)
    if wind_profile is None:
        rng = np.random.default_rng(getattr(args, "seed", 0))
        duration_s = getattr(args, "warmup_s", 0.0) + len(path) / getattr(args, "rate", 20.0)
        wind_profile = sample_wind_profile(rng, duration_s, "uniform_weak_indoor")
    runtime_root = Path(getattr(args, "runtime_config_dir", ".runtime/phase_b"))
    runtime_root.mkdir(parents=True, exist_ok=True)
    temp = tempfile.TemporaryDirectory(prefix=f"{session_id}_", dir=runtime_root)
    config_dir = write_isolated_config(source_config_dir, Path(temp.name) / "config", wind_profile)
    builder = SceneBuilder(frame_map=FrameMap.from_scenario(source_config_dir))
    for i, spec in enumerate(specs):
        builder.add_object(_object(
            f"virtual_source_{i}", spec,
            model=getattr(args, "emission_model", "legacy_coupled"),
            emission_scale=float(getattr(args, "emission_scale", 1.0)),
            concentration_exponent=float(
                getattr(args, "emission_concentration_exponent", 1.0)
            ),
            filaments_per_second=int(getattr(args, "filaments_per_second", 20)),
            background_ratio=(
                float(getattr(args, "background_emission_ratio", 1.0))
                if i > 0 else 1.0
            ),
        ))
    expanded = np.concatenate([
        np.repeat(positions[i:i + 1], len(spec.gases), axis=0) for i, spec in enumerate(specs)
    ])
    scene_id = session_id
    builder.export_gaden_scene(
        config_dir,
        scene_id=scene_id,
        initial_positions_gaden=expanded,
        buoyancy_tau_s=float(getattr(args, "buoyancy_tau_s", 1.0)),
        original_terminal_velocity_enabled=bool(
            getattr(args, "original_terminal_velocity", False)
        ),
    )
    parallel = bool(getattr(args, "parallel_gaden", False))
    if parallel:
        worker_id = f"domain{os.environ.get('ROS_DOMAIN_ID', 'unset')}_pid{os.getpid()}"
        server = IsolatedGadenServerManager(
            config_dir, scene_id=scene_id, log_dir=args.server_log_dir,
            worker_id=worker_id,
        )
    else:
        server = GadenServerManager(
            config_dir, scene_id=scene_id, log_dir=args.server_log_dir,
        )
    bridge = None
    try:
        server.start(timeout=args.wait_timeout, kill_stale=not parallel)
        bridge = GadenBridge()
        if not bridge.wait_for_server(args.wait_timeout):
            raise RuntimeError("GADEN /odor_value service unavailable")
        bridge.reset_time()
        for _ in range(max(0, int(round(args.warmup_s * args.rate)))):
            bridge.publish_source_poses(expanded)
            bridge.step()
        query_path = np.asarray(path, np.float32)
        single_point = query_path.ndim == 2
        if single_point:
            query_path = query_path[:, None, :]
        if query_path.ndim != 3 or query_path.shape[2] != 3:
            raise ValueError("path must have shape (time, 3) or (time, points, 3)")
        ppm = np.zeros((len(query_path), query_path.shape[1], len(GAS_ORDER)), np.float32)
        for t, points in enumerate(query_path):
            bridge.publish_source_poses(expanded)
            bridge.step()
            readings = bridge.query_ppm(points)
            for point_index, reading in enumerate(readings):
                ppm[t, point_index] = [float(reading.get(gas, 0.0)) for gas in GAS_ORDER]
        return ppm[:, 0] if single_point else ppm
    finally:
        if bridge is not None:
            bridge.close()
        server.stop()
        temp.cleanup()


def collect(args: argparse.Namespace) -> None:
    scenario_config = resolve_scenario(args.scenario, args.scenario_config)
    frame_map = FrameMap.from_scenario(scenario_config)
    base_source = frame_map.robosuite_to_gaden([0.0, 0.0, 0.8])
    local_calibration = _load_local_ppm_calibration(args.local_ppm_calibration)
    if local_calibration is not None:
        fitted = local_calibration["emission_configuration"]
        active = {
            "emission_concentration_exponent": args.emission_concentration_exponent,
            "emission_scale": args.emission_scale,
            "filaments_per_second": args.filaments_per_second,
            "warmup_s": args.warmup_s,
            "wind_regime": args.wind_regime,
            "buoyancy_tau_s": args.buoyancy_tau_s,
        }
        mismatched = [
            name for name, value in active.items() if fitted.get(name) != value
        ]
        if mismatched:
            raise ValueError(
                "local-ppm calibration does not match active emission configuration: "
                + ", ".join(mismatched)
            )
    index = 0
    episode_attempt = 0
    attempted_by_path = {name: 0 for name in PATH_NAMES}
    accepted_by_path = {name: 0 for name in PATH_NAMES}
    while index < args.trajectories:
        if episode_attempt >= args.max_attempt_multiplier:
            raise RuntimeError(
                f"could not collect valid trajectory {index} after {episode_attempt} attempts; "
                "inspect plume placement or relax observability thresholds"
            )
        session_id = f"virtual_{index:06d}"
        split = select_split(session_id, args.seed, args.split)
        if (Path(args.out_dir) / split / f"{session_id}.npz").is_file():
            index += 1
            episode_attempt = 0
            continue
        episode_seed = args.seed + index * args.max_attempt_multiplier + episode_attempt
        episode_attempt += 1
        rng = np.random.default_rng(episode_seed)
        source = base_source + rng.uniform([-0.4, -0.6, 0.0], [0.4, 0.6, 0.0])
        gas_index, name = scheduled_target(index, args.seed)
        concentration_stratum = (
            sum(
                scheduled_target(previous, args.seed)[0] == gas_index
                for previous in range(index)
            )
            if gas_index is not None else None
        )
        desired_spec = _scheduled_source(
            gas_index, rng, args.nominal_concentration_sampling,
            concentration_stratum=concentration_stratum,
            concentration_min_ppm=args.concentration_min_ppm,
            concentration_max_ppm=args.concentration_max_ppm,
        )
        spec = (
            _map_desired_local_to_nominal(
                desired_spec, local_calibration,
                allow_extrapolation=args.allow_local_ppm_extrapolation,
            )
            if local_calibration is not None and desired_spec.gases else desired_spec
        )
        wind_regime = (
            scheduled_wind_regime(index, args.seed)
            if args.wind_regime == "sample" else args.wind_regime
        )
        wind_profile = (
            zero_wind_profile(args.warmup_s + args.max_duration_s)
            if wind_regime == "zero"
            else sample_wind_profile(rng, args.warmup_s + args.max_duration_s, wind_regime)
        )
        if spec.gases:
            attempted_by_path[name] += 1
        specs, positions = [spec], source[None]
        background_orientations = []
        if spec.gases and rng.random() < args.background_probability:
            for background in _background_sources(rng, spec, args.background_max_gases):
                if wind_regime == "zero":
                    orientation = "radial_zero_wind"
                    offset = sample_background_offset(
                        rng, args.background_min_distance, args.background_max_distance,
                    )
                else:
                    orientation = str(rng.choice(
                        ("upwind", "crosswind", "downwind"), p=(0.25, 0.50, 0.25),
                    ))
                    offset = sample_background_offset(
                        rng, args.background_min_distance, args.background_max_distance,
                        wind_profile.nominal_direction, orientation,
                    )
                specs.append(background)
                positions = np.vstack((positions, source + offset))
                background_orientations.append(orientation)
        steps = max(2, int(round(rng.uniform(args.min_duration_s, args.max_duration_s) * args.rate)))
        sniff_distance_m = float(rng.uniform(args.min_sniff_distance, args.max_sniff_distance))
        path, phase = scripted_path(name, steps, source, sniff_distance_m, rng=rng)
        if not spec.gases:
            ppm = np.zeros((steps, 6), np.float32)
        else:
            ppm = _run_gaden(args, session_id, specs, positions, path, wind_profile)
        valid, target_fraction = observable_primary_mask(
            spec, ppm, min_target_ppm=args.min_target_ppm,
            min_target_fraction=args.min_target_fraction, rate=args.rate,
            recovery_s=args.recovery_valid_s,
            max_target_ppm=args.max_target_ppm,
            max_total_ppm=args.max_total_ppm,
            evidence_s=args.min_evidence_s,
            trend_s=args.trend_window_s,
            max_decline_fraction=args.max_decline_fraction,
        )
        first_usable_endpoint = int(round(args.required_history_s * args.rate)) - 1
        satisfies_training_criteria = bool(
            not spec.gases
            or np.any(np.flatnonzero(valid) >= first_usable_endpoint)
        )
        if spec.gases and not satisfies_training_criteria:
            late = np.arange(steps) >= first_usable_endpoint
            diagnostic_target = ppm[:, int(np.flatnonzero(spec.concentration_ppm)[0])]
            disposition = "save calibration sample" if args.calibration_mode else "reject"
            print(
                f"[phase-b] {disposition} attempt={episode_attempt} path={name}: "
                "primary gas was not observable at a usable endpoint "
                f"(late_target_max={float(diagnostic_target[late].max()):.4g} ppm, "
                f"late_fraction_max={float(target_fraction[late].max()):.3f})"
            )
            if not args.calibration_mode:
                continue
        if spec.gases and satisfies_training_criteria:
            accepted_by_path[name] += 1
        noise_std_v = float(rng.uniform(args.min_noise_std_v, args.max_noise_std_v))
        voltage, rs_r0 = simulate_mox(
            ppm, args.rate, measurement_noise_std_v=noise_std_v, rng=rng,
        )
        if spec.gases:
            target_index = int(np.flatnonzero(spec.concentration_ppm)[0])
            target_ppm = ppm[:, target_index].astype(np.float32)
        else:
            target_ppm = np.zeros(steps, np.float32)
        dominant, _ = observable_primary_mask(
            spec, ppm, min_target_ppm=args.min_target_ppm,
            max_target_ppm=args.max_target_ppm,
            max_total_ppm=args.max_total_ppm,
            min_target_fraction=args.min_target_fraction,
            rate=args.rate, evidence_s=args.min_evidence_s, recovery_s=0.0,
            trend_s=args.trend_window_s,
            max_decline_fraction=args.max_decline_fraction,
        )
        recovery = valid & ~dominant & bool(spec.gases)
        if args.original_terminal_velocity:
            transport_mode = "original_gaden_ablation"
        elif args.buoyancy_tau_s > 0:
            transport_mode = "experimental_buoyancy_ablation"
        elif wind_regime != "zero":
            transport_mode = "wind_sensitivity"
        else:
            transport_mode = "passive_zero_wind"
        labels = source_labels(spec, ppm, valid, recovery)
        labels["target_fraction"] = target_fraction.astype(np.float32)
        labels["target_ppm"] = target_ppm
        labels["sniff_elapsed_s"] = _elapsed_while_true(dominant, args.rate)
        labels["endpoint_kind"] = np.where(
            dominant, "dominant",
            np.where(recovery, "recovery", "clean" if not spec.gases else "invalid"),
        )
        write_episode(
            args.out_dir, session_id=session_id, split=split, domain="virtual_gaden", rate=args.rate,
            voltage=voltage, rs_r0=rs_r0, ppm=ppm, time_s=np.arange(steps) / args.rate,
            position=path, phase=phase, labels=labels,
            metadata={"path_type": name, "seed": episode_seed,
                      "gaden_scenario": args.scenario,
                      "containment_model": "uncontained_point_sources",
                      "sniff_distance_m": sniff_distance_m,
                      "source_positions": positions.tolist(),
                      "background_distances_m": [
                          float(np.linalg.norm(position - positions[0]))
                          for position in positions[1:]
                      ],
                      "background_wind_orientations": background_orientations,
                      "wind_regime": wind_profile.regime,
                      "wind_update_s": 1.0,
                      "wind_vectors_m_s": wind_profile.vectors_m_s.tolist(),
                      "mean_wind_vector_m_s": wind_profile.mean_vector_m_s.tolist(),
                      "mean_wind_speed_m_s": float(np.linalg.norm(wind_profile.mean_vector_m_s)),
                      "source_concentrations_ppm": [s.concentration_ppm.tolist() for s in specs],
                      "desired_local_concentration_ppm": (
                          desired_spec.concentration_ppm.tolist()
                      ),
                      "local_ppm_calibration": args.local_ppm_calibration,
                      "local_ppm_extrapolation_enabled": (
                          args.allow_local_ppm_extrapolation
                      ),
                      "legacy_equivalent_emission_strengths": [
                          [_emission_strength(float(ppm)) for ppm in s.concentration_ppm if ppm > 0]
                          for s in specs
                      ],
                      "emission_model": args.emission_model,
                      "emission_scale": args.emission_scale,
                      "emission_concentration_exponent": (
                          args.emission_concentration_exponent
                      ),
                      "nominal_concentration_sampling": (
                          args.nominal_concentration_sampling
                      ),
                      "filaments_per_second": args.filaments_per_second,
                      "background_emission_ratio": args.background_emission_ratio,
                      "source_emission_parameters": [
                          [
                              _emission_parameters(
                                  float(value), model=args.emission_model,
                                  emission_scale=args.emission_scale,
                                  concentration_exponent=(
                                      args.emission_concentration_exponent
                                  ),
                                  filaments_per_second=args.filaments_per_second,
                                  background_ratio=(
                                      args.background_emission_ratio if i > 0 else 1.0
                                  ),
                              )
                              for value in source_spec.concentration_ppm if value > 0
                          ]
                          for i, source_spec in enumerate(specs)
                      ],
                      "evaluation_group": spec.evaluation_group,
                      "calibration_mode": args.calibration_mode,
                      "trajectory_satisfies_training_criteria": satisfies_training_criteria,
                      "valid_endpoint_count": int(valid.sum()),
                      "dominant_endpoint_count": int(dominant.sum()),
                      "recovery_endpoint_count": int(recovery.sum()),
                      "min_target_ppm": args.min_target_ppm,
                      "max_target_ppm": args.max_target_ppm,
                      "max_total_ppm": args.max_total_ppm,
                      "min_target_fraction": args.min_target_fraction,
                      "min_evidence_s": args.min_evidence_s,
                      "warmup_s": args.warmup_s,
                      "trend_window_s": args.trend_window_s,
                      "max_decline_fraction": args.max_decline_fraction,
                      "required_history_s": args.required_history_s,
                      "buoyancy_tau_s": args.buoyancy_tau_s,
                      "transport_mode": transport_mode,
                      "original_terminal_velocity_enabled": args.original_terminal_velocity,
                      "measurement_noise_std_v": noise_std_v,
                      "mean_dominant_target_fraction": (
                          float(target_fraction[dominant].mean()) if np.any(dominant) else None
                      )},
        )
        print(f"[phase-b] {index + 1}/{args.trajectories} {name}")
        index += 1
        episode_attempt = 0

    attempted = sum(attempted_by_path.values())
    accepted = sum(accepted_by_path.values())
    print(
        f"[phase-b] non-clean satisfaction: {accepted}/{attempted} "
        f"({accepted / max(attempted, 1):.1%})"
    )
    for name in PATH_NAMES:
        path_attempted = attempted_by_path[name]
        path_accepted = accepted_by_path[name]
        print(
            f"[phase-b]   {name}: {path_accepted}/{path_attempted} "
            f"({path_accepted / max(path_attempted, 1):.1%})"
        )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trajectories", type=int, default=1000, help="Pilot target: 1000")
    p.add_argument("--out-dir", default="datasets/odor_encoder/virtual_single_v6")
    p.add_argument("--rate", type=float, default=20.0)
    p.add_argument("--min-duration-s", type=float, default=24.0)
    p.add_argument("--max-duration-s", type=float, default=32.0)
    p.add_argument("--warmup-s", type=float, default=8.0)
    p.add_argument(
        "--emission-model", choices=("separated", "legacy_coupled"),
        default="separated",
    )
    p.add_argument(
        "--emission-scale", type=float, default=0.02,
        help="multiplicative scale in the separated C_filament mapping",
    )
    p.add_argument(
        "--emission-concentration-exponent", type=float, default=1.0,
        help=("C_filament = scale * 10 * (nominal_ppm / 10) ** exponent; "
              "1.0 preserves the linear mapping"),
    )
    p.add_argument(
        "--filaments-per-second", type=int, default=20,
        help="fixed lambda_emit used independently of C_filament",
    )
    p.add_argument(
        "--background-emission-ratio", type=float, default=0.10,
        help="additional C_filament multiplier for neighboring sources",
    )
    p.add_argument(
        "--buoyancy-tau-s", type=float, default=0.0,
        help="experimental filament buoyancy relaxation in seconds; 0 disables it",
    )
    p.add_argument(
        "--original-terminal-velocity", action="store_true",
        help="use the historical dimensionally flawed GADEN term for ablation only",
    )
    p.add_argument("--scenario", default="10x6_uniform")
    p.add_argument("--scenario-config", default="config1")
    p.add_argument("--server-log-dir", default=".roslog/odor_encoder_phase_b")
    p.add_argument("--runtime-config-dir", default=".runtime/phase_b")
    p.add_argument("--wait-timeout", type=float, default=60.0)
    p.add_argument("--background-probability", type=float, default=1.0)
    p.add_argument(
        "--nominal-concentration-sampling",
        choices=("log_uniform", "linear_uniform", "stratified_log"),
        default="log_uniform",
    )
    p.add_argument(
        "--local-ppm-calibration",
        help="inverse calibration JSON; sampled concentrations become desired local ppm",
    )
    p.add_argument(
        "--allow-local-ppm-extrapolation", action="store_true",
        help="allow desired local ppm outside the fitted mapping support",
    )
    p.add_argument("--concentration-min-ppm", type=float, default=1.0)
    p.add_argument("--concentration-max-ppm", type=float, default=100.0)
    p.add_argument("--background-max-gases", type=int, default=2)
    p.add_argument("--background-min-distance", type=float, default=0.20)
    p.add_argument("--background-max-distance", type=float, default=0.30)
    p.add_argument("--min-sniff-distance", type=float, default=0.01)
    p.add_argument("--max-sniff-distance", type=float, default=0.05)
    p.add_argument("--min-target-ppm", type=float, default=0.1)
    p.add_argument("--max-target-ppm", type=float, default=200.0)
    p.add_argument("--max-total-ppm", type=float, default=200.0)
    p.add_argument("--min-target-fraction", type=float, default=0.90)
    p.add_argument("--min-evidence-s", type=float, default=3.0)
    p.add_argument("--trend-window-s", type=float, default=1.0)
    p.add_argument("--max-decline-fraction", type=float, default=0.15)
    p.add_argument("--recovery-valid-s", type=float, default=0.0)
    p.add_argument("--required-history-s", type=float, default=20.0)
    p.add_argument(
        "--max-attempt-multiplier", type=int, default=50,
        help="maximum plume/layout retries per scheduled trajectory (default: 50)",
    )
    p.add_argument(
        "--calibration-mode", action="store_true",
        help=("save every attempted trajectory, including samples outside the "
              "training acceptance limits; never use this output for training"),
    )
    p.add_argument(
        "--wind-regime", choices=("zero", "sample", *WIND_REGIMES), default="zero",
        help=("sample uses stagnant/weak/vent weights "
              f"{WIND_REGIME_WEIGHTS.tolist()}; choose one value for a controlled run"),
    )
    p.add_argument("--seed", type=int, default=10000)
    p.add_argument("--min-noise-std-v", type=float, default=0.0)
    p.add_argument("--max-noise-std-v", type=float, default=0.01)
    p.add_argument("--split", choices=("auto", "train", "validation", "test"), default="auto")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.trajectories < 1 or args.rate <= 0:
        raise SystemExit("--trajectories and --rate must be positive")
    if not np.isfinite(args.buoyancy_tau_s) or args.buoyancy_tau_s < 0:
        raise SystemExit("--buoyancy-tau-s must be finite and non-negative")
    if args.original_terminal_velocity and args.buoyancy_tau_s != 0:
        raise SystemExit("legacy terminal velocity and experimental buoyancy are mutually exclusive")
    if not 0.0 <= args.background_probability <= 1.0:
        raise SystemExit("--background-probability must be in [0, 1]")
    if not 0 < args.concentration_min_ppm < args.concentration_max_ppm:
        raise SystemExit("concentration bounds must satisfy 0 < min < max")
    if (
        args.emission_scale <= 0
        or args.emission_concentration_exponent <= 0
        or args.filaments_per_second < 1
    ):
        raise SystemExit("separated emission controls must be positive")
    if not 0 < args.background_emission_ratio <= 1:
        raise SystemExit("--background-emission-ratio must be in (0, 1]")
    if not 0 < args.background_min_distance <= args.background_max_distance:
        raise SystemExit("background distances must satisfy 0 < min <= max")
    if not 0 < args.min_sniff_distance <= args.max_sniff_distance:
        raise SystemExit("sniff distances must satisfy 0 < min <= max")
    if (
        args.min_target_ppm < 0
        or args.max_target_ppm <= args.min_target_ppm
        or args.max_total_ppm < args.max_target_ppm
        or not 0.0 < args.min_target_fraction <= 1.0
        or args.min_evidence_s < 0
        or args.trend_window_s < 0
        or not 0.0 <= args.max_decline_fraction <= 1.0
    ):
        raise SystemExit("observability thresholds are invalid")
    if args.recovery_valid_s < 0 or args.required_history_s <= 0 or args.max_attempt_multiplier < 1:
        raise SystemExit("recovery duration must be nonnegative and max attempts must be positive")
    if args.required_history_s > args.max_duration_s:
        raise SystemExit("--required-history-s cannot exceed --max-duration-s")
    if not 0 <= args.min_noise_std_v <= args.max_noise_std_v:
        raise SystemExit("noise values must satisfy 0 <= min <= max")
    collect(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

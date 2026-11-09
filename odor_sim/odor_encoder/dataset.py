"""Shared odor-source sampling, labels, splitting, and episode IO."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from odor_sim.sensors.mox_pid import DEFAULT_MOX_MODELS, MOXSensor

GAS_ORDER = (
    "ethanol", "methanol", "acetone", "acetic_acid", "ethyl_acetate", "toluene",
)
CLASS_ORDER = ("air", *GAS_ORDER)
SENSOR_ORDER = tuple(DEFAULT_MOX_MODELS)
SAMPLE_RATE_HZ = 20.0
HELD_OUT_COMBINATIONS = frozenset(
    {
        ("ethanol", "toluene"),
        ("acetic_acid", "methanol"),
        ("acetone", "ethyl_acetate", "toluene"),
        ("acetic_acid", "ethanol", "ethyl_acetate", "methanol"),
    }
)


@dataclass(frozen=True)
class MixtureSpec:
    concentration_ppm: np.ndarray
    evaluation_group: str = "seen"

    @property
    def presence(self) -> np.ndarray:
        return (self.concentration_ppm > 0).astype(np.float32)

    @property
    def fractions(self) -> np.ndarray:
        total = float(self.concentration_ppm.sum())
        return (self.concentration_ppm / total).astype(np.float32) if total else np.zeros(6, np.float32)

    @property
    def gases(self) -> tuple[str, ...]:
        return tuple(g for g, c in zip(GAS_ORDER, self.concentration_ppm) if c > 0)


def sample_single_source(
    rng: np.random.Generator, *, clean_probability: float = 0.10,
) -> MixtureSpec:
    """Sample clean air or one primary gas at a log-uniform 1-100 ppm.

    Trace gases from other sources belong in the measured ppm signal, not in
    this target specification. This keeps the identity label clean and
    teaches the encoder to ignore low-level cross-plume interference.
    """
    if rng.random() < clean_probability:
        return MixtureSpec(np.zeros(len(GAS_ORDER), np.float32), "clean")
    vector = np.zeros(len(GAS_ORDER), np.float32)
    gas_index = int(rng.integers(len(GAS_ORDER)))
    vector[gas_index] = 10.0 ** rng.uniform(0.0, 2.0)
    return MixtureSpec(vector, "single_gas")


def sample_background(
    rng: np.random.Generator, primary: MixtureSpec, *, max_gases: int = 2,
    min_fraction: float = 0.002, max_fraction: float = 0.03,
) -> MixtureSpec:
    """Sample weak non-target gases relative to the primary concentration."""
    if not primary.gases or max_gases < 1:
        return MixtureSpec(np.zeros(len(GAS_ORDER), np.float32), "background")
    primary_index = int(np.flatnonzero(primary.concentration_ppm)[0])
    candidates = np.delete(np.arange(len(GAS_ORDER)), primary_index)
    count = int(rng.integers(1, min(max_gases, len(candidates)) + 1))
    selected = rng.choice(candidates, count, replace=False)
    vector = np.zeros(len(GAS_ORDER), np.float32)
    primary_ppm = float(primary.concentration_ppm[primary_index])
    vector[selected] = primary_ppm * np.exp(
        rng.uniform(np.log(min_fraction), np.log(max_fraction), count)
    )
    return MixtureSpec(vector, "background")


def assign_split(session_id: str, seed: int = 0) -> str:
    """Deterministic 70/15/15 assignment for a complete session."""
    digest = hashlib.blake2b(f"{seed}:{session_id}".encode(), digest_size=8).digest()
    value = int.from_bytes(digest, "big") / 2**64
    return "train" if value < 0.70 else "validation" if value < 0.85 else "test"


def select_split(session_id: str, seed: int, requested: str = "auto") -> str:
    """Return a deterministic split, or an explicit evaluation-only override."""
    if requested == "auto":
        return assign_split(session_id, seed)
    if requested not in ("train", "validation", "test"):
        raise ValueError(f"unknown split {requested!r}")
    return requested


def _cardinality(rng: np.random.Generator) -> int:
    bucket = int(rng.choice(5, p=[0.10, 0.30, 0.30, 0.20, 0.10]))
    return bucket if bucket < 4 else int(rng.integers(4, 7))


def _sample_seen_combo(rng: np.random.Generator, count: int) -> tuple[str, ...]:
    while True:
        combo = tuple(sorted(rng.choice(GAS_ORDER, count, replace=False).tolist()))
        if combo not in HELD_OUT_COMBINATIONS:
            return combo


def sample_mixture(rng: np.random.Generator, *, split: str, force_nonempty: bool = False) -> MixtureSpec:
    """Sample the requested clean/30/30/20/10 cardinality distribution.

    Reserved combinations and the reserved 3:1 ratio occur only in test.
    Concentrations are approximately log-uniform in the calibrated 1-100 ppm range.
    """
    count = _cardinality(rng)
    if force_nonempty:
        count = max(1, count)
    if count == 0:
        return MixtureSpec(np.zeros(6, np.float32), "pure")

    group = "pure" if count == 1 else "seen"
    candidates = [c for c in HELD_OUT_COMBINATIONS if len(c) == count]
    if split == "test" and candidates and rng.random() < 0.35:
        combo = candidates[int(rng.integers(len(candidates)))]
        group = "unseen_combination"
    else:
        combo = _sample_seen_combo(rng, count)

    if split == "test" and count >= 2 and group == "seen" and rng.random() < 0.35:
        weights = np.ones(count)
        weights[0] = 3.0
        rng.shuffle(weights)
        weights /= weights.sum()
        group = "unseen_ratio"
    else:
        weights = rng.dirichlet(np.full(count, 0.7))
    total = (10.0 ** rng.uniform(0.0, 2.0)) * count
    concentrations = np.clip(total * weights, 1.0, 100.0)
    vector = np.zeros(6, np.float32)
    for gas, ppm in zip(combo, concentrations):
        vector[GAS_ORDER.index(gas)] = ppm
    return MixtureSpec(vector, group)


def simulate_mox(
    ppm: np.ndarray,
    rate: float,
    *,
    measurement_noise_std_v: float = 0.0,
    rng: np.random.Generator | None = None,
    precondition_ppm: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Run one uninterrupted MOX history without resetting between rows.

    ``precondition_ppm`` advances the same sensor instances before recording,
    allowing a non-baseline initial state without leaking warm-up samples into
    the saved sequence. Voltage noise is measurement-only; ``rs_r0`` retains
    the physical filtered sensor state.
    """
    ppm = np.asarray(ppm, np.float32)
    if ppm.ndim != 2 or ppm.shape[1] != 6:
        raise ValueError(f"ppm must be (T, 6), got {ppm.shape}")
    sensors = [MOXSensor(model=model, rate=rate) for model in SENSOR_ORDER]
    if measurement_noise_std_v < 0:
        raise ValueError("measurement_noise_std_v must be non-negative")
    if precondition_ppm is not None:
        precondition_ppm = np.asarray(precondition_ppm, np.float32)
        if precondition_ppm.ndim != 2 or precondition_ppm.shape[1] != len(GAS_ORDER):
            raise ValueError("precondition_ppm must have shape (T, 6)")
        for row in precondition_ppm:
            values = dict(zip(GAS_ORDER, row.tolist()))
            for sensor in sensors:
                sensor.update(values)
    voltage = np.empty((len(ppm), 8), np.float32)
    rs_r0 = np.empty_like(voltage)
    for t, row in enumerate(ppm):
        values = dict(zip(GAS_ORDER, row.tolist()))
        for channel, sensor in enumerate(sensors):
            sensor.update(values)
            voltage[t, channel] = sensor.voltage
            rs_r0[t, channel] = sensor.rs_r0
    if measurement_noise_std_v > 0:
        rng = np.random.default_rng() if rng is None else rng
        voltage += rng.normal(0.0, measurement_noise_std_v, voltage.shape).astype(np.float32)
        np.clip(voltage, 0.0, 5.0, out=voltage)
    return voltage, rs_r0


def source_labels(
    spec: MixtureSpec, ppm: np.ndarray, valid: np.ndarray,
    recovery: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Source identity/fractions plus local total intensity at every timestep."""
    ppm = np.asarray(ppm, np.float32)
    n = len(ppm)
    totals = ppm.sum(1)
    recovery = np.zeros(n, bool) if recovery is None else np.asarray(recovery, bool)
    if recovery.shape != (n,):
        raise ValueError(f"recovery must be ({n},), got {recovery.shape}")
    class_index = np.zeros(n, np.int64)
    if spec.gases:
        primary_index = int(np.flatnonzero(spec.concentration_ppm)[0])
        class_index[(totals >= 1e-4) & ~recovery] = primary_index + 1
    return {
        "label_presence": np.repeat(spec.presence[None], n, axis=0),
        "label_fractions": np.repeat(spec.fractions[None], n, axis=0),
        "label_intensity": np.log1p(totals).astype(np.float32),
        "label_clean": ((totals < 1e-4) & ~recovery).astype(np.float32),
        "label_recovery": recovery.astype(np.float32),
        "label_valid": np.asarray(valid, bool),
        "label_class": class_index,
        "label_class_valid": np.asarray(valid, bool),
    }


def current_class_labels(
    class_index: np.ndarray,
    ppm: np.ndarray,
    valid: np.ndarray,
    recovery: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Build per-timestep air/single-gas targets for transition sequences."""
    class_index = np.asarray(class_index, np.int64)
    ppm = np.asarray(ppm, np.float32)
    valid = np.asarray(valid, bool)
    n = len(ppm)
    if class_index.shape != (n,) or valid.shape != (n,):
        raise ValueError("class_index and valid must match the ppm time dimension")
    if np.any((class_index < 0) | (class_index >= len(CLASS_ORDER))):
        raise ValueError("class_index contains an unknown class")
    recovery = np.zeros(n, bool) if recovery is None else np.asarray(recovery, bool)
    presence = np.zeros((n, len(GAS_ORDER)), np.float32)
    gas_rows = class_index > 0
    presence[np.flatnonzero(gas_rows), class_index[gas_rows] - 1] = 1.0
    totals = ppm.sum(1)
    return {
        "label_presence": presence,
        "label_fractions": presence.copy(),
        "label_intensity": np.log1p(totals).astype(np.float32),
        "label_clean": ((class_index == 0) & ~recovery).astype(np.float32),
        "label_recovery": recovery.astype(np.float32),
        "label_valid": valid,
        "label_class": class_index,
        "label_class_valid": valid.copy(),
    }


def write_episode(
    out_dir: str | Path, *, session_id: str, split: str, domain: str, rate: float,
    voltage: np.ndarray, rs_r0: np.ndarray, ppm: np.ndarray, time_s: np.ndarray,
    position: np.ndarray, phase: np.ndarray, labels: dict[str, np.ndarray], metadata: dict,
    extra_arrays: dict[str, np.ndarray] | None = None,
    class_order: tuple[str, ...] | list[str] = CLASS_ORDER,
) -> Path:
    """Write one complete sequence. Causal windows are extracted only at training time."""
    path = Path(out_dir) / split / f"{session_id}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        "voltage": np.asarray(voltage, np.float32),
        "rs_r0": np.asarray(rs_r0, np.float32),
        "ppm": np.asarray(ppm, np.float32),
        "time_s": np.asarray(time_s, np.float32),
        "position": np.asarray(position, np.float32),
        "phase": np.asarray(phase, dtype="U48"),
        "gas_order": np.asarray(GAS_ORDER),
        "class_order": np.asarray(class_order),
        "sensor_order": np.asarray(SENSOR_ORDER),
        **labels,
    }
    if extra_arrays:
        overlap = set(arrays).intersection(extra_arrays)
        if overlap:
            raise ValueError(f"extra_arrays duplicates built-in keys: {sorted(overlap)}")
        arrays.update(extra_arrays)
    np.savez_compressed(path, **arrays)
    document = {
        "schema_version": 3, "session_id": session_id, "split": split,
        "domain": domain, "rate_hz": rate, "gas_order": list(GAS_ORDER),
        "class_order": list(class_order),
        "sensor_order": list(SENSOR_ORDER), "windows_extracted": False, **metadata,
    }
    path.with_suffix(".json").write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    return path

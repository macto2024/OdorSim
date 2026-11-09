"""Airflow profiles and isolated GADEN configurations for odor collection."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml


WIND_REGIMES = (
    "stagnant_correlated",
    "uniform_weak_indoor",
    "vent_like",
)
WIND_REGIME_WEIGHTS = np.asarray([0.40, 0.45, 0.15], dtype=float)
WIND_UPDATE_S = 1.0


@dataclass(frozen=True)
class WindProfile:
    """One episode's spatially uniform wind vectors, sampled once per second."""

    regime: str
    vectors_m_s: np.ndarray
    nominal_direction: np.ndarray

    @property
    def mean_vector_m_s(self) -> np.ndarray:
        return np.asarray(self.vectors_m_s, float).mean(axis=0)

    @property
    def speed_m_s(self) -> np.ndarray:
        return np.linalg.norm(self.vectors_m_s, axis=1)


def _horizontal_direction(rng: np.random.Generator) -> np.ndarray:
    angle = float(rng.uniform(-np.pi, np.pi))
    return np.asarray([np.cos(angle), np.sin(angle), 0.0])


def zero_wind_profile(duration_s: float) -> WindProfile:
    """An exactly zero, spatially uniform airflow profile."""
    count = max(2, int(np.ceil(duration_s / WIND_UPDATE_S)) + 2)
    return WindProfile(
        "zero",
        np.zeros((count, 3), np.float32),
        np.asarray([1.0, 0.0, 0.0], np.float32),
    )


def sample_wind_profile(
    rng: np.random.Generator,
    duration_s: float,
    regime: str | None = None,
) -> WindProfile:
    """Sample a plausible indoor profile without claiming CFD circulation.

    All vectors are spatially uniform. Only ``stagnant_correlated`` changes
    slowly over time; its autoregressive update represents weak room drift,
    not a resolved vortex or recirculation zone.
    """
    if regime is None or regime == "sample":
        regime = str(rng.choice(WIND_REGIMES, p=WIND_REGIME_WEIGHTS))
    if regime not in WIND_REGIMES:
        raise ValueError(f"unknown wind regime {regime!r}")
    count = max(2, int(np.ceil(duration_s / WIND_UPDATE_S)) + 2)
    direction = _horizontal_direction(rng)

    if regime == "stagnant_correlated":
        vector = direction * float(rng.uniform(0.003, 0.012))
        vector[2] = float(rng.uniform(-0.002, 0.002))
        vectors = np.empty((count, 3), float)
        for index in range(count):
            innovation = rng.normal(0.0, [0.0025, 0.0025, 0.0008])
            vector = 0.94 * vector + 0.06 * direction * 0.008 + innovation
            horizontal = np.linalg.norm(vector[:2])
            if horizontal > 0.03:
                vector[:2] *= 0.03 / horizontal
            vector[2] = np.clip(vector[2], -0.006, 0.006)
            vectors[index] = vector
    else:
        speed_range = {
            "uniform_weak_indoor": (0.03, 0.12),
            "vent_like": (0.15, 0.30),
        }[regime]
        speed = float(rng.uniform(*speed_range))
        vertical_fraction = float(rng.uniform(-0.08, 0.08))
        vector = direction * speed
        vector[2] = speed * vertical_fraction
        vectors = np.repeat(vector[None], count, axis=0)

    mean_horizontal = np.asarray(vectors[:, :2].mean(axis=0), float)
    norm = np.linalg.norm(mean_horizontal)
    nominal = direction if norm < 1e-9 else np.asarray([*mean_horizontal / norm, 0.0])
    return WindProfile(regime, vectors.astype(np.float32), nominal.astype(np.float32))


def write_isolated_config(
    source_config_dir: str | Path,
    destination_config_dir: str | Path,
    profile: WindProfile,
) -> Path:
    """Write a private config that reuses geometry but owns its wind CSV."""
    source_config_dir = Path(source_config_dir).resolve()
    destination_config_dir = Path(destination_config_dir).resolve()
    destination_config_dir.mkdir(parents=True, exist_ok=True)
    with open(source_config_dir / "config.yaml") as stream:
        config = yaml.safe_load(stream) or {}

    for key in ("models", "outlets_models"):
        config[key] = [
            str((source_config_dir / entry).resolve()) if not Path(entry).is_absolute() else entry
            for entry in config.get(key, [])
        ]
    wind_dir = destination_config_dir / "wind"
    wind_dir.mkdir(parents=True, exist_ok=True)
    wind_prefix = wind_dir / "wind"
    np.savetxt(
        wind_dir / "wind_0.csv", profile.vectors_m_s,
        delimiter=",", fmt="%.8f",
    )
    config["unprocessed_wind_files"] = str(wind_prefix)
    config["uniformWind"] = True
    with open(destination_config_dir / "config.yaml", "w") as stream:
        yaml.safe_dump(config, stream, sort_keys=False)
    return destination_config_dir

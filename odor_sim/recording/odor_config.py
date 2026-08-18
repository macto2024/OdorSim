"""Canonical OVLA odor-collection timing, sensor, and VOC configuration.

All derived dimensions (sniff/filter step counts, channel count) come from
this dataclass so collectors and the RLDS builder do not hard-code 200 / 400 / 5.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from odor_sim.config.gas_types import EXCEL_VOC_GAS_NAMES
from odor_sim.sensors.mox_pid import DEFAULT_MOX_MODELS

# Smellnet_chkpt / ScentFormerAB class names (single strength; no 1.0 suffix).
DEFAULT_VOC_POOL: tuple[str, ...] = EXCEL_VOC_GAS_NAMES

DEFAULT_SAMPLING_RATE = 20.0
DEFAULT_SNIFF_DURATION_S = 10.0
FILTER_DURATION_FACTOR = 1.5


@dataclass(frozen=True)
class OdorConfig:
    """One sniff protocol + sensor/VOC contract for a collection run."""

    odor_sampling_rate: float = DEFAULT_SAMPLING_RATE
    sniff_duration_s: float = DEFAULT_SNIFF_DURATION_S
    filter_duration_s: "float | None" = None
    mox_models: tuple[str, ...] = field(default_factory=lambda: tuple(DEFAULT_MOX_MODELS))
    voc_pool: tuple[str, ...] = DEFAULT_VOC_POOL

    def __post_init__(self) -> None:
        if float(self.odor_sampling_rate) <= 0:
            raise ValueError(f"odor_sampling_rate must be > 0, got {self.odor_sampling_rate}")
        if float(self.sniff_duration_s) <= 0:
            raise ValueError(f"sniff_duration_s must be > 0, got {self.sniff_duration_s}")
        if self.filter_duration_s is not None and float(self.filter_duration_s) < 0:
            raise ValueError(
                f"filter_duration_s must be >= 0, got {self.filter_duration_s}"
            )
        if not self.mox_models:
            raise ValueError("mox_models must be non-empty")
        if not self.voc_pool:
            raise ValueError("voc_pool must be non-empty")

    @property
    def filter_s(self) -> float:
        """Nose-rest duration: explicit override, else 1.5 × sniff."""
        if self.filter_duration_s is None:
            return FILTER_DURATION_FACTOR * float(self.sniff_duration_s)
        return float(self.filter_duration_s)

    @property
    def sniff_steps(self) -> int:
        return max(1, int(round(float(self.sniff_duration_s) * float(self.odor_sampling_rate))))

    @property
    def filter_steps(self) -> int:
        return max(1, int(round(self.filter_s * float(self.odor_sampling_rate))))

    @property
    def num_odor_channels(self) -> int:
        return len(self.mox_models)

    @property
    def control_freq(self) -> int:
        return int(round(float(self.odor_sampling_rate)))

    def with_updates(self, **kwargs) -> "OdorConfig":
        return replace(self, **kwargs)

    def to_meta(self) -> dict:
        """JSON-safe dict for episode ``meta.json`` / the collection manifest."""
        return {
            "odor_sampling_rate": float(self.odor_sampling_rate),
            "sniff_duration_s": float(self.sniff_duration_s),
            "filter_duration_s": float(self.filter_s),
            "mox_models": list(self.mox_models),
            "voc_pool": list(self.voc_pool),
            "num_odor_channels": self.num_odor_channels,
            "sniff_steps": self.sniff_steps,
            "filter_steps": self.filter_steps,
        }


DEFAULT_ODOR_CONFIG = OdorConfig()


def odor_config_from_args(
    *,
    sniff_duration_s: "float | None" = None,
    filter_duration_s: "float | None" = None,
    odor_sampling_rate: "float | None" = None,
    mox_models: "list[str] | tuple[str, ...] | None" = None,
    voc_pool: "list[str] | tuple[str, ...] | None" = None,
    sample_hold_s: "float | None" = None,
) -> OdorConfig:
    """Build a config from CLI-style optional overrides.

    ``sniff_duration_s`` wins over the legacy alias ``sample_hold_s``.
    ``filter_duration_s=None`` keeps the 1.5× sniff default.
    """
    sniff = sniff_duration_s
    if sniff is None:
        sniff = sample_hold_s
    if sniff is None:
        sniff = DEFAULT_SNIFF_DURATION_S
    kwargs: dict = {"sniff_duration_s": float(sniff)}
    if filter_duration_s is not None:
        kwargs["filter_duration_s"] = float(filter_duration_s)
    if odor_sampling_rate is not None:
        kwargs["odor_sampling_rate"] = float(odor_sampling_rate)
    if mox_models is not None:
        kwargs["mox_models"] = tuple(str(m) for m in mox_models)
    if voc_pool is not None:
        kwargs["voc_pool"] = tuple(str(v) for v in voc_pool)
    return OdorConfig(**kwargs)

"""OdorProfile: the multi-VOC odor description attached to an object.

An object emits a *mixture* of volatile organic compounds (VOCs). Each *active*
VOC component (``strength > 0``) becomes one live GADEN source, co-located at
the object's position (plus an optional per-VOC local offset). ``strength`` is
an abstract 0..1 knob that maps to GADEN emission parameters
(``filamentPPMcenter`` and ``numFilaments_sec``); ``strength == 0`` means the
object does not emit that gas at all (no GADEN source is created for it).

This lets every object declare the full MOX gas panel with explicit zeros, so
the recorded ppm vector has a fixed layout while only the emitting gases cost
any GADEN compute.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math

from odor_sim.config.gas_types import GasType, MOX_SUPPORTED, gaden_name, parse_gas_type

# strength (0..1) -> GADEN emission params.
# Linear interpolation between a "barely detectable" and a "strong" source,
# tuned against the Phase 2 scenario (ethanol at ppmCenter 20, 20 filaments/s
# gave ~189 ppm downwind).
_PPM_CENTER_MIN = 5.0
_PPM_CENTER_MAX = 40.0
_FILAMENTS_MIN = 5
_FILAMENTS_MAX = 40


def calibrated_emission_parameters(
    nominal_ppm: float,
    *,
    emission_scale: float = 0.06,
    concentration_exponent: float = 0.5,
    filaments_per_second: int = 15,
) -> dict[str, float | int]:
    """Return the separated empirical source controls used by encoder data.

    ``nominal_ppm`` is a source condition, not the concentration measured at
    the e-nose. The local-ppm calibration maps desired local ppm back to this
    source condition before this function is called.
    """
    nominal_ppm = float(nominal_ppm)
    emission_scale = float(emission_scale)
    concentration_exponent = float(concentration_exponent)
    filaments_per_second = int(filaments_per_second)
    if not math.isfinite(nominal_ppm) or nominal_ppm <= 0:
        raise ValueError("nominal_ppm must be finite and positive")
    if not math.isfinite(emission_scale) or emission_scale <= 0:
        raise ValueError("emission_scale must be finite and positive")
    if not math.isfinite(concentration_exponent) or concentration_exponent <= 0:
        raise ValueError("concentration_exponent must be finite and positive")
    if filaments_per_second <= 0:
        raise ValueError("filaments_per_second must be positive")
    return {
        "filament_ppm_center": (
            emission_scale
            * 10.0
            * (nominal_ppm / 10.0) ** concentration_exponent
        ),
        "num_filaments_sec": filaments_per_second,
    }


@dataclass(frozen=True)
class VOCComponent:
    """One volatile compound emitted by an object.

    Args:
        gas_type: which GADEN gas (must be MOX-supported for the e-nose).
        strength: abstract emission strength in [0, 1].
        local_offset: (x, y, z) offset of this source from the object body
            origin, expressed in the object's local frame (meters). Lets one
            object emit different VOCs from slightly different points.
    """

    gas_type: GasType
    strength: float = 0.5
    local_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    filament_ppm_center_override: float | None = None
    num_filaments_sec_override: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "gas_type", parse_gas_type(self.gas_type))
        if not 0.0 <= self.strength <= 1.0:
            raise ValueError(f"strength must be in [0, 1], got {self.strength}")
        if len(self.local_offset) != 3:
            raise ValueError("local_offset must be a 3-tuple (x, y, z)")
        if (
            self.filament_ppm_center_override is not None
            and (
                not math.isfinite(self.filament_ppm_center_override)
                or self.filament_ppm_center_override <= 0
            )
        ):
            raise ValueError("filament_ppm_center must be finite and positive")
        if (
            self.num_filaments_sec_override is not None
            and self.num_filaments_sec_override < 1
        ):
            raise ValueError("num_filaments_sec must be a positive integer")

    @property
    def filament_ppm_center(self) -> float:
        if self.filament_ppm_center_override is not None:
            return float(self.filament_ppm_center_override)
        return _PPM_CENTER_MIN + self.strength * (_PPM_CENTER_MAX - _PPM_CENTER_MIN)

    @property
    def num_filaments_sec(self) -> int:
        if self.num_filaments_sec_override is not None:
            return int(self.num_filaments_sec_override)
        return int(round(_FILAMENTS_MIN + self.strength * (_FILAMENTS_MAX - _FILAMENTS_MIN)))

    @property
    def is_active(self) -> bool:
        """True if this VOC actually emits (strength > 0 -> a GADEN source)."""
        return self.strength > 0.0

    def is_mox_detectable(self) -> bool:
        return self.gas_type in MOX_SUPPORTED

    def gaden_gas_name(self) -> str:
        return gaden_name(self.gas_type)


@dataclass
class OdorProfile:
    """A mixture of VOC components emitted by a single object."""

    components: list[VOCComponent] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.components:
            raise ValueError("OdorProfile must have at least one VOC component")

    def __len__(self) -> int:
        return len(self.components)

    def __iter__(self):
        return iter(self.components)

    def active_components(self) -> list[VOCComponent]:
        """VOCs that actually emit (strength > 0); these become GADEN sources."""
        return [c for c in self.components if c.is_active]

    @property
    def num_sources(self) -> int:
        """Number of GADEN sources this profile expands into (active VOCs only)."""
        return len(self.active_components())

    @property
    def gas_types(self) -> list[GasType]:
        """Every declared gas, active or not (the fixed ppm-vector layout)."""
        return [c.gas_type for c in self.components]

    def undetectable_components(self) -> list[VOCComponent]:
        """VOCs the stock MOX sensor cannot model (useful for validation)."""
        return [c for c in self.components if not c.is_mox_detectable()]

    @classmethod
    def from_spec(cls, spec: list) -> "OdorProfile":
        """Build from a list of dicts/tuples.

        Each entry may be:
          - a dict: {gas_type, strength?, local_offset?,
                     filament_ppm_center?, num_filaments_sec?}
          - a tuple/list: (gas_type, strength?, local_offset?)
        """
        components: list[VOCComponent] = []
        for entry in spec:
            if isinstance(entry, dict):
                components.append(
                    VOCComponent(
                        gas_type=parse_gas_type(entry["gas_type"]),
                        strength=float(entry.get("strength", 0.5)),
                        local_offset=tuple(entry.get("local_offset", (0.0, 0.0, 0.0))),
                        filament_ppm_center_override=(
                            float(entry["filament_ppm_center"])
                            if "filament_ppm_center" in entry else None
                        ),
                        num_filaments_sec_override=(
                            int(entry["num_filaments_sec"])
                            if "num_filaments_sec" in entry else None
                        ),
                    )
                )
            elif isinstance(entry, (tuple, list)):
                gas = parse_gas_type(entry[0])
                strength = float(entry[1]) if len(entry) > 1 else 0.5
                offset = tuple(entry[2]) if len(entry) > 2 else (0.0, 0.0, 0.0)
                components.append(VOCComponent(gas, strength, offset))
            else:
                raise TypeError(f"Unsupported VOC spec entry: {entry!r}")
        return cls(components=components)

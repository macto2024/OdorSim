"""OdorProfile: the multi-VOC odor description attached to an object.

An object emits a *mixture* of volatile organic compounds (VOCs). Each VOC
component becomes one live GADEN source, co-located at the object's position
(plus an optional per-VOC local offset). ``strength`` is an abstract 0..1 knob
that maps to GADEN emission parameters (``filamentPPMcenter`` and
``numFilaments_sec``).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from odor_sim.config.gas_types import GasType, MOX_SUPPORTED, gaden_name, parse_gas_type

# strength (0..1) -> GADEN emission params.
# Linear interpolation between a "barely detectable" and a "strong" source,
# tuned against the Phase 2 scenario (ethanol at ppmCenter 20, 20 filaments/s
# gave ~189 ppm downwind).
_PPM_CENTER_MIN = 5.0
_PPM_CENTER_MAX = 40.0
_FILAMENTS_MIN = 5
_FILAMENTS_MAX = 40


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

    def __post_init__(self) -> None:
        object.__setattr__(self, "gas_type", parse_gas_type(self.gas_type))
        if not 0.0 <= self.strength <= 1.0:
            raise ValueError(f"strength must be in [0, 1], got {self.strength}")
        if len(self.local_offset) != 3:
            raise ValueError("local_offset must be a 3-tuple (x, y, z)")

    @property
    def filament_ppm_center(self) -> float:
        return _PPM_CENTER_MIN + self.strength * (_PPM_CENTER_MAX - _PPM_CENTER_MIN)

    @property
    def num_filaments_sec(self) -> int:
        return int(round(_FILAMENTS_MIN + self.strength * (_FILAMENTS_MAX - _FILAMENTS_MIN)))

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

    @property
    def num_sources(self) -> int:
        """Number of GADEN sources this profile expands into (one per VOC)."""
        return len(self.components)

    @property
    def gas_types(self) -> list[GasType]:
        return [c.gas_type for c in self.components]

    def undetectable_components(self) -> list[VOCComponent]:
        """VOCs the stock MOX sensor cannot model (useful for validation)."""
        return [c for c in self.components if not c.is_mox_detectable()]

    @classmethod
    def from_spec(cls, spec: list) -> "OdorProfile":
        """Build from a list of dicts/tuples.

        Each entry may be:
          - a dict: {gas_type, strength?, local_offset?}
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

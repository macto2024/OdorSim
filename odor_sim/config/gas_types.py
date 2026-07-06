"""GADEN gas type definitions and sensor-support tables.

The integer values MUST match GADEN's ``gaden::GasType`` enum
(``gaden/gaden_common/third_party/gaden_core/include/gaden/datatypes/GasTypes.hpp``)
because the same integer is written into the generated GADEN ``sim.yaml``
files and interpreted by the C++ core.
"""

from __future__ import annotations

from enum import IntEnum


class GasType(IntEnum):
    """Mirror of ``gaden::GasType`` (backing type ``int``)."""

    unknown = -1
    ethanol = 0
    methane = 1
    hydrogen = 2
    propanol = 3
    chlorine = 4
    fluorine = 5
    acetone = 6
    neon = 7
    helium = 8
    biogas = 9
    butane = 10
    carbon_dioxide = 11
    carbon_monoxide = 12
    smoke = 13


# GADEN's ``to_string`` spellings, keyed by GasType. Kept separate from the
# enum member names so we can use pythonic names (carbon_dioxide) while still
# emitting exactly what GADEN prints/parses.
GADEN_GAS_NAMES: dict[GasType, str] = {
    GasType.unknown: "unknown",
    GasType.ethanol: "ethanol",
    GasType.methane: "methane",
    GasType.hydrogen: "hydrogen",
    GasType.propanol: "propanol",
    GasType.chlorine: "chlorine",
    GasType.fluorine: "flurorine",  # NOTE: GADEN spells it 'flurorine'
    GasType.acetone: "acetone",
    GasType.neon: "neon",
    GasType.helium: "helium",
    GasType.biogas: "biogas",
    GasType.butane: "butane",
    GasType.carbon_dioxide: "carbonDioxide",
    GasType.carbon_monoxide: "carbonMonoxide",
    GasType.smoke: "smoke",
}

# The stock MOX sensor model (fake_gas_sensor.cpp) only has response tables for
# these gases. VOC recipes intended for the e-nose must use this set unless the
# sensor tables are extended.
MOX_SUPPORTED: frozenset[GasType] = frozenset(
    {
        GasType.ethanol,
        GasType.methane,
        GasType.hydrogen,
        GasType.propanol,
        GasType.chlorine,
        GasType.fluorine,
        GasType.acetone,
    }
)

# The stock PID sensor only models these three.
PID_SUPPORTED: frozenset[GasType] = frozenset(
    {GasType.ethanol, GasType.methane, GasType.hydrogen}
)


def parse_gas_type(value: "str | int | GasType") -> GasType:
    """Coerce a name (either python or GADEN spelling) or int into a GasType."""
    if isinstance(value, GasType):
        return value
    if isinstance(value, int):
        return GasType(value)
    if isinstance(value, str):
        key = value.strip().lower()
        # match python enum names (carbon_dioxide) and GADEN spellings
        for gt in GasType:
            if gt.name.lower() == key:
                return gt
        for gt, name in GADEN_GAS_NAMES.items():
            if name.lower() == key:
                return gt
        # tolerate the common 'fluorine' spelling
        if key == "fluorine":
            return GasType.fluorine
    raise ValueError(f"Unknown gas type: {value!r}")


def gaden_name(gas: GasType) -> str:
    """Return GADEN's canonical string for a gas type."""
    return GADEN_GAS_NAMES[gas]

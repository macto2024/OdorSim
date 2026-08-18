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
    methanol = 14
    acetic_acid = 15
    ethyl_acetate = 16
    toluene = 17


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
    GasType.methanol: "methanol",
    GasType.acetic_acid: "acetic_acid",
    GasType.ethyl_acetate: "ethyl_acetate",
    GasType.toluene: "toluene",
}

# Excel-calibrated / ScentFormer VOC panel. OVLA liquids (``ethanol``, …)
# declare this full layout (active gas at strength 1, the rest at 0) so
# recorded ppm columns stay stable. Integer IDs 14–17 require patched GADEN.
EXCEL_VOC_GASES: tuple[GasType, ...] = (
    GasType.ethanol,
    GasType.methanol,
    GasType.acetone,
    GasType.acetic_acid,
    GasType.ethyl_acetate,
    GasType.toluene,
)
EXCEL_VOC_GAS_NAMES: tuple[str, ...] = tuple(g.name for g in EXCEL_VOC_GASES)

# Gas types that OdorSim's calibrated Python MOX array may consume. The legacy
# GADEN fake_gas_sensor.cpp remains unchanged; it is not used for these curves.
MOX_SUPPORTED: frozenset[GasType] = frozenset(
    {
        GasType.ethanol,
        GasType.methane,
        GasType.hydrogen,
        GasType.propanol,
        GasType.chlorine,
        GasType.fluorine,
        GasType.acetone,
        GasType.methanol,
        GasType.acetic_acid,
        GasType.ethyl_acetate,
        GasType.toluene,
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

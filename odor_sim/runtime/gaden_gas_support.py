"""Fail fast when collection needs GADEN IDs 14–17 but the binary is unpatched."""

from __future__ import annotations

import os
from pathlib import Path

from odor_sim.config.gas_types import GasType, parse_gas_type

_REPO_ROOT = Path(__file__).resolve().parents[2]
_EXCEL_GAS_IDS = {
    GasType.methanol,
    GasType.acetic_acid,
    GasType.ethyl_acetate,
    GasType.toluene,
}
_EXCEL_GAS_MARKERS = (b"methanol", b"acetic_acid", b"ethyl_acetate", b"toluene")


def find_rt_server() -> "Path | None":
    """Return the ``odor_gaden_rt`` ``rt_server`` binary if it is on disk."""
    prefixes = [p for p in os.environ.get("AMENT_PREFIX_PATH", "").split(os.pathsep) if p]
    candidates = [
        Path(prefix) / "lib" / "odor_gaden_rt" / "rt_server" for prefix in prefixes
    ]
    candidates.append(
        _REPO_ROOT / "ros2_ws" / "install" / "odor_gaden_rt" / "lib" / "odor_gaden_rt" / "rt_server"
    )
    for path in candidates:
        if path.is_file():
            return path
    return None


def gaden_binary_supports_excel_gases(binary: "str | Path | None" = None) -> bool:
    """True if ``rt_server`` was built with methanol / acetic_acid / ethyl_acetate / toluene."""
    path = Path(binary) if binary is not None else find_rt_server()
    if path is None or not path.is_file():
        return False
    data = path.read_bytes()
    return all(marker in data for marker in _EXCEL_GAS_MARKERS)


def active_gaden_gas_names(env) -> list[str]:
    """GADEN names for VOCs that actually emit (strength > 0)."""
    names: list[str] = []
    seen: set[str] = set()
    for obj in getattr(env, "odor_objects", []) or []:
        profile = getattr(obj, "odor_profile", None)
        if profile is None:
            continue
        for comp in profile.active_components():
            name = comp.gaden_gas_name()
            if name not in seen:
                seen.add(name)
                names.append(name)
    return names


def excel_gases_in(gas_types) -> list[str]:
    """Subset of ``gas_types`` that need GADEN IDs 14–17."""
    needed: list[str] = []
    for value in gas_types or []:
        try:
            gas = parse_gas_type(value)
        except ValueError:
            continue
        if gas in _EXCEL_GAS_IDS:
            needed.append(gas.name)
    return needed


def require_gaden_excel_gases(gas_types) -> None:
    """Raise if the live GADEN binary cannot simulate the requested Excel gases."""
    needed = excel_gases_in(gas_types)
    if not needed:
        return
    binary = find_rt_server()
    if gaden_binary_supports_excel_gases(binary):
        return
    where = str(binary) if binary is not None else "(odor_gaden_rt rt_server not found)"
    raise RuntimeError(
        "GADEN cannot simulate "
        + ", ".join(needed)
        + f" (IDs 14–17). Binary {where} was built without those gas types. "
        "Apply patches/gaden_core_add_excel_gases.patch, rebuild gaden_common "
        "and odor_gaden_rt, then source setup/activate.sh before collecting."
    )

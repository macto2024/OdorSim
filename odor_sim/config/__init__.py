"""odor_sim.config: VOC recipe table, gas types, and the robosuite<->GADEN frame map.

NOTE: ``get_recipe`` / ``list_recipes`` live in :mod:`odor_sim.config.recipes`
and are intentionally NOT re-exported here. ``recipes`` depends on
:mod:`odor_sim.envs.odor_profile`, so importing it eagerly from this package
``__init__`` would create an import cycle (config -> envs -> config). Import
them directly from ``odor_sim.config.recipes`` instead.
"""

from odor_sim.config.frame_map import FrameMap
from odor_sim.config.gas_types import (
    GADEN_GAS_NAMES,
    MOX_SUPPORTED,
    PID_SUPPORTED,
    GasType,
    gaden_name,
    parse_gas_type,
)

__all__ = [
    "FrameMap",
    "GasType",
    "GADEN_GAS_NAMES",
    "MOX_SUPPORTED",
    "PID_SUPPORTED",
    "gaden_name",
    "parse_gas_type",
]

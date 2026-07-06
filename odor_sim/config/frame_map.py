"""Frame map between robosuite world coordinates and the GADEN environment.

Single affine transform::

    p_gaden = R @ p_robosuite + t

Both frames are metric (meters). The default map uses identity rotation and a
translation chosen so the robosuite table workspace lands on the Phase 2 test
source location in the 10x6 room, with robosuite +x aligned to GADEN +x (the
uniform wind direction).

GADEN's cell size (and, when available, the occupancy-grid origin/bounds) are
read from the scenario's ``config.yaml`` so the map stays tied to a concrete
scenario.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

# robosuite table center defaults to (0, 0, 0.8); map it to this GADEN point,
# which is where the Phase 2 test source sits in the 10x6 room.
_DEFAULT_ROBOSUITE_ANCHOR = np.array([0.0, 0.0, 0.8])
_DEFAULT_GADEN_ANCHOR = np.array([2.0, 3.0, 0.5])


@dataclass
class FrameMap:
    """Affine map robosuite <-> GADEN (``p_gaden = R @ p_robosuite + t``)."""

    rotation: np.ndarray = field(default_factory=lambda: np.eye(3))
    translation: np.ndarray = field(
        default_factory=lambda: _DEFAULT_GADEN_ANCHOR - _DEFAULT_ROBOSUITE_ANCHOR
    )
    cell_size: float = 0.1
    gaden_min: "np.ndarray | None" = None
    gaden_max: "np.ndarray | None" = None

    def __post_init__(self) -> None:
        self.rotation = np.asarray(self.rotation, dtype=float).reshape(3, 3)
        self.translation = np.asarray(self.translation, dtype=float).reshape(3)
        if self.gaden_min is not None:
            self.gaden_min = np.asarray(self.gaden_min, dtype=float).reshape(3)
        if self.gaden_max is not None:
            self.gaden_max = np.asarray(self.gaden_max, dtype=float).reshape(3)

    def robosuite_to_gaden(self, p) -> np.ndarray:
        p = np.asarray(p, dtype=float).reshape(3)
        return self.rotation @ p + self.translation

    def gaden_to_robosuite(self, p) -> np.ndarray:
        p = np.asarray(p, dtype=float).reshape(3)
        return self.rotation.T @ (p - self.translation)

    def is_in_gaden_bounds(self, p) -> bool:
        """True if a GADEN-frame point lies in the known room bounds (if set)."""
        if self.gaden_min is None or self.gaden_max is None:
            return True
        p = np.asarray(p, dtype=float).reshape(3)
        return bool(np.all(p >= self.gaden_min) and np.all(p <= self.gaden_max))

    @classmethod
    def from_scenario(
        cls,
        scenario_config_dir: "str | Path",
        rotation=None,
        translation=None,
        robosuite_anchor=_DEFAULT_ROBOSUITE_ANCHOR,
        gaden_anchor=_DEFAULT_GADEN_ANCHOR,
    ) -> "FrameMap":
        """Build a FrameMap from a GADEN scenario config directory.

        Reads ``config.yaml`` for ``cell_size``. If ``translation`` is not given
        it is derived so that ``robosuite_anchor`` maps to ``gaden_anchor`` under
        the (identity by default) rotation.
        """
        cfg_dir = Path(scenario_config_dir)
        cfg_path = cfg_dir / "config.yaml"
        cell_size = 0.1
        if cfg_path.exists():
            with open(cfg_path, "r") as f:
                cfg = yaml.safe_load(f) or {}
            cell_size = float(cfg.get("cell_size", 0.1))

        R = np.eye(3) if rotation is None else np.asarray(rotation, dtype=float).reshape(3, 3)
        if translation is None:
            t = np.asarray(gaden_anchor, dtype=float) - R @ np.asarray(robosuite_anchor, dtype=float)
        else:
            t = np.asarray(translation, dtype=float)

        gmin, gmax = _read_occupancy_bounds(cfg_dir)
        return cls(rotation=R, translation=t, cell_size=cell_size, gaden_min=gmin, gaden_max=gmax)


def _read_occupancy_bounds(cfg_dir: Path):
    """Best-effort read of GADEN occupancy-grid bounds from OccupancyGrid3D.csv.

    The grid is only produced after preprocessing, so this returns (None, None)
    if the file is absent (the common Phase 3 case, before running the node).
    """
    for candidate in cfg_dir.rglob("OccupancyGrid3D.csv"):
        try:
            header = {}
            with open(candidate, "r") as f:
                for _ in range(4):
                    line = f.readline().strip()
                    if not line:
                        break
                    parts = line.split()
                    if len(parts) == 2:
                        header[parts[0]] = float(parts[1])
            if {"env_min_x", "env_min_y", "env_min_z", "env_max_x", "env_max_y", "env_max_z"} <= header.keys():
                gmin = np.array([header["env_min_x"], header["env_min_y"], header["env_min_z"]])
                gmax = np.array([header["env_max_x"], header["env_max_y"], header["env_max_z"]])
                return gmin, gmax
        except (OSError, ValueError):
            continue
    return None, None

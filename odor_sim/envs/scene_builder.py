"""Flatten OdorObjects into a GADEN source list + object->source index map.

This is the bridge-facing bookkeeping layer of Phase 3. It does NOT talk to
ROS. It:

  * flattens every object's VOC components into a single ordered source list
    (the order defines the source index used by ``/gaden/source_poses``),
  * keeps ``object_id -> [source indices]`` so Phase 4 can move all of an
    object's sources together,
  * (optionally) exports a GADEN scenario (per-source ``sim.yaml`` +
    ``scene.yaml``) that ``odor_gaden_rt`` can load, so the C++ source list
    matches this Python source list index-for-index.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from odor_sim.config.frame_map import FrameMap
from odor_sim.envs.odor_object import OdorObject
from odor_sim.envs.odor_profile import VOCComponent

# GADEN per-source simulation defaults (mirror scenarios/10x6_uniform sim.yaml).
_SIM_DEFAULTS = {
    "deltaTime": 0.05,
    "windIterationDeltaTime": 1,
    "temperature": 298,
    "pressure": 1,
    "filamentInitialSigma": 10,
    "filamentGrowthGamma": 10,
    "filamentNoise_std": 0.02,
    "expectedNumIterations": 24000,
    "saveResults": False,
    "saveDeltaTime": 0.5,
    "preCalculateConcentrations": False,
}


@dataclass
class SourceEntry:
    """One flattened GADEN source (one VOC of one object)."""

    index: int
    object_id: str
    voc_index: int  # position within the object's OdorProfile
    component: VOCComponent

    @property
    def gas_type(self):
        return self.component.gas_type


class SceneBuilder:
    """Flattens odor objects into GADEN sources and tracks the mapping.

    Args:
        frame_map: robosuite <-> GADEN transform. Used both when exporting
            initial source positions and when converting live object poses to
            GADEN-frame source poses (Phase 4).
    """

    def __init__(self, frame_map: "FrameMap | None" = None):
        self.frame_map = frame_map or FrameMap()
        self._sources: list[SourceEntry] = []
        self._object_to_sources: dict[str, list[int]] = {}
        self._objects: dict[str, OdorObject] = {}

    def add_object(self, obj: OdorObject) -> list[int]:
        """Register an OdorObject; returns the source indices it occupies."""
        if obj.object_id in self._object_to_sources:
            raise ValueError(f"Object {obj.object_id!r} already added")
        indices: list[int] = []
        for voc_index, comp in enumerate(obj.odor_profile):
            idx = len(self._sources)
            self._sources.append(
                SourceEntry(index=idx, object_id=obj.object_id, voc_index=voc_index, component=comp)
            )
            indices.append(idx)
        self._object_to_sources[obj.object_id] = indices
        self._objects[obj.object_id] = obj
        return indices

    def add_objects(self, objects) -> None:
        for obj in objects:
            self.add_object(obj)

    @property
    def num_sources(self) -> int:
        return len(self._sources)

    @property
    def sources(self) -> list[SourceEntry]:
        return list(self._sources)

    @property
    def object_to_sources(self) -> dict[str, list[int]]:
        return {k: list(v) for k, v in self._object_to_sources.items()}

    def sources_for(self, object_id: str) -> list[int]:
        return list(self._object_to_sources[object_id])

    def gas_types(self) -> list:
        return [s.gas_type for s in self._sources]

    def gaden_source_positions(self, object_world_poses: dict) -> np.ndarray:
        """Compute GADEN-frame source positions from object world poses.

        Args:
            object_world_poses: ``{object_id: (pos3, mat3x3 or None)}`` in the
                robosuite world frame.

        Returns:
            (num_sources, 3) array of GADEN-frame source positions, ordered by
            source index (i.e. ready to become a ``PoseArray`` for
            ``/gaden/source_poses``).
        """
        out = np.zeros((self.num_sources, 3), dtype=float)
        for object_id, indices in self._object_to_sources.items():
            if object_id not in object_world_poses:
                raise KeyError(f"Missing world pose for object {object_id!r}")
            pos, mat = object_world_poses[object_id]
            obj = self._objects[object_id]
            world_positions = obj.source_positions(pos, mat)
            for local_i, src_idx in enumerate(indices):
                out[src_idx] = self.frame_map.robosuite_to_gaden(world_positions[local_i])
        return out

    def export_gaden_scene(
        self,
        config_dir: "str | Path",
        scene_id: str = "scene1",
        initial_positions_gaden=None,
        wind_files: str = "../../wind_simulations/uniform/wind",
    ) -> Path:
        """Write per-source ``sim.yaml`` + a ``scene.yaml`` for odor_gaden_rt.

        The generated source list is index-aligned with :attr:`sources`, so the
        C++ node's source *i* is exactly this builder's source *i*.

        Args:
            config_dir: GADEN environment configuration directory (the one that
                contains ``config.yaml``); ``simulations/`` and ``scenes/`` are
                created/overwritten under it.
            scene_id: name of the scene file to write (``scenes/<scene_id>.yaml``).
            initial_positions_gaden: optional (num_sources, 3) initial source
                positions in GADEN frame. Defaults to the frame-map image of the
                robosuite origin for every source.
        """
        config_dir = Path(config_dir)
        sims_dir = config_dir / "simulations"
        scenes_dir = config_dir / "scenes"
        sims_dir.mkdir(parents=True, exist_ok=True)
        scenes_dir.mkdir(parents=True, exist_ok=True)

        if initial_positions_gaden is None:
            default_pos = self.frame_map.robosuite_to_gaden([0.0, 0.0, 0.8])
            initial_positions_gaden = np.tile(default_pos, (self.num_sources, 1))
        initial_positions_gaden = np.asarray(initial_positions_gaden, dtype=float).reshape(-1, 3)

        scene_sims = []
        for src in self._sources:
            sim_id = f"src{src.index:03d}_{src.object_id}_{src.component.gaden_gas_name()}"
            pos = initial_positions_gaden[src.index].tolist()
            sim_doc = {
                "source": {
                    "sourceType": "point",
                    "position": [round(float(v), 4) for v in pos],
                    "gasType": int(src.gas_type),
                },
                "filamentPPMcenter": round(float(src.component.filament_ppm_center), 3),
                "numFilaments_sec": int(src.component.num_filaments_sec),
                "windLooping": {"loop": False, "from": 0, "to": 0},
                **_SIM_DEFAULTS,
            }
            sim_path = sims_dir / sim_id / "sim.yaml"
            sim_path.parent.mkdir(parents=True, exist_ok=True)
            with open(sim_path, "w") as f:
                yaml.safe_dump(sim_doc, f, sort_keys=False)
            scene_sims.append({"sim": sim_id, "gas_color": _gas_color(src.gas_type)})

        scene_doc = {
            "playback_initial_iteration": 0,
            "playback_loop": {"loop": False, "from": 0, "to": 0},
            "simulations": scene_sims,
        }
        scene_path = scenes_dir / f"{scene_id}.yaml"
        with open(scene_path, "w") as f:
            yaml.safe_dump(scene_doc, f, sort_keys=False)
        return scene_path


def _gas_color(gas_type) -> list:
    """Deterministic-ish RGB per gas for RViz, matching GADEN's [r,g,b]."""
    palette = {
        0: [0.0, 0.8, 0.2],   # ethanol
        1: [0.8, 0.8, 0.0],   # methane
        2: [0.0, 0.6, 0.9],   # hydrogen
        3: [0.6, 0.2, 0.8],   # propanol
        4: [0.9, 0.9, 0.2],   # chlorine
        5: [0.2, 0.9, 0.9],   # fluorine
        6: [0.9, 0.4, 0.1],   # acetone
    }
    return palette.get(int(gas_type), [0.7, 0.7, 0.7])

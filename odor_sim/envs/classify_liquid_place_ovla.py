"""ClassifyLiquidPlaceOVLA: vision-similar cups, odor is the remaining cue.

Two identical porcelain (white) mugs plus a plate in tight spawn slots, using
the closer OdorPlace camera so both cups and the gripper stay in frame. Liquid
identity is randomized onto left/right slots so neither appearance nor position
leaks the instructed VOC.
"""

from __future__ import annotations

import numpy as np
from robosuite.utils.placement_samplers import (
    SequentialCompositeSampler,
    UniformRandomSampler,
)

from odor_sim.config.gas_types import EXCEL_VOC_GAS_NAMES
from odor_sim.envs.classify_liquid_place import ClassifyLiquidPlace
from odor_sim.envs.odor_object import OdorXMLObject, make_odor_object

CUP_CATALOG = "porcelain_mug"


class ClassifyLiquidPlaceOVLA(ClassifyLiquidPlace):
    """Smell-conditioned place task with matched white cups and tight layout."""

    agentview_pos = (0.5, 0.0, 1.35)
    agentview_fovy = 45.0

    plate_x_range = (-0.06, 0.06)
    plate_y_range = (-0.06, 0.06)
    # Two small cup slots, split left/right so XY does not identify the liquid.
    cup_left_x_range = (0.12, 0.20)
    cup_left_y_range = (-0.16, -0.08)
    cup_right_x_range = (0.12, 0.20)
    cup_right_y_range = (0.08, 0.16)

    @property
    def gas_types(self) -> list[str]:
        """Fixed Excel / ScentFormer ppm columns, independent of the inert plate."""
        return list(EXCEL_VOC_GAS_NAMES)

    def _porcelain_cup(self):
        matches = [
            cup
            for cup in self._cup_pool
            if CUP_CATALOG in str(cup.xml_path)
        ]
        if not matches:
            raise RuntimeError(
                f"{CUP_CATALOG} missing from liquid catalog cup pool"
            )
        return matches[0]

    def _make_odor_objects(self):
        """Both liquids in porcelain_mug; plate is the catalog place target."""
        from odor_sim.config.recipes import get_recipe

        cup = self._porcelain_cup()
        objects = []
        for name in self._object_names:
            profile = get_recipe(self._liquid_recipes[name], path=self._recipe_path)
            objects.append(
                OdorXMLObject(
                    name=name,
                    odor_profile=profile,
                    xml_path=cup.xml_path,
                    rotation=cup.rotation,
                    rotation_axis=cup.rotation_axis,
                )
            )
        objects.append(
            make_odor_object(
                self._place_target_name,
                instance_name=self._place_target_name,
                rng=self.rng,
            )
        )
        return objects

    def _cup_slot_ranges(self) -> list[tuple[list[float], list[float]]]:
        return [
            (list(self.cup_left_x_range), list(self.cup_left_y_range)),
            (list(self.cup_right_x_range), list(self.cup_right_y_range)),
        ]

    def _build_placement_initializer(self) -> SequentialCompositeSampler:
        """Plate near center; cups randomly assigned to left/right tight slots."""
        sampler = SequentialCompositeSampler(name="ClassifyLiquidPlaceOVLASampler")
        plate_name = self._place_target_name
        cups = [obj for obj in self.odor_objects if obj.object_id != plate_name]
        order = list(range(len(cups)))
        self.rng.shuffle(order)
        slots = self._cup_slot_ranges()
        if len(cups) > len(slots):
            raise ValueError(
                f"ClassifyLiquidPlaceOVLA supports at most {len(slots)} cups, "
                f"got {len(cups)}"
            )

        def _append(obj, x_range, y_range):
            rotation = getattr(obj, "rotation", None)
            rotation_axis = getattr(obj, "rotation_axis", "z") or "z"
            sampler.append_sampler(
                UniformRandomSampler(
                    name=f"{obj.object_id}_sampler",
                    mujoco_objects=obj,
                    x_range=x_range,
                    y_range=y_range,
                    rotation=rotation,
                    rotation_axis=rotation_axis,
                    ensure_object_boundary_in_range=False,
                    ensure_valid_placement=True,
                    reference_pos=self.table_offset,
                    z_offset=0.01,
                    rng=self.rng,
                )
            )

        plate = next(obj for obj in self.odor_objects if obj.object_id == plate_name)
        _append(plate, list(self.plate_x_range), list(self.plate_y_range))
        for slot_i, cup_i in enumerate(order):
            x_range, y_range = slots[slot_i]
            _append(cups[cup_i], x_range, y_range)
        return sampler

    @property
    def cup_catalog_name(self) -> str:
        return CUP_CATALOG

    def spawn_layout(self) -> dict:
        """World XY of cups and plate after reset (for episode metadata)."""
        poses = self.get_object_world_poses()
        layout = {}
        for obj in self.odor_objects:
            pos = np.asarray(poses[obj.object_id][0], dtype=float).ravel()[:3]
            layout[obj.object_id] = {
                "pos": [float(pos[0]), float(pos[1]), float(pos[2])],
                "catalog": (
                    CUP_CATALOG
                    if obj.object_id != self._place_target_name
                    else self._place_target_name
                ),
            }
        return layout

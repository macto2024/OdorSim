"""OdorPlace: vision-only pick a catalog cup and place it on a plate.

Spawns N catalog mug meshes (unique instance ids) plus one place target
(default ``plate``). Success is LIBERO-style ``On``: target cup above the
plate, in contact, XY centers within ``on_xy_threshold``. No liquids / odor
classification — intended for OpenVLA backbone fine-tuning demos.
"""

from __future__ import annotations

import numpy as np
from robosuite.utils.placement_samplers import (
    SequentialCompositeSampler,
    UniformRandomSampler,
)

from odor_sim.envs.base import OdorManipulationEnv
from odor_sim.envs.odor_object import make_odor_object

# Catalog mug names used by OdorPlace / scripted collection.
MUG_POOL = ("porcelain_mug", "white_yellow_mug", "red_coffee_mug")


def nice_mug_name(catalog: str) -> str:
    """Human-readable mug name for language instructions (e.g. red_coffee_mug → red coffee mug)."""
    n = str(catalog).replace("_", " ")
    if n.endswith(" small"):
        n = n[: -len(" small")]
    return n


def _default_instruction(cup_catalog_names: list[str], target_catalog: str, place_target: str) -> str:
    """Name the cup type when unique among spawned cups; else generic 'cup'."""
    place = place_target.replace("_", " ")
    if cup_catalog_names.count(target_catalog) == 1:
        return f"put the {nice_mug_name(target_catalog)} on the {place}"
    return f"put the cup on the {place}"


class OdorPlace(OdorManipulationEnv):
    """Pick a target catalog cup and place it on a single plate.

    Args:
        cups: catalog mug name(s) to spawn (see ``objects.yaml`` / ``MUG_POOL``).
            A string or list; length is the cup count. Instance ids are
            ``{catalog}_{i}`` so duplicates are allowed.
        target_cup: which cup to pick. Accepts:
            * catalog name if it appears exactly once among ``cups``
            * instance id (``porcelain_mug_0``)
            * integer index into ``cups``
            Defaults to the first cup.
        place_target: catalog place-surface name (default ``"plate"``).
        instruction: language label; generated if empty.
    """

    # Max XY distance (m) between cup and plate body centers for On-success.
    # Slightly looser than ClassifyLiquidPlace (0.03) — mug meshes are wide.
    on_xy_threshold = 0.05

    # Stock robosuite framing rather than the pulled-back base default. The
    # spawn box below is deliberately small enough to stay in frame here, and
    # the closer camera buys ~40% more pixels per centimetre of height (2.8
    # vs 2.0 px/cm at 256²) — the cue the policy uses to judge when the
    # gripper is low enough to close on the mug.
    agentview_pos = (0.5, 0.0, 1.35)
    agentview_fovy = 45.0

    # Spawn bands in table-frame XY (m), sized from the agentview frustum so
    # both the mug and the gripper hovering above it stay in view for the whole
    # episode. Cups sit forward of the plate, toward the camera.
    plate_x_range = (-0.06, 0.06)
    plate_y_range = (-0.06, 0.06)
    cup_x_range = (0.12, 0.22)
    cup_y_range = (-0.17, 0.17)

    def __init__(
        self,
        robots="Panda",
        cups=None,
        target_cup: "str | int | None" = None,
        place_target: str = "plate",
        instruction: str = "",
        scenario_config_dir=None,
        cup_x_range=None,
        cup_y_range=None,
        plate_x_range=None,
        plate_y_range=None,
        **kwargs,
    ):
        from odor_sim.config.objects import get_object_spec

        if cups is None:
            cups_list = ["porcelain_mug"]
        elif isinstance(cups, str):
            cups_list = [cups]
        else:
            cups_list = list(cups)
        if not cups_list:
            raise ValueError("OdorPlace requires at least one cup")
        if not place_target:
            raise ValueError("OdorPlace requires a place_target")
        get_object_spec(place_target)
        for name in cups_list:
            get_object_spec(name)

        self._cup_catalog_names = list(cups_list)
        self._cup_instance_ids = [f"{name}_{i}" for i, name in enumerate(cups_list)]
        self._place_target_name = place_target

        for attr, value in (
            ("cup_x_range", cup_x_range),
            ("cup_y_range", cup_y_range),
            ("plate_x_range", plate_x_range),
            ("plate_y_range", plate_y_range),
        ):
            if value is not None:
                lo, hi = (float(v) for v in value)
                setattr(self, attr, (lo, hi))

        self._target_index = self._resolve_target_index(target_cup)
        self._target_instance_id = self._cup_instance_ids[self._target_index]
        self._target_catalog_name = self._cup_catalog_names[self._target_index]

        if not instruction:
            instruction = _default_instruction(
                self._cup_catalog_names,
                self._target_catalog_name,
                place_target,
            )

        super().__init__(
            robots=robots,
            instruction=instruction,
            scenario_config_dir=scenario_config_dir,
            **kwargs,
        )

    def _resolve_target_index(self, target_cup: "str | int | None") -> int:
        if target_cup is None:
            return 0
        if isinstance(target_cup, (int, np.integer)):
            idx = int(target_cup)
            if idx < 0 or idx >= len(self._cup_catalog_names):
                raise ValueError(
                    f"target_cup index {idx} out of range for "
                    f"{len(self._cup_catalog_names)} cups"
                )
            return idx
        target = str(target_cup)
        if target in self._cup_instance_ids:
            return self._cup_instance_ids.index(target)
        matches = [i for i, n in enumerate(self._cup_catalog_names) if n == target]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise ValueError(
                f"target_cup catalog name {target!r} is ambiguous among "
                f"{self._cup_catalog_names!r}; pass an instance id "
                f"(e.g. {target}_0) or an index"
            )
        raise ValueError(
            f"target_cup {target!r} not among cups {self._cup_catalog_names!r} "
            f"or instance ids {self._cup_instance_ids!r}"
        )

    def _make_odor_objects(self):
        objects = [
            make_odor_object(name, instance_name=inst, rng=self.rng)
            for name, inst in zip(self._cup_catalog_names, self._cup_instance_ids)
        ]
        objects.append(
            make_odor_object(
                self._place_target_name,
                instance_name=self._place_target_name,
                rng=self.rng,
            )
        )
        return objects

    def _build_placement_initializer(self) -> SequentialCompositeSampler:
        """Plate near center; cups in a forward reachable band (non-overlapping).

        The bands come from the class attributes rather than ``spawn_half``:
        they are sized to the agentview frustum, not to the table, so widening
        them silently pushes mugs out of frame. ``ensure_valid_placement``
        still keeps centers ~0.16 m apart (mug/plate horizontal_radius ~0.08).
        """
        plate_x_range = list(self.plate_x_range)
        plate_y_range = list(self.plate_y_range)
        cup_x_range = list(self.cup_x_range)
        cup_y_range = list(self.cup_y_range)
        sampler = SequentialCompositeSampler(name="OdorPlaceSampler")
        for obj in self.odor_objects:
            rotation = getattr(obj, "rotation", None)
            rotation_axis = getattr(obj, "rotation_axis", "z") or "z"
            if obj.object_id == self._place_target_name:
                x_range, y_range = plate_x_range, plate_y_range
            else:
                x_range, y_range = cup_x_range, cup_y_range
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
        return sampler

    @property
    def object_names(self) -> list[str]:
        """Instance ids of spawned cups (place target excluded), spawn order."""
        return list(self._cup_instance_ids)

    @property
    def cup_catalog_names(self) -> list[str]:
        """Catalog names of spawned cups (may contain duplicates)."""
        return list(self._cup_catalog_names)

    @property
    def target_object_name(self) -> str:
        """Instance id of the cup that must be placed."""
        return self._target_instance_id

    @property
    def target_catalog_name(self) -> str:
        """Catalog name of the target cup type."""
        return self._target_catalog_name

    @property
    def target_object(self):
        """The cup object that must end On the place target."""
        for obj in self.odor_objects:
            if obj.object_id == self._target_instance_id:
                return obj
        raise RuntimeError(
            f"target cup {self._target_instance_id!r} not among odor_objects"
        )

    @property
    def place_target_name(self) -> str:
        return self._place_target_name

    @property
    def place_target(self):
        for obj in self.odor_objects:
            if obj.object_id == self._place_target_name:
                return obj
        raise RuntimeError(
            f"place_target {self._place_target_name!r} not among odor_objects"
        )

    def _object_body_pos(self, object_id: str) -> np.ndarray:
        return np.array(self.sim.data.body_xpos[self._odor_body_ids[object_id]])

    def reward(self, action=None):
        reward = 0.0
        if self._check_success():
            reward = 2.25
        elif self.reward_shaping:
            dist = self._gripper_to_target(
                gripper=self.robots[0].gripper[self.robots[0].arms[0]],
                target=self.target_object.root_body,
                target_type="body",
                return_distance=True,
            )
            reward += 1 - np.tanh(10.0 * dist)
            if self._check_grasp(
                gripper=self.robots[0].gripper[self.robots[0].arms[0]],
                object_geoms=self.target_object,
            ):
                reward += 0.25
        if self.reward_scale is not None:
            reward *= self.reward_scale / 2.25
        return reward

    def _check_success(self):
        """True when the target cup is On the place target (LIBERO-style)."""
        cup = self.target_object
        surface = self.place_target
        cup_pos = self._object_body_pos(cup.object_id)
        surface_pos = self._object_body_pos(surface.object_id)
        return (
            surface_pos[2] <= cup_pos[2]
            and self.check_contact(cup, surface)
            and (
                np.linalg.norm(cup_pos[:2] - surface_pos[:2]) < self.on_xy_threshold
            )
        )

"""ClassifyLiquidPlace: pick a liquid cup by smell and place it on a target.

Combines :class:`~odor_sim.envs.classify_liquid.ClassifyLiquid` (liquid odor
class, random cup mesh) with a LIBERO-style ``On`` success check: the target
liquid's cup must rest on a catalog place target (default ``plate``).
"""

from __future__ import annotations

import numpy as np

from odor_sim.envs.classify_liquid import ClassifyLiquid
from odor_sim.envs.odor_object import make_odor_object


class ClassifyLiquidPlace(ClassifyLiquid):
    """Place the target liquid's cup onto a place-target object.

    Args:
        liquids / target_liquid / instruction / liquid_catalog_path /
            recipe_path: same as :class:`ClassifyLiquid`. Default instruction is
            ``put the cup of {liquid} on the {place_target}``.
        place_target: catalog object name for the put-on surface (see
            ``odor_sim/config/objects.yaml``). Defaults to ``"plate"``. Must not
            collide with a spawned liquid name. The place target uses its
            catalog recipe (``plate`` is inert / odorless).
    """

    # Max XY distance (m) between cup and place-target body centers (LIBERO On).
    on_xy_threshold = 0.03

    def __init__(
        self,
        robots="Panda",
        liquids=None,
        target_liquid: "str | None" = None,
        place_target: str = "plate",
        instruction: str = "",
        scenario_config_dir=None,
        liquid_catalog_path=None,
        recipe_path=None,
        **kwargs,
    ):
        from odor_sim.config.objects import get_object_spec

        if not place_target:
            raise ValueError("ClassifyLiquidPlace requires a place_target")
        # Validate catalog entry up front (unknown name fails at construction).
        get_object_spec(place_target)
        self._place_target_name = place_target

        if liquids is None:
            liquids_list = ["water"]
        elif isinstance(liquids, str):
            liquids_list = [liquids]
        else:
            liquids_list = list(liquids)
        if place_target in liquids_list:
            raise ValueError(
                f"place_target {place_target!r} collides with a spawned liquid "
                f"name {liquids_list!r}"
            )

        tgt = target_liquid if target_liquid is not None else liquids_list[0]
        if not instruction:
            instruction = (
                f"put the cup of {tgt.replace('_', ' ')} on the "
                f"{place_target.replace('_', ' ')}"
            )

        super().__init__(
            robots=robots,
            liquids=liquids,
            target_liquid=target_liquid,
            instruction=instruction,
            scenario_config_dir=scenario_config_dir,
            liquid_catalog_path=liquid_catalog_path,
            recipe_path=recipe_path,
            **kwargs,
        )

    def _make_odor_objects(self):
        """Liquid cups (random meshes) plus the place-target catalog object."""
        objects = super()._make_odor_objects()
        objects.append(
            make_odor_object(
                self._place_target_name,
                instance_name=self._place_target_name,
                rng=self.rng,
            )
        )
        return objects

    @property
    def place_target_name(self) -> str:
        """Catalog name of the put-on target."""
        return self._place_target_name

    @property
    def place_target(self):
        """The object the target liquid cup must be placed on."""
        for obj in self.odor_objects:
            if obj.object_id == self._place_target_name:
                return obj
        raise RuntimeError(
            f"place_target {self._place_target_name!r} not among odor_objects"
        )

    def _object_body_pos(self, object_id: str) -> np.ndarray:
        return np.array(self.sim.data.body_xpos[self._odor_body_ids[object_id]])

    def _check_success(self):
        """True when the target liquid cup is On the place target (LIBERO-style)."""
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

"""OdorLift: a Lift-style task with an odor-emitting cube.

Mirrors robosuite's Lift, but the cube is an :class:`OdorObject` carrying a
multi-VOC :class:`OdorProfile`, and the env exposes the EE odor-sensor pose and
the object->source map for the Phase 4 bridge.
"""

from __future__ import annotations

import numpy as np

from odor_sim.envs.base import OdorManipulationEnv
from odor_sim.envs.odor_object import OdorObject
from odor_sim.envs.odor_profile import OdorProfile


class OdorLift(OdorManipulationEnv):
    """Lift a single odor-emitting cube off the table.

    Args:
        recipe: name of a VOC recipe (see ``odor_sim/config/voc_recipes.yaml``)
            OR an explicit :class:`OdorProfile`. Defaults to ``"ripe_fruit"``.
        instruction: task language label. A sensible default is generated from
            the recipe if not provided.
        cube_size: (half-x, half-y, half-z) of the cube.
    """

    def __init__(
        self,
        robots="Panda",
        recipe="ripe_fruit",
        instruction: str = "",
        cube_size=(0.022, 0.022, 0.022),
        scenario_config_dir=None,
        **kwargs,
    ):
        if isinstance(recipe, OdorProfile):
            self._profile = recipe
            self._recipe_name = "custom"
        else:
            from odor_sim.config.recipes import get_recipe

            self._profile = get_recipe(recipe)
            self._recipe_name = recipe
        self._cube_size = cube_size

        if not instruction:
            instruction = f"pick up the {self._recipe_name.replace('_', ' ')} object"

        super().__init__(
            robots=robots,
            instruction=instruction,
            scenario_config_dir=scenario_config_dir,
            **kwargs,
        )

    def _make_odor_objects(self):
        cube = OdorObject(
            name="odor_cube",
            odor_profile=self._profile,
            size=self._cube_size,
            rgba=(0.7, 0.35, 0.15, 1.0),
            rng=self.rng,
        )
        return [cube]

    @property
    def cube(self) -> OdorObject:
        return self.odor_objects[0]

    def reward(self, action=None):
        reward = 0.0
        if self._check_success():
            reward = 2.25
        elif self.reward_shaping:
            dist = self._gripper_to_target(
                gripper=self.robots[0].gripper[self.robots[0].arms[0]],
                target=self.cube.root_body,
                target_type="body",
                return_distance=True,
            )
            reward += 1 - np.tanh(10.0 * dist)
            if self._check_grasp(
                gripper=self.robots[0].gripper[self.robots[0].arms[0]], object_geoms=self.cube
            ):
                reward += 0.25
        if self.reward_scale is not None:
            reward *= self.reward_scale / 2.25
        return reward

    def _check_success(self):
        cube_height = self.sim.data.body_xpos[self._odor_body_ids[self.cube.object_id]][2]
        table_height = self.model.mujoco_arena.table_offset[2]
        return cube_height > table_height + 0.04

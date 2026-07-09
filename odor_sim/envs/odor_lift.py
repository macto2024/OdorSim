"""OdorLift: a Lift-style task with odor-emitting catalog objects.

Mirrors robosuite's Lift, but the object(s) come from the OdorSim object catalog
(:mod:`odor_sim.config.objects`) -- each name bundles geometry (a primitive box
or a mesh) with its own dedicated VOC :class:`OdorProfile`. The task lifts the
first object; extra objects act as odor-emitting distractors. The env exposes
the EE odor-sensor pose and the object->source map for the Phase 4 bridge.
"""

from __future__ import annotations

import numpy as np

from odor_sim.envs.base import OdorManipulationEnv
from odor_sim.envs.odor_object import OdorObject, make_odor_object


class OdorLift(OdorManipulationEnv):
    """Lift a catalog object off the table.

    Args:
        objects: catalog object name(s) to spawn (see
            ``odor_sim/config/objects.yaml``). A single string or a list; by
            default the first entry is the lift target, extras are odor
            distractors. Defaults to ``["odor_cube"]``. Each object uses its
            own dedicated catalog recipe (there is no per-task recipe override).
        target_object: which spawned object is the lift target. Must be one of
            the names in ``objects``. Defaults to the first spawned object.
        instruction: task language label. Generated from the target object name
            if not provided.
    """

    def __init__(
        self,
        robots="Panda",
        objects=None,
        target_object: "str | None" = None,
        instruction: str = "",
        scenario_config_dir=None,
        **kwargs,
    ):
        if objects is None:
            objects = ["odor_cube"]
        elif isinstance(objects, str):
            objects = [objects]
        self._object_names = list(objects)
        if not self._object_names:
            raise ValueError("OdorLift requires at least one object")

        if target_object is None:
            self._target_index = 0
        elif target_object in self._object_names:
            self._target_index = self._object_names.index(target_object)
        else:
            raise ValueError(
                f"target_object {target_object!r} is not among the spawned "
                f"objects {self._object_names!r}"
            )
        self._target_name = self._object_names[self._target_index]

        if not instruction:
            instruction = f"pick up the {self._target_name.replace('_', ' ')}"

        super().__init__(
            robots=robots,
            instruction=instruction,
            scenario_config_dir=scenario_config_dir,
            **kwargs,
        )

    def _make_odor_objects(self):
        return [make_odor_object(name, rng=self.rng) for name in self._object_names]

    @property
    def object_names(self) -> list[str]:
        """Catalog names of the spawned objects (in spawn order)."""
        return list(self._object_names)

    @property
    def target_object_name(self) -> str:
        """Catalog name of the lift target."""
        return self._target_name

    @property
    def target_object(self):
        """The object the task rewards lifting."""
        return self.odor_objects[self._target_index]

    @property
    def cube(self) -> OdorObject:
        """Backward-compatible alias for :attr:`target_object`."""
        return self.odor_objects[self._target_index]

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

    # Height (m) the target must rise above its resting pose to count as lifted.
    lift_height = 0.04

    def _check_success(self):
        object_id = self.target_object.object_id
        return self.object_body_z(object_id) > self.object_rest_z(object_id) + self.lift_height

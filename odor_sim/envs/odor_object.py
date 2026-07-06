"""OdorObject: a robosuite object that also carries an OdorProfile.

Kept deliberately thin: it is a normal :class:`BoxObject` (so it drops straight
into robosuite's placement/observable machinery) with an attached
:class:`OdorProfile`. The scene builder reads ``.odor_profile`` to expand the
object into one GADEN source per VOC component.
"""

from __future__ import annotations

import numpy as np
from robosuite.models.objects import BoxObject

from odor_sim.envs.odor_profile import OdorProfile


class OdorObject(BoxObject):
    """A box-shaped object emitting a multi-VOC :class:`OdorProfile`.

    Args:
        name: unique object name (also the ``object_id`` used in the
            object -> source map).
        odor_profile: the mixture this object emits.
        size: (half-x, half-y, half-z) box size in meters.
        rgba / material / density / friction: passed through to BoxObject.
    """

    def __init__(
        self,
        name: str,
        odor_profile: OdorProfile,
        size=(0.022, 0.022, 0.022),
        rgba=(0.55, 0.35, 0.15, 1.0),
        material=None,
        density=None,
        friction=None,
        rng=None,
    ):
        if not isinstance(odor_profile, OdorProfile):
            raise TypeError("odor_profile must be an OdorProfile")
        self.odor_profile = odor_profile
        super().__init__(
            name=name,
            size=size,
            rgba=rgba,
            material=material,
            density=density,
            friction=friction,
            rng=rng,
        )

    @property
    def object_id(self) -> str:
        return self.name

    @property
    def num_sources(self) -> int:
        return self.odor_profile.num_sources

    def source_positions(self, body_pos, body_mat=None) -> list[np.ndarray]:
        """World positions of this object's VOC sources.

        Args:
            body_pos: (3,) world position of the object body.
            body_mat: optional (3, 3) world rotation of the object body. If
                given, per-VOC local offsets are rotated into world frame.

        Returns:
            list of (3,) world positions, one per VOC component (in profile
            order).
        """
        body_pos = np.asarray(body_pos, dtype=float).reshape(3)
        R = np.asarray(body_mat, dtype=float).reshape(3, 3) if body_mat is not None else np.eye(3)
        out = []
        for comp in self.odor_profile:
            offset = np.asarray(comp.local_offset, dtype=float).reshape(3)
            out.append(body_pos + R @ offset)
        return out

    @classmethod
    def from_recipe(cls, name: str, recipe: str, recipe_path=None, **kwargs) -> "OdorObject":
        """Construct from a named recipe in the VOC recipe table."""
        # Local import avoids a config <-> envs import cycle.
        from odor_sim.config.recipes import get_recipe

        profile = get_recipe(recipe, path=recipe_path)
        return cls(name=name, odor_profile=profile, **kwargs)

"""OdorObject: robosuite objects that also carry an OdorProfile.

The scene builder and bridge only ever read three things off an object:
``.object_id``, ``.odor_profile`` and ``.source_positions()`` -- the geometry is
irrelevant to them. So the odor API lives in :class:`OdorObjectMixin`, and the
concrete classes differ only in geometry:

  * :class:`OdorObject`     -- a primitive :class:`BoxObject`.
  * :class:`OdorXMLObject`  -- a mesh-backed :class:`MujocoXMLObject` (LIBERO
    assets or any custom robosuite object XML).

Use :func:`make_odor_object` to build either kind by catalog name.
"""

from __future__ import annotations

import numpy as np
from robosuite.models.objects import BoxObject, MujocoXMLObject

from odor_sim.envs.odor_profile import OdorProfile


class OdorObjectMixin:
    """The geometry-agnostic odor API attached to every odor object.

    Concrete subclasses must set ``self.odor_profile`` (an :class:`OdorProfile`)
    before use, and may set ``self.rotation`` / ``self.rotation_axis`` placement
    metadata (consumed by the env's placement sampler; mirrors LIBERO's
    per-object rotation attributes).
    """

    odor_profile: OdorProfile
    rotation = None
    rotation_axis = "z"

    @property
    def object_id(self) -> str:
        return self.name

    @property
    def num_sources(self) -> int:
        return self.odor_profile.num_sources

    def source_positions(self, body_pos, body_mat=None) -> list[np.ndarray]:
        """World positions of this object's *active* VOC sources.

        Args:
            body_pos: (3,) world position of the object body.
            body_mat: optional (3, 3) world rotation of the object body. If
                given, per-VOC local offsets are rotated into world frame.

        Returns:
            list of (3,) world positions, one per active VOC component (in
            profile order), index-aligned with the SceneBuilder's sources.
        """
        body_pos = np.asarray(body_pos, dtype=float).reshape(3)
        R = np.asarray(body_mat, dtype=float).reshape(3, 3) if body_mat is not None else np.eye(3)
        out = []
        for comp in self.odor_profile.active_components():
            offset = np.asarray(comp.local_offset, dtype=float).reshape(3)
            out.append(body_pos + R @ offset)
        return out


class OdorObject(OdorObjectMixin, BoxObject):
    """A box-shaped object emitting a multi-VOC :class:`OdorProfile`.

    Args:
        name: unique object name (also the ``object_id`` used in the
            object -> source map).
        odor_profile: the mixture this object emits.
        size: (half-x, half-y, half-z) box size in meters.
        rgba / material / density / friction: passed through to BoxObject.
        rotation / rotation_axis: placement metadata for the env sampler.
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
        rotation=None,
        rotation_axis="z",
    ):
        if not isinstance(odor_profile, OdorProfile):
            raise TypeError("odor_profile must be an OdorProfile")
        self.odor_profile = odor_profile
        self.rotation = rotation
        self.rotation_axis = rotation_axis
        BoxObject.__init__(
            self,
            name=name,
            size=size,
            rgba=rgba,
            material=material,
            density=density,
            friction=friction,
            rng=rng,
        )

    @classmethod
    def from_recipe(cls, name: str, recipe: str, recipe_path=None, **kwargs) -> "OdorObject":
        """Construct from a named recipe in the VOC recipe table."""
        # Local import avoids a config <-> envs import cycle.
        from odor_sim.config.recipes import get_recipe

        profile = get_recipe(recipe, path=recipe_path)
        return cls(name=name, odor_profile=profile, **kwargs)


class OdorXMLObject(OdorObjectMixin, MujocoXMLObject):
    """A mesh-backed object (LIBERO / custom CAD) emitting an OdorProfile.

    Loads a self-contained robosuite object XML (mesh + texture paths relative
    to the XML), following LIBERO's constructor conventions so LIBERO assets
    drop in unchanged.

    Args:
        name: unique object name (also the ``object_id``).
        odor_profile: the mixture this object emits.
        xml_path: path to the object XML.
        rotation / rotation_axis: placement metadata for the env sampler.
        joints / obj_type / duplicate_collision_geoms: forwarded to
            :class:`MujocoXMLObject` (defaults match LIBERO movable objects).
    """

    def __init__(
        self,
        name: str,
        odor_profile: OdorProfile,
        xml_path,
        rotation=None,
        rotation_axis="z",
        joints=None,
        obj_type="all",
        duplicate_collision_geoms=False,
    ):
        if not isinstance(odor_profile, OdorProfile):
            raise TypeError("odor_profile must be an OdorProfile")
        self.odor_profile = odor_profile
        self.rotation = rotation
        self.rotation_axis = rotation_axis
        if joints is None:
            joints = [dict(type="free", damping="0.0005")]
        MujocoXMLObject.__init__(
            self,
            str(xml_path),
            name=name,
            joints=joints,
            obj_type=obj_type,
            duplicate_collision_geoms=duplicate_collision_geoms,
        )


def make_odor_object(
    catalog_name: str,
    *,
    instance_name: "str | None" = None,
    catalog_path=None,
    recipe_path=None,
    rng=None,
):
    """Build an odor object (box or mesh) from a catalog entry.

    Each catalog object uses its own dedicated recipe; there is no override.

    Args:
        catalog_name: name in ``objects.yaml`` (see
            :func:`odor_sim.config.objects.list_objects`).
        instance_name: robosuite object name / ``object_id`` (defaults to
            ``catalog_name``).
        catalog_path / recipe_path: optional overrides for the catalog file and
            recipe file locations.
        rng: numpy RNG forwarded to primitive objects.

    Returns:
        An :class:`OdorObject` or :class:`OdorXMLObject`.
    """
    # Local imports avoid a config <-> envs import cycle.
    from odor_sim.config.objects import get_object_spec, resolve_geometry_xml
    from odor_sim.config.recipes import get_recipe

    spec = get_object_spec(catalog_name, path=catalog_path)
    profile = get_recipe(spec.recipe, path=recipe_path)

    name = instance_name or catalog_name
    geom = spec.geometry
    gtype = geom["type"]

    if gtype == "box":
        return OdorObject(
            name=name,
            odor_profile=profile,
            size=tuple(geom.get("size", (0.022, 0.022, 0.022))),
            rgba=tuple(geom.get("rgba", (0.55, 0.35, 0.15, 1.0))),
            rotation=spec.rotation,
            rotation_axis=spec.rotation_axis,
            rng=rng,
        )

    xml_path = resolve_geometry_xml(geom)
    return OdorXMLObject(
        name=name,
        odor_profile=profile,
        xml_path=xml_path,
        rotation=spec.rotation,
        rotation_axis=spec.rotation_axis,
    )

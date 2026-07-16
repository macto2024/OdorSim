"""ClassifyLiquid: a Lift-style task where the smell is the class, not the look.

Like :class:`~odor_sim.envs.odor_lift.OdorLift`, the robot lifts an object off
the table. But here the task is chosen by *liquid* (see
:mod:`odor_sim.config.liquids`): each liquid carries its own VOC
:class:`~odor_sim.envs.odor_profile.OdorProfile`, while its *container* is a cup
mesh drawn at random from a shared pool every episode. So the visual (which cup)
is decorrelated from the label (which liquid), forcing a policy to classify by
odor rather than appearance.
"""

from __future__ import annotations

from odor_sim.envs.odor_lift import OdorLift
from odor_sim.envs.base import OdorManipulationEnv
from odor_sim.envs.odor_object import OdorXMLObject


class ClassifyLiquid(OdorLift):
    """Lift a cup of liquid; the liquid (odor) is the class, the cup is random.

    Args:
        liquids: liquid name(s) to spawn (see ``odor_sim/config/liquids.yaml``).
            A single string or a list; by default the first entry is the lift
            target and extras are odor distractors. Defaults to ``["water"]``.
            Each liquid uses its own dedicated VOC recipe; its cup mesh is picked
            at random (per episode) from the catalog's shared ``cups`` pool.
        target_liquid: which spawned liquid is the lift target. Must be one of
            the names in ``liquids``. Defaults to the first spawned liquid.
        instruction: task language label. Generated from the target liquid name
            if not provided.
        liquid_catalog_path / recipe_path: optional overrides for the liquid
            catalog file and recipe file locations (mainly for tests).
    """

    def __init__(
        self,
        robots="Panda",
        liquids=None,
        target_liquid: "str | None" = None,
        instruction: str = "",
        scenario_config_dir=None,
        liquid_catalog_path=None,
        recipe_path=None,
        **kwargs,
    ):
        from odor_sim.config.liquids import get_cup_pool, get_liquid_spec

        if liquids is None:
            liquids = ["water"]
        elif isinstance(liquids, str):
            liquids = [liquids]
        self._object_names = list(liquids)
        if not self._object_names:
            raise ValueError("ClassifyLiquid requires at least one liquid")

        # Resolve (and validate) each liquid's dedicated recipe up front so an
        # unknown liquid name fails loudly at construction, not at first reset.
        self._liquid_recipes = {
            name: get_liquid_spec(name, path=liquid_catalog_path).recipe
            for name in self._object_names
        }
        self._recipe_path = recipe_path
        # Shared cup-mesh pool; a fresh choice is drawn per liquid every reset.
        self._cup_pool = get_cup_pool(path=liquid_catalog_path)
        if not self._cup_pool:
            raise ValueError("ClassifyLiquid requires a non-empty cup pool")

        if target_liquid is None:
            self._target_index = 0
        elif target_liquid in self._object_names:
            self._target_index = self._object_names.index(target_liquid)
        else:
            raise ValueError(
                f"target_liquid {target_liquid!r} is not among the spawned "
                f"liquids {self._object_names!r}"
            )
        self._target_name = self._object_names[self._target_index]

        if not instruction:
            instruction = f"pick up the cup of {self._target_name.replace('_', ' ')}"

        # Skip OdorLift.__init__ (it expects ``objects``); go straight to the
        # base env with our already-prepared liquid/target/instruction state.
        OdorManipulationEnv.__init__(
            self,
            robots=robots,
            instruction=instruction,
            scenario_config_dir=scenario_config_dir,
            **kwargs,
        )

    def _make_odor_objects(self):
        """One mesh object per liquid, each in a randomly chosen cup.

        Called from ``_load_model`` on every (hard) reset, so the cup assignment
        is re-sampled each episode while the odor stays tied to the liquid.
        """
        from odor_sim.config.recipes import get_recipe

        objects = []
        for name in self._object_names:
            cup = self._cup_pool[int(self.rng.integers(len(self._cup_pool)))]
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
        return objects

    @property
    def liquid_names(self) -> list[str]:
        """Catalog names of the spawned liquids (alias of :attr:`object_names`)."""
        return list(self._object_names)

    @property
    def target_liquid_name(self) -> str:
        """Catalog name of the lift-target liquid."""
        return self._target_name

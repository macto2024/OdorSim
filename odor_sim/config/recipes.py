"""Loader for the VOC recipe table (``voc_recipes.yaml``)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

_RECIPE_FILE = Path(__file__).with_name("voc_recipes.yaml")


@lru_cache(maxsize=1)
def _load_raw(path: str) -> dict:
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    recipes = data.get("recipes", {}) if isinstance(data, dict) else {}
    if not recipes:
        raise ValueError(f"No 'recipes' found in {path}")
    return recipes


def list_recipes(path: "str | Path | None" = None) -> list[str]:
    """Return the available recipe names."""
    p = str(path or _RECIPE_FILE)
    return sorted(_load_raw(p).keys())


def get_recipe(name: str, path: "str | Path | None" = None):
    """Look up a recipe by name and return an :class:`OdorProfile`."""
    # Local import avoids a config <-> envs import cycle.
    from odor_sim.envs.odor_profile import OdorProfile

    p = str(path or _RECIPE_FILE)
    recipes = _load_raw(p)
    if name not in recipes:
        raise KeyError(f"Recipe {name!r} not found. Available: {sorted(recipes)}")
    return OdorProfile.from_spec(recipes[name])

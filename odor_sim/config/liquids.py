"""Loader for the liquid catalog (``liquids.yaml``).

The ClassifyLiquid task classifies a liquid by its *smell* while showing it in a
randomly chosen *cup*. So, unlike :mod:`odor_sim.config.objects` (which binds one
name to one fixed geometry + recipe), a liquid binds only to a VOC recipe, and
the geometry is drawn per episode from a shared pool of cup meshes. This module
loads both halves: the ``cups`` pool and the ``liquids`` name -> recipe map.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Union

import yaml

_CATALOG_FILE = Path(__file__).with_name("liquids.yaml")

# Repo root: .../odor_sim/config/liquids.py -> parents[2] == repo root.
_REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class CupSpec:
    """One interchangeable cup mesh: resolved XML path + placement metadata."""

    xml_path: Path
    rotation: Union[None, float, list] = None
    rotation_axis: str = "z"


@dataclass(frozen=True)
class LiquidSpec:
    """A resolved liquid entry: its name and dedicated VOC recipe."""

    name: str
    recipe: str


@lru_cache(maxsize=1)
def _load_raw(path: str) -> dict:
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Malformed liquid catalog {path}")
    if not data.get("liquids"):
        raise ValueError(f"No 'liquids' found in {path}")
    if not data.get("cups"):
        raise ValueError(f"No 'cups' pool found in {path}")
    return data


def _resolve_xml(raw: str) -> Path:
    """Resolve a cup XML path (relative to repo root or absolute)."""
    p = Path(raw)
    if not p.is_absolute():
        p = _REPO_ROOT / p
    if not p.is_file():
        raise FileNotFoundError(f"Cup XML not found: {p}")
    return p.resolve()


def list_liquids(path: "str | Path | None" = None) -> list[str]:
    """Return the available liquid names."""
    p = str(path or _CATALOG_FILE)
    return sorted(_load_raw(p)["liquids"].keys())


def get_liquid_spec(name: str, path: "str | Path | None" = None) -> LiquidSpec:
    """Look up and validate a liquid entry, returning a :class:`LiquidSpec`."""
    p = str(path or _CATALOG_FILE)
    liquids = _load_raw(p)["liquids"]
    if name not in liquids:
        raise KeyError(f"Liquid {name!r} not found. Available: {sorted(liquids)}")
    entry = liquids[name] or {}
    recipe = entry.get("recipe")
    if not recipe:
        raise ValueError(f"Liquid {name!r} is missing a 'recipe'")
    return LiquidSpec(name=name, recipe=recipe)


def get_cup_pool(path: "str | Path | None" = None) -> list[CupSpec]:
    """Return the pool of cup meshes (validated, XML paths resolved)."""
    p = str(path or _CATALOG_FILE)
    cups = _load_raw(p)["cups"]
    pool: list[CupSpec] = []
    for entry in cups:
        if not isinstance(entry, dict) or not entry.get("path"):
            raise ValueError(f"Each cup entry needs a 'path'; got {entry!r}")
        pool.append(
            CupSpec(
                xml_path=_resolve_xml(entry["path"]),
                rotation=entry.get("rotation"),
                rotation_axis=entry.get("rotation_axis", "z"),
            )
        )
    return pool

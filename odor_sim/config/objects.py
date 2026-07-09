"""Loader for the object catalog (``objects.yaml``).

The catalog binds a single name (e.g. ``"mango"``) to both a geometry spec and a
dedicated VOC recipe, so a task can request an object by name and get its shape
and its smell together. This mirrors the recipe-table pattern in
:mod:`odor_sim.config.recipes`; geometry resolution (primitive box params or a
self-contained mesh XML path) lives here so the env layer stays thin.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Union

import yaml

_CATALOG_FILE = Path(__file__).with_name("objects.yaml")

# Repo root: .../odor_sim/config/objects.py -> parents[2] == repo root.
_REPO_ROOT = Path(__file__).resolve().parents[2]

_VALID_GEOMETRY_TYPES = frozenset({"box", "xml"})


@dataclass(frozen=True)
class ObjectSpec:
    """A resolved catalog entry: geometry + dedicated recipe + placement meta."""

    name: str
    geometry: dict
    recipe: str
    rotation: Union[None, float, list] = None
    rotation_axis: str = "z"


@lru_cache(maxsize=1)
def _load_raw(path: str) -> dict:
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    objects = data.get("objects", {}) if isinstance(data, dict) else {}
    if not objects:
        raise ValueError(f"No 'objects' found in {path}")
    return objects


def list_objects(path: "str | Path | None" = None) -> list[str]:
    """Return the available catalog object names."""
    p = str(path or _CATALOG_FILE)
    return sorted(_load_raw(p).keys())


def resolve_geometry_xml(geometry: dict) -> Path:
    """Resolve the object XML path for an ``xml`` geometry spec.

    Paths are relative to the repo root (or absolute). The referenced XML is a
    self-contained robosuite object (mesh/texture paths relative to itself).
    """
    gtype = geometry.get("type")
    if gtype != "xml":
        raise ValueError(f"Geometry type {gtype!r} has no XML to resolve")
    raw = geometry.get("path")
    if not raw:
        raise ValueError("xml geometry requires a 'path'")
    p = Path(raw)
    if not p.is_absolute():
        p = _REPO_ROOT / p
    if not p.is_file():
        raise FileNotFoundError(f"Object XML not found: {p}")
    return p.resolve()


def _validate_entry(name: str, entry: dict) -> None:
    geometry = entry.get("geometry")
    if not isinstance(geometry, dict):
        raise ValueError(f"Object {name!r} is missing a 'geometry' mapping")
    gtype = geometry.get("type")
    if gtype not in _VALID_GEOMETRY_TYPES:
        raise ValueError(
            f"Object {name!r} has invalid geometry type {gtype!r}; "
            f"must be one of {sorted(_VALID_GEOMETRY_TYPES)}"
        )
    if gtype == "xml":
        # Fail fast (and clearly) if the referenced asset is missing.
        resolve_geometry_xml(geometry)
    if not entry.get("recipe"):
        raise ValueError(f"Object {name!r} is missing a 'recipe'")


def get_object_spec(name: str, path: "str | Path | None" = None) -> ObjectSpec:
    """Look up and validate a catalog entry, returning an :class:`ObjectSpec`."""
    p = str(path or _CATALOG_FILE)
    objects = _load_raw(p)
    if name not in objects:
        raise KeyError(f"Object {name!r} not found. Available: {sorted(objects)}")
    entry = objects[name]
    _validate_entry(name, entry)
    return ObjectSpec(
        name=name,
        geometry=dict(entry["geometry"]),
        recipe=entry["recipe"],
        rotation=entry.get("rotation"),
        rotation_axis=entry.get("rotation_axis", "z"),
    )

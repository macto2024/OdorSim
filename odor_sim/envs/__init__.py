"""odor_sim.envs: robosuite 1.5.2 task-authoring layer for odor-aware manipulation.

Exposes the OdorObject/OdorProfile abstractions, the SceneBuilder that flattens
objects into GADEN sources (with the object->source index map), the base
odor-aware manipulation env, and the OdorLift reference task.
"""

from odor_sim.envs.base import OdorManipulationEnv
from odor_sim.envs.odor_lift import OdorLift
from odor_sim.envs.odor_object import OdorObject
from odor_sim.envs.odor_profile import OdorProfile, VOCComponent
from odor_sim.envs.scene_builder import SceneBuilder, SourceEntry

__all__ = [
    "OdorManipulationEnv",
    "OdorLift",
    "OdorObject",
    "OdorProfile",
    "VOCComponent",
    "SceneBuilder",
    "SourceEntry",
]

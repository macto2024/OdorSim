"""odor_sim.recording: Phase 5 dataset pipeline (synthesis + LeRobot export).

Two-stage pipeline (docs/SCHEMA.md, docs/phase5_recording.plan.md):

  * :mod:`odor_sim.recording.synthesize` (5b) - offline MOX/PID feature
    synthesis over raw episode dirs (``features.npz`` + ``features_meta.json``).
  * :mod:`odor_sim.recording.convert` (5c) - convert raw episode dirs to a
    LeRobotDataset v3.0.
"""

from odor_sim.recording.convert import convert_episodes
from odor_sim.recording.synthesize import synthesize_episode

__all__ = ["synthesize_episode", "convert_episodes"]

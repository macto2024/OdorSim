"""Backward-compatible alias for :mod:`odor_sim.bridge.collect_place_oft`.

The vision-only pick-place batch collector was renamed to ``collect_place_oft``
so ``collect_odor*`` can mean the odor-conditioned task. Prefer::

    python -m odor_sim.bridge.collect_place_oft ...
"""

from odor_sim.bridge.collect_place_oft import *  # noqa: F401,F403
from odor_sim.bridge.collect_place_oft import main

if __name__ == "__main__":
    raise SystemExit(main())

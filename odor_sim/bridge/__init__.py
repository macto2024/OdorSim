"""odor_sim.bridge: rclpy coupling to odor_gaden_rt + teleop collection app.

``GadenBridge`` (Phase 4a) publishes object-expanded source poses + step ticks
and queries ground-truth ppm at the EE. ``export_scene`` writes a GADEN scene
matching an env's source list. ``teleop`` (Phase 4b) is the interactive data
collection tool.

``GadenBridge`` is imported lazily so that ROS-free helpers (e.g. the scene
exporter) can be used without ``rclpy`` on the path.
"""

__all__ = ["GadenBridge"]


def __getattr__(name):
    if name == "GadenBridge":
        from odor_sim.bridge.gaden_bridge import GadenBridge

        return GadenBridge
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

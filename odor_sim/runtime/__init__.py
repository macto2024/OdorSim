"""odor_sim.runtime: lifecycle glue behind :func:`odor_sim.make`.

``GadenServerManager`` (Phase 4.5b) spawns and health-checks the ``odor_gaden_rt``
C++ server as a subprocess. ``OdorCosimSession`` (Phase 4.5c) composes a
robosuite env + :class:`~odor_sim.bridge.gaden_bridge.GadenBridge` + the server
manager into a single ``reset()`` / ``step()`` / ``close()`` co-simulation
object. Both are imported lazily so ROS-free callers need not have ``rclpy`` on
the path.
"""

__all__ = ["GadenServerManager", "OdorCosimSession", "OdorMonitor"]


def __getattr__(name):
    if name == "GadenServerManager":
        from odor_sim.runtime.gaden_server import GadenServerManager

        return GadenServerManager
    if name == "OdorCosimSession":
        from odor_sim.runtime.session import OdorCosimSession

        return OdorCosimSession
    if name == "OdorMonitor":
        from odor_sim.runtime.odor_monitor import OdorMonitor

        return OdorMonitor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

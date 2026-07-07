"""odor_sim: co-simulation glue between robosuite and GADEN for odor-aware VLA data.

Subpackages (populated across build phases):
    envs      - robosuite task authoring, OdorObject/OdorProfile, RM65 robot
    bridge    - rclpy client to the GADEN real-time server + teleop app
    sensors   - MOX/PID e-nose model (ppm -> voltage), shared offline + eval
    recording - LeRobot dataset writer
    policy    - PolicyAdapter interface + closed-loop eval harness
    config    - VOC recipe table, frame map, scenario ids

Unified co-simulation facade (Phase 4.5)::

    import odor_sim as odorsim
    with odorsim.make("OdorLift", recipe="ripe_fruit") as cosim:
        obs = cosim.reset()
        obs, reward, done, info = cosim.step(action)   # info["ppm"] at the EE

``make`` and ``OdorCosimSession`` are imported lazily so that plain
``import odor_sim`` does not pull in robosuite / rclpy.
"""

__version__ = "0.0.1"

__all__ = ["make", "OdorCosimSession"]


def __getattr__(name):
    if name == "make":
        from odor_sim._make import make

        return make
    if name == "OdorCosimSession":
        from odor_sim.runtime.session import OdorCosimSession

        return OdorCosimSession
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

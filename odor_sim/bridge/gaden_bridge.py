"""rclpy bridge between a robosuite env and the ``odor_gaden_rt`` server.

This is the Phase 4a coupling layer. It has no robosuite dependency itself; it
only consumes the pose/source APIs exposed by the Phase 3 env
(:class:`~odor_sim.envs.base.OdorManipulationEnv`) and speaks ROS 2 to the
running GADEN real-time node:

    pub  /gaden/source_poses  geometry_msgs/PoseArray  (one pose per source)
    pub  /gaden/step          std_msgs/Empty           (advance one deltaTime)
    sub  /gaden/sim_time      std_msgs/Float32         (lockstep confirmation)
    cli  /odor_value          gaden_msgs/srv/GasPosition  (per-gas ppm at EE)
    cli  /wind_value          gaden_msgs/srv/WindPosition (optional)

Per control step the driver publishes the object-expanded source poses (already
in GADEN frame via the env's SceneBuilder + FrameMap), ticks the simulation one
step in lockstep, then queries the ground-truth per-gas ppm at the EE odor
sensor. It returns that ppm plus the poses so a recorder (mining) or the sensor
model (eval) can consume it.

The GADEN node must be running in **lockstep** mode (``stepOnTimer:=false``,
the default) and with a scene whose source list matches the env's
``scene_builder`` index-for-index. Use :mod:`odor_sim.bridge.export_scene` to
generate a matching scene before launching the node.
"""

from __future__ import annotations

import time

import numpy as np
import rclpy
from gaden_msgs.srv import GasPosition, WindPosition
from geometry_msgs.msg import Pose, PoseArray
from rclpy.node import Node
from std_msgs.msg import Empty, Float32


class GadenBridge:
    """Drives ``odor_gaden_rt`` in lockstep from a robosuite control loop.

    Args:
        node_name: rclpy node name.
        service_timeout: seconds to wait for an ``/odor_value`` response.
        step_timeout: seconds to wait for ``/gaden/sim_time`` to confirm each
            ``/gaden/step`` (lockstep sync). Set to 0 to fire-and-forget.
        settle_time: small pause (s) after publishing source poses before the
            step tick, so the source update is delivered before the step is
            processed by the (single-threaded) server. Default 2 ms.
        frame_id: header frame id stamped on published messages.
        auto_init: call ``rclpy.init()`` if the context is not already inited.
    """

    def __init__(
        self,
        node_name: str = "odor_bridge",
        service_timeout: float = 5.0,
        step_timeout: float = 2.0,
        settle_time: float = 0.002,
        frame_id: str = "map",
        auto_init: bool = True,
    ):
        self._owns_context = False
        if auto_init and not rclpy.ok():
            rclpy.init()
            self._owns_context = True

        self.node = Node(node_name)
        self.service_timeout = float(service_timeout)
        self.step_timeout = float(step_timeout)
        self.settle_time = float(settle_time)
        self.frame_id = frame_id

        self._source_pub = self.node.create_publisher(PoseArray, "/gaden/source_poses", 10)
        self._step_pub = self.node.create_publisher(Empty, "/gaden/step", 20)
        self._odor_cli = self.node.create_client(GasPosition, "/odor_value")
        self._wind_cli = self.node.create_client(WindPosition, "/wind_value")

        self._sim_time = 0.0
        self._sim_time_msgs = 0
        self.node.create_subscription(Float32, "/gaden/sim_time", self._on_sim_time, 10)

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    def wait_for_server(self, timeout: float = 10.0) -> bool:
        """Block until the ``/odor_value`` service is available."""
        return self._odor_cli.wait_for_service(timeout_sec=timeout)

    def close(self) -> None:
        self.node.destroy_node()
        if self._owns_context and rclpy.ok():
            rclpy.shutdown()

    def __enter__(self) -> "GadenBridge":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ------------------------------------------------------------------ #
    # Sim time / spinning
    # ------------------------------------------------------------------ #
    def _on_sim_time(self, msg: Float32) -> None:
        self._sim_time = float(msg.data)
        self._sim_time_msgs += 1

    @property
    def sim_time(self) -> float:
        return self._sim_time

    def _spin_until(self, predicate, timeout: float) -> bool:
        """Spin the node until ``predicate()`` is true or ``timeout`` elapses."""
        deadline = time.time() + timeout
        while rclpy.ok() and not predicate():
            remaining = deadline - time.time()
            if remaining <= 0.0:
                return False
            rclpy.spin_once(self.node, timeout_sec=min(remaining, 0.05))
        return predicate()

    # ------------------------------------------------------------------ #
    # Source poses + stepping
    # ------------------------------------------------------------------ #
    def publish_source_poses(self, positions_gaden) -> None:
        """Publish (num_sources, 3) GADEN-frame source positions.

        Index ``i`` of the array moves source ``i`` in the server's scene, so
        the caller must pass positions ordered like the env's SceneBuilder
        (i.e. ``env.get_gaden_source_poses()``).
        """
        positions = np.asarray(positions_gaden, dtype=float).reshape(-1, 3)
        msg = PoseArray()
        msg.header.frame_id = self.frame_id
        msg.header.stamp = self.node.get_clock().now().to_msg()
        for p in positions:
            pose = Pose()
            pose.position.x = float(p[0])
            pose.position.y = float(p[1])
            pose.position.z = float(p[2])
            msg.poses.append(pose)
        self._source_pub.publish(msg)

    def step(self, n: int = 1) -> bool:
        """Advance the GADEN sim ``n`` steps, waiting for lockstep confirmation.

        Returns True if ``/gaden/sim_time`` confirmed all ``n`` steps within
        ``step_timeout`` (or immediately if ``step_timeout`` is 0).
        """
        confirmed = True
        for _ in range(n):
            target = self._sim_time_msgs + 1
            self._step_pub.publish(Empty())
            if self.step_timeout > 0.0:
                ok = self._spin_until(lambda t=target: self._sim_time_msgs >= t, self.step_timeout)
                confirmed = confirmed and ok
            else:
                rclpy.spin_once(self.node, timeout_sec=0.0)
        return confirmed

    # ------------------------------------------------------------------ #
    # Queries
    # ------------------------------------------------------------------ #
    def query_ppm(self, points_gaden) -> list[dict]:
        """Query per-gas ppm at one or more GADEN-frame points.

        Args:
            points_gaden: (3,) or (N, 3) GADEN-frame query points.

        Returns:
            list of ``{gas_name: ppm}`` dicts, one per query point. ``gas_name``
            is GADEN's canonical spelling (e.g. ``"ethanol"``).
        """
        pts = np.asarray(points_gaden, dtype=float).reshape(-1, 3)
        req = GasPosition.Request()
        req.x = pts[:, 0].tolist()
        req.y = pts[:, 1].tolist()
        req.z = pts[:, 2].tolist()

        future = self._odor_cli.call_async(req)
        if not self._spin_until(future.done, self.service_timeout):
            raise TimeoutError("/odor_value did not respond within service_timeout")
        res = future.result()

        gas_names = list(res.gas_type)
        out: list[dict] = []
        for cell in res.positions:
            out.append({gas_names[i]: float(cell.concentration[i]) for i in range(len(gas_names))})
        return out

    def query_wind(self, points_gaden) -> np.ndarray:
        """Query wind vectors at GADEN-frame points; returns (N, 3) [u, v, w]."""
        pts = np.asarray(points_gaden, dtype=float).reshape(-1, 3)
        req = WindPosition.Request()
        req.x = pts[:, 0].tolist()
        req.y = pts[:, 1].tolist()
        req.z = pts[:, 2].tolist()

        future = self._wind_cli.call_async(req)
        if not self._spin_until(future.done, self.service_timeout):
            raise TimeoutError("/wind_value did not respond within service_timeout")
        res = future.result()
        return np.stack([np.array(res.u), np.array(res.v), np.array(res.w)], axis=1)

    # ------------------------------------------------------------------ #
    # Per-control-step driver
    # ------------------------------------------------------------------ #
    def step_env(self, env, advance: bool = True) -> dict:
        """Run one bridge cycle for the current state of ``env``.

        Sequence: publish object-expanded source poses -> tick one GADEN step
        (lockstep) -> query ground-truth per-gas ppm at the EE odor sensor.

        Args:
            env: an :class:`~odor_sim.envs.base.OdorManipulationEnv` (must expose
                ``get_gaden_source_poses``, ``get_enose_site_pose`` and
                ``frame_map``).
            advance: if False, skip the ``/gaden/step`` tick (query the current
                plume without advancing time).

        Returns:
            dict with keys:
                ``sim_time``            current GADEN sim time (s)
                ``ee_pos_world``        (3,) EE odor-sensor pose, robosuite frame
                ``ee_pos_gaden``        (3,) same point mapped to GADEN frame
                ``source_poses_gaden``  (num_sources, 3) published positions
                ``ppm``                 ``{gas_name: ppm}`` at the EE
                ``gas_types``           list of gas names present in the scene
        """
        source_poses = env.get_gaden_source_poses()
        self.publish_source_poses(source_poses)

        ee_pos_world, _ = env.get_enose_site_pose()
        ee_pos_gaden = env.frame_map.robosuite_to_gaden(ee_pos_world)

        if advance:
            if self.settle_time > 0.0:
                time.sleep(self.settle_time)
            self.step(1)

        ppm = self.query_ppm(ee_pos_gaden)[0]
        return {
            "sim_time": self._sim_time,
            "ee_pos_world": np.asarray(ee_pos_world, dtype=float),
            "ee_pos_gaden": np.asarray(ee_pos_gaden, dtype=float),
            "source_poses_gaden": np.asarray(source_poses, dtype=float),
            "ppm": ppm,
            "gas_types": list(ppm.keys()),
        }

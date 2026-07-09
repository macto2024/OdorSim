"""``OdorCosimSession``: env + bridge + server as one co-simulation object.

Phase 4.5c. Composition (not subclassing) of three already-built pieces:

  * a robosuite :class:`~odor_sim.envs.base.OdorManipulationEnv` (physics + poses),
  * a :class:`~odor_sim.bridge.gaden_bridge.GadenBridge` (ROS lockstep client),
  * a :class:`~odor_sim.runtime.gaden_server.GadenServerManager` (C++ subprocess).

``reset()`` / ``step()`` / ``close()`` drive all three together. ``step()``
order is: robosuite ``env.step(action)`` first, then the bridge publishes the
post-step source poses, ticks GADEN one step in lockstep, and queries the
ground-truth per-gas ppm at the EE, which is returned in ``info``.

The session is created by :func:`odor_sim.make`; construct it directly only for
advanced wiring. Any of ``bridge`` / ``server`` may be ``None`` (e.g. the
``auto_start_gaden=False`` dry-run path), in which case ``step()`` runs the
robosuite env alone and reports an empty ppm dict.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


class OdorCosimSession:
    """Bundles the robosuite env, the GADEN bridge, and the server process.

    Args:
        env: the constructed :class:`OdorManipulationEnv` subclass.
        bridge: a connected :class:`GadenBridge`, or ``None`` for env-only.
        server: a running :class:`GadenServerManager`, or ``None`` if the caller
            manages the server (``connect_only``) or there is none (dry run).
        scene_path: path to the exported GADEN scene YAML (for provenance).
        scene_id: the scene id loaded by the server.
        config_dir: the GADEN environment configuration directory.
        owns_server: if True, :meth:`close` stops ``server``; if False (e.g.
            ``connect_only``), the server is left running.
        odor_monitor: optional :class:`~odor_sim.runtime.odor_monitor.OdorMonitor`.
    """

    def __init__(
        self,
        env,
        *,
        bridge=None,
        server=None,
        scene_path: "str | Path | None" = None,
        scene_id: "str | None" = None,
        config_dir: "str | Path | None" = None,
        owns_server: bool = True,
        odor_monitor=None,
    ):
        self.env = env
        self.bridge = bridge
        self.server = server
        self.scene_path = Path(scene_path) if scene_path else None
        self.scene_id = scene_id
        self.config_dir = Path(config_dir) if config_dir else None
        self.owns_server = bool(owns_server)
        self.odor_monitor = odor_monitor
        self._closed = False

    # ------------------------------------------------------------------ #
    @property
    def has_gaden(self) -> bool:
        """True if a bridge is wired (ppm queries available)."""
        return self.bridge is not None

    def reset(self):
        """Reset the robosuite env; reset GADEN time and publish source poses.

        Returns the robosuite observation (with ``obs["instruction"]``).
        """
        obs = self.env.reset()
        if self.bridge is not None:
            # Start a fresh GADEN episode (simTime -> 0, filaments cleared) so
            # recorded sim_time is episode-relative, then prime the server with
            # the reset source layout (no time advance).
            self.bridge.reset_time()
            self.bridge.publish_source_poses(self.env.get_gaden_source_poses())
        if self.odor_monitor is not None:
            self.odor_monitor.reset()
        return obs

    def step(self, action):
        """Advance physics one control step, then GADEN one step in lockstep.

        Order: ``env.step(action)`` -> publish post-step source poses ->
        ``/gaden/step`` -> query ``/odor_value`` at the EE.

        Returns:
            ``(obs, reward, done, info)`` where ``info`` gains, when a bridge is
            present, ``info["ppm"]`` (``{gas_name: ppm}`` at the EE),
            ``info["sim_time"]`` and ``info["gaden_source_poses"]``.
        """
        obs, reward, done, info = self.env.step(action)
        if self.bridge is not None:
            rec = self.bridge.step_env(self.env, advance=True)
            info["ppm"] = rec["ppm"]
            info["sim_time"] = rec["sim_time"]
            info["gaden_source_poses"] = rec["source_poses_gaden"]
            info["gas_types"] = rec["gas_types"]
            if self.odor_monitor is not None:
                self.odor_monitor.record(rec["sim_time"], rec["ppm"])
        else:
            info.setdefault("ppm", {})
        return obs, reward, done, info

    # ------------------------------------------------------------------ #
    def close(self) -> None:
        """Tear down env, bridge, and (if owned) the server process."""
        if self._closed:
            return
        self._closed = True
        if self.odor_monitor is not None:
            try:
                self.odor_monitor.close()
            except Exception:  # noqa: BLE001
                pass
        # order: bridge first (stops ROS traffic), then server, then env.
        if self.bridge is not None:
            try:
                self.bridge.close()
            except Exception:  # noqa: BLE001
                pass
        if self.server is not None and self.owns_server:
            try:
                self.server.stop()
            except Exception:  # noqa: BLE001
                pass
        try:
            self.env.close()
        except Exception:  # noqa: BLE001
            pass

    def __enter__(self) -> "OdorCosimSession":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ------------------------------------------------------------------ #
    # Convenience passthroughs (keep the robosuite surface reachable)
    # ------------------------------------------------------------------ #
    @property
    def instruction(self) -> str:
        return self.env.instruction

    @property
    def action_dim(self) -> int:
        return self.env.action_dim

    def render(self, *args, **kwargs):
        return self.env.render(*args, **kwargs)

    def zero_action(self) -> np.ndarray:
        """A no-op robosuite action vector (arm delta zero)."""
        return np.zeros(self.env.action_dim, dtype=float)

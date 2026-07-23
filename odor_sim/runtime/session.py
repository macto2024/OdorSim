"""``OdorCosimSession``: env + bridge + server as one co-simulation object.

Phase 4.5c. Composition (not subclassing) of three already-built pieces:

  * a robosuite :class:`~odor_sim.envs.base.OdorManipulationEnv` (physics + poses),
  * a :class:`~odor_sim.bridge.gaden_bridge.GadenBridge` (ROS lockstep client),
  * a :class:`~odor_sim.runtime.gaden_server.GadenServerManager` (C++ subprocess).

``reset()`` / ``step()`` / ``close()`` drive all three together. ``step()``
order is: robosuite ``env.step(action)`` first, then the bridge publishes the
post-step source poses, ticks GADEN one step in lockstep, and queries the
ground-truth per-gas ppm at the EE, which is returned in ``info``.

When ``odor_mode`` is ``continuous`` or ``discrete``, the session streams MOX
e-nose voltage into ``info``. By default **all** models in
:data:`~odor_sim.sensors.mox_pid.MOX_MODELS` run in parallel:

  * ``info["enose_voltages"]`` — ``{model: voltage}`` for every active sensor
  * ``info["enose_voltage"]`` — primary model scalar (``TGS2620`` when present,
    else the first selected model) for ``/act`` / back-compat

Modes are mutually exclusive (continuous vs discrete), not multi-stream of both
valve behaviors. Any of ``bridge`` / ``server`` may be ``None`` (dry run).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence, Union

import numpy as np

from odor_sim.config.gas_types import GADEN_GAS_NAMES
from odor_sim.sensors.mox_pid import MOX_MODELS, ContinuousEnose, SamplingEnose

ODOR_MODES = ("none", "continuous", "discrete")
SAMPLE, IDLE, FILTER = 1, 0, -1
PRIMARY_MOX_MODEL = MOX_MODELS[0]  # TGS2620 — matches offline synthesize


def resolve_mox_models(
    mox_model: "str | Sequence[str] | None" = None,
) -> list[str]:
    """Normalize a ``mox_model`` spec to an ordered list of valid MOX ids.

    ``None``, ``\"all\"``, or an empty sequence → all of :data:`MOX_MODELS`.
    A comma-separated string or a sequence selects a subset (order preserved,
    duplicates dropped). Primary preference for ``enose_voltage`` is
    :data:`PRIMARY_MOX_MODEL` when it is in the list, else the first entry.
    """
    if mox_model is None:
        return list(MOX_MODELS)
    if isinstance(mox_model, str):
        text = mox_model.strip()
        if not text or text.lower() == "all":
            return list(MOX_MODELS)
        models = [p.strip() for p in text.split(",") if p.strip()]
    else:
        models = [str(p).strip() for p in mox_model if str(p).strip()]
    if not models:
        return list(MOX_MODELS)
    out: list[str] = []
    for name in models:
        if name not in MOX_MODELS:
            raise ValueError(f"Unknown MOX model {name!r}; choose from {MOX_MODELS}")
        if name not in out:
            out.append(name)
    return out


def primary_mox_model(models: Sequence[str]) -> str:
    """Canonical primary id used for ``info['enose_voltage']``."""
    if PRIMARY_MOX_MODEL in models:
        return PRIMARY_MOX_MODEL
    return str(models[0])


def _serialize_sample_window(window, rate: float) -> dict:
    """Serializable dict for one completed :class:`~odor_sim.sensors.mox_pid.SampleWindow`."""
    odor_class = window.odor_class
    odor_name = odor_class.name if odor_class is not None else None
    return {
        "trigger_step": int(window.trigger_step),
        "duration_steps": int(window.duration_steps),
        "duration_s": float(window.duration_steps / rate) if rate else float(window.duration_s or 0.0),
        "odor_class": odor_name,
        "odor_class_gaden": GADEN_GAS_NAMES.get(odor_class) if odor_class is not None else None,
        "voltage_trace": np.asarray(window.voltage_trace, dtype=np.float32),
        "ppm_trace": [dict(frame) for frame in window.ppm_trace],
    }


def _make_one_enose(
    odor_mode: str,
    *,
    mox_model: str,
    load_resistance: float | None,
    vcc: float,
    rate: float,
) -> "ContinuousEnose | SamplingEnose":
    kwargs: dict[str, Any] = {"model": mox_model, "rate": float(rate), "vcc": float(vcc)}
    if load_resistance is not None:
        kwargs["load_resistance"] = float(load_resistance)
    if odor_mode == "continuous":
        return ContinuousEnose(**kwargs)
    return SamplingEnose(**kwargs)


def _build_enoses(
    odor_mode: str,
    *,
    mox_models: Sequence[str],
    load_resistance: float | None,
    vcc: float,
    rate: float,
) -> tuple[str, "dict[str, ContinuousEnose | SamplingEnose]"]:
    """Allocate one ContinuousEnose / SamplingEnose per model (or empty)."""
    mode = str(odor_mode or "none").strip().lower()
    if mode not in ODOR_MODES:
        raise ValueError(f"odor_mode must be one of {ODOR_MODES}, got {odor_mode!r}")
    if mode == "none":
        return mode, {}
    enoses = {
        name: _make_one_enose(
            mode,
            mox_model=name,
            load_resistance=load_resistance,
            vcc=vcc,
            rate=rate,
        )
        for name in mox_models
    }
    return mode, enoses


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
        sensor_monitor: optional :class:`~odor_sim.runtime.sensor_monitor.SensorMonitor`.
        odor_mode: ``none`` | ``continuous`` | ``discrete`` live voltage path.
        mox_model: MOX model id(s) to stream. ``None`` / ``\"all\"`` (default)
            runs every model in :data:`MOX_MODELS` in parallel. A string, comma
            list, or sequence selects a subset.
        load_resistance: voltage-divider RL [ohms]; ``None`` uses each model R0.
        vcc: divider supply voltage (V).
        enose_rate: sensor update rate (Hz); defaults to env ``control_freq``.
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
        sensor_monitor=None,
        odor_mode: str = "none",
        mox_model: "str | Sequence[str] | None" = None,
        load_resistance: "float | None" = None,
        vcc: float = 5.0,
        enose_rate: "float | None" = None,
    ):
        self.env = env
        self.bridge = bridge
        self.server = server
        self.scene_path = Path(scene_path) if scene_path else None
        self.scene_id = scene_id
        self.config_dir = Path(config_dir) if config_dir else None
        self.owns_server = bool(owns_server)
        self.odor_monitor = odor_monitor
        self.sensor_monitor = sensor_monitor
        self._closed = False
        self._sensor_step = 0

        rate = float(enose_rate) if enose_rate is not None else float(
            getattr(env, "control_freq", 20) or 20
        )
        self.mox_models = resolve_mox_models(mox_model)
        self.mox_model = primary_mox_model(self.mox_models)  # primary / back-compat
        self.load_resistance = load_resistance
        self.vcc = float(vcc)
        self.enose_rate = rate
        self.odor_mode, self._enoses = _build_enoses(
            odor_mode,
            mox_models=self.mox_models,
            load_resistance=self.load_resistance,
            vcc=self.vcc,
            rate=self.enose_rate,
        )
        # Legacy alias used by older callers / teleop checks.
        self._enose = self._enoses.get(self.mox_model) if self._enoses else None

    # ------------------------------------------------------------------ #
    @property
    def has_gaden(self) -> bool:
        """True if a bridge is wired (ppm queries available)."""
        return self.bridge is not None

    @property
    def has_enose(self) -> bool:
        """True if one or more live voltage sensors are attached."""
        return bool(self._enoses)

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
        if self.sensor_monitor is not None:
            self.sensor_monitor.reset()
        self._sensor_step = 0
        self.reset_enose()
        return obs

    def reset_enose(self) -> None:
        """Reset all live MOX sensors without resetting the env / GADEN."""
        for enose in self._enoses.values():
            enose.reset()

    def _split_action(self, action) -> tuple[np.ndarray, int]:
        """Split robosuite action vs optional trailing ``enose_state`` (discrete)."""
        action = np.asarray(action, dtype=float).ravel()
        env_dim = int(self.env.action_dim)
        if self.odor_mode == "discrete" and action.size == env_dim + 1:
            enose_state = int(np.rint(action[-1]))
            if enose_state not in (SAMPLE, IDLE, FILTER):
                # Soft clamp to nearest valid token.
                if enose_state > 0:
                    enose_state = SAMPLE
                elif enose_state < 0:
                    enose_state = FILTER
                else:
                    enose_state = IDLE
            return action[:env_dim], enose_state
        return action[:env_dim] if action.size >= env_dim else np.resize(action, env_dim), IDLE

    def _update_enose(self, info: dict, enose_state: int) -> None:
        """Fill ``info`` voltage fields from all attached e-noses."""
        if not self._enoses:
            return
        ppm = info.get("ppm") or {}
        voltages: dict[str, float] = {}
        sample_windows: dict[str, dict | None] = {}
        sampling_active = False

        if self.odor_mode == "continuous":
            for name, enose in self._enoses.items():
                voltages[name] = float(enose.update(ppm))
            info["enose_voltages"] = voltages
            info["enose_voltage"] = float(voltages[self.mox_model])
            info["mox_models"] = list(self.mox_models)
            info["primary_mox_model"] = self.mox_model
            info["enose_state"] = IDLE
            info["sampling_active"] = False
            info["sample_window"] = None
            info["sample_windows"] = {name: None for name in self.mox_models}
            return

        # discrete — shared enose_state gates every sensor
        for name, enose in self._enoses.items():
            result = enose.update(ppm, enose_state)
            voltages[name] = float(result["voltage"])
            sampling_active = sampling_active or bool(result["sampling_active"])
            completed = result.get("completed_window")
            sample_windows[name] = (
                _serialize_sample_window(completed, self.enose_rate)
                if completed is not None
                else None
            )

        info["enose_voltages"] = voltages
        info["enose_voltage"] = float(voltages[self.mox_model])
        info["mox_models"] = list(self.mox_models)
        info["primary_mox_model"] = self.mox_model
        info["enose_state"] = int(enose_state)
        info["sampling_active"] = bool(sampling_active)
        info["sample_windows"] = sample_windows
        info["sample_window"] = sample_windows.get(self.mox_model)

    def _record_sensor_monitor(self, info: dict) -> None:
        """Push live voltages into the optional SensorMonitor."""
        if self.sensor_monitor is None or "enose_voltages" not in info:
            return
        if "sim_time" in info:
            t = float(info["sim_time"])
        else:
            self._sensor_step += 1
            t = self._sensor_step / max(self.enose_rate, 1e-6)
            info.setdefault("sim_time", t)
        self.sensor_monitor.record(
            t,
            info["enose_voltages"],
            enose_state=info.get("enose_state"),
            sampling_active=info.get("sampling_active"),
        )

    def step(self, action):
        """Advance physics one control step, then GADEN one step in lockstep.

        Order: ``env.step(action)`` -> publish post-step source poses ->
        ``/gaden/step`` -> query ``/odor_value`` at the EE -> optional enose update.

        For ``odor_mode="discrete"``, pass ``action`` of length
        ``env.action_dim + 1`` with trailing ``enose_state`` in ``{1,0,-1}``.

        Returns:
            ``(obs, reward, done, info)`` where ``info`` gains, when a bridge is
            present, ``info["ppm"]`` (``{gas_name: ppm}`` at the EE),
            ``info["sim_time"]`` and ``info["gaden_source_poses"]``. When e-noses
            are attached, also ``info["enose_voltage"]`` (primary),
            ``info["enose_voltages"]`` (all models), and discrete sampling metadata.
        """
        env_action, enose_state = self._split_action(action)
        obs, reward, done, info = self.env.step(env_action)
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
        self._update_enose(info, enose_state)
        self._record_sensor_monitor(info)
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
        if self.sensor_monitor is not None:
            try:
                self.sensor_monitor.close()
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
        """Env action dim, plus 1 for trailing ``enose_state`` in discrete mode."""
        dim = int(self.env.action_dim)
        if self.odor_mode == "discrete":
            return dim + 1
        return dim

    def render(self, *args, **kwargs):
        return self.env.render(*args, **kwargs)

    def zero_action(self) -> np.ndarray:
        """A no-op action vector (arm delta zero; discrete appends idle enose)."""
        return np.zeros(self.action_dim, dtype=float)

"""Interactive teleoperation + data-mining app (Phase 4b / 5a).

Drives an :class:`~odor_sim.envs.base.OdorManipulationEnv` with a robosuite
device (keyboard / SpaceMouse) while co-stepping the GADEN plume through the
:class:`~odor_sim.bridge.gaden_bridge.GadenBridge`. Each control step records
the robosuite action, the ternary ``enose_state`` token, an auto-hold/sampling
mask, the ground-truth per-gas ppm at the EE, robot proprio, and the task
instruction. Episodes are written to disk (raw ppm(t) always; live
``enose_voltage_*`` when the co-sim ``odor_mode`` is continuous/discrete).
Offline ``odor_sim.recording.synthesize`` remains optional for other MOX models.

Phase 5a extends mining with camera frames (``agentview`` + wrist) and
structured proprio (joint pos/vel, EE pose, gripper).

Two entry points:

  * :meth:`TeleopSession.run_interactive` - human drives with a device + viewer.
  * :meth:`TeleopSession.run_scripted` - fixed action/enose schedule, headless;
    used by the Phase 4b smoke test (no display, no human).

E-nose keys (added on top of robosuite's keyboard controls; digits 3/4/5
avoid the contested 0/1/2 range):
    ``3`` sample (auto: ~7 s sample hold -> ~10 s filter -> idle)
    ``4`` idle (immediate; cancels an in-flight sequence)
    ``5`` filter/purge (immediate; cancels an in-flight sequence)
Auto-hold: one ``sample`` press freezes the arm for the sample window (motion
ignored) so every sniff is a clean stationary dwell; those steps carry a
``sampling_active`` mask. After sample, the valve auto-switches to FILTER
for the filter window, then IDLE. The HUD and ``[enose]`` log lines report
the phase and remaining time.

Since Phase 4.5 the GADEN server is spawned automatically via ``odor_sim.make``,
so a single command is enough (no separate server terminal)::

    python -m odor_sim.bridge.teleop --env OdorLift --objects milk --robots Panda \\
        --camera agentview --device keyboard --odor-monitor

For the ClassifyLiquid task, choose liquids instead of objects; each liquid is
shown in a randomly chosen cup every episode (the odor, not the cup, is the
class)::

    python -m odor_sim.bridge.teleop --env ClassifyLiquid \\
        --liquids water coke wine --target-liquid coke --robots Panda

ClassifyLiquidPlace is the same liquid/cup setup, but success is placing the
target liquid's cup on a catalog place target (default plate)::

    python -m odor_sim.bridge.teleop --env ClassifyLiquidPlace \\
        --liquids water coke wine --target-liquid coke --place-target plate
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

# enose tokens
SAMPLE, IDLE, FILTER = 1, 0, -1
DEFAULT_SCENARIO = "10x6_uniform"

# Mining camera set (Phase 5 decision): table overview + wrist, 256x256 RGB.
MINING_CAMERAS = ("agentview", "robot0_eye_in_hand")
# robosuite camera name -> on-disk frames/ subdir name.
_CAMERA_DIRNAMES = {"agentview": "agentview", "robot0_eye_in_hand": "wrist"}
# Structured proprio: robosuite obs key -> episode.npz key.
_PROPRIO_KEYS = {
    "robot0_joint_pos": "joint_pos",
    "robot0_joint_vel": "joint_vel",
    "robot0_eef_pos": "eef_pos",
    "robot0_eef_quat": "eef_quat",
    "robot0_gripper_qpos": "gripper_qpos",
}


def _camera_dirname(camera: str) -> str:
    return _CAMERA_DIRNAMES.get(camera, camera)


def _save_png(path: "str | Path", image: np.ndarray) -> None:
    from PIL import Image

    Image.fromarray(np.ascontiguousarray(image)).save(str(path))


def _controller_type(controller_config: dict) -> str:
    """Human-readable arm controller type from a composite controller config."""
    body_parts = controller_config.get("body_parts", {}) if controller_config else {}
    arm = body_parts.get("right")
    if not isinstance(arm, dict):
        arm = next((v for v in body_parts.values() if isinstance(v, dict)), {})
    return arm.get("type") or controller_config.get("type", "?")


def _validate_robot(robots: str) -> str:
    from robosuite.robots import ROBOT_CLASS_MAPPING

    if robots not in ROBOT_CLASS_MAPPING:
        available = ", ".join(sorted(ROBOT_CLASS_MAPPING.keys()))
        raise ValueError(f"Unknown robot {robots!r}. Available robosuite robots: {available}")
    return robots


class EpisodeRecorder:
    """Accumulates per-step records and writes one episode to disk.

    Stores raw ppm(t) + state/action/enose token/mask/instruction. When the
    session streams live voltage, also stores ``enose_voltage_continuous`` or
    ``enose_voltage_sampling`` (mode-specific) aligned 1:1 with ppm.
    Phase 5a adds structured proprio (joint pos/vel, EE pose, gripper) and, when
    ``camera_map`` is given, per-step camera frames written as PNG sequences
    under ``frames/<subdir>/``.

    Also stores supervised odor labels per step: ``source_identity`` (class
    index of the target object) and ``source_distance`` (e-nose site to target
    body center, meters), plus ``enose_pos`` / ``target_pos`` for provenance.

    Args:
        camera_map: ``{obs_image_key: frames_subdir}`` (e.g.
            ``{"agentview_image": "agentview", "robot0_eye_in_hand_image": "wrist"}``).
            Empty/None disables frame capture (headless path).
        target_index: class index of the active odor source within ``objects``.
        num_classes: number of spawned object classes for this episode.
    """

    def __init__(
        self,
        out_dir: "str | Path",
        instruction: str,
        gas_types: list,
        meta: dict,
        camera_map: "dict | None" = None,
        target_index: int = 0,
        num_classes: int = 1,
    ):
        self.out_dir = Path(out_dir)
        self.instruction = instruction
        self.gas_types = list(gas_types)
        self.meta = dict(meta)
        self.camera_map = dict(camera_map or {})
        self.target_index = int(target_index)
        self.num_classes = int(num_classes)
        self._rows: list[dict] = []
        self._image_size: "tuple | None" = None
        self.ep_dir: "Path | None" = None

    def _ensure_dir(self) -> Path:
        """Create (once) the timestamped episode dir and any frame subdirs."""
        if self.ep_dir is not None:
            return self.ep_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        ep_dir = self.out_dir / f"episode_{stamp}"
        suffix = 1
        while ep_dir.exists():
            ep_dir = self.out_dir / f"episode_{stamp}_{suffix}"
            suffix += 1
        ep_dir.mkdir(parents=True)
        for subdir in self.camera_map.values():
            (ep_dir / "frames" / subdir).mkdir(parents=True, exist_ok=True)
        self.ep_dir = ep_dir
        return ep_dir

    def add(
        self,
        *,
        sim_time,
        state,
        action,
        enose_state,
        sampling_active,
        ppm,
        source_distance: float,
        enose_pos,
        target_pos,
        proprio: "dict | None" = None,
        frames: "dict | None" = None,
        enose_voltage: "float | None" = None,
        enose_voltages: "dict | None" = None,
    ):
        row = {
            "sim_time": float(sim_time),
            "state": np.asarray(state, dtype=np.float32),
            "action": np.asarray(action, dtype=np.float32),
            "enose_state": int(enose_state),
            "sampling_active": bool(sampling_active),
            "ppm": np.asarray([float(ppm.get(g, 0.0)) for g in self.gas_types], dtype=np.float32),
            "source_identity": self.target_index,
            "source_distance": float(source_distance),
            "enose_pos": np.asarray(enose_pos, dtype=np.float32).ravel()[:3],
            "target_pos": np.asarray(target_pos, dtype=np.float32).ravel()[:3],
        }
        if enose_voltage is not None:
            row["enose_voltage"] = float(enose_voltage)
        if enose_voltages:
            # Preserve model order from meta["mox_models"] when present.
            model_order = list(self.meta.get("mox_models") or enose_voltages.keys())
            row["enose_voltages"] = np.asarray(
                [float(enose_voltages.get(m, 0.0)) for m in model_order],
                dtype=np.float32,
            )
        if proprio:
            row["proprio"] = {
                k: np.asarray(v, dtype=np.float32).ravel() for k, v in proprio.items()
            }
        self._rows.append(row)

        if frames and self.camera_map:
            step_idx = len(self._rows) - 1
            ep_dir = self._ensure_dir()
            for obs_key, subdir in self.camera_map.items():
                img = frames.get(obs_key)
                if img is None:
                    continue
                img = np.asarray(img, dtype=np.uint8)
                if self._image_size is None:
                    self._image_size = (int(img.shape[0]), int(img.shape[1]))
                _save_png(ep_dir / "frames" / subdir / f"{step_idx:06d}.png", img)

    def __len__(self) -> int:
        return len(self._rows)

    def _stack_proprio(self) -> dict:
        if not self._rows or "proprio" not in self._rows[0]:
            return {}
        keys = list(self._rows[0]["proprio"].keys())
        out = {}
        for key in keys:
            if all("proprio" in r and key in r["proprio"] for r in self._rows):
                out[key] = np.stack([r["proprio"][key] for r in self._rows])
        return out

    def write(self, success: bool) -> "Path | None":
        if not self._rows:
            print("[teleop] empty episode; nothing written")
            return None
        ep_dir = self._ensure_dir()

        t_count = len(self._rows)
        arrays = {
            "sim_time": np.array([r["sim_time"] for r in self._rows], dtype=np.float32),
            "state": np.stack([r["state"] for r in self._rows]),
            "action": np.stack([r["action"] for r in self._rows]),
            "enose_state": np.array([r["enose_state"] for r in self._rows], dtype=np.int8),
            "sampling_active": np.array([r["sampling_active"] for r in self._rows], dtype=bool),
            "ppm": np.stack([r["ppm"] for r in self._rows]),
            "source_identity": np.full(t_count, self.target_index, dtype=np.int64),
            "source_distance": np.array(
                [r["source_distance"] for r in self._rows], dtype=np.float32
            ),
            "enose_pos": np.stack([r["enose_pos"] for r in self._rows]),
            "target_pos": np.stack([r["target_pos"] for r in self._rows]),
        }
        proprio = self._stack_proprio()
        arrays.update(proprio)
        odor_mode = str(self.meta.get("odor_mode") or "none")
        mox_models = list(self.meta.get("mox_models") or [])
        if self._rows and "enose_voltage" in self._rows[0]:
            voltage = np.array([r["enose_voltage"] for r in self._rows], dtype=np.float32)
            # Mode-specific primary keys match offline synthesize / LeRobot convert.
            if odor_mode == "discrete":
                arrays["enose_voltage_sampling"] = voltage
            else:
                arrays["enose_voltage_continuous"] = voltage
            arrays["enose_voltage"] = voltage
        if self._rows and "enose_voltages" in self._rows[0]:
            multi = np.stack([r["enose_voltages"] for r in self._rows]).astype(np.float32)
            arrays["enose_voltages"] = multi  # (T, n_models)
            # Also write per-model 1-D streams for easy offline inspection.
            if mox_models and multi.ndim == 2 and multi.shape[1] == len(mox_models):
                suffix = "sampling" if odor_mode == "discrete" else "continuous"
                for i, name in enumerate(mox_models):
                    arrays[f"enose_voltage_{suffix}_{name}"] = multi[:, i]
        np.savez(ep_dir / "episode.npz", **arrays)

        class_names = list(self.meta.get("objects") or [])
        if "enose_voltage" in arrays:
            models_note = ",".join(mox_models) if mox_models else "primary"
            note = (
                f"raw ppm(t) + live {odor_mode} voltage stored "
                f"({models_note}); offline synthesize still optional"
            )
        else:
            note = "raw ppm(t) stored; voltage synthesized offline (Phase 5b)"
        meta = {
            "instruction": self.instruction,
            "gas_types": self.gas_types,
            "num_steps": t_count,
            "success": bool(success),
            "note": note,
            "proprio_keys": sorted(proprio.keys()),
            "source_identity_index": self.target_index,
            "class_names": class_names,
            "num_classes": self.num_classes,
            "distance_reference": "enose_site_to_target_body_center",
            **self.meta,
        }
        if self.camera_map:
            meta["cameras"] = {
                subdir: f"frames/{subdir}" for subdir in dict.fromkeys(self.camera_map.values())
            }
            if self._image_size is not None:
                meta["image_size"] = list(self._image_size)
        with open(ep_dir / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)
        print(f"[teleop] wrote {len(self._rows)} steps -> {ep_dir} (success={success})")
        return ep_dir


class TeleopSession:
    """Owns the co-sim (env + bridge + server) and the per-step record logic."""

    def __init__(
        self,
        env: str = "OdorLift",
        objects=None,
        target_object: str | None = None,
        liquids=None,
        target_liquid: str | None = None,
        place_target: str | None = None,
        scenario: str = DEFAULT_SCENARIO,
        sample_hold_s: float = 7.0,
        filter_hold_s: float = 10.0,
        use_bridge: bool = True,
        out_dir: str = "datasets/teleop",
        has_renderer: bool = False,
        control_freq: int = 20,
        odor_monitor=False,
        sensor_monitor=False,
        robots: str = "Panda",
        controller: str | None = None,
        render_camera: str = "agentview",
        goal_update_mode: str = "target",
        pos_sensitivity: float = 1.0,
        rot_sensitivity: float = 1.0,
        success_hold_steps: int = 10,
        record_frames: bool = False,
        camera_size: int = 256,
        mining_cameras=MINING_CAMERAS,
        odor_mode: str = "none",
        mox_model=None,
        load_resistance: float | None = None,
        vcc: float = 5.0,
    ):
        import odor_sim as odorsim
        from robosuite.controllers import load_composite_controller_config
        from odor_sim.runtime.session import resolve_mox_models

        robots = _validate_robot(robots)
        if goal_update_mode not in ("target", "achieved"):
            raise ValueError(f"goal_update_mode must be 'target' or 'achieved', got {goal_update_mode!r}")
        if success_hold_steps < 0:
            raise ValueError(f"success_hold_steps must be >= 0, got {success_hold_steps}")
        odor_mode = str(odor_mode or "none").strip().lower()
        if odor_mode not in ("none", "continuous", "discrete"):
            raise ValueError(
                f"odor_mode must be 'none', 'continuous', or 'discrete', got {odor_mode!r}"
            )
        mox_models = resolve_mox_models(mox_model)

        self.robots = robots
        self.goal_update_mode = goal_update_mode
        self.pos_sensitivity = pos_sensitivity
        self.rot_sensitivity = rot_sensitivity
        self.success_hold_steps = int(success_hold_steps)
        self.control_freq = control_freq
        self.sample_hold_steps = int(round(sample_hold_s * control_freq))
        self.filter_hold_steps = int(round(filter_hold_s * control_freq))
        self.out_dir = out_dir
        self._vis_wrapped = False
        self.record_frames = bool(record_frames)
        self.mining_cameras = list(mining_cameras)
        self.camera_size = int(camera_size)
        self.odor_mode = odor_mode
        self.mox_models = mox_models
        self.mox_model = mox_models[0]  # display / primary hint; session picks TGS2620
        self.camera_map = (
            {f"{c}_image": _camera_dirname(c) for c in self.mining_cameras}
            if self.record_frames
            else {}
        )

        controller_config = load_composite_controller_config(controller=controller, robot=robots)
        self.controller_type = _controller_type(controller_config)

        if self.record_frames:
            cam_kwargs = {
                "has_offscreen_renderer": True,
                "use_camera_obs": True,
                "camera_names": self.mining_cameras,
                "camera_heights": self.camera_size,
                "camera_widths": self.camera_size,
            }
        else:
            cam_kwargs = {"has_offscreen_renderer": False, "use_camera_obs": False}

        # Only forward ``objects``/``target_object`` (catalog tasks like
        # OdorLift), ``liquids``/``target_liquid`` (ClassifyLiquid*), or
        # ``place_target`` (ClassifyLiquidPlace) when set, so each env only
        # receives the kwargs it understands.
        object_kwargs = {"objects": list(objects)} if objects else {}
        if target_object:
            object_kwargs["target_object"] = target_object
        if liquids:
            object_kwargs["liquids"] = list(liquids)
        if target_liquid:
            object_kwargs["target_liquid"] = target_liquid
        if place_target:
            object_kwargs["place_target"] = place_target

        self.cosim = odorsim.make(
            env,
            scenario=scenario,
            auto_start_gaden=use_bridge,
            bridge=use_bridge,
            odor_monitor=odor_monitor if use_bridge else False,
            sensor_monitor=sensor_monitor,
            odor_mode=odor_mode,
            mox_model=mox_models,
            load_resistance=load_resistance,
            vcc=vcc,
            enose_site_offset=(0.0, 0.0, -0.02),
            robots=robots,
            controller_configs=controller_config,
            render_camera=render_camera,
            ignore_done=True,
            reward_shaping=True,
            has_renderer=has_renderer,
            control_freq=control_freq,
            horizon=100000,
            **object_kwargs,
            **cam_kwargs,
        )
        self.env = self.cosim.env
        self.bridge = self.cosim.bridge
        self.scenario = scenario
        self.scene_id = self.cosim.scene_id

        # Fixed gas-vector layout from the objects' full profiles (incl. inert
        # strength-0 gases), so recorded ppm has the same columns every run.
        self.gas_types = list(self.env.gas_types)
        self._seq_phase: str | None = None  # None | "sample" | "filter"
        self._phase_left = 0

    def close(self):
        self.cosim.close()

    def _proprio_state(self, obs) -> np.ndarray:
        keys = [
            k
            for k in obs
            if k.startswith("robot0_")
            and isinstance(obs[k], np.ndarray)
            and obs[k].ndim == 1
            and not k.endswith("image")
            and not k.endswith("depth")
        ]
        if not keys:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate([np.asarray(obs[k], dtype=np.float32).ravel() for k in sorted(keys)])

    @staticmethod
    def _structured_proprio(obs) -> dict:
        out = {}
        for obs_key, schema_key in _PROPRIO_KEYS.items():
            val = obs.get(obs_key)
            if isinstance(val, np.ndarray):
                out[schema_key] = np.asarray(val, dtype=np.float32).ravel()
        return out

    def _extract_frames(self, obs) -> dict:
        frames = {}
        for obs_key in self.camera_map:
            img = obs.get(obs_key)
            if isinstance(img, np.ndarray):
                frames[obs_key] = np.asarray(img[::-1], dtype=np.uint8)
        return frames

    def _zero_arm(self, env_action: np.ndarray, gripper_dof: int) -> np.ndarray:
        held = env_action.copy()
        if gripper_dof > 0:
            held[:-gripper_dof] = 0.0
        else:
            held[:] = 0.0
        return held

    def _cancel_enose_sequence(self) -> None:
        self._seq_phase = None
        self._phase_left = 0

    def _apply_hold(
        self,
        env_action: np.ndarray,
        enose_state: int,
        gripper_dof: int,
        forced: bool = False,
    ):
        """Apply sample→filter→idle auto sequence, with immediate 4/5 overrides.

        Pressing sample (``3``) starts: SAMPLE hold -> FILTER window -> IDLE.
        A *forced* idle/filter (operator just pressed ``4`` / ``5``) cancels
        the sequence immediately. Latched idle after a one-shot sample does
        not cancel — the auto sequence continues.
        """
        if forced and enose_state == IDLE:
            self._cancel_enose_sequence()
            return env_action, IDLE, False

        if forced and enose_state == FILTER:
            self._cancel_enose_sequence()
            return env_action, FILTER, False

        if enose_state == SAMPLE:
            self._seq_phase = "sample"
            self._phase_left = self.sample_hold_steps

        if self._seq_phase == "sample":
            held = self._zero_arm(env_action, gripper_dof)
            self._phase_left -= 1
            if self._phase_left <= 0:
                self._seq_phase = "filter"
                self._phase_left = self.filter_hold_steps
            return held, SAMPLE, True

        if self._seq_phase == "filter":
            self._phase_left -= 1
            if self._phase_left <= 0:
                self._cancel_enose_sequence()
            return env_action, FILTER, False

        return env_action, enose_state, False

    def step(
        self,
        recorder: EpisodeRecorder,
        env_action: np.ndarray,
        enose_state: int,
        gripper_dof: int,
        enose_forced: bool = False,
    ):
        env_action, eff_enose, sampling = self._apply_hold(
            env_action, enose_state, gripper_dof, forced=enose_forced
        )

        # Drive physics + GADEN + live MOX through OdorCosimSession so
        # info["enose_voltage"] matches eval / offline ContinuousEnose /
        # SamplingEnose when odor_mode is set.
        if self.cosim.odor_mode == "discrete":
            step_action = np.concatenate(
                [np.asarray(env_action, dtype=float).ravel(), [float(eff_enose)]]
            )
        else:
            step_action = np.asarray(env_action, dtype=float).ravel()

        obs, reward, done, info = self.cosim.step(step_action)

        ppm = info.get("ppm") or {}
        sim_time = float(info.get("sim_time", 0.0))
        enose_pos = self.env.get_enose_site_pose()[0]

        # Prefer session sampling_active when discrete (sensor valve); else teleop hold mask.
        if self.cosim.odor_mode == "discrete" and "sampling_active" in info:
            sampling = bool(info["sampling_active"])

        target_name = self.env.target_object_name
        target_pos = self.env.get_object_world_poses()[target_name][0]
        source_distance = float(
            np.linalg.norm(np.asarray(enose_pos, dtype=float) - np.asarray(target_pos, dtype=float))
        )

        enose_voltage = info.get("enose_voltage")
        enose_voltages = info.get("enose_voltages")
        recorder.add(
            sim_time=sim_time,
            state=self._proprio_state(obs),
            action=np.concatenate([np.asarray(env_action, dtype=float).ravel(), [eff_enose]]),
            enose_state=eff_enose,
            sampling_active=sampling,
            ppm=ppm,
            source_distance=source_distance,
            enose_pos=enose_pos,
            target_pos=target_pos,
            proprio=self._structured_proprio(obs),
            frames=self._extract_frames(obs) if self.camera_map else None,
            enose_voltage=enose_voltage,
            enose_voltages=enose_voltages,
        )
        return obs, reward, done, info, eff_enose, sampling

    def _reset_gaden(self) -> None:
        """Start a fresh GADEN episode and prime it with the current source layout.

        Called after ``env.reset()`` so each recorded episode's ``sim_time`` starts
        near zero instead of accumulating GADEN server time across the session.
        """
        if self.bridge is None:
            return
        if not self.bridge.reset_time():
            print(
                "[teleop] warning: GADEN /gaden/reset did not confirm; "
                "sim_time may carry over from prior episodes. "
                "Rebuild odor_gaden_rt and restart teleop (no manual rt_server)."
            )
        self.bridge.publish_source_poses(self.env.get_gaden_source_poses())

    def _new_recorder(self, instruction: str) -> EpisodeRecorder:
        object_names = list(getattr(self.env, "object_names", None) or [])
        target_name = getattr(self.env, "target_object_name", None)
        if target_name is not None and target_name in object_names:
            target_index = object_names.index(target_name)
        else:
            target_index = 0
        num_classes = max(len(object_names), 1)
        meta = {
            "objects": object_names or None,
            "target_object": target_name,
            "robots": self.robots,
            "control_freq": self.control_freq,
            "sample_hold_steps": self.sample_hold_steps,
            "filter_hold_steps": self.filter_hold_steps,
            "success_hold_steps": self.success_hold_steps,
            "action_layout": "[robosuite_action..., enose_state]",
            "controller_type": self.controller_type,
            "scenario": self.scenario,
            "scene_id": self.scene_id,
            "odor_mode": self.odor_mode,
            "mox_models": list(self.cosim.mox_models),
            "mox_model": self.cosim.mox_model,  # primary
            "vcc": float(self.cosim.vcc),
        }
        return EpisodeRecorder(
            self.out_dir,
            instruction,
            self.gas_types,
            meta,
            camera_map=self.camera_map,
            target_index=target_index,
            num_classes=num_classes,
        )

    def _update_success_hold(self, hold_count: int) -> tuple[int, bool]:
        if self.success_hold_steps <= 0:
            return hold_count, False
        if hold_count == 0:
            return hold_count, True
        if self.env._check_success():
            if hold_count > 0:
                return hold_count - 1, False
            return self.success_hold_steps, False
        return -1, False

    def run_scripted(self, arm_actions, enose_schedule, instruction: str = "", success: bool = True):
        instruction = instruction or self.env.instruction
        recorder = self._new_recorder(instruction)
        self.env.reset()
        self._reset_gaden()
        self.cosim.reset_enose()
        if self.cosim.odor_monitor is not None:
            self.cosim.odor_monitor.reset()
        if self.cosim.sensor_monitor is not None:
            self.cosim.sensor_monitor.reset()
        gripper_dof = self._gripper_dof()

        for a, e in zip(arm_actions, enose_schedule):
            self.step(recorder, np.asarray(a, dtype=float), int(e), gripper_dof)

        return recorder.write(success=success)

    def _gripper_dof(self) -> int:
        robot = self.env.robots[0]
        arm = robot.arms[0]
        try:
            return robot.gripper[arm].dof
        except Exception:  # noqa: BLE001
            return 1

    def run_interactive(self, device_name: str = "keyboard", max_fr: int = 20):
        from copy import deepcopy

        from robosuite.controllers.composite.composite_controller import WholeBody
        from robosuite.wrappers import VisualizationWrapper

        if not self._vis_wrapped:
            self.env = VisualizationWrapper(self.env)
            self.cosim.env = self.env
            self._vis_wrapped = True

        device, enose_keys = self._make_device(device_name)
        gripper_dof = self._gripper_dof()

        print(self._controls_help())
        print(f"[teleop] robot={self.robots} odor_mode={self.odor_mode} mox={list(self.cosim.mox_models)}")
        while True:
            instruction = self.env.instruction
            recorder = self._new_recorder(instruction)
            self.env.reset()
            self._reset_gaden()
            self.cosim.reset_enose()
            if self.cosim.odor_monitor is not None:
                self.cosim.odor_monitor.reset()
            if self.cosim.sensor_monitor is not None:
                self.cosim.sensor_monitor.reset()
            self.env.render()
            device.start_control()
            self._cancel_enose_sequence()
            enose_keys.reset()
            task_completion_hold_count = -1

            for robot in self.env.robots:
                robot.print_action_info_dict()

            active_robot = self.env.robots[0]
            prev_gripper = {
                f"{arm}_gripper": np.repeat([0], active_robot.gripper[arm].dof)
                for arm in active_robot.arms
                if active_robot.gripper[arm].dof > 0
            }
            prev_eff, prev_samp = None, None

            while True:
                start = time.time()
                input_ac = device.input2action(goal_update_mode=self.goal_update_mode)
                if input_ac is None:
                    break

                action_dict = deepcopy(input_ac)
                for arm in active_robot.arms:
                    if isinstance(active_robot.composite_controller, WholeBody):
                        ctype = active_robot.composite_controller.joint_action_policy.input_type
                    else:
                        ctype = active_robot.part_controllers[arm].input_type
                    action_dict[arm] = input_ac[f"{arm}_{'delta' if ctype == 'delta' else 'abs'}"]

                env_action = active_robot.create_action_vector(action_dict)
                for g in prev_gripper:
                    prev_gripper[g] = action_dict[g]

                enose_state, enose_forced = enose_keys.consume()
                _, _, _, _, eff, samp = self.step(
                    recorder, env_action, enose_state, gripper_dof, enose_forced=enose_forced
                )
                self.env.render()
                if (eff, samp) != (prev_eff, prev_samp):
                    token = {SAMPLE: "SAMPLE", IDLE: "idle", FILTER: "FILTER"}.get(eff, "?")
                    if self._seq_phase is not None and self._phase_left > 0:
                        hold_s = self._phase_left / float(self.env.control_freq)
                        print(
                            f"\n[enose] state={token}  phase={self._seq_phase}  "
                            f"~{hold_s:.1f}s remaining",
                            flush=True,
                        )
                    elif samp:
                        print(f"\n[enose] state={token}  sampling=ON", flush=True)
                    else:
                        print(
                            f"\n[enose] state={token}  sampling=OFF  valve closed",
                            flush=True,
                        )
                    prev_eff, prev_samp = eff, samp
                self._hud(
                    instruction,
                    eff,
                    samp,
                    len(recorder),
                    success_hold_count=task_completion_hold_count,
                    hold_left=self._phase_left,
                    seq_phase=self._seq_phase,
                    control_freq=float(self.env.control_freq),
                )

                task_completion_hold_count, end_on_success = self._update_success_hold(
                    task_completion_hold_count
                )
                if end_on_success:
                    break

                if max_fr:
                    diff = 1.0 / max_fr - (time.time() - start)
                    if diff > 0:
                        time.sleep(diff)

            success = self.env._check_success()
            recorder.write(success=success)
            if not self._ask_continue():
                break

    def _make_device(self, device_name: str):
        from odor_sim.bridge._enose_keys import EnoseKeyState

        enose_keys = EnoseKeyState()
        if device_name == "keyboard":
            from robosuite.devices import Keyboard

            device = Keyboard(
                env=self.env,
                pos_sensitivity=self.pos_sensitivity,
                rot_sensitivity=self.rot_sensitivity,
            )
            enose_keys.attach_to_keyboard(device)
        elif device_name == "spacemouse":
            try:
                from robosuite.devices import SpaceMouse
            except ImportError as exc:
                raise RuntimeError(
                    "SpaceMouse is unavailable: the `hid` module is not installed or robosuite "
                    "could not load the SpaceMouse driver. Use --device keyboard instead."
                ) from exc

            device = SpaceMouse(
                env=self.env,
                pos_sensitivity=self.pos_sensitivity,
                rot_sensitivity=self.rot_sensitivity,
            )
        else:
            raise ValueError(f"Unknown device {device_name!r}")
        return device, enose_keys

    @staticmethod
    def _controls_help() -> str:
        from odor_sim.bridge._enose_keys import KEY_FILTER, KEY_IDLE, KEY_SAMPLE

        return (
            "\n=== OdorSim teleop ===\n"
            "  arm/gripper: standard robosuite keyboard controls\n"
            f"  e-nose:  {KEY_SAMPLE} = sample/sniff "
            f"(auto: ~7 s SAMPLE hold -> ~10 s FILTER -> idle)\n"
            f"           {KEY_IDLE} = idle now (cancels sequence)\n"
            f"           {KEY_FILTER} = filter/purge now (cancels sequence)\n"
            "  HUD line shows enose phase + countdown\n"
            "  end episode: the device reset key (Ctrl+Q on keyboard)\n"
            "  sustained task success auto-ends the episode (collect-style)\n"
        )

    @staticmethod
    def _hud(
        instruction,
        enose_state,
        sampling,
        step_i,
        success_hold_count=-1,
        hold_left=0,
        seq_phase=None,
        control_freq=20.0,
    ):
        token = {SAMPLE: "SAMPLE", IDLE: "idle", FILTER: "FILTER"}.get(enose_state, "?")
        if seq_phase and hold_left > 0 and control_freq > 0:
            hold_s = hold_left / float(control_freq)
            if seq_phase == "sample":
                tag = f" [SNIFFING hold={hold_s:.1f}s left -> then FILTER]"
            elif seq_phase == "filter":
                tag = f" [FILTERING purge={hold_s:.1f}s left -> then idle]"
            else:
                tag = f" [{seq_phase} {hold_s:.1f}s left]"
        elif sampling:
            tag = " [SNIFFING]"
        else:
            tag = " [valve closed]" if enose_state in (IDLE, FILTER) else ""
        success_tag = ""
        if success_hold_count > 0:
            success_tag = f"  [SUCCESS {success_hold_count}]"
        print(
            f"\r[{step_i:5d}] enose={token}{tag}{success_tag}  | {instruction}   ",
            end="",
            flush=True,
        )

    @staticmethod
    def _ask_continue() -> bool:
        try:
            ans = input("\nAnother episode? [Y/n] ").strip().lower()
        except EOFError:
            return False
        return ans in ("", "y", "yes")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", default="OdorLift", help="registered task name")
    parser.add_argument(
        "--objects",
        nargs="+",
        default=None,
        metavar="NAME",
        help=(
            "catalog object name(s) to spawn (see odor_sim/config/objects.yaml); "
            "first is the lift target unless --target-object is given, extras "
            "are odor distractors. Default: the task's own default object."
        ),
    )
    parser.add_argument(
        "--target-object",
        default=None,
        metavar="NAME",
        help=(
            "which spawned object is the lift target (must be one of --objects); "
            "default: the first object."
        ),
    )
    parser.add_argument(
        "--liquids",
        nargs="+",
        default=None,
        metavar="NAME",
        help=(
            "ClassifyLiquid / ClassifyLiquidPlace: liquid name(s) to spawn (see "
            "odor_sim/config/liquids.yaml); each is shown in a random cup every "
            "episode. First is the target unless --target-liquid is given, "
            "extras are odor distractors."
        ),
    )
    parser.add_argument(
        "--target-liquid",
        default=None,
        metavar="NAME",
        help=(
            "ClassifyLiquid / ClassifyLiquidPlace: which spawned liquid is the "
            "task target (must be one of --liquids); default: the first liquid."
        ),
    )
    parser.add_argument(
        "--place-target",
        default=None,
        metavar="NAME",
        help=(
            "ClassifyLiquidPlace only: catalog object to put the target liquid "
            "on (see odor_sim/config/objects.yaml); default: plate."
        ),
    )
    parser.add_argument("--scenario", default=DEFAULT_SCENARIO, help="GADEN scenario name or config-dir path")
    parser.add_argument("--device", default="keyboard", choices=["keyboard", "spacemouse"])
    parser.add_argument(
        "--robots",
        default="Panda",
        help="robosuite robot name (e.g. Panda, UR5e, Kinova3)",
    )
    parser.add_argument(
        "--controller",
        default=None,
        help="robosuite composite controller name or config path (default: robot default)",
    )
    parser.add_argument(
        "--camera",
        default="agentview",
        help="on-screen viewer camera name (render_camera)",
    )
    parser.add_argument(
        "--pos-sensitivity",
        type=float,
        default=1.0,
        help="keyboard / SpaceMouse position input scale",
    )
    parser.add_argument(
        "--rot-sensitivity",
        type=float,
        default=1.0,
        help="keyboard / SpaceMouse rotation input scale",
    )
    parser.add_argument(
        "--goal-update-mode",
        default="target",
        choices=["target", "achieved"],
        help="how the device updates arm goals (see collect_human_demonstrations)",
    )
    parser.add_argument("--sample-hold-s", type=float, default=7.0)
    parser.add_argument(
        "--filter-hold-s",
        type=float,
        default=10.0,
        help="auto FILTER duration after sample before returning to idle (default 10 s)",
    )
    parser.add_argument(
        "--success-hold-steps",
        type=int,
        default=10,
        help="consecutive successful steps before auto-ending episode (0=disable)",
    )
    parser.add_argument("--out-dir", default="datasets/teleop")
    parser.add_argument(
        "--no-frames",
        action="store_true",
        help=f"disable camera-frame mining (default records {', '.join(MINING_CAMERAS)})",
    )
    parser.add_argument(
        "--camera-size",
        type=int,
        default=256,
        help="mining camera frame height/width in px (default 256)",
    )
    parser.add_argument("--no-bridge", action="store_true", help="drive arm only, no GADEN/ppm")
    parser.add_argument(
        "--odor-mode",
        default="none",
        choices=["none", "continuous", "discrete"],
        help="live MOX voltage: none (ppm only), continuous (always-on), "
        "discrete (valve-gated by e-nose keys). Default none for back-compat",
    )
    parser.add_argument(
        "--mox-model",
        nargs="*",
        default=None,
        metavar="MODEL",
        help="MOX model(s) for live voltage. Omit = all five "
        "(TGS2620 TGS2600 TGS2611 TGS2610 TGS2612). "
        "Example: --mox-model TGS2620 TGS2600",
    )
    parser.add_argument(
        "--vcc",
        type=float,
        default=5.0,
        help="MOX voltage-divider supply (V)",
    )
    parser.add_argument(
        "--odor-monitor",
        nargs="?",
        const=True,
        default=False,
        metavar="MODE",
        help="live ppm log/plot: default log+plot; MODE=log or plot for one only",
    )
    parser.add_argument(
        "--sensor-monitor",
        nargs="?",
        const=True,
        default=False,
        metavar="MODE",
        help="live MOX voltage log/plot (needs --odor-mode continuous|discrete): "
        "default log+plot; MODE=log or plot for one only",
    )
    args = parser.parse_args(argv)

    odor_monitor = False
    if args.odor_monitor is not False:
        odor_monitor = args.odor_monitor if args.odor_monitor is not True else True
    sensor_monitor = False
    if args.sensor_monitor is not False:
        sensor_monitor = args.sensor_monitor if args.sensor_monitor is not True else True

    session = TeleopSession(
        env=args.env,
        objects=args.objects,
        target_object=args.target_object,
        liquids=args.liquids,
        target_liquid=args.target_liquid,
        place_target=args.place_target,
        scenario=args.scenario,
        sample_hold_s=args.sample_hold_s,
        filter_hold_s=args.filter_hold_s,
        use_bridge=not args.no_bridge,
        out_dir=args.out_dir,
        has_renderer=True,
        odor_monitor=odor_monitor,
        sensor_monitor=sensor_monitor,
        robots=args.robots,
        controller=args.controller,
        render_camera=args.camera,
        goal_update_mode=args.goal_update_mode,
        pos_sensitivity=args.pos_sensitivity,
        rot_sensitivity=args.rot_sensitivity,
        success_hold_steps=args.success_hold_steps,
        record_frames=not args.no_frames,
        camera_size=args.camera_size,
        odor_mode=args.odor_mode,
        mox_model=args.mox_model,
        vcc=args.vcc,
    )
    try:
        session.run_interactive(device_name=args.device)
    finally:
        session.close()
        # Ensure the GADEN server is gone even if close() missed it.
        from odor_sim.runtime.gaden_server import GadenServerManager

        GadenServerManager.kill_all()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

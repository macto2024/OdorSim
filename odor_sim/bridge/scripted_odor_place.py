"""Scripted smell → decide → pick-place collector for ClassifyLiquidPlaceOVLA.

Two identical porcelain cups with ScentFormer VOC liquids. The arm hovers
above each cup, SAMPLE-holds for ``sniff_duration_s`` (default 10 s) so the
e-nose can respond, then:

* if that cup is the instructed liquid → grasp and place on the plate
* otherwise → move aside, FILTER-hold for ``1.5 * sniff_duration_s``, try
  the next cup
* if no cup matches the instruction → return to the post-reset home pose

The first cup is uniformly random (``target_first_frac=0.5``). Ground-truth
identity drives the scripted decision so demos are reliable; the recorded
SAMPLE voltages are what a policy must learn from.

Example::

    python -m odor_sim.bridge.scripted_odor_place \\
        --liquids ethanol methanol --instructed ethanol \\
        --num-demos 5 --out-dir datasets/teleop/odor_place_scripted
"""

from __future__ import annotations

import argparse
import re
import shutil
import time
import uuid
from pathlib import Path

import numpy as np

from odor_sim.bridge.scripted_place import (
    DEFAULT_ACTION_NOISE,
    DEFAULT_ACTION_NOISE_FLOOR,
    DEFAULT_GRASP_XY_TOL,
    DEFAULT_GRASP_Z_TOL,
    DEFAULT_MIN_SPEED,
    DEFAULT_MIN_SPEED_DIST,
    DEFAULT_PRESS_Z,
    DEFAULT_SLOW_RADIUS,
    GRIPPER_CLOSE,
    GRIPPER_OPEN,
    OSC_POS_SCALE,
    ScriptedPlaceCollector,
    add_motion_shaping_args,
    motion_shaping_kwargs,
)
from odor_sim.bridge.teleop import (
    MINING_CAMERAS,
    EpisodeRecorder,
    _camera_dirname,
    _controller_type,
)
from odor_sim.recording.odor_config import (
    DEFAULT_ODOR_CONFIG,
    FILTER_DURATION_FACTOR,
    OdorConfig,
    odor_config_from_args,
)
from odor_sim.recording.odor_events import (
    PHASE_CONTROL,
    PHASE_FILTER_HOLD,
    PHASE_SNIFF_HOLD,
    write_odor_events,
)
from odor_sim.runtime.session import FILTER, IDLE, SAMPLE

DEFAULT_OUT_DIR = "datasets/teleop/odor_place_scripted"
DEFAULT_LIQUIDS = DEFAULT_ODOR_CONFIG.voc_pool[:2]
DEFAULT_SAMPLE_HOLD_S = DEFAULT_ODOR_CONFIG.sniff_duration_s
DEFAULT_FILTER_HOLD_S = DEFAULT_ODOR_CONFIG.filter_s
DEFAULT_SNIFF_Z = 0.12  # meters above cup body center
DEFAULT_FILTER_OFFSET_Y = 0.0  # unused; FILTER hovers above the plate (in-frame)
# When the instructed VOC is on the table: P(sniff it first). 0.5 = uniform.
DEFAULT_TARGET_FIRST_FRAC = 0.5
OVLA_ENV_NAME = "ClassifyLiquidPlaceOVLA"


def liquid_display_name(catalog: str) -> str:
    """Language label for a liquid catalog id — drop strength suffix.

    ``propanol1.0`` / ``ethanol0.3`` → ``propanol`` / ``ethanol``.
    Underscores become spaces; unknown forms pass through with ``_`` → `` ``.
    """
    name = str(catalog).strip()
    # Strip trailing strength: 1.0, 0.3, 0.7, etc.
    name = re.sub(r"(\d+\.\d+)$", "", name)
    name = name.replace("_", " ").strip()
    return name or str(catalog).replace("_", " ")


def build_instruction(instructed_liquid: str, place_target: str = "plate") -> str:
    return (
        f"put the cup of {liquid_display_name(instructed_liquid)} on the "
        f"{str(place_target).replace('_', ' ')}"
    )


class ScriptedOdorPlaceCollector(ScriptedPlaceCollector):
    """Smell-conditioned pick-place on :class:`ClassifyLiquidPlaceOVLA`."""

    def __init__(
        self,
        *,
        out_dir: "str | Path" = DEFAULT_OUT_DIR,
        liquids=None,
        instructed_liquid: str = "ethanol",
        place_target: str = "plate",
        instruction: str = "",
        robots: str = "Panda",
        controller: str = "BASIC",
        control_freq: int = 20,
        camera_size: int = 256,
        mining_cameras=MINING_CAMERAS,
        record_frames: bool = True,
        has_renderer: bool = False,
        render_camera: str = "agentview",
        seed: "int | None" = None,
        odor_mode: str = "discrete",
        sample_hold_s: "float | None" = None,
        filter_hold_s: "float | None" = None,
        sniff_duration_s: "float | None" = None,
        sniff_z: float = DEFAULT_SNIFF_Z,
        filter_offset_y: float = DEFAULT_FILTER_OFFSET_Y,
        # P(sniff the instructed cup first) when it is on the table.
        # Remainder: sniff a distractor first, FILTER, then the target.
        target_first_frac: float = DEFAULT_TARGET_FIRST_FRAC,
        odor_config: "OdorConfig | None" = None,
        mox_models=None,
        env_name: str = OVLA_ENV_NAME,
        # Motion / grasp knobs (same defaults as ScriptedPlaceCollector).
        pos_gain: float = 8.0,
        max_pos_action: float = 1.0,
        xy_tol: float = 0.015,
        z_tol: float = 0.015,
        slow_radius: float = DEFAULT_SLOW_RADIUS,
        min_speed: float = DEFAULT_MIN_SPEED,
        min_speed_dist: float = DEFAULT_MIN_SPEED_DIST,
        action_noise: float = DEFAULT_ACTION_NOISE,
        action_noise_floor: float = DEFAULT_ACTION_NOISE_FLOOR,
        grasp_xy_tol: float = DEFAULT_GRASP_XY_TOL,
        grasp_z_tol: float = DEFAULT_GRASP_Z_TOL,
        press_z: float = DEFAULT_PRESS_Z,
        hover_z: float = 0.18,
        grasp_z_offset: float = -0.02,
        place_z: float = 0.01,
        lift_z: float = 0.16,
        retreat_z: float = 0.20,
        rim_radius: "float | None" = None,
        rim_radius_frac: float = 0.85,
        phase_max_steps: int = 200,
        grip_hold_steps: int = 15,
        grasp_dwell_s: float = 0.75,
        max_grasp_attempts: int = 3,
        settle_steps: int = 20,
        max_episode_steps: int = 4000,
        verbose: bool = True,
        # Parallel GADEN: each worker must use its own scene_id / log dir and
        # set kill_stale_gaden=False so siblings are not wiped.
        kill_stale_gaden: bool = True,
        scene_id: "str | None" = None,
        server_log_dir: "str | Path | None" = None,
        worker_id: int = 0,
    ):
        import odor_sim as odorsim
        from robosuite.controllers import load_composite_controller_config

        liquids = list(liquids) if liquids is not None else list(DEFAULT_LIQUIDS)
        if len(liquids) < 1:
            raise ValueError("need at least one liquid cup")
        instructed = str(instructed_liquid).strip()
        if not instructed:
            raise ValueError("instructed_liquid must be non-empty")

        if odor_config is None:
            odor_config = odor_config_from_args(
                sniff_duration_s=sniff_duration_s,
                filter_duration_s=filter_hold_s,
                odor_sampling_rate=float(control_freq),
                mox_models=mox_models,
                voc_pool=liquids,
                sample_hold_s=sample_hold_s,
            )
        else:
            updates = {}
            if sniff_duration_s is not None:
                updates["sniff_duration_s"] = float(sniff_duration_s)
            elif sample_hold_s is not None:
                updates["sniff_duration_s"] = float(sample_hold_s)
            if filter_hold_s is not None:
                updates["filter_duration_s"] = float(filter_hold_s)
            if mox_models is not None:
                updates["mox_models"] = tuple(str(m) for m in mox_models)
            if updates:
                odor_config = odor_config.with_updates(**updates)
        self.odor_config = odor_config
        self.env_name = str(env_name)

        self.out_dir = Path(out_dir)
        self.verbose = bool(verbose)
        self.liquids = liquids
        self.instructed_liquid = instructed
        self.target_present = instructed in liquids
        self.place_target = place_target
        self.instruction_override = instruction
        self.robots = robots
        self.control_freq = int(self.odor_config.control_freq)
        self.record_frames = bool(record_frames)
        self.mining_cameras = list(mining_cameras)
        self.camera_size = int(camera_size)
        self.has_renderer = bool(has_renderer)
        self.seed = seed
        self.odor_mode = str(odor_mode).strip().lower()
        if self.odor_mode not in ("discrete", "continuous", "none"):
            raise ValueError(f"odor_mode must be discrete|continuous|none, got {odor_mode!r}")

        self.sample_hold_s = float(self.odor_config.sniff_duration_s)
        self.filter_hold_s = float(self.odor_config.filter_s)
        self.sniff_z = float(sniff_z)
        self.filter_offset_y = float(filter_offset_y)
        self.target_first_frac = float(np.clip(target_first_frac, 0.0, 1.0))

        self.pos_gain = float(pos_gain)
        self.max_pos_action = float(max_pos_action)
        self.xy_tol = float(xy_tol)
        self.z_tol = float(z_tol)
        self.slow_radius = max(0.0, float(slow_radius))
        self.min_speed = max(0.0, float(min_speed))
        self.min_speed_dist = max(0.0, float(min_speed_dist))
        self.action_noise = max(0.0, float(action_noise))
        self.action_noise_floor = float(np.clip(action_noise_floor, 0.0, 1.0))
        self.grasp_xy_tol = float(grasp_xy_tol)
        self.grasp_z_tol = float(grasp_z_tol)
        self.press_z = float(press_z)
        self.hover_z = float(hover_z)
        self.rng = np.random.default_rng(seed)
        self.grasp_z_offset = float(grasp_z_offset)
        self.place_z = float(place_z)
        self.lift_z = float(lift_z)
        self.retreat_z = float(retreat_z)
        self.rim_radius = None if rim_radius is None else float(rim_radius)
        self.rim_radius_frac = float(rim_radius_frac)
        self.phase_max_steps = int(phase_max_steps)
        self.grip_hold_steps = int(grip_hold_steps)
        self.grasp_dwell_s = float(grasp_dwell_s)
        self.max_grasp_attempts = max(1, int(max_grasp_attempts))
        self.settle_steps = int(settle_steps)
        self.max_episode_steps = int(max_episode_steps)
        self.kill_stale_gaden = bool(kill_stale_gaden)
        self._scene_id_override = scene_id
        self._server_log_dir = (
            None if server_log_dir is None else Path(server_log_dir)
        )
        self.worker_id = int(worker_id)

        # Unused parent fields (kept so inherited helpers don't AttributeError).
        self.cups = list(liquids)
        self.target_cup = 0

        self.camera_map = (
            {f"{c}_image": _camera_dirname(c) for c in self.mining_cameras}
            if self.record_frames
            else {}
        )

        controller_config = load_composite_controller_config(
            controller=controller, robot=robots
        )
        self.controller_type = _controller_type(controller_config)

        if self.record_frames:
            cam_kwargs = {
                "camera_names": list(self.mining_cameras),
                "camera_heights": self.camera_size,
                "camera_widths": self.camera_size,
            }
        else:
            cam_kwargs = {
                "use_camera_obs": False,
                "has_offscreen_renderer": False,
            }

        # Env needs a target_liquid among spawned liquids. When the instructed
        # VOC is absent we still pick a dummy env target (first cup) and rely on
        # our own success check (return home without placing).
        env_target = instructed if self.target_present else liquids[0]
        if not instruction:
            instruction = build_instruction(instructed, place_target)

        make_kwargs = {
            "liquids": list(liquids),
            "target_liquid": env_target,
            "place_target": place_target,
            "instruction": instruction,
            "robots": robots,
            "controller_configs": controller_config,
            "control_freq": self.control_freq,
            "has_renderer": self.has_renderer,
            "has_offscreen_renderer": True,
            "render_camera": render_camera,
            "ignore_done": True,
            "reward_shaping": True,
            "horizon": 100000,
            "use_camera_obs": True,
            **cam_kwargs,
        }
        if seed is not None:
            make_kwargs["seed"] = int(seed)

        # Defaults of odorsim.make: export=True, bridge=True, auto_start_gaden=True.
        # Vision-only scripted_place turns those off; odor needs the full stack:
        # export scene YAML → spawn odor_gaden_rt → GadenBridge for ppm.
        want_odor = self.odor_mode != "none"
        make_extra = {}
        if self._scene_id_override:
            make_extra["scene_id"] = str(self._scene_id_override)
        if self._server_log_dir is not None:
            make_extra["server_log_dir"] = str(self._server_log_dir)
        self.cosim = odorsim.make(
            self.env_name,
            auto_start_gaden=want_odor,
            bridge=want_odor,
            export=want_odor,
            odor_mode=self.odor_mode,
            kill_stale_gaden=self.kill_stale_gaden,
            mox_model=list(self.odor_config.mox_models),
            **make_extra,
            **make_kwargs,
        )
        self.env = self.cosim.env
        self.gas_types = list(getattr(self.env, "gas_types", []) or [])
        self.scenario = "10x6_uniform"
        self.scene_id = getattr(self.cosim, "scene_id", self.env_name)
        self.home_pos: "np.ndarray | None" = None
        self._active_enose = IDLE
        self._active_phase = PHASE_CONTROL
        self._sniff_object_id: "str | None" = None
        self._odor_events: list[dict] = []
        self._episode_id: "str | None" = None

    # ------------------------------------------------------------------ #
    # Action packing / recording
    # ------------------------------------------------------------------ #
    def _pack(self, env_action: np.ndarray, enose_state: int = IDLE) -> np.ndarray:
        env_action = np.asarray(env_action, dtype=float).ravel()
        if self.cosim.odor_mode == "discrete":
            return np.concatenate([env_action, [float(enose_state)]])
        return env_action

    def _new_recorder(self, instruction: str) -> EpisodeRecorder:
        liquids = list(getattr(self.env, "object_names", None) or self.liquids)
        if self.instructed_liquid in liquids:
            target_index = liquids.index(self.instructed_liquid)
        else:
            target_index = -1  # instructed VOC absent from the scene
        mox_models = list(self.odor_config.mox_models)
        spawn_layout = {}
        if hasattr(self.env, "spawn_layout"):
            try:
                spawn_layout = self.env.spawn_layout()
            except Exception:  # noqa: BLE001
                spawn_layout = {}
        meta = {
            "objects": liquids,
            "liquids": list(self.liquids),
            "instructed_liquid": self.instructed_liquid,
            "target_present": self.target_present,
            "target_object": self.instructed_liquid if self.target_present else None,
            "target_catalog": self.instructed_liquid,
            "place_target": getattr(self.env, "place_target_name", self.place_target),
            "cup_catalog": getattr(self.env, "cup_catalog_name", "porcelain_mug"),
            "spawn_layout": spawn_layout,
            "robots": self.robots,
            "control_freq": self.control_freq,
            "action_layout": (
                "[robosuite_action..., enose_state]"
                if self.odor_mode == "discrete"
                else "[robosuite_action...]"
            ),
            "controller_type": self.controller_type,
            "scenario": self.scenario,
            "scene_id": self.scene_id,
            "episode_id": self._episode_id or str(uuid.uuid4()),
            "seed": None if self.seed is None else int(self.seed),
            "odor_mode": self.odor_mode,
            "collector": "scripted_odor_place",
            "env_name": self.env_name,
            "schema_version": "ovla-odor-0.1",
            **self.odor_config.to_meta(),
            "mox_model": mox_models[0] if mox_models else None,
        }
        return EpisodeRecorder(
            self.out_dir,
            instruction,
            self.gas_types,
            meta,
            camera_map=self.camera_map,
            target_index=max(target_index, 0),
            num_classes=max(len(liquids), 1),
        )

    def _record_step(self, recorder: EpisodeRecorder, obs, action: np.ndarray, info: dict):
        ppm = info.get("ppm") or {}
        sim_time = float(info.get("sim_time", 0.0))
        enose_pos = self.env.get_enose_site_pose()[0]
        # Distance to the cup currently being sniffed (or instructed target).
        oid = self._sniff_object_id
        if oid is None or oid not in self.env.get_object_world_poses():
            oid = (
                self.instructed_liquid
                if self.target_present
                else self.liquids[0]
            )
        target_pos = self.env.get_object_world_poses()[oid][0]
        source_distance = float(
            np.linalg.norm(
                np.asarray(enose_pos, dtype=float) - np.asarray(target_pos, dtype=float)
            )
        )
        packed = np.asarray(action, dtype=float).ravel()
        if self.odor_mode == "discrete" and packed.size == self.env.action_dim:
            packed = np.concatenate([packed, [float(self._active_enose)]])
        enose_state = int(info.get("enose_state", self._active_enose))
        recorder.add(
            sim_time=sim_time,
            state=self._proprio_state(obs),
            action=packed,
            enose_state=enose_state,
            sampling_active=bool(info.get("sampling_active", enose_state == SAMPLE)),
            ppm=ppm,
            source_distance=source_distance,
            enose_pos=enose_pos,
            target_pos=target_pos,
            proprio=self._structured_proprio(obs),
            frames=self._extract_frames(obs) if self.camera_map else None,
            enose_voltage=info.get("enose_voltage"),
            enose_voltages=info.get("enose_voltages"),
            phase=int(self._active_phase),
        )

    def _step(self, recorder: EpisodeRecorder, obs, action: np.ndarray):
        packed = self._pack(action, self._active_enose)
        obs, reward, done, info = self.cosim.step(packed)
        if self.has_renderer:
            self.env.render()
        self._record_step(recorder, obs, packed, info)
        return obs, reward, done, info

    def _hold_enose(
        self,
        recorder,
        obs,
        gripper: float,
        n_steps: int,
        enose_state: int,
        *,
        target_pos: "np.ndarray | None" = None,
    ):
        """Hold arm (optionally servo to target_pos) under a fixed enose state."""
        self._active_enose = int(enose_state)
        if int(enose_state) == SAMPLE:
            self._active_phase = PHASE_SNIFF_HOLD
        elif int(enose_state) == FILTER:
            self._active_phase = PHASE_FILTER_HOLD
        else:
            self._active_phase = PHASE_CONTROL
        for _ in range(max(0, n_steps)):
            if target_pos is not None:
                action = self._pose_delta_action(
                    obs, target_pos, gripper, track_fingers=False, noise=False
                )
            else:
                action = np.zeros(self._action_dim(), dtype=float)
                gdof = self._gripper_dof()
                if gdof > 0:
                    action[-gdof:] = gripper
            obs, _, _, _ = self._step(recorder, obs, action)
        return obs

    # ------------------------------------------------------------------ #
    # Sniff / filter / home
    # ------------------------------------------------------------------ #
    def _sniff_hover_pos(self, cup_id: str) -> np.ndarray:
        pos = np.asarray(self.env.get_object_world_poses()[cup_id][0], dtype=float).ravel()[:3]
        out = pos.copy()
        out[2] = float(pos[2]) + self.sniff_z
        return out

    def _filter_aside_pos(self, obs) -> np.ndarray:
        """Elevated hover over the plate so FILTER stays inside agentview.

        Discrete mode already feeds clean air during FILTER, so the move only
        has to leave the cup. The old +Y lateral offset walked the arm off the
        tight OVLA camera frustum.
        """
        cur = self._eef_pos(obs)
        aside = cur.copy()
        plate_id = getattr(self.env, "place_target_name", None) or self.place_target
        try:
            plate = np.asarray(self._object_pos(plate_id), dtype=float).ravel()[:3]
            aside[0] = float(plate[0])
            aside[1] = float(plate[1]) + float(self.filter_offset_y)
        except Exception:  # noqa: BLE001
            aside[0] = float(getattr(self.env, "table_offset", np.zeros(3))[0])
            aside[1] = float(getattr(self.env, "table_offset", np.zeros(3))[1]) + float(
                self.filter_offset_y
            )
        home_z = (
            float(self.home_pos[2]) if self.home_pos is not None else float(cur[2])
        )
        aside[2] = max(float(cur[2]), home_z)
        return aside

    def _sniff_cup(self, recorder, obs, cup_id: str):
        """Approach cup, SAMPLE-hold, return (obs, is_instructed)."""
        self._sniff_object_id = cup_id
        hover = self._sniff_hover_pos(cup_id)
        self._active_enose = IDLE
        self._active_phase = PHASE_CONTROL
        self._log(f"approach sniff {cup_id!r} → {self._fmt_xyz(hover)}")
        obs, ok = self._go_to(
            recorder,
            obs,
            hover,
            GRIPPER_OPEN,
            track_fingers=False,
            phase=f"sniff_approach_{cup_id}",
        )
        if not ok:
            return obs, False, False

        sample_steps = self.odor_config.sniff_steps
        start_step = len(recorder)
        self._log(f"SAMPLE {self.sample_hold_s:.1f}s ({sample_steps} steps) over {cup_id!r}")
        obs = self._hold_enose(
            recorder, obs, GRIPPER_OPEN, sample_steps, SAMPLE, target_pos=hover
        )
        end_step = len(recorder) - 1
        is_match = cup_id == self.instructed_liquid
        self._log(f"sniff decision GT: match={is_match} (instructed={self.instructed_liquid!r})")
        enose_pos = self.env.get_enose_site_pose()[0]
        cup_pos = self.env.get_object_world_poses()[cup_id][0]
        self._odor_events.append(
            {
                "event_id": len(self._odor_events),
                "object_id": cup_id,
                "voc_class": cup_id,
                "start_step": start_step,
                "end_step": end_step,
                "decision": "grasp" if is_match else "reject",
                "robot_sniff_pose": np.asarray(hover, dtype=float).ravel()[:3],
                "sensor_object_distance": float(
                    np.linalg.norm(
                        np.asarray(enose_pos, dtype=float) - np.asarray(cup_pos, dtype=float)
                    )
                ),
            }
        )
        self._active_enose = IDLE
        self._active_phase = PHASE_CONTROL
        return obs, True, is_match

    def _filter_between(self, recorder, obs):
        """Move aside and FILTER-hold so the MOX can decay before the next sniff."""
        aside = self._filter_aside_pos(obs)
        self._active_enose = IDLE
        self._active_phase = PHASE_CONTROL
        self._log(f"filter aside → {self._fmt_xyz(aside)}")
        obs, ok = self._go_to(
            recorder,
            obs,
            aside,
            GRIPPER_OPEN,
            track_fingers=False,
            phase="filter_aside",
        )
        filter_steps = self.odor_config.filter_steps
        self._log(f"FILTER {self.filter_hold_s:.1f}s ({filter_steps} steps)")
        obs = self._hold_enose(
            recorder, obs, GRIPPER_OPEN, filter_steps, FILTER, target_pos=aside
        )
        self._active_enose = IDLE
        self._active_phase = PHASE_CONTROL
        return obs, ok

    def _go_home(self, recorder, obs):
        if self.home_pos is None:
            return obs, True
        self._active_enose = IDLE
        self._active_phase = PHASE_CONTROL
        self._log(f"return home → {self._fmt_xyz(self.home_pos)}")
        obs, ok = self._go_to(
            recorder,
            obs,
            self.home_pos,
            GRIPPER_OPEN,
            track_fingers=False,
            phase="home",
        )
        obs = self._hold_enose(
            recorder, obs, GRIPPER_OPEN, self.settle_steps, IDLE, target_pos=self.home_pos
        )
        return obs, ok

    def _sniff_order(self) -> list[str]:
        """Ordered cup ids to sniff this episode.

        When the instructed VOC is on the table:
          * with prob ``target_first_frac`` (default 0.4) sniff it first
          * otherwise sniff a distractor first, then the target (then any rest)

        When absent, shuffle all on-table cups (no target to prioritize).
        """
        cups = list(self.liquids)
        if not self.target_present or self.instructed_liquid not in cups:
            self.rng.shuffle(cups)
            return cups

        target = self.instructed_liquid
        distractors = [c for c in cups if c != target]
        self.rng.shuffle(distractors)

        if self.rng.random() < self.target_first_frac or not distractors:
            return [target] + distractors
        # 60% default: wrong cup first → FILTER → target.
        return [distractors[0], target] + distractors[1:]

    # ------------------------------------------------------------------ #
    # Pick-place (reuses parent motion helpers; cup_id = liquid name)
    # ------------------------------------------------------------------ #
    def _pick_and_place(self, recorder, obs, cup_id: str, *, from_sniff: bool = False):
        """Grasp ``cup_id`` and place on the plate.

        When ``from_sniff`` is True the EE is already hovering above the cup
        after SAMPLE, so we skip the raise / high re-approach and slide to the
        rim grasp XY at the current height, then descend.
        """
        plate_id = self.env.place_target_name
        dwell_steps = max(1, int(round(self.grasp_dwell_s * self.control_freq)))
        cup_z0 = float(self._object_pos(cup_id)[2])
        min_lift = 0.05
        self._active_enose = IDLE
        self._active_phase = PHASE_CONTROL

        def fail(reason: str):
            self._log(f"pick_place {reason}")
            return obs, False

        def _plan_grasp():
            cup_pos = self._object_pos(cup_id)
            # target_object may be a different liquid; resolve by id.
            cup_obj = None
            for o in self.env.odor_objects:
                if o.object_id == cup_id:
                    cup_obj = o
                    break
            if cup_obj is None:
                raise RuntimeError(f"cup {cup_id!r} not among odor_objects")
            rim_xy_ = self._rim_grasp_xy(cup_pos, cup_obj)
            cup_bot_z, cup_top_z = self._object_collision_z_bounds(cup_id)
            grasp_finger_z_ = cup_top_z + self.grasp_z_offset
            safe_z_ = max(grasp_finger_z_ + self.hover_z, cup_top_z + self.hover_z)
            return cup_pos, rim_xy_, grasp_finger_z_, safe_z_

        cup, rim_xy, grasp_finger_z, safe_z = _plan_grasp()
        if from_sniff:
            # Already above the cup from the sniff hover — keep height, only
            # recenter XY onto the rim pinch point, then descend.
            fingers = self._finger_midpoint_pos()
            approach_z = max(float(fingers[2]), float(grasp_finger_z) + 0.03)
            rim_hover = np.array([rim_xy[0], rim_xy[1], approach_z], dtype=float)
            self._log(
                f"grasp from sniff (skip raise): rim_hover={self._fmt_xyz(rim_hover)}"
            )
            obs, ok = self._go_to(
                recorder,
                obs,
                rim_hover,
                GRIPPER_OPEN,
                track_fingers=True,
                phase="sniff_to_rim",
            )
            if not ok:
                return fail("sniff_to_rim failed")
        else:
            fingers = self._finger_midpoint_pos()
            raise_target = np.array([fingers[0], fingers[1], safe_z], dtype=float)
            obs, ok = self._go_to(
                recorder,
                obs,
                raise_target,
                GRIPPER_OPEN,
                track_fingers=True,
                phase="raise",
            )
            if not ok:
                return fail("raise failed")

        pickup_finger_z = float(grasp_finger_z)
        grasp_xy_offset = np.asarray(rim_xy[:2], dtype=float) - np.asarray(
            cup[:2], dtype=float
        )
        grasped = False
        for g_attempt in range(1, self.max_grasp_attempts + 1):
            if len(recorder) >= self.max_episode_steps:
                return fail("episode cap during grasp")
            cup, rim_xy, grasp_finger_z, safe_z = _plan_grasp()

            # First attempt after a sniff is already at rim XY; later retries
            # (or non-sniff entry) need the high hover.
            need_hover = not (from_sniff and g_attempt == 1)
            if need_hover:
                hover = np.array([rim_xy[0], rim_xy[1], safe_z], dtype=float)
                obs, ok = self._go_to(
                    recorder,
                    obs,
                    hover,
                    GRIPPER_OPEN,
                    xy_first=True,
                    track_fingers=True,
                    phase=f"hover_{g_attempt}",
                )
                if not ok:
                    continue

            grasp = np.array([rim_xy[0], rim_xy[1], grasp_finger_z], dtype=float)
            obs, ok = self._go_to(
                recorder,
                obs,
                grasp,
                GRIPPER_OPEN,
                track_fingers=True,
                xy_tol=self.grasp_xy_tol,
                z_tol=self.grasp_z_tol,
                phase=f"grasp_descend_{g_attempt}",
            )
            if not ok:
                continue
            press = grasp.copy()
            press[2] = float(grasp[2]) - self.press_z
            obs = self._hold_at(recorder, obs, press, GRIPPER_OPEN, dwell_steps)
            obs = self._hold_at(recorder, obs, press, GRIPPER_CLOSE, dwell_steps)
            pickup_finger_z = float(self._finger_midpoint_pos()[2])
            grasp_xy_offset = np.asarray(rim_xy[:2], dtype=float) - np.asarray(
                cup[:2], dtype=float
            )
            fingers = self._finger_midpoint_pos()
            lift_target = np.array(
                [fingers[0], fingers[1], float(fingers[2]) + self.lift_z], dtype=float
            )
            obs, ok = self._go_to(
                recorder,
                obs,
                lift_target,
                GRIPPER_CLOSE,
                track_fingers=True,
                phase=f"lift_{g_attempt}",
            )
            cup_z_after = (
                float(self.env.object_body_z(cup_id))
                if hasattr(self.env, "object_body_z")
                else float(self._object_pos(cup_id)[2])
            )
            if ok and cup_z_after >= cup_z0 + min_lift:
                grasped = True
                break
            # reopen + raise for retry
            obs = self._hold(recorder, obs, GRIPPER_OPEN, max(self.grip_hold_steps, 10))
            fingers = self._finger_midpoint_pos()
            obs, _ = self._go_to(
                recorder,
                obs,
                np.array([fingers[0], fingers[1], safe_z], dtype=float),
                GRIPPER_OPEN,
                track_fingers=True,
                phase=f"regrasp_raise_{g_attempt}",
            )

        if not grasped:
            return fail("cup not lifted")

        plate = self._object_pos(plate_id)
        place_xy = np.asarray(plate[:2], dtype=float) + grasp_xy_offset
        fingers = self._finger_midpoint_pos()
        above_plate = np.array(
            [place_xy[0], place_xy[1], float(fingers[2])], dtype=float
        )
        obs, ok = self._go_to(
            recorder,
            obs,
            above_plate,
            GRIPPER_CLOSE,
            xy_first=True,
            track_fingers=True,
            phase="above_plate",
        )
        if not ok:
            return fail("above_plate failed")

        place = np.array(
            [place_xy[0], place_xy[1], pickup_finger_z + self.place_z], dtype=float
        )
        obs, ok = self._go_to(
            recorder, obs, place, GRIPPER_CLOSE, track_fingers=True, phase="place_descend"
        )
        if not ok:
            return fail("place_descend failed")
        obs = self._hold_at(recorder, obs, place, GRIPPER_OPEN, self.grip_hold_steps)
        obs = self._hold(recorder, obs, GRIPPER_OPEN, self.settle_steps)
        fingers = self._finger_midpoint_pos()
        retreat = fingers.copy()
        retreat[2] = float(fingers[2]) + self.retreat_z
        obs, _ = self._go_to(
            recorder,
            obs,
            retreat,
            GRIPPER_OPEN,
            max_steps=80,
            track_fingers=True,
            phase="retreat",
        )
        return obs, True

    # ------------------------------------------------------------------ #
    # Episode
    # ------------------------------------------------------------------ #
    def run_episode(self) -> "Path | None":
        obs = self.cosim.reset()
        instruction = self.instruction_override or self.env.instruction
        if not self.instruction_override:
            # Prefer language that names the *instructed* liquid even when absent.
            # Display name drops strength (propanol1.0 → propanol).
            instruction = build_instruction(self.instructed_liquid, self.place_target)
            self.env.instruction = instruction

        self._episode_id = str(uuid.uuid4())
        recorder = self._new_recorder(instruction)
        self._active_enose = IDLE
        self._active_phase = PHASE_CONTROL
        self._sniff_object_id = None
        self._odor_events = []

        zero = np.zeros(self._action_dim(), dtype=float)
        gdof = self._gripper_dof()
        if gdof > 0:
            zero[-gdof:] = GRIPPER_OPEN
        info0 = {"ppm": {}, "sim_time": 0.0}
        if self.cosim.has_enose:
            info0.update(self.cosim.snapshot_enose(IDLE))
        self._record_step(recorder, obs, self._pack(zero, IDLE), info0)

        def fail(reason: str = "failed"):
            self._log(f"episode {reason}")
            self._discard_recorder(recorder)
            return None

        obs = self._hold_enose(
            recorder, obs, GRIPPER_OPEN, max(self.settle_steps, 20), IDLE
        )
        self.home_pos = self._eef_pos(obs).copy()
        self._log(
            f"home={self._fmt_xyz(self.home_pos)} liquids={self.liquids} "
            f"instructed={self.instructed_liquid!r} present={self.target_present}"
        )

        order = self._sniff_order()
        self._log(f"sniff order={order}")

        placed = False
        for i, cup_id in enumerate(order):
            if len(recorder) >= self.max_episode_steps:
                return fail("episode cap")
            obs, ok, is_match = self._sniff_cup(recorder, obs, cup_id)
            if not ok:
                return fail(f"sniff approach failed for {cup_id}")
            if is_match:
                obs, ok = self._pick_and_place(
                    recorder, obs, cup_id, from_sniff=True
                )
                if not ok:
                    return fail("pick_place failed")
                placed = True
                break
            # Wrong cup (or instructed absent): purge before the next sniff,
            # or before going home after the last distractor.
            if i < len(order) - 1 or not self.target_present:
                obs, ok = self._filter_between(recorder, obs)
                if not ok:
                    return fail("filter aside failed")

        if placed:
            if self.target_present and not self.env._check_success():
                return fail("success check false after place")
            self._log("episode SUCCESS (placed)")
            return self._finish_episode(recorder, success=True)

        # Instructed liquid absent (or never matched): return home.
        obs, ok = self._go_home(recorder, obs)
        if not ok:
            return fail("home failed")
        if self.target_present:
            # Should have found it — treat as failure so we don't poison demos.
            return fail("target present but never matched")
        self._log("episode SUCCESS (absent → home)")
        return self._finish_episode(recorder, success=True)

    def _finish_episode(self, recorder: EpisodeRecorder, *, success: bool) -> "Path | None":
        path = recorder.write(success=success)
        if path is None:
            return None
        if self._odor_events:
            try:
                write_odor_events(
                    path,
                    self._odor_events,
                    mox_models=list(self.odor_config.mox_models),
                    gas_types=list(self.gas_types),
                    sniff_steps=self.odor_config.sniff_steps,
                )
            except Exception as exc:  # noqa: BLE001
                self._log(f"odor_events write failed: {exc}")
                self._discard_recorder(recorder)
                return None
        return path

    def collect(self, num_demos: int, *, max_attempts: "int | None" = None) -> list[Path]:
        if num_demos <= 0:
            return []
        max_attempts = int(max_attempts) if max_attempts is not None else max(num_demos * 20, 20)
        saved: list[Path] = []
        attempts = 0
        while len(saved) < num_demos and attempts < max_attempts:
            attempts += 1
            t0 = time.time()
            try:
                path = self.run_episode()
            except Exception as exc:  # noqa: BLE001
                print(f"[scripted_odor_place] attempt {attempts} error: {exc}")
                path = None
            dt = time.time() - t0
            if path is not None:
                saved.append(path)
                print(
                    f"[scripted_odor_place] saved {len(saved)}/{num_demos} "
                    f"({path.name}, {dt:.1f}s, attempt {attempts})"
                )
            else:
                print(
                    f"[scripted_odor_place] attempt {attempts} failed "
                    f"({dt:.1f}s); successes={len(saved)}/{num_demos}"
                )
        return saved


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Scripted smell→decide→pick-place collector (ClassifyLiquidPlaceOVLA)"
    )
    p.add_argument(
        "--liquids",
        nargs="+",
        default=list(DEFAULT_LIQUIDS),
        help="Spawned liquid cups (object ids / catalog names)",
    )
    p.add_argument(
        "--instructed",
        default="ethanol",
        help="Liquid named in the language instruction (may be absent from --liquids)",
    )
    p.add_argument("--place-target", default="plate")
    p.add_argument("--instruction", default="", help="Override language instruction")
    p.add_argument("--num-demos", type=int, default=5)
    p.add_argument("--max-attempts", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    p.add_argument("--robots", default="Panda")
    p.add_argument("--control-freq", type=int, default=int(DEFAULT_ODOR_CONFIG.control_freq))
    p.add_argument("--camera-size", type=int, default=256)
    p.add_argument("--no-frames", action="store_true")
    p.add_argument("--render", action="store_true")
    p.add_argument("--max-episode-steps", type=int, default=4000)
    p.add_argument("--odor-mode", default="discrete", choices=("discrete", "continuous", "none"))
    p.add_argument(
        "--sniff-duration",
        type=float,
        default=None,
        help=(
            f"Seconds the arm hovers above a cup while SAMPLE-ing "
            f"(default {DEFAULT_SAMPLE_HOLD_S})"
        ),
    )
    p.add_argument(
        "--sample-hold-s",
        type=float,
        default=None,
        help="Legacy alias for --sniff-duration",
    )
    p.add_argument(
        "--filter-hold-s",
        type=float,
        default=None,
        help=(
            "FILTER nose-rest seconds after a wrong cup "
            f"(default {FILTER_DURATION_FACTOR}× sniff = {DEFAULT_FILTER_HOLD_S} at "
            f"{DEFAULT_SAMPLE_HOLD_S}s sniff)"
        ),
    )
    p.add_argument("--sniff-z", type=float, default=DEFAULT_SNIFF_Z)
    p.add_argument(
        "--filter-offset-y",
        type=float,
        default=DEFAULT_FILTER_OFFSET_Y,
        help="Optional Y nudge from the plate hover used during FILTER (default 0; stays in frame)",
    )
    p.add_argument(
        "--target-first-frac",
        type=float,
        default=DEFAULT_TARGET_FIRST_FRAC,
        help=(
            "When the instructed VOC is on the table, probability of sniffing "
            f"it first (default {DEFAULT_TARGET_FIRST_FRAC}). Remainder sniff a "
            "distractor first, FILTER, then the target."
        ),
    )
    p.add_argument("--quiet", action="store_true")
    add_motion_shaping_args(p)
    return p


def main(argv=None) -> int:
    args = build_argparser().parse_args(argv)
    with ScriptedOdorPlaceCollector(
        out_dir=args.out_dir,
        liquids=list(args.liquids),
        instructed_liquid=args.instructed,
        place_target=args.place_target,
        instruction=args.instruction,
        robots=args.robots,
        control_freq=args.control_freq,
        camera_size=args.camera_size,
        record_frames=not args.no_frames,
        has_renderer=args.render,
        seed=int(args.seed),
        odor_mode=args.odor_mode,
        sniff_duration_s=args.sniff_duration,
        sample_hold_s=args.sample_hold_s,
        filter_hold_s=args.filter_hold_s,
        sniff_z=float(args.sniff_z),
        filter_offset_y=float(args.filter_offset_y),
        target_first_frac=float(args.target_first_frac),
        max_episode_steps=int(args.max_episode_steps),
        verbose=not args.quiet,
        **motion_shaping_kwargs(args),
    ) as collector:
        print(
            f"[scripted_odor_place] liquids={args.liquids} "
            f"instructed={args.instructed!r} present={collector.target_present} "
            f"instruction={collector.env.instruction!r}"
        )
        saved = collector.collect(int(args.num_demos), max_attempts=args.max_attempts)
    print(f"[scripted_odor_place] done: saved {len(saved)} demos under {args.out_dir}")
    return 0 if saved else 1


if __name__ == "__main__":
    raise SystemExit(main())

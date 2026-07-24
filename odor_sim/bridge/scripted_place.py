"""Scripted OSC pick-place collector for the OdorPlace task.

Drives the EE toward cup/plate waypoints with OSC_POSE position deltas
(robosuite's operational-space controller — practical “IK to object”),
records successful episodes with :class:`EpisodeRecorder` (same layout as
teleop), and skips GADEN (``auto_start_gaden=False``).

Example::

    python -m odor_sim.bridge.scripted_place \\
        --num-demos 20 --num-cups 2 --out-dir datasets/teleop/odorplace_scripted
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from odor_sim.bridge.teleop import (
    MINING_CAMERAS,
    EpisodeRecorder,
    _camera_dirname,
    _controller_type,
    _PROPRIO_KEYS,
)
from odor_sim.envs.odor_place import MUG_POOL

# Gripper action for robosuite OSC_POSE / GRIP (Panda): +1 closes, -1 opens.
GRIPPER_OPEN = -1.0
GRIPPER_CLOSE = 1.0

DEFAULT_OUT_DIR = "datasets/teleop/odorplace_scripted"

# OSC_POSE maps action in [-1, 1] -> EE delta in meters (BASIC default ±0.05 m).
OSC_POS_SCALE = 0.05

# Default mug rim radius (m) if the object has no usable horizontal_radius.
DEFAULT_RIM_RADIUS = 0.045
# Place EE this fraction of radius out from center so jaws pinch the wall.
RIM_RADIUS_FRAC = 0.85


class ScriptedPlaceCollector:
    """Closed-loop OSC waypoint pick → place collector."""

    def __init__(
        self,
        *,
        out_dir: "str | Path" = DEFAULT_OUT_DIR,
        cups=None,
        target_cup=None,
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
        # Motion gains / thresholds (meters, normalized action units, steps).
        pos_gain: float = 8.0,
        max_pos_action: float = 1.0,
        xy_tol: float = 0.015,
        z_tol: float = 0.015,
        hover_z: float = 0.18,
        # Finger-midpoint grasp Z relative to collision AABB max Z (m).
        # Default: 2 cm below rim so pads pinch the wall (1 mm was too shallow).
        grasp_z_offset: float = -0.02,
        # Extra clearance above pickup finger Z when placing (m).
        place_z: float = 0.01,
        lift_z: float = 0.16,
        retreat_z: float = 0.20,
        rim_radius: "float | None" = None,
        rim_radius_frac: float = RIM_RADIUS_FRAC,
        phase_max_steps: int = 200,
        grip_hold_steps: int = 15,
        settle_steps: int = 20,
        max_episode_steps: int = 1200,
        verbose: bool = True,
    ):
        import odor_sim as odorsim
        from robosuite.controllers import load_composite_controller_config

        self.out_dir = Path(out_dir)
        self.verbose = bool(verbose)
        self.cups = list(cups) if cups is not None else ["porcelain_mug"]
        self.target_cup = target_cup
        self.place_target = place_target
        self.instruction_override = instruction
        self.robots = robots
        self.control_freq = control_freq
        self.record_frames = bool(record_frames)
        self.mining_cameras = list(mining_cameras)
        self.camera_size = int(camera_size)
        self.has_renderer = bool(has_renderer)
        self.seed = seed

        self.pos_gain = float(pos_gain)
        self.max_pos_action = float(max_pos_action)
        self.xy_tol = float(xy_tol)
        self.z_tol = float(z_tol)
        self.hover_z = float(hover_z)
        self.grasp_z_offset = float(grasp_z_offset)
        self.place_z = float(place_z)
        self.lift_z = float(lift_z)
        self.retreat_z = float(retreat_z)
        self.rim_radius = None if rim_radius is None else float(rim_radius)
        self.rim_radius_frac = float(rim_radius_frac)
        self.phase_max_steps = int(phase_max_steps)
        self.grip_hold_steps = int(grip_hold_steps)
        self.settle_steps = int(settle_steps)
        self.max_episode_steps = int(max_episode_steps)

        self.camera_map = (
            {f"{c}_image": _camera_dirname(c) for c in self.mining_cameras}
            if self.record_frames
            else {}
        )

        controller_config = load_composite_controller_config(
            controller=controller, robot=robots
        )
        self.controller_type = _controller_type(controller_config)

        # Always enable offscreen cams for mining; disable only via record_frames=False
        # still keeps camera_names so obs keys exist when frames are requested.
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

        make_kwargs = {
            "cups": list(self.cups),
            "place_target": self.place_target,
            "robots": robots,
            "controller_configs": controller_config,
            "control_freq": control_freq,
            "has_renderer": self.has_renderer,
            "has_offscreen_renderer": True,
            "render_camera": render_camera,
            "ignore_done": True,
            "reward_shaping": True,
            "horizon": 100000,
            "use_camera_obs": True,
            **cam_kwargs,
        }
        if self.target_cup is not None:
            make_kwargs["target_cup"] = self.target_cup
        if self.instruction_override:
            make_kwargs["instruction"] = self.instruction_override
        if seed is not None:
            make_kwargs["seed"] = int(seed)

        self.cosim = odorsim.make(
            "OdorPlace",
            auto_start_gaden=False,
            bridge=False,
            export=False,
            odor_mode="none",
            **make_kwargs,
        )
        self.env = self.cosim.env
        self.gas_types = list(getattr(self.env, "gas_types", []) or [])
        self.scenario = "10x6_uniform"
        self.scene_id = getattr(self.cosim, "scene_id", "OdorPlace")

    def close(self):
        self.cosim.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    def _gripper_dof(self) -> int:
        robot = self.env.robots[0]
        arm = robot.arms[0]
        try:
            return int(robot.gripper[arm].dof)
        except Exception:  # noqa: BLE001
            return 1

    def _action_dim(self) -> int:
        return int(self.env.action_dim)

    @staticmethod
    def _proprio_state(obs) -> np.ndarray:
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
        return np.concatenate(
            [np.asarray(obs[k], dtype=np.float32).ravel() for k in sorted(keys)]
        )

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

    def _eef_pos(self, obs) -> np.ndarray:
        for key in ("robot0_eef_pos", "robot0_right_eef_pos"):
            if key in obs:
                return np.asarray(obs[key], dtype=float).ravel()[:3]
        # Fallback: grip site
        return np.asarray(self.env.get_enose_site_pose()[0], dtype=float).ravel()[:3]

    def _finger_midpoint_pos(self) -> np.ndarray:
        """World position between the two Panda fingertip bodies.

        This is the point we want on the cup rim — not ``robot0_eef_pos`` (grip site).
        """
        sim = self.env.sim
        tip_names = (
            "gripper0_right_finger_joint1_tip",
            "gripper0_right_finger_joint2_tip",
        )
        tips = []
        for name in tip_names:
            try:
                bid = sim.model.body_name2id(name)
            except Exception:  # noqa: BLE001
                continue
            tips.append(np.asarray(sim.data.body_xpos[bid], dtype=float).ravel()[:3])
        if len(tips) == 2:
            return 0.5 * (tips[0] + tips[1])
        # Fallback: grip site if tip bodies are missing.
        return self._eef_pos({})

    def _track_pos(self, obs, *, track_fingers: bool) -> np.ndarray:
        return self._finger_midpoint_pos() if track_fingers else self._eef_pos(obs)

    def _object_pos(self, object_id: str) -> np.ndarray:
        return np.asarray(self.env.get_object_world_poses()[object_id][0], dtype=float)

    def _object_collision_z_bounds(self, object_id: str) -> tuple[float, float]:
        """World-frame min/max Z of collision geoms for ``object_id`` (AABB).

        Prefers group-0 / contacting geoms. Falls back to ``top_site`` /
        ``bottom_site`` if no collision geoms are found.
        """
        sim = self.env.sim
        model, data = sim.model, sim.data
        root = self.env._odor_body_ids[object_id]
        body_ids = {root}
        for bid in range(model.nbody):
            b = bid
            while b > 0:
                if b == root:
                    body_ids.add(bid)
                    break
                b = int(model.body_parentid[b])

        zs: list[float] = []
        for gid in range(model.ngeom):
            if int(model.geom_bodyid[gid]) not in body_ids:
                continue
            # Skip visual-only meshes (group 1, no contacts).
            if int(model.geom_group[gid]) == 1:
                continue
            if int(model.geom_contype[gid]) == 0 and int(model.geom_conaffinity[gid]) == 0:
                continue
            pos = np.asarray(data.geom_xpos[gid], dtype=float).ravel()[:3]
            mat = np.asarray(data.geom_xmat[gid], dtype=float).reshape(3, 3)
            size = np.asarray(model.geom_size[gid], dtype=float).ravel()
            gtype = int(model.geom_type[gid])
            # MuJoCo geom types: sphere=2, capsule=3, ellipsoid=4, cylinder=5, box=6, mesh=7
            if gtype == 6:  # box
                corners = np.array(
                    [
                        [sx, sy, sz]
                        for sx in (-size[0], size[0])
                        for sy in (-size[1], size[1])
                        for sz in (-size[2], size[2])
                    ],
                    dtype=float,
                )
                world = corners @ mat.T + pos
                zs.extend(world[:, 2].tolist())
            elif gtype == 2:  # sphere
                zs.extend([pos[2] - size[0], pos[2] + size[0]])
            elif gtype in (3, 5):  # capsule / cylinder: half-height size[1], radius size[0]
                axis = mat[:, 2]
                hh = float(size[1]) if size.size > 1 else float(size[0])
                rad = float(size[0])
                for sign in (-1.0, 1.0):
                    c = pos + sign * hh * axis
                    zs.extend([c[2] - rad, c[2] + rad])
            else:
                # ellipsoid / mesh: conservative radius bound
                r = float(np.max(size)) if size.size else float(model.geom_rbound[gid])
                zs.extend([pos[2] - r, pos[2] + r])

        if zs:
            return float(min(zs)), float(max(zs))

        # Fallback: placement sites (often inaccurate on scanned mugs).
        body = self._object_pos(object_id)
        obj = None
        for o in self.env.odor_objects:
            if o.object_id == object_id:
                obj = o
                break
        top_off = float(np.asarray(getattr(obj, "top_offset", [0, 0, 0.04])).ravel()[2])
        bot_off = float(np.asarray(getattr(obj, "bottom_offset", [0, 0, -0.06])).ravel()[2])
        return float(body[2] + bot_off), float(body[2] + top_off)

    def _cup_rim_radius(self, cup_obj) -> float:
        """XY radius used for rim offset (prefer object horizontal_radius)."""
        if self.rim_radius is not None:
            return float(self.rim_radius)
        hr = getattr(cup_obj, "horizontal_radius", None)
        if hr is not None:
            # Placement sites were enlarged (~0.08); actual rim is smaller.
            return float(min(max(hr * 0.55, 0.03), 0.055))
        return DEFAULT_RIM_RADIUS

    def _rim_grasp_xy(self, cup_pos: np.ndarray, cup_obj) -> np.ndarray:
        """World XY on the cup rim for a parallel-jaw pinch (not the center).

        Panda's default downward EE opens along world Y, so we offset along +Y
        (or toward the robot if that is clearer): one finger inside the wall,
        one outside.
        """
        radius = self._cup_rim_radius(cup_obj) * self.rim_radius_frac
        # Prefer +Y (gripper open axis). Flip toward robot (-x) if cup is far +y.
        direction = np.array([0.0, 1.0], dtype=float)
        # Bias slightly toward robot base so the approach is reachable.
        to_robot = np.array([-cup_pos[0], -cup_pos[1]], dtype=float)
        n = np.linalg.norm(to_robot)
        if n > 1e-6:
            to_robot /= n
            # Blend: mostly gripper-axis, a bit toward robot for reachability.
            direction = 0.7 * direction + 0.3 * to_robot
            direction = direction / (np.linalg.norm(direction) + 1e-9)
        return cup_pos[:2] + radius * direction

    def _pose_delta_action(
        self,
        obs,
        target_pos: np.ndarray,
        gripper: float,
        *,
        track_fingers: bool = False,
    ) -> np.ndarray:
        """Normalized OSC_POSE delta in [-1, 1] toward ``target_pos`` (+ gripper).

        When ``track_fingers`` is True, the error is measured from the fingertip
        midpoint (not the EEF grip site). OSC still commands the arm via EEF
        deltas — same direction as the finger error for small offsets.
        """
        cur = self._track_pos(obs, track_fingers=track_fingers)
        err = np.asarray(target_pos, dtype=float).ravel()[:3] - cur
        unit = (self.pos_gain * err) / OSC_POS_SCALE
        unit = np.clip(unit, -self.max_pos_action, self.max_pos_action)
        action = np.zeros(self._action_dim(), dtype=float)
        n_pose = min(3, action.size)
        action[:n_pose] = unit[:n_pose]
        gdof = self._gripper_dof()
        if gdof > 0 and action.size >= gdof:
            action[-gdof:] = gripper
        return action

    def _at_target(
        self,
        obs,
        target_pos: np.ndarray,
        *,
        check_z: bool = True,
        track_fingers: bool = False,
    ) -> bool:
        cur = self._track_pos(obs, track_fingers=track_fingers)
        xy_ok = np.linalg.norm(cur[:2] - target_pos[:2]) < self.xy_tol
        if not check_z:
            return bool(xy_ok)
        z_ok = abs(float(cur[2] - target_pos[2])) < self.z_tol
        return bool(xy_ok and z_ok)

    def _holding_target(self) -> bool:
        try:
            return bool(
                self.env._check_grasp(
                    gripper=self.env.robots[0].gripper[self.env.robots[0].arms[0]],
                    object_geoms=self.env.target_object,
                )
            )
        except Exception:  # noqa: BLE001
            return False

    def _new_recorder(self, instruction: str) -> EpisodeRecorder:
        object_names = list(getattr(self.env, "object_names", None) or [])
        target_name = getattr(self.env, "target_object_name", None)
        if target_name is not None and target_name in object_names:
            target_index = object_names.index(target_name)
        else:
            target_index = 0
        meta = {
            "objects": object_names or None,
            "cups": list(getattr(self.env, "cup_catalog_names", []) or self.cups),
            "target_object": target_name,
            "target_catalog": getattr(self.env, "target_catalog_name", None),
            "place_target": getattr(self.env, "place_target_name", self.place_target),
            "robots": self.robots,
            "control_freq": self.control_freq,
            "action_layout": "[robosuite_action..., enose_state]",
            "controller_type": self.controller_type,
            "scenario": self.scenario,
            "scene_id": self.scene_id,
            "odor_mode": "none",
            "collector": "scripted_place",
            "mox_models": [],
            "mox_model": None,
        }
        return EpisodeRecorder(
            self.out_dir,
            instruction,
            self.gas_types,
            meta,
            camera_map=self.camera_map,
            target_index=target_index,
            num_classes=max(len(object_names), 1),
        )

    def _record_step(self, recorder: EpisodeRecorder, obs, action: np.ndarray, info: dict):
        ppm = info.get("ppm") or {}
        sim_time = float(info.get("sim_time", 0.0))
        enose_pos = self.env.get_enose_site_pose()[0]
        target_name = self.env.target_object_name
        target_pos = self.env.get_object_world_poses()[target_name][0]
        source_distance = float(
            np.linalg.norm(
                np.asarray(enose_pos, dtype=float) - np.asarray(target_pos, dtype=float)
            )
        )
        env_action = np.asarray(action, dtype=float).ravel()
        recorder.add(
            sim_time=sim_time,
            state=self._proprio_state(obs),
            action=np.concatenate([env_action, [0]]),  # trailing enose_state=IDLE
            enose_state=0,
            sampling_active=False,
            ppm=ppm,
            source_distance=source_distance,
            enose_pos=enose_pos,
            target_pos=target_pos,
            proprio=self._structured_proprio(obs),
            frames=self._extract_frames(obs) if self.camera_map else None,
        )

    def _step(self, recorder: EpisodeRecorder, obs, action: np.ndarray):
        obs, reward, done, info = self.cosim.step(action)
        if self.has_renderer:
            self.env.render()
        self._record_step(recorder, obs, action, info)
        return obs, reward, done, info

    def _hold(self, recorder, obs, gripper: float, n_steps: int):
        action = np.zeros(self._action_dim(), dtype=float)
        gdof = self._gripper_dof()
        if gdof > 0:
            action[-gdof:] = gripper
        for _ in range(max(0, n_steps)):
            obs, _, _, _ = self._step(recorder, obs, action)
        return obs

    @staticmethod
    def _fmt_xyz(p) -> str:
        a = np.asarray(p, dtype=float).ravel()
        return f"[{a[0]:+.4f}, {a[1]:+.4f}, {a[2]:+.4f}]"

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(f"[scripted_place] {msg}", flush=True)

    def _log_reach(self, phase: str, obs, target_pos, *, track_fingers: bool) -> None:
        tracked = self._track_pos(obs, track_fingers=track_fingers)
        what = "fingers" if track_fingers else "eef"
        err = np.asarray(target_pos, dtype=float).ravel()[:3] - tracked
        self._log(
            f"{phase}: target={self._fmt_xyz(target_pos)}  "
            f"{what}={self._fmt_xyz(tracked)}  "
            f"eef={self._fmt_xyz(self._eef_pos(obs))}  "
            f"err={self._fmt_xyz(err)}"
        )

    def _go_to(
        self,
        recorder,
        obs,
        target_pos: np.ndarray,
        gripper: float,
        *,
        check_z: bool = True,
        max_steps: "int | None" = None,
        xy_first: bool = False,
        track_fingers: bool = False,
        phase: str = "go_to",
    ):
        """Move tracked point (EEF or finger midpoint) toward ``target_pos``."""
        limit = self.phase_max_steps if max_steps is None else int(max_steps)
        self._log_reach(f"{phase} start", obs, target_pos, track_fingers=track_fingers)
        if xy_first:
            cur = self._track_pos(obs, track_fingers=track_fingers)
            xy_target = np.array(
                [target_pos[0], target_pos[1], max(float(cur[2]), float(target_pos[2]))],
                dtype=float,
            )
            xy_target[2] = max(float(cur[2]), float(target_pos[2]) + 0.05)
            self._log(f"{phase} xy_first target={self._fmt_xyz(xy_target)}")
            half = max(limit // 2, 1)
            for _ in range(half):
                if self._at_target(
                    obs, xy_target, check_z=False, track_fingers=track_fingers
                ):
                    break
                action = self._pose_delta_action(
                    obs, xy_target, gripper, track_fingers=track_fingers
                )
                if action.size >= 3:
                    action[2] = 0.0
                obs, _, _, _ = self._step(recorder, obs, action)
                if len(recorder) >= self.max_episode_steps:
                    self._log_reach(
                        f"{phase} FAIL (episode cap during xy)",
                        obs,
                        target_pos,
                        track_fingers=track_fingers,
                    )
                    return obs, False
            limit = max(limit - half, 1)

        for step_i in range(limit):
            if self._at_target(
                obs, target_pos, check_z=check_z, track_fingers=track_fingers
            ):
                self._log_reach(
                    f"{phase} ok (step {step_i})",
                    obs,
                    target_pos,
                    track_fingers=track_fingers,
                )
                return obs, True
            action = self._pose_delta_action(
                obs, target_pos, gripper, track_fingers=track_fingers
            )
            obs, _, _, _ = self._step(recorder, obs, action)
            if len(recorder) >= self.max_episode_steps:
                self._log_reach(
                    f"{phase} FAIL (episode cap)",
                    obs,
                    target_pos,
                    track_fingers=track_fingers,
                )
                return obs, False
        ok = self._at_target(
            obs, target_pos, check_z=check_z, track_fingers=track_fingers
        )
        self._log_reach(
            f"{phase} {'ok' if ok else 'FAIL (timeout)'}",
            obs,
            target_pos,
            track_fingers=track_fingers,
        )
        return obs, ok

    @staticmethod
    def _discard_recorder(recorder: EpisodeRecorder) -> None:
        """Remove a partial episode dir (frames may have been written mid-run)."""
        import shutil

        ep_dir = getattr(recorder, "ep_dir", None)
        if ep_dir is not None and Path(ep_dir).exists():
            shutil.rmtree(ep_dir, ignore_errors=True)

    def run_episode(self) -> "Path | None":
        """Run one scripted pick-place; write episode only on success."""
        obs = self.cosim.reset()
        instruction = self.env.instruction
        recorder = self._new_recorder(instruction)
        # Record the post-reset observation with a zero action (start frame).
        zero = np.zeros(self._action_dim(), dtype=float)
        gdof = self._gripper_dof()
        if gdof > 0:
            zero[-gdof:] = GRIPPER_OPEN
        self._record_step(recorder, obs, zero, {"ppm": {}, "sim_time": 0.0})

        cup_id = self.env.target_object_name
        plate_id = self.env.place_target_name

        def fail(reason: str = "failed"):
            self._log(f"episode {reason}")
            self._discard_recorder(recorder)
            return None

        # Open gripper, settle, then rim-grasp using finger-midpoint tracking.
        obs = self._hold(recorder, obs, GRIPPER_OPEN, max(self.settle_steps, 20))

        cup = self._object_pos(cup_id)
        cup_obj = self.env.target_object
        rim_xy = self._rim_grasp_xy(cup, cup_obj)
        cup_z0 = float(cup[2])
        table_z = float(np.asarray(self.env.table_offset).ravel()[2])
        # Placement sites are often wrong on scanned mugs — use collision AABB.
        cup_bot_z, cup_top_z = self._object_collision_z_bounds(cup_id)
        cup_height = cup_top_z - cup_bot_z
        top_off = float(np.asarray(getattr(cup_obj, "top_offset", [0, 0, 0])).ravel()[2])
        site_top_z = cup_z0 + top_off
        # Descend fingers below collision rim so pads pinch the wall.
        grasp_finger_z = cup_top_z + self.grasp_z_offset
        safe_z = max(grasp_finger_z + self.hover_z, cup_top_z + self.hover_z)
        rim_r = self._cup_rim_radius(cup_obj)
        self._log(
            f"plan cup_id={cup_id!r} cup_body={self._fmt_xyz(cup)} "
            f"table_z={table_z:.4f} coll_bot_z={cup_bot_z:.4f} coll_top_z={cup_top_z:.4f} "
            f"cup_height={cup_height:.4f} (site_top_z={site_top_z:.4f} "
            f"site_vs_coll={site_top_z - cup_top_z:+.4f})"
        )
        self._log(
            f"plan rim_r={rim_r:.4f} rim_xy={self._fmt_xyz([rim_xy[0], rim_xy[1], 0])} "
            f"grasp_z={grasp_finger_z:.4f} (=coll_top_z{self.grasp_z_offset:+.4f}) "
            f"safe_z={safe_z:.4f}"
        )

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

        # Hover above the rim (finger midpoint), then descend to estimated rim Z.
        hover = np.array([rim_xy[0], rim_xy[1], safe_z], dtype=float)
        obs, ok = self._go_to(
            recorder,
            obs,
            hover,
            GRIPPER_OPEN,
            xy_first=True,
            track_fingers=True,
            phase="hover",
        )
        if not ok:
            return fail("hover failed")

        grasp = np.array([rim_xy[0], rim_xy[1], grasp_finger_z], dtype=float)
        obs, ok = self._go_to(
            recorder,
            obs,
            grasp,
            GRIPPER_OPEN,
            track_fingers=True,
            phase="grasp_descend",
        )
        if not ok:
            return fail("grasp_descend failed")

        # Close gripper — jaws pinch the rim wall.
        self._log("close gripper")
        obs = self._hold(recorder, obs, GRIPPER_CLOSE, self.grip_hold_steps)
        # Actual pickup height (may differ slightly from planned grasp_finger_z).
        pickup_finger_z = float(self._finger_midpoint_pos()[2])
        # Rim grasp is offset from cup center; keep that offset so the cup
        # body lands on the plate center (not the rim / edge).
        grasp_xy_offset = np.asarray(rim_xy[:2], dtype=float) - np.asarray(
            cup[:2], dtype=float
        )
        self._log(
            f"grasp_xy_offset={self._fmt_xyz([grasp_xy_offset[0], grasp_xy_offset[1], 0])} "
            f"(rim relative to cup center)"
        )

        # Lift from current finger height (not cup body + lift_z — after a rim
        # grasp fingers are already ~cup_top, so body+lift_z barely rises).
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
            phase="lift",
        )
        if not ok:
            return fail("lift failed")
        cup_z_after = float(self.env.object_body_z(cup_id))
        min_lift = 0.05
        self._log(
            f"lift check cup_z={cup_z_after:.4f} "
            f"(need >={cup_z0 + min_lift:.4f}, delta={cup_z_after - cup_z0:+.4f})"
        )
        if cup_z_after < cup_z0 + min_lift:
            return fail("cup not lifted")

        # Translate above plate at current lift height (don't drop early).
        # Gripper XY = plate center + rim offset → cup center on plate center.
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

        # Place at ~pickup finger Z + small clearance (not plate+const — that
        # drives into the mug when grasping at the rim).
        plate = self._object_pos(plate_id)
        place_xy = np.asarray(plate[:2], dtype=float) + grasp_xy_offset
        place = np.array(
            [place_xy[0], place_xy[1], pickup_finger_z + self.place_z],
            dtype=float,
        )
        self._log(
            f"place_descend target={self._fmt_xyz(place)} "
            f"(pickup_z={pickup_finger_z:.4f}{self.place_z:+.4f}, "
            f"xy=plate+grasp_offset)"
        )
        obs, ok = self._go_to(
            recorder,
            obs,
            place,
            GRIPPER_CLOSE,
            track_fingers=True,
            phase="place_descend",
        )
        if not ok:
            return fail("place_descend failed")

        # Open gripper + settle
        self._log("open gripper / settle")
        obs = self._hold(recorder, obs, GRIPPER_OPEN, self.grip_hold_steps)
        obs = self._hold(recorder, obs, GRIPPER_OPEN, self.settle_steps)

        # Retreat upward from current finger height.
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
        obs = self._hold(recorder, obs, GRIPPER_OPEN, self.settle_steps)

        if not self.env._check_success():
            return fail("success check false")
        self._log("episode SUCCESS")
        return recorder.write(success=True)

    def collect(self, num_demos: int, *, max_attempts: "int | None" = None) -> list[Path]:
        """Collect ``num_demos`` successful episodes (retry on failure)."""
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
                print(f"[scripted_place] attempt {attempts} error: {exc}")
                path = None
            dt = time.time() - t0
            if path is not None:
                saved.append(path)
                print(
                    f"[scripted_place] saved {len(saved)}/{num_demos} "
                    f"({path.name}, {dt:.1f}s, attempt {attempts})"
                )
            else:
                print(
                    f"[scripted_place] attempt {attempts} failed "
                    f"({dt:.1f}s); successes={len(saved)}/{num_demos}"
                )
        if len(saved) < num_demos:
            print(
                f"[scripted_place] stopped early: {len(saved)}/{num_demos} "
                f"after {attempts} attempts"
            )
        return saved


def _sample_cups(rng: np.random.Generator, num_cups: int, pool=MUG_POOL) -> list[str]:
    pool = list(pool)
    if num_cups <= 0:
        raise ValueError("num_cups must be >= 1")
    # Prefer unique types when possible for clearer language labels.
    if num_cups <= len(pool):
        return [str(x) for x in rng.choice(pool, size=num_cups, replace=False)]
    return [str(x) for x in rng.choice(pool, size=num_cups, replace=True)]


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Scripted OSC pick-place collector for OdorPlace")
    p.add_argument("--cups", nargs="+", default=None, help="Catalog mug names to spawn")
    p.add_argument(
        "--num-cups",
        type=int,
        default=2,
        help="If --cups omitted, sample this many mugs from the pool (default 2)",
    )
    p.add_argument(
        "--target-cup",
        default=None,
        help="Target cup (catalog name if unique, instance id, or integer index)",
    )
    p.add_argument("--place-target", default="plate")
    p.add_argument("--instruction", default="", help="Override language instruction")
    p.add_argument("--num-demos", type=int, default=5, help="Successful demos to save")
    p.add_argument("--max-attempts", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    p.add_argument("--robots", default="Panda")
    p.add_argument("--control-freq", type=int, default=20)
    p.add_argument("--camera-size", type=int, default=256)
    p.add_argument("--no-frames", action="store_true", help="Skip PNG frame mining")
    p.add_argument("--render", action="store_true", help="On-screen MJViewer")
    p.add_argument("--max-episode-steps", type=int, default=800)
    p.add_argument(
        "--grasp-z-offset",
        type=float,
        default=-0.02,
        help=(
            "Finger midpoint Z relative to collision AABB max (m). "
            "Default -0.02 = 2 cm below rim for wall pinch."
        ),
    )
    p.add_argument(
        "--quiet",
        action="store_true",
        help="Disable per-phase reach logs (target / fingers / eef / err)",
    )
    return p


def main(argv=None):
    args = build_argparser().parse_args(argv)
    rng = np.random.default_rng(args.seed)

    cups = list(args.cups) if args.cups else _sample_cups(rng, int(args.num_cups))
    target = args.target_cup
    if target is None:
        target = 0  # first cup; placements still randomize each reset
    elif isinstance(target, str) and target.isdigit():
        target = int(target)

    with ScriptedPlaceCollector(
        out_dir=args.out_dir,
        cups=cups,
        target_cup=target,
        place_target=args.place_target,
        instruction=args.instruction,
        robots=args.robots,
        control_freq=args.control_freq,
        camera_size=args.camera_size,
        record_frames=not args.no_frames,
        has_renderer=args.render,
        seed=int(args.seed),
        max_episode_steps=args.max_episode_steps,
        grasp_z_offset=float(args.grasp_z_offset),
        verbose=not args.quiet,
    ) as collector:
        print(
            f"[scripted_place] cups={cups} target={target!r} "
            f"instruction={collector.env.instruction!r}"
        )
        saved = collector.collect(
            int(args.num_demos),
            max_attempts=args.max_attempts,
        )

    print(f"[scripted_place] done: saved {len(saved)} demos under {args.out_dir}")
    return 0 if saved else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Base odor-aware manipulation environment (robosuite 1.5.2).

Adds three things on top of robosuite's :class:`ManipulationEnv`:

  1. a per-task **language instruction** (metadata field, injected into every
     observation as ``obs["instruction"]``),
  2. an **EE odor-sensor site** (the gripper's grip site plus a configurable
     local offset) whose world pose is read each step,
  3. an **OdorObject** layer with a :class:`SceneBuilder` that flattens objects
     into GADEN sources and exposes the ``object_id -> [source indices]`` map.

No ROS / GADEN runtime coupling lives here (that is Phase 4). The env only
*produces* the poses and the source map the bridge will consume.
"""

from __future__ import annotations

from collections import OrderedDict

import numpy as np
from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.arenas import TableArena
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.placement_samplers import UniformRandomSampler
from robosuite.utils.transform_utils import convert_quat

from odor_sim.config.frame_map import FrameMap
from odor_sim.envs.odor_object import OdorObject
from odor_sim.envs.scene_builder import SceneBuilder


class OdorManipulationEnv(ManipulationEnv):
    """Single-arm tabletop env carrying odor objects + an EE odor sensor.

    Subclasses implement :meth:`_make_odor_objects` (return a list of
    :class:`OdorObject`) and may override :meth:`reward` / :meth:`_check_success`.

    Args:
        instruction: natural-language task instruction (the VLA label).
        scenario_config_dir: GADEN environment configuration directory, used to
            build the :class:`FrameMap`. Optional; a default map is used if None.
        enose_site_offset: (x, y, z) offset of the odor sensor from the gripper
            grip site, in the grip-site local frame (meters).
        frame_map: explicit FrameMap (overrides scenario_config_dir).
    """

    def __init__(
        self,
        robots="Panda",
        instruction: str = "",
        scenario_config_dir=None,
        enose_site_offset=(0.0, 0.0, 0.0),
        frame_map: "FrameMap | None" = None,
        env_configuration="default",
        controller_configs=None,
        gripper_types="default",
        initialization_noise="default",
        table_full_size=(0.8, 0.8, 0.05),
        table_friction=(1.0, 5e-3, 1e-4),
        use_camera_obs=True,
        use_object_obs=True,
        reward_scale=1.0,
        reward_shaping=False,
        placement_initializer=None,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="frontview",
        render_collision_mesh=False,
        render_visual_mesh=True,
        render_gpu_device_id=-1,
        control_freq=20,
        lite_physics=True,
        horizon=1000,
        ignore_done=False,
        hard_reset=True,
        camera_names="agentview",
        camera_heights=256,
        camera_widths=256,
        camera_depths=False,
        camera_segmentations=None,
        renderer="mjviewer",
        renderer_config=None,
        seed=None,
    ):
        self.instruction = instruction
        self.enose_site_offset = np.asarray(enose_site_offset, dtype=float).reshape(3)

        if frame_map is not None:
            self.frame_map = frame_map
        elif scenario_config_dir is not None:
            self.frame_map = FrameMap.from_scenario(scenario_config_dir)
        else:
            self.frame_map = FrameMap()

        self.table_full_size = table_full_size
        self.table_friction = table_friction
        self.table_offset = np.array((0, 0, 0.8))

        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping
        self.use_object_obs = use_object_obs
        self.placement_initializer = placement_initializer

        # populated in _load_model / _setup_references
        self.odor_objects: list[OdorObject] = []
        self.scene_builder: "SceneBuilder | None" = None
        self._odor_body_ids: dict[str, int] = {}

        super().__init__(
            robots=robots,
            env_configuration=env_configuration,
            controller_configs=controller_configs,
            base_types="default",
            gripper_types=gripper_types,
            initialization_noise=initialization_noise,
            use_camera_obs=use_camera_obs,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_camera=render_camera,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            render_gpu_device_id=render_gpu_device_id,
            control_freq=control_freq,
            lite_physics=lite_physics,
            horizon=horizon,
            ignore_done=ignore_done,
            hard_reset=hard_reset,
            camera_names=camera_names,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            camera_depths=camera_depths,
            camera_segmentations=camera_segmentations,
            renderer=renderer,
            renderer_config=renderer_config,
            seed=seed,
        )

    # ------------------------------------------------------------------ #
    # Subclass hooks
    # ------------------------------------------------------------------ #
    def _make_odor_objects(self) -> list[OdorObject]:
        """Return the OdorObjects for this task. Must be implemented."""
        raise NotImplementedError

    # ------------------------------------------------------------------ #
    # Model / references / observables
    # ------------------------------------------------------------------ #
    def _load_model(self):
        super()._load_model()

        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)

        mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )
        mujoco_arena.set_origin([0, 0, 0])

        self.odor_objects = self._make_odor_objects()
        if not self.odor_objects:
            raise ValueError("_make_odor_objects returned no objects")

        # Build the GADEN source map from the odor objects.
        self.scene_builder = SceneBuilder(frame_map=self.frame_map)
        self.scene_builder.add_objects(self.odor_objects)

        if self.placement_initializer is not None:
            self.placement_initializer.reset()
            self.placement_initializer.add_objects(self.odor_objects)
        else:
            self.placement_initializer = UniformRandomSampler(
                name="OdorObjectSampler",
                mujoco_objects=self.odor_objects,
                x_range=[-0.05, 0.05],
                y_range=[-0.05, 0.05],
                rotation=None,
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=self.table_offset,
                z_offset=0.01,
                rng=self.rng,
            )

        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=self.odor_objects,
        )

    def _setup_references(self):
        super()._setup_references()
        self._odor_body_ids = {
            obj.object_id: self.sim.model.body_name2id(obj.root_body) for obj in self.odor_objects
        }

    @property
    def object_to_sources(self) -> dict:
        """``object_id -> [gaden source indices]`` (feeds the Phase 4 bridge)."""
        return self.scene_builder.object_to_sources if self.scene_builder else {}

    # ------------------------------------------------------------------ #
    # Pose readouts (consumed by the Phase 4 bridge / recorder)
    # ------------------------------------------------------------------ #
    def _eef_arm(self) -> str:
        return self.robots[0].arms[0]

    def get_enose_site_pose(self):
        """World pose of the EE odor sensor site.

        Returns:
            (pos (3,), rot_mat (3, 3)) in the robosuite world frame, with the
            configurable ``enose_site_offset`` applied in the grip-site frame.
        """
        arm = self._eef_arm()
        site_id = self.robots[0].eef_site_id[arm]
        grip_pos = np.array(self.sim.data.site_xpos[site_id])
        grip_mat = np.array(self.sim.data.site_xmat[site_id]).reshape(3, 3)
        sensor_pos = grip_pos + grip_mat @ self.enose_site_offset
        return sensor_pos, grip_mat

    def get_object_world_poses(self) -> dict:
        """``object_id -> (pos (3,), rot_mat (3, 3))`` for all odor objects."""
        poses = {}
        for obj in self.odor_objects:
            bid = self._odor_body_ids[obj.object_id]
            pos = np.array(self.sim.data.body_xpos[bid])
            mat = np.array(self.sim.data.body_xmat[bid]).reshape(3, 3)
            poses[obj.object_id] = (pos, mat)
        return poses

    def get_gaden_source_poses(self) -> np.ndarray:
        """(num_sources, 3) GADEN-frame source positions for the current state."""
        return self.scene_builder.gaden_source_positions(self.get_object_world_poses())

    # ------------------------------------------------------------------ #
    # Observables
    # ------------------------------------------------------------------ #
    def _setup_observables(self):
        observables = super()._setup_observables()

        modality = "object"

        @sensor(modality="odor")
        def enose_site_pos(obs_cache):
            pos, _ = self.get_enose_site_pose()
            return pos

        sensors = [enose_site_pos]

        if self.use_object_obs:
            for obj in self.odor_objects:
                oid = obj.object_id

                def _pos(obs_cache, oid=oid):
                    return np.array(self.sim.data.body_xpos[self._odor_body_ids[oid]])

                def _quat(obs_cache, oid=oid):
                    return convert_quat(
                        np.array(self.sim.data.body_xquat[self._odor_body_ids[oid]]), to="xyzw"
                    )

                _pos.__name__ = f"{oid}_pos"
                _quat.__name__ = f"{oid}_quat"
                sensors.append(sensor(modality=modality)(_pos))
                sensors.append(sensor(modality=modality)(_quat))

        for s in sensors:
            observables[s.__name__] = Observable(
                name=s.__name__, sensor=s, sampling_rate=self.control_freq
            )

        return observables

    # ------------------------------------------------------------------ #
    # Instruction injection
    # ------------------------------------------------------------------ #
    def _inject_instruction(self, observations):
        if isinstance(observations, (dict, OrderedDict)):
            observations["instruction"] = self.instruction
        return observations

    def reset(self):
        obs = super().reset()
        return self._inject_instruction(obs)

    def step(self, action):
        obs, reward, done, info = super().step(action)
        return self._inject_instruction(obs), reward, done, info

    # ------------------------------------------------------------------ #
    # Reset placement
    # ------------------------------------------------------------------ #
    def _reset_internal(self):
        super()._reset_internal()
        if not self.deterministic_reset:
            object_placements = self.placement_initializer.sample()
            for obj_pos, obj_quat, obj in object_placements.values():
                self.sim.data.set_joint_qpos(
                    obj.joints[0], np.concatenate([np.array(obj_pos), np.array(obj_quat)])
                )

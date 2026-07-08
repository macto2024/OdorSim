"""Interactive teleoperation + data-mining app (Phase 4b).

Drives an :class:`~odor_sim.envs.base.OdorManipulationEnv` with a robosuite
device (keyboard / SpaceMouse) while co-stepping the GADEN plume through the
:class:`~odor_sim.bridge.gaden_bridge.GadenBridge`. Each control step records
the robosuite action, the ternary ``enose_state`` token, an auto-hold/sampling
mask, the ground-truth per-gas ppm at the EE, robot proprio, and the task
instruction. Episodes are written to disk (raw ppm(t) is stored; voltage is
synthesized offline in Phase 5).

Two entry points:

  * :meth:`TeleopSession.run_interactive` - human drives with a device + viewer.
  * :meth:`TeleopSession.run_scripted` - fixed action/enose schedule, headless;
    used by the Phase 4b smoke test (no display, no human).

E-nose keys (added on top of robosuite's keyboard controls):
    ``1`` sample (auto-hold ~7 s)   ``0`` idle    ``2`` filter/purge
Auto-hold: one ``sample`` press freezes the arm for the sample window (motion
ignored) so every sniff is a clean stationary dwell; those steps carry a
``sampling_active`` mask.

Since Phase 4.5 the GADEN server is spawned automatically via ``odor_sim.make``,
so a single command is enough (no separate server terminal)::

    python -m odor_sim.bridge.teleop --env OdorLift --recipe ripe_fruit --robots Panda \\
        --camera agentview --device keyboard --odor-monitor
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


def _validate_robot(robots: str) -> str:
    from robosuite.robots import ROBOT_CLASS_MAPPING

    if robots not in ROBOT_CLASS_MAPPING:
        available = ", ".join(sorted(ROBOT_CLASS_MAPPING.keys()))
        raise ValueError(f"Unknown robot {robots!r}. Available robosuite robots: {available}")
    return robots


class EpisodeRecorder:
    """Accumulates per-step records and writes one episode to disk.

    Stores raw ppm(t) (NOT voltage) + state/action/enose token/mask/instruction,
    matching the Phase 4b/5 "capture a superset, no voltage at mine time" rule.
    """

    def __init__(self, out_dir: "str | Path", instruction: str, gas_types: list, meta: dict):
        self.out_dir = Path(out_dir)
        self.instruction = instruction
        self.gas_types = list(gas_types)
        self.meta = dict(meta)
        self._rows: list[dict] = []

    def add(self, *, sim_time, state, action, enose_state, sampling_active, ppm):
        self._rows.append(
            {
                "sim_time": float(sim_time),
                "state": np.asarray(state, dtype=np.float32),
                "action": np.asarray(action, dtype=np.float32),
                "enose_state": int(enose_state),
                "sampling_active": bool(sampling_active),
                "ppm": np.asarray([float(ppm.get(g, 0.0)) for g in self.gas_types], dtype=np.float32),
            }
        )

    def __len__(self) -> int:
        return len(self._rows)

    def write(self, success: bool) -> "Path | None":
        if not self._rows:
            print("[teleop] empty episode; nothing written")
            return None
        self.out_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        ep_dir = self.out_dir / f"episode_{stamp}"
        ep_dir.mkdir(parents=True, exist_ok=True)

        np.savez(
            ep_dir / "episode.npz",
            sim_time=np.array([r["sim_time"] for r in self._rows], dtype=np.float32),
            state=np.stack([r["state"] for r in self._rows]),
            action=np.stack([r["action"] for r in self._rows]),
            enose_state=np.array([r["enose_state"] for r in self._rows], dtype=np.int8),
            sampling_active=np.array([r["sampling_active"] for r in self._rows], dtype=bool),
            ppm=np.stack([r["ppm"] for r in self._rows]),
        )
        meta = {
            "instruction": self.instruction,
            "gas_types": self.gas_types,
            "num_steps": len(self._rows),
            "success": bool(success),
            "note": "raw ppm(t) stored; voltage synthesized offline (Phase 4c/5)",
            **self.meta,
        }
        with open(ep_dir / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)
        print(f"[teleop] wrote {len(self._rows)} steps -> {ep_dir} (success={success})")
        return ep_dir


class TeleopSession:
    """Owns the co-sim (env + bridge + server) and the per-step record logic.

    The whole GADEN stack is brought up by :func:`odor_sim.make`, so there is no
    separate server terminal: constructing a ``TeleopSession`` with
    ``use_bridge=True`` exports the scene, spawns ``odor_gaden_rt`` in lockstep,
    and connects the bridge.

    Args:
        env: registered task name (e.g. ``"OdorLift"``).
        recipe: VOC recipe for the task object.
        scenario: logical GADEN scenario name (or a config-dir path).
        sample_hold_s: auto-hold duration on a ``sample`` press (seconds).
        use_bridge: co-step GADEN and record ppm. If False, run arm-only
            (no server, ppm recorded as zeros) - to sanity-check driving alone.
        out_dir: dataset output directory.
        robots: robosuite robot name (e.g. ``"Panda"``, ``"UR5e"``).
        controller: robosuite composite controller name or config path; ``None``
            uses the robot default.
        render_camera: on-screen viewer camera (e.g. ``"agentview"``).
        goal_update_mode: passed to the device ``input2action`` (``target`` or
            ``achieved``).
        pos_sensitivity: keyboard / SpaceMouse position scale.
        rot_sensitivity: keyboard / SpaceMouse rotation scale.
        success_hold_steps: consecutive successful control steps before auto-ending
            an interactive episode (``0`` disables; default ``10`` like collect).
    """

    def __init__(
        self,
        env: str = "OdorLift",
        recipe: str = "ripe_fruit",
        scenario: str = DEFAULT_SCENARIO,
        sample_hold_s: float = 7.0,
        use_bridge: bool = True,
        out_dir: str = "datasets/teleop",
        has_renderer: bool = False,
        control_freq: int = 20,
        odor_monitor=False,
        robots: str = "Panda",
        controller: str | None = None,
        render_camera: str = "agentview",
        goal_update_mode: str = "target",
        pos_sensitivity: float = 1.0,
        rot_sensitivity: float = 1.0,
        success_hold_steps: int = 10,
    ):
        import odor_sim as odorsim
        from robosuite.controllers import load_composite_controller_config

        robots = _validate_robot(robots)
        if goal_update_mode not in ("target", "achieved"):
            raise ValueError(f"goal_update_mode must be 'target' or 'achieved', got {goal_update_mode!r}")
        if success_hold_steps < 0:
            raise ValueError(f"success_hold_steps must be >= 0, got {success_hold_steps}")

        self.robots = robots
        self.goal_update_mode = goal_update_mode
        self.pos_sensitivity = pos_sensitivity
        self.rot_sensitivity = rot_sensitivity
        self.success_hold_steps = int(success_hold_steps)
        self.control_freq = control_freq
        self.sample_hold_steps = int(round(sample_hold_s * control_freq))
        self.out_dir = out_dir
        self._vis_wrapped = False

        controller_config = load_composite_controller_config(controller=controller, robot=robots)

        self.cosim = odorsim.make(
            env,
            recipe=recipe,
            scenario=scenario,
            auto_start_gaden=use_bridge,
            bridge=use_bridge,
            odor_monitor=odor_monitor if use_bridge else False,
            enose_site_offset=(0.0, 0.0, -0.02),
            robots=robots,
            controller_configs=controller_config,
            render_camera=render_camera,
            ignore_done=True,
            reward_shaping=True,
            has_renderer=has_renderer,
            has_offscreen_renderer=False,
            use_camera_obs=False,
            control_freq=control_freq,
            horizon=100000,
        )
        self.env = self.cosim.env
        self.bridge = self.cosim.bridge

        # De-duplicate gas names (the server returns one ppm per *unique* gas),
        # preserving source order.
        self.gas_types = list(
            dict.fromkeys(s.component.gaden_gas_name() for s in self.env.scene_builder.sources)
        )

        # auto-hold state
        self._hold_left = 0

    # ------------------------------------------------------------------ #
    def close(self):
        self.cosim.close()

    def _proprio_state(self, obs) -> np.ndarray:
        keys = [k for k in obs if k.startswith("robot0_") and isinstance(obs[k], np.ndarray)]
        if not keys:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate([np.asarray(obs[k], dtype=np.float32).ravel() for k in sorted(keys)])

    def _apply_hold(self, env_action: np.ndarray, enose_state: int, gripper_dof: int):
        """Auto-hold: freeze the arm for the sample window on a sample press.

        Returns (env_action, effective_enose_state, sampling_active).
        """
        if enose_state == SAMPLE and self._hold_left == 0:
            self._hold_left = self.sample_hold_steps

        if self._hold_left > 0:
            held = env_action.copy()
            if gripper_dof > 0:
                held[:-gripper_dof] = 0.0  # zero arm delta, keep gripper command
            else:
                held[:] = 0.0
            self._hold_left -= 1
            return held, SAMPLE, True
        return env_action, enose_state, False

    def step(self, recorder: EpisodeRecorder, env_action: np.ndarray, enose_state: int, gripper_dof: int):
        """One control step: auto-hold, publish+step GADEN, query ppm, record."""
        env_action, eff_enose, sampling = self._apply_hold(env_action, enose_state, gripper_dof)

        if self.bridge is not None:
            rec = self.bridge.step_env(self.env)  # publishes poses, ticks, queries EE ppm
            sim_time, ppm = rec["sim_time"], rec["ppm"]
            if self.cosim.odor_monitor is not None:
                self.cosim.odor_monitor.record(sim_time, ppm)
        else:
            sim_time, ppm = 0.0, {}

        obs, reward, done, info = self.env.step(env_action)
        state = self._proprio_state(obs)
        recorder.add(
            sim_time=sim_time,
            state=state,
            action=np.concatenate([env_action, [eff_enose]]),  # action + enose dim
            enose_state=eff_enose,
            sampling_active=sampling,
            ppm=ppm,
        )
        return obs, reward, done, info, eff_enose, sampling

    def _new_recorder(self, instruction: str) -> EpisodeRecorder:
        meta = {
            "recipe": getattr(self.env, "_recipe_name", "?"),
            "robots": self.robots,
            "control_freq": self.control_freq,
            "sample_hold_steps": self.sample_hold_steps,
            "success_hold_steps": self.success_hold_steps,
            "action_layout": "[robosuite_action..., enose_state]",
        }
        return EpisodeRecorder(self.out_dir, instruction, self.gas_types, meta)

    def _update_success_hold(self, hold_count: int) -> tuple[int, bool]:
        """Advance collect-style sustained-success counter.

        Returns:
            (new_count, should_break_episode)
        """
        if self.success_hold_steps <= 0:
            return hold_count, False

        if hold_count == 0:
            return hold_count, True

        if self.env._check_success():
            if hold_count > 0:
                return hold_count - 1, False
            return self.success_hold_steps, False

        return -1, False

    # ------------------------------------------------------------------ #
    # Headless scripted episode (for smoke testing without a human/device)
    # ------------------------------------------------------------------ #
    def run_scripted(self, arm_actions, enose_schedule, instruction: str = "", success: bool = True):
        """Run a fixed schedule with no device; write the episode.

        Args:
            arm_actions: (T, action_dim) robosuite actions.
            enose_schedule: length-T ternary enose tokens.
            instruction: episode language label (defaults to env instruction).
        """
        instruction = instruction or self.env.instruction
        recorder = self._new_recorder(instruction)
        self.env.reset()
        if self.cosim.odor_monitor is not None:
            self.cosim.odor_monitor.reset()
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

    # ------------------------------------------------------------------ #
    # Interactive episode (device + on-screen viewer)
    # ------------------------------------------------------------------ #
    def run_interactive(self, device_name: str = "keyboard", max_fr: int = 20):
        from copy import deepcopy

        from robosuite.controllers.composite.composite_controller import WholeBody
        from robosuite.wrappers import VisualizationWrapper

        if not self._vis_wrapped:
            self.env = VisualizationWrapper(self.env)
            self._vis_wrapped = True

        device, enose_keys = self._make_device(device_name)
        gripper_dof = self._gripper_dof()

        print(self._controls_help())
        print(f"[teleop] robot={self.robots}")
        while True:
            instruction = self.env.instruction
            recorder = self._new_recorder(instruction)
            self.env.reset()
            if self.cosim.odor_monitor is not None:
                self.cosim.odor_monitor.reset()
            self.env.render()
            device.start_control()
            self._hold_left = 0
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

            while True:
                start = time.time()
                input_ac = device.input2action(goal_update_mode=self.goal_update_mode)
                if input_ac is None:  # device reset -> end episode
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

                enose_state = enose_keys.consume()
                _, _, _, _, eff, samp = self.step(recorder, env_action, enose_state, gripper_dof)
                self.env.render()
                self._hud(
                    instruction,
                    eff,
                    samp,
                    len(recorder),
                    success_hold_count=task_completion_hold_count,
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
                    "could not load the SpaceMouse driver. robosuite officially supports "
                    "SpaceMouse on macOS only. On Linux install `pip install hidapi`, system "
                    "libhidapi, and udev rules, and connect a 3Dconnexion device. "
                    "Use --device keyboard instead."
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
        return (
            "\n=== OdorSim teleop ===\n"
            "  arm/gripper: standard robosuite keyboard controls\n"
            "  e-nose:  1 = sample (auto-hold ~7 s)   0 = idle   2 = filter/purge\n"
            "  end episode: the device reset key (Ctrl+Q on keyboard)\n"
            "  sustained task success auto-ends the episode (collect-style)\n"
        )

    @staticmethod
    def _hud(instruction, enose_state, sampling, step_i, success_hold_count=-1):
        token = {SAMPLE: "SAMPLE", IDLE: "idle", FILTER: "FILTER"}.get(enose_state, "?")
        tag = " [SNIFF]" if sampling else ""
        success_tag = ""
        if success_hold_count > 0:
            success_tag = f"  [SUCCESS {success_hold_count}]"
        print(f"\r[{step_i:5d}] enose={token}{tag}{success_tag}  | {instruction}", end="", flush=True)

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
    parser.add_argument("--recipe", default="ripe_fruit")
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
        "--success-hold-steps",
        type=int,
        default=10,
        help="consecutive successful steps before auto-ending episode (0=disable)",
    )
    parser.add_argument("--out-dir", default="datasets/teleop")
    parser.add_argument("--no-bridge", action="store_true", help="drive arm only, no GADEN/ppm")
    parser.add_argument(
        "--odor-monitor",
        nargs="?",
        const=True,
        default=False,
        metavar="MODE",
        help="live ppm log/plot: default log+plot; MODE=log or plot for one only",
    )
    args = parser.parse_args(argv)

    odor_monitor = False
    if args.odor_monitor is not False:
        odor_monitor = args.odor_monitor if args.odor_monitor is not True else True

    session = TeleopSession(
        env=args.env,
        recipe=args.recipe,
        scenario=args.scenario,
        sample_hold_s=args.sample_hold_s,
        use_bridge=not args.no_bridge,
        out_dir=args.out_dir,
        has_renderer=True,
        odor_monitor=odor_monitor,
        robots=args.robots,
        controller=args.controller,
        render_camera=args.camera,
        goal_update_mode=args.goal_update_mode,
        pos_sensitivity=args.pos_sensitivity,
        rot_sensitivity=args.rot_sensitivity,
        success_hold_steps=args.success_hold_steps,
    )
    try:
        session.run_interactive(device_name=args.device)
    finally:
        session.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

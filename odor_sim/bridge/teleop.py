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

Run (needs odor_gaden_rt in lockstep + a matching scene; see README)::

    python -m odor_sim.bridge.teleop --recipe ripe_fruit --device keyboard
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from odor_sim.bridge.gaden_bridge import GadenBridge

# enose tokens
SAMPLE, IDLE, FILTER = 1, 0, -1
CONFIG_DIR = "scenarios/10x6_uniform/environment_configurations/config1"


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
    """Owns the env + bridge and the shared per-step control/record logic.

    Args:
        recipe: VOC recipe for the OdorLift cube.
        config_dir: GADEN scenario config directory (for the frame map).
        sample_hold_s: auto-hold duration on a ``sample`` press (seconds).
        use_bridge: co-step GADEN and record ppm. If False, run arm-only
            (ppm recorded as zeros) - useful to sanity-check driving alone.
        out_dir: dataset output directory.
    """

    def __init__(
        self,
        recipe: str = "ripe_fruit",
        config_dir: str = CONFIG_DIR,
        sample_hold_s: float = 7.0,
        use_bridge: bool = True,
        out_dir: str = "datasets/teleop",
        has_renderer: bool = False,
        control_freq: int = 20,
    ):
        from odor_sim.envs.odor_lift import OdorLift

        self.control_freq = control_freq
        self.sample_hold_steps = int(round(sample_hold_s * control_freq))
        self.out_dir = out_dir

        self.env = OdorLift(
            robots="Panda",
            recipe=recipe,
            scenario_config_dir=config_dir,
            enose_site_offset=(0.0, 0.0, -0.02),
            has_renderer=has_renderer,
            has_offscreen_renderer=False,
            use_camera_obs=False,
            control_freq=control_freq,
            horizon=100000,
        )
        # De-duplicate gas names (the server returns one ppm per *unique* gas),
        # preserving source order.
        self.gas_types = list(
            dict.fromkeys(s.component.gaden_gas_name() for s in self.env.scene_builder.sources)
        )

        self.bridge = None
        if use_bridge:
            self.bridge = GadenBridge(step_timeout=2.0)
            if not self.bridge.wait_for_server(timeout=10.0):
                raise RuntimeError(
                    "odor_gaden_rt /odor_value not available. Start it in lockstep "
                    "mode with a scene matching this env (see export_scene)."
                )

        # auto-hold state
        self._hold_left = 0

    # ------------------------------------------------------------------ #
    def close(self):
        if self.bridge is not None:
            self.bridge.close()
        self.env.close()

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
            "control_freq": self.control_freq,
            "sample_hold_steps": self.sample_hold_steps,
            "action_layout": "[robosuite_action..., enose_state]",
        }
        return EpisodeRecorder(self.out_dir, instruction, self.gas_types, meta)

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

        device, enose_keys = self._make_device(device_name)
        gripper_dof = self._gripper_dof()

        print(self._controls_help())
        while True:
            instruction = self.env.instruction
            recorder = self._new_recorder(instruction)
            self.env.reset()
            self.env.render()
            device.start_control()
            self._hold_left = 0
            enose_keys.reset()

            active_robot = self.env.robots[0]
            prev_gripper = {
                f"{arm}_gripper": np.repeat([0], active_robot.gripper[arm].dof)
                for arm in active_robot.arms
                if active_robot.gripper[arm].dof > 0
            }

            while True:
                start = time.time()
                input_ac = device.input2action()
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
                self._hud(instruction, eff, samp, len(recorder))

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

            device = Keyboard(env=self.env, pos_sensitivity=1.0, rot_sensitivity=1.0)
            enose_keys.attach_to_keyboard(device)
        elif device_name == "spacemouse":
            from robosuite.devices import SpaceMouse

            device = SpaceMouse(env=self.env, pos_sensitivity=1.0, rot_sensitivity=1.0)
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
        )

    @staticmethod
    def _hud(instruction, enose_state, sampling, step_i):
        token = {SAMPLE: "SAMPLE", IDLE: "idle", FILTER: "FILTER"}.get(enose_state, "?")
        tag = " [SNIFF]" if sampling else ""
        print(f"\r[{step_i:5d}] enose={token}{tag}  | {instruction}", end="", flush=True)

    @staticmethod
    def _ask_continue() -> bool:
        try:
            ans = input("\nAnother episode? [Y/n] ").strip().lower()
        except EOFError:
            return False
        return ans in ("", "y", "yes")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", default="ripe_fruit")
    parser.add_argument("--config-dir", default=CONFIG_DIR)
    parser.add_argument("--device", default="keyboard", choices=["keyboard", "spacemouse"])
    parser.add_argument("--sample-hold-s", type=float, default=7.0)
    parser.add_argument("--out-dir", default="datasets/teleop")
    parser.add_argument("--no-bridge", action="store_true", help="drive arm only, no GADEN/ppm")
    args = parser.parse_args(argv)

    session = TeleopSession(
        recipe=args.recipe,
        config_dir=args.config_dir,
        sample_hold_s=args.sample_hold_s,
        use_bridge=not args.no_bridge,
        out_dir=args.out_dir,
        has_renderer=True,
    )
    try:
        session.run_interactive(device_name=args.device)
    finally:
        session.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

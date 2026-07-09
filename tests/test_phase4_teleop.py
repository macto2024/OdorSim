"""Phase 4b test: teleop recording pipeline (headless, scripted; no human).

Needs odor_gaden_rt running in lockstep with a scene matching the env (same
setup as test_phase4_bridge.py). Runs a fixed action + enose schedule through
TeleopSession.run_scripted and validates the written episode:

    # Terminal 1  (server, once)
    source setup/activate.sh
    python -m odor_sim.bridge.export_scene --scene-id rt_scene --recipe ripe_fruit
    ros2 run odor_gaden_rt rt_server --ros-args \
        -p scenarioPath:=$PWD/scenarios/10x6_uniform/environment_configurations/config1 \
        -p sceneID:=rt_scene

    # Terminal 2
    source setup/activate.sh
    python tests/test_phase4_teleop.py

PASS: episode.npz + meta.json written; auto-hold freezes the arm and sets the
sampling mask for the sample window; action carries the appended enose dim;
raw ppm(t) recorded for the scene gases.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")


def main() -> int:
    from odor_sim.bridge.teleop import SAMPLE, TeleopSession

    hold_s = 0.5  # 10 steps at 20 Hz -> keep the test short
    out_dir = tempfile.mkdtemp(prefix="teleop_test_")
    session = TeleopSession(
        recipe="ripe_fruit",
        sample_hold_s=hold_s,
        use_bridge=True,
        out_dir=out_dir,
        has_renderer=False,
        control_freq=20,
    )

    failures = 0

    def check(name, ok, detail=""):
        nonlocal failures
        if ok:
            print(f"PASS  {name}  {detail}")
        else:
            failures += 1
            print(f"FAIL  {name}  {detail}")

    try:
        adim = session.env.action_dim
        hold_steps = session.sample_hold_steps
        T = 40
        sample_at = 5

        # nonzero arm command on dim 0 so we can see the auto-hold zero it out;
        # trigger a single sample at step `sample_at`.
        arm_actions = [np.zeros(adim) for _ in range(T)]
        for a in arm_actions:
            a[0] = 0.3

        enose = [0] * T
        enose[sample_at] = SAMPLE

        ep_dir = session.run_scripted(arm_actions, enose, instruction="pick up the ripe fruit object")
        check("episode_written", ep_dir is not None and (ep_dir / "episode.npz").exists(), str(ep_dir))

        data = np.load(ep_dir / "episode.npz")
        meta = json.loads((ep_dir / "meta.json").read_text())

        check("num_steps", data["action"].shape[0] == T, f"{data['action'].shape[0]} == {T}")
        check("action_has_enose_dim", data["action"].shape[1] == adim + 1, f"{data['action'].shape[1]} == {adim + 1}")

        # enose token appended as the last action dim; sample window latched to 1
        enose_col = data["action"][:, -1]
        window = slice(sample_at, sample_at + hold_steps)
        check("sample_window_enose_is_1", np.all(enose_col[window] == SAMPLE), f"{enose_col.tolist()}")
        check(
            "sampling_mask_matches_window",
            int(data["sampling_active"].sum()) == hold_steps,
            f"active={int(data['sampling_active'].sum())} expected={hold_steps}",
        )

        # auto-hold zeroed the arm command (first dim) during the window
        arm_dim0 = data["action"][window, 0]
        check("auto_hold_freezes_arm", np.allclose(arm_dim0, 0.0), f"arm[0] in window = {arm_dim0[:3]}...")
        # before the sample it was the commanded 0.3
        check("arm_command_active_pre_sample", abs(data["action"][0, 0] - 0.3) < 1e-5, f"{data['action'][0,0]}")

        # raw ppm(t) recorded for the two scene gases, and shows nonzero somewhere
        check("ppm_columns", data["ppm"].shape[1] == len(session.gas_types), f"gas_types={session.gas_types}")
        check("ppm_recorded_nonzero", float(data["ppm"].max()) > 0.0, f"max ppm = {float(data['ppm'].max()):.3f}")

        # sim_time must be monotonic (latched per step, no stale queued messages)
        # and episode-relative (GADEN reset on episode start -> starts near 0).
        st = data["sim_time"]
        back = int(np.sum(np.diff(st) < 0))
        check("sim_time_monotonic", bool(np.all(np.diff(st) >= 0)), f"backward jumps={back}")
        check("sim_time_starts_near_zero", float(st[0]) <= 0.2, f"first sim_time={float(st[0]):.3f}")

        check("meta_instruction", bool(meta.get("instruction")), meta.get("instruction", ""))
        check("meta_gas_types", meta.get("gas_types") == session.gas_types, str(meta.get("gas_types")))

    finally:
        session.close()

    total = 13
    print(f"\n{total - failures}/{total} checks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

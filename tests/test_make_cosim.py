"""Phase 4.5c test: odor_sim.make() full co-simulation, no manual terminal.

Unlike test_phase4_bridge.py (which needs the server started by hand), this
brings up the whole stack from a single ``make()`` call: it exports the scene
from the env, spawns ``odor_gaden_rt`` as a managed subprocess, connects the
bridge, and steps the co-sim in lockstep. Run it from an activated shell:

    source setup/activate.sh
    python tests/test_make_cosim.py       # 7/7 checks
    pytest tests/test_make_cosim.py

PASS criteria:
  * make() returns a session with a live server + connected bridge;
  * co-stepping advances GADEN sim_time in lockstep;
  * with sources parked at the EE the plume builds and reports the scene gases;
  * step() info carries a well-formed ppm/source record;
  * close() tears the server down cleanly;
  * odor_monitor='log' prints per-VOC lines (4.5f).
"""

from __future__ import annotations

import os
import sys
from io import StringIO

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np

import odor_sim as odorsim

SCENARIO = "10x6_uniform"


def _make_cosim(odor_monitor=False):
    return odorsim.make(
        "OdorLift",
        recipe="ripe_fruit",
        scenario=SCENARIO,
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        control_freq=20,
        horizon=100000,
        enose_site_offset=(0.0, 0.0, -0.02),
        wait_timeout=60.0,
        odor_monitor=odor_monitor,
    )


def test_odor_monitor_log_during_cosim(capsys):
    """4.5f: odor_monitor='log' prints gas names without plot window."""
    cosim = _make_cosim(odor_monitor="log")
    try:
        assert cosim.odor_monitor is not None
        cosim.reset()
        for _ in range(5):
            cosim.step(cosim.zero_action())
    finally:
        cosim.close()
    out = capsys.readouterr().out
    assert "ethanol" in out
    assert "acetone" in out
    assert "[odor]" in out


def main() -> int:
    failures = 0

    def check(name, ok, detail=""):
        nonlocal failures
        if ok:
            print(f"PASS  {name}  {detail}")
        else:
            failures += 1
            print(f"FAIL  {name}  {detail}")

    cosim = _make_cosim()

    try:
        check(
            "make_brings_up_server_and_bridge",
            cosim.server is not None and cosim.server.alive and cosim.has_gaden,
            f"scene_id={cosim.scene_id}",
        )

        obs = cosim.reset()
        check("reset_returns_instruction", bool(obs.get("instruction")), obs.get("instruction", ""))

        n = cosim.env.scene_builder.num_sources
        # 1) lockstep: co-stepping advances GADEN sim_time.
        _, _, _, info0 = cosim.step(cosim.zero_action())
        t0 = info0["sim_time"]
        for _ in range(15):
            _, _, _, info = cosim.step(cosim.zero_action())
        check("lockstep_sim_time_advances", info["sim_time"] > t0, f"{t0:.3f} -> {info['sim_time']:.3f}")

        # 2) step() info is well-formed.
        ok = (
            isinstance(info["ppm"], dict)
            and np.asarray(info["gaden_source_poses"]).shape == (n, 3)
            and isinstance(info["sim_time"], float)
        )
        check("step_info_shape", ok, f"keys={sorted(info.keys())}")

        # 3) Park the sources at the EE and let the plume build; ppm should rise
        #    and report the scene gases (ethanol + acetone). Drive the bridge
        #    directly (same trick as the Phase 4 bridge test).
        ee_world, _ = cosim.env.get_enose_site_pose()
        ee_gaden = cosim.env.frame_map.robosuite_to_gaden(ee_world)
        at_ee = np.tile(ee_gaden, (n, 1))
        totals = []
        ppm = {}
        for _ in range(200):
            cosim.bridge.publish_source_poses(at_ee)
            cosim.bridge.step(1)
            ppm = cosim.bridge.query_ppm(ee_gaden)[0]
            totals.append(sum(ppm.values()))
        check("ppm_rises_when_source_at_ee", totals[-1] > 1.0, f"total ppm = {totals[-1]:.2f}")
        check("scene_gases_reported", set(ppm) >= {"ethanol", "acetone"}, f"gases={sorted(ppm)}")
    finally:
        cosim.close()

    server = cosim.server
    check("clean_shutdown", server is not None and not server.alive, "server stopped")

    # 4.5f log-mode smoke (headless)
    from contextlib import redirect_stdout

    log_cosim = _make_cosim(odor_monitor="log")
    try:
        buf = StringIO()
        with redirect_stdout(buf):
            log_cosim.reset()
            for _ in range(5):
                log_cosim.step(log_cosim.zero_action())
        log_out = buf.getvalue()
        check(
            "odor_monitor_log",
            log_cosim.odor_monitor is not None and "ethanol" in log_out and "[odor]" in log_out,
            "log lines present",
        )
    finally:
        log_cosim.close()

    total = 7
    print(f"\n{total - failures}/{total} checks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

"""Phase 5b test: offline MOX feature synthesis (no ROS / no GADEN).

Builds a synthetic raw episode fixture (fabricated ppm(t) + enose_state), then
runs :func:`odor_sim.recording.synthesize.synthesize_episode` and validates the
written ``features.npz`` / ``features_meta.json``:

  * continuous + sampling voltage arrays, length T,
  * continuous voltage departs from the clean-air baseline while ppm is nonzero,
  * at least one completed sample window carries a ground-truth ``odor_class``,
  * re-running with different MOX params changes the voltage without re-mining.

    source setup/activate.sh
    python tests/test_phase5_synthesize.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np


def _make_fixture(ep_dir: Path) -> None:
    """Write a synthetic episode.npz + meta.json (ethanol plume + one sniff)."""
    T = 30
    gas_types = ["ethanol", "acetone"]
    ppm = np.zeros((T, 2), dtype=np.float32)
    ppm[10:20, 0] = 50.0  # ethanol plume steps 10..19
    ppm[10:20, 1] = 3.0   # a little acetone

    enose = np.zeros(T, dtype=np.int8)
    enose[12:18] = 1  # sample window inside the plume

    ep_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        ep_dir / "episode.npz",
        sim_time=np.arange(T, dtype=np.float32) * 0.05,
        action=np.zeros((T, 8), dtype=np.float32),
        enose_state=enose,
        sampling_active=(enose == 1),
        ppm=ppm,
        state=np.zeros((T, 4), dtype=np.float32),
    )
    meta = {
        "instruction": "pick up the ripe fruit object",
        "gas_types": gas_types,
        "control_freq": 20,
        "num_steps": T,
    }
    (ep_dir / "meta.json").write_text(json.dumps(meta, indent=2))


def main() -> int:
    from odor_sim.recording.synthesize import synthesize_episode

    failures = 0

    def check(name, ok, detail=""):
        nonlocal failures
        if ok:
            print(f"PASS  {name}  {detail}")
        else:
            failures += 1
            print(f"FAIL  {name}  {detail}")

    root = Path(tempfile.mkdtemp(prefix="synth_test_"))
    ep_dir = root / "episode_fixture"
    _make_fixture(ep_dir)

    feats_path = synthesize_episode(ep_dir, mox_model="TGS2620")
    check("features_written", feats_path.exists() and (ep_dir / "features_meta.json").exists(), str(feats_path))

    feats = np.load(feats_path, allow_pickle=True)
    fmeta = json.loads((ep_dir / "features_meta.json").read_text())

    cont = feats["enose_voltage_continuous"]
    samp = feats["enose_voltage_sampling"]
    check("continuous_len", cont.shape == (30,), f"{cont.shape}")
    check("sampling_len", samp.shape == (30,), f"{samp.shape}")

    baseline = fmeta["baseline_voltage"]
    # continuous sensor sees the plume every step -> voltage moves off baseline.
    check(
        "continuous_responds_to_plume",
        float(np.max(np.abs(cont - baseline))) > 1e-3,
        f"max|dV|={float(np.max(np.abs(cont - baseline))):.4f} baseline={baseline:.4f}",
    )

    windows = list(feats["sample_windows"])
    check("one_sample_window", len(windows) == 1, f"n={len(windows)}")
    if windows:
        w = windows[0]
        check("window_odor_class_ethanol", w["odor_class"] == "ethanol", str(w["odor_class"]))
        check("window_duration_steps", w["duration_steps"] == 6, f"{w['duration_steps']}")
        check("window_voltage_trace", np.asarray(w["voltage_trace"]).shape[0] == 6, str(np.asarray(w["voltage_trace"]).shape))

    check(
        "meta_window_odor_class",
        fmeta["num_sample_windows"] == 1 and fmeta["sample_windows"][0]["odor_class"] == "ethanol",
        str(fmeta.get("sample_windows")),
    )

    # Re-runnable: a different MOX model yields a different voltage stream.
    synthesize_episode(ep_dir, mox_model="TGS2600")
    cont2 = np.load(ep_dir / "features.npz", allow_pickle=True)["enose_voltage_continuous"]
    check(
        "rerun_changes_voltage",
        not np.allclose(cont, cont2),
        f"max|delta|={float(np.max(np.abs(cont - cont2))):.4f}",
    )

    total = 9
    print(f"\n{total - failures}/{total} checks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

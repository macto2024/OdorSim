"""Phase 5b: offline MOX/PID feature synthesis over raw episode dirs.

Raw episodes store ground-truth ``ppm(t)`` at the EE (not voltage), per the
Phase 5 "no voltage at mine time" rule (docs/SCHEMA.md). This module replays a
recorded ppm(t) series through the shared sensor model
(:mod:`odor_sim.sensors.mox_pid`) to derive:

  * ``enose_voltage_continuous`` - always-exposed MOX voltage stream,
  * ``enose_voltage_sampling``   - valve-gated stream (sees the plume only while
    ``enose_state == 1``),
  * ``sample_windows``           - completed sampling windows with voltage/ppm
    traces and a ground-truth ``odor_class`` label.

Results are written next to the episode as ``features.npz`` + ``features_meta.json``.
Because synthesis is a pure function of the stored ``ppm`` + ``enose_state``, it
is fully re-runnable: change the MOX model / load resistor / Vcc and re-run to
get different voltages without re-mining.

Usage::

    source setup/activate.sh
    python -m odor_sim.recording.synthesize datasets/teleop
    python -m odor_sim.recording.synthesize datasets/teleop/episode_20260708_170047 \\
        --mox-model TGS2600 --vcc 5.0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from odor_sim.config.gas_types import GADEN_GAS_NAMES
from odor_sim.sensors.mox_pid import (
    MOX_MODELS,
    MOXSensor,
    synthesize_continuous,
    synthesize_sampling,
)


def _ppm_series(episode: "np.lib.npyio.NpzFile", gas_types: list) -> list[dict]:
    """Rebuild the per-step ``{gas_name: ppm}`` series from ``episode.npz``."""
    ppm = np.asarray(episode["ppm"], dtype=float)
    if ppm.ndim != 2 or ppm.shape[1] != len(gas_types):
        raise ValueError(
            f"ppm array shape {ppm.shape} does not match gas_types {gas_types}"
        )
    return [{gas_types[j]: float(ppm[t, j]) for j in range(len(gas_types))} for t in range(ppm.shape[0])]


def _odor_class_name(gt) -> "str | None":
    if gt is None:
        return None
    return gt.name


def _window_record(window, rate: float) -> dict:
    """Serializable dict for one completed :class:`SampleWindow`."""
    return {
        "trigger_step": int(window.trigger_step),
        "duration_steps": int(window.duration_steps),
        "duration_s": float(window.duration_steps / rate),
        "odor_class": _odor_class_name(window.odor_class),
        "odor_class_gaden": GADEN_GAS_NAMES.get(window.odor_class) if window.odor_class is not None else None,
        "voltage_trace": np.asarray(window.voltage_trace, dtype=np.float32),
        "ppm_trace": [dict(frame) for frame in window.ppm_trace],
    }


def synthesize_episode(
    ep_dir: "str | Path",
    *,
    mox_model: str = "TGS2620",
    load_resistance: "float | None" = None,
    vcc: float = 5.0,
    rate: "float | None" = None,
    overwrite: bool = True,
) -> Path:
    """Synthesize + write ``features.npz`` / ``features_meta.json`` for one episode.

    Args:
        ep_dir: raw episode dir (must contain ``episode.npz`` + ``meta.json``).
        mox_model: MOX model id (see :data:`odor_sim.sensors.mox_pid.MOX_MODELS`).
        load_resistance: voltage-divider RL [ohms]; defaults to the model's R0.
        vcc: divider supply voltage (V).
        rate: control frequency (Hz) for the sensor transient; defaults to the
            episode's ``control_freq`` meta (or 20).
        overwrite: if False and ``features.npz`` exists, skip (returns its path).

    Returns:
        Path to the written ``features.npz``.
    """
    ep_dir = Path(ep_dir)
    npz_path = ep_dir / "episode.npz"
    meta_path = ep_dir / "meta.json"
    if not npz_path.is_file():
        raise FileNotFoundError(f"no episode.npz in {ep_dir}")

    features_path = ep_dir / "features.npz"
    if features_path.is_file() and not overwrite:
        print(f"[synth] {features_path} exists; skipping (--no-overwrite)")
        return features_path

    meta = json.loads(meta_path.read_text()) if meta_path.is_file() else {}
    gas_types = list(meta.get("gas_types", []))
    if rate is None:
        rate = float(meta.get("control_freq", 20))

    episode = np.load(npz_path)
    if not gas_types:
        raise ValueError(f"{meta_path} has no gas_types; cannot map ppm columns")
    ppm_series = _ppm_series(episode, gas_types)
    enose_states = [int(x) for x in np.asarray(episode["enose_state"]).ravel()]

    def _make_sensor() -> MOXSensor:
        return MOXSensor(model=mox_model, rate=rate, load_resistance=load_resistance, vcc=vcc)

    continuous = synthesize_continuous(ppm_series, sensor=_make_sensor())
    sampling, windows = synthesize_sampling(ppm_series, enose_states, sensor=_make_sensor())

    window_records = [_window_record(w, rate) for w in windows]

    np.savez(
        features_path,
        enose_voltage_continuous=np.asarray(continuous, dtype=np.float32),
        enose_voltage_sampling=np.asarray(sampling, dtype=np.float32),
        sample_windows=np.array(window_records, dtype=object),
    )

    baseline = _make_sensor().baseline_voltage()
    features_meta = {
        "sensor_model": mox_model,
        "sensor_type": "MOX",
        "R0": _make_sensor().R0,
        "RL": _make_sensor().load_resistance,
        "Vcc": vcc,
        "rate": rate,
        "baseline_voltage": baseline,
        "num_steps": int(np.asarray(episode["ppm"]).shape[0]),
        "gas_types": gas_types,
        "num_sample_windows": len(window_records),
        "sample_windows": [
            {
                "trigger_step": w["trigger_step"],
                "duration_steps": w["duration_steps"],
                "duration_s": w["duration_s"],
                "odor_class": w["odor_class"],
                "odor_class_gaden": w["odor_class_gaden"],
            }
            for w in window_records
        ],
        "note": "offline synthesis from episode.npz ppm(t)+enose_state (Phase 5b)",
    }
    with open(ep_dir / "features_meta.json", "w") as f:
        json.dump(features_meta, f, indent=2)

    print(
        f"[synth] {ep_dir.name}: {features_meta['num_steps']} steps, "
        f"{len(window_records)} sample window(s), model={mox_model} -> {features_path.name}"
    )
    return features_path


def _find_episode_dirs(root: Path) -> list[Path]:
    """Resolve a path to a list of episode dirs (single or parent-of-many)."""
    if (root / "episode.npz").is_file():
        return [root]
    return sorted(p.parent for p in root.glob("episode_*/episode.npz"))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "paths",
        nargs="+",
        help="episode dir(s) or parent dir(s) containing episode_*/ subdirs",
    )
    parser.add_argument(
        "--mox-model",
        default="TGS2620",
        choices=list(MOX_MODELS),
        help="MOX sensor model (default TGS2620)",
    )
    parser.add_argument(
        "--load-resistance",
        type=float,
        default=None,
        help="voltage-divider RL [ohms] (default: model R0)",
    )
    parser.add_argument("--vcc", type=float, default=5.0, help="divider supply voltage (V)")
    parser.add_argument(
        "--rate",
        type=float,
        default=None,
        help="control frequency (Hz) for the transient (default: episode control_freq)",
    )
    parser.add_argument(
        "--no-overwrite",
        action="store_true",
        help="skip episodes that already have features.npz",
    )
    args = parser.parse_args(argv)

    episodes: list[Path] = []
    for p in args.paths:
        episodes.extend(_find_episode_dirs(Path(p)))
    if not episodes:
        print("[synth] no episodes found")
        return 1

    for ep in episodes:
        synthesize_episode(
            ep,
            mox_model=args.mox_model,
            load_resistance=args.load_resistance,
            vcc=args.vcc,
            rate=args.rate,
            overwrite=not args.no_overwrite,
        )
    print(f"[synth] done: {len(episodes)} episode(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

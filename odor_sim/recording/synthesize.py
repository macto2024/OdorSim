"""Phase 5b: offline MOX/PID feature synthesis over raw episode dirs.

Raw episodes store ground-truth ``ppm(t)`` at the EE (not voltage), per the
Phase 5 "no voltage at mine time" rule. This module replays a
recorded ppm(t) series through the shared sensor model
(:mod:`odor_sim.sensors.mox_pid`) to derive:

  * ``enose_voltage_continuous`` - always-exposed MOX voltage stream,
  * ``enose_voltage_sampling``   - valve-gated stream (sees the plume only while
    ``enose_state == 1``),
  * ``sample_windows``           - completed sampling windows with voltage/ppm
    traces and a ground-truth ``odor_class`` label.
  * ``source_identity`` / ``source_distance`` - carried from ``episode.npz``
    when present (aligned 1:1 with the voltage streams).

Results are written next to the episode as ``features.npz`` + ``features_meta.json``.
Because synthesis is a pure function of the stored ``ppm`` + ``enose_state``, it
is fully re-runnable: change the MOX model / load resistor / Vcc and re-run to
get different voltages without re-mining.

By default (no ``--mox-model``) every model in
:data:`odor_sim.sensors.mox_pid.MOX_MODELS` is synthesized, writing one
``features_<MODEL>.npz`` / ``features_meta_<MODEL>.json`` pair per model plus a
canonical ``features.npz`` / ``features_meta.json`` for the primary model
(``TGS2620``) that downstream conversion consumes. Passing ``--mox-model X``
synthesizes only model ``X`` into the canonical ``features.npz``.

Usage::

    source setup/activate.sh
    python -m odor_sim.recording.synthesize datasets/teleop          # all models
    python -m odor_sim.recording.synthesize datasets/teleop/episode_20260708_170047 \\
        --mox-model TGS2600 --vcc 5.0                                # single model
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


#: Canonical (model-agnostic) output basename consumed by :mod:`convert`.
CANONICAL_NAME = "features"
#: Primary model mirrored into :data:`CANONICAL_NAME` when synthesizing many.
PRIMARY_MOX_MODEL = MOX_MODELS[0]


def _write_features(
    ep_dir: Path,
    basename: str,
    *,
    continuous,
    sampling,
    window_records: list[dict],
    features_meta: dict,
    source_identity: "np.ndarray | None" = None,
    source_distance: "np.ndarray | None" = None,
) -> Path:
    """Write one ``<basename>.npz`` + ``<basename>_meta.json`` pair."""
    features_path = ep_dir / f"{basename}.npz"
    arrays = {
        "enose_voltage_continuous": np.asarray(continuous, dtype=np.float32),
        "enose_voltage_sampling": np.asarray(sampling, dtype=np.float32),
        "sample_windows": np.array(window_records, dtype=object),
    }
    if source_identity is not None:
        arrays["source_identity"] = np.asarray(source_identity, dtype=np.int64).ravel()
    if source_distance is not None:
        arrays["source_distance"] = np.asarray(source_distance, dtype=np.float32).ravel()
    np.savez(features_path, **arrays)
    with open(ep_dir / f"{basename}_meta.json", "w") as f:
        json.dump(features_meta, f, indent=2)
    return features_path


def _check_label_lengths(
    *,
    num_steps: int,
    continuous,
    sampling,
    source_identity: "np.ndarray | None",
    source_distance: "np.ndarray | None",
) -> None:
    """Raise if voltage streams and optional labels disagree on length ``T``."""
    checks = {
        "enose_voltage_continuous": len(continuous),
        "enose_voltage_sampling": len(sampling),
    }
    if source_identity is not None:
        checks["source_identity"] = int(np.asarray(source_identity).shape[0])
    if source_distance is not None:
        checks["source_distance"] = int(np.asarray(source_distance).shape[0])
    mismatches = {name: n for name, n in checks.items() if n != num_steps}
    if mismatches:
        raise ValueError(
            f"label/voltage length mismatch vs ppm T={num_steps}: {mismatches}"
        )


def synthesize_episode(
    ep_dir: "str | Path",
    *,
    mox_model: str = PRIMARY_MOX_MODEL,
    load_resistance: "float | None" = None,
    vcc: float = 5.0,
    rate: "float | None" = None,
    overwrite: bool = True,
    output_names: "list[str] | None" = None,
) -> Path:
    """Synthesize + write features for one episode with a single MOX model.

    Args:
        ep_dir: raw episode dir (must contain ``episode.npz`` + ``meta.json``).
        mox_model: MOX model id (see :data:`odor_sim.sensors.mox_pid.MOX_MODELS`).
        load_resistance: voltage-divider RL [ohms]; defaults to the model's R0.
        vcc: divider supply voltage (V).
        rate: control frequency (Hz) for the sensor transient; defaults to the
            episode's ``control_freq`` meta (or 20).
        overwrite: if False and the primary output ``.npz`` exists, skip.
        output_names: output basenames to write (each gets ``<name>.npz`` +
            ``<name>_meta.json``). Defaults to ``[CANONICAL_NAME]``.

    Returns:
        Path to the primary (first) written ``.npz``.
    """
    ep_dir = Path(ep_dir)
    names = list(output_names) if output_names else [CANONICAL_NAME]
    npz_path = ep_dir / "episode.npz"
    meta_path = ep_dir / "meta.json"
    if not npz_path.is_file():
        raise FileNotFoundError(f"no episode.npz in {ep_dir}")

    primary_path = ep_dir / f"{names[0]}.npz"
    if primary_path.is_file() and not overwrite:
        print(f"[synth] {primary_path} exists; skipping (--no-overwrite)")
        return primary_path

    meta = json.loads(meta_path.read_text()) if meta_path.is_file() else {}
    gas_types = list(meta.get("gas_types", []))
    if rate is None:
        rate = float(meta.get("control_freq", 20))

    episode = np.load(npz_path)
    if not gas_types:
        raise ValueError(f"{meta_path} has no gas_types; cannot map ppm columns")
    ppm_series = _ppm_series(episode, gas_types)
    enose_states = [int(x) for x in np.asarray(episode["enose_state"]).ravel()]
    num_steps = int(np.asarray(episode["ppm"]).shape[0])

    source_identity = None
    source_distance = None
    if "source_identity" in episode.files:
        source_identity = np.asarray(episode["source_identity"], dtype=np.int64).ravel()
    if "source_distance" in episode.files:
        source_distance = np.asarray(episode["source_distance"], dtype=np.float32).ravel()

    def _make_sensor() -> MOXSensor:
        return MOXSensor(model=mox_model, rate=rate, load_resistance=load_resistance, vcc=vcc)

    continuous = synthesize_continuous(ppm_series, sensor=_make_sensor())
    sampling, windows = synthesize_sampling(ppm_series, enose_states, sensor=_make_sensor())

    _check_label_lengths(
        num_steps=num_steps,
        continuous=continuous,
        sampling=sampling,
        source_identity=source_identity,
        source_distance=source_distance,
    )

    window_records = [_window_record(w, rate) for w in windows]

    baseline = _make_sensor().baseline_voltage()
    features_meta = {
        "sensor_model": mox_model,
        "sensor_type": "MOX",
        "R0": _make_sensor().R0,
        "RL": _make_sensor().load_resistance,
        "Vcc": vcc,
        "rate": rate,
        "baseline_voltage": baseline,
        "num_steps": num_steps,
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
    if "class_names" in meta:
        features_meta["class_names"] = list(meta["class_names"])
    elif "objects" in meta:
        features_meta["class_names"] = list(meta["objects"] or [])
    if "num_classes" in meta:
        features_meta["num_classes"] = int(meta["num_classes"])
    elif "class_names" in features_meta:
        features_meta["num_classes"] = len(features_meta["class_names"])
    if "source_identity_index" in meta:
        features_meta["source_identity_index"] = int(meta["source_identity_index"])
    elif source_identity is not None and source_identity.size:
        features_meta["source_identity_index"] = int(source_identity[0])

    paths = [
        _write_features(
            ep_dir,
            name,
            continuous=continuous,
            sampling=sampling,
            window_records=window_records,
            features_meta=features_meta,
            source_identity=source_identity,
            source_distance=source_distance,
        )
        for name in names
    ]

    print(
        f"[synth] {ep_dir.name}: {features_meta['num_steps']} steps, "
        f"{len(window_records)} sample window(s), model={mox_model} "
        f"-> {', '.join(p.name for p in paths)}"
    )
    return paths[0]


def synthesize_episode_models(
    ep_dir: "str | Path",
    *,
    mox_models: "list[str] | None" = None,
    load_resistance: "float | None" = None,
    vcc: float = 5.0,
    rate: "float | None" = None,
    overwrite: bool = True,
) -> list[Path]:
    """Synthesize one episode across one or many MOX models.

    A single model writes only the canonical ``features.npz`` /
    ``features_meta.json``. Multiple models write one
    ``features_<MODEL>.npz`` / ``features_meta_<MODEL>.json`` pair each, and the
    primary model (:data:`PRIMARY_MOX_MODEL`, or the first requested) is also
    mirrored into the canonical ``features.npz`` for downstream conversion.

    Args:
        ep_dir: raw episode dir (must contain ``episode.npz`` + ``meta.json``).
        mox_models: models to synthesize; defaults to all of
            :data:`odor_sim.sensors.mox_pid.MOX_MODELS`.
        load_resistance: voltage-divider RL [ohms]; defaults to each model's R0.
        vcc: divider supply voltage (V).
        rate: control frequency (Hz); defaults to the episode's ``control_freq``.
        overwrite: if False, skip a model whose primary output already exists.

    Returns:
        Primary ``.npz`` path written for each synthesized model.
    """
    models = list(mox_models) if mox_models else list(MOX_MODELS)
    if not models:
        raise ValueError("no MOX models to synthesize")

    # Put the primary model first so its canonical mirror reflects the default.
    if len(models) > 1 and PRIMARY_MOX_MODEL in models:
        models = [PRIMARY_MOX_MODEL] + [m for m in models if m != PRIMARY_MOX_MODEL]

    single = len(models) == 1
    paths: list[Path] = []
    for i, model in enumerate(models):
        if single:
            output_names = [CANONICAL_NAME]
        else:
            output_names = [f"{CANONICAL_NAME}_{model}"]
            if i == 0:
                output_names.append(CANONICAL_NAME)
        paths.append(
            synthesize_episode(
                ep_dir,
                mox_model=model,
                load_resistance=load_resistance,
                vcc=vcc,
                rate=rate,
                overwrite=overwrite,
                output_names=output_names,
            )
        )
    return paths


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
        default=None,
        choices=list(MOX_MODELS),
        help="MOX sensor model; default synthesizes ALL models "
        f"({', '.join(MOX_MODELS)})",
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

    mox_models = [args.mox_model] if args.mox_model else list(MOX_MODELS)
    for ep in episodes:
        synthesize_episode_models(
            ep,
            mox_models=mox_models,
            load_resistance=args.load_resistance,
            vcc=args.vcc,
            rate=args.rate,
            overwrite=not args.no_overwrite,
        )
    print(
        f"[synth] done: {len(episodes)} episode(s) x {len(mox_models)} model(s) "
        f"({', '.join(mox_models)})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

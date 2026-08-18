"""Per-sniff odor event I/O and control-phase tags.

Raw episodes store a full control-rate stream (including hover ticks). Sniff
windows used by ScentFormer / RLDS live in ``odor_events.npz`` + ``odor_events.json``,
sliced from SAMPLE holds. Phase tags on ``episode.npz['phase']`` let the RLDS
builder drop sniff/filter hovers without relying on the no-op action filter.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

PHASE_CONTROL = 0
PHASE_SNIFF_HOLD = 1
PHASE_FILTER_HOLD = 2
PHASE_NAMES = ("control", "sniff_hold", "filter_hold")


def phase_name(value: int) -> str:
    idx = int(value)
    if 0 <= idx < len(PHASE_NAMES):
        return PHASE_NAMES[idx]
    return f"unknown_{idx}"


def write_odor_events(
    ep_dir: "str | Path",
    events: list[dict],
    *,
    mox_models: list[str],
    gas_types: list[str],
    sniff_steps: int,
) -> None:
    """Write ``odor_events.npz`` + ``odor_events.json`` under ``ep_dir``.

    Each event dict must contain:
      ``event_id``, ``object_id``, ``voc_class``, ``start_step``, ``end_step``,
      ``decision`` (``grasp`` / ``reject``), and optionally ``robot_sniff_pose``,
      ``sensor_object_distance``, ``start_sim_time``, ``end_sim_time``.

    Voltage / ppm / sim_time are sliced from ``episode.npz`` using
    ``[start_step, end_step]`` inclusive.
    """
    ep_dir = Path(ep_dir)
    data = np.load(ep_dir / "episode.npz", allow_pickle=False)
    n = len(events)
    c = len(mox_models)
    g = len(gas_types)
    t_sniff = int(sniff_steps)

    voltage = np.zeros((n, t_sniff, c), dtype=np.float32)
    ppm = np.zeros((n, t_sniff, max(g, 1)), dtype=np.float32)
    sim_time = np.zeros((n, t_sniff), dtype=np.float32)

    records = []
    for i, ev in enumerate(events):
        start = int(ev["start_step"])
        end = int(ev["end_step"])
        span = end - start + 1
        if span != t_sniff:
            raise ValueError(
                f"event {ev.get('event_id')} span {span} != sniff_steps {t_sniff} "
                f"(start={start}, end={end})"
            )
        if "enose_voltages" in data.files:
            voltage[i] = np.asarray(data["enose_voltages"][start : end + 1], dtype=np.float32)
        elif "enose_voltage" in data.files:
            voltage[i, :, 0] = np.asarray(data["enose_voltage"][start : end + 1], dtype=np.float32)
        if "ppm" in data.files and g > 0:
            ppm[i, :, :g] = np.asarray(data["ppm"][start : end + 1], dtype=np.float32)
        if "sim_time" in data.files:
            sim_time[i] = np.asarray(data["sim_time"][start : end + 1], dtype=np.float32)

        rec = {
            "event_id": int(ev["event_id"]),
            "object_id": str(ev["object_id"]),
            "voc_class": str(ev.get("voc_class") or ev["object_id"]),
            "start_step": start,
            "end_step": end,
            "start_sim_time": float(ev.get("start_sim_time", sim_time[i, 0])),
            "end_sim_time": float(ev.get("end_sim_time", sim_time[i, -1])),
            "decision": str(ev.get("decision", "reject")),
            "robot_sniff_pose": [
                float(x) for x in np.asarray(ev.get("robot_sniff_pose", [0, 0, 0])).ravel()[:3]
            ],
            "sensor_object_distance": float(ev.get("sensor_object_distance", 0.0)),
        }
        records.append(rec)

    np.savez(
        ep_dir / "odor_events.npz",
        voltage=voltage,
        ppm=ppm,
        sim_time=sim_time,
    )
    payload = {
        "num_events": n,
        "sniff_steps": t_sniff,
        "mox_models": list(mox_models),
        "gas_types": list(gas_types),
        "events": records,
    }
    (ep_dir / "odor_events.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )


def load_odor_events(ep_dir: "str | Path") -> dict:
    """Return ``{"meta": ..., "voltage": ..., "ppm": ..., "sim_time": ...}``."""
    ep_dir = Path(ep_dir)
    meta = json.loads((ep_dir / "odor_events.json").read_text(encoding="utf-8"))
    arrays = np.load(ep_dir / "odor_events.npz", allow_pickle=False)
    return {
        "meta": meta,
        "voltage": np.asarray(arrays["voltage"], dtype=np.float32),
        "ppm": np.asarray(arrays["ppm"], dtype=np.float32),
        "sim_time": np.asarray(arrays["sim_time"], dtype=np.float32),
    }


def collapse_keep_mask(phase: np.ndarray, sniff_end_steps: list[int]) -> np.ndarray:
    """True for control steps plus the last SAMPLE step of each sniff (decision frame)."""
    phase = np.asarray(phase, dtype=np.int8).ravel()
    keep = np.ones(phase.shape[0], dtype=bool)
    keep[phase == PHASE_SNIFF_HOLD] = False
    keep[phase == PHASE_FILTER_HOLD] = False
    for idx in sniff_end_steps:
        i = int(idx)
        if 0 <= i < keep.shape[0]:
            keep[i] = True
    return keep

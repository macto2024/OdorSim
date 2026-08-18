"""Tests for OVLA odor collection config, events, RLDS collapse, and env layout."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("MUJOCO_GL", "egl")

from odor_sim.config.gas_types import EXCEL_VOC_GAS_NAMES
from odor_sim.recording.odor_config import (
    DEFAULT_ODOR_CONFIG,
    FILTER_DURATION_FACTOR,
    OdorConfig,
    odor_config_from_args,
)
from odor_sim.recording.odor_events import (
    PHASE_CONTROL,
    PHASE_FILTER_HOLD,
    PHASE_SNIFF_HOLD,
    collapse_keep_mask,
    write_odor_events,
)
from odor_sim.recording.build_rlds_odor_ovla import attach_odor, collapse_episode_steps


def test_odor_config_derives_steps_from_rate_and_duration():
    cfg = OdorConfig(odor_sampling_rate=20.0, sniff_duration_s=10.0)
    assert cfg.sniff_steps == 200
    assert cfg.filter_s == pytest.approx(FILTER_DURATION_FACTOR * 10.0)
    assert cfg.filter_steps == 300
    assert cfg.num_odor_channels == 8
    assert cfg.control_freq == 20


def test_odor_config_filter_override_and_five_second_sniff():
    cfg = OdorConfig(sniff_duration_s=5.0, filter_duration_s=2.0, odor_sampling_rate=20.0)
    assert cfg.sniff_steps == 100
    assert cfg.filter_steps == 40
    cfg2 = odor_config_from_args(sniff_duration_s=7.0)
    assert cfg2.sniff_steps == 140
    assert cfg2.filter_s == pytest.approx(10.5)


def test_default_voc_pool_matches_scentformer_gases():
    assert list(DEFAULT_ODOR_CONFIG.voc_pool) == list(EXCEL_VOC_GAS_NAMES)


def test_collapse_keep_mask_keeps_decision_frames_only():
    phase = np.array(
        [0, 0, 1, 1, 1, 0, 2, 2, 0, 1, 1, 0],
        dtype=np.int8,
    )
    keep = collapse_keep_mask(phase, sniff_end_steps=[4, 10])
    assert keep.tolist() == [
        True,
        True,
        False,
        False,
        True,
        True,
        False,
        False,
        True,
        False,
        True,
        True,
    ]


def test_attach_odor_uses_latest_completed_sniff(tmp_path: Path):
    t_sniff, c = 4, 8
    voltage = np.stack(
        [
            np.full((t_sniff, c), 1.0, dtype=np.float32),
            np.full((t_sniff, c), 2.0, dtype=np.float32),
        ]
    )
    keep = np.array([0, 4, 5, 10], dtype=np.int64)
    odor, valid = attach_odor(
        keep,
        voltage=voltage,
        end_steps=np.array([4, 10], dtype=np.int64),
        t_sniff=t_sniff,
        num_channels=c,
    )
    assert valid.tolist() == [False, True, True, True]
    assert float(odor[0].max()) == 0.0
    assert float(odor[1, 0, 0]) == 1.0
    assert float(odor[3, 0, 0]) == 2.0


def test_write_odor_events_slices_episode_npz(tmp_path: Path):
    t, c, g = 12, 8, 2
    sniff = 3
    voltages = np.arange(t * c, dtype=np.float32).reshape(t, c)
    np.savez(
        tmp_path / "episode.npz",
        enose_voltages=voltages,
        ppm=np.ones((t, g), dtype=np.float32),
        sim_time=np.arange(t, dtype=np.float32) / 20.0,
    )
    events = [
        {
            "event_id": 0,
            "object_id": "ethanol",
            "voc_class": "ethanol",
            "start_step": 2,
            "end_step": 4,
            "decision": "reject",
            "robot_sniff_pose": [0.1, 0.0, 1.0],
            "sensor_object_distance": 0.12,
        }
    ]
    write_odor_events(
        tmp_path,
        events,
        mox_models=[f"m{i}" for i in range(c)],
        gas_types=["ethanol", "methanol"],
        sniff_steps=sniff,
    )
    arrays = np.load(tmp_path / "odor_events.npz")
    assert arrays["voltage"].shape == (1, sniff, c)
    np.testing.assert_array_equal(arrays["voltage"][0], voltages[2:5])
    meta = json.loads((tmp_path / "odor_events.json").read_text())
    assert meta["events"][0]["decision"] == "reject"


def test_collapse_episode_steps_drops_holds_and_zero_actions():
    phase = np.array([0, 1, 1, 1, 0, 0], dtype=np.int8)
    action7 = np.zeros((6, 7), dtype=np.float32)
    action7[0, 0] = 0.5
    action7[3, 0] = 0.2  # decision frame
    action7[5, 0] = 0.4
    # index 4 is control but zero arm → no-op
    keep, n_hold, n_noop = collapse_episode_steps(
        phase=phase, action7=action7, sniff_end_steps=[3]
    )
    assert n_hold == 2  # two sniff_hold steps dropped (1,2); 3 kept
    assert 4 not in keep.tolist()
    assert keep.tolist() == [0, 3, 5]


def test_liquid_aliases_exist():
    from odor_sim.config.liquids import get_liquid_spec

    for name in DEFAULT_ODOR_CONFIG.voc_pool:
        spec = get_liquid_spec(name)
        assert spec.recipe


def test_ovla_1_0_recipes_use_excel_gas_panel():
    from odor_sim.config.liquids import get_liquid_spec
    from odor_sim.config.recipes import excel_panel_spec, get_recipe
    from odor_sim.sensors.mox_pid import DEFAULT_MOX_MODELS, get_mox_profile

    for liquid in DEFAULT_ODOR_CONFIG.voc_pool:
        spec = get_liquid_spec(liquid)
        profile = get_recipe(spec.recipe)
        voc = liquid.replace("1.0", "")
        expected = excel_panel_spec(voc, strength=1.0)
        assert [c.gas_type.name for c in profile.components] == [row["gas_type"] for row in expected]
        assert [c.strength for c in profile.components] == [row["strength"] for row in expected]
        assert [c.gas_type.name for c in profile.components] == list(EXCEL_VOC_GAS_NAMES)
        active = [c for c in profile.active_components()]
        assert len(active) == 1
        assert active[0].gas_type.name == voc

    for alias in ("ethanol1.0", "methanol1.0", "acetone1.0"):
        spec = get_liquid_spec(alias)
        assert get_recipe(spec.recipe).active_components()[0].gas_type.name == alias.replace("1.0", "")

    for model in DEFAULT_MOX_MODELS:
        sensitivity = get_mox_profile(model)["sensitivity"]
        for gas in EXCEL_VOC_GAS_NAMES:
            assert sensitivity[gas].get("runtime_supported") is True


def test_episode_recorder_keeps_voltages_if_start_frame_omits_them(tmp_path: Path):
    from odor_sim.bridge.teleop import EpisodeRecorder

    rec = EpisodeRecorder(
        tmp_path,
        "put the cup of ethanol on the plate",
        list(EXCEL_VOC_GAS_NAMES),
        {"mox_models": ["101", "203"], "odor_mode": "discrete"},
    )
    rec.add(
        sim_time=0.0,
        state=np.zeros(3, dtype=np.float32),
        action=np.zeros(2, dtype=np.float32),
        enose_state=0,
        sampling_active=False,
        ppm={},
        source_distance=0.1,
        enose_pos=[0.0, 0.0, 1.0],
        target_pos=[0.1, 0.0, 0.8],
    )
    rec.add(
        sim_time=0.05,
        state=np.zeros(3, dtype=np.float32),
        action=np.zeros(2, dtype=np.float32),
        enose_state=1,
        sampling_active=True,
        ppm={"ethanol": 12.0},
        source_distance=0.1,
        enose_pos=[0.0, 0.0, 1.0],
        target_pos=[0.1, 0.0, 0.8],
        enose_voltage=1.4,
        enose_voltages={"101": 1.4, "203": 1.25},
    )
    ep_dir = rec.write(True)
    data = np.load(ep_dir / "episode.npz")
    assert data["enose_voltages"].shape == (2, 2)
    assert float(data["enose_voltages"][1, 0]) == pytest.approx(1.4)
    assert float(data["ppm"][1, 0]) == pytest.approx(12.0)


def test_require_gaden_excel_gases_checks_binary(tmp_path: Path, monkeypatch):
    from odor_sim.runtime import gaden_gas_support as support

    fake = tmp_path / "rt_server"
    fake.write_bytes(b"odor_gaden_rt ethanol acetone")
    monkeypatch.setattr(support, "find_rt_server", lambda: fake)
    with pytest.raises(RuntimeError, match="IDs 14"):
        support.require_gaden_excel_gases(["methanol"])
    support.require_gaden_excel_gases(["ethanol", "acetone"])


def test_classify_liquid_place_ovla_porcelain_and_slots():
    import odor_sim as odorsim
    from odor_sim.envs.classify_liquid_place_ovla import (
        CUP_CATALOG,
        ClassifyLiquidPlaceOVLA,
    )

    with odorsim.make(
        "ClassifyLiquidPlaceOVLA",
        liquids=["ethanol", "methanol"],
        target_liquid="ethanol",
        auto_start_gaden=False,
        bridge=False,
        export=False,
        odor_mode="none",
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        control_freq=20,
        horizon=50,
        seed=0,
    ) as cosim:
        env = cosim.env
        assert isinstance(env, ClassifyLiquidPlaceOVLA)
        assert env.gas_types == list(EXCEL_VOC_GAS_NAMES)
        cosim.reset()
        assert env.cup_catalog_name == CUP_CATALOG
        poses = env.get_object_world_poses()
        plate = np.asarray(poses[env.place_target_name][0], dtype=float)
        assert env.plate_x_range[0] - 0.05 <= plate[0] - env.table_offset[0] <= env.plate_x_range[1] + 0.05
        cups_y = []
        for lid in env.object_names:
            pos = np.asarray(poses[lid][0], dtype=float)
            cups_y.append(float(pos[1] - env.table_offset[1]))
        assert min(cups_y) < 0 < max(cups_y)
        layout = env.spawn_layout()
        assert env.place_target_name in layout
        assert all(layout[n]["catalog"] == CUP_CATALOG for n in env.object_names)

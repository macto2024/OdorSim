"""Tests for OVLA odor collection config, events, RLDS collapse, and env layout."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("MUJOCO_GL", "egl")

from odor_sim.config.frame_map import FrameMap
from odor_sim.config.gas_types import EXCEL_VOC_GAS_NAMES
from odor_sim.recording.odor_config import (
    DEFAULT_ODOR_CONFIG,
    FILTER_DURATION_FACTOR,
    OdorConfig,
    odor_config_from_args,
)
from odor_sim.recording.odor_events import (
    DEFAULT_SNIFF_LANDMARKS_S,
    PHASE_CONTROL,
    PHASE_FILTER_HOLD,
    PHASE_SNIFF_HOLD,
    collapse_keep_mask,
    sniff_landmark_steps,
    write_odor_events,
)
from odor_sim.recording.build_rlds_odor_ovla import (
    attach_continuous_odor,
    attach_odor,
    collapse_episode_steps,
)
from odor_sim.envs.odor_profile import calibrated_emission_parameters
from odor_sim.bridge.collect_odor_place_oft import (
    _plan_target_first,
    _sample_paired_local_ppm,
)
from odor_sim.bridge.scripted_odor_place import ScriptedOdorPlaceCollector
from odor_sim.envs.classify_liquid_place_ovla import ClassifyLiquidPlaceOVLA


def test_default_frame_map_preserves_tabletop_height():
    table_top = np.array([0.0, 0.0, 0.8])
    expected_gaden = np.array([2.0, 3.0, 0.8])

    direct = FrameMap()
    from_scenario = FrameMap.from_scenario(
        "scenarios/10x6_uniform/environment_configurations/config1"
    )

    np.testing.assert_allclose(direct.robosuite_to_gaden(table_top), expected_gaden)
    np.testing.assert_allclose(
        from_scenario.robosuite_to_gaden(table_top), expected_gaden
    )
    np.testing.assert_allclose(
        direct.gaden_to_robosuite(expected_gaden), table_top
    )


def test_half_target_first_plan_is_balanced_and_retry_stable():
    rng = np.random.default_rng(7)
    planned = [_plan_target_first(i, 0.5, rng) for i in range(30)]
    assert planned == [i % 2 == 0 for i in range(30)]
    assert sum(planned) == 15

    for pair_index in range(30):
        pair_orders = [
            _plan_target_first(pair_index + repeat * 30, 0.5, rng)
            for repeat in range(4)
        ]
        assert sum(pair_orders) == 2

    # Once planned, batch collection passes an exact 0/1 probability to every
    # retry rather than drawing a new order from each attempt's seed.
    assert _plan_target_first(0, 0.0, rng) is False
    assert _plan_target_first(1, 1.0, rng) is True


def test_paired_source_sampler_is_symmetric_and_stays_calibrated():
    rng = np.random.default_rng(11)
    log_ratios = []
    for _ in range(1000):
        ppm, base, ratio = _sample_paired_local_ppm(
            rng,
            ["ethanol", "acetone"],
            local_ppm_min=5.0,
            local_ppm_max=100.0,
            ratio_min=0.8,
            ratio_max=1.25,
        )
        assert ppm["ethanol"] * ppm["acetone"] == pytest.approx(base**2)
        assert 5.0 <= ppm["ethanol"] <= 100.0
        assert 5.0 <= ppm["acetone"] <= 100.0
        log_ratios.append(np.log(ratio))
    assert abs(float(np.mean(log_ratios))) < 0.02


def test_ovla_cup_slot_spacing_is_between_22_and_30_cm():
    min_dy = (
        ClassifyLiquidPlaceOVLA.cup_right_y_range[0]
        - ClassifyLiquidPlaceOVLA.cup_left_y_range[1]
    )
    max_dy = (
        ClassifyLiquidPlaceOVLA.cup_right_y_range[1]
        - ClassifyLiquidPlaceOVLA.cup_left_y_range[0]
    )
    max_dx = (
        ClassifyLiquidPlaceOVLA.cup_right_x_range[1]
        - ClassifyLiquidPlaceOVLA.cup_left_x_range[0]
    )
    assert min_dy == pytest.approx(0.22)
    assert np.hypot(max_dx, max_dy) < 0.30


def test_recent_decision_evidence_uses_window_median_not_final_spike():
    collector = object.__new__(ScriptedOdorPlaceCollector)
    collector.decision_window_s = 1.0
    collector.control_freq = 20
    collector.decision_min_total_ppm = 5.0
    collector.decision_dominance_threshold = 0.70
    collector.decision_min_pass_fraction = 0.80
    collector.gas_types = ["ethanol", "acetone"]

    class Recorder:
        pass

    recorder = Recorder()
    recorder._rows = [
        {"ppm": np.array([4.0, 6.0], dtype=np.float32)} for _ in range(19)
    ] + [{"ppm": np.array([9.0, 1.0], dtype=np.float32)}]
    evidence = collector._recent_decision_evidence(
        recorder, cup_id="ethanol", end_step=19
    )
    assert evidence["median_dominance"] == pytest.approx(0.4)
    assert evidence["dominance_pass_fraction"] == pytest.approx(0.05)
    assert evidence["evidence_accepted"] is False

    recorder._rows = [
        {"ppm": np.array([8.0, 2.0], dtype=np.float32)} for _ in range(20)
    ]
    evidence = collector._recent_decision_evidence(
        recorder, cup_id="ethanol", end_step=19
    )
    assert evidence["median_dominance"] == pytest.approx(0.8)
    assert evidence["evidence_accepted"] is True


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


def test_sniff_landmarks_are_sparse_deterministic_and_include_decision():
    events = [{"start_step": 20, "end_step": 219}]
    first = sniff_landmark_steps(
        events, rate_hz=20.0, landmarks_s=DEFAULT_SNIFF_LANDMARKS_S,
        jitter_s=0.1, seed=7,
    )
    second = sniff_landmark_steps(
        events, rate_hz=20.0, landmarks_s=DEFAULT_SNIFF_LANDMARKS_S,
        jitter_s=0.1, seed=7,
    )
    assert first == second
    assert first[-1] == 219
    assert len(first) == len(DEFAULT_SNIFF_LANDMARKS_S) + 1
    assert all(20 <= index <= 219 for index in first)


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


def test_attach_odor_uses_causal_active_sniff_prefix_and_padding_mask():
    voltage = np.arange(4 * 2, dtype=np.float32).reshape(1, 4, 2) + 1
    odor, valid, padding = attach_odor(
        np.array([1, 2, 3, 5, 6]),
        voltage=voltage,
        start_steps=np.array([2]),
        end_steps=np.array([5]),
        t_sniff=4,
        num_channels=2,
        return_padding_mask=True,
    )
    assert valid.tolist() == [False, True, True, True, True]
    assert padding[0].tolist() == [True, True, True, True]
    assert padding[1].tolist() == [True, True, True, False]
    assert padding[2].tolist() == [True, True, False, False]
    np.testing.assert_array_equal(odor[1, -1], voltage[0, 0])
    np.testing.assert_array_equal(odor[2, -2:], voltage[0, :2])
    np.testing.assert_array_equal(odor[3], voltage[0])
    np.testing.assert_array_equal(odor[4], voltage[0])


def test_continuous_odor_history_is_causal_and_crosses_sniff_boundaries():
    voltage = np.arange(12 * 2, dtype=np.float32).reshape(12, 2)
    odor, valid, padding = attach_continuous_odor(
        np.array([1, 7, 11]), voltage=voltage, history_steps=5,
    )
    assert valid.tolist() == [True, True, True]
    assert padding[0].tolist() == [True, True, True, False, False]
    np.testing.assert_array_equal(odor[1], voltage[3:8])
    np.testing.assert_array_equal(odor[2], voltage[7:12])


def test_calibrated_emission_separates_concentration_and_filament_rate():
    params = calibrated_emission_parameters(
        10.0, emission_scale=0.06, concentration_exponent=0.5,
        filaments_per_second=15,
    )
    assert params["filament_ppm_center"] == pytest.approx(0.6)
    assert params["num_filaments_sec"] == 15


def test_active_second_sniff_replaces_completed_first_sniff_causally():
    voltage = np.stack(
        [
            np.ones((4, 2), dtype=np.float32),
            np.full((4, 2), 2.0, dtype=np.float32),
        ]
    )
    odor, valid, padding = attach_odor(
        np.array([6, 11]),
        voltage=voltage,
        start_steps=np.array([2, 10]),
        end_steps=np.array([5, 13]),
        t_sniff=4,
        num_channels=2,
        return_padding_mask=True,
    )
    assert valid.tolist() == [True, True]
    np.testing.assert_array_equal(odor[0], voltage[0])
    assert padding[0].tolist() == [False, False, False, False]
    assert padding[1].tolist() == [True, True, False, False]
    np.testing.assert_array_equal(odor[1, -2:], voltage[1, :2])
    assert not np.any(odor[1] == 1.0)


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
            "decision_window_s": 1.0,
            "median_total_ppm": 20.0,
            "median_intended_ppm": 16.0,
            "median_dominance": 0.8,
            "presence_pass_fraction": 1.0,
            "dominance_pass_fraction": 1.0,
            "evidence_accepted": True,
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
    assert meta["events"][0]["median_dominance"] == pytest.approx(0.8)
    assert meta["events"][0]["evidence_accepted"] is True


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


def test_collapse_episode_steps_protects_zero_action_sniff_landmarks():
    phase = np.array([0, 1, 1, 1, 1, 0], dtype=np.int8)
    action7 = np.zeros((6, 7), dtype=np.float32)
    action7[0, 0] = 0.5
    action7[5, 0] = 0.4
    keep, n_hold, n_noop = collapse_episode_steps(
        phase=phase,
        action7=action7,
        sniff_end_steps=[4],
        sniff_landmarks=[2],
    )
    assert keep.tolist() == [0, 2, 4, 5]
    assert n_hold == 2
    assert n_noop == 0


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

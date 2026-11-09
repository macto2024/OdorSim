"""Focused tests for staged odor-encoder collection and training helpers."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import yaml
try:
    import torch
except ImportError:  # Training dependency is optional for collection-only installs.
    torch = None

from odor_sim.odor_encoder.dataset import (
    CLASS_ORDER, GAS_ORDER, HELD_OUT_COMBINATIONS, SENSOR_ORDER, MixtureSpec, assign_split,
    current_class_labels,
    sample_background, sample_mixture, sample_single_source, simulate_mox,
    source_labels, write_episode,
)
from odor_sim.config.frame_map import FrameMap
from odor_sim.odor_encoder.phase_a import PATTERNS, concentration_envelope
from odor_sim.odor_encoder.phase_c import build_sequence as build_onset_recovery
from odor_sim.odor_encoder.phase_d import (
    build_sequence as build_carryover, event_timing_arrays,
)
from odor_sim.odor_encoder.phase_b import (
    PATH_NAMES, PATH_WEIGHTS, _emission_parameters, _emission_strength,
    _scheduled_source, observable_primary_mask,
    sample_background_offset, scheduled_target, scheduled_wind_regime,
    scripted_path,
)
from odor_sim.odor_encoder.wind import (
    WIND_REGIMES, WIND_REGIME_WEIGHTS, WindProfile, sample_wind_profile,
    write_isolated_config, zero_wind_profile,
)
from odor_sim.odor_encoder.two_source_rviz import _crosswind_profile
from odor_sim.odor_encoder.supplement_phase_d import (
    _base_seed, _next_index, _phase_d_report,
)
from odor_sim.envs.odor_profile import OdorProfile
from odor_sim.envs.scene_builder import SceneBuilder
from odor_sim.runtime.gaden_server import IsolatedGadenServerManager
if torch is not None:
    from odor_sim.odor_encoder.train_ab import OdorTransformer, _loss
    from odor_sim.odor_encoder.train_abcd import (
        BaselineScentFormer, DualTimescaleScentFormer, PhysicalWindows, PREFIXES,
        _binary_metrics as _abcd_binary_metrics, _loss as _abcd_loss,
        _wilson_interval,
    )
    from odor_sim.odor_encoder.physical_pipeline import (
        PAIR_SPLIT_CYCLE, PHASE_ROS_DOMAINS, _balanced_desired_ppm, _phase_a_mode,
        _phase_a_desired_ppm, _phase_a_role_is_target,
        _scheduled_split, physical_labels, scheduled_pair,
    )


class TestSampling(unittest.TestCase):
    def test_phase_d_supplement_helpers_preserve_indices_and_seed_base(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "test").mkdir()
            (root / "test" / "physical_d_000007_ethanol_to_acetone.json").write_text("{}")
            (root / "validation").mkdir()
            (root / "validation" / "physical_d_000012_acetone_to_ethanol.json").write_text("{}")
            self.assertEqual(_next_index(root), 13)
        metadata = {
            "episode_id": "physical_d_000012_acetone_to_ethanol",
            "seed": 51000 + 30_000_000 + 12 * 1009,
        }
        self.assertEqual(_base_seed(metadata), 51000)
        report = {"datasets": [{"label": "A"}, {"label": "D", "root": "d"}]}
        self.assertEqual(_phase_d_report(report)["root"], "d")

    def test_physical_phases_have_distinct_ros_domains(self):
        self.assertEqual(set(PHASE_ROS_DOMAINS), {"A", "B", "C", "D"})
        self.assertEqual(len(set(PHASE_ROS_DOMAINS.values())), 4)

    def test_isolated_server_command_avoids_legacy_cleanup_patterns(self):
        manager = IsolatedGadenServerManager(
            Path("/tmp/not_started"), worker_id="domain82_test",
        )
        command = " ".join(manager._command())
        self.assertNotIn("ros2 run odor_gaden_rt rt_server", command)
        self.assertNotIn("/odor_gaden_rt/lib/odor_gaden_rt/rt_server", command)

    def test_split_is_stable_and_close_to_70_15_15(self):
        values = [assign_split(f"episode_{i}", 7) for i in range(10000)]
        fractions = {name: values.count(name) / len(values) for name in ("train", "validation", "test")}
        self.assertAlmostEqual(fractions["train"], 0.70, delta=0.02)
        self.assertAlmostEqual(fractions["validation"], 0.15, delta=0.02)
        self.assertAlmostEqual(fractions["test"], 0.15, delta=0.02)

    def test_reserved_combinations_never_enter_train(self):
        rng = np.random.default_rng(4)
        for _ in range(10000):
            spec = sample_mixture(rng, split="train")
            self.assertNotIn(tuple(sorted(spec.gases)), HELD_OUT_COMBINATIONS)

    def test_cardinality_distribution(self):
        rng = np.random.default_rng(8)
        counts = np.asarray([len(sample_mixture(rng, split="train").gases) for _ in range(20000)])
        expected = {0: 0.10, 1: 0.30, 2: 0.30, 3: 0.20}
        for count, probability in expected.items():
            self.assertAlmostEqual(float(np.mean(counts == count)), probability, delta=0.02)
        self.assertAlmostEqual(float(np.mean(counts >= 4)), 0.10, delta=0.02)

    def test_single_source_target_excludes_background_gases(self):
        rng = np.random.default_rng(19)
        primary = sample_single_source(rng, clean_probability=0.0)
        background = sample_background(rng, primary, max_gases=2)
        self.assertEqual(len(primary.gases), 1)
        self.assertGreaterEqual(len(background.gases), 1)
        self.assertTrue(set(primary.gases).isdisjoint(background.gases))
        measured_ppm = np.repeat(
            (primary.concentration_ppm + background.concentration_ppm)[None], 20, axis=0
        )
        labels = source_labels(primary, measured_ppm, np.ones(20, bool))
        self.assertEqual(int(labels["label_presence"][0].sum()), 1)
        self.assertEqual(int(np.count_nonzero(measured_ppm[0])), 1 + len(background.gases))


class TestSequences(unittest.TestCase):
    def test_zero_wind_is_exactly_zero(self):
        profile = zero_wind_profile(40.0)
        self.assertEqual(profile.regime, "zero")
        self.assertTrue(np.all(profile.vectors_m_s == 0))

    def test_two_source_crosswind_is_perpendicular_to_cup_axis(self):
        profile = WindProfile(
            "stagnant_correlated",
            np.asarray([[0.01, 0.0, 0.0], [0.02, 0.0, 0.001]], np.float32),
            np.asarray([1.0, 0.0, 0.0], np.float32),
        )
        positions = np.asarray([[2.0, 2.8, 0.87], [2.0, 3.2, 0.87]])
        crosswind = _crosswind_profile(profile, positions, seed=42)
        cup_axis = positions[1, :2] - positions[0, :2]
        self.assertAlmostEqual(
            float(crosswind.vectors_m_s[:, :2].mean(0) @ cup_axis), 0.0, places=7,
        )

    def test_onset_recovery_keeps_air_and_gas_targets_separate(self):
        ppm, classes, phase, valid, recovery, bounds = build_onset_recovery(
            gas_index=2, concentration_ppm=10.0, rate=20.0,
            baseline_s=2.0, exposure_s=7.0, recovery_s=15.0,
            onset_loss_delay_s=0.5, recovery_loss_delay_s=1.0,
            rng=np.random.default_rng(2),
        )
        self.assertTrue(np.all(classes[:bounds["gas_start"]] == 0))
        self.assertTrue(np.all(classes[bounds["gas_start"]:bounds["recovery_start"]] == 3))
        self.assertTrue(np.all(classes[bounds["recovery_start"]:] == 0))
        self.assertTrue(np.all(ppm[bounds["recovery_start"]:] == 0))
        self.assertTrue(np.all(recovery[bounds["recovery_start"]:]))
        self.assertFalse(valid[bounds["recovery_start"]])
        labels = current_class_labels(classes, ppm, valid, recovery)
        self.assertEqual(labels["label_class"].shape, (len(ppm),))

    def test_carryover_sequence_has_no_inserted_full_recovery(self):
        ppm, classes, _, valid, recovery, bounds = build_carryover(
            np.random.default_rng(5), 0, 4, rate_hz=20.0,
        )
        a0 = bounds["gas_a_start_index"]
        transit = bounds["transit_start_index"]
        b0 = bounds["gas_b_start_index"]
        self.assertTrue(np.all(classes[a0:transit] == 1))
        self.assertTrue(np.all(classes[transit:b0] == 0))
        self.assertTrue(np.all(classes[b0:] == 5))
        self.assertTrue(np.all(ppm[transit:b0] == 0))
        self.assertTrue(np.all(recovery[transit:b0]))
        self.assertFalse(valid[b0])

        timing = event_timing_arrays(len(ppm), bounds, 20.0)
        self.assertTrue(np.all(timing["event_id"][a0:transit] == 1))
        self.assertTrue(np.all(timing["event_id"][transit:b0] == 2))
        self.assertTrue(np.all(timing["event_id"][b0:] == 3))
        self.assertEqual(float(timing["sniff_elapsed_s"][b0]), 0.0)
        self.assertTrue(np.isnan(timing["sniff_elapsed_s"][transit]))

    def test_one_simulation_preserves_mox_state_into_clean_air(self):
        ppm = np.zeros((240, len(GAS_ORDER)), np.float32)
        ppm[:160, 0] = 50.0
        _, continuous = simulate_mox(ppm, 20.0)
        _, reset_air = simulate_mox(np.zeros((80, len(GAS_ORDER)), np.float32), 20.0)
        self.assertGreater(float(np.max(np.abs(continuous[160] - reset_air[0]))), 1e-3)

    def test_scene_builder_writes_configurable_buoyancy_tau(self):
        builder = SceneBuilder()
        builder.add_object(SimpleNamespace(
            object_id="source",
            odor_profile=OdorProfile.from_spec([("ethanol", 1.0)]),
        ))
        with tempfile.TemporaryDirectory() as directory:
            config_dir = Path(directory)
            builder.export_gaden_scene(
                config_dir, buoyancy_tau_s=0.5, filament_noise_std=0.0,
            )
            sim_path = next((config_dir / "simulations").glob("*/sim.yaml"))
            sim = yaml.safe_load(sim_path.read_text())
            self.assertEqual(sim["buoyancyTau"], 0.5)
            self.assertFalse(sim["originalTerminalVelocity"])
            self.assertEqual(sim["filamentNoise_std"], 0.0)
            with self.assertRaisesRegex(ValueError, "non-negative"):
                builder.export_gaden_scene(config_dir, buoyancy_tau_s=-1.0)

    def test_scene_builder_preserves_separated_low_emission_controls(self):
        builder = SceneBuilder()
        builder.add_object(SimpleNamespace(
            object_id="source",
            odor_profile=OdorProfile.from_spec([{
                "gas_type": "ethanol",
                "filament_ppm_center": 0.0002,
                "num_filaments_sec": 17,
            }]),
        ))
        with tempfile.TemporaryDirectory() as directory:
            config_dir = Path(directory)
            builder.export_gaden_scene(config_dir)
            sim_path = next((config_dir / "simulations").glob("*/sim.yaml"))
            simulation = yaml.safe_load(sim_path.read_text())
            self.assertEqual(simulation["filamentPPMcenter"], 0.0002)
            self.assertEqual(simulation["numFilaments_sec"], 17)

    def test_scene_builder_applies_shared_encoder_emission_model(self):
        builder = SceneBuilder()
        builder.add_object(SimpleNamespace(
            object_id="cup_a",
            odor_profile=OdorProfile.from_spec([("ethanol", 1.0)]),
        ))
        configured = builder.configure_calibrated_emission(
            {"cup_a": 10.0}, emission_scale=0.06,
            concentration_exponent=0.5, filaments_per_second=15,
        )
        self.assertAlmostEqual(configured["cup_a"]["filament_ppm_center"], 0.6)
        self.assertEqual(builder.sources[0].component.num_filaments_sec, 15)

    def test_all_static_patterns_have_exposure_and_baseline(self):
        self.assertNotIn("pulses", PATTERNS)
        self.assertIn("constant", PATTERNS)
        rng = np.random.default_rng(2)
        for name in PATTERNS:
            envelope, valid = concentration_envelope(name, 400, 20, rng)
            self.assertEqual(envelope.shape, (400,))
            self.assertTrue(np.any(envelope > 0))
            self.assertTrue(np.any(envelope == 0))
            self.assertFalse(valid[0])

    def test_virtual_path_distribution_and_shapes(self):
        self.assertAlmostEqual(float(PATH_WEIGHTS.sum()), 1.0)
        self.assertAlmostEqual(float(PATH_WEIGHTS[:2].sum()), 0.80)
        self.assertNotIn("leave_enter", PATH_NAMES)
        self.assertNotIn("circle", PATH_NAMES)
        self.assertNotIn("zigzag", PATH_NAMES)
        for name in PATH_NAMES:
            path, phase = scripted_path(name, 240, np.asarray([2.0, 3.0, 0.5]))
            self.assertEqual(path.shape, (240, 3))
            self.assertEqual(phase.shape, (240,))
            self.assertTrue(np.isfinite(path).all())

    def test_default_tabletop_maps_to_gaden_z_0p8(self):
        mapped = FrameMap().robosuite_to_gaden([0.0, 0.0, 0.8])
        np.testing.assert_allclose(mapped, [2.0, 3.0, 0.8])

    def test_virtual_sniff_distance_is_respected(self):
        source = np.asarray([2.0, 3.0, 0.5])
        for distance in (0.01, 0.03, 0.05):
            path, _ = scripted_path("stationary_downwind", 40, source, distance)
            measured = np.linalg.norm(path - source, axis=1)
            np.testing.assert_allclose(measured, distance, atol=1e-6)

    def test_virtual_schedule_balances_gases_and_paths(self):
        targets = [scheduled_target(index, 37) for index in range(400)]
        self.assertEqual(sum(gas is None for gas, _ in targets), 40)
        for gas_index in range(len(GAS_ORDER)):
            gas_paths = [path for gas, path in targets if gas == gas_index]
            self.assertEqual(len(gas_paths), 60)
            self.assertEqual([gas_paths.count(path) for path in PATH_NAMES], [27, 21, 6, 6])

    def test_wind_schedule_has_exact_requested_proportions(self):
        regimes = [scheduled_wind_regime(index, 37) for index in range(20)]
        self.assertEqual([regimes.count(name) for name in WIND_REGIMES], [8, 9, 3])

    def test_virtual_emission_strength_is_log_scaled(self):
        self.assertAlmostEqual(_emission_strength(1.0), 0.15)
        self.assertAlmostEqual(_emission_strength(100.0), 1.0)
        self.assertLess(_emission_strength(2.0), _emission_strength(20.0))

    def test_nonlinear_filament_concentration_mapping(self):
        def center(ppm, exponent):
            return _emission_parameters(
                ppm, model="separated", emission_scale=0.06,
                concentration_exponent=exponent, filaments_per_second=15,
                background_ratio=1.0,
            )["filament_ppm_center"]

        self.assertAlmostEqual(center(10.0, 0.5), 0.6)
        self.assertAlmostEqual(center(1.0, 1.0), 0.06)
        self.assertAlmostEqual(center(100.0, 1.0), 6.0)
        self.assertLess(center(100.0, 0.5) / center(1.0, 0.5), 100.0)

    def test_nominal_concentration_sampling_modes_stay_in_range(self):
        for mode in ("log_uniform", "linear_uniform"):
            rng = np.random.default_rng(11)
            values = [
                _scheduled_source(0, rng, mode).concentration_ppm[0]
                for _ in range(100)
            ]
            self.assertGreaterEqual(min(values), 1.0)
            self.assertLessEqual(max(values), 100.0)

    def test_background_offset_uses_true_cup_separation(self):
        rng = np.random.default_rng(31)
        distances = [
            np.linalg.norm(sample_background_offset(rng, 0.20, 0.30))
            for _ in range(1000)
        ]
        self.assertGreaterEqual(min(distances), 0.20)
        self.assertLessEqual(max(distances), 0.30)
        self.assertGreater(np.ptp(distances), 0.09)

    def test_background_orientations_are_relative_to_wind(self):
        rng = np.random.default_rng(31)
        wind = np.asarray([1.0, 0.0, 0.0])
        upwind = sample_background_offset(rng, 0.2, 0.2, wind, "upwind")
        downwind = sample_background_offset(rng, 0.2, 0.2, wind, "downwind")
        crosswind = sample_background_offset(rng, 0.2, 0.2, wind, "crosswind")
        self.assertLess(upwind[0], 0)
        self.assertGreater(downwind[0], 0)
        self.assertAlmostEqual(float(crosswind[0]), 0.0, places=7)
        self.assertAlmostEqual(abs(float(crosswind[1])), 0.2, places=7)

    def test_wind_regime_speed_ranges_and_persistence(self):
        self.assertAlmostEqual(float(WIND_REGIME_WEIGHTS.sum()), 1.0)
        stagnant = sample_wind_profile(np.random.default_rng(1), 30, WIND_REGIMES[0])
        weak = sample_wind_profile(np.random.default_rng(2), 30, WIND_REGIMES[1])
        vent = sample_wind_profile(np.random.default_rng(3), 30, WIND_REGIMES[2])
        horizontal = np.linalg.norm(stagnant.vectors_m_s[:, :2], axis=1)
        self.assertLessEqual(float(horizontal.max()), 0.030001)
        self.assertGreaterEqual(float(weak.speed_m_s.min()), 0.03)
        self.assertLessEqual(float(weak.speed_m_s.max()), 0.121)
        self.assertGreaterEqual(float(vent.speed_m_s.min()), 0.15)
        self.assertLessEqual(float(vent.speed_m_s.max()), 0.301)
        np.testing.assert_allclose(
            weak.vectors_m_s,
            np.repeat(weak.vectors_m_s[0:1], len(weak.vectors_m_s), axis=0),
        )
        np.testing.assert_allclose(
            vent.vectors_m_s,
            np.repeat(vent.vectors_m_s[0:1], len(vent.vectors_m_s), axis=0),
        )
        changes = np.linalg.norm(np.diff(stagnant.vectors_m_s, axis=0), axis=1)
        self.assertGreater(float(changes.max()), 0.0)
        self.assertLess(float(np.median(changes)), 0.01)

    def test_isolated_wind_config_reuses_absolute_geometry(self):
        source = Path("scenarios/10x6_uniform/environment_configurations/config1")
        profile = sample_wind_profile(np.random.default_rng(4), 10, "uniform_weak_indoor")
        with tempfile.TemporaryDirectory() as directory:
            config_dir = write_isolated_config(source, Path(directory) / "config", profile)
            config = (config_dir / "config.yaml").read_text()
            self.assertIn("uniformWind: true", config)
            self.assertIn(str(config_dir / "wind" / "wind"), config)
            rows = np.loadtxt(config_dir / "wind" / "wind_0.csv", delimiter=",")
            np.testing.assert_allclose(rows, profile.vectors_m_s, atol=1e-7)

    def test_observability_requires_target_signal_and_dominance(self):
        vector = np.zeros(len(GAS_ORDER), np.float32)
        vector[0] = 10.0
        spec = MixtureSpec(vector, "single_gas")
        ppm = np.zeros((5, len(GAS_ORDER)), np.float32)
        ppm[0, 0] = 0.01
        ppm[1, :2] = (1.0, 0.1)
        ppm[2, :2] = (1.0, 1.0)
        valid, fraction = observable_primary_mask(
            spec, ppm, min_target_ppm=0.05, min_target_fraction=0.75,
            rate=1.0, recovery_s=2.0,
        )
        self.assertEqual(valid.tolist(), [False, True, False, True, False])
        self.assertAlmostEqual(float(fraction[1]), 1.0 / 1.1, places=5)

    def test_observability_can_require_sustained_calibrated_evidence(self):
        vector = np.zeros(len(GAS_ORDER), np.float32)
        vector[0] = 10.0
        spec = MixtureSpec(vector, "single_gas")
        ppm = np.zeros((6, len(GAS_ORDER)), np.float32)
        ppm[:, 0] = (0.0, 50.0, 50.0, 50.0, 400.0, 400.0)
        valid, _ = observable_primary_mask(
            spec, ppm, min_target_ppm=1.0, max_target_ppm=100.0,
            max_total_ppm=100.0, min_target_fraction=0.9,
            rate=1.0, evidence_s=3.0, recovery_s=0.0,
        )
        self.assertEqual(valid.tolist(), [False, False, True, True, False, False])

    def test_observability_masks_falling_concentration(self):
        vector = np.zeros(len(GAS_ORDER), np.float32)
        vector[0] = 10.0
        spec = MixtureSpec(vector, "single_gas")
        ppm = np.zeros((7, len(GAS_ORDER)), np.float32)
        ppm[:, 0] = (1.0, 2.0, 4.0, 8.0, 8.0, 4.0, 2.0)
        valid, _ = observable_primary_mask(
            spec, ppm, min_target_ppm=0.1, min_target_fraction=0.9,
            rate=1.0, evidence_s=1.0, recovery_s=0.0,
            trend_s=1.0, max_decline_fraction=0.15,
        )
        self.assertEqual(valid.tolist(), [True, True, True, True, True, False, False])

    def test_explicit_filament_parameters_are_independent(self):
        profile = OdorProfile.from_spec([{
            "gas_type": "ethanol",
            "strength": 0.5,
            "filament_ppm_center": 0.4,
            "num_filaments_sec": 20,
        }])
        component = profile.components[0]
        self.assertEqual(component.filament_ppm_center, 0.4)
        self.assertEqual(component.num_filaments_sec, 20)

    def test_full_episode_schema(self):
        rng = np.random.default_rng(3)
        spec = sample_mixture(rng, split="train", force_nonempty=True)
        ppm = np.repeat(spec.concentration_ppm[None], 20, axis=0)
        voltage, rs_r0 = simulate_mox(ppm, 20)
        labels = source_labels(spec, ppm, np.ones(20, bool))
        with tempfile.TemporaryDirectory() as directory:
            path = write_episode(
                directory, session_id="schema", split="train", domain="test", rate=20,
                voltage=voltage, rs_r0=rs_r0, ppm=ppm, time_s=np.arange(20) / 20,
                position=np.zeros((20, 3)), phase=np.full(20, "test"), labels=labels,
                metadata={"evaluation_group": spec.evaluation_group},
            )
            data = np.load(path)
            self.assertEqual(data["voltage"].shape, (20, len(SENSOR_ORDER)))
            self.assertEqual(data["ppm"].shape, (20, len(GAS_ORDER)))
            self.assertEqual(data["label_presence"].shape, (20, len(GAS_ORDER)))
            self.assertEqual(data["label_recovery"].shape, (20,))
            self.assertEqual(data["label_class"].shape, (20,))
            self.assertEqual(tuple(data["class_order"].tolist()), CLASS_ORDER)
            self.assertTrue(Path(path).with_suffix(".json").is_file())


class TestModel(unittest.TestCase):
    def test_physical_split_cycle_is_exact_70_15_15(self):
        self.assertEqual(PAIR_SPLIT_CYCLE.count("train"), 14)
        self.assertEqual(PAIR_SPLIT_CYCLE.count("validation"), 3)
        self.assertEqual(PAIR_SPLIT_CYCLE.count("test"), 3)

    def test_physical_pilot_splits_are_balanced(self):
        for episodes, expected in (
            (300, {"train": 210, "validation": 45, "test": 45}),
            (510, {"train": 360, "validation": 75, "test": 75}),
        ):
            counts = {name: 0 for name in ("train", "validation", "test")}
            by_pair = {}
            for index in range(episodes):
                pair = scheduled_pair(index, 91)
                split = _scheduled_split(index, *pair)
                counts[split] += 1
                by_pair.setdefault(pair, set()).add(split)
            self.assertEqual(counts, expected)
            self.assertTrue(all(splits == set(counts) for splits in by_pair.values()))

    def test_physical_pair_schedule_balances_all_directions(self):
        pairs = [scheduled_pair(index, 91) for index in range(30)]
        self.assertEqual(len(set(pairs)), 30)
        self.assertTrue(all(a != b for a, b in pairs))

    def test_physical_source_conditions_are_balanced_by_gas_and_role(self):
        pairs = [scheduled_pair(index, 91) for index in range(30)]
        for role in (0, 1):
            by_gas = []
            for gas_index in range(len(GAS_ORDER)):
                values = sorted(
                    _balanced_desired_ppm(index, 91, gas_index, role, 1.0, 100.0)
                    for index, pair in enumerate(pairs)
                    if pair[role] == gas_index
                )
                by_gas.append(values)
            for values in by_gas[1:]:
                np.testing.assert_allclose(values, by_gas[0])

    def test_phase_a_modes_cover_every_mode_in_each_pair_block(self):
        pairs = [scheduled_pair(index, 91) for index in range(30)]
        modes = [_phase_a_mode(index, *pair) for index, pair in enumerate(pairs)]
        self.assertEqual(
            {mode: modes.count(mode) for mode in set(modes)},
            {
                "near_a": 5, "near_b": 5, "between": 5,
                "offset_a": 5, "offset_b": 5, "jitter_a": 5,
            },
        )

    def test_phase_a_targeted_concentrations_share_distribution_across_gases(self):
        targeted = {gas_index: [] for gas_index in range(len(GAS_ORDER))}
        for index in range(300):
            pair = scheduled_pair(index, 91)
            mode = _phase_a_mode(index, *pair)
            for role, gas_index in enumerate(pair):
                if _phase_a_role_is_target(mode, role):
                    targeted[gas_index].append(
                        _phase_a_desired_ppm(index, 91, gas_index, role, 1.0, 100.0)
                    )
        shortest = min(map(len, targeted.values()))
        reference = targeted[0][:shortest]
        for values in targeted.values():
            self.assertGreaterEqual(len(values), 24)
            np.testing.assert_allclose(values[:shortest], reference)

    def test_physical_labels_mask_no_gas_mixed_and_falling(self):
        ppm = np.zeros((8, len(GAS_ORDER)), np.float32)
        ppm[:, 0] = (0, 1, 3, 8, 8, 6, 4, 2)
        ppm[4, 1] = 7
        _, rs_r0 = simulate_mox(ppm, 2.0)
        labels, state = physical_labels(
            ppm, rs_r0, rate=2.0, presence_threshold_ppm=1.0,
            gas_class_min_ppm=3.0, dominant_threshold=0.90,
            slope_window_s=0.5, rising_rate_per_s=0.03,
            falling_rate_per_s=0.08, memory_start_index=5,
            recovery_residual_threshold=0.0,
        )
        self.assertEqual(state["gas_mask_reason"][0], "NO_GAS")
        self.assertEqual(state["gas_mask_reason"][4], "MIXED")
        self.assertEqual(state["gas_mask_reason"][6], "FALLING")
        self.assertFalse(bool(labels["gas_loss_mask"][0]))
        self.assertFalse(bool(labels["gas_loss_mask"][4]))
        self.assertFalse(bool(labels["gas_loss_mask"][6]))
        self.assertEqual(float(labels["label_presence"][0]), 0.0)
        self.assertGreater(float(labels["label_recovery"][5]), 0.0)

    @unittest.skipIf(torch is None, "PyTorch is not installed in this environment")
    def test_abcd_baseline_and_dual_output_shapes(self):
        baseline = BaselineScentFormer(40, dim=32, heads=4, layers=1, dropout=0.0)
        baseline_out = baseline(torch.zeros(2, 40, 8), torch.zeros(2, 40, dtype=torch.bool))
        self.assertEqual(baseline_out["embedding"].shape, (2, 32))
        self.assertEqual(
            baseline.encode(
                torch.zeros(2, 40, 8), torch.zeros(2, 40, dtype=torch.bool),
            ).shape,
            (2, 32),
        )
        self.assertEqual(baseline_out["gas_logits"].shape, (2, len(GAS_ORDER)))
        self.assertTrue(torch.allclose(
            baseline_out["gas_logits"].softmax(1).sum(1), torch.ones(2), atol=1e-6,
        ))

        dual = DualTimescaleScentFormer(20, dim=32, heads=4, layers=1, dropout=0.0)
        dual_out = dual(
            torch.zeros(2, 20, 8), torch.zeros(2, 20, dtype=torch.bool),
            torch.zeros(2, 20, 8), torch.zeros(2, 20, dtype=torch.bool),
        )
        self.assertEqual(dual_out["embedding"].shape, (2, 32))
        self.assertEqual(
            dual.encode(
                torch.zeros(2, 20, 8), torch.zeros(2, 20, dtype=torch.bool),
                torch.zeros(2, 20, 8), torch.zeros(2, 20, dtype=torch.bool),
            ).shape,
            (2, 32),
        )
        self.assertEqual(dual_out["gas_logits"].shape, (2, len(GAS_ORDER)))
        self.assertEqual(PREFIXES["baseline"], (1, 3, 5, 7, 10, 20))
        self.assertEqual(PREFIXES["dual_timescale"][-2:], (30, 40))

    def test_abcd_recovery_loss_can_be_phase_masked(self):
        outputs = {
            "gas_logits": torch.zeros(2, len(GAS_ORDER)),
            "presence": torch.zeros(2),
            "intensity": torch.zeros(2),
            "recovery": torch.tensor((-10.0, 10.0)),
        }
        batch = {
            "gas_target": torch.zeros(2, dtype=torch.long),
            "gas_mask": torch.ones(2),
            "presence_target": torch.ones(2),
            "intensity_target": torch.zeros(2),
            "recovery_target": torch.tensor((1.0, 0.0)),
            "recovery_mask": torch.zeros(2),
        }
        losses = _abcd_loss(outputs, batch)
        self.assertEqual(float(losses["recovery"]), 0.0)

    def test_binary_metric_marks_single_class_support_not_applicable(self):
        metrics = _abcd_binary_metrics({"tp": 0, "tn": 10, "fp": 0, "fn": 0})
        self.assertFalse(metrics["applicable"])
        self.assertEqual(metrics["positive_n"], 0)
        self.assertEqual(metrics["negative_n"], 10)

    def test_wilson_interval_reflects_sample_count(self):
        small = _wilson_interval(2, 2)
        large = _wilson_interval(100, 100)
        self.assertLess(small[0], large[0])
        self.assertEqual(small[1], 1.0)
        self.assertEqual(large[1], 1.0)
        self.assertIsNone(_wilson_interval(0, 0))

    def test_dual_history_uses_four_sample_lowpass_blocks(self):
        dataset = PhysicalWindows.__new__(PhysicalWindows)
        dataset.history_steps = 16
        dataset.recent_steps = 8
        dataset.downsample = 4
        signal = np.repeat(np.arange(16, dtype=np.float32)[:, None], 8, axis=1)
        recent, recent_mask, history, history_mask = dataset._dual(signal)
        self.assertEqual(recent.shape, (8, 8))
        self.assertFalse(recent_mask.any())
        np.testing.assert_allclose(history[:, 0], [1.5, 5.5, 9.5, 13.5])
        self.assertFalse(history_mask.any())

    @unittest.skipIf(torch is None, "PyTorch is not installed in this environment")
    def test_five_head_forward_shapes(self):
        model = OdorTransformer(max_steps=200, dim=32, heads=4, layers=2)
        self.assertEqual(tuple(model.scent_token.shape), (1, 1, 32))
        outputs = model(torch.zeros(3, 200, 8), torch.zeros(3, 200, dtype=torch.bool))
        self.assertEqual(outputs[0].shape, (3, 6))
        self.assertEqual(outputs[1].shape, (3, 6))
        self.assertEqual(outputs[2].shape, (3,))
        self.assertEqual(outputs[3].shape, (3,))
        self.assertEqual(outputs[4].shape, (3,))

    @unittest.skipIf(torch is None, "PyTorch is not installed in this environment")
    def test_component_losses_sum_to_total(self):
        outputs = (
            torch.zeros(2, 6), torch.zeros(2, 6), torch.ones(2),
            torch.zeros(2), torch.zeros(2),
        )
        presence = torch.tensor([[1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]], dtype=torch.float32)
        fractions = presence.clone()
        targets = (
            presence, fractions, torch.ones(2), torch.tensor([0.0, 1.0]),
            torch.tensor([1.0, 0.0]),
        )
        losses = _loss(outputs, targets)
        component_sum = sum(
            losses[name]
            for name in ("presence", "primary_gas", "intensity", "clean", "recovery")
        )
        self.assertTrue(torch.allclose(losses["total"], component_sum))

    @unittest.skipIf(torch is None, "PyTorch is not installed in this environment")
    def test_primary_gas_class_weights_affect_confused_class_loss(self):
        fraction_logits = torch.tensor([
            [4, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 4, 0],
        ], dtype=torch.float32)
        outputs = (
            torch.zeros(2, 6), fraction_logits, torch.ones(2),
            torch.zeros(2), torch.zeros(2),
        )
        presence = torch.tensor([
            [1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1],
        ], dtype=torch.float32)
        targets = (presence, presence.clone(), torch.ones(2), torch.zeros(2), torch.zeros(2))
        unweighted = _loss(outputs, targets)["primary_gas"]
        weighted = _loss(outputs, targets, torch.tensor([1, 1, 1, 1, 1, 1.4]))["primary_gas"]
        self.assertGreater(float(weighted), float(unweighted))


if __name__ == "__main__":
    unittest.main()

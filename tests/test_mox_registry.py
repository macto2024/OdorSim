"""Tests for calibrated MOX registry behavior."""

import unittest
from pathlib import Path

import yaml

from odor_sim.config.gas_types import (
    EXCEL_VOC_GASES,
    GADEN_GAS_NAMES,
    MOX_SUPPORTED,
    GasType,
    parse_gas_type,
)
from odor_sim.config.liquids import get_liquid_spec
from odor_sim.config.recipes import get_recipe
from odor_sim.runtime.session import resolve_mox_models
from odor_sim.sensors import DEFAULT_MOX_MODELS, MOX_MODELS, MOXSensor
from odor_sim.sensors.mox_pid import get_mox_profile

EXPECTED_FAMILIES = {
    "101", "104", "203", "302", "303", "401", "43_SINTERED",
    "43_UNSINTERED", "500-1", "509", "601", "701", "TGS2600",
    "TGS2602", "TGS2610", "TGS2611", "TGS2620",
}


class MoxRegistryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        registry_path = (
            Path(__file__).resolve().parents[1]
            / "odor_sim"
            / "config"
            / "mox_sensors.yaml"
        )
        cls.registry = yaml.safe_load(registry_path.read_text(encoding="utf-8"))

    def test_only_median_first_nominal_models_are_selectable(self):
        self.assertEqual(set(MOX_MODELS), EXPECTED_FAMILIES)
        for model in MOX_MODELS:
            self.assertEqual(
                get_mox_profile(model)["aggregation_method"],
                "source-selected adaptive late-air/gas-minimum ratio per channel; "
                "median Ra/Rg across channels at each concentration; fit once",
            )

    def test_source_selection_policy(self):
        expected = {
            "TGS2610": "CROSS2024",
            "TGS2611": "CROSS2024",
            "TGS2600": "CROSS2024",
            "TGS2620": "MULTI2023",
            "TGS2602": "MULTI2023",
            "500-1": "MULTI2023",
            "43_SINTERED": "MULTI2023",
            "43_UNSINTERED": "MULTI2023",
            "601": "MULTI2023",
            "701": "MULTI2023",
            "101": "CROSS2024",
            "203": "CROSS2024",
            "302": "CROSS2024",
            "401": "CROSS2024",
            "509": "CROSS2024",
            "104": "PRECOAT2024",
            "303": "PRECOAT2024",
        }
        for model, source in expected.items():
            self.assertEqual(get_mox_profile(model)["source_ids"], [source])

    def test_nominal_profiles_store_six_gases(self):
        gases = {
            "methanol", "ethanol", "acetic_acid",
            "ethyl_acetate", "toluene", "acetone",
        }
        for model in MOX_MODELS:
            self.assertEqual(set(get_mox_profile(model)["sensitivity"]), gases)

    def test_profiles_store_positive_workbook_dynamics(self):
        self.assertEqual(self.registry["schema_version"], 4)
        for model in MOX_MODELS:
            profile = get_mox_profile(model)
            self.assertGreater(profile["tau_rise_s"], 0.0)
            self.assertGreater(profile["tau_decay_s"], 0.0)
            for curve in profile["sensitivity"].values():
                self.assertGreater(curve["coefficient_k"], 0.0)
                self.assertGreater(curve["exponent_n"], 0.0)
                self.assertGreaterEqual(curve["rmse_rs_rair"], 0.0)
                self.assertNotIn("a", curve)
                self.assertNotIn("b", curve)
                self.assertGreater(curve["tau_rise_s"], 0.0)
                self.assertGreater(curve["tau_decay_s"], 0.0)
                self.assertIn("workbook", curve["tau_rise_source"])
                self.assertIn("workbook", curve["tau_decay_source"])

    def test_excel_gases_have_stable_gaden_ids_and_names(self):
        expected = {
            "methanol": 14,
            "acetic_acid": 15,
            "ethyl_acetate": 16,
            "toluene": 17,
        }
        for name, gas_id in expected.items():
            gas_type = parse_gas_type(name)
            self.assertEqual(int(gas_type), gas_id)
            self.assertEqual(GADEN_GAS_NAMES[gas_type], name)
            self.assertIn(gas_type, MOX_SUPPORTED)

    def test_extended_gases_are_selectable_three_cup_liquids(self):
        for gas in (
            "ethanol",
            "methanol",
            "acetone",
            "acetic_acid",
            "ethyl_acetate",
            "toluene",
        ):
            recipe_name = get_liquid_spec(gas).recipe
            profile = get_recipe(recipe_name)
            self.assertEqual(profile.num_sources, 1)
            self.assertEqual(profile.gas_types, list(EXCEL_VOC_GASES))
            self.assertEqual(profile.active_components()[0].gas_type, parse_gas_type(gas))

    def test_all_excel_gases_are_active_at_runtime(self):
        sensor = MOXSensor("601")
        for gas in (
            "methanol",
            "ethanol",
            "acetic_acid",
            "ethyl_acetate",
            "toluene",
            "acetone",
        ):
            self.assertLess(sensor._ideal_rs_r0({gas: 100.0}), 1.0, gas)

    def test_window_ratio_is_median_aggregated_before_inversion(self):
        curve = get_mox_profile("TGS2610")["sensitivity"]["ethyl_acetate"]
        expected_ra_rg = 1.6385485674547215
        self.assertAlmostEqual(
            curve["median_ra_rg_points"]["100"],
            expected_ra_rg,
        )
        self.assertAlmostEqual(
            curve["measured_points"]["100"],
            1.0 / expected_ra_rg,
        )

    def test_nominal_selection_preserves_order(self):
        self.assertEqual(resolve_mox_models(), list(DEFAULT_MOX_MODELS))
        self.assertEqual(resolve_mox_models("all"), list(MOX_MODELS))
        self.assertEqual(resolve_mox_models(["all"]), list(MOX_MODELS))
        with self.assertRaises(ValueError):
            resolve_mox_models(["all", "101"])
        self.assertEqual(
            resolve_mox_models(["601", "TGS2610", "601"]),
            ["601", "TGS2610"],
        )

    def test_range_handling(self):
        sensor = MOXSensor("601")
        clean = sensor._ideal_rs_r0({"ethanol": 0.0})
        half = sensor._ideal_rs_r0({"ethanol": 0.5})
        one = sensor._ideal_rs_r0({"ethanol": 1.0})
        self.assertEqual(clean, 1.0)
        self.assertGreater(half, one)
        self.assertLess(half, clean)
        # Runtime sensitivity is not silently clamped at the workbook limit;
        # physical dataset validation is responsible for flagging out-of-range ppm.
        self.assertLess(
            sensor._ideal_rs_r0({"acetone": 1000.0}),
            sensor._ideal_rs_r0({"acetone": 100.0}),
        )

    def test_runtime_uses_conductance_power_law_and_additive_mixture(self):
        sensor = MOXSensor("TGS2610")
        ethanol = get_mox_profile("TGS2610")["sensitivity"]["ethanol"]
        acetone = get_mox_profile("TGS2610")["sensitivity"]["acetone"]
        ethanol_excess = ethanol["coefficient_k"] * 10.0 ** ethanol["exponent_n"]
        acetone_excess = acetone["coefficient_k"] * 10.0 ** acetone["exponent_n"]
        self.assertAlmostEqual(
            sensor._ideal_rs_r0({"ethanol": 10.0}),
            1.0 / (1.0 + ethanol_excess),
        )
        self.assertAlmostEqual(
            sensor._ideal_rs_r0({"ethanol": 10.0, "acetone": 10.0}),
            1.0 / (1.0 + ethanol_excess + acetone_excess),
        )

    def test_unsupported_gas_is_ignored(self):
        self.assertEqual(MOXSensor("601")._ideal_rs_r0({"methane": 100.0}), 1.0)

    def test_default_rl_and_voltage_output(self):
        sensor = MOXSensor("601")
        self.assertEqual(sensor.load_resistance, 0.25 * sensor.R0)
        self.assertAlmostEqual(sensor.baseline_voltage(), 1.0)
        sensor.update({})
        sensor.update({"ethanol": 100.0})
        self.assertGreater(sensor.voltage, sensor.baseline_voltage())

    def test_runtime_uses_gas_specific_rise_and_recovery(self):
        sensor = MOXSensor("601", rate=1.0)
        curve = get_mox_profile("601")["sensitivity"]["toluene"]
        sensor.update({})
        ideal = sensor._ideal_rs_r0({"toluene": 100.0})
        rise_alpha = 1.0 / (curve["tau_rise_s"] + 1.0)
        sensor.update({"toluene": 100.0})
        self.assertAlmostEqual(
            sensor.rs_r0,
            rise_alpha * ideal + (1.0 - rise_alpha),
        )
        previous = sensor.rs_r0
        decay_alpha = 1.0 / (curve["tau_decay_s"] + 1.0)
        sensor.update({})
        self.assertAlmostEqual(
            sensor.rs_r0,
            decay_alpha + (1.0 - decay_alpha) * previous,
        )


if __name__ == "__main__":
    unittest.main()

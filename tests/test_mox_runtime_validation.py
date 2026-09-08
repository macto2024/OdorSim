"""Regression tests for the executable MOX runtime validator."""

import unittest

from odor_sim.sensors.mox_pid import MOXSensor
from tools.validate_mox_runtime import validate_case


class MoxRuntimeValidationTests(unittest.TestCase):
    def test_detectable_step_reproduces_curve_and_dynamics(self):
        result, trace = validate_case(
            "TGS2610", "acetone", 10.0, retain_trace=True
        )
        self.assertTrue(result["response_detectable"])
        self.assertTrue(result["all_pass"])
        self.assertLess(result["runtime_ideal_absolute_error"], 1e-12)
        self.assertLess(result["rise_tau_relative_error"], 0.01)
        self.assertLess(result["recovery_tau_relative_error"], 0.01)
        self.assertTrue(trace)

    def test_one_ppm_response_is_not_clamped_to_clean_air(self):
        result, trace = validate_case(
            "TGS2620", "methanol", 1.0, retain_trace=True
        )
        self.assertTrue(result["response_detectable"])
        self.assertLess(result["runtime_ideal_rs_rair"], 1.0)
        self.assertIsNotNone(result["rise_tau_measured_s"])
        self.assertIsNotNone(result["recovery_tau_measured_s"])
        self.assertTrue(result["all_pass"])
        self.assertTrue(trace)

    def test_out_of_range_ppm_is_not_clamped_to_calibration_boundary(self):
        sensor = MOXSensor("TGS2610")
        at_100 = sensor._ideal_rs_r0({"acetone": 100.0})
        at_1000 = sensor._ideal_rs_r0({"acetone": 1000.0})
        self.assertLess(at_1000, at_100)


if __name__ == "__main__":
    unittest.main()

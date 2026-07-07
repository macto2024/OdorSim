"""Phase 4c test: shared MOX/PID e-nose sensor model (no ROS / no robosuite).

Runnable two ways:
    python tests/test_phase4_sensor.py     # prints PASS/FAIL for each check
    pytest tests/test_phase4_sensor.py     # standard test discovery

Validates the pure-Python re-implementation of GADEN's fake_gas_sensor MOX/PID
math, the voltage divider, and the continuous vs sampling state machine over a
shared ppm(t) trace.
"""

from __future__ import annotations

import sys

from odor_sim.config.gas_types import GasType
from odor_sim.sensors.mox_pid import (
    MOXSensor,
    PIDSensor,
    SamplingEnose,
    synthesize_continuous,
    synthesize_sampling,
)


def test_mox_baseline_first_reading():
    s = MOXSensor("TGS2620", rate=20.0)
    rs = s.update({})  # first reading latches baseline, no response
    assert abs(rs - s.sensitivity_air * s.R0) < 1e-6
    # baseline voltage is the low end of the range
    assert 0.0 < s.voltage < s.vcc


def test_mox_rises_and_saturates_toward_steady_state():
    s = MOXSensor("TGS2620", rate=20.0)
    s.update({})  # baseline
    base_v = s.voltage
    # constant ethanol exposure: resistance should drop, voltage should rise
    v_prev = base_v
    for _ in range(400):  # ~20 s at 20 Hz -> well past tau
        s.update({"ethanol": 200.0})
    v_final = s.voltage
    assert v_final > base_v, "voltage should rise above baseline under gas"
    # resistance dropped below the clean-air resistance
    assert s.resistance < s.sensitivity_air * s.R0
    # transient is monotone-ish on the way up (first step already increases)
    s.reset()
    s.update({})
    s.update({"ethanol": 200.0})
    assert s.voltage >= v_prev - 1e-9 or s.voltage > base_v


def test_mox_decays_back_toward_baseline_when_gas_removed():
    s = MOXSensor("TGS2620", rate=20.0)
    s.update({})
    for _ in range(400):
        s.update({"ethanol": 200.0})
    v_exposed = s.voltage
    for _ in range(2000):  # long purge in clean air
        s.update({})
    v_purged = s.voltage
    assert v_purged < v_exposed
    # returns close to the clean-air baseline
    assert abs(v_purged - s.baseline_voltage()) < 0.05 * s.vcc


def test_mox_mixture_stronger_than_single_component():
    single = MOXSensor("TGS2620", rate=20.0)
    mix = MOXSensor("TGS2620", rate=20.0)
    single.update({})
    mix.update({})
    for _ in range(400):
        single.update({"ethanol": 100.0})
        mix.update({"ethanol": 100.0, "acetone": 100.0})
    # more total gas -> lower resistance -> higher voltage
    assert mix.voltage > single.voltage


def test_pid_weighted_sum_and_blindness():
    pid = PIDSensor(use_correction=True)
    # ethanol correction factor 10.47 -> attenuated
    v_eth = pid.update({"ethanol": 100.0})
    assert abs(v_eth - 100.0 / 10.47) < 1e-6
    # methane/hydrogen are invisible to the PID (factor 0)
    assert pid.update({"methane": 500.0}) == 0.0
    assert pid.update({"hydrogen": 500.0}) == 0.0
    # without correction it's a plain sum
    plain = PIDSensor(use_correction=False)
    assert abs(plain.update({"ethanol": 30.0, "acetone": 20.0}) - 50.0) < 1e-6


def test_continuous_vs_sampling_from_same_ppm_series():
    # Build a ppm(t) trace: clean, then a burst of ethanol, then clean again.
    n = 600
    ppm_series = []
    for i in range(n):
        ppm_series.append({"ethanol": 150.0} if 100 <= i < 400 else {"ethanol": 0.0})

    # enose_state schedule: idle, then sample during part of the burst, then purge
    states = []
    for i in range(n):
        if 150 <= i < 350:
            states.append(SamplingEnose.SAMPLE)
        elif i >= 350:
            states.append(SamplingEnose.FILTER)
        else:
            states.append(SamplingEnose.IDLE)

    cont = synthesize_continuous(ppm_series, model="TGS2620", rate=20.0)
    samp, windows = synthesize_sampling(ppm_series, states, model="TGS2620", rate=20.0)

    assert len(cont) == n and len(samp) == n

    base = MOXSensor("TGS2620").baseline_voltage()
    # Continuous sensor responds during the burst even before sampling starts.
    assert cont[120] > base + 1e-3
    # Sampling sensor is still at baseline at step 120 (idle, valve closed).
    assert abs(samp[120] - base) < 1e-3
    # During the sample window the gated sensor rises above baseline.
    assert samp[300] > base + 1e-3
    # Exactly one completed sample window, labeled ethanol.
    assert len(windows) == 1
    w = windows[0]
    assert w.trigger_step == 150
    assert w.duration_steps == 200
    assert w.odor_class == GasType.ethanol
    assert len(w.voltage_trace) == 200


def test_sampling_window_odor_class_picks_dominant_gas():
    n = 200
    ppm_series = [{"ethanol": 10.0, "acetone": 120.0} for _ in range(n)]
    states = [SamplingEnose.SAMPLE] * n
    _, windows = synthesize_sampling(ppm_series, states, model="TGS2620", rate=20.0)
    assert len(windows) == 1
    assert windows[0].odor_class == GasType.acetone


def _run_all():
    tests = [
        test_mox_baseline_first_reading,
        test_mox_rises_and_saturates_toward_steady_state,
        test_mox_decays_back_toward_baseline_when_gas_removed,
        test_mox_mixture_stronger_than_single_component,
        test_pid_weighted_sum_and_blindness,
        test_continuous_vs_sampling_from_same_ppm_series,
        test_sampling_window_odor_class_picks_dominant_gas,
    ]
    failures = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except Exception as e:  # noqa: BLE001
            failures += 1
            print(f"FAIL  {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failures}/{len(tests)} checks passed")
    return failures


if __name__ == "__main__":
    sys.exit(1 if _run_all() else 0)

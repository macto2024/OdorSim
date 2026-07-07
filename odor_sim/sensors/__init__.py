"""odor_sim.sensors: shared MOX/PID e-nose model (ppm -> resistance/voltage).

Pure-Python re-implementation of GADEN's ``fake_gas_sensor.cpp`` plus a
voltage divider and an active-sampling state machine. Used both offline
(feature synthesis over stored ppm(t)) and online (streaming at eval).
"""

from odor_sim.sensors.mox_pid import (
    MOX_MODELS,
    ContinuousEnose,
    MOXSensor,
    PIDSensor,
    SampleWindow,
    SamplingEnose,
    synthesize_continuous,
    synthesize_sampling,
)

__all__ = [
    "MOX_MODELS",
    "MOXSensor",
    "PIDSensor",
    "ContinuousEnose",
    "SamplingEnose",
    "SampleWindow",
    "synthesize_continuous",
    "synthesize_sampling",
]

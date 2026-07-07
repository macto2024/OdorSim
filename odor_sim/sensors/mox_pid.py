"""Shared MOX / PID e-nose sensor model (Phase 4c).

Pure-Python re-implementation of GADEN's ``fake_gas_sensor.cpp`` MOX and PID
models, plus a load-resistor voltage divider and an active-sampling state
machine. Consumes a per-gas ppm reading (or a stored ppm(t) series) and
produces sensor resistance / voltage. Used two ways:

  * **offline feature synthesis** (Phase 5): run over a recorded ppm(t) series
    to derive continuous and sampling voltage streams,
  * **online eval** (Phase 6): stream ppm each control step to feed a policy.

MOX model (per gas ``g`` at concentration ``c`` ppm)::

    RS/R0 = A_g * c ^ B_g                       (line in loglog scale)
    RS/R0 clamped to <= Sensitivity_Air (baseline in clean air)
    resistance_variation = sum_g (Sensitivity_Air - RS/R0_g)   (mixture)
    RS/R0 = Sensitivity_Air - resistance_variation  (>= 0.01)
    transient: low-pass filter with tau_rise / tau_decay
    Rs = (RS/R0) * R0                            (ohms)

Voltage divider (our addition; GADEN publishes raw ohms)::

    Vout = Vcc * RL / (Rs + RL)                  (Vcc = 5 V)

PID model: weighted ppm sum (optional per-gas correction factors).

Sensor characterization constants are copied verbatim from
``gaden/simulated_gas_sensor/src/fake_gas_sensor.h`` (TGS2620/2600/2611/2610/
2612 and the miniRAE-Lite PID). Gas indices match GADEN's ``GasType`` enum
(ethanol=0 ... acetone=6), so a :class:`~odor_sim.config.gas_types.GasType`
integer *is* the row index into the sensitivity tables.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from odor_sim.config.gas_types import GasType, parse_gas_type

# ---------------------------------------------------------------------------- #
# MOX characterization (from fake_gas_sensor.h). Row order per model:
#   ethanol, methane, hydrogen, propanol, chlorine, fluorine, acetone
# ---------------------------------------------------------------------------- #
MOX_MODELS = ("TGS2620", "TGS2600", "TGS2611", "TGS2610", "TGS2612")

# Reference resistance R0 [ohms]
_R0 = {
    "TGS2620": 3000.0,
    "TGS2600": 50000.0,
    "TGS2611": 3740.0,
    "TGS2610": 3740.0,
    "TGS2612": 4500.0,
}

# RS/R0 in clean air (baseline)
_SENSITIVITY_AIR = {
    "TGS2620": 21.0,
    "TGS2600": 1.0,
    "TGS2611": 8.8,
    "TGS2610": 10.3,
    "TGS2612": 19.5,
}

# (tau_rise, tau_decay) seconds; identical across gases in the stock model
_TAU = {
    "TGS2620": (2.96, 15.71),
    "TGS2600": (4.8, 18.75),
    "TGS2611": (3.44, 6.35),
    "TGS2610": (3.44, 6.35),
    "TGS2612": (3.44, 6.35),
}

# RS/R0 = A * conc^B ; (A, B) per gas index 0..6
_SENSITIVITY_LINELOGLOG = {
    "TGS2620": [
        (62.32, -0.7155),  # ethanol
        (120.6, -0.4877),  # methane
        (24.45, -0.5546),  # hydrogen
        (120.6, -0.4877),  # propanol
        (120.6, -0.4877),  # chlorine
        (120.6, -0.4877),  # fluorine
        (120.6, -0.4877),  # acetone
    ],
    "TGS2600": [
        (0.6796, -0.3196),
        (1.018, -0.07284),
        (0.6821, -0.3532),
        (1.018, -0.07284),
        (1.018, -0.07284),
        (1.018, -0.07284),
        (1.018, -0.07284),
    ],
    "TGS2611": [
        (51.11, -0.3658),
        (38.46, -0.4289),
        (41.3, -0.3614),
        (38.46, -0.4289),
        (38.46, -0.4289),
        (38.46, -0.4289),
        (38.46, -0.4289),
    ],
    "TGS2610": [
        (106.1, -0.5008),
        (63.91, -0.5372),
        (66.78, -0.4888),
        (63.91, -0.5372),
        (63.91, -0.5372),
        (63.91, -0.5372),
        (63.91, -0.5372),
    ],
    "TGS2612": [
        (31.35, -0.09115),
        (146.2, -0.5916),
        (19.5, 0.0),
        (146.2, -0.5916),
        (146.2, -0.5916),
        (146.2, -0.5916),
        (146.2, -0.5916),
    ],
}

# PID correction factors (index 0..6). 0.0 means the PID is blind to that gas.
_PID_CORRECTION_FACTORS = [10.47, 0.0, 0.0, 2.7, 1.0, 0.0, 1.4]


def _normalize_ppm(ppm_by_gas: dict) -> dict:
    """Map arbitrary gas keys (name/int/GasType) to GasType, dropping zeros."""
    out: dict[GasType, float] = {}
    for key, value in ppm_by_gas.items():
        gt = parse_gas_type(key)
        out[gt] = out.get(gt, 0.0) + float(value)
    return out


class MOXSensor:
    """A single MOX gas sensor with GADEN's loglog sensitivity + transient.

    Args:
        model: one of :data:`MOX_MODELS`.
        rate: control frequency (Hz); sets the transient low-pass alpha exactly
            as the stock node's ``node_rate``.
        load_resistance: RL [ohms] for the voltage divider. Defaults to the
            model's R0, giving a monotone 0..Vcc reading that rises with
            concentration. A modeling parameter (GADEN itself outputs raw ohms).
        vcc: supply voltage for the divider (V).
    """

    def __init__(
        self,
        model: str = "TGS2620",
        rate: float = 20.0,
        load_resistance: "float | None" = None,
        vcc: float = 5.0,
    ):
        if model not in MOX_MODELS:
            raise ValueError(f"Unknown MOX model {model!r}; choose from {MOX_MODELS}")
        self.model = model
        self.rate = float(rate)
        self.R0 = _R0[model]
        self.sensitivity_air = _SENSITIVITY_AIR[model]
        self.tau_rise, self.tau_decay = _TAU[model]
        self._sens = _SENSITIVITY_LINELOGLOG[model]
        self.load_resistance = float(load_resistance) if load_resistance is not None else self.R0
        self.vcc = float(vcc)
        self.reset()

    def reset(self) -> None:
        """Reset to the clean-air baseline (RS/R0 = Sensitivity_Air)."""
        self._rs_r0 = self.sensitivity_air
        self._prev = self.sensitivity_air
        self._first = True

    # -- ideal (steady-state) response, no transient -- #
    def _ideal_rs_r0(self, ppm_by_gas: dict) -> float:
        resistance_variation = 0.0
        for gt, conc in _normalize_ppm(ppm_by_gas).items():
            idx = int(gt)
            if not 0 <= idx <= 6:
                raise ValueError(f"MOX sensor is not configured for gas {gt.name!r}")
            if conc <= 0.0:
                continue
            a, b = self._sens[idx]
            rs_r0 = a * (conc ** b)
            if rs_r0 > self.sensitivity_air:
                rs_r0 = self.sensitivity_air
            resistance_variation += self.sensitivity_air - rs_r0
        rs_r0 = self.sensitivity_air - resistance_variation
        return max(rs_r0, 0.01)

    def update(self, ppm_by_gas: dict) -> float:
        """Advance one control step; returns sensor resistance Rs [ohms].

        Mirrors ``simulate_mox_as_line_loglog``: the first call latches the
        baseline (no gas response), subsequent calls apply the sensitivity
        curve then a rise/decay low-pass filter.
        """
        if self._first:
            self._rs_r0 = self.sensitivity_air
            self._prev = self._rs_r0
            self._first = False
            return self.resistance

        ideal = self._ideal_rs_r0(ppm_by_gas)
        tau = self.tau_rise if ideal < self._prev else self.tau_decay
        dt = 1.0 / self.rate
        alpha = dt / (tau + dt)
        self._rs_r0 = alpha * ideal + (1.0 - alpha) * self._prev
        self._prev = self._rs_r0
        return self.resistance

    @property
    def rs_r0(self) -> float:
        """Current filtered RS/R0 ratio."""
        return self._rs_r0

    @property
    def resistance(self) -> float:
        """Current sensor resistance Rs [ohms]."""
        return self._rs_r0 * self.R0

    @property
    def voltage(self) -> float:
        """Voltage-divider reading Vout = Vcc * RL / (Rs + RL) [volts]."""
        rs = self.resistance
        return self.vcc * self.load_resistance / (rs + self.load_resistance)

    def baseline_voltage(self) -> float:
        """Voltage at the clean-air baseline (before any exposure)."""
        rs = self.sensitivity_air * self.R0
        return self.vcc * self.load_resistance / (rs + self.load_resistance)


class PIDSensor:
    """Photo-ionization detector: weighted ppm sum (no transient).

    Args:
        use_correction: divide each gas by its PID correction factor (gases
            with factor 0, e.g. methane/hydrogen, are invisible to the PID).
        vcc / full_scale_ppm: optional linear ppm->voltage mapping for
            ``voltage`` (clamped to ``vcc``).
    """

    def __init__(self, use_correction: bool = True, vcc: float = 5.0, full_scale_ppm: float = 500.0):
        self.use_correction = bool(use_correction)
        self.vcc = float(vcc)
        self.full_scale_ppm = float(full_scale_ppm)
        self._value = 0.0

    def reset(self) -> None:
        self._value = 0.0

    def update(self, ppm_by_gas: dict) -> float:
        """Return accumulated (corrected) ppm for the current reading."""
        total = 0.0
        for gt, conc in _normalize_ppm(ppm_by_gas).items():
            if conc <= 0.0:
                continue
            if self.use_correction:
                idx = int(gt)
                if not 0 <= idx <= 6:
                    raise ValueError(f"PID is not configured for gas {gt.name!r}")
                factor = _PID_CORRECTION_FACTORS[idx]
                if factor != 0.0:
                    total += conc / factor
            else:
                total += conc
        self._value = total
        return total

    @property
    def value(self) -> float:
        """Current PID reading in (corrected) ppm."""
        return self._value

    @property
    def voltage(self) -> float:
        v = self.vcc * self._value / self.full_scale_ppm
        return min(v, self.vcc)


@dataclass
class SampleWindow:
    """A completed active-sampling window (produced by :class:`SamplingEnose`)."""

    trigger_step: int
    duration_steps: int
    voltage_trace: list = field(default_factory=list)
    ppm_trace: list = field(default_factory=list)  # list of {gas_name: ppm}
    odor_class: "GasType | None" = None  # ground-truth dominant gas over window

    @property
    def duration_s(self) -> float:
        return self._duration_s

    def _finalize(self, rate: float, dominant: "GasType | None") -> None:
        self._duration_s = self.duration_steps / rate
        self.odor_class = dominant


class ContinuousEnose:
    """Always-exposed e-nose: applies the dynamic model every step.

    Thin wrapper over a MOX (or PID) sensor that feeds it the live ppm each
    control step and exposes the streaming voltage.
    """

    def __init__(self, sensor: "MOXSensor | PIDSensor | None" = None, **mox_kwargs):
        self.sensor = sensor if sensor is not None else MOXSensor(**mox_kwargs)

    def reset(self) -> None:
        self.sensor.reset()

    def update(self, ppm_by_gas: dict) -> float:
        """Advance one step; returns the streaming voltage."""
        self.sensor.update(ppm_by_gas)
        return self.sensor.voltage


class SamplingEnose:
    """Valve-gated e-nose driven by the ternary ``enose_state`` action.

    The sensor only sees the plume while ``enose_state == 1`` (sample); in
    ``idle(0)`` and ``filter(-1)`` it is exposed to clean air (ppm = 0) so its
    dynamic response decays/purges toward baseline. Completed sample windows are
    collected with their voltage/ppm traces and a ground-truth odor-class label
    (the gas with the highest mean ppm over the window).

    States: ``1`` sample, ``0`` idle, ``-1`` filter/purge.
    """

    SAMPLE = 1
    IDLE = 0
    FILTER = -1

    def __init__(self, sensor: "MOXSensor | None" = None, **mox_kwargs):
        self.sensor = sensor if sensor is not None else MOXSensor(**mox_kwargs)
        self.reset()

    def reset(self) -> None:
        self.sensor.reset()
        self._step = -1
        self._in_sample = False
        self._cur: "SampleWindow | None" = None
        self.windows: list[SampleWindow] = []

    def update(self, ppm_by_gas: dict, enose_state: int) -> dict:
        """Advance one step under ``enose_state``.

        Returns a dict: ``voltage``, ``sampling_active`` (bool), and
        ``completed_window`` (a :class:`SampleWindow` on the step a sample
        window closes, else None).
        """
        self._step += 1
        sampling = int(enose_state) == self.SAMPLE
        exposure = ppm_by_gas if sampling else {}
        self.sensor.update(exposure)
        voltage = self.sensor.voltage

        completed = None
        if sampling:
            if not self._in_sample:
                self._cur = SampleWindow(trigger_step=self._step, duration_steps=0)
                self._in_sample = True
            self._cur.duration_steps += 1
            self._cur.voltage_trace.append(voltage)
            self._cur.ppm_trace.append(dict(ppm_by_gas))
        else:
            if self._in_sample and self._cur is not None:
                completed = self._close_window()

        return {
            "voltage": voltage,
            "sampling_active": sampling,
            "completed_window": completed,
        }

    def finish(self) -> "SampleWindow | None":
        """Close any open sample window (call at episode end)."""
        if self._in_sample and self._cur is not None:
            return self._close_window()
        return None

    def _close_window(self) -> SampleWindow:
        window = self._cur
        window._finalize(self.sensor.rate, _dominant_gas(window.ppm_trace))
        self.windows.append(window)
        self._in_sample = False
        self._cur = None
        return window


def _dominant_gas(ppm_trace: list) -> "GasType | None":
    """Gas with the highest mean ppm over a window (ground-truth class label)."""
    totals: dict[GasType, float] = {}
    for frame in ppm_trace:
        for gt, conc in _normalize_ppm(frame).items():
            totals[gt] = totals.get(gt, 0.0) + conc
    if not totals:
        return None
    best = max(totals.items(), key=lambda kv: kv[1])
    return best[0] if best[1] > 0.0 else None


def synthesize_continuous(ppm_series, sensor: "MOXSensor | None" = None, **mox_kwargs) -> list:
    """Offline: continuous voltage stream from a stored ppm(t) series.

    Args:
        ppm_series: iterable of ``{gas_name: ppm}`` per step.
    """
    enose = ContinuousEnose(sensor=sensor, **mox_kwargs)
    enose.reset()
    return [enose.update(frame) for frame in ppm_series]


def synthesize_sampling(ppm_series, enose_states, sensor: "MOXSensor | None" = None, **mox_kwargs):
    """Offline: sampling voltage stream + completed windows from ppm(t).

    Args:
        ppm_series: iterable of ``{gas_name: ppm}`` per step.
        enose_states: iterable of ternary ``enose_state`` tokens per step.

    Returns:
        (voltages, windows) where voltages is a per-step list and windows is the
        list of :class:`SampleWindow` closed during the series.
    """
    enose = SamplingEnose(sensor=sensor, **mox_kwargs)
    enose.reset()
    voltages = []
    for frame, state in zip(ppm_series, enose_states):
        voltages.append(enose.update(frame, state)["voltage"])
    enose.finish()
    return voltages, list(enose.windows)

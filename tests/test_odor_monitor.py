"""Unit tests for OdorMonitor (Phase 4.5f) — no ROS, no robosuite."""

from __future__ import annotations

import sys

import pytest

from odor_sim.runtime.odor_monitor import OdorMonitor, parse_odor_monitor_spec


def test_parse_odor_monitor_spec():
    assert parse_odor_monitor_spec(False) == {"enabled": False}
    assert parse_odor_monitor_spec(True) == {"enabled": True, "log": True, "plot": True}
    assert parse_odor_monitor_spec("log") == {"enabled": True, "log": True, "plot": False}
    assert parse_odor_monitor_spec("plot") == {"enabled": True, "log": False, "plot": True}
    d = parse_odor_monitor_spec({"log": True, "plot": False, "history_s": 30})
    assert d == {"enabled": True, "log": True, "plot": False, "history_s": 30}
    with pytest.raises(ValueError):
        parse_odor_monitor_spec("invalid")


def test_record_accumulates_history():
    mon = OdorMonitor(["ethanol", "acetone"], log=False, plot=False)
    mon.record(0.05, {"ethanol": 1.0, "acetone": 2.0})
    mon.record(0.10, {"ethanol": 3.0, "acetone": 4.0})
    h = mon.history
    assert h["sim_time"].tolist() == [0.05, 0.10]
    assert h["ethanol"].tolist() == [1.0, 3.0]
    assert h["acetone"].tolist() == [2.0, 4.0]


def test_reset_clears_history():
    mon = OdorMonitor(["ethanol"], log=False, plot=False)
    mon.record(1.0, {"ethanol": 5.0})
    mon.reset()
    assert len(mon.history["sim_time"]) == 0


def test_log_throttle(capsys):
    mon = OdorMonitor(["ethanol"], log=True, plot=False, log_every_n=2)
    mon.record(0.1, {"ethanol": 1.0})
    mon.record(0.2, {"ethanol": 2.0})
    mon.record(0.3, {"ethanol": 3.0})
    out = capsys.readouterr().out
    assert out.count("[odor]") == 1
    assert "t=0.20s" in out
    assert "ethanol=2.00" in out


def test_trim_history_by_sim_time():
    mon = OdorMonitor(["ethanol"], log=False, plot=False, history_s=0.15)
    for i in range(10):
        mon.record(i * 0.05, {"ethanol": float(i)})
    t = mon.history["sim_time"]
    # last sample t=0.45; window keeps t >= 0.30
    assert t.min() >= 0.30 - 1e-6
    assert len(t) == 4


def test_missing_gas_defaults_to_zero():
    mon = OdorMonitor(["ethanol", "acetone"], log=False, plot=False)
    mon.record(0.0, {"ethanol": 7.0})
    assert mon.history["acetone"][0] == 0.0


def _run_all():
    import pytest

    return pytest.main([__file__, "-q"])


if __name__ == "__main__":
    sys.exit(_run_all())

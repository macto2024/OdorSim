"""Live GADEN ppm and eight-channel MOX plots for the two-source RViz demo."""

from __future__ import annotations

import argparse
import time

import numpy as np

from odor_sim.bridge.gaden_bridge import GadenBridge
from odor_sim.odor_encoder.dataset import GAS_ORDER, SENSOR_ORDER
from odor_sim.runtime.odor_monitor import OdorMonitor
from odor_sim.runtime.sensor_monitor import SensorMonitor
from odor_sim.sensors.mox_pid import MOXSensor


def run(args: argparse.Namespace) -> int:
    sources = np.asarray(args.source_position, dtype=float)
    sensor_position = np.asarray(args.sensor_position, dtype=float)
    if sources.shape != (2, 3):
        raise SystemExit("exactly two --source-position x y z values are required")

    bridge = GadenBridge(node_name="two_source_rviz_live_monitor")
    odor = OdorMonitor(
        [args.gas_a, args.gas_b], log=True, plot=not args.no_plots,
        history_s=args.history_s, log_every_n=max(1, round(args.rate_hz)),
    )
    sensor_plot = SensorMonitor(
        list(SENSOR_ORDER), log=True, plot=not args.no_plots,
        history_s=args.history_s, log_every_n=max(1, round(args.rate_hz)),
    )
    sensors = [MOXSensor(model=model, rate=args.rate_hz) for model in SENSOR_ORDER]
    try:
        if not bridge.wait_for_server(args.wait_timeout):
            raise RuntimeError("GADEN /odor_value service unavailable")
        time.sleep(args.startup_delay_s)
        period = 1.0 / args.rate_hz
        bridge.publish_source_poses(sources)
        reading = bridge.query_ppm(sensor_position[None, :])[0]
        odor.record(0.0, reading)
        initial_voltages = {}
        for model, sensor in zip(SENSOR_ORDER, sensors):
            sensor.update(reading)
            initial_voltages[model] = sensor.voltage
        sensor_plot.record(0.0, initial_voltages)
        next_sample = time.monotonic()
        step_index = 0
        while True:
            bridge.publish_source_poses(sources)
            if not bridge.step(1):
                raise TimeoutError("GADEN did not confirm the visualization step")
            step_index += 1
            reading = bridge.query_ppm(sensor_position[None, :])[0]
            sim_time = step_index / args.rate_hz
            odor.record(sim_time, reading)
            voltages = {}
            for model, sensor in zip(SENSOR_ORDER, sensors):
                sensor.update(reading)
                voltages[model] = sensor.voltage
            sensor_plot.record(sim_time, voltages)
            next_sample += period
            time.sleep(max(0.0, next_sample - time.monotonic()))
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        # The parent stops the isolated ROS context when RViz closes.
        if "context is invalid" in str(exc) or "shutdown" in str(exc).lower():
            return 0
        raise
    finally:
        odor.close()
        sensor_plot.close()
        bridge.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gas-a", choices=GAS_ORDER, required=True)
    parser.add_argument("--gas-b", choices=GAS_ORDER, required=True)
    parser.add_argument(
        "--source-position", action="append", nargs=3, type=float, default=[],
        metavar=("X", "Y", "Z"),
    )
    parser.add_argument("--sensor-position", nargs=3, type=float, required=True)
    parser.add_argument("--rate-hz", type=float, default=20.0)
    parser.add_argument("--history-s", type=float, default=120.0)
    parser.add_argument("--wait-timeout", type=float, default=60.0)
    parser.add_argument("--startup-delay-s", type=float, default=3.0)
    parser.add_argument("--no-plots", action="store_true", help=argparse.SUPPRESS)
    return parser


if __name__ == "__main__":
    raise SystemExit(run(build_parser().parse_args()))

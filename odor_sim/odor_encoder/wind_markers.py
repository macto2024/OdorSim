"""Publish a simple room and the current uniform wind field for RViz."""

from __future__ import annotations

import argparse
from pathlib import Path
import time

import numpy as np


def run(args: argparse.Namespace) -> int:
    import rclpy
    from geometry_msgs.msg import Point
    from rclpy.executors import ExternalShutdownException
    from rclpy.qos import DurabilityPolicy, QoSProfile
    from visualization_msgs.msg import Marker, MarkerArray

    vectors = np.loadtxt(Path(args.wind_file), delimiter=",", ndmin=2)
    rclpy.init()
    node = rclpy.create_node("odor_sim_wind_markers")
    qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
    room_pub = node.create_publisher(MarkerArray, "/odor_sim/room_visualization", qos)
    wind_pub = node.create_publisher(MarkerArray, "/odor_sim/wind_visualization", qos)

    def cube(marker_id, position, scale, color):
        marker = Marker()
        marker.header.frame_id = "map"
        marker.ns = "room"
        marker.id = marker_id
        marker.type = Marker.CUBE
        marker.action = Marker.ADD
        marker.pose.position.x, marker.pose.position.y, marker.pose.position.z = position
        marker.pose.orientation.w = 1.0
        marker.scale.x, marker.scale.y, marker.scale.z = scale
        marker.color.r, marker.color.g, marker.color.b, marker.color.a = color
        return marker

    room = MarkerArray()
    room.markers = [
        cube(0, (5.0, 3.0, -0.025), (10.4, 6.4, 0.05), (0.30, 0.34, 0.38, 0.75)),
        cube(1, (-0.175, 3.0, 1.25), (0.05, 6.4, 2.5), (0.55, 0.58, 0.62, 0.30)),
        cube(2, (10.175, 3.0, 1.25), (0.05, 6.4, 2.5), (0.55, 0.58, 0.62, 0.30)),
        cube(3, (5.0, -0.175, 1.25), (10.4, 0.05, 2.5), (0.55, 0.58, 0.62, 0.30)),
        cube(4, (5.0, 6.175, 1.25), (10.4, 0.05, 2.5), (0.55, 0.58, 0.62, 0.30)),
    ]
    if args.tabletop:
        room.markers.append(
            cube(
                5,
                (args.table_center_x, args.table_center_y, args.tabletop_z - 0.025),
                (args.table_size_x, args.table_size_y, 0.05),
                (0.32, 0.34, 0.36, 0.85),
            )
        )
        leg_offset_x = 0.5 * args.table_size_x - 0.06
        leg_offset_y = 0.5 * args.table_size_y - 0.06
        for marker_id, (dx, dy) in enumerate(
            ((-leg_offset_x, -leg_offset_y), (-leg_offset_x, leg_offset_y),
             (leg_offset_x, -leg_offset_y), (leg_offset_x, leg_offset_y)),
            start=6,
        ):
            room.markers.append(cube(
                marker_id,
                (args.table_center_x + dx, args.table_center_y + dy, args.tabletop_z / 2),
                (0.06, 0.06, args.tabletop_z),
                (0.25, 0.27, 0.29, 0.9),
            ))

    if args.sensor_position:
        sensor = Marker()
        sensor.header.frame_id = "map"
        sensor.ns = "enose"
        sensor.id = 30
        sensor.type = Marker.SPHERE
        sensor.action = Marker.ADD
        sensor.pose.position.x, sensor.pose.position.y, sensor.pose.position.z = (
            float(value) for value in args.sensor_position
        )
        sensor.pose.orientation.w = 1.0
        sensor.scale.x = sensor.scale.y = sensor.scale.z = 0.035
        sensor.color.r, sensor.color.g, sensor.color.b, sensor.color.a = 1.0, 0.05, 0.05, 1.0
        room.markers.append(sensor)

    start_time = time.monotonic()

    def publish():
        now = node.get_clock().now().to_msg()
        for marker in room.markers:
            marker.header.stamp = now
        room_pub.publish(room)

        index = min(int(time.monotonic() - start_time), len(vectors) - 1)
        vector = vectors[index]
        speed = float(np.linalg.norm(vector))
        direction = vector / speed if speed > 1e-9 else np.zeros(3)
        # Keep weak drift visible while preserving relative length above 0.03 m/s.
        display_length = args.arrow_scale * (
            0.18 if speed < 0.03 else min(1.2, 0.18 + 3.0 * speed)
        )
        wind = MarkerArray()
        marker_id = 0
        for x in (1.0, 3.0, 5.0, 7.0, 9.0):
            for y in (1.0, 3.0, 5.0):
                arrow = Marker()
                arrow.header.frame_id = "map"
                arrow.header.stamp = now
                arrow.ns = "uniform_wind"
                arrow.id = marker_id
                marker_id += 1
                arrow.type = Marker.ARROW
                arrow.action = Marker.ADD
                start = np.asarray([x, y, 0.12])
                end = start + direction * display_length
                arrow.points = [Point(x=float(start[0]), y=float(start[1]), z=float(start[2])),
                                Point(x=float(end[0]), y=float(end[1]), z=float(end[2]))]
                arrow.scale.x, arrow.scale.y, arrow.scale.z = (
                    0.025 * args.arrow_scale,
                    0.07 * args.arrow_scale,
                    0.10 * args.arrow_scale,
                )
                arrow.color.r, arrow.color.g, arrow.color.b, arrow.color.a = 0.15, 0.75, 1.0, 0.85
                wind.markers.append(arrow)

        label = Marker()
        label.header.frame_id = "map"
        label.header.stamp = now
        label.ns = "wind_label"
        label.id = marker_id
        label.type = Marker.TEXT_VIEW_FACING
        label.action = Marker.ADD
        label.pose.position.x, label.pose.position.y, label.pose.position.z = 2.0, 3.0, 1.25
        label.pose.orientation.w = 1.0
        label.scale.z = 0.22
        label.color.r, label.color.g, label.color.b, label.color.a = 0.9, 0.95, 1.0, 1.0
        label.text = f"{args.regime}: {speed:.3f} m/s"
        wind.markers.append(label)

        for source_index, source in enumerate(args.source_label):
            text, x, y, z = source
            source_label = Marker()
            source_label.header.frame_id = "map"
            source_label.header.stamp = now
            source_label.ns = "source_labels"
            source_label.id = marker_id + source_index + 1
            source_label.type = Marker.TEXT_VIEW_FACING
            source_label.action = Marker.ADD
            source_label.pose.position.x = float(x)
            source_label.pose.position.y = float(y)
            source_label.pose.position.z = float(z) + 0.18
            source_label.pose.orientation.w = 1.0
            source_label.scale.z = 0.10
            source_label.color.r = 1.0
            source_label.color.g = 1.0
            source_label.color.b = 1.0
            source_label.color.a = 1.0
            source_label.text = text
            wind.markers.append(source_label)
        wind_pub.publish(wind)

    timer = node.create_timer(0.2, publish)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        timer.cancel()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wind-file", required=True)
    parser.add_argument("--regime", required=True)
    parser.add_argument("--tabletop", action="store_true")
    parser.add_argument("--table-center-x", type=float, default=2.0)
    parser.add_argument("--table-center-y", type=float, default=3.0)
    parser.add_argument("--tabletop-z", type=float, default=0.8)
    parser.add_argument("--table-size-x", type=float, default=0.8)
    parser.add_argument("--table-size-y", type=float, default=0.8)
    parser.add_argument("--arrow-scale", type=float, default=1.0)
    parser.add_argument("--sensor-position", nargs=3, type=float)
    parser.add_argument(
        "--source-label", action="append", nargs=4, default=[],
        metavar=("TEXT", "X", "Y", "Z"),
    )
    return parser


if __name__ == "__main__":
    raise SystemExit(run(build_parser().parse_args()))

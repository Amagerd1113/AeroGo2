#!/usr/bin/env python3
import bisect
import math
import pathlib
import statistics
import struct
import sys

import rosbag2_py
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import Imu, PointCloud2


def stamp_seconds(message):
    stamp = message.header.stamp
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def mean_std(values):
    if not values:
        return float("nan"), float("nan")
    return statistics.fmean(values), statistics.pstdev(values)


def percentile(values, fraction):
    if not values:
        return float("nan")
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(fraction * (len(ordered) - 1)))]


def reversals(values):
    return sum(current <= previous for previous, current in zip(values, values[1:]))


def main():
    if len(sys.argv) != 2:
        raise SystemExit("usage: 04_probe_ros2_bag.py /absolute/path/to/ros2_bag")
    bag_path = pathlib.Path(sys.argv[1]).expanduser().resolve()
    if not (bag_path / "metadata.yaml").is_file():
        raise SystemExit(f"ERROR: metadata.yaml not found under {bag_path}")

    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(bag_path), storage_id="sqlite3"),
        rosbag2_py.ConverterOptions(
            input_serialization_format="cdr", output_serialization_format="cdr"
        ),
    )

    imu_rows = []
    cloud_stamps = []
    imu_record_offsets = []
    cloud_record_offsets = []
    cloud_ranges = []
    cloud_time_spans = []
    cloud_nonfinite_samples = 0

    while reader.has_next():
        topic, payload, record_time_ns = reader.read_next()
        if topic == "/utlidar/imu":
            message = deserialize_message(payload, Imu)
            header_time = stamp_seconds(message)
            acceleration = message.linear_acceleration
            angular = message.angular_velocity
            values = (
                float(acceleration.x),
                float(acceleration.y),
                float(acceleration.z),
                float(angular.x),
                float(angular.y),
                float(angular.z),
            )
            if all(math.isfinite(value) for value in values):
                imu_rows.append((header_time,) + values)
            imu_record_offsets.append(record_time_ns * 1e-9 - header_time)
        elif topic == "/utlidar/cloud":
            message = deserialize_message(payload, PointCloud2)
            header_time = stamp_seconds(message)
            cloud_stamps.append(header_time)
            cloud_record_offsets.append(record_time_ns * 1e-9 - header_time)
            field_offsets = {field.name: field.offset for field in message.fields}
            required = {"x", "y", "z", "time"}
            if not required.issubset(field_offsets):
                raise SystemExit(f"ERROR: missing PointCloud2 fields: {required - field_offsets.keys()}")
            endian = ">" if message.is_bigendian else "<"
            point_count = int(message.width) * int(message.height)
            point_stride = max(1, point_count // 200)
            point_times = []
            for index in range(point_count):
                base = index * int(message.point_step)
                point_time = struct.unpack_from(endian + "f", message.data, base + field_offsets["time"])[0]
                if math.isfinite(point_time):
                    point_times.append(point_time)
                if index % point_stride == 0:
                    x = struct.unpack_from(endian + "f", message.data, base + field_offsets["x"])[0]
                    y = struct.unpack_from(endian + "f", message.data, base + field_offsets["y"])[0]
                    z = struct.unpack_from(endian + "f", message.data, base + field_offsets["z"])[0]
                    if all(math.isfinite(value) for value in (x, y, z)):
                        cloud_ranges.append(math.sqrt(x * x + y * y + z * z))
                    else:
                        cloud_nonfinite_samples += 1
            if point_times:
                cloud_time_spans.append(max(point_times) - min(point_times))

    if len(imu_rows) < 2 or len(cloud_stamps) < 2:
        raise SystemExit("ERROR: bag does not contain enough IMU/cloud samples")

    imu_stamps = [row[0] for row in imu_rows]
    static_limit = imu_stamps[0] + 8.0
    static_rows = [row for row in imu_rows if row[0] <= static_limit]
    if len(static_rows) < 2:
        raise SystemExit("ERROR: fewer than two IMU samples in the first 8 seconds")

    axis_stats = []
    for index in range(1, 7):
        axis_stats.append(mean_std([row[index] for row in static_rows]))
    acceleration_norms = [math.sqrt(row[1] ** 2 + row[2] ** 2 + row[3] ** 2) for row in static_rows]
    angular_norms = [math.sqrt(row[4] ** 2 + row[5] ** 2 + row[6] ** 2) for row in static_rows]
    acceleration_norm_mean, acceleration_norm_std = mean_std(acceleration_norms)
    angular_norm_mean, angular_norm_std = mean_std(angular_norms)

    nearest_deltas = []
    for cloud_stamp in cloud_stamps:
        location = bisect.bisect_left(imu_stamps, cloud_stamp)
        candidates = []
        if location < len(imu_stamps):
            candidates.append(abs(imu_stamps[location] - cloud_stamp))
        if location > 0:
            candidates.append(abs(imu_stamps[location - 1] - cloud_stamp))
        if candidates:
            nearest_deltas.append(min(candidates) * 1000.0)

    imu_offset_mean, imu_offset_std = mean_std(imu_record_offsets)
    cloud_offset_mean, cloud_offset_std = mean_std(cloud_record_offsets)
    print(f"imu_count={len(imu_rows)}")
    print(f"cloud_count={len(cloud_stamps)}")
    print(f"imu_header_duration_s={imu_stamps[-1] - imu_stamps[0]:.6f}")
    print(f"cloud_header_duration_s={cloud_stamps[-1] - cloud_stamps[0]:.6f}")
    print(f"imu_nonmonotonic={reversals(imu_stamps)}")
    print(f"cloud_nonmonotonic={reversals(cloud_stamps)}")
    print(f"static_imu_samples={len(static_rows)}")
    print("static_acc_mean_xyz=" + ",".join(f"{axis_stats[index][0]:.6f}" for index in range(3)))
    print("static_acc_std_xyz=" + ",".join(f"{axis_stats[index][1]:.6f}" for index in range(3)))
    print(f"static_acc_norm_mean={acceleration_norm_mean:.6f}")
    print(f"static_acc_norm_std={acceleration_norm_std:.6f}")
    print("static_gyro_mean_xyz=" + ",".join(f"{axis_stats[index][0]:.6f}" for index in range(3, 6)))
    print("static_gyro_std_xyz=" + ",".join(f"{axis_stats[index][1]:.6f}" for index in range(3, 6)))
    print(f"static_gyro_norm_mean={angular_norm_mean:.6f}")
    print(f"static_gyro_norm_std={angular_norm_std:.6f}")
    print(f"nearest_imu_delta_ms_mean={statistics.fmean(nearest_deltas):.6f}")
    print(f"nearest_imu_delta_ms_p95={percentile(nearest_deltas, 0.95):.6f}")
    print(f"nearest_imu_delta_ms_max={max(nearest_deltas):.6f}")
    print(f"imu_record_minus_header_s_mean={imu_offset_mean:.6f}")
    print(f"imu_record_minus_header_s_std={imu_offset_std:.6f}")
    print(f"cloud_record_minus_header_s_mean={cloud_offset_mean:.6f}")
    print(f"cloud_record_minus_header_s_std={cloud_offset_std:.6f}")
    print(f"record_offset_topic_difference_ms={(cloud_offset_mean - imu_offset_mean) * 1000.0:.6f}")
    print(f"cloud_time_span_ms_mean={statistics.fmean(cloud_time_spans) * 1000.0:.6f}")
    print(f"cloud_time_span_ms_max={max(cloud_time_spans) * 1000.0:.6f}")
    print(f"cloud_range_m_median={percentile(cloud_ranges, 0.50):.6f}")
    print(f"cloud_range_m_p95={percentile(cloud_ranges, 0.95):.6f}")
    print(f"cloud_range_m_max={max(cloud_ranges):.6f}")
    print(f"cloud_nonfinite_sampled={cloud_nonfinite_samples}")


if __name__ == "__main__":
    main()

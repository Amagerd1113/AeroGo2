#!/usr/bin/env python3
import csv
import math
import pathlib
import sys


def find_column(fieldnames, suffix):
    for name in fieldnames:
        if name.endswith(suffix):
            return name
    raise KeyError(suffix)


def parse_time(raw):
    value = float(raw)
    return value / 1e9 if abs(value) > 1e12 else value


def main():
    if len(sys.argv) != 2:
        raise SystemExit("usage: 03_analyze_pointlio_csv.py pointlio_odom.csv")
    csv_path = pathlib.Path(sys.argv[1])
    if not csv_path.is_file() or csv_path.stat().st_size == 0:
        raise SystemExit(f"ERROR: odometry CSV is empty: {csv_path}")

    with csv_path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames:
            raise SystemExit("ERROR: odometry CSV has no header")
        time_key = "%time" if "%time" in reader.fieldnames else reader.fieldnames[0]
        x_key = find_column(reader.fieldnames, ".pose.pose.position.x")
        y_key = find_column(reader.fieldnames, ".pose.pose.position.y")
        z_key = find_column(reader.fieldnames, ".pose.pose.position.z")
        qx_key = find_column(reader.fieldnames, ".pose.pose.orientation.x")
        qy_key = find_column(reader.fieldnames, ".pose.pose.orientation.y")
        qz_key = find_column(reader.fieldnames, ".pose.pose.orientation.z")
        qw_key = find_column(reader.fieldnames, ".pose.pose.orientation.w")
        samples = []
        for row in reader:
            try:
                values = tuple(float(row[key]) for key in (x_key, y_key, z_key, qx_key, qy_key, qz_key, qw_key))
                timestamp = parse_time(row[time_key])
            except (KeyError, TypeError, ValueError):
                continue
            if math.isfinite(timestamp) and all(math.isfinite(value) for value in values):
                samples.append((timestamp,) + values)

    if len(samples) < 2:
        raise SystemExit(f"ERROR: only {len(samples)} valid odometry samples")

    first = samples[0]
    last = samples[-1]
    duration = max(0.0, last[0] - first[0])
    dx, dy, dz = last[1] - first[1], last[2] - first[2], last[3] - first[3]
    path_length = 0.0
    max_step = 0.0
    for previous, current in zip(samples, samples[1:]):
        step = math.sqrt(sum((current[index] - previous[index]) ** 2 for index in (1, 2, 3)))
        path_length += step
        max_step = max(max_step, step)

    def yaw(sample):
        qx, qy, qz, qw = sample[4], sample[5], sample[6], sample[7]
        return math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))

    yaw_delta = (yaw(last) - yaw(first) + math.pi) % (2.0 * math.pi) - math.pi
    rate = (len(samples) - 1) / duration if duration > 0.0 else 0.0
    print(f"samples={len(samples)}")
    print(f"duration_s={duration:.3f}")
    print(f"rate_hz={rate:.3f}")
    print(f"dx_m={dx:.4f}")
    print(f"dy_m={dy:.4f}")
    print(f"dz_m={dz:.4f}")
    print(f"net_distance_m={math.sqrt(dx * dx + dy * dy + dz * dz):.4f}")
    print(f"path_length_m={path_length:.4f}")
    print(f"yaw_change_deg={math.degrees(yaw_delta):.3f}")
    print(f"max_single_step_m={max_step:.4f}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import pathlib
import shutil


source_path = (
    pathlib.Path.home()
    / "aerogo2_slam_ws"
    / "src"
    / "point_lio_unilidar"
    / "src"
    / "preprocess.cpp"
)
backup_path = source_path.with_suffix(".cpp.aerogo2_before_time_guard")

if not source_path.is_file():
    raise SystemExit(f"ERROR: Point-LIO source not found: {source_path}")

source_text = source_path.read_text(encoding="utf-8")
marker = "const float point_time_ms = pl_orig.points[i].time * time_unit_scale;"

if marker in source_text:
    print("L1_TIME_GUARD=already_applied")
    raise SystemExit(0)

old_text = """      added_pt.intensity = pl_orig.points[i].intensity;

      added_pt.curvature = pl_orig.points[i].time * time_unit_scale; 

      if (added_pt.x * added_pt.x + added_pt.y * added_pt.y + added_pt.z * added_pt.z > (blind * blind))
"""
new_text = """      added_pt.intensity = pl_orig.points[i].intensity;

      // Unitree L1 normally reports a relative point time within one scan
      // (about 68 ms on this Go2).  Drop corrupt timestamps instead of
      // allowing a multi-second outlier to destabilize IMU propagation.
      const float point_time_ms = pl_orig.points[i].time * time_unit_scale;
      if (!std::isfinite(point_time_ms) || point_time_ms < 0.0f || point_time_ms > 100.0f)
      {
        continue;
      }
      added_pt.curvature = point_time_ms;

      if (added_pt.x * added_pt.x + added_pt.y * added_pt.y + added_pt.z * added_pt.z > (blind * blind))
"""

if source_text.count(old_text) != 1:
    raise SystemExit("ERROR: expected Unitree handler block was not found exactly once; nothing changed")

if "#include <cmath>" not in source_text:
    include_anchor = '#include "preprocess.h"\n'
    if source_text.count(include_anchor) != 1:
        raise SystemExit("ERROR: preprocess include anchor not found; nothing changed")
    source_text = source_text.replace(include_anchor, include_anchor + "\n#include <cmath>\n", 1)

if not backup_path.exists():
    shutil.copy2(source_path, backup_path)

source_text = source_text.replace(old_text, new_text, 1)
source_path.write_text(source_text, encoding="utf-8")
print(f"BACKUP={backup_path}")
print("L1_TIME_GUARD=applied")

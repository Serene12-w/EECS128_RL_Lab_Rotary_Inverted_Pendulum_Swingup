"""Render a side-by-side pendulum replay from saved Simulink angle traces."""

import argparse
import csv
import math
from pathlib import Path

import cv2
import numpy as np


def read_trace(path):
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    return [(float(row["time_s"]), float(row["pendulum_angle_deg"])) for row in rows]


def angle_at(trace, time_s):
    index = min(round(time_s / 0.01), len(trace) - 1)
    return trace[index][1]


def draw_pendulum(frame, center_x, angle_deg, label):
    pivot_y, length = 370, 220
    angle = math.radians(angle_deg)
    tip = (round(center_x + length * math.sin(angle)),
           round(pivot_y - length * math.cos(angle)))
    upright = abs(angle_deg) <= 15
    color = (68, 190, 92) if upright else (70, 150, 245)
    cv2.line(frame, (center_x, pivot_y), (center_x, pivot_y - length),
             (90, 90, 90), 2)
    for boundary in (-15, 15):
        a = math.radians(boundary)
        edge = (round(center_x + length * math.sin(a)),
                round(pivot_y - length * math.cos(a)))
        cv2.line(frame, (center_x, pivot_y), edge, (80, 110, 80), 1)
    cv2.line(frame, (center_x, pivot_y), tip, color, 14, cv2.LINE_AA)
    cv2.circle(frame, (center_x, pivot_y), 16, (215, 215, 215), -1, cv2.LINE_AA)
    cv2.circle(frame, tip, 18, color, -1, cv2.LINE_AA)
    cv2.putText(frame, label, (center_x - 175, 100), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (245, 245, 245), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Angle: {angle_deg:+.1f} deg", (center_x - 175, 650),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (245, 245, 245), 2, cv2.LINE_AA)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evaluation_dir", type=Path)
    parser.add_argument("--trial", type=int, default=1)
    parser.add_argument("--damping", type=float, default=0.25)
    args = parser.parse_args()
    folder = args.evaluation_dir
    suffix = f"d{args.damping:.3f}_trial{args.trial}.csv"
    baseline = read_trace(folder / f"angle_baseline_{suffix}")
    trained = read_trace(folder / f"angle_dampingArm_pm10pct_{suffix}")
    output = folder / f"swingup_comparison_d{args.damping:.3f}_trial{args.trial}.mp4"
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"),
                             30, (1280, 720))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot create {output}")
    try:
        for frame_index in range(151):
            time_s = frame_index / 30
            frame = np.full((720, 1280, 3), (32, 32, 32), dtype="uint8")
            cv2.line(frame, (640, 80), (640, 680), (75, 75, 75), 2)
            draw_pendulum(frame, 320, angle_at(baseline, time_s), "Lab 6a baseline")
            draw_pendulum(frame, 960, angle_at(trained, time_s), "Lab 6b randomized")
            cv2.putText(frame, f"t = {time_s:.2f} s   arm damping = {args.damping:.3f}",
                        (290, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.85,
                        (245, 245, 245), 2, cv2.LINE_AA)
            cv2.putText(frame, "Green = within upright +/-15 deg", (415, 690),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (190, 190, 190), 1, cv2.LINE_AA)
            writer.write(frame)
    finally:
        writer.release()
    print(output)


if __name__ == "__main__":
    main()

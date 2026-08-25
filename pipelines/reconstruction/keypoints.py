from __future__ import annotations

import json
import os
import re

import numpy as np


def get_frame_number(filename):
    match = re.findall(r"\d+", filename)
    return int(match[-1]) if match else -1


def filter_frame_numbers(frame_numbers, frame_range):
    sorted_frames = sorted(frame_numbers)
    if frame_range in (None, "all", "auto", []):
        return sorted_frames
    try:
        start = int(frame_range[0])
        end = int(frame_range[1])
    except (TypeError, ValueError, IndexError):
        return sorted_frames
    if end < start:
        start, end = end, start
    return [frame for frame in sorted_frames if start <= frame <= end]


def normalize_camera_label(label: str) -> str:
    match = re.search(r"cam0*(\d+)$", str(label).lower())
    return f"cam{int(match.group(1))}" if match else str(label).lower()


def camera_sort_key(label: str):
    normalized = normalize_camera_label(label)
    match = re.search(r"cam(\d+)$", normalized)
    return (int(match.group(1)) if match else 9999, str(label).lower())


def pose_json_dirs(project_dir: str):
    pose_dir = os.path.join(project_dir, "pose")
    if not os.path.isdir(pose_dir):
        return {}
    dirs = {}
    for name in os.listdir(pose_dir):
        match = re.match(r"^(cam\d+)_json$", name, re.IGNORECASE)
        if match:
            label = normalize_camera_label(match.group(1))
            dirs[label] = os.path.join(pose_dir, name)
    return dict(sorted(dirs.items(), key=lambda item: camera_sort_key(item[0])))


def person_to_keypoints(person):
    return np.asarray(person["pose_keypoints_2d"], dtype=np.float64).reshape(26, 3)

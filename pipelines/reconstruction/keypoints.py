from __future__ import annotations

import json
import os
import re

import numpy as np


def get_frame_number(filename):
    match = re.findall(r"\d+", filename)
    return int(match[-1]) if match else -1


def select_person_by_index(people, person_idx: int, camera_name: str, frame_idx: int):
    """Select by zero-based position in JSON people[], not by person_id."""
    if not (0 <= person_idx < len(people)):
        raise ValueError(
            f"{camera_name}_person_idx={person_idx} is out of range for frame {frame_idx} "
            f"({camera_name} people={len(people)})."
        )
    return people[person_idx]


def load_synchronized_kps(cam1_dir, cam2_dir, cam1_person_idx: int = 0, cam2_person_idx: int = 0):
    cam1_files = {get_frame_number(filename): filename for filename in os.listdir(cam1_dir) if filename.endswith(".json")}
    cam2_files = {get_frame_number(filename): filename for filename in os.listdir(cam2_dir) if filename.endswith(".json")}
    common_frames = sorted(set(cam1_files).intersection(cam2_files))

    kp1_list, kp2_list, valid_frames = [], [], []
    for frame_idx in common_frames:
        path1 = os.path.join(cam1_dir, cam1_files[frame_idx])
        path2 = os.path.join(cam2_dir, cam2_files[frame_idx])
        with open(path1) as file1, open(path2) as file2:
            data1, data2 = json.load(file1), json.load(file2)
        people1 = data1.get("people", []) or []
        people2 = data2.get("people", []) or []
        if len(people1) == 0 or len(people2) == 0:
            continue

        person1 = select_person_by_index(people1, cam1_person_idx, "cam1", frame_idx)
        person2 = select_person_by_index(people2, cam2_person_idx, "cam2", frame_idx)

        kp1_list.append(np.array(person1["pose_keypoints_2d"]).reshape(26, 3))
        kp2_list.append(np.array(person2["pose_keypoints_2d"]).reshape(26, 3))
        valid_frames.append(frame_idx)
    return np.array(kp1_list), np.array(kp2_list), valid_frames


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

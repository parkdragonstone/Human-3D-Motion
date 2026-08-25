"""2D keypoint input for keypoint-based extrinsic calibration.

Calibration runs inside the analysis pipeline, after pose estimation has already written
OpenPose-style JSON per camera under ``<project>/pose/<cam>_json``. Reading those files
back costs nothing and guarantees the calibration sees exactly the keypoints the
reconstruction will later triangulate; re-running RTMPose here would be slower and could
disagree with them.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

import numpy as np

from .skeleton import NUM_JOINTS


@dataclass
class CameraPose2D:
    camera_label: str
    keypoints: np.ndarray       # (N, J, 2) pixels
    scores: np.ndarray          # (N, J) confidence, 0 where missing
    image_size: tuple[int, int]  # (width, height)
    frame_indices: np.ndarray


def _largest_person(people) -> np.ndarray | None:
    """Pick the person covering the most image area, i.e. the calibration subject.

    Pose JSON carries no bounding boxes, so the extent of the confident keypoints
    stands in for one.
    """
    best = None
    best_area = -1.0
    for person in people:
        flat = person.get("pose_keypoints_2d") or []
        if len(flat) < NUM_JOINTS * 3:
            continue
        array = np.asarray(flat, dtype=np.float64).reshape(-1, 3)[:NUM_JOINTS]
        valid = np.isfinite(array).all(axis=1) & (array[:, 2] > 0)
        if valid.sum() < 4:
            continue
        points = array[valid, :2]
        spread = points.max(axis=0) - points.min(axis=0)
        area = float(spread[0] * spread[1]) * float(np.mean(array[valid, 2]))
        if area > best_area:
            best_area = area
            best = array
    return best


def load_camera_pose2d(
    pose_json_dir: str,
    camera_label: str,
    image_size,
    *,
    frame_stride: int = 5,
    max_frames: int = 300,
) -> CameraPose2D:
    """Read one camera's pose JSON directory into dense arrays."""
    from ...reconstruction.keypoints import get_frame_number

    if not os.path.isdir(pose_json_dir):
        raise ValueError(f"pose_json_dir_not_found: {pose_json_dir}")

    files = sorted(
        (name for name in os.listdir(pose_json_dir) if name.endswith(".json")),
        key=get_frame_number,
    )
    stride = max(1, int(frame_stride))
    selected = files[::stride][:max(1, int(max_frames))]

    keypoints = np.zeros((len(selected), NUM_JOINTS, 2), dtype=np.float64)
    scores = np.zeros((len(selected), NUM_JOINTS), dtype=np.float64)
    frame_indices = []

    for position, name in enumerate(selected):
        frame_indices.append(get_frame_number(name))
        with open(os.path.join(pose_json_dir, name), encoding="utf-8") as handle:
            payload = json.load(handle)
        person = _largest_person(payload.get("people") or [])
        if person is None:
            continue
        finite = np.isfinite(person).all(axis=1)
        keypoints[position][finite] = person[finite, :2]
        scores[position][finite] = np.clip(person[finite, 2], 0.0, 1.0)

    return CameraPose2D(
        camera_label=camera_label,
        keypoints=keypoints,
        scores=scores,
        image_size=(int(image_size[0]), int(image_size[1])),
        frame_indices=np.asarray(frame_indices, dtype=int),
    )


def load_all_cameras(
    pose_dirs_by_label: dict[str, str],
    image_sizes_by_label: dict[str, tuple[int, int]],
    *,
    frame_stride: int = 5,
    max_frames: int = 300,
    progress=None,
) -> list[CameraPose2D]:
    """Load every camera, trimmed to a common frame count."""
    cameras: list[CameraPose2D] = []
    for label, directory in pose_dirs_by_label.items():
        if progress:
            progress(f"reading 2D keypoints for {label}")
        cameras.append(
            load_camera_pose2d(
                directory,
                label,
                image_sizes_by_label[label],
                frame_stride=frame_stride,
                max_frames=max_frames,
            )
        )

    shortest = min(len(camera.frame_indices) for camera in cameras)
    for camera in cameras:
        camera.keypoints = camera.keypoints[:shortest]
        camera.scores = camera.scores[:shortest]
        camera.frame_indices = camera.frame_indices[:shortest]
    return cameras


def stack_cameras(cameras: list[CameraPose2D]):
    """Stack per-camera results into ``(C, N, J, 2)``, ``(C, N, J)`` and image sizes."""
    p2d = np.stack([camera.keypoints for camera in cameras], axis=0)
    s2d = np.nan_to_num(np.stack([camera.scores for camera in cameras], axis=0), nan=0.0)
    missing = ~np.isfinite(p2d).all(axis=-1)
    p2d = np.nan_to_num(p2d, nan=0.0)
    s2d[missing] = 0.0
    return p2d, s2d, [camera.image_size for camera in cameras]

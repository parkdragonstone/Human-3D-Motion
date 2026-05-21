"""Compatibility entrypoint for the pose estimation package."""

from .pose_estimation import (
    keypoints_ids,
    keypoints_names,
    process_frame,
    run_poseEstimation,
    save_to_openpose,
    setup_backend_device,
    setup_detector,
    setup_pose_solver,
    wrapping_detector,
)

__all__ = [
    "keypoints_ids",
    "keypoints_names",
    "process_frame",
    "run_poseEstimation",
    "save_to_openpose",
    "setup_backend_device",
    "setup_detector",
    "setup_pose_solver",
    "wrapping_detector",
]

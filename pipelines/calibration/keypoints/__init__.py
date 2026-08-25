"""Keypoint-based extrinsic calibration: a moving person replaces the calibration board.

Port of the RTMPose + VideoPose3D route of
`flodelaplace/lab-camera-dynamic-calibrator <https://github.com/flodelaplace/lab-camera-dynamic-calibrator>`_
(MIT), rebuilt on the pose output this project already produces.

Pipeline
--------
1. The 2D joints written by pose estimation are read back per camera (:mod:`.pose2d`).
2. VideoPose3D lifts each view to root-relative 3D (:mod:`.lift3d`) -- the *directions* of
   the bones are what matter, the scale it invents is discarded.
3. A linear solve turns shared bone directions plus 2D bearings into all camera poses at
   once (:mod:`.linear`).
4. Bundle adjustment refines poses, optionally the focal lengths, and the joint cloud
   (:mod:`.bundle`).
5. The scene is gravity-aligned, put on the floor and scaled by the subject stature
   (:mod:`.scale`), and comes out Z-up like every other calibration target.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from . import skeleton
from .bundle import mean_reprojection_error, run_bundle_adjustment
from .geometry import triangulate_points
from .intrinsics import approximate_intrinsic, resolve_intrinsics
from .lift3d import MissingLifterCheckpoint, lift_all_cameras
from .linear import (
    bearing_vectors,
    calibrate_linear,
    degenerate_eigenvalue_ratio,
    joints_to_orientations,
    joints_to_projections,
)
from .pose2d import load_all_cameras, stack_cameras
from .scale import align_and_scale


DEFAULT_OPTIONS: dict[str, Any] = {
    "subject_height_m": 1.70,
    "frame_stride": 5,
    "max_frames": 300,
    "conf_threshold": 0.5,
    "ba_iterations": 2,
    "videopose3d_checkpoint": None,
}

MIN_FRAMES = 12
MIN_COVISIBLE_JOINTS = 6


def _undistort(p2d, intrinsics):
    """Remove lens distortion so the pinhole model the solvers assume actually holds."""
    import cv2

    corrected = np.array(p2d, dtype=np.float64, copy=True)
    for camera, intrinsic in enumerate(intrinsics):
        distortion = np.asarray(intrinsic.get("dist_coeffs") or [], dtype=np.float64).reshape(-1)
        if distortion.size == 0 or not np.any(np.abs(distortion) > 1e-12):
            continue
        K = np.asarray(intrinsic["camera_matrix"], dtype=np.float64)
        flat = corrected[camera].reshape(-1, 1, 2)
        undistorted = cv2.undistortPoints(flat, K, distortion.reshape(-1, 1), P=K)
        corrected[camera] = undistorted.reshape(corrected[camera].shape)
    return corrected


def _select_frames(s2d, conf_threshold: float, bone_joints) -> np.ndarray:
    """Keep frames carrying enough joints seen confidently by *every* camera.

    Only fully covisible observations reach the solvers, so a frame nobody agrees on is
    dead weight; filtering the whole frame on one occluded wrist would be wasteful the
    other way round, hence a minimum count rather than an all-or-nothing test.
    """
    covisible = (s2d[:, :, bone_joints] > conf_threshold).all(axis=0)
    return covisible.sum(axis=1) >= MIN_COVISIBLE_JOINTS


def run_keypoint_calibration(
    pose_dirs_by_label: dict[str, str],
    image_sizes_by_label: dict[str, tuple[int, int]],
    intrinsics_by_label: dict[str, dict[str, Any]],
    options: dict[str, Any] | None = None,
    progress: Callable[[str], None] | None = None,
    refine_focal: bool = False,
) -> dict[str, Any]:
    """Estimate extrinsics for every camera from a person moving in the scene."""
    settings = {**DEFAULT_OPTIONS, **{k: v for k, v in (options or {}).items() if v not in (None, "")}}

    labels = list(pose_dirs_by_label.keys())
    if len(labels) < 2:
        return {"ok": False, "error": f"need_at_least_2_cameras: {len(labels)}"}
    missing = [label for label in labels if intrinsics_by_label.get(label) is None]
    if missing:
        return {"ok": False, "error": f"intrinsic_not_found_for_cameras: {', '.join(missing)}"}

    def report(message: str) -> None:
        if progress:
            progress(message)

    conf_threshold = float(settings["conf_threshold"])

    # --- 1. 2D keypoints from the pose stage --------------------------------------
    try:
        cameras = load_all_cameras(
            pose_dirs_by_label,
            image_sizes_by_label,
            frame_stride=int(settings["frame_stride"]),
            max_frames=int(settings["max_frames"]),
            progress=progress,
        )
    except ValueError as exc:
        return {"ok": False, "error": str(exc)}

    p2d, s2d, image_sizes = stack_cameras(cameras)
    intrinsics = [intrinsics_by_label[label] for label in labels]
    K_all = np.array([np.asarray(item["camera_matrix"], dtype=np.float64) for item in intrinsics])
    p2d = _undistort(p2d, intrinsics)

    keep = _select_frames(s2d, conf_threshold, skeleton.BONE_JOINTS)
    if int(keep.sum()) < MIN_FRAMES:
        return {
            "ok": False,
            "error": f"not_enough_covisible_frames: {int(keep.sum())} < {MIN_FRAMES}",
            "hint": "the subject must be visible in every camera at the same time",
        }
    p2d, s2d = p2d[:, keep], s2d[:, keep]
    report(f"{int(keep.sum())} frames are covisible in all {len(labels)} cameras")

    # --- 2. 3D lifting -------------------------------------------------------------
    try:
        p3d, s3d = lift_all_cameras(
            p2d, s2d, image_sizes,
            checkpoint=settings["videopose3d_checkpoint"],
            progress=progress,
        )
    except (MissingLifterCheckpoint, RuntimeError) as exc:
        return {"ok": False, "error": str(exc)}

    # --- 3. Linear calibration -----------------------------------------------------
    report("solving linear extrinsic calibration")
    mask = s2d > conf_threshold
    orientations = joints_to_orientations(p3d, mask & (s3d > 0), skeleton.BONES)
    if orientations.size == 0 or orientations.shape[1] < 3:
        return {"ok": False, "error": "not_enough_valid_bone_orientations"}

    projections = joints_to_projections(p2d, mask)
    if projections.shape[1] < 6:
        return {"ok": False, "error": f"not_enough_covisible_joints: {projections.shape[1]}"}

    bearings = bearing_vectors(projections, K_all)
    degeneracy = degenerate_eigenvalue_ratio(orientations)

    R_linear, t_linear, _ = calibrate_linear(orientations, bearings)
    if R_linear is None:
        return {
            "ok": False,
            "error": "linear_calibration_degenerate",
            "hint": "the subject moved too little; record more varied motion across the volume",
            "degeneracy": degeneracy,
        }

    linear_points = triangulate_points(p2d, s2d, K_all, R_linear, t_linear, conf_threshold)
    linear_error = mean_reprojection_error(
        K_all, R_linear, t_linear, linear_points, p2d, s2d, conf_threshold
    )
    report(f"linear solution: mean reprojection {linear_error:.2f} px")

    # --- 4. Bundle adjustment ------------------------------------------------------
    R_refined, t_refined, points, K_refined, ba_report = run_bundle_adjustment(
        K_all, R_linear, t_linear, p2d, s2d, p3d, s3d, skeleton.BONES,
        conf_threshold=conf_threshold,
        iterations=int(settings["ba_iterations"]),
        refine_focal=refine_focal,
        progress=progress,
    )

    # --- 5. Metric scaling and gravity alignment -----------------------------------
    report("aligning the scene to gravity and scaling to metres")
    subject_height = settings.get("subject_height_m")
    R_final, t_final, scene_points, scale_report = align_and_scale(
        R_refined, t_refined, points, float(subject_height) if subject_height else None
    )

    final_error = mean_reprojection_error(
        K_refined, R_final, t_final, scene_points, p2d, s2d, conf_threshold
    )

    # --- 6. Result payload ---------------------------------------------------------
    import cv2

    cameras_payload: dict[str, Any] = {}
    refined_intrinsics: dict[str, Any] = {}
    for index, label in enumerate(labels):
        rotation_vector = cv2.Rodrigues(np.asarray(R_final[index], dtype=np.float64))[0].reshape(3)
        translation = np.asarray(t_final[index], dtype=np.float64).reshape(3)
        per_camera = mean_reprojection_error(
            K_refined[index:index + 1], R_final[index:index + 1], t_final[index:index + 1],
            scene_points, p2d[index:index + 1], s2d[index:index + 1], conf_threshold,
        )
        cameras_payload[label] = {
            "ok": True,
            "rvec": rotation_vector.tolist(),
            "tvec": translation.tolist(),
            "reproj_rms_px": float(per_camera),
            "matched_points": int(np.sum(s2d[index] > conf_threshold)),
            "image_size": list(image_sizes[index]),
        }
        source = intrinsics_by_label[label]
        refined_intrinsics[label] = {
            "camera_matrix": K_refined[index].tolist(),
            "dist_coeffs": list(source.get("dist_coeffs") or [0.0] * 5),
            "image_size": list(image_sizes[index]),
            "approximate": bool(source.get("approximate")),
            "focal_refined": bool(refine_focal),
        }

    return {
        "ok": True,
        "camera_labels": labels,
        "cameras": cameras_payload,
        "intrinsics": refined_intrinsics,
        "method": "keypoints",
        "frames_used": int(keep.sum()),
        "frame_stride": int(settings["frame_stride"]),
        "conf_threshold": conf_threshold,
        "degeneracy": float(degeneracy),
        "mean_reprojection_px": {
            "linear": float(linear_error),
            "bundle_adjustment": float(ba_report["mean_reprojection_px"]),
            "final": float(final_error),
        },
        "bundle_adjustment": ba_report,
        "scene": scale_report,
        "subject_height_m": float(subject_height) if subject_height else None,
    }


__all__ = [
    "run_keypoint_calibration",
    "approximate_intrinsic",
    "resolve_intrinsics",
    "DEFAULT_OPTIONS",
    "MissingLifterCheckpoint",
]

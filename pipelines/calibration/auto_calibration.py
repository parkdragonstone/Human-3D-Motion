"""Automatic calibration for analysis sessions that have no calibration file.

Runs between pose estimation and 3D reconstruction. The person already detected in the
footage becomes the calibration target, so a session recorded without a board can still
be reconstructed:

* intrinsics come from the upload when there is one, otherwise from a resolution-derived
  guess whose focal length the bundle adjustment then refines;
* extrinsics come from :mod:`pipelines.calibration.keypoints`;
* the result is written to ``<session>/analysis_calibration.json`` in the same shape the
  board-based targets produce, so reconstruction and later re-runs consume it unchanged.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .keypoints import resolve_intrinsics, run_keypoint_calibration
from ..reconstruction.keypoints import normalize_camera_label, pose_json_dirs


OUTPUT_NAME = "analysis_calibration.json"

DEFAULT_SETTINGS: dict[str, Any] = {
    "frame_stride": 5,
    "max_frames": 300,
    "conf_threshold": 0.5,
    "ba_iterations": 2,
    "focal_ratio": 0.9,
}


def _image_sizes(config: dict, camera_labels: list[str]) -> dict[str, tuple[int, int]]:
    base = config.get("base") or {}
    sizes: dict[str, tuple[int, int]] = {}
    for label in camera_labels:
        resolution = base.get(f"resolution_{label}")
        if not resolution or len(resolution) < 2 or int(resolution[0]) <= 0:
            raise ValueError(f"resolution_missing_for_camera: {label}")
        sizes[label] = (int(resolution[0]), int(resolution[1]))
    return sizes


def _subject_height_m(config: dict) -> float | None:
    """Session subject height is recorded in centimetres."""
    try:
        height_cm = float((config.get("subject") or {}).get("height") or 0)
    except (TypeError, ValueError):
        return None
    if not 50.0 <= height_cm <= 250.0:
        return None
    return height_cm / 100.0


def _payload(result: dict[str, Any], approximate: bool) -> dict[str, Any]:
    """Wrap the solver output in the schema the analysis loader already understands."""
    extrinsic = {key: value for key, value in result.items() if key != "intrinsics"}
    return {
        "ok": True,
        "mode": "EXTR",
        "source": "auto_keypoints",
        "approximate_intrinsics": approximate,
        "camera_labels": result.get("camera_labels", []),
        "intrinsics": result.get("intrinsics", {}),
        "extrinsic": extrinsic,
    }


def run_auto_calibration(config: dict, emit_log=None) -> dict[str, Any] | None:
    """Calibrate from the detected keypoints and persist the result.

    Returns the written payload, or ``None`` when calibration could not be produced.
    """

    def log(message: str, level: str = "info") -> None:
        if callable(emit_log):
            emit_log(message, level)

    project_dir = Path((config.get("paths") or {}).get("project_dir") or "")
    if not project_dir.is_dir():
        log("Auto calibration: project directory is missing.", "error")
        return None

    pose_dirs = pose_json_dirs(str(project_dir))
    if len(pose_dirs) < 2:
        log(f"Auto calibration: need pose output from at least two cameras, found {len(pose_dirs)}.", "error")
        return None

    camera_labels = list(pose_dirs.keys())
    try:
        image_sizes = _image_sizes(config, camera_labels)
    except ValueError as exc:
        log(f"Auto calibration: {exc}", "error")
        return None

    uploaded = config.get("calibration") if isinstance(config.get("calibration"), dict) else None
    intrinsics, approximate = resolve_intrinsics(
        camera_labels,
        image_sizes,
        uploaded,
        float((config.get("auto_calibration") or {}).get("focal_ratio", DEFAULT_SETTINGS["focal_ratio"])),
    )
    if approximate:
        log(
            "Auto calibration: no camera intrinsics supplied; starting from a "
            "resolution-based estimate and refining the focal length during bundle adjustment.",
            "info",
        )
    else:
        log("Auto calibration: using the uploaded camera intrinsics.", "info")

    settings = {**DEFAULT_SETTINGS, **(config.get("auto_calibration") or {})}
    height_m = _subject_height_m(config)
    if height_m is None:
        log(
            "Auto calibration: subject height is missing or out of range, so the scene "
            "keeps an arbitrary scale.",
            "warning",
        )

    log("Auto calibration: estimating camera poses from the detected person.", "info")
    result = run_keypoint_calibration(
        pose_dirs,
        image_sizes,
        intrinsics,
        {
            "subject_height_m": height_m,
            "frame_stride": settings.get("frame_stride"),
            "max_frames": settings.get("max_frames"),
            "conf_threshold": settings.get("conf_threshold"),
            "ba_iterations": settings.get("ba_iterations"),
            "videopose3d_checkpoint": settings.get("videopose3d_checkpoint"),
        },
        progress=lambda message: log(f"Auto calibration: {message}", "info"),
        refine_focal=approximate,
    )

    if not result.get("ok"):
        detail = result.get("error") or "calibration_failed"
        hint = result.get("hint")
        log(f"Auto calibration failed: {detail}" + (f" ({hint})" if hint else ""), "error")
        return None

    payload = _payload(result, approximate)
    output_path = project_dir / OUTPUT_NAME
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    errors = result.get("mean_reprojection_px", {})
    log(
        "Auto calibration complete: mean reprojection "
        f"{errors.get('final', float('nan')):.2f} px over {result.get('frames_used')} frames. "
        f"Saved {output_path.name}.",
        "info",
    )
    return payload


def bundle_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Flatten a written payload into the calibration bundle reconstruction expects."""
    bundle: dict[str, Any] = {}
    extrinsic = payload.get("extrinsic") or {}
    cameras = extrinsic.get("cameras")
    if isinstance(cameras, dict):
        bundle["cameras"] = {normalize_camera_label(label): camera for label, camera in cameras.items()}
        bundle["camera_labels"] = sorted(bundle["cameras"].keys())
    for label, intrinsic in (payload.get("intrinsics") or {}).items():
        bundle[normalize_camera_label(label)] = intrinsic
    bundle["object_points_unit_used"] = "m"
    return bundle

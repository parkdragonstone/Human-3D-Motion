from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .opencv import ensure_cv2


def first_video_frame(cv2, video_path: str):
    cap = cv2.VideoCapture(str(video_path))
    try:
        if not cap.isOpened():
            return None
        ok, frame = cap.read()
        if not ok or frame is None:
            return None
        return frame
    finally:
        cap.release()


def draw_labeled_points(cv2, frame, points, color, title: str):
    canvas = frame.copy()
    for index, point in enumerate(points or [], start=1):
        try:
            u = int(round(float(point.get("u"))))
            v = int(round(float(point.get("v"))))
        except Exception:
            continue
        cv2.circle(canvas, (u, v), 8, color, -1, lineType=cv2.LINE_AA)
        cv2.circle(canvas, (u, v), 12, (255, 255, 255), 2, lineType=cv2.LINE_AA)
        cv2.putText(canvas, str(point.get("id", index)), (u + 10, v - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, title, (18, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (215, 255, 67), 2, cv2.LINE_AA)
    return canvas


def project_object_points(cv2, intr: dict[str, Any], object_points: list[dict[str, Any]], rvec_value, tvec_value):
    camera_matrix = np.asarray(intr.get("camera_matrix"), dtype=np.float64)
    dist_coeffs = np.asarray(intr.get("dist_coeffs"), dtype=np.float64).reshape(-1, 1)
    rvec_value = np.asarray(rvec_value, dtype=np.float64).reshape(-1)
    tvec_value = np.asarray(tvec_value, dtype=np.float64).reshape(-1)
    if rvec_value.size != 3 or tvec_value.size != 3:
        return None
    object_array = np.asarray(
        [[float(point.get("x")), float(point.get("y")), float(point.get("z"))] for point in object_points],
        dtype=np.float64,
    ).reshape(-1, 3)
    img_pts, _ = cv2.projectPoints(
        object_array,
        rvec_value.reshape(3, 1),
        tvec_value.reshape(3, 1),
        camera_matrix,
        dist_coeffs,
    )
    projected = img_pts.reshape(-1, 2)
    return [
        {"id": point.get("id", index), "u": float(projected[index, 0]), "v": float(projected[index, 1])}
        for index, point in enumerate(object_points)
    ]


def draw_projection_overlay(cv2, frame, actual_points, projected_points, title: str):
    canvas = frame.copy()
    actual_map = {
        str(point.get("id")): point
        for point in actual_points or []
        if point.get("id") is not None
    }
    for index, projected in enumerate(projected_points or [], start=1):
        try:
            u = int(round(float(projected.get("u"))))
            v = int(round(float(projected.get("v"))))
        except Exception:
            continue
        point_id = projected.get("id", index)
        actual = actual_map.get(str(point_id))
        if actual is not None:
            try:
                actual_u = int(round(float(actual.get("u"))))
                actual_v = int(round(float(actual.get("v"))))
                cv2.line(canvas, (actual_u, actual_v), (u, v), (255, 255, 255), 1, cv2.LINE_AA)
                cv2.circle(canvas, (actual_u, actual_v), 7, (80, 200, 80), -1, lineType=cv2.LINE_AA)
                cv2.circle(canvas, (u, v), 8, (255, 255, 255), 2, lineType=cv2.LINE_AA)
                cv2.putText(canvas, str(point_id), (u + 10, v - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
            except Exception:
                pass
        else:
            cv2.circle(canvas, (u, v), 8, (255, 255, 255), 2, lineType=cv2.LINE_AA)
    cv2.putText(canvas, title, (18, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (215, 255, 67), 2, cv2.LINE_AA)
    cv2.putText(canvas, "green=clicked  white=reprojection", (18, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 230, 230), 1, cv2.LINE_AA)
    return canvas


def save_extrinsic_scene_result_images_multi(
    video_by_label: dict[str, Path],
    intrinsics_by_label: dict[str, dict[str, Any]],
    image_points_by_camera: dict[str, list[dict[str, Any]]],
    object_points: list[dict[str, Any]],
    extrinsic: dict[str, Any],
) -> dict[str, str] | None:
    cv2 = ensure_cv2()
    result_images: dict[str, str] = {}
    first_video = next(iter(video_by_label.values()), None)
    if first_video is None:
        return None
    extrinsic_ok = bool(extrinsic.get("ok"))

    def write(image, name: str) -> str:
        out_path = first_video.parent / name
        cv2.imwrite(str(out_path), image)
        return str(out_path)

    for label, video in video_by_label.items():
        frame = first_video_frame(cv2, str(video))
        points = image_points_by_camera.get(label, [])
        if frame is None:
            continue
        intrinsic = intrinsics_by_label.get(label)
        rvec = extrinsic.get(f"rvec_{label}")
        tvec = extrinsic.get(f"tvec_{label}")
        reprojection = project_object_points(cv2, intrinsic, object_points, rvec, tvec) if extrinsic_ok and intrinsic else None
        if reprojection is not None:
            overlay = draw_projection_overlay(cv2, frame, points, reprojection, f"{label.upper()} REPROJECTION")
        else:
            title = f"{label.upper()} REPROJECTION UNAVAILABLE"
            if not extrinsic_ok and extrinsic.get("error"):
                title = f"{title}: {extrinsic.get('error')}"
            overlay = draw_labeled_points(cv2, frame, points, (80, 80, 255), title)
        result_images[f"{label}_reprojection"] = write(overlay, f"extrinsic_{label}_reprojection.jpg")

    return result_images or None

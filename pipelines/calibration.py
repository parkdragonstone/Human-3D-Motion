"""
Camera calibration utilities (intrinsic / extrinsic).

- Intrinsic: chessboard video -> cameraMatrix + distCoeffs (OpenCV)
- Extrinsic: two videos observing same ArUco markers -> relative pose between cameras
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class IntrinsicResult:
    ok: bool
    error: str | None = None
    rms: float | None = None
    image_size: tuple[int, int] | None = None  # (w,h)
    camera_matrix: np.ndarray | None = None    # 3x3
    dist_coeffs: np.ndarray | None = None      # (k,)
    used_frames: int = 0
    used_corners: int = 0
    frames_read: int = 0
    frames_checked: int = 0
    frames_found: int = 0

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "error": self.error,
            "rms": self.rms,
            "image_size": list(self.image_size) if self.image_size else None,
            "camera_matrix": self.camera_matrix.tolist() if self.camera_matrix is not None else None,
            "dist_coeffs": self.dist_coeffs.reshape(-1).tolist() if self.dist_coeffs is not None else None,
            "used_frames": int(self.used_frames),
            "used_corners": int(self.used_corners),
            "frames_read": int(self.frames_read),
            "frames_checked": int(self.frames_checked),
            "frames_found": int(self.frames_found),
        }


def _ensure_cv2():
    try:
        import cv2  # type: ignore
    except Exception as e:
        raise RuntimeError(f"OpenCV not available: {e}")
    return cv2


def calibrate_intrinsic_from_video(
    video_path: str,
    board_cols: int,
    board_rows: int,
    square_size_mm: float,
    checker_board_type: str = "auto",
    sample_every: int = 10,
    max_samples: int = 60,
    debug_out_path: str | None = None,
    aruco_dictionary: str = "DICT_4X4_50",
    marker_size_mm: float | None = None,
    ) -> IntrinsicResult:
    """
    Parameters
    ----------
    board_cols / board_rows: chessboard inner corners (e.g. 9x6)
    square_size_mm: real size per square (same unit used for output; mm recommended)
    """
    _ensure_cv2()

    board_cols = int(board_cols)
    board_rows = int(board_rows)
    checker_board_type = _checker_board_type(checker_board_type)
    if checker_board_type == "charuco":
        return _calibrate_charuco_intrinsic_from_video(
            video_path,
            board_cols,
            board_rows,
            square_size_mm,
            marker_size_mm or square_size_mm * 0.72,
            aruco_dictionary,
            sample_every,
            max_samples,
            debug_out_path,
        )
    if board_cols < 3 or board_rows < 3:
        return IntrinsicResult(ok=False, error="board_size_too_small")
    if square_size_mm <= 0:
        return IntrinsicResult(ok=False, error="bad_square_size")

    return _calibrate_chessboard_intrinsic_from_video(
        video_path,
        board_cols,
        board_rows,
        square_size_mm,
        sample_every,
        max_samples,
        debug_out_path,
    )


def _calibrate_chessboard_intrinsic_from_video(
    video_path: str,
    board_cols: int,
    board_rows: int,
    square_size_mm: float,
    sample_every: int,
    max_samples: int,
    debug_out_path: str | None,
) -> IntrinsicResult:
    cv2 = _ensure_cv2()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return IntrinsicResult(ok=False, error=f"cannot_open_video: {video_path}")

    objp = np.zeros((board_rows * board_cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:board_cols, 0:board_rows].T.reshape(-1, 2)
    objp *= float(square_size_mm)

    objpoints: list[np.ndarray] = []
    imgpoints: list[np.ndarray] = []

    used = 0
    corners_total = 0
    frames_read = 0
    frames_checked = 0
    frames_found = 0

    frame_idx = 0
    image_size = None

    # prefer SB (more robust) if available
    find_sb = getattr(cv2, "findChessboardCornersSB", None)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    dbg_frames = [] if debug_out_path else None
    try:
        while used < max_samples:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            frame_idx += 1
            frames_read += 1
            if sample_every > 1 and (frame_idx % sample_every) != 0:
                continue

            frames_checked += 1
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape[:2]
            if image_size is None:
                image_size = (w, h)

            pattern = (board_cols, board_rows)
            found = False
            corners = None

            if callable(find_sb):
                try:
                    # OpenCV API: (ret, corners) or (ret, corners, meta)
                    sb_ret = find_sb(gray, pattern)
                    if isinstance(sb_ret, tuple):
                        if len(sb_ret) >= 2:
                            found = bool(sb_ret[0])
                            corners = sb_ret[1]
                        else:
                            found = False
                            corners = None
                    else:
                        # older/odd builds: return only corners
                        corners = sb_ret
                        found = corners is not None
                except Exception:
                    found = False
                    corners = None

            if not found:
                try:
                    found, corners = cv2.findChessboardCorners(
                        gray,
                        pattern,
                        flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE,
                    )
                except Exception:
                    found = False
                    corners = None
                if found and corners is not None:
                    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

            if not found or corners is None:
                continue

            frames_found += 1
            objpoints.append(objp.copy())
            imgpoints.append(corners.astype(np.float32))
            # debug frame
            if dbg_frames is not None:
                dbg = frame.copy()
                try:
                    cv2.drawChessboardCorners(dbg, pattern, corners, True)
                except Exception:
                    pass
                dbg_frames.append(dbg)
            used += 1
            corners_total += int(corners.shape[0])
    finally:
        cap.release()

    if used < 8:
        return IntrinsicResult(
            ok=False,
            error=f"not_enough_samples: {used}",
            used_frames=used,
            used_corners=corners_total,
            frames_read=frames_read,
            frames_checked=frames_checked,
            frames_found=frames_found,
        )

    try:
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            objpoints,
            imgpoints,
            image_size,
            None,
            None,
        )
        # write debug video if requested
        if dbg_frames:
            try:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                h, w = dbg_frames[0].shape[:2]
                vw = cv2.VideoWriter(debug_out_path, fourcc, 10.0, (w, h))
                for f in dbg_frames:
                    if f.shape[0] != h or f.shape[1] != w:
                        f = cv2.resize(f, (w, h))
                    vw.write(f)
                vw.release()
            except Exception:
                pass
        # ret == RMS reprojection error
        return IntrinsicResult(
            ok=True,
            rms=float(ret),
            image_size=image_size,
            camera_matrix=np.asarray(mtx, dtype=np.float64),
            dist_coeffs=np.asarray(dist, dtype=np.float64).reshape(-1),
            used_frames=used,
            used_corners=corners_total,
            frames_read=frames_read,
            frames_checked=frames_checked,
            frames_found=frames_found,
        )
    except Exception as e:
        return IntrinsicResult(
            ok=False,
            error=f"calibrate_failed: {e}",
            used_frames=used,
            used_corners=corners_total,
            frames_read=frames_read,
            frames_checked=frames_checked,
            frames_found=frames_found,
        )


def run_calibration_folder(folder_path: str, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    folder = Path(folder_path)
    if not folder.is_dir():
        raise ValueError(f"calibration_folder_not_found: {folder_path}")

    mode, project_name = _folder_mode_and_project(folder)
    if mode == "INTR":
        return run_intrinsic_calibration_folder(folder, project_name, metadata)
    if mode == "EXTR":
        return run_extrinsic_calibration_folder(folder, project_name, metadata)
    raise ValueError(f"unknown_calibration_mode: {mode}")


def run_intrinsic_calibration_folder(
    folder: Path,
    project_name: str,
    metadata_override: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metadata = _read_metadata(folder)
    if metadata_override:
        metadata.update({key: value for key, value in metadata_override.items() if value not in (None, "")})
    _require_calibration_metadata(metadata)
    board_cols = int(metadata["checker_board_columns"])
    board_rows = int(metadata["checker_board_rows"])
    square_size_mm = float(metadata["checker_board_size_mm"])
    checker_board_type = _checker_board_type(str(metadata.get("checker_board_type", "auto")))
    aruco_dictionary = str(metadata.get("aruco_dictionary", "DICT_4X4_50"))
    marker_size_mm = float(metadata.get("marker_size_mm") or square_size_mm * 0.72)
    videos = _video_files(folder)
    if not videos:
        raise ValueError("calibration_videos_not_found")

    intrinsics: dict[str, Any] = {}
    for video in videos:
        camera_label = _camera_label_from_path(video)
        debug_path = folder / f"intrinsic_debug_{camera_label}.mp4"
        result = calibrate_intrinsic_from_video(
            str(video),
            board_cols,
            board_rows,
            square_size_mm,
            checker_board_type=checker_board_type,
            aruco_dictionary=aruco_dictionary,
            marker_size_mm=marker_size_mm,
            debug_out_path=str(debug_path),
        )
        intrinsics[camera_label] = {
            "video": video.name,
            **result.to_jsonable(),
        }

    payload: dict[str, Any] = {
        "ok": any(item.get("ok") for item in intrinsics.values()),
        "mode": "INTR",
        "project_name": project_name,
        "checker_board_size_mm": square_size_mm,
        "checker_board_type": checker_board_type,
        "aruco_dictionary": aruco_dictionary,
        "marker_size_mm": marker_size_mm,
        "checker_board_columns": board_cols,
        "checker_board_rows": board_rows,
        "intrinsics": intrinsics,
    }

    output_path = folder / "intrinsic_calibration.json"
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return {"ok": payload["ok"], "output_path": str(output_path), **payload}


def run_extrinsic_calibration_folder(
    folder: Path,
    project_name: str,
    metadata_override: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metadata = _read_metadata(folder)
    if metadata_override:
        metadata.update({key: value for key, value in metadata_override.items() if value not in (None, "")})

    intrinsic_bundle = metadata.get("intrinsic_calibration")
    if not isinstance(intrinsic_bundle, dict):
        return {
            "ok": False,
            "mode": "EXTR",
            "project_name": project_name,
            "error": "intrinsic_calibration_upload_required",
        }

    videos = _video_files(folder)
    if len(videos) < 2:
        return {
            "ok": False,
            "mode": "EXTR",
            "project_name": project_name,
            "error": f"need_at_least_2_extrinsic_videos: {len(videos)}",
        }

    calibration_videos = videos[:4]
    video_by_label = {_camera_label_from_path(video): video for video in calibration_videos}
    camera_labels = list(video_by_label.keys())
    intrinsics_by_label = {
        label: _intrinsic_for_label(intrinsic_bundle, label)
        for label in camera_labels
    }
    missing_intrinsics = [label for label, intrinsic in intrinsics_by_label.items() if intrinsic is None]
    if missing_intrinsics:
        return {
            "ok": False,
            "mode": "EXTR",
            "project_name": project_name,
            "error": f"intrinsic_not_found_for_cameras: {', '.join(missing_intrinsics)}",
        }

    object_points = _parse_object_points(metadata.get("object_points"))
    image_points_by_camera = _parse_image_points_by_camera(metadata, camera_labels)
    if object_points and all(image_points_by_camera.get(label) for label in camera_labels):
        extrinsic = calibrate_extrinsic_scene_from_points_multi(
            {label: intrinsic for label, intrinsic in intrinsics_by_label.items() if intrinsic is not None},
            object_points,
            image_points_by_camera,
        )
        result_images = None
        try:
            result_images = _save_extrinsic_scene_result_images_multi(
                video_by_label,
                {label: intrinsic for label, intrinsic in intrinsics_by_label.items() if intrinsic is not None},
                image_points_by_camera,
                object_points,
                extrinsic,
            )
        except Exception:
            result_images = None
        payload: dict[str, Any] = {
            "ok": bool(extrinsic.get("ok")),
            "mode": "EXTR",
            "project_name": project_name,
            "camera_labels": camera_labels,
            "videos": [video.name for video in calibration_videos],
            "object_points": object_points,
            "image_points_by_camera": image_points_by_camera,
            "intrinsic_calibration": intrinsic_bundle,
            "extrinsic": extrinsic,
        }
        for index, label in enumerate(camera_labels, start=1):
            payload[f"image_points_cam{index}"] = image_points_by_camera.get(label, [])
        if result_images is not None:
            payload["result_images"] = result_images
        if not payload["ok"]:
            payload["error"] = str(extrinsic.get("error") or "extrinsic_calibration_failed")
        output_path = folder / "extrinsic_calibration.json"
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return {"output_path": str(output_path), **payload}

    marker_preset = str(metadata.get("aruco_dictionary") or intrinsic_bundle.get("aruco_dictionary") or "DICT_4X4_50")
    marker_length_mm = float(
        metadata.get("marker_size_mm")
        or intrinsic_bundle.get("marker_size_mm")
        or intrinsic_bundle.get("checker_board_size_mm")
        or 30.0
    )
    extrinsic = calibrate_extrinsic_from_aruco_videos(
        str(calibration_videos[0]),
        str(calibration_videos[1]),
        intrinsics_by_label[camera_labels[0]],
        intrinsics_by_label[camera_labels[1]],
        marker_preset=marker_preset,
        marker_length_mm=marker_length_mm,
    )

    payload: dict[str, Any] = {
        "ok": bool(extrinsic.get("ok")),
        "mode": "EXTR",
        "project_name": project_name,
        "camera_labels": camera_labels[:2],
        "videos": [video.name for video in calibration_videos[:2]],
        "marker_preset": marker_preset,
        "marker_length_mm": marker_length_mm,
        "object_points": object_points,
        "intrinsic_calibration": intrinsic_bundle,
        "extrinsic": extrinsic,
    }
    if not payload["ok"]:
        payload["error"] = str(extrinsic.get("error") or "extrinsic_calibration_failed")

    output_path = folder / "extrinsic_calibration.json"
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return {"output_path": str(output_path), **payload}


def _calibrate_charuco_intrinsic_from_video(
    video_path: str,
    board_cols: int,
    board_rows: int,
    square_size_mm: float,
    marker_size_mm: float,
    dictionary_name: str,
    sample_every: int,
    max_samples: int,
    debug_out_path: str | None,
) -> IntrinsicResult:
    cv2 = _ensure_cv2()
    if not hasattr(cv2, "aruco"):
        return IntrinsicResult(ok=False, error="opencv_aruco_not_available")
    if board_cols < 2 or board_rows < 2:
        return IntrinsicResult(ok=False, error="board_size_too_small")
    if square_size_mm <= 0 or marker_size_mm <= 0 or marker_size_mm >= square_size_mm:
        return IntrinsicResult(ok=False, error="bad_charuco_size")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return IntrinsicResult(ok=False, error=f"cannot_open_video: {video_path}")

    aruco_dict = _aruco_dict_from_preset(cv2, dictionary_name)
    squares_x = board_cols + 1
    squares_y = board_rows + 1
    board = _create_charuco_board(cv2, squares_x, squares_y, square_size_mm, marker_size_mm, aruco_dict)
    charuco_detector = _charuco_detector(cv2, board)
    detector = _aruco_detector(cv2, aruco_dict)
    if charuco_detector is None and not hasattr(cv2.aruco, "interpolateCornersCharuco"):
        return IntrinsicResult(ok=False, error="opencv_charuco_detector_not_available")
    objpoints = []
    imgpoints = []
    frames_read = 0
    frames_checked = 0
    frames_found = 0
    image_size = None
    debug_frames = [] if debug_out_path else None
    frame_idx = 0

    try:
        while frames_found < max_samples:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            frame_idx += 1
            frames_read += 1
            if sample_every > 1 and (frame_idx % sample_every) != 0:
                continue
            frames_checked += 1
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape[:2]
            if image_size is None:
                image_size = (w, h)
            if charuco_detector is not None:
                detected = charuco_detector.detectBoard(gray)
                charuco_corners = detected[0] if len(detected) > 0 else None
                charuco_ids = detected[1] if len(detected) > 1 else None
                count = 0 if charuco_ids is None else len(charuco_ids)
            else:
                marker_corners, marker_ids = _detect_aruco_markers(cv2, detector, gray, aruco_dict)
                if marker_ids is None or len(marker_ids) == 0:
                    continue
                count, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
                    marker_corners,
                    marker_ids,
                    gray,
                    board,
                )
            if charuco_ids is None or charuco_corners is None or int(count) < 4:
                continue
            objp, imgp = board.matchImagePoints(charuco_corners, charuco_ids)
            if objp is None or imgp is None or len(objp) < 4:
                continue
            frames_found += 1
            objpoints.append(np.asarray(objp, dtype=np.float32).reshape(-1, 3))
            imgpoints.append(np.asarray(imgp, dtype=np.float32).reshape(-1, 1, 2))
            if debug_frames is not None:
                dbg = frame.copy()
                try:
                    cv2.aruco.drawDetectedCornersCharuco(dbg, charuco_corners, charuco_ids)
                except Exception:
                    pass
                debug_frames.append(dbg)
    finally:
        cap.release()

    if frames_found < 6:
        return IntrinsicResult(
            ok=False,
            error=f"not_enough_samples: {frames_found}",
            frames_read=frames_read,
            frames_checked=frames_checked,
            frames_found=frames_found,
        )

    try:
        ret, mtx, dist, _rvecs, _tvecs = cv2.calibrateCamera(
            objpoints,
            imgpoints,
            image_size,
            None,
            None,
        )
        _write_debug_video(cv2, debug_out_path, debug_frames)
        return IntrinsicResult(
            ok=True,
            rms=float(ret),
            image_size=image_size,
            camera_matrix=np.asarray(mtx, dtype=np.float64),
            dist_coeffs=np.asarray(dist, dtype=np.float64).reshape(-1),
            used_frames=frames_found,
            used_corners=int(sum(len(points) for points in imgpoints)),
            frames_read=frames_read,
            frames_checked=frames_checked,
            frames_found=frames_found,
        )
    except Exception as exc:
        return IntrinsicResult(
            ok=False,
            error=f"charuco_calibrate_failed: {exc}",
            frames_read=frames_read,
            frames_checked=frames_checked,
            frames_found=frames_found,
        )


def _aruco_dict_from_preset(cv2, preset: str):
    preset = (preset or "").strip().lower()
    if preset in ("dict_4x4_50", "aruco_4x4_50", "charuco_4x4_50"):
        return cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    if preset in ("dict_5x5_100", "aruco_5x5_100"):
        return cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
    if preset in ("dict_6x6_250", "aruco_6x6_250"):
        return cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    # fallback
    return cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)


def _create_charuco_board(cv2, squares_x, squares_y, square_size, marker_size, aruco_dict):
    if hasattr(cv2.aruco, "CharucoBoard_create"):
        return cv2.aruco.CharucoBoard_create(squares_x, squares_y, square_size, marker_size, aruco_dict)
    return cv2.aruco.CharucoBoard((squares_x, squares_y), square_size, marker_size, aruco_dict)


def _aruco_detector(cv2, aruco_dict):
    if hasattr(cv2.aruco, "ArucoDetector"):
        params = cv2.aruco.DetectorParameters()
        return cv2.aruco.ArucoDetector(aruco_dict, params)
    return None


def _charuco_detector(cv2, board):
    if hasattr(cv2.aruco, "CharucoDetector"):
        return cv2.aruco.CharucoDetector(board)
    return None


def _detect_aruco_markers(cv2, detector, gray, aruco_dict):
    if detector is not None:
        corners, ids, _rejected = detector.detectMarkers(gray)
        return corners, ids
    params = cv2.aruco.DetectorParameters_create()
    corners, ids, _rejected = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=params)
    return corners, ids


def _write_debug_video(cv2, debug_out_path: str | None, frames) -> None:
    if not debug_out_path or not frames:
        return
    try:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        h, w = frames[0].shape[:2]
        writer = cv2.VideoWriter(debug_out_path, fourcc, 10.0, (w, h))
        for frame in frames:
            if frame.shape[0] != h or frame.shape[1] != w:
                frame = cv2.resize(frame, (w, h))
            writer.write(frame)
        writer.release()
    except Exception:
        pass


def _first_video_frame(cv2, video_path: str):
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


def _draw_labeled_points(cv2, frame, points, color, title: str):
    canvas = frame.copy()
    for index, point in enumerate(points or [], start=1):
        try:
            u = int(round(float(point.get("u"))))
            v = int(round(float(point.get("v"))))
        except Exception:
            continue
        cv2.circle(canvas, (u, v), 8, color, -1, lineType=cv2.LINE_AA)
        cv2.circle(canvas, (u, v), 12, (255, 255, 255), 2, lineType=cv2.LINE_AA)
        cv2.putText(
            canvas,
            str(point.get("id", index)),
            (u + 10, v - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    cv2.putText(
        canvas,
        title,
        (18, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (215, 255, 67),
        2,
        cv2.LINE_AA,
    )
    return canvas


def _project_object_points(cv2, intr: dict[str, Any], object_points: list[dict[str, Any]], rvec_value, tvec_value):
    K = np.asarray(intr.get("camera_matrix"), dtype=np.float64)
    D = np.asarray(intr.get("dist_coeffs"), dtype=np.float64).reshape(-1, 1)
    rvec_value = np.asarray(rvec_value, dtype=np.float64).reshape(-1)
    tvec_value = np.asarray(tvec_value, dtype=np.float64).reshape(-1)
    if rvec_value.size != 3 or tvec_value.size != 3:
        return None
    rvec = rvec_value.reshape(3, 1)
    tvec = tvec_value.reshape(3, 1)
    obj = np.asarray([[float(p.get("x")), float(p.get("y")), float(p.get("z"))] for p in object_points], dtype=np.float64).reshape(-1, 3)
    img_pts, _ = cv2.projectPoints(obj, rvec, tvec, K, D)
    projected = img_pts.reshape(-1, 2)
    return [
        {"id": point.get("id", index), "u": float(projected[index, 0]), "v": float(projected[index, 1])}
        for index, point in enumerate(object_points)
    ]


def _draw_projection_overlay(cv2, frame, actual_points, projected_points, title: str):
    canvas = frame.copy()
    actual_map = {}
    for point in actual_points or []:
        pid = point.get("id")
        if pid is not None:
            actual_map[str(pid)] = point
    for index, projected in enumerate(projected_points or [], start=1):
        try:
            u = int(round(float(projected.get("u"))))
            v = int(round(float(projected.get("v"))))
        except Exception:
            continue
        pid = projected.get("id", index)
        actual = actual_map.get(str(pid))
        if actual is not None:
            try:
                au = int(round(float(actual.get("u"))))
                av = int(round(float(actual.get("v"))))
                cv2.line(canvas, (au, av), (u, v), (255, 255, 255), 1, cv2.LINE_AA)
                cv2.circle(canvas, (au, av), 7, (80, 200, 80), -1, lineType=cv2.LINE_AA)
                cv2.circle(canvas, (u, v), 8, (255, 255, 255), 2, lineType=cv2.LINE_AA)
                cv2.putText(canvas, str(pid), (u + 10, v - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
            except Exception:
                pass
        else:
            cv2.circle(canvas, (u, v), 8, (255, 255, 255), 2, lineType=cv2.LINE_AA)
    cv2.putText(canvas, title, (18, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (215, 255, 67), 2, cv2.LINE_AA)
    cv2.putText(canvas, "green=clicked  white=reprojection", (18, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 230, 230), 1, cv2.LINE_AA)
    return canvas


def _save_extrinsic_scene_result_images_multi(
    video_by_label: dict[str, Path],
    intrinsics_by_label: dict[str, dict[str, Any]],
    image_points_by_camera: dict[str, list[dict[str, Any]]],
    object_points: list[dict[str, Any]],
    extrinsic: dict[str, Any],
) -> dict[str, str] | None:
    cv2 = _ensure_cv2()
    result_images: dict[str, str] = {}
    first_video = next(iter(video_by_label.values()), None)
    if first_video is None:
        return None
    extrinsic_ok = bool(extrinsic.get("ok"))

    def _write(image, name: str) -> str:
        out_path = first_video.parent / name
        cv2.imwrite(str(out_path), image)
        return str(out_path)

    for label, video in video_by_label.items():
        frame = _first_video_frame(cv2, str(video))
        points = image_points_by_camera.get(label, [])
        if frame is None:
            continue
        intrinsic = intrinsics_by_label.get(label)
        rvec = extrinsic.get(f"rvec_{label}")
        tvec = extrinsic.get(f"tvec_{label}")
        reproj = _project_object_points(cv2, intrinsic, object_points, rvec, tvec) if extrinsic_ok and intrinsic else None
        if reproj is not None:
            overlay = _draw_projection_overlay(cv2, frame, points, reproj, f"{label.upper()} REPROJECTION")
        else:
            title = f"{label.upper()} REPROJECTION UNAVAILABLE"
            if not extrinsic_ok and extrinsic.get("error"):
                title = f"{title}: {extrinsic.get('error')}"
            overlay = _draw_labeled_points(cv2, frame, points, (80, 80, 255), title)
        result_images[f"{label}_reprojection"] = _write(overlay, f"extrinsic_{label}_reprojection.jpg")

    return result_images or None


def _estimate_marker_poses_in_frame(cv2, gray, camera_matrix, dist_coeffs, aruco_dict, marker_length_mm: float):
    params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, params)
    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None or len(ids) == 0:
        return {}
    ids = ids.reshape(-1).astype(int)
    rvecs, tvecs, _obj = cv2.aruco.estimatePoseSingleMarkers(
        corners, float(marker_length_mm), camera_matrix, dist_coeffs
    )
    out = {}
    for i, mid in enumerate(ids):
        out[int(mid)] = (rvecs[i].reshape(3), tvecs[i].reshape(3))
    return out


def _rvec_tvec_to_T(cv2, rvec, tvec):
    R, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3))
    t = np.asarray(tvec, dtype=np.float64).reshape(3, 1)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3:] = t
    return T


def _avg_rotation(rot_mats: list[np.ndarray]) -> np.ndarray:
    # Markley / SVD based average of rotation matrices
    M = np.zeros((3, 3), dtype=np.float64)
    for R in rot_mats:
        M += R
    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    return R


def calibrate_extrinsic_from_aruco_videos(
    video_cam1: str,
    video_cam2: str,
    intr1: dict,
    intr2: dict,
    marker_preset: str = "aruco_4x4_50",
    marker_length_mm: float = 30.0,
    sample_every: int = 5,
    max_samples: int = 120,
    ) -> dict[str, Any]:
    """
    Returns relative transform from cam1 to cam2 (T_cam1_cam2).
    """
    cv2 = _ensure_cv2()
    if not hasattr(cv2, "aruco"):
        return {"ok": False, "error": "opencv_aruco_not_available"}

    try:
        K1 = np.asarray(intr1.get("camera_matrix"), dtype=np.float64)
        D1 = np.asarray(intr1.get("dist_coeffs"), dtype=np.float64).reshape(-1)
        K2 = np.asarray(intr2.get("camera_matrix"), dtype=np.float64)
        D2 = np.asarray(intr2.get("dist_coeffs"), dtype=np.float64).reshape(-1)
    except Exception as e:
        return {"ok": False, "error": f"bad_intrinsic_payload: {e}"}

    cap1 = cv2.VideoCapture(video_cam1)
    cap2 = cv2.VideoCapture(video_cam2)
    if not cap1.isOpened() or not cap2.isOpened():
        if cap1:
            cap1.release()
        if cap2:
            cap2.release()
        return {"ok": False, "error": "cannot_open_videos"}

    aruco_dict = _aruco_dict_from_preset(cv2, marker_preset)

    Ts: list[np.ndarray] = []
    used_frames = 0
    common_obs = 0
    i = 0

    try:
        while len(Ts) < max_samples:
            ok1, f1 = cap1.read()
            ok2, f2 = cap2.read()
            if not ok1 or not ok2 or f1 is None or f2 is None:
                break
            i += 1
            if sample_every > 1 and (i % sample_every) != 0:
                continue

            g1 = cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY)
            g2 = cv2.cvtColor(f2, cv2.COLOR_BGR2GRAY)

            poses1 = _estimate_marker_poses_in_frame(cv2, g1, K1, D1, aruco_dict, marker_length_mm)
            poses2 = _estimate_marker_poses_in_frame(cv2, g2, K2, D2, aruco_dict, marker_length_mm)

            common = set(poses1.keys()) & set(poses2.keys())
            if not common:
                continue

            used_frames += 1

            # for this frame, average over common markers
            frame_Ts = []
            for mid in sorted(common):
                r1, t1 = poses1[mid]
                r2, t2 = poses2[mid]
                T1m = _rvec_tvec_to_T(cv2, r1, t1)  # cam1 -> marker
                T2m = _rvec_tvec_to_T(cv2, r2, t2)  # cam2 -> marker
                Tm2 = np.linalg.inv(T2m)            # marker -> cam2
                T12 = T1m @ Tm2                     # cam1 -> cam2
                frame_Ts.append(T12)
                common_obs += 1

            if frame_Ts:
                # average within frame (rotation + translation)
                Rs = [T[:3, :3] for T in frame_Ts]
                ts = [T[:3, 3] for T in frame_Ts]
                Ravg = _avg_rotation(Rs)
                tavg = np.mean(np.stack(ts, axis=0), axis=0)
                Tavg = np.eye(4, dtype=np.float64)
                Tavg[:3, :3] = Ravg
                Tavg[:3, 3] = tavg
                Ts.append(Tavg)
    finally:
        cap1.release()
        cap2.release()

    if len(Ts) < 6:
        return {
            "ok": False,
            "error": f"not_enough_common_marker_frames: {len(Ts)}",
            "used_frames": used_frames,
            "common_obs": common_obs,
        }

    Rfinal = _avg_rotation([T[:3, :3] for T in Ts])
    tfinal = np.mean(np.stack([T[:3, 3] for T in Ts], axis=0), axis=0)
    Tfinal = np.eye(4, dtype=np.float64)
    Tfinal[:3, :3] = Rfinal
    Tfinal[:3, 3] = tfinal

    return {
        "ok": True,
        "used_frames": used_frames,
        "common_obs": common_obs,
        "T_cam1_cam2": Tfinal.tolist(),
        "R_cam1_cam2": Rfinal.tolist(),
        "t_cam1_cam2": tfinal.tolist(),
    }


def _scene_point_map(points, keys):
    out = {}
    for item in (points or []):
        pid = item.get("id")
        if pid in (None, ""):
            continue
        values = []
        ok = True
        for key in keys:
            try:
                values.append(float(item.get(key)))
            except Exception:
                ok = False
                break
        if ok:
            out[pid] = values
    return out


def _solve_scene_pnp(cv2, objp3, img2d, K, D):
    for flag in (cv2.SOLVEPNP_ITERATIVE, cv2.SOLVEPNP_EPNP):
        try:
            ok, rvec, tvec, inliers = cv2.solvePnPRansac(
                objp3,
                img2d,
                K,
                D,
                flags=flag,
                reprojectionError=6.0,
                confidence=0.999,
                iterationsCount=200,
            )
            if ok:
                return ok, rvec, tvec, inliers
        except Exception:
            pass
    try:
        if hasattr(cv2, "SOLVEPNP_AP3P"):
            ok, rvec, tvec = cv2.solvePnP(objp3, img2d, K, D, flags=cv2.SOLVEPNP_AP3P)
            if ok:
                return ok, rvec, tvec, None
    except Exception:
        pass
    try:
        z_range = float(np.ptp(objp3[:, 2]))
        obj_scale = max(float(np.max(np.ptp(objp3, axis=0))), 1e-12)
        if z_range <= max(1e-6, 1e-3 * obj_scale) and hasattr(cv2, "SOLVEPNP_IPPE"):
            ok, rvec, tvec = cv2.solvePnP(objp3, img2d, K, D, flags=cv2.SOLVEPNP_IPPE)
            if ok:
                return ok, rvec, tvec, None
    except Exception:
        pass
    return False, None, None, None


def _scene_reproj_rms(cv2, K, D, rvec, tvec, objp3, imgp2, inliers):
    proj, _ = cv2.projectPoints(objp3, rvec, tvec, K, D)
    diff = proj.reshape(-1, 2) - imgp2
    if inliers is not None and len(inliers) > 0:
        diff = diff[inliers.reshape(-1).astype(int)]
    return float(np.sqrt(np.mean(np.sum(diff * diff, axis=1)))) if diff.size else 0.0


def _scene_reproj_rms_cm(cv2, K, D, rvec, tvec, objp3, imgp2, inliers):
    fx = float(K[0, 0])
    fy = float(K[1, 1])
    R, _ = cv2.Rodrigues(rvec)
    t = np.asarray(tvec, dtype=np.float64).reshape(3, 1)
    Xc = (R @ objp3.T + t).T
    Z = np.maximum(1e-6, Xc[:, 2])
    proj, _ = cv2.projectPoints(objp3, rvec, tvec, K, D)
    diff = proj.reshape(-1, 2) - imgp2
    if inliers is not None and len(inliers) > 0:
        idx = inliers.reshape(-1).astype(int)
        diff = diff[idx]
        Z = Z[idx]
    dx_m = (Z / fx) * diff[:, 0]
    dy_m = (Z / fy) * diff[:, 1]
    d_m = np.sqrt(dx_m * dx_m + dy_m * dy_m)
    rms_m = float(np.sqrt(np.mean(d_m * d_m))) if d_m.size else 0.0
    return rms_m * 100.0


def calibrate_extrinsic_scene_from_points_multi(
    intrinsics_by_label: dict[str, dict[str, Any]],
    object_points: list[dict],
    image_points_by_camera: dict[str, list[dict]],
    ) -> dict[str, Any]:
    cv2 = _ensure_cv2()
    if not isinstance(object_points, list) or len(object_points) < 6:
        return {"ok": False, "error": "need_at_least_6_object_points"}

    obj = _scene_point_map(object_points, ("x", "y", "z"))
    cameras: dict[str, dict[str, Any]] = {}
    transforms: dict[str, np.ndarray] = {}
    errors: dict[str, str] = {}

    for label, intrinsic in intrinsics_by_label.items():
        try:
            K = np.asarray(intrinsic.get("camera_matrix"), dtype=np.float64)
            D = np.asarray(intrinsic.get("dist_coeffs"), dtype=np.float64).reshape(-1, 1)
        except Exception as exc:
            errors[label] = f"bad_intrinsic_payload: {exc}"
            continue

        img = _scene_point_map(image_points_by_camera.get(label, []), ("u", "v"))
        common = sorted(set(obj.keys()) & set(img.keys()))
        if len(common) < 6:
            errors[label] = f"not_enough_matched_points: {len(common)}"
            continue

        objp = np.asarray([obj[i] for i in common], dtype=np.float64).reshape(-1, 3)
        imgp = np.asarray([img[i] for i in common], dtype=np.float64).reshape(-1, 2)
        ok, rvec, tvec, inliers = _solve_scene_pnp(cv2, objp, imgp, K, D)
        if not ok:
            errors[label] = "solvepnp_not_ok"
            continue

        rms_px = _scene_reproj_rms(cv2, K, D, rvec, tvec, objp, imgp, inliers)
        rms_cm = _scene_reproj_rms_cm(cv2, K, D, rvec, tvec, objp, imgp, inliers)
        cameras[label] = {
            "ok": True,
            "rvec": np.asarray(rvec, dtype=np.float64).reshape(3).tolist(),
            "tvec": np.asarray(tvec, dtype=np.float64).reshape(3).tolist(),
            "matched_points": int(len(common)),
            "inliers": int(len(inliers) if inliers is not None else len(common)),
            "reproj_rms_px": rms_px,
            "reproj_rms_cm": rms_cm,
            "point_ids": common,
        }
        transforms[label] = _rvec_tvec_to_T(cv2, rvec, tvec)

    if len(cameras) < 2:
        return {
            "ok": False,
            "error": "need_at_least_2_solved_cameras",
            "camera_errors": errors,
            "cameras": cameras,
        }

    solved_labels = list(cameras.keys())
    result: dict[str, Any] = {
        "ok": True,
        "camera_labels": solved_labels,
        "cameras": cameras,
    }
    return result


def calibrate_extrinsic_scene_from_points(
    intr1: dict,
    intr2: dict,
    object_points: list[dict],
    image_points_cam1: list[dict],
    image_points_cam2: list[dict],
) -> dict[str, Any]:
    """
    Solve PnP for each camera from (3D object points) <-> (2D image points on first frame),
    then compute relative transform cam1->cam2.
    """
    cv2 = _ensure_cv2()

    try:
        K1 = np.asarray(intr1.get("camera_matrix"), dtype=np.float64)
        D1 = np.asarray(intr1.get("dist_coeffs"), dtype=np.float64).reshape(-1, 1)
        K2 = np.asarray(intr2.get("camera_matrix"), dtype=np.float64)
        D2 = np.asarray(intr2.get("dist_coeffs"), dtype=np.float64).reshape(-1, 1)
    except Exception as e:
        return {"ok": False, "error": f"bad_intrinsic_payload: {e}"}

    if not isinstance(object_points, list) or len(object_points) < 6:
        return {"ok": False, "error": "need_at_least_6_object_points"}

    def _map_points(lst, keys):
        out = {}
        for it in (lst or []):
            pid = it.get("id")
            if pid in (None, ""):
                continue
            ok = True
            vals = []
            for k in keys:
                v = it.get(k, None)
                if v is None:
                    ok = False
                    break
                try:
                    vals.append(float(v))
                except Exception:
                    ok = False
                    break
            if ok:
                out[pid] = vals
        return out

    obj = _map_points(object_points, ("x", "y", "z"))
    img1 = _map_points(image_points_cam1, ("u", "v"))
    img2 = _map_points(image_points_cam2, ("u", "v"))

    common = sorted(set(obj.keys()) & set(img1.keys()) & set(img2.keys()))
    if len(common) < 6:
        return {"ok": False, "error": f"not_enough_matched_points: {len(common)}"}

    objp = np.asarray([obj[i] for i in common], dtype=np.float64).reshape(-1, 3)
    ip1 = np.asarray([img1[i] for i in common], dtype=np.float64).reshape(-1, 2)
    ip2 = np.asarray([img2[i] for i in common], dtype=np.float64).reshape(-1, 2)

    # Use RANSAC for robustness
    def _solve_with_fallback(objp3, img2d, K, D):
        # 1) RANSAC ITERATIVE
        try:
            ok, rvec, tvec, inl = cv2.solvePnPRansac(
                objp3, img2d, K, D, flags=cv2.SOLVEPNP_ITERATIVE,
                reprojectionError=6.0, confidence=0.999, iterationsCount=200
            )
            if ok:
                return ok, rvec, tvec, inl
        except Exception:
            pass
        # 2) RANSAC EPNP
        try:
            ok, rvec, tvec, inl = cv2.solvePnPRansac(
                objp3, img2d, K, D, flags=cv2.SOLVEPNP_EPNP,
                reprojectionError=6.0, confidence=0.999, iterationsCount=200
            )
            if ok:
                return ok, rvec, tvec, inl
        except Exception:
            pass
        # 3) Non-RANSAC AP3P (general fallback; works well when ITERATIVE/RANSAC are unstable)
        try:
            if hasattr(cv2, "SOLVEPNP_AP3P"):
                ok, rvec, tvec = cv2.solvePnP(objp3, img2d, K, D, flags=cv2.SOLVEPNP_AP3P)
                if ok:
                    return ok, rvec, tvec, None
        except Exception:
            pass

        # 4) Non-RANSAC IPPE for coplanar
        try:
            # Relative coplanar check (object point units may be meters / mm depending on input).
            z_range = float(np.ptp(objp3[:, 2]))  # max-min
            obj_scale = float(np.max(np.ptp(objp3, axis=0)))
            obj_scale = max(obj_scale, 1e-12)
            # Consider coplanar if z variation is extremely small compared to overall object scale.
            # (tuned to be forgiving: ~1e-3 relative)
            coplanar = z_range <= max(1e-6, 1e-3 * obj_scale)

            if coplanar and hasattr(cv2, "SOLVEPNP_IPPE"):
                ok, rvec, tvec = cv2.solvePnP(objp3, img2d, K, D, flags=cv2.SOLVEPNP_IPPE)
                inl = None
                if ok:
                    return ok, rvec, tvec, inl
        except Exception:
            pass
        return False, None, None, None

    ok1, rvec1, tvec1, inliers1 = _solve_with_fallback(objp, ip1, K1, D1)
    ok2, rvec2, tvec2, inliers2 = _solve_with_fallback(objp, ip2, K2, D2)

    if not ok1 or not ok2:
        return {"ok": False, "error": "solvepnp_not_ok", "matched_points": int(len(common)), "common_ids": common}

    # Reprojection error (inlier-aware)
    def _reproj_rms(K, D, rvec, tvec, objp3, imgp2, inliers):
        proj, _ = cv2.projectPoints(objp3, rvec, tvec, K, D)
        proj = proj.reshape(-1, 2)
        diff = proj - imgp2
        if inliers is not None and len(inliers) > 0:
            idx = inliers.reshape(-1).astype(int)
            diff = diff[idx]
        err = np.sqrt(np.mean(np.sum(diff * diff, axis=1)))
        return float(err)

    def _reproj_rms_cm(K, D, rvec, tvec, objp3, imgp2, inliers):
        """
        픽셀 오차를 간단한 1차 근사로 mm→cm로 변환:
          dX_mm ≈ Z / fx * du,  dY_mm ≈ Z / fy * dv
        (Z는 각 포인트의 카메라 좌표계 깊이, fx/fy는 K)
        """
        fx = float(K[0, 0]); fy = float(K[1, 1])
        R, _ = cv2.Rodrigues(rvec)
        t = np.asarray(tvec, dtype=np.float64).reshape(3, 1)
        Xc = (R @ objp3.T + t).T  # Nx3, object-point units
        Z = np.maximum(1e-6, Xc[:, 2])
        proj, _ = cv2.projectPoints(objp3, rvec, tvec, K, D)
        proj = proj.reshape(-1, 2)
        diff = proj - imgp2  # px
        if inliers is not None and len(inliers) > 0:
            idx = inliers.reshape(-1).astype(int)
            diff = diff[idx]; Z = Z[idx]
        # Object points are entered in meters in the UI. Convert meters to centimeters.
        dx_m = (Z / fx) * diff[:, 0]
        dy_m = (Z / fy) * diff[:, 1]
        d_m = np.sqrt(dx_m * dx_m + dy_m * dy_m)
        rms_m = float(np.sqrt(np.mean(d_m * d_m))) if d_m.size else 0.0
        return rms_m * 100.0

    rms1 = _reproj_rms(K1, D1, rvec1, tvec1, objp, ip1, inliers1)
    rms2 = _reproj_rms(K2, D2, rvec2, tvec2, objp, ip2, inliers2)
    rms1_cm = _reproj_rms_cm(K1, D1, rvec1, tvec1, objp, ip1, inliers1)
    rms2_cm = _reproj_rms_cm(K2, D2, rvec2, tvec2, objp, ip2, inliers2)

    return {
        "ok": True,
        "rvec_cam1": np.asarray(rvec1, dtype=np.float64).reshape(3).tolist(),
        "tvec_cam1": np.asarray(tvec1, dtype=np.float64).reshape(3).tolist(),
        "rvec_cam2": np.asarray(rvec2, dtype=np.float64).reshape(3).tolist(),
        "tvec_cam2": np.asarray(tvec2, dtype=np.float64).reshape(3).tolist(),
        "matched_points": len(common),
        "inliers_cam1": int(len(inliers1) if inliers1 is not None else 0),
        "inliers_cam2": int(len(inliers2) if inliers2 is not None else 0),
        "reproj_rms_cam1_px": rms1,
        "reproj_rms_cam2_px": rms2,
        "reproj_rms_cam1_cm": rms1_cm,
        "reproj_rms_cam2_cm": rms2_cm,
        "point_ids": common,
    }


def _folder_mode_and_project(folder: Path) -> tuple[str, str]:
    parts = folder.name.split("_", 2)
    if len(parts) == 3 and parts[0] == "CALIB" and parts[1] in {"INTR", "EXTR"}:
        return parts[1], parts[2]
    raise ValueError(f"bad_calibration_folder_name: {folder.name}")


def _intrinsic_for_label(bundle: dict[str, Any], camera_label: str) -> dict[str, Any] | None:
    label = camera_label.lower()
    candidates = [
        bundle.get("intrinsics", {}).get(label) if isinstance(bundle.get("intrinsics"), dict) else None,
        bundle.get(label),
        bundle.get(f"intrinsic_{label}"),
        bundle.get(label.upper()),
        bundle.get(f"intrinsic_{label.upper()}"),
    ]
    for candidate in candidates:
        if isinstance(candidate, dict) and candidate.get("camera_matrix") is not None:
            return candidate
    return None


def _parse_object_points(value) -> list[dict[str, Any]]:
    if isinstance(value, list):
        return [point for point in value if isinstance(point, dict)]
    points: list[dict[str, Any]] = []
    for index, line in enumerate(str(value or "").splitlines()):
        parts = [part.strip() for part in line.split(",") if part.strip()]
        if len(parts) == 3:
            point_id = index
            coords = parts
        elif len(parts) >= 4:
            point_id = parts[0]
            coords = parts[1:4]
        else:
            continue
        try:
            points.append({
                "id": point_id,
                "x": float(coords[0]),
                "y": float(coords[1]),
                "z": float(coords[2]),
            })
        except ValueError:
            continue
    return points


def _parse_image_points(value) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    points: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        try:
            points.append({
                "id": item.get("id"),
                "u": float(item.get("u")),
                "v": float(item.get("v")),
            })
        except (TypeError, ValueError):
            continue
    return points


def _parse_image_points_by_camera(metadata: dict[str, Any], camera_labels: list[str]) -> dict[str, list[dict[str, Any]]]:
    raw = metadata.get("image_points_by_camera")
    points_by_camera: dict[str, list[dict[str, Any]]] = {}
    if isinstance(raw, dict):
        for label, points in raw.items():
            normalized = str(label).lower()
            points_by_camera[normalized] = _parse_image_points(points)

    for index, label in enumerate(camera_labels, start=1):
        if label not in points_by_camera:
            points_by_camera[label] = _parse_image_points(metadata.get(f"image_points_cam{index}"))
    return points_by_camera


def _checker_board_type(value: str) -> str:
    normalized = str(value or "auto").strip().lower()
    aliases = {
        "auto": "chessboard",
        "sb": "chessboard",
        "standard": "chessboard",
        "chess": "chessboard",
        "chessboard": "chessboard",
        "charuco": "charuco",
        "charucoboard": "charuco",
    }
    if normalized in aliases:
        return aliases[normalized]
    raise ValueError(f"unknown_checker_board_type: {value}")


def _read_metadata(folder: Path) -> dict[str, Any]:
    metadata_path = folder / "calibration.json"
    if not metadata_path.is_file():
        return {}
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def _require_calibration_metadata(metadata: dict[str, Any]) -> None:
    required = ["checker_board_columns", "checker_board_rows", "checker_board_size_mm"]
    missing = [key for key in required if metadata.get(key) in (None, "")]
    if missing:
        raise ValueError(f"calibration_metadata_missing: {', '.join(missing)}")


def _video_files(folder: Path) -> list[Path]:
    return sorted(
        path for path in folder.iterdir()
        if path.is_file()
        and path.suffix.lower() in {".mp4", ".mov", ".webm", ".avi"}
        and not path.stem.lower().startswith("intrinsic_debug_")
    )


def _camera_label_from_path(path: Path) -> str:
    import re

    match = re.search(r"(cam\d+)$", path.stem, re.IGNORECASE)
    return match.group(1).lower() if match else path.stem.lower()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run camera calibration for a CALIB_* folder.")
    parser.add_argument("folder", help="CALIB_INTR_* or CALIB_EXTR_* folder path")
    args = parser.parse_args()
    try:
        result = run_calibration_folder(args.folder)
    except Exception as exc:
        result = {"ok": False, "error": str(exc)}
        print(json.dumps(result, indent=2))
        sys.exit(1)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

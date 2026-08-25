"""
Camera calibration utilities (intrinsic / extrinsic).

- Intrinsic: chessboard video -> cameraMatrix + distCoeffs (OpenCV)
- Extrinsic: two videos observing same ArUco markers -> relative pose between cameras
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np

from .aruco import (
    aruco_detector as _aruco_detector,
    aruco_dict_from_preset as _aruco_dict_from_preset,
    charuco_detector as _charuco_detector,
    create_charuco_board as _create_charuco_board,
    detect_aruco_markers as _detect_aruco_markers,
    write_debug_video as _write_debug_video,
)
from .metadata import (
    camera_label_from_path as _camera_label_from_path,
    checker_board_type as _checker_board_type,
    folder_mode_and_project as _folder_mode_and_project,
    intrinsic_for_label as _intrinsic_for_label,
    parse_image_points_by_camera as _parse_image_points_by_camera,
    parse_object_points as _parse_object_points,
    read_metadata as _read_metadata,
    require_calibration_metadata as _require_calibration_metadata,
    video_files as _video_files,
)
from .models import IntrinsicResult
from .opencv import ensure_cv2 as _ensure_cv2
from .reprojection import (
    save_extrinsic_scene_result_images_multi as _save_extrinsic_scene_result_images_multi,
)


logger = logging.getLogger(__name__)


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
                    cv2.drawChessboardCorners(
                        dbg, pattern, np.asarray(corners, dtype=np.float32).reshape(-1, 1, 2), True
                    )
                except Exception as exc:
                    logger.warning(f"chessboard debug overlay failed: {exc}")
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
            "checker_board_type": metadata.get("checker_board_type"),
            "board_position": metadata.get("board_position") or metadata.get("chessboard_orientation"),
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
    # For ChArUco, Column/Row are the number of squares (a 10x7 board is 10 by 7
    # squares); Chessboard keeps counting inner corners. A 3x3 board is the smallest
    # that still yields the four charuco corners matchImagePoints needs.
    if board_cols < 3 or board_rows < 3:
        return IntrinsicResult(ok=False, error="board_size_too_small")
    if square_size_mm <= 0 or marker_size_mm <= 0 or marker_size_mm >= square_size_mm:
        return IntrinsicResult(ok=False, error="bad_charuco_size")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return IntrinsicResult(ok=False, error=f"cannot_open_video: {video_path}")

    aruco_dict = _aruco_dict_from_preset(cv2, dictionary_name)
    board = _create_charuco_board(cv2, board_cols, board_rows, square_size_mm, marker_size_mm, aruco_dict)
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
                    # OpenCV 5 hands back (N, 2) corners and (N,) ids, but the drawer
                    # still asserts the (N, 1, 2) layout OpenCV 4 used.
                    cv2.aruco.drawDetectedCornersCharuco(
                        dbg,
                        np.asarray(charuco_corners, dtype=np.float32).reshape(-1, 1, 2),
                        np.asarray(charuco_ids).reshape(-1, 1),
                    )
                except Exception as exc:
                    logger.warning(f"charuco debug overlay failed: {exc}")
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


def _is_axis_aligned_coplanar(objp3: np.ndarray) -> bool:
    axis_ranges = np.ptp(objp3, axis=0)
    obj_scale = max(float(np.max(axis_ranges)), 1e-12)
    return float(np.min(axis_ranges)) <= max(1e-6, 1e-3 * obj_scale)


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
        if _is_axis_aligned_coplanar(objp3) and hasattr(cv2, "SOLVEPNP_IPPE"):
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
    Compatibility wrapper for the old two-camera result shape.
    """
    result = calibrate_extrinsic_scene_from_points_multi(
        {"cam1": intr1, "cam2": intr2},
        object_points,
        {"cam1": image_points_cam1, "cam2": image_points_cam2},
    )
    if not result.get("ok"):
        return result

    cameras = result.get("cameras") or {}
    cam1 = cameras.get("cam1") or {}
    cam2 = cameras.get("cam2") or {}
    common_ids = sorted(set(cam1.get("point_ids") or []) & set(cam2.get("point_ids") or []))

    return {
        "ok": True,
        "rvec_cam1": cam1.get("rvec"),
        "tvec_cam1": cam1.get("tvec"),
        "rvec_cam2": cam2.get("rvec"),
        "tvec_cam2": cam2.get("tvec"),
        "matched_points": min(int(cam1.get("matched_points") or 0), int(cam2.get("matched_points") or 0)),
        "inliers_cam1": int(cam1.get("inliers") or 0),
        "inliers_cam2": int(cam2.get("inliers") or 0),
        "reproj_rms_cam1_px": cam1.get("reproj_rms_px"),
        "reproj_rms_cam2_px": cam2.get("reproj_rms_px"),
        "reproj_rms_cam1_cm": cam1.get("reproj_rms_cm"),
        "reproj_rms_cam2_cm": cam2.get("reproj_rms_cm"),
        "point_ids": common_ids,
    }


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

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

from pipelines.calibration.aruco import (
    aruco_detector,
    aruco_dict_from_preset,
    charuco_detector,
    create_charuco_board,
    detect_aruco_markers,
)


class OpenCvCalibrationBoardCornerDetector:
    def detect_first_match(self, path: str, config: dict[str, Any]) -> dict[str, Any]:
        import cv2

        video_path = Path(path)
        capture = cv2.VideoCapture(str(video_path))
        fallback_frame = None
        fallback_shape = None
        best_partial_frame = None
        best_partial_shape = None
        best_partial_index = 0
        best_partial_detection: dict[str, Any] | None = None
        best_partial_count = 0
        frame_index = 0
        last_detection: dict[str, Any] | None = None
        try:
            while True:
                ok, frame = capture.read()
                if not ok or frame is None:
                    break
                if fallback_frame is None:
                    fallback_frame = frame.copy()
                    fallback_shape = frame.shape[:2]

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                height, width = gray.shape[:2]
                board_type = str(config.get("checker_board_type") or "chessboard").strip().lower()
                if board_type == "charuco":
                    detected = _detect_charuco_frame(cv2, gray, config)
                else:
                    detected = _detect_chessboard_frame(cv2, gray, config)
                last_detection = detected
                point_count = len(detected.get("corners") or [])
                if point_count > best_partial_count:
                    best_partial_frame = frame.copy()
                    best_partial_shape = gray.shape[:2]
                    best_partial_index = frame_index
                    best_partial_detection = detected
                    best_partial_count = point_count
                if detected.get("found"):
                    return {
                        "width": int(width),
                        "height": int(height),
                        "frame_index": frame_index,
                        "frames_checked": frame_index + 1,
                        "image": _frame_data_url(cv2, frame, video_path.name),
                        **detected,
                    }
                frame_index += 1
        finally:
            capture.release()

        if fallback_frame is None or fallback_shape is None:
            raise ValueError(f"cannot_read_first_frame: {video_path.name}")

        if best_partial_frame is not None and best_partial_shape is not None and best_partial_detection is not None:
            height, width = best_partial_shape
            return {
                "width": int(width),
                "height": int(height),
                "frame_index": best_partial_index,
                "frames_checked": frame_index,
                "image": _frame_data_url(cv2, best_partial_frame, video_path.name),
                **best_partial_detection,
                "found": False,
                "error": best_partial_detection.get("error") or "partial_board_corners_found",
            }

        height, width = fallback_shape
        board_type = str(config.get("checker_board_type") or "chessboard").strip().lower()
        return {
            "width": int(width),
            "height": int(height),
            "frame_index": 0,
            "frames_checked": frame_index,
            "image": _frame_data_url(cv2, fallback_frame, video_path.name),
            "found": False,
            "checker_board_type": board_type,
            "corners": [],
            "error": (last_detection or {}).get("error") or f"{board_type}_corners_not_found",
        }


def _detect_chessboard_frame(cv2, gray, config: dict[str, Any]) -> dict[str, Any]:
    columns = int(config.get("checker_board_columns") or 0)
    rows = int(config.get("checker_board_rows") or 0)
    pattern = (columns, rows)
    found = False
    corners = None

    find_sb = getattr(cv2, "findChessboardCornersSB", None)
    if callable(find_sb):
        try:
            result = find_sb(gray, pattern)
            if isinstance(result, tuple) and len(result) >= 2:
                found = bool(result[0])
                corners = result[1]
            else:
                corners = result
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
            if found and corners is not None:
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        except Exception:
            found = False
            corners = None

    points = []
    if found and corners is not None:
        for index, corner in enumerate(corners.reshape(-1, 2), start=1):
            points.append({"id": index, "u": float(corner[0]), "v": float(corner[1])})

    return {
        "found": bool(found and len(points) == columns * rows),
        "checker_board_type": "chessboard",
        "corners": points,
        "error": None if points else "chessboard_corners_not_found",
    }


def _frame_data_url(cv2, frame, video_name: str) -> str:
    ok, buffer = cv2.imencode(".jpg", frame)
    if not ok:
        raise ValueError(f"cannot_encode_frame: {video_name}")
    encoded = base64.b64encode(buffer.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def _detect_charuco_frame(cv2, gray, config: dict[str, Any]) -> dict[str, Any]:
    if not hasattr(cv2, "aruco"):
        return {
            "found": False,
            "checker_board_type": "charuco",
            "corners": [],
            "error": "opencv_aruco_not_available",
        }

    columns = int(config.get("checker_board_columns") or 0)
    rows = int(config.get("checker_board_rows") or 0)
    square_size_mm = float(config.get("checker_board_size_mm") or 0)
    marker_size_mm = float(config.get("marker_size_mm") or square_size_mm * 0.72)
    # Column/Row count squares for ChArUco, so the inner corners the detector returns
    # are one fewer in each direction.
    if columns < 3 or rows < 3 or square_size_mm <= 0 or marker_size_mm <= 0 or marker_size_mm >= square_size_mm:
        return {
            "found": False,
            "checker_board_type": "charuco",
            "corners": [],
            "error": "bad_charuco_setup",
        }

    aruco_dict = aruco_dict_from_preset(cv2, str(config.get("aruco_dictionary") or "DICT_4X4_50"))
    board = create_charuco_board(cv2, columns, rows, square_size_mm, marker_size_mm, aruco_dict)
    detector = charuco_detector(cv2, board)
    marker_detector = aruco_detector(cv2, aruco_dict)

    if detector is not None:
        detected = detector.detectBoard(gray)
        charuco_corners = detected[0] if len(detected) > 0 else None
        charuco_ids = detected[1] if len(detected) > 1 else None
    elif hasattr(cv2.aruco, "interpolateCornersCharuco"):
        marker_corners, marker_ids = detect_aruco_markers(cv2, marker_detector, gray, aruco_dict)
        if marker_ids is None or len(marker_ids) == 0:
            charuco_corners = None
            charuco_ids = None
        else:
            _count, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
                marker_corners,
                marker_ids,
                gray,
                board,
            )
    else:
        return {
            "found": False,
            "checker_board_type": "charuco",
            "corners": [],
            "error": "opencv_charuco_detector_not_available",
        }

    points = []
    if charuco_corners is not None and charuco_ids is not None:
        ids = charuco_ids.reshape(-1)
        for marker_id, corner in zip(ids, charuco_corners.reshape(-1, 2)):
            points.append({"id": int(marker_id) + 1, "u": float(corner[0]), "v": float(corner[1])})

    return {
        "found": len(points) == (columns - 1) * (rows - 1),
        "checker_board_type": "charuco",
        "corners": points,
        "error": None if points else "charuco_corners_not_found",
    }

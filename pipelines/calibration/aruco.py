def aruco_dict_from_preset(cv2, preset: str):
    preset = (preset or "").strip().lower()
    if preset in ("dict_4x4_50", "aruco_4x4_50", "charuco_4x4_50"):
        return cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    if preset in ("dict_5x5_100", "aruco_5x5_100"):
        return cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
    if preset in ("dict_6x6_250", "aruco_6x6_250"):
        return cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    return cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)


def create_charuco_board(cv2, squares_x, squares_y, square_size, marker_size, aruco_dict):
    if hasattr(cv2.aruco, "CharucoBoard_create"):
        return cv2.aruco.CharucoBoard_create(squares_x, squares_y, square_size, marker_size, aruco_dict)
    return cv2.aruco.CharucoBoard((squares_x, squares_y), square_size, marker_size, aruco_dict)


def aruco_detector(cv2, aruco_dict):
    if hasattr(cv2.aruco, "ArucoDetector"):
        params = cv2.aruco.DetectorParameters()
        return cv2.aruco.ArucoDetector(aruco_dict, params)
    return None


def charuco_detector(cv2, board):
    if hasattr(cv2.aruco, "CharucoDetector"):
        return cv2.aruco.CharucoDetector(board)
    return None


def detect_aruco_markers(cv2, detector, gray, aruco_dict):
    if detector is not None:
        corners, ids, _rejected = detector.detectMarkers(gray)
        return corners, ids
    params = cv2.aruco.DetectorParameters_create()
    corners, ids, _rejected = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=params)
    return corners, ids


def write_debug_video(cv2, debug_out_path: str | None, frames) -> None:
    if not debug_out_path or not frames:
        return
    try:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        height, width = frames[0].shape[:2]
        writer = cv2.VideoWriter(debug_out_path, fourcc, 10.0, (width, height))
        for frame in frames:
            if frame.shape[0] != height or frame.shape[1] != width:
                frame = cv2.resize(frame, (width, height))
            writer.write(frame)
        writer.release()
    except Exception:
        pass

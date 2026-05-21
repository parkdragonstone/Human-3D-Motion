def ensure_cv2():
    try:
        import cv2  # type: ignore
    except Exception as exc:
        raise RuntimeError(f"OpenCV not available: {exc}") from exc
    return cv2

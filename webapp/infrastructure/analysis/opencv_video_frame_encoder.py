from __future__ import annotations

import base64
from pathlib import Path


class OpenCvVideoFrameEncoder:
    def first_frame_data_url(self, path: str) -> str:
        import cv2

        video_path = Path(path)
        capture = cv2.VideoCapture(str(video_path))
        try:
            ok, frame = capture.read()
        finally:
            capture.release()
        if not ok or frame is None:
            raise ValueError(f"cannot_read_first_frame: {video_path.name}")
        ok, buffer = cv2.imencode(".jpg", frame)
        if not ok:
            raise ValueError(f"cannot_encode_first_frame: {video_path.name}")
        encoded = base64.b64encode(buffer.tobytes()).decode("ascii")
        return f"data:image/jpeg;base64,{encoded}"

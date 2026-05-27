from __future__ import annotations


class OpenCvVideoMetadataReader:
    def read_metadata(self, path: str) -> tuple[float, list[int]]:
        import cv2

        capture = cv2.VideoCapture(path)
        try:
            fps = float(capture.get(cv2.CAP_PROP_FPS) or 0)
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        finally:
            capture.release()
        return fps, [width, height]

    def read_frame_count(self, path: str) -> int:
        import cv2

        capture = cv2.VideoCapture(path)
        try:
            return int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        finally:
            capture.release()

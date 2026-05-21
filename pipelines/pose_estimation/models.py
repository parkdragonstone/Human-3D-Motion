from __future__ import annotations

from pathlib import Path

import numpy as np
from ultralytics import YOLO


MODEL_DIR = Path(__file__).resolve().parents[1] / "models"


def setup_detector(device, det_score_threshold, det_iou, det_nms, mode: str = "normal"):
    """Setup pose detector based on config."""
    detector = YOLO(str(MODEL_DIR / mode / "yolo_ckpt.pt"))
    detector_cfg = {
        "device": device,
        "conf": det_score_threshold,
        "iou": det_iou,
        "nms": det_nms,
        "verbose": False,
    }
    return detector, detector_cfg


class WrappingDetector:
    def __init__(self, detector, detector_cfg):
        self.model = detector
        self.cfg = detector_cfg

    def __call__(self, frame_bgr: np.ndarray) -> np.ndarray:
        result = self.model.predict(frame_bgr, classes=[0], **self.cfg)[0]
        if result.boxes is None or len(result.boxes) == 0:
            return np.zeros((0, 4), dtype=np.float32)
        return result.boxes.xyxy.detach().cpu().numpy().astype(np.float32)


def setup_pose_solver(mode: str, backend: str, device: str):
    """Setup pose solver based on config."""
    pose_onnx = str(MODEL_DIR / mode / "rtmpose_end2end.onnx")
    try:
        from rtmlib.tools.pose_estimation.rtmpose import RTMPose
    except Exception:
        try:
            from rtmlib.tools.pose_estimation import RTMPose
        except Exception as exc:
            raise ImportError("Cannot import RTMPose from rtmlib. Please check your rtmlib version.") from exc
    return RTMPose(
        pose_onnx,
        model_input_size=(192, 256),
        to_openpose=False,
        backend=backend,
        device=device,
    )

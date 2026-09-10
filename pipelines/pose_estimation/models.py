from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from rtmlib.tools.object_detection.yolox import YOLOX


logger = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).resolve().parents[1] / "models"

# The bundled export is a fixed-shape [1, 3, 640, 640] graph, so this is the
# model geometry rather than a tunable.
DETECTOR_INPUT_SIZE = (640, 640)
_COCO_PERSON_CLASS = 0

# Kept as a safety net for execution providers that cannot run the graph. The
# bundled NMS-free export runs on CoreML, but an export with a baked-in NMS
# module does not: CoreML mis-infers the rank of its outputs and aborts, on
# onnxruntime 1.26 for any frame that detects nobody and from 1.27 for every
# frame. Empty frames are routine mid-recording, so such an export has to fall
# back to CPU rather than fail partway through a video.
_DETECTOR_FALLBACKS = (("openvino", "cpu"), ("onnxruntime", "cpu"))


class _PersonYOLOX(YOLOX):
    """YOLOX that returns people only, for either ONNX export shape.

    rtmlib's ``det_mode="human"`` does not actually filter by class on a raw
    multi-class export, so a skier comes back as two boxes; and on an export
    with a baked-in NMS module it applies a hard-coded 0.3 score cut that
    silently discards ``score_thr``. Both paths are fixed here so that
    ``det_score_threshold`` reaches the detector and only people come back.
    """

    def postprocess(self, outputs, ratio: float = 1.0):
        if outputs.shape[-1] == 5:
            # Baked-in NMS: person-only weights, so the score cut is all we owe.
            boxes = outputs[0, :, :4] / ratio
            scores = outputs[0, :, 4]
            return boxes[scores > self.score_thr]
        previous_mode, self.det_mode = self.det_mode, "multiclass"
        try:
            boxes, class_ids = super().postprocess(outputs, ratio)
        finally:
            self.det_mode = previous_mode
        return boxes[class_ids == _COCO_PERSON_CLASS] if len(boxes) else boxes


def setup_detector(device, det_score_threshold, mode: str = "normal", backend: str = "onnxruntime"):
    """Setup pose detector based on config.

    The bundled Human-Art graphs run NMS internally, so the score threshold is
    the only detector knob the config can still influence; rtmlib's own
    ``nms_thr`` default covers the unused non-baked-NMS code path.
    """
    detector_onnx = MODEL_DIR / mode / "detector_end2end.onnx"
    if not detector_onnx.is_file():
        raise FileNotFoundError(f"detector_model_not_found: {detector_onnx}")

    requested = (backend, device)
    candidates = [requested] + [c for c in _DETECTOR_FALLBACKS if c != requested]
    errors = []
    for candidate_backend, candidate_device in candidates:
        try:
            detector = _PersonYOLOX(
                str(detector_onnx),
                model_input_size=DETECTOR_INPUT_SIZE,
                det_mode="human",
                score_thr=det_score_threshold,
                backend=candidate_backend,
                device=candidate_device,
            )
            # Probe on a frame that detects nobody: execution providers can
            # fail only once a graph runs, and that is the input CoreML gets
            # wrong, so anything that cannot survive it would die mid-video.
            detector(np.zeros((*DETECTOR_INPUT_SIZE, 3), dtype=np.uint8))
        except Exception as exc:
            errors.append(f"{candidate_backend}/{candidate_device}: {exc}")
            continue
        if (candidate_backend, candidate_device) != requested:
            logger.warning(
                "Detector backend/device %s/%s unusable, falling back to %s/%s",
                backend, device, candidate_backend, candidate_device,
            )
        return detector, {}

    raise RuntimeError("detector_backend_unavailable: " + " | ".join(errors))


class WrappingDetector:
    def __init__(self, detector, detector_cfg):
        self.model = detector
        self.cfg = detector_cfg

    def __call__(self, frame_bgr: np.ndarray) -> np.ndarray:
        boxes = self.model(frame_bgr, **self.cfg)
        if boxes is None or len(boxes) == 0:
            return np.zeros((0, 4), dtype=np.float32)
        return np.asarray(boxes, dtype=np.float32).reshape(-1, 4)


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

from __future__ import annotations

import json
import re
from pathlib import Path

import cv2

from webapp.domain.entities import CaptureSession
from webapp.domain.ports import AnalysisRunner, LogEmitter


class AnalysisPipelineService:
    def __init__(self, analysis_runner: AnalysisRunner):
        self._analysis_runner = analysis_runner

    def default_config(self) -> dict:
        from pipelines.config import DEFAULT_CONFIG

        return DEFAULT_CONFIG

    def run(self, session: CaptureSession, user_config: dict, emit_log: LogEmitter) -> None:
        config = self.config_for_session(session, user_config)
        self._analysis_runner.run(config, emit_log)

    def config_for_session(self, session: CaptureSession, user_config: dict) -> dict:
        from pipelines.config import DEFAULT_CONFIG, deep_merge

        config = deep_merge(DEFAULT_CONFIG, user_config if isinstance(user_config, dict) else {})
        videos = sorted(session.videos, key=lambda video: video.camera_label)
        if len(videos) < 2:
            raise ValueError("analysis_requires_two_videos")

        config.setdefault("paths", {})
        config["paths"]["project_dir"] = str(Path(session.session_path).resolve())
        config.setdefault("base", {})
        config["base"].setdefault("frame_range", "auto")

        detected_fps = 0.0
        for video in videos[:4]:
            camera_path = Path(video.path).resolve()
            fps, resolution = _video_metadata(camera_path)
            detected_fps = detected_fps or fps
            camera_label = _normalize_camera_label(video.camera_label)
            config["paths"][camera_label] = str(camera_path)
            config["base"][f"resolution_{camera_label}"] = resolution

        config["base"]["fps"] = detected_fps or 30
        config["subject"] = {
            "name": session.subject.name,
            "height": session.subject.height_cm,
            "weight": session.subject.weight_kg,
            "hand": session.subject.hand,
        }

        calibration_path, calibration_bundle = _analysis_calibration_bundle(session)
        if calibration_path is not None:
            config.setdefault("lifting", {})
            config["lifting"]["camera_intrinsic_file"] = str(calibration_path)
        if calibration_bundle:
            config["calibration"] = calibration_bundle
        return config


def _normalize_camera_label(label: str) -> str:
    match = re.search(r"cam0*(\d+)$", str(label).lower())
    return f"cam{int(match.group(1))}" if match else str(label).lower()


def _analysis_calibration_bundle(session: CaptureSession) -> tuple[Path | None, dict | None]:
    for calibration_path in (
        Path(session.session_path) / "analysis_calibration.json",
        Path(session.session_path) / "extrinsic_calibration.json",
        Path(session.session_path) / "intrinsic_calibration.json",
    ):
        if not calibration_path.is_file():
            continue
        payload = json.loads(calibration_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            continue
        bundle: dict = {}
        extrinsic = payload.get("extrinsic")
        if isinstance(extrinsic, dict) and extrinsic.get("ok"):
            bundle.update(extrinsic)
            cameras = extrinsic.get("cameras")
            if isinstance(cameras, dict):
                normalized_cameras = {_normalize_camera_label(label): camera for label, camera in cameras.items()}
                bundle["cameras"] = normalized_cameras
                bundle["camera_labels"] = sorted(normalized_cameras.keys())
                camera_labels = [
                    _normalize_camera_label(label)
                    for label in extrinsic.get("camera_labels", [])
                    if _normalize_camera_label(label) in normalized_cameras
                ]
                if len(camera_labels) >= 2:
                    first = normalized_cameras.get(camera_labels[0])
                    second = normalized_cameras.get(camera_labels[1])
                    if isinstance(first, dict) and isinstance(second, dict):
                        bundle["rvec_cam1"] = first.get("rvec")
                        bundle["tvec_cam1"] = first.get("tvec")
                        bundle["rvec_cam2"] = second.get("rvec")
                        bundle["tvec_cam2"] = second.get("tvec")
        elif payload.get("rvec_cam1") is not None and payload.get("rvec_cam2") is not None:
            bundle.update(payload)

        intrinsics = payload.get("intrinsics")
        if not isinstance(intrinsics, dict):
            intrinsic_payload = payload.get("intrinsic_calibration")
            if isinstance(intrinsic_payload, dict):
                intrinsics = intrinsic_payload.get("intrinsics")
        if isinstance(intrinsics, dict):
            for label, intrinsic in intrinsics.items():
                if isinstance(intrinsic, dict):
                    bundle[str(label).lower()] = intrinsic
                    bundle[_normalize_camera_label(label)] = intrinsic
            cam1_intrinsic = intrinsics.get("cam1") or intrinsics.get("intrinsic_cam1")
            cam2_intrinsic = intrinsics.get("cam2") or intrinsics.get("intrinsic_cam2")
            if isinstance(cam1_intrinsic, dict):
                bundle["cam1"] = cam1_intrinsic
            if isinstance(cam2_intrinsic, dict):
                bundle["cam2"] = cam2_intrinsic
        if bundle:
            return calibration_path, bundle
    return None, None


def _video_metadata(path: Path) -> tuple[float, list[int]]:
    capture = cv2.VideoCapture(str(path))
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    finally:
        capture.release()
    return fps, [width, height]

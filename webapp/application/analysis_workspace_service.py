from __future__ import annotations

import json
from pathlib import Path
from typing import IO

from webapp.application.analysis_job_service import AnalysisJobService
from webapp.application.analysis_pipeline_service import AnalysisPipelineService
from webapp.application.analysis_result_service import AnalysisResultService
from webapp.application.capture_service import CaptureService
from webapp.application.session_query_service import SessionQueryService
from webapp.domain.entities import CaptureSession
from webapp.domain.ports import SessionCatalog


class AnalysisWorkspaceService:
    def __init__(
        self,
        capture_service: CaptureService,
        sessions: SessionCatalog,
        session_query_service: SessionQueryService,
        analysis_service: AnalysisPipelineService,
        analysis_job_service: AnalysisJobService,
        analysis_result_service: AnalysisResultService,
    ) -> None:
        self._capture_service = capture_service
        self._sessions = sessions
        self._session_query_service = session_query_service
        self._analysis_service = analysis_service
        self._analysis_job_service = analysis_job_service
        self._analysis_result_service = analysis_result_service

    def default_config(self) -> dict:
        return self._analysis_service.default_config()

    def page_root(self, requested_root: str | None) -> str:
        if requested_root:
            return self._capture_service.set_storage_root(requested_root)
        return self._capture_service.get_storage_root()

    def list_sessions(self, requested_root: str | None) -> list[CaptureSession]:
        root = str(requested_root or self._capture_service.get_storage_root()).strip()
        if root:
            root = self._capture_service.set_storage_root(root)
        return self._sessions.list_sessions(root)

    def video_file(self, raw_path: str) -> Path:
        path = Path(str(raw_path or "")).expanduser().resolve()
        if not path.is_file() or path.suffix.lower() not in {".mp4", ".mov", ".webm", ".avi"}:
            raise FileNotFoundError("video_not_found")
        return path

    def upload_calibration(
        self,
        session_path: str,
        filename: str,
        stream: IO[bytes] | None,
    ) -> dict:
        session = self._session_query_service.require_by_path(session_path)
        if not filename:
            raise ValueError("calibration_file_required")
        if Path(filename).suffix.lower() != ".json":
            raise ValueError("calibration_file_must_be_json")
        if stream is None:
            raise ValueError("calibration_file_required")

        payload = json.load(stream)
        if not isinstance(payload, dict):
            raise ValueError("calibration_json_must_be_object")
        mode = str(payload.get("mode") or ("EXTR" if payload.get("extrinsic") else "INTR")).upper()
        target = Path(session.session_path) / "analysis_calibration.json"
        target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return {"filename": filename, "mode": mode, "path": str(target)}

    def start_job(self, session_path: str, config: dict) -> dict:
        session = self._session_query_service.require_by_path(session_path)
        return self._analysis_job_service.start(session, config)

    def get_job(self, job_id: str) -> dict | None:
        return self._analysis_job_service.get(job_id)

    def list_pose3d_files(self, session_path: str) -> list[str]:
        session = self._session_query_service.require_by_path(session_path)
        return self._analysis_result_service.list_pose3d_files(Path(session.session_path))

    def pose3d_data(self, session_path: str, filename: str) -> dict:
        session = self._session_query_service.require_by_path(session_path)
        filename = str(filename or "").strip()
        if not filename or "/" in filename or "\\" in filename:
            raise ValueError("invalid_trc_file")
        trc_path = Path(session.session_path) / "pose-3d" / filename
        return self._analysis_result_service.pose3d_data_from_trc(trc_path)

    def keypoint_frame(self, session_path: str, camera_label: str, frame: int) -> dict:
        session = self._session_query_service.require_by_path(session_path)
        return self._analysis_result_service.keypoint_frame_from_json(session.session_path, camera_label, frame)

    def save_keypoint_frame(
        self,
        session_path: str,
        camera_label: str,
        frame: int,
        keypoints: list,
    ) -> None:
        session = self._session_query_service.require_by_path(session_path)
        if not isinstance(keypoints, list):
            raise ValueError("people_keypoints_required")
        self._analysis_result_service.save_keypoint_frame_to_json(
            session.session_path,
            camera_label,
            frame,
            keypoints,
        )

    def render_keypoint_video(self, session_path: str, camera_label: str) -> Path:
        session = self._session_query_service.require_by_path(session_path)
        return self._analysis_result_service.render_pose_video_from_keypoints(session, camera_label)

    def kinematics_summary(self, session_path: str) -> dict:
        session = self._session_query_service.require_by_path(session_path)
        return self._analysis_result_service.kinematics_summary(Path(session.session_path))

    def kinematics_timeseries(self, session_path: str, signal: str) -> dict:
        session = self._session_query_service.require_by_path(session_path)
        return self._analysis_result_service.kinematics_timeseries(Path(session.session_path), signal)

from __future__ import annotations

import os
from dataclasses import dataclass

from webapp.application import (
    AnalysisJobService,
    AnalysisPipelineService,
    AnalysisResultService,
    AnalysisWorkspaceService,
    CalibrationRecordingService,
    CalibrationService,
    CaptureRecordingService,
    CaptureService,
    MediaViewService,
    PhoneCaptureService,
    SessionQueryService,
    StorageRootService,
)
from webapp.infrastructure.analysis import (
    OpenCvVideoFrameEncoder,
    OpenCvVideoMetadataReader,
    PipelineAnalysisConfigProvider,
    PipelineAnalysisResultGateway,
    PipelineAnalysisRunner,
    PipelineCalibrationRunner,
)
from webapp.infrastructure.camera import ModeCameraController, PhoneCameraController, UrlCameraController
from webapp.infrastructure.persistence import FileSessionCatalog, JsonSettingsRepository
from webapp.infrastructure.system import TkinterDirectorySelector


@dataclass(frozen=True)
class AppServices:
    media_view_service: MediaViewService
    storage_root_service: StorageRootService
    capture_service: CaptureService
    capture_recording_service: CaptureRecordingService
    analysis_workspace_service: AnalysisWorkspaceService
    phone_service: PhoneCaptureService
    calibration_service: CalibrationService
    calibration_recording_service: CalibrationRecordingService


def create_app_services() -> AppServices:
    settings = JsonSettingsRepository(os.environ.get("BASEBALL_MOTION_SETTINGS", "webapp_data/settings.json"))
    sessions = FileSessionCatalog()
    session_query_service = SessionQueryService(sessions)
    camera_controller = _camera_controller_from_env(settings)
    video_metadata_reader = OpenCvVideoMetadataReader()
    media_view_service = MediaViewService(video_metadata_reader)
    video_frame_encoder = OpenCvVideoFrameEncoder()
    directory_selector = TkinterDirectorySelector()
    capture_service = CaptureService(camera_controller, sessions, settings)
    storage_root_service = StorageRootService(capture_service, directory_selector)
    analysis_service = AnalysisPipelineService(
        PipelineAnalysisRunner(),
        PipelineAnalysisConfigProvider(),
        video_metadata_reader,
    )
    analysis_job_service = AnalysisJobService(analysis_service)
    analysis_result_service = AnalysisResultService(PipelineAnalysisResultGateway())
    phone_service = PhoneCaptureService(settings)
    capture_recording_service = CaptureRecordingService(capture_service, phone_service)
    analysis_workspace_service = AnalysisWorkspaceService(
        capture_service,
        sessions,
        session_query_service,
        analysis_service,
        analysis_job_service,
        analysis_result_service,
    )
    calibration_service = CalibrationService(settings, PipelineCalibrationRunner(), video_frame_encoder)
    calibration_recording_service = CalibrationRecordingService(
        capture_service,
        calibration_service,
        phone_service,
        camera_controller,
    )

    capture_service.configure_cameras(
        settings.get_camera_count(),
        settings.get_ccb_url(),
        settings.get_live_view_frame_rate(),
    )
    capture_service.configure_capture_mode(
        settings.get_capture_mode(),
        settings.get_phone_camera_count(),
        settings.get_phone_frame_rate(),
        settings.get_phone_resolution(),
    )

    return AppServices(
        media_view_service=media_view_service,
        storage_root_service=storage_root_service,
        capture_service=capture_service,
        capture_recording_service=capture_recording_service,
        analysis_workspace_service=analysis_workspace_service,
        phone_service=phone_service,
        calibration_service=calibration_service,
        calibration_recording_service=calibration_recording_service,
    )


def _camera_controller_from_env(settings: JsonSettingsRepository) -> ModeCameraController:
    sony_count = int(os.environ.get("BASEBALL_MOTION_CAMERA_COUNT", str(settings.get_camera_count())))
    return ModeCameraController(
        settings,
        UrlCameraController(camera_count=sony_count),
        PhoneCameraController(camera_count=settings.get_phone_camera_count()),
    )

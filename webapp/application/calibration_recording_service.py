from __future__ import annotations

from dataclasses import dataclass

from webapp.application.calibration_service import ActiveCalibration, CalibrationService
from webapp.application.capture_service import CaptureService
from webapp.application.phone_capture_service import PhoneCaptureService
from webapp.domain.ports import CameraController


@dataclass(frozen=True)
class CalibrationRecordingResult:
    calibration: ActiveCalibration
    status: str
    phone_command: dict | None = None


class CalibrationRecordingService:
    def __init__(
        self,
        capture_service: CaptureService,
        calibration_service: CalibrationService,
        phone_service: PhoneCaptureService,
        camera_controller: CameraController,
    ) -> None:
        self._capture_service = capture_service
        self._calibration_service = calibration_service
        self._phone_service = phone_service
        self._camera_controller = camera_controller

    def start(
        self,
        mode: str,
        project_name: str,
        intrinsic_camera_label: str,
        metadata: dict,
        phone_session_token: str,
        base_url: str,
    ) -> CalibrationRecordingResult:
        connected_cameras = [camera for camera in self._capture_service.list_cameras() if camera.connected]
        record_camera_ids = [camera.camera_id for camera in connected_cameras]
        if mode == "INTR":
            selected_label = str(intrinsic_camera_label or "").strip()
            if not selected_label:
                raise ValueError("intrinsic_camera_required")
            save_camera_labels = {selected_label}
        else:
            save_camera_labels = {camera.label for camera in connected_cameras}

        calibration = self._calibration_service.start(
            mode,
            project_name,
            record_camera_ids,
            save_camera_labels,
            metadata,
        )
        self._camera_controller.start_recording(record_camera_ids)
        phone_command = None
        if self._capture_service.camera_settings()["capture_mode"] == "phone":
            token = str(phone_session_token or self._phone_service.current_or_create_draft(base_url).token)
            self._phone_service.start_calibration(
                token,
                calibration.mode,
                calibration.project_name,
                str(calibration.output_dir),
                save_camera_labels,
            )
            phone_command = {"command": "start", "token": token}
        return CalibrationRecordingResult(calibration=calibration, status="recording", phone_command=phone_command)

    def stop(self, phone_session_token: str, base_url: str) -> CalibrationRecordingResult:
        calibration = self._calibration_service.stop()
        self._camera_controller.stop_recording(
            calibration.record_camera_ids,
            str(calibration.output_dir),
            self._calibration_service.recording_subject(calibration),
            calibration.timestamp,
        )
        if calibration.mode == "INTR":
            self._calibration_service.remove_unselected_videos(calibration)
        phone_command = None
        if self._capture_service.camera_settings()["capture_mode"] == "phone":
            token = str(phone_session_token or self._phone_service.current_or_create_draft(base_url).token)
            self._phone_service.stop_calibration(token)
            phone_command = {"command": "stop", "token": token}
        return CalibrationRecordingResult(calibration=calibration, status="captured", phone_command=phone_command)

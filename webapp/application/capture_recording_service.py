from __future__ import annotations

from dataclasses import dataclass

from webapp.application.capture_service import CaptureService
from webapp.application.phone_capture_service import PhoneCaptureService
from webapp.domain.entities import CaptureSession, SubjectInfo


@dataclass(frozen=True)
class CaptureRecordingResult:
    session: CaptureSession
    status: str
    phone_command: dict | None = None


class CaptureRecordingService:
    def __init__(
        self,
        capture_service: CaptureService,
        phone_service: PhoneCaptureService,
    ) -> None:
        self._capture_service = capture_service
        self._phone_service = phone_service

    def start(
        self,
        subject: SubjectInfo,
        camera_ids: list[str],
        phone_session_token: str,
        base_url: str,
    ) -> CaptureRecordingResult:
        session = self._capture_service.start_capture(subject, camera_ids)
        phone_command = None
        if self._capture_service.camera_settings()["capture_mode"] == "phone":
            token = str(phone_session_token or self._phone_service.current_or_create_draft(base_url).token)
            self._phone_service.start_session(token, session)
            phone_command = {"command": "start", "token": token}
        return CaptureRecordingResult(session=session, status="recording", phone_command=phone_command)

    def stop(self, phone_session_token: str, base_url: str) -> CaptureRecordingResult:
        settings_payload = self._capture_service.camera_settings()
        token = str(phone_session_token or self._phone_service.current_or_create_draft(base_url).token)
        session = self._capture_service.stop_capture()
        phone_command = None
        if settings_payload["capture_mode"] == "phone":
            phone_command = {"command": "stop", "token": token}
        return CaptureRecordingResult(session=session, status="captured", phone_command=phone_command)

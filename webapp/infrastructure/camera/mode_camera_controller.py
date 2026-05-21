from __future__ import annotations

from webapp.domain.entities import CameraStatus, CaptureVideo, SubjectInfo
from webapp.domain.ports import CameraController, SettingsRepository


class ModeCameraController(CameraController):
    def __init__(
        self,
        settings: SettingsRepository,
        sony_controller: CameraController,
        phone_controller: CameraController,
    ) -> None:
        self._settings = settings
        self._sony_controller = sony_controller
        self._phone_controller = phone_controller

    def list_cameras(self) -> list[CameraStatus]:
        return self._active_controller().list_cameras()

    def start_recording(self, camera_ids: list[str]) -> None:
        self._active_controller().start_recording(camera_ids)

    def stop_recording(
        self,
        camera_ids: list[str],
        session_dir: str,
        subject: SubjectInfo,
        timestamp: str,
    ) -> list[CaptureVideo]:
        return self._active_controller().stop_recording(camera_ids, session_dir, subject, timestamp)

    def set_camera_count(self, camera_count: int) -> None:
        if hasattr(self._sony_controller, "set_camera_count"):
            self._sony_controller.set_camera_count(camera_count)

    def set_phone_camera_count(self, camera_count: int) -> None:
        if hasattr(self._phone_controller, "set_camera_count"):
            self._phone_controller.set_camera_count(camera_count)

    def set_live_view_frame_rate(self, frame_rate: str) -> None:
        if hasattr(self._sony_controller, "set_live_view_frame_rate"):
            self._sony_controller.set_live_view_frame_rate(frame_rate)

    def mark_phone_connected(self, camera_label: str) -> None:
        if hasattr(self._phone_controller, "mark_phone_connected"):
            self._phone_controller.mark_phone_connected(camera_label)

    def mark_phone_blocked(self, camera_label: str, message: str) -> None:
        if hasattr(self._phone_controller, "mark_phone_blocked"):
            self._phone_controller.mark_phone_blocked(camera_label, message)

    def mark_phone_disconnected(self, camera_label: str) -> None:
        if hasattr(self._phone_controller, "mark_phone_disconnected"):
            self._phone_controller.mark_phone_disconnected(camera_label)

    def _active_controller(self) -> CameraController:
        if self._settings.get_capture_mode() == "phone":
            return self._phone_controller
        return self._sony_controller

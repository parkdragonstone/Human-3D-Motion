from __future__ import annotations

from typing import Protocol

from webapp.domain.entities import CameraStatus, CaptureSession, CaptureVideo, SubjectInfo


class CameraController(Protocol):
    def list_cameras(self) -> list[CameraStatus]:
        ...

    def start_recording(self, camera_ids: list[str]) -> None:
        ...

    def stop_recording(
        self,
        camera_ids: list[str],
        session_dir: str,
        subject: SubjectInfo,
        timestamp: str,
    ) -> list[CaptureVideo]:
        ...


class SessionCatalog(Protocol):
    def list_sessions(self, storage_root: str, limit: int = 50) -> list[CaptureSession]:
        ...


class SettingsRepository(Protocol):
    def get_storage_root(self) -> str:
        ...

    def set_storage_root(self, storage_root: str) -> None:
        ...

    def get_camera_count(self) -> int:
        ...

    def set_camera_count(self, camera_count: int) -> None:
        ...

    def get_ccb_url(self) -> str:
        ...

    def set_ccb_url(self, ccb_url: str) -> None:
        ...

    def get_live_view_frame_rate(self) -> str:
        ...

    def set_live_view_frame_rate(self, frame_rate: str) -> None:
        ...

    def get_capture_mode(self) -> str:
        ...

    def set_capture_mode(self, capture_mode: str) -> None:
        ...

    def get_phone_camera_count(self) -> int:
        ...

    def set_phone_camera_count(self, camera_count: int) -> None:
        ...

    def get_phone_frame_rate(self) -> int:
        ...

    def set_phone_frame_rate(self, frame_rate: int) -> None:
        ...

    def get_phone_orientation(self) -> str:
        ...

    def set_phone_orientation(self, orientation: str) -> None:
        ...

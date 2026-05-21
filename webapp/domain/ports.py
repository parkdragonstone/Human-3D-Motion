from __future__ import annotations

from typing import Callable, Protocol

from webapp.domain.entities import CameraStatus, CaptureSession, CaptureVideo, SubjectInfo


LogEmitter = Callable[[str, str], None]


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


class AnalysisRunner(Protocol):
    def run(self, config: dict, emit_log: LogEmitter) -> None:
        ...


class CalibrationRunner(Protocol):
    def run(self, folder_path: str, metadata: dict | None = None) -> dict:
        ...


class AnalysisResultGateway(Protocol):
    def pose3d_data_from_trc(self, trc_path: str) -> dict:
        ...

    def keypoint_frame_from_json(self, session_path: str, camera_label: str, frame: int) -> dict:
        ...

    def save_keypoint_frame_to_json(
        self,
        session_path: str,
        camera_label: str,
        frame: int,
        people_keypoints: list,
    ) -> None:
        ...

    def render_pose_video_from_keypoints(self, session: CaptureSession, camera_label: str) -> str:
        ...

    def latest_kinematics_csv_file(self, session_path: str) -> str | None:
        ...

    def read_csv_columns(self, csv_path: str) -> dict[str, list[float]]:
        ...

    def recalculate_kinematics_event_markers(self, csv_path: str) -> list[dict[str, float | int | str]]:
        ...

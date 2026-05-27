from __future__ import annotations

import re
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from webapp.domain.entities import CameraStatus, CaptureSession, SubjectInfo
from webapp.domain.ports import CameraController, SessionCatalog, SettingsRepository


@dataclass
class ActiveCapture:
    session_id: str
    subject: SubjectInfo
    timestamp: str
    camera_ids: list[str]
    session_dir: Path


class CaptureService:
    def __init__(
        self,
        camera_controller: CameraController,
        sessions: SessionCatalog,
        settings: SettingsRepository,
    ) -> None:
        self._camera_controller = camera_controller
        self._sessions = sessions
        self._settings = settings
        self._active_capture: ActiveCapture | None = None

    def list_cameras(self):
        ccb_url = self._settings.get_ccb_url()
        frame_rate = self._settings.get_live_view_frame_rate()
        capture_mode = self._settings.get_capture_mode()
        return [
            CameraStatus(
                camera_id=camera.camera_id,
                label=camera.label,
                connected=camera.connected,
                recording=camera.recording,
                live_view_url=(camera.live_view_url or ccb_url) if capture_mode == "sony" else camera.live_view_url,
                live_view_frame_rate=frame_rate if capture_mode == "sony" else str(self._settings.get_phone_frame_rate()),
                last_error=camera.last_error,
            )
            for camera in self._camera_controller.list_cameras()
        ]

    def mark_phone_connected(self, camera_label: str) -> None:
        if hasattr(self._camera_controller, "mark_phone_connected"):
            self._camera_controller.mark_phone_connected(camera_label)

    def mark_phone_blocked(self, camera_label: str, message: str) -> None:
        if hasattr(self._camera_controller, "mark_phone_blocked"):
            self._camera_controller.mark_phone_blocked(camera_label, message)

    def mark_phone_disconnected(self, camera_label: str) -> None:
        if hasattr(self._camera_controller, "mark_phone_disconnected"):
            self._camera_controller.mark_phone_disconnected(camera_label)

    def configure_cameras(self, camera_count: int, ccb_url: str, live_view_frame_rate: str) -> None:
        self._settings.set_camera_count(camera_count)
        self._settings.set_ccb_url(ccb_url)
        self._settings.set_live_view_frame_rate(live_view_frame_rate)
        if hasattr(self._camera_controller, "set_camera_count"):
            self._camera_controller.set_camera_count(camera_count)
        if hasattr(self._camera_controller, "set_live_view_frame_rate"):
            self._camera_controller.set_live_view_frame_rate(live_view_frame_rate)

    def configure_capture_mode(
        self,
        capture_mode: str,
        phone_camera_count: int | None = None,
        phone_frame_rate: int | None = None,
        phone_orientation: str | None = None,
    ) -> None:
        self._settings.set_capture_mode(capture_mode)
        if phone_camera_count is not None:
            self._settings.set_phone_camera_count(phone_camera_count)
            if hasattr(self._camera_controller, "set_phone_camera_count"):
                self._camera_controller.set_phone_camera_count(phone_camera_count)
        if phone_frame_rate is not None:
            self._settings.set_phone_frame_rate(phone_frame_rate)
        if phone_orientation is not None:
            self._settings.set_phone_orientation(phone_orientation)

    def camera_settings(self) -> dict:
        return {
            "camera_count": self._settings.get_camera_count(),
            "ccb_url": self._settings.get_ccb_url(),
            "live_view_frame_rate": self._settings.get_live_view_frame_rate(),
            "capture_mode": self._settings.get_capture_mode(),
            "phone_camera_count": self._settings.get_phone_camera_count(),
            "phone_frame_rate": self._settings.get_phone_frame_rate(),
            "phone_orientation": self._settings.get_phone_orientation(),
            "backend": self._camera_controller.__class__.__name__,
        }

    def get_storage_root(self) -> str:
        return self._settings.get_storage_root()

    def set_storage_root(self, storage_root: str) -> str:
        path = Path(storage_root).expanduser().resolve()
        path.mkdir(parents=True, exist_ok=True)
        self._settings.set_storage_root(str(path))
        return str(path)

    def start_capture(self, subject: SubjectInfo, camera_ids: list[str]) -> CaptureSession:
        if self._active_capture is not None:
            raise RuntimeError("capture_already_running")
        if not camera_ids:
            raise ValueError("select_at_least_one_camera")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_id = timestamp
        session_name = _session_name(subject, timestamp)
        session_dir = Path(self.get_storage_root()) / session_name
        session_dir.mkdir(parents=True, exist_ok=True)

        self._camera_controller.start_recording(camera_ids)
        session = CaptureSession(
            session_id=session_id,
            subject=subject,
            timestamp=timestamp,
            session_path=str(session_dir),
            status="recording",
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )
        self._active_capture = ActiveCapture(session_id, subject, timestamp, camera_ids, session_dir)
        return session

    def stop_capture(self) -> CaptureSession:
        if self._active_capture is None:
            raise RuntimeError("capture_not_running")

        active = self._active_capture
        videos = self._camera_controller.stop_recording(
            active.camera_ids,
            str(active.session_dir),
            active.subject,
            active.timestamp,
        )
        session = CaptureSession(
            session_id=active.session_id,
            subject=active.subject,
            timestamp=active.timestamp,
            session_path=str(active.session_dir),
            status="captured",
            videos=videos,
            updated_at=datetime.now(),
        )
        self._active_capture = None
        return session

    def active_capture(self) -> ActiveCapture | None:
        return self._active_capture

    def list_sessions(self):
        return self._sessions.list_sessions(self.get_storage_root())

    def delete_session(self, session_id: str) -> CaptureSession:
        if self._active_capture is not None and self._active_capture.session_id == session_id:
            raise RuntimeError("cannot_delete_active_capture")

        session = next((item for item in self.list_sessions() if item.session_id == session_id), None)
        if session is None:
            raise ValueError("session_not_found")

        storage_root = Path(self.get_storage_root()).resolve()
        session_path = Path(session.session_path).resolve()
        if session_path == storage_root or storage_root not in session_path.parents:
            raise ValueError("session_path_outside_storage_root")
        if not session_path.exists():
            raise ValueError("session_path_not_found")

        if session_path.is_dir():
            shutil.rmtree(session_path)
        else:
            session_path.unlink()
        return session


def _session_name(subject: SubjectInfo, timestamp: str) -> str:
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", subject.name.strip()).strip("_")
    if not safe_name:
        safe_name = "subject"
    safe_hand = _safe_hand(subject.hand)
    return f"{safe_name}_{subject.height_cm}_{subject.weight_kg}_{safe_hand}_{timestamp}"


def _safe_hand(hand: str) -> str:
    value = str(hand or "right").strip().lower()
    return value if value in {"right", "left"} else "right"

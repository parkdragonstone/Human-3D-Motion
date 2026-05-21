from __future__ import annotations

from webapp.domain.entities import CameraStatus, CaptureVideo, SubjectInfo
from webapp.domain.ports import CameraController


class PhoneCameraController(CameraController):
    """Camera controller for phone clients paired by QR code."""

    def __init__(self, camera_count: int = 2) -> None:
        self._camera_count = camera_count
        self._recording_ids: set[str] = set()
        self._connected_labels: set[str] = set()
        self._last_errors: dict[str, str] = {}

    def list_cameras(self) -> list[CameraStatus]:
        return [
            CameraStatus(
                camera_id=f"phone-{idx:02d}",
                label=f"cam{idx:02d}",
                connected=f"cam{idx:02d}" in self._connected_labels,
                recording=f"phone-{idx:02d}" in self._recording_ids,
                last_error=None
                if f"cam{idx:02d}" in self._connected_labels
                else self._last_errors.get(f"cam{idx:02d}") or "Phone is not paired.",
            )
            for idx in range(1, self._camera_count + 1)
        ]

    def start_recording(self, camera_ids: list[str]) -> None:
        valid = {c.camera_id for c in self.list_cameras()}
        unknown = sorted(set(camera_ids) - valid)
        if unknown:
            raise ValueError(f"unknown_camera_ids: {', '.join(unknown)}")
        self._recording_ids.update(camera_ids)

    def stop_recording(
        self,
        camera_ids: list[str],
        session_dir: str,
        subject: SubjectInfo,
        timestamp: str,
    ) -> list[CaptureVideo]:
        self._recording_ids.difference_update(camera_ids)
        return []

    def set_camera_count(self, camera_count: int) -> None:
        self._camera_count = max(1, min(16, int(camera_count)))
        valid = {c.camera_id for c in self.list_cameras()}
        valid_labels = {c.label for c in self.list_cameras()}
        self._recording_ids = self._recording_ids & valid
        self._connected_labels = self._connected_labels & valid_labels
        self._last_errors = {label: error for label, error in self._last_errors.items() if label in valid_labels}

    def mark_phone_connected(self, camera_label: str) -> None:
        self._connected_labels.add(camera_label)
        self._last_errors.pop(camera_label, None)

    def mark_phone_blocked(self, camera_label: str, message: str) -> None:
        self._connected_labels.discard(camera_label)
        self._last_errors[camera_label] = message or "Phone camera is blocked."

    def mark_phone_disconnected(self, camera_label: str) -> None:
        self._connected_labels.discard(camera_label)
        self._last_errors[camera_label] = "Phone is not paired."

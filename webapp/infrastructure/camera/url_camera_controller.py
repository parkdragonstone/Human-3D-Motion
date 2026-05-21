from __future__ import annotations

from pathlib import Path

from webapp.domain.entities import CameraStatus, CaptureVideo, SubjectInfo
from webapp.domain.ports import CameraController


class UrlCameraController(CameraController):
    """URL-based camera adapter for the CCB-WD1 web control workflow."""

    def __init__(self, camera_count: int = 2) -> None:
        self._camera_count = camera_count
        self._live_view_frame_rate = "low"
        self._recording_ids: set[str] = set()

    def list_cameras(self) -> list[CameraStatus]:
        cameras = [
            CameraStatus(
                camera_id=f"ccb-{idx:02d}",
                label=f"cam{idx:02d}",
                connected=True,
                live_view_frame_rate=self._live_view_frame_rate,
            )
            for idx in range(1, self._camera_count + 1)
        ]
        return [
            CameraStatus(
                camera_id=c.camera_id,
                label=c.label,
                connected=c.connected,
                recording=c.camera_id in self._recording_ids,
                live_view_url=c.live_view_url,
                live_view_frame_rate=self._live_view_frame_rate,
                last_error=c.last_error,
            )
            for c in cameras
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
        output_dir = Path(session_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        cameras_by_id = {c.camera_id: c for c in self.list_cameras()}
        videos: list[CaptureVideo] = []

        for camera_id in camera_ids:
            camera = cameras_by_id[camera_id]
            filename = f"{subject.name}_{subject.height_cm}_{subject.weight_kg}_{subject.hand}_{timestamp}_{camera.label}.mp4"
            output_path = output_dir / filename
            output_path.write_bytes(b"")
            videos.append(CaptureVideo(camera_id=camera_id, camera_label=camera.label, path=str(output_path)))

        self._recording_ids.difference_update(camera_ids)
        return videos

    def set_camera_count(self, camera_count: int) -> None:
        self._camera_count = max(1, min(16, int(camera_count)))
        valid = {c.camera_id for c in self.list_cameras()}
        self._recording_ids = self._recording_ids & valid

    def set_live_view_frame_rate(self, frame_rate: str) -> None:
        self._live_view_frame_rate = frame_rate if frame_rate in {"low", "standard"} else "low"

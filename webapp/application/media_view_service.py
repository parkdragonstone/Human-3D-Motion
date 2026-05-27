from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from webapp.domain.entities import CaptureSession, SubjectInfo
from webapp.domain.ports import VideoMetadataReader


@dataclass(frozen=True)
class VideoView:
    camera_id: str
    camera_label: str
    path: str
    filename: str
    fps: float
    frame_count: int
    size_bytes: int
    pose_video_path: str | None
    pose_video_mtime: int | None


@dataclass(frozen=True)
class SessionView:
    session_id: str
    subject: SubjectInfo
    timestamp: str
    session_path: str
    status: str
    videos: list[VideoView]
    created_at: datetime | None
    updated_at: datetime | None


@dataclass(frozen=True)
class CalibrationVideoView:
    camera_label: str
    path: str
    filename: str
    size_bytes: int


@dataclass(frozen=True)
class CalibrationRecordView:
    mode: str
    project_name: str
    folder_name: str
    output_dir: str
    updated_at: datetime
    videos: list[CalibrationVideoView]


class MediaViewService:
    def __init__(self, video_metadata_reader: VideoMetadataReader) -> None:
        self._video_metadata_reader = video_metadata_reader

    def session_view(self, session: CaptureSession) -> SessionView:
        return SessionView(
            session_id=session.session_id,
            subject=session.subject,
            timestamp=session.timestamp,
            session_path=session.session_path,
            status=session.status,
            videos=[self._video_view(video, session.session_path) for video in session.videos],
            created_at=session.created_at,
            updated_at=session.updated_at,
        )

    def calibration_record_view(self, record) -> CalibrationRecordView:
        return CalibrationRecordView(
            mode=record.mode,
            project_name=record.project_name,
            folder_name=record.folder_name,
            output_dir=str(record.output_dir),
            updated_at=record.updated_at,
            videos=[self._calibration_video_view(video) for video in record.videos],
        )

    def _video_view(self, video, session_path: str) -> VideoView:
        path = Path(video.path)
        size_bytes = path.stat().st_size if path.is_file() else 0
        fps = 0.0
        frame_count = 0
        if path.is_file():
            fps, _resolution = self._video_metadata_reader.read_metadata(str(path))
            frame_count = self._video_metadata_reader.read_frame_count(str(path))

        pose_path = Path(session_path or path.parent) / "pose" / f"{_pose_label(video.camera_label)}_pose.mp4"
        pose_exists = pose_path.is_file()
        return VideoView(
            camera_id=video.camera_id,
            camera_label=video.camera_label,
            path=str(path),
            filename=path.name,
            fps=fps,
            frame_count=frame_count,
            size_bytes=size_bytes,
            pose_video_path=str(pose_path) if pose_exists else None,
            pose_video_mtime=int(pose_path.stat().st_mtime) if pose_exists else None,
        )

    def _calibration_video_view(self, video) -> CalibrationVideoView:
        path = Path(video.path)
        size_bytes = path.stat().st_size if path.is_file() else 0
        return CalibrationVideoView(
            camera_label=video.camera_label,
            path=str(path),
            filename=path.name,
            size_bytes=size_bytes,
        )


def _pose_label(camera_label: str) -> str:
    match = re.search(r"cam0*(\d+)$", str(camera_label).lower())
    return f"cam{int(match.group(1))}" if match else str(camera_label)

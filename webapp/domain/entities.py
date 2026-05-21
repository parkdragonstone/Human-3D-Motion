from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass(frozen=True)
class CameraStatus:
    camera_id: str
    label: str
    connected: bool
    recording: bool = False
    live_view_url: str | None = None
    live_view_frame_rate: str = "low"
    last_error: str | None = None


@dataclass(frozen=True)
class SubjectInfo:
    name: str
    height_cm: int
    weight_kg: int
    hand: str = "right"


@dataclass(frozen=True)
class CaptureVideo:
    camera_id: str
    camera_label: str
    path: str


@dataclass(frozen=True)
class CaptureSession:
    session_id: str
    subject: SubjectInfo
    timestamp: str
    session_path: str
    status: str
    videos: list[CaptureVideo] = field(default_factory=list)
    created_at: datetime | None = None
    updated_at: datetime | None = None

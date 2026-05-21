from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

from webapp.domain.entities import CaptureSession, CaptureVideo, SubjectInfo
from webapp.domain.ports import SessionCatalog


SESSION_FILE_RE = re.compile(
    r"^(?P<name>.+)_(?P<height>\d+)_(?P<weight>\d+)(?:_(?P<hand>right|left))?_(?P<date>\d{8})_(?P<time>\d{6})_(?P<camera>cam\d+)\.(?:mp4|avi)$",
    re.IGNORECASE,
)

VIDEO_EXTENSIONS = {".mp4", ".avi"}


class FileSessionCatalog(SessionCatalog):
    def list_sessions(self, storage_root: str, limit: int = 50) -> list[CaptureSession]:
        root = Path(storage_root)
        if not root.is_dir():
            return []

        grouped: dict[str, list[tuple[Path, re.Match[str]]]] = {}
        for video in root.rglob("*"):
            if not video.is_file() or video.suffix.lower() not in VIDEO_EXTENSIONS:
                continue
            match = SESSION_FILE_RE.match(video.name)
            if not match:
                continue
            timestamp = f"{match.group('date')}_{match.group('time')}"
            grouped.setdefault(timestamp, []).append((video, match))

        sessions = [self._to_session(timestamp, items) for timestamp, items in grouped.items()]
        sessions.sort(key=lambda session: session.timestamp, reverse=True)
        return sessions[:limit]

    def _to_session(self, timestamp: str, items: list[tuple[Path, re.Match[str]]]) -> CaptureSession:
        first_path, first_match = sorted(items, key=lambda item: item[1].group("camera"))[0]
        videos = [
            CaptureVideo(
                camera_id=match.group("camera").lower(),
                camera_label=match.group("camera").lower(),
                path=str(path),
            )
            for path, match in sorted(items, key=lambda item: item[1].group("camera"))
        ]
        subject = SubjectInfo(
            name=first_match.group("name"),
            height_cm=int(first_match.group("height")),
            weight_kg=int(first_match.group("weight")),
            hand=(first_match.group("hand") or "right").lower(),
        )
        newest_mtime = max(path.stat().st_mtime for path, _ in items)
        updated_at = datetime.fromtimestamp(newest_mtime)
        session_dir = first_path.parent
        return CaptureSession(
            session_id=timestamp,
            subject=subject,
            timestamp=timestamp,
            session_path=str(session_dir),
            status="captured",
            videos=videos,
            created_at=updated_at,
            updated_at=updated_at,
        )

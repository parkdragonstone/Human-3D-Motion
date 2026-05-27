from __future__ import annotations

from pathlib import Path

from webapp.domain.entities import CaptureSession
from webapp.domain.ports import SessionCatalog


class SessionQueryService:
    def __init__(self, sessions: SessionCatalog) -> None:
        self._sessions = sessions

    def find_by_path(self, session_path: str | Path) -> CaptureSession | None:
        resolved = self._resolve_path(session_path)
        return next(
            (
                session
                for session in self._sessions.list_sessions(str(resolved.parent))
                if Path(session.session_path).resolve() == resolved
            ),
            None,
        )

    def require_by_path(self, session_path: str | Path) -> CaptureSession:
        resolved = self._resolve_path(session_path)
        if not resolved.is_dir():
            raise ValueError("session_path_not_found")
        session = self.find_by_path(resolved)
        if session is None:
            raise ValueError("session_not_found")
        return session

    @staticmethod
    def _resolve_path(session_path: str | Path) -> Path:
        return Path(str(session_path)).expanduser().resolve()

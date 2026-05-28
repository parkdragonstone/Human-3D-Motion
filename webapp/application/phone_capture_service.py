from __future__ import annotations

import base64
import io
import mimetypes
import secrets
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO
from urllib.parse import urljoin

from webapp.domain.entities import CaptureSession
from webapp.domain.ports import SettingsRepository


@dataclass(frozen=True)
class PhoneSlot:
    camera_id: str
    camera_label: str
    join_url: str
    qr_data_url: str


@dataclass
class PhoneDraft:
    token: str
    slots: list[PhoneSlot]


@dataclass(frozen=True)
class PhoneCalibration:
    mode: str
    project_name: str
    output_dir: str
    save_camera_labels: set[str]


@dataclass(frozen=True)
class PhoneVideoUpload:
    stream: BinaryIO | None
    content_type: str
    user_agent: str
    actual_fps: str | None
    actual_width: str | None
    actual_height: str | None


class PhoneCaptureService:
    def __init__(self, settings: SettingsRepository) -> None:
        self._settings = settings
        self._draft: PhoneDraft | None = None
        self._active_sessions: dict[str, CaptureSession] = {}
        self._active_calibrations: dict[str, PhoneCalibration] = {}

    def current_or_create_draft(self, base_url: str) -> PhoneDraft:
        if (
            self._draft is None
            or len(self._draft.slots) != self._settings.get_phone_camera_count()
            or not _draft_matches_base_url(self._draft, base_url)
        ):
            self._draft = self.create_draft(base_url)
        return self._draft

    def create_draft(self, base_url: str) -> PhoneDraft:
        token = secrets.token_urlsafe(18)
        slots = [
            self._slot(base_url, token, idx)
            for idx in range(1, self._settings.get_phone_camera_count() + 1)
        ]
        self._draft = PhoneDraft(token=token, slots=slots)
        return self._draft

    def settings_payload(self) -> dict:
        resolution = self._settings.get_phone_resolution()
        return {
            "frame_rate": self._settings.get_phone_frame_rate(),
            "resolution": _resolution_size(resolution),
            "camera_count": self._settings.get_phone_camera_count(),
        }

    def start_session(self, token: str, session: CaptureSession) -> None:
        self._active_sessions[token] = session
        self._active_calibrations.pop(token, None)

    def stop_session(self, token: str) -> CaptureSession | None:
        return self._active_sessions.get(token)

    def start_calibration(
        self,
        token: str,
        mode: str,
        project_name: str,
        output_dir: str,
        save_camera_labels: set[str],
    ) -> None:
        self._active_sessions.pop(token, None)
        self._active_calibrations[token] = PhoneCalibration(
            mode=mode,
            project_name=project_name,
            output_dir=output_dir,
            save_camera_labels=save_camera_labels,
        )

    def stop_calibration(self, token: str) -> PhoneCalibration | None:
        return self._active_calibrations.get(token)

    def save_upload(
        self,
        token: str,
        camera_label: str,
        upload: PhoneVideoUpload,
    ) -> dict:
        calibration = self._active_calibrations.get(token)
        if calibration is not None:
            return self._save_calibration_upload(
                calibration,
                camera_label,
                upload,
            )

        session = self._active_sessions.get(token)
        if session is None:
            raise ValueError("phone_session_not_active")

        if upload.stream is None:
            raise ValueError("video_file_required")

        session_dir = Path(session.session_path)
        session_dir.mkdir(parents=True, exist_ok=True)
        content_type = upload.content_type or "video/mp4"
        extension = mimetypes.guess_extension(content_type.split(";")[0]) or ".mp4"
        if extension == ".m4v":
            extension = ".mp4"
        output_path = session_dir / (
            f"{session.subject.name}_{session.subject.height_cm}_{session.subject.weight_kg}_{session.subject.hand}_"
            f"{session.timestamp}_{camera_label}{extension}"
        )
        _write_upload(upload.stream, output_path)
        return {"path": str(output_path)}

    def _save_calibration_upload(
        self,
        calibration: PhoneCalibration,
        camera_label: str,
        upload: PhoneVideoUpload,
    ) -> dict:
        if camera_label not in calibration.save_camera_labels:
            return {"skipped": True, "camera_label": camera_label}

        if upload.stream is None:
            raise ValueError("video_file_required")

        output_dir = Path(calibration.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        content_type = upload.content_type or "video/mp4"
        extension = mimetypes.guess_extension(content_type.split(";")[0]) or ".mp4"
        if extension == ".m4v":
            extension = ".mp4"
        output_path = output_dir / f"{calibration.mode}_{calibration.project_name}_{camera_label}{extension}"
        _write_upload(upload.stream, output_path)
        return {"path": str(output_path)}

    def _slot(self, base_url: str, token: str, idx: int) -> PhoneSlot:
        camera_label = f"cam{idx:02d}"
        join_url = urljoin(base_url, f"phone-capture/{token}/{camera_label}")
        return PhoneSlot(
            camera_id=f"phone-{idx:02d}",
            camera_label=camera_label,
            join_url=join_url,
            qr_data_url=_qr_data_url(join_url),
        )


def _qr_data_url(value: str) -> str:
    try:
        import qrcode
        import qrcode.image.svg

        image_factory = qrcode.image.svg.SvgPathImage
        image = qrcode.make(value, image_factory=image_factory)
        buffer = io.BytesIO()
        image.save(buffer)
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return f"data:image/svg+xml;base64,{encoded}"
    except Exception:
        encoded = base64.b64encode(value.encode("utf-8")).decode("ascii")
        return f"data:text/plain;base64,{encoded}"


def _write_upload(stream: BinaryIO, output_path: Path) -> None:
    with output_path.open("wb") as target:
        shutil.copyfileobj(stream, target)


def _draft_matches_base_url(draft: PhoneDraft, base_url: str) -> bool:
    return all(slot.join_url.startswith(base_url) for slot in draft.slots)


def _resolution_size(resolution: str) -> str:
    return "1920x1080" if resolution == "1080" else "1280x720"

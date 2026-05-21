from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from webapp.domain.ports import SettingsRepository


@dataclass
class ActiveCalibration:
    calibration_id: str
    mode: str
    project_name: str
    timestamp: str
    output_dir: Path
    record_camera_ids: list[str]
    save_camera_labels: set[str]
    metadata: dict = field(default_factory=dict)


@dataclass
class CalibrationVideo:
    camera_label: str
    path: Path


@dataclass
class CalibrationRecord:
    mode: str
    project_name: str
    folder_name: str
    output_dir: Path
    updated_at: datetime
    videos: list[CalibrationVideo] = field(default_factory=list)


class CalibrationService:
    def __init__(self, settings: SettingsRepository) -> None:
        self._settings = settings
        self._active: ActiveCalibration | None = None

    def start(
        self,
        mode: str,
        project_name: str,
        record_camera_ids: list[str],
        save_camera_labels: set[str],
        metadata: dict,
    ) -> ActiveCalibration:
        if self._active is not None:
            raise RuntimeError("calibration_already_running")
        if not record_camera_ids:
            raise ValueError("select_at_least_one_camera")
        normalized_mode = _mode(mode)
        safe_project = _safe_project_name(project_name)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(self._settings.get_storage_root()) / f"CALIB_{normalized_mode}_{safe_project}"
        output_dir.mkdir(parents=True, exist_ok=True)

        calibration = ActiveCalibration(
            calibration_id=timestamp,
            mode=normalized_mode,
            project_name=safe_project,
            timestamp=timestamp,
            output_dir=output_dir,
            record_camera_ids=record_camera_ids,
            save_camera_labels=save_camera_labels,
            metadata=metadata,
        )
        self._active = calibration
        self._write_metadata(calibration, "recording")
        return calibration

    def stop(self) -> ActiveCalibration:
        if self._active is None:
            raise RuntimeError("calibration_not_running")
        calibration = self._active
        self._active = None
        self._write_metadata(calibration, "captured")
        return calibration

    def active(self) -> ActiveCalibration | None:
        return self._active

    def list_calibrations(self) -> list[CalibrationRecord]:
        root = Path(self._settings.get_storage_root())
        if not root.exists():
            return []

        records: list[CalibrationRecord] = []
        for path in root.iterdir():
            if not path.is_dir():
                continue
            match = re.match(r"^CALIB_(INTR|EXTR)_(.+)$", path.name)
            if not match:
                continue
            records.append(
                CalibrationRecord(
                    mode=match.group(1),
                    project_name=match.group(2),
                    folder_name=path.name,
                    output_dir=path,
                    updated_at=datetime.fromtimestamp(path.stat().st_mtime),
                    videos=_calibration_videos(path),
                )
            )
        return sorted(records, key=lambda record: record.updated_at, reverse=True)

    def delete_calibration(self, folder_name: str) -> CalibrationRecord:
        record = next((item for item in self.list_calibrations() if item.folder_name == folder_name), None)
        if record is None:
            raise ValueError("calibration_not_found")
        if self._active is not None and self._active.output_dir == record.output_dir:
            raise RuntimeError("calibration_is_recording")

        storage_root = Path(self._settings.get_storage_root()).resolve()
        calibration_path = record.output_dir.resolve()
        if calibration_path == storage_root or storage_root not in calibration_path.parents:
            raise ValueError("calibration_path_outside_storage_root")
        shutil.rmtree(calibration_path)
        return record

    def run_calibration(self, folder_name: str, metadata: dict | None = None) -> dict:
        record = next((item for item in self.list_calibrations() if item.folder_name == folder_name), None)
        if record is None:
            raise ValueError("calibration_not_found")
        if self._active is not None and self._active.output_dir == record.output_dir:
            raise RuntimeError("calibration_is_recording")

        storage_root = Path(self._settings.get_storage_root()).resolve()
        calibration_path = record.output_dir.resolve()
        if calibration_path == storage_root or storage_root not in calibration_path.parents:
            raise ValueError("calibration_path_outside_storage_root")

        from pipelines.calibration import run_calibration_folder

        return run_calibration_folder(str(calibration_path), metadata)

    def _write_metadata(self, calibration: ActiveCalibration, status: str) -> None:
        payload = {
            "calibration_id": calibration.calibration_id,
            "mode": calibration.mode,
            "project_name": calibration.project_name,
            "timestamp": calibration.timestamp,
            "status": status,
            "record_camera_ids": calibration.record_camera_ids,
            "save_camera_labels": sorted(calibration.save_camera_labels),
            **calibration.metadata,
        }
        (calibration.output_dir / "calibration.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _mode(value: str) -> str:
    normalized = str(value or "INTR").strip().upper()
    return normalized if normalized in {"INTR", "EXTR"} else "INTR"


def _safe_project_name(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "").strip()).strip("_")
    if not safe:
        raise ValueError("project_name_required")
    return safe


def _calibration_videos(path: Path) -> list[CalibrationVideo]:
    videos: list[CalibrationVideo] = []
    for video in sorted(path.iterdir()):
        if (
            not video.is_file()
            or video.suffix.lower() not in {".mp4", ".mov", ".webm", ".avi"}
            or video.stem.lower().startswith("intrinsic_debug_")
        ):
            continue
        videos.append(CalibrationVideo(camera_label=_camera_label_from_video(video), path=video))
    return videos


def _camera_label_from_video(video: Path) -> str:
    match = re.search(r"(cam\d+)$", video.stem, re.IGNORECASE)
    return match.group(1).lower() if match else "camera"

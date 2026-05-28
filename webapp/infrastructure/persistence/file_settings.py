from __future__ import annotations

import json
import os
from pathlib import Path

from webapp.domain.ports import SettingsRepository


class JsonSettingsRepository(SettingsRepository):
    def __init__(self, settings_path: str = "webapp_data/settings.json") -> None:
        self._settings_path = Path(settings_path)
        self._settings_path.parent.mkdir(parents=True, exist_ok=True)

    def get_storage_root(self) -> str:
        data = self._read()
        storage_root = os.environ.get("BASEBALL_MOTION_STORAGE") or data.get("storage_root")
        if storage_root:
            path = Path(storage_root).expanduser().resolve()
        else:
            path = Path("recordings").resolve()
            self.set_storage_root(str(path))
        path.mkdir(parents=True, exist_ok=True)
        return str(path)

    def set_storage_root(self, storage_root: str) -> None:
        path = Path(storage_root).expanduser().resolve()
        path.mkdir(parents=True, exist_ok=True)
        data = self._read()
        data["storage_root"] = str(path)
        data.pop("analysis_root", None)
        self._settings_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def get_camera_count(self) -> int:
        data = self._read()
        try:
            return max(1, min(16, int(data.get("camera_count", 1))))
        except (TypeError, ValueError):
            return 1

    def set_camera_count(self, camera_count: int) -> None:
        data = self._read()
        data["camera_count"] = max(1, min(16, int(camera_count)))
        self._settings_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def get_ccb_url(self) -> str:
        data = self._read()
        return str(data.get("ccb_url") or "http://169.254.200.200/").strip()

    def set_ccb_url(self, ccb_url: str) -> None:
        data = self._read()
        data["ccb_url"] = ccb_url.strip() or "http://169.254.200.200/"
        self._settings_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def get_live_view_frame_rate(self) -> str:
        data = self._read()
        value = str(data.get("live_view_frame_rate") or "low").lower()
        return value if value in {"low", "standard"} else "low"

    def set_live_view_frame_rate(self, frame_rate: str) -> None:
        data = self._read()
        value = str(frame_rate or "low").lower()
        data["live_view_frame_rate"] = value if value in {"low", "standard"} else "low"
        self._settings_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def get_capture_mode(self) -> str:
        data = self._read()
        value = str(data.get("capture_mode") or "sony").lower()
        return value if value in {"sony", "phone"} else "sony"

    def set_capture_mode(self, capture_mode: str) -> None:
        data = self._read()
        value = str(capture_mode or "sony").lower()
        data["capture_mode"] = value if value in {"sony", "phone"} else "sony"
        self._settings_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def get_phone_camera_count(self) -> int:
        data = self._read()
        try:
            return max(1, min(16, int(data.get("phone_camera_count", 2))))
        except (TypeError, ValueError):
            return 2

    def set_phone_camera_count(self, camera_count: int) -> None:
        data = self._read()
        data["phone_camera_count"] = max(1, min(16, int(camera_count)))
        self._settings_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def get_phone_frame_rate(self) -> int:
        data = self._read()
        try:
            value = int(data.get("phone_frame_rate", 120))
        except (TypeError, ValueError):
            return 120
        return value if value in {30, 60, 120, 240} else 120

    def set_phone_frame_rate(self, frame_rate: int) -> None:
        data = self._read()
        try:
            value = int(frame_rate)
        except (TypeError, ValueError):
            value = 120
        data["phone_frame_rate"] = value if value in {30, 60, 120, 240} else 120
        self._settings_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def get_phone_resolution(self) -> str:
        data = self._read()
        value = str(data.get("phone_resolution") or "720").lower().removesuffix("p")
        return value if value in {"720", "1080"} else "720"

    def set_phone_resolution(self, resolution: str) -> None:
        data = self._read()
        value = str(resolution or "720").lower().removesuffix("p")
        data["phone_resolution"] = value if value in {"720", "1080"} else "720"
        self._settings_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def _read(self) -> dict:
        if not self._settings_path.is_file():
            return {}
        try:
            return json.loads(self._settings_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}

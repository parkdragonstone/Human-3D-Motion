from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class IntrinsicResult:
    ok: bool
    error: str | None = None
    rms: float | None = None
    image_size: tuple[int, int] | None = None
    camera_matrix: np.ndarray | None = None
    dist_coeffs: np.ndarray | None = None
    used_frames: int = 0
    used_corners: int = 0
    frames_read: int = 0
    frames_checked: int = 0
    frames_found: int = 0

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "error": self.error,
            "rms": self.rms,
            "image_size": list(self.image_size) if self.image_size else None,
            "camera_matrix": self.camera_matrix.tolist() if self.camera_matrix is not None else None,
            "dist_coeffs": self.dist_coeffs.reshape(-1).tolist() if self.dist_coeffs is not None else None,
            "used_frames": int(self.used_frames),
            "used_corners": int(self.used_corners),
            "frames_read": int(self.frames_read),
            "frames_checked": int(self.frames_checked),
            "frames_found": int(self.frames_found),
        }

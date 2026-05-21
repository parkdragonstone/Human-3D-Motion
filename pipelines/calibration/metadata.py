from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


def folder_mode_and_project(folder: Path) -> tuple[str, str]:
    parts = folder.name.split("_", 2)
    if len(parts) == 3 and parts[0] == "CALIB" and parts[1] in {"INTR", "EXTR"}:
        return parts[1], parts[2]
    raise ValueError(f"bad_calibration_folder_name: {folder.name}")


def intrinsic_for_label(bundle: dict[str, Any], camera_label: str) -> dict[str, Any] | None:
    label = camera_label.lower()
    candidates = [
        bundle.get("intrinsics", {}).get(label) if isinstance(bundle.get("intrinsics"), dict) else None,
        bundle.get(label),
        bundle.get(f"intrinsic_{label}"),
        bundle.get(label.upper()),
        bundle.get(f"intrinsic_{label.upper()}"),
    ]
    for candidate in candidates:
        if isinstance(candidate, dict) and candidate.get("camera_matrix") is not None:
            return candidate
    return None


def parse_object_points(value) -> list[dict[str, Any]]:
    if isinstance(value, list):
        return [point for point in value if isinstance(point, dict)]
    points: list[dict[str, Any]] = []
    for index, line in enumerate(str(value or "").splitlines()):
        parts = [part.strip() for part in line.split(",") if part.strip()]
        if len(parts) == 3:
            point_id = index
            coords = parts
        elif len(parts) >= 4:
            point_id = parts[0]
            coords = parts[1:4]
        else:
            continue
        try:
            points.append({
                "id": point_id,
                "x": float(coords[0]),
                "y": float(coords[1]),
                "z": float(coords[2]),
            })
        except ValueError:
            continue
    return points


def parse_image_points(value) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    points: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        try:
            points.append({
                "id": item.get("id"),
                "u": float(item.get("u")),
                "v": float(item.get("v")),
            })
        except (TypeError, ValueError):
            continue
    return points


def parse_image_points_by_camera(metadata: dict[str, Any], camera_labels: list[str]) -> dict[str, list[dict[str, Any]]]:
    raw = metadata.get("image_points_by_camera")
    points_by_camera: dict[str, list[dict[str, Any]]] = {}
    if isinstance(raw, dict):
        for label, points in raw.items():
            points_by_camera[str(label).lower()] = parse_image_points(points)

    for index, label in enumerate(camera_labels, start=1):
        if label not in points_by_camera:
            points_by_camera[label] = parse_image_points(metadata.get(f"image_points_cam{index}"))
    return points_by_camera


def checker_board_type(value: str) -> str:
    normalized = str(value or "auto").strip().lower()
    aliases = {
        "auto": "chessboard",
        "sb": "chessboard",
        "standard": "chessboard",
        "chess": "chessboard",
        "chessboard": "chessboard",
        "charuco": "charuco",
        "charucoboard": "charuco",
    }
    if normalized in aliases:
        return aliases[normalized]
    raise ValueError(f"unknown_checker_board_type: {value}")


def read_metadata(folder: Path) -> dict[str, Any]:
    metadata_path = folder / "calibration.json"
    if not metadata_path.is_file():
        return {}
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def require_calibration_metadata(metadata: dict[str, Any]) -> None:
    required = ["checker_board_columns", "checker_board_rows", "checker_board_size_mm"]
    missing = [key for key in required if metadata.get(key) in (None, "")]
    if missing:
        raise ValueError(f"calibration_metadata_missing: {', '.join(missing)}")


def video_files(folder: Path) -> list[Path]:
    return sorted(
        path for path in folder.iterdir()
        if path.is_file()
        and path.suffix.lower() in {".mp4", ".mov", ".webm", ".avi"}
        and not path.stem.lower().startswith("intrinsic_debug_")
    )


def camera_label_from_path(path: Path) -> str:
    match = re.search(r"(cam\d+)$", path.stem, re.IGNORECASE)
    return match.group(1).lower() if match else path.stem.lower()

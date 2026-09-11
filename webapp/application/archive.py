"""Zip helpers shared by the report export and the per-folder video export."""
from __future__ import annotations

import os
import tempfile
import zipfile
from pathlib import Path

VIDEO_SUFFIXES = {".mp4", ".avi"}
# Already-compressed media; deflating it costs CPU and saves nothing.
_STORED_SUFFIXES = {".mp4", ".avi", ".mov", ".webm", ".jpg", ".jpeg", ".png"}


def build_archive(files: list[Path], root: Path) -> Path:
    """Zip `files` with paths relative to `root`, into a temp file the caller deletes."""
    handle, temp_path = tempfile.mkstemp(prefix="h3dm_export_", suffix=".zip")
    os.close(handle)
    archive_path = Path(temp_path)
    try:
        with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as archive:
            for path in files:
                compression = (
                    zipfile.ZIP_STORED if path.suffix.lower() in _STORED_SUFFIXES
                    else zipfile.ZIP_DEFLATED
                )
                archive.write(path, path.relative_to(root), compress_type=compression)
    except Exception:
        archive_path.unlink(missing_ok=True)
        raise
    return archive_path


def is_visible_file(path: Path, root: Path) -> bool:
    """Skip dot-files and anything nested under a dot-directory."""
    return path.is_file() and not any(part.startswith(".") for part in path.relative_to(root).parts)


def resolve_inside(folder_path: str, storage_root: Path) -> Path:
    """Resolve a caller-supplied folder, refusing anything outside the storage root."""
    target = Path(str(folder_path or "").strip()).expanduser().resolve()
    if target != storage_root and storage_root not in target.parents:
        raise ValueError("path_outside_storage_root")
    if not target.is_dir():
        raise ValueError("folder_not_found")
    return target

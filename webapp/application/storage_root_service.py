from __future__ import annotations

from dataclasses import dataclass

from webapp.application.capture_service import CaptureService
from webapp.domain.ports import DirectorySelector


@dataclass(frozen=True)
class StorageRootSelection:
    root: str
    cancelled: bool
    manual_required: bool = False
    error: str | None = None


class StorageRootService:
    def __init__(
        self,
        capture_service: CaptureService,
        directory_selector: DirectorySelector,
    ) -> None:
        self._capture_service = capture_service
        self._directory_selector = directory_selector

    def get(self) -> str:
        return self._capture_service.get_storage_root()

    def set(self, storage_root: str) -> str:
        return self._capture_service.set_storage_root(storage_root)

    def select(self, requested_root: str, manual: bool = False) -> StorageRootSelection:
        requested_root = str(requested_root or "").strip()
        if manual and requested_root:
            return StorageRootSelection(root=self.set(requested_root), cancelled=False)

        current_root = self.get()
        try:
            selected_path = self._directory_selector.select_directory(current_root)
        except RuntimeError as exc:
            return StorageRootSelection(
                root=current_root,
                cancelled=True,
                manual_required=True,
                error=str(exc),
            )
        if selected_path is None:
            return StorageRootSelection(root=current_root, cancelled=True)
        return StorageRootSelection(root=self.set(selected_path), cancelled=False)

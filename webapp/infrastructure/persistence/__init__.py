"""Persistence adapters."""

from webapp.infrastructure.persistence.file_session_catalog import FileSessionCatalog
from webapp.infrastructure.persistence.file_settings import JsonSettingsRepository

__all__ = [
    "FileSessionCatalog",
    "JsonSettingsRepository",
]

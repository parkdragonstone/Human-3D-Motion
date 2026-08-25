from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_DIALOG_TITLE = "Select storage folder"
_APPLESCRIPT_USER_CANCELLED = "-128"


class TkinterDirectorySelector:
    def select_directory(self, initial_dir: str) -> str | None:
        if sys.platform == "darwin":
            # AppKit forbids creating windows outside the main thread, and Flask serves
            # requests on worker threads, so Tk would abort the process on macOS.
            return _select_directory_with_osascript(initial_dir)
        return _select_directory_with_tkinter(initial_dir)


def _select_directory_with_tkinter(initial_dir: str) -> str | None:
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        selected = filedialog.askdirectory(initialdir=initial_dir, title=_DIALOG_TITLE)
        root.destroy()
        return selected or None
    except Exception as exc:
        raise RuntimeError(f"folder_dialog_unavailable: {exc}") from exc


def _select_directory_with_osascript(initial_dir: str) -> str | None:
    script = "\n".join([
        "activate",
        f'set chosenFolder to choose folder with prompt "{_DIALOG_TITLE}"'
        f"{_applescript_default_location(initial_dir)}",
        "return POSIX path of chosenFolder",
    ])
    try:
        completed = subprocess.run(
            ["osascript", "-e", script],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as exc:
        raise RuntimeError(f"folder_dialog_unavailable: {exc}") from exc

    if completed.returncode != 0:
        message = (completed.stderr or "").strip()
        if _APPLESCRIPT_USER_CANCELLED in message:
            return None
        raise RuntimeError(f"folder_dialog_unavailable: {message or 'osascript_failed'}")

    selected = completed.stdout.strip().rstrip("/")
    return selected or None


def _applescript_default_location(initial_dir: str) -> str:
    candidate = str(initial_dir or "").strip()
    if not candidate or not Path(candidate).is_dir():
        return ""
    return f' default location POSIX file "{_escape_applescript_string(candidate)}"'


def _escape_applescript_string(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')

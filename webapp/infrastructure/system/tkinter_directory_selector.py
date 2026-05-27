from __future__ import annotations


class TkinterDirectorySelector:
    def select_directory(self, initial_dir: str) -> str | None:
        try:
            import tkinter as tk
            from tkinter import filedialog

            root = tk.Tk()
            root.withdraw()
            root.attributes("-topmost", True)
            selected = filedialog.askdirectory(initialdir=initial_dir, title="Select storage folder")
            root.destroy()
            return selected or None
        except Exception as exc:
            raise RuntimeError(f"folder_dialog_unavailable: {exc}") from exc

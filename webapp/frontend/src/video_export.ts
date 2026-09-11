// Export the videos of one session or calibration folder, from the list cards on
// the Capture and Calibration pages. Importing this module installs the handler.

interface SaveFilePickerOptions {
  suggestedName?: string;
  types?: Array<{ description: string; accept: Record<string, string[]> }>;
}

function suggestedArchiveName(folderPath: string): string {
  const folder = folderPath.split(/[\\/]/).filter(Boolean).pop() || "session";
  return `${folder}_videos.zip`;
}

async function exportFolderVideos(folderPath: string, button: HTMLButtonElement): Promise<void> {
  const suggestedName = suggestedArchiveName(folderPath);
  const picker = (window as unknown as {
    showSaveFilePicker?: (options: SaveFilePickerOptions) => Promise<FileSystemFileHandle>;
  }).showSaveFilePicker;

  let handle: FileSystemFileHandle | null = null;
  if (picker) {
    try {
      handle = await picker({
        suggestedName,
        types: [{ description: "Zip archive", accept: { "application/zip": [".zip"] } }],
      });
    } catch {
      return; // The operator dismissed the save dialog.
    }
  }

  const label = button.textContent;
  button.disabled = true;
  button.textContent = "...";
  try {
    const response = await fetch("/api/videos/export", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ path: folderPath }),
    });
    if (!response.ok) throw new Error(await response.text());
    if (handle && response.body) {
      // Streamed straight to disk so a large session never sits in memory.
      await response.body.pipeTo(await handle.createWritable());
    } else {
      const url = URL.createObjectURL(await response.blob());
      const link = document.createElement("a");
      link.href = url;
      link.download = suggestedName;
      link.click();
      URL.revokeObjectURL(url);
    }
  } catch (error) {
    window.alert(error instanceof Error ? error.message : "Export failed.");
  } finally {
    button.disabled = false;
    button.textContent = label;
  }
}

document.addEventListener("click", (event) => {
  const button = (event.target as HTMLElement).closest<HTMLButtonElement>("[data-export-videos]");
  const folderPath = button?.dataset.exportVideos;
  if (!button || !folderPath) return;
  void exportFolderVideos(folderPath, button);
});

from __future__ import annotations


class PipelineCalibrationRunner:
    def run(self, folder_path: str, metadata: dict | None = None) -> dict:
        from pipelines.calibration import run_calibration_folder

        return run_calibration_folder(folder_path, metadata)

from __future__ import annotations

from pathlib import Path


class PipelineReportRunner:
    """Reduces a trial to the gait parameters the report cards are drawn from.

    The analysis pipeline stops after inverse kinematics, so this runs later, once the
    Report page has supplied the motion and the direction the subject walked in.
    """

    def build_gait_parameters(
        self,
        session_path: str,
        motion: str,
        walking_direction: str,
        subject_metadata: dict | None,
        filter_config: dict | None,
    ) -> dict:
        from pipelines.gait_report import export_gait_parameters_csv, heading_degrees

        output_path, parameters, heading = export_gait_parameters_csv(
            Path(session_path),
            walking_direction,
            filter_config,
            subject_metadata=subject_metadata,
            motion=motion,
        )
        return {
            "csv_path": str(output_path),
            "parameters": parameters,
            "heading": {
                "degrees": round(heading_degrees(heading["heading"]), 2),
                "source": heading["source"],
                "confident": bool(heading["confident"]),
            },
        }

    def read_gait_parameters(self, session_path: str) -> tuple[str, list[dict]] | None:
        from pipelines.gait_report import read_gait_parameters_csv

        csv_path = _latest_parameters_csv(Path(session_path))
        if csv_path is None:
            return None
        return str(csv_path), read_gait_parameters_csv(csv_path)


def _latest_parameters_csv(session_dir: Path) -> Path | None:
    files = [path for path in session_dir.glob("*_gait_parameters.csv") if path.is_file()]
    if not files:
        return None
    return max(files, key=lambda path: path.stat().st_mtime)

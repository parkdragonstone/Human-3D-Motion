from __future__ import annotations

from pathlib import Path

from webapp.application.session_query_service import SessionQueryService
from webapp.domain.ports import ReportRunner

MOTIONS = ("Walking",)
# "auto" reads the heading off the walking path; the axes stay as overrides.
WALKING_DIRECTIONS = ("auto", "+x", "-x", "+z", "-z")


class ReportService:
    def __init__(
        self,
        report_runner: ReportRunner,
        config_provider,
        session_query_service: SessionQueryService,
    ) -> None:
        self._report_runner = report_runner
        self._config_provider = config_provider
        self._session_query_service = session_query_service

    def options(self) -> dict:
        return {"motions": list(MOTIONS), "walking_directions": list(WALKING_DIRECTIONS)}

    def run(self, session_path: str, motion: str, walking_direction: str) -> dict:
        session = self._session_query_service.require_by_path(session_path)
        motion = self._validated_motion(motion)
        walking_direction = self._validated_direction(walking_direction)
        result = self._report_runner.build_gait_parameters(
            str(Path(session.session_path).resolve()),
            motion,
            walking_direction,
            {
                "name": session.subject.name,
                "height": session.subject.height_cm,
                "weight": session.subject.weight_kg,
                "hand": session.subject.hand,
            },
            self._kinematics_filter(),
        )
        return {
            "available": True,
            "motion": motion,
            "walking_direction": walking_direction,
            "csv_file": Path(result["csv_path"]).name,
            "parameters": result["parameters"],
            "heading": result["heading"],
        }

    def parameters(self, session_path: str) -> dict:
        session = self._session_query_service.require_by_path(session_path)
        result = self._report_runner.read_gait_parameters(str(Path(session.session_path).resolve()))
        if result is None:
            return {"available": False, "parameters": []}
        csv_path, parameters = result
        return {"available": True, "csv_file": Path(csv_path).name, "parameters": parameters}

    def _kinematics_filter(self) -> dict:
        kinematics = self._config_provider.default_config().get("kinematics") or {}
        return kinematics.get("filter") or {}

    @staticmethod
    def _validated_motion(motion: str) -> str:
        normalized = str(motion or "").strip()
        for candidate in MOTIONS:
            if normalized.lower() == candidate.lower():
                return candidate
        raise ValueError("invalid_motion")

    @staticmethod
    def _validated_direction(walking_direction: str) -> str:
        normalized = str(walking_direction or "auto").strip().lower()
        if normalized not in WALKING_DIRECTIONS:
            raise ValueError("invalid_walking_direction")
        return normalized

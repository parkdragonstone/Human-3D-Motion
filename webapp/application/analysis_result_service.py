from __future__ import annotations

from pathlib import Path

from webapp.domain.entities import CaptureSession
from webapp.domain.ports import AnalysisResultGateway


class AnalysisResultService:
    def __init__(self, result_gateway: AnalysisResultGateway) -> None:
        self._result_gateway = result_gateway

    def pose3d_data_from_trc(self, trc_path: Path) -> dict:
        return self._result_gateway.pose3d_data_from_trc(str(trc_path))

    def keypoint_frame_from_json(self, session_path: str, camera_label: str, frame: int) -> dict:
        return self._result_gateway.keypoint_frame_from_json(session_path, camera_label, frame)

    def save_keypoint_frame_to_json(
        self,
        session_path: str,
        camera_label: str,
        frame: int,
        people_keypoints: list,
    ) -> None:
        self._result_gateway.save_keypoint_frame_to_json(session_path, camera_label, frame, people_keypoints)

    def render_pose_video_from_keypoints(self, session: CaptureSession, camera_label: str) -> Path:
        return Path(self._result_gateway.render_pose_video_from_keypoints(session, camera_label))

    def latest_kinematics_csv_file(self, session_path: Path) -> Path | None:
        csv_path = self._result_gateway.latest_kinematics_csv_file(str(session_path))
        return Path(csv_path) if csv_path is not None else None

    def read_csv_columns(self, csv_path: Path) -> dict[str, list[float]]:
        return self._result_gateway.read_csv_columns(str(csv_path))

    def recalculate_kinematics_event_markers(
        self,
        csv_path: Path,
    ) -> list[dict[str, float | int | str]]:
        return self._result_gateway.recalculate_kinematics_event_markers(str(csv_path))

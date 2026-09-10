from __future__ import annotations

import math
from pathlib import Path

from webapp.domain.entities import CaptureSession
from webapp.domain.ports import AnalysisResultGateway


class AnalysisResultService:
    def __init__(self, result_gateway: AnalysisResultGateway) -> None:
        self._result_gateway = result_gateway

    def pose3d_data_from_trc(self, trc_path: Path) -> dict:
        return self._result_gateway.pose3d_data_from_trc(str(trc_path))

    def list_pose3d_files(self, session_path: Path) -> list[str]:
        return self._result_gateway.list_pose3d_files(str(session_path))

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

    def kinematics_series(self, session_path: Path) -> dict[str, list[float]]:
        return self._result_gateway.kinematics_series(str(session_path))

    def kinematics_summary(self, session_path: Path) -> dict:
        columns = self.kinematics_series(session_path)
        if not columns:
            return {"available": False, "signals": [], "unit": "deg", "events": []}
        signals = [signal for signal in _kinematics_signals() if signal["key"] in columns]
        return {
            "available": True,
            "source": "kinematics/*.mot",
            "unit": "deg",
            "signals": signals,
            "events": [],
        }

    def kinematics_timeseries(self, session_path: Path, signal: str) -> dict:
        signal_map = {item["key"]: item for item in _kinematics_signals()}
        if signal not in signal_map:
            raise ValueError("invalid_signal")
        columns = self.kinematics_series(session_path)
        if not columns:
            raise ValueError("kinematics_mot_not_found")
        if signal not in columns:
            raise ValueError("signal_not_found")
        return {
            "unit": signal_map[signal]["unit"],
            "time": _finite_or_null(columns.get("time", [])),
            "values": _finite_or_null(columns.get(signal, [])),
        }


def _kinematics_signals() -> list[dict[str, str]]:
    angle_signals = [
        {"key": "hip_flexion_l", "label": "Hip Flexion", "side": "Left", "category": "hip"},
        {"key": "hip_flexion_r", "label": "Hip Flexion", "side": "Right", "category": "hip"},
        {"key": "hip_adduction_l", "label": "Hip Abduction", "side": "Left", "category": "hip"},
        {"key": "hip_adduction_r", "label": "Hip Abduction", "side": "Right", "category": "hip"},
        {"key": "hip_rotation_l", "label": "Hip Rotation", "side": "Left", "category": "hip"},
        {"key": "hip_rotation_r", "label": "Hip Rotation", "side": "Right", "category": "hip"},
        {"key": "pelvis_tilt", "label": "Pelvis Tilt", "side": "Center", "category": "pelvis"},
        {"key": "pelvis_list", "label": "Pelvis List", "side": "Center", "category": "pelvis"},
        {"key": "pelvis_rotation", "label": "Pelvis Rotation", "side": "Center", "category": "pelvis"},
        {"key": "lumbar_Flex_Ext", "label": "Flexion", "side": "Center", "category": "hip_shoulder"},
        {"key": "lumbar_Lat_Bending", "label": "Lateral Bend", "side": "Center", "category": "hip_shoulder"},
        {"key": "lumbar_axial_rotation", "label": "Rotation", "side": "Center", "category": "hip_shoulder"},
        {"key": "trunk_tilt_global", "label": "Trunk Tilt", "side": "Global", "category": "trunk"},
        {"key": "trunk_list_global", "label": "Trunk List", "side": "Global", "category": "trunk"},
        {"key": "trunk_rotation_global", "label": "Trunk Rotation", "side": "Global", "category": "trunk"},
        {"key": "knee_angle_l", "label": "Knee Flexion", "side": "Left", "category": "knee"},
        {"key": "knee_angle_r", "label": "Knee Flexion", "side": "Right", "category": "knee"},
        {"key": "ankle_angle_l", "label": "Ankle Dorsiflexion", "side": "Left", "category": "ankle"},
        {"key": "ankle_angle_r", "label": "Ankle Dorsiflexion", "side": "Right", "category": "ankle"},
        {"key": "arm_flex_l", "label": "Shoulder Flexion", "side": "Left", "category": "shoulder"},
        {"key": "arm_flex_r", "label": "Shoulder Flexion", "side": "Right", "category": "shoulder"},
        {"key": "arm_add_l", "label": "Shoulder Adduction", "side": "Left", "category": "shoulder"},
        {"key": "arm_add_r", "label": "Shoulder Adduction", "side": "Right", "category": "shoulder"},
        {"key": "arm_rot_l", "label": "Shoulder Rotation", "side": "Left", "category": "shoulder"},
        {"key": "arm_rot_r", "label": "Shoulder Rotation", "side": "Right", "category": "shoulder"},
        {"key": "elbow_flex_l", "label": "Elbow Flexion", "side": "Left", "category": "elbow"},
        {"key": "elbow_flex_r", "label": "Elbow Flexion", "side": "Right", "category": "elbow"},
        {"key": "pro_sup_l", "label": "Pronation Supination", "side": "Left", "category": "elbow"},
        {"key": "pro_sup_r", "label": "Pronation Supination", "side": "Right", "category": "elbow"},
    ]
    signals = [{**signal, "kind": "angle", "unit": "deg"} for signal in angle_signals]
    signals.extend([
        {
            **signal,
            "key": f"{signal['key']}_velocity",
            "kind": "velocity",
            "unit": "deg/s",
        }
        for signal in angle_signals
    ])
    return signals


def _finite_or_null(values: list[float]) -> list[float | None]:
    return [round(value, 4) if isinstance(value, float) and math.isfinite(value) else None for value in values]

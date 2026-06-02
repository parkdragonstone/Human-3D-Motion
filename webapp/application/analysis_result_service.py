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

    def latest_kinematics_csv_file(self, session_path: Path) -> Path | None:
        csv_path = self._result_gateway.latest_kinematics_csv_file(str(session_path))
        return Path(csv_path) if csv_path is not None else None

    def read_csv_columns(self, csv_path: Path) -> dict[str, list[float]]:
        return self._result_gateway.read_csv_columns(str(csv_path))

    def kinematics_summary(self, session_path: Path) -> dict:
        csv_path = self.latest_kinematics_csv_file(session_path)
        if csv_path is None:
            return {"available": False, "signals": [], "unit": "deg"}
        columns = self.read_csv_columns(csv_path)
        signals = [signal for signal in _kinematics_signals() if signal["key"] in columns]
        return {
            "available": True,
            "file": csv_path.name,
            "unit": "deg",
            "signals": signals,
            "events": self.kinematics_event_markers(csv_path, columns),
        }

    def kinematics_timeseries(self, session_path: Path, signal: str) -> dict:
        signal_map = {item["key"]: item for item in _kinematics_signals()}
        if signal not in signal_map:
            raise ValueError("invalid_signal")
        csv_path = self.latest_kinematics_csv_file(session_path)
        if csv_path is None:
            raise ValueError("kinematics_csv_not_found")
        columns = self.read_csv_columns(csv_path)
        if signal not in columns:
            raise ValueError("signal_not_found")
        return {
            "unit": signal_map[signal]["unit"],
            "time": _finite_or_null(columns.get("time", [])),
            "values": _finite_or_null(columns.get(signal, [])),
        }

    def kinematics_event_markers(
        self,
        csv_path: Path,
        columns: dict[str, list[float]],
    ) -> list[dict[str, float | int | str]]:
        recalculated = self.recalculate_kinematics_event_markers(csv_path)
        if recalculated:
            return recalculated
        if not all(f"{key}_time" in columns for key in ("knee_high", "mer", "ball_release")):
            return []
        events = [
            ("knee_high", "KH", "Knee High"),
            ("mer", "MER", "Max Shoulder External Rotation"),
            ("ball_release", "BR", "Ball Release"),
        ]
        markers: list[dict[str, float | int | str]] = []
        for key, label, description in events:
            frame = _first_finite(columns.get(f"{key}_frame", []))
            time = _first_finite(columns.get(f"{key}_time", []))
            if time is None:
                continue
            marker: dict[str, float | int | str] = {
                "key": key,
                "label": label,
                "description": description,
                "time": time,
            }
            if frame is not None:
                marker["frame"] = int(frame)
            markers.append(marker)
        return markers

    def recalculate_kinematics_event_markers(
        self,
        csv_path: Path,
    ) -> list[dict[str, float | int | str]]:
        return self._result_gateway.recalculate_kinematics_event_markers(str(csv_path))


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
        {"key": "L5_S1_Flex_Ext", "label": "Flexion", "side": "Center", "category": "hip_shoulder"},
        {"key": "L5_S1_Lat_Bending", "label": "Lateral Bend", "side": "Center", "category": "hip_shoulder"},
        {"key": "L5_S1_axial_rotation", "label": "Rotation", "side": "Center", "category": "hip_shoulder"},
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


def _first_finite(values: list[float]) -> float | None:
    for value in values:
        if isinstance(value, float) and math.isfinite(value):
            return value
    return None


def _finite_or_null(values: list[float]) -> list[float | None]:
    return [round(value, 4) if isinstance(value, float) and math.isfinite(value) else None for value in values]

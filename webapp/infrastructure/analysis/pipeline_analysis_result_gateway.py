from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from webapp.domain.entities import CaptureSession


class PipelineAnalysisResultGateway:
    def list_pose3d_files(self, session_path: str) -> list[str]:
        pose3d_dir = Path(session_path) / "pose-3d"
        if not pose3d_dir.is_dir():
            return []
        return sorted(path.name for path in pose3d_dir.glob("*.trc") if path.is_file())

    def pose3d_data_from_trc(self, trc_path: str) -> dict:
        from pipelines.utilities import read_trc

        path = Path(trc_path)
        if not path.is_file() or path.suffix.lower() != ".trc":
            raise ValueError("trc_file_not_found")

        coordinates, _frames_col, time_col, markers, header = read_trc(path)
        fps = 30.0
        try:
            fps = float(header[2].split("\t")[0])
        except Exception:
            pass

        frames = []
        for _, row in coordinates.iterrows():
            values = row.tolist()
            frame_points = []
            for marker_index in range(len(markers)):
                offset = marker_index * 3
                frame_points.append([
                    _safe_float(values[offset] if offset < len(values) else None),
                    _safe_float(values[offset + 1] if offset + 1 < len(values) else None),
                    _safe_float(values[offset + 2] if offset + 2 < len(values) else None),
                ])
            frames.append(frame_points)

        return {
            "file": path.name,
            "markers": markers,
            "fps": fps,
            "num_frames": len(frames),
            "time": [_safe_float(value) for value in time_col.tolist()],
            "frames": frames,
        }

    def keypoint_frame_from_json(self, session_path: str, camera_label: str, frame: int) -> dict:
        path = _keypoint_json_path(session_path, camera_label, frame)
        payload = json.loads(path.read_text(encoding="utf-8"))
        people_keypoints = []
        max_keypoints = 0
        for person_index, person in enumerate(payload.get("people") or []):
            if not isinstance(person, dict):
                continue
            raw_keypoints = person.get("pose_keypoints_2d") or []
            keypoints = []
            for index in range(0, len(raw_keypoints), 3):
                keypoints.append({
                    "x": _json_number(raw_keypoints[index] if index < len(raw_keypoints) else None),
                    "y": _json_number(raw_keypoints[index + 1] if index + 1 < len(raw_keypoints) else None),
                    "score": _json_number(raw_keypoints[index + 2] if index + 2 < len(raw_keypoints) else None),
                })
            max_keypoints = max(max_keypoints, len(keypoints))
            people_keypoints.append({"person_index": person_index, "keypoints": keypoints})
        return {
            "camera_label": _normalized_pose_camera_label(camera_label),
            "frame": frame,
            "file": path.name,
            "keypoint_names": _keypoint_names(max_keypoints),
            "people": people_keypoints,
        }

    def save_keypoint_frame_to_json(
        self,
        session_path: str,
        camera_label: str,
        frame: int,
        people_keypoints: list,
    ) -> None:
        path = _keypoint_json_path(session_path, camera_label, frame)
        payload = json.loads(path.read_text(encoding="utf-8"))
        people = payload.setdefault("people", [])
        if not people:
            people.append({})
        for person_payload in people_keypoints:
            if not isinstance(person_payload, dict):
                raise ValueError("invalid_person_keypoints")
            person_index = int(person_payload.get("person_index", 0))
            while len(people) <= person_index:
                people.append({})
            flattened = []
            for point in person_payload.get("keypoints") or []:
                if not isinstance(point, dict):
                    raise ValueError("invalid_keypoint")
                for key in ("x", "y", "score"):
                    value = point.get(key)
                    if value is None or value == "":
                        flattened.append(None)
                        continue
                    number = float(value)
                    flattened.append(number if math.isfinite(number) else None)
            people[person_index]["pose_keypoints_2d"] = flattened
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    def render_pose_video_from_keypoints(self, session: CaptureSession, camera_label: str) -> str:
        from pipelines.utilities import draw_bounding_box, draw_keypts, draw_skel, setup_video, transcode_to_h264

        pose_label = _normalized_pose_camera_label(camera_label)
        video = next(
            (item for item in session.videos if _normalized_pose_camera_label(item.camera_label) == pose_label),
            None,
        )
        if video is None:
            raise ValueError("video_not_found")

        session_path = Path(session.session_path)
        json_dir = session_path / "pose" / f"{pose_label}_json"
        if not json_dir.is_dir():
            raise ValueError("keypoint_json_dir_not_found")
        output_dir = session_path / "pose"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{pose_label}_pose.mp4"
        temp_path = output_dir / f"{pose_label}_pose_tmp.mp4"

        cap, writer, _width, _height, _fps = setup_video(Path(video.path), temp_path, True)
        frame_index = 0
        try:
            while True:
                success, frame = cap.read()
                if not success:
                    break
                json_path = json_dir / f"{pose_label}_{frame_index:06d}.json"
                if json_path.is_file():
                    frame = _draw_keypoints_json(frame, json_path, draw_bounding_box, draw_keypts, draw_skel)
                writer.write(frame)
                frame_index += 1
        finally:
            cap.release()
            if writer is not None:
                writer.release()

        transcode_to_h264(temp_path)
        if output_path.exists():
            output_path.unlink()
        temp_path.replace(output_path)
        return str(output_path)

    def latest_kinematics_csv_file(self, session_path: str) -> str | None:
        path = Path(session_path)
        csv_files = [csv_path for csv_path in path.glob("*_keypoints_kinematics.csv") if csv_path.is_file()]
        if not csv_files:
            csv_files = [csv_path for csv_path in path.glob("*.csv") if csv_path.is_file()]
        if not csv_files:
            return None
        return str(max(csv_files, key=lambda csv_path: csv_path.stat().st_mtime))

    def read_csv_columns(self, csv_path: str) -> dict[str, list[float]]:
        path = Path(csv_path)
        if not path.is_file():
            raise ValueError("kinematics_csv_not_found")

        with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames:
                raise ValueError("invalid_csv_header")
            columns = {field: [] for field in reader.fieldnames}
            for row in reader:
                for field in columns:
                    try:
                        value = float(row.get(field, "nan"))
                    except ValueError:
                        value = float("nan")
                    columns[field].append(value)
        return columns

    def recalculate_kinematics_event_markers(self, csv_path: str) -> list[dict[str, float | int | str]]:
        from pipelines.parameters import extract_pitching_events_from_dataframe

        try:
            df = pd.read_csv(csv_path)
            if "hand" not in df.columns:
                return []
            hand_values = df["hand"].dropna()
            if hand_values.empty:
                return []
            events = extract_pitching_events_from_dataframe(df, str(hand_values.iloc[0]), _infer_csv_fps(df))
        except Exception:
            return []

        labels = {
            "knee_high": ("KH", "Knee High"),
            "mer": ("MER", "Max Shoulder External Rotation"),
            "ball_release": ("BR", "Ball Release"),
        }
        markers: list[dict[str, float | int | str]] = []
        for key, event in events.items():
            label, description = labels[key]
            time = event.get("time")
            if time is None:
                continue
            marker: dict[str, float | int | str] = {
                "key": key,
                "label": label,
                "description": description,
                "time": float(time),
            }
            frame = event.get("frame")
            if frame is not None:
                marker["frame"] = int(frame)
            markers.append(marker)
        return markers


def _safe_float(value) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return round(number, 4) if math.isfinite(number) else None


def _normalized_pose_camera_label(label: str) -> str:
    import re

    match = re.search(r"cam0*(\d+)$", str(label).lower())
    return f"cam{int(match.group(1))}" if match else str(label).strip().lower()


def _keypoint_json_path(session_path: str, camera_label: str, frame: int) -> Path:
    pose_label = _normalized_pose_camera_label(camera_label)
    if not pose_label:
        raise ValueError("camera_label_required")
    if frame < 0:
        raise ValueError("invalid_frame")
    path = Path(session_path) / "pose" / f"{pose_label}_json" / f"{pose_label}_{frame:06d}.json"
    if not path.is_file():
        raise ValueError("keypoint_json_not_found")
    return path


def _keypoint_names(count: int) -> list[str]:
    halpe26 = [
        "Nose", "LEye", "REye", "LEar", "REar", "LShoulder", "RShoulder", "LElbow", "RElbow",
        "LWrist", "RWrist", "LHip", "RHip", "LKnee", "RKnee", "LAnkle", "RAnkle", "Head",
        "Neck", "Hip", "LBigToe", "RBigToe", "LSmallToe", "RSmallToe", "LHeel", "RHeel",
    ]
    return [halpe26[index] if index < len(halpe26) else f"Keypoint {index}" for index in range(count)]


def _json_number(value) -> float | None:
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _draw_keypoints_json(frame, json_path: Path, draw_bounding_box, draw_keypts, draw_skel):
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    keypoints = []
    scores = []
    for person in payload.get("people") or []:
        if not isinstance(person, dict):
            continue
        raw = np.asarray(person.get("pose_keypoints_2d") or [], dtype=float)
        if raw.size == 0 or raw.size % 3 != 0:
            continue
        points = raw.reshape((-1, 3))
        keypoints.append(points[:, :2])
        scores.append(points[:, 2])
    if not keypoints:
        return frame

    keypoints_array = np.asarray(keypoints, dtype=float)
    scores_array = np.asarray(scores, dtype=float)
    frame = draw_bounding_box(
        frame,
        keypoints_array[:, :, 0],
        keypoints_array[:, :, 1],
        fontSize=2,
    )
    frame = draw_keypts(frame, keypoints_array[:, :, 0], keypoints_array[:, :, 1], scores_array, cmap_str="RdYlGn")
    return draw_skel(frame, keypoints_array[:, :, 0], keypoints_array[:, :, 1])


def _infer_csv_fps(df) -> float:
    if "time" not in df.columns:
        return 60.0

    time_values = pd.to_numeric(df["time"], errors="coerce").to_numpy(dtype=float)
    dt = np.diff(time_values)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if len(dt) == 0:
        return 60.0
    return 1.0 / float(np.median(dt))

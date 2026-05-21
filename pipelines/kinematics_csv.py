"""Export combined 3D keypoint and kinematics CSV files."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from scipy.spatial.transform import Rotation

from .parameters import extract_pitching_events_from_dataframe
from .utilities import read_trc


PELVIS_GLOBAL_COLUMNS = ("pelvis_tilt", "pelvis_list", "pelvis_rotation")
L5_S1_JOINT_COLUMNS = ("L5_S1_Flex_Ext", "L5_S1_Lat_Bending", "L5_S1_axial_rotation")
TRUNK_GLOBAL_COLUMNS = ("trunk_tilt_global", "trunk_list_global", "trunk_rotation_global")
OPENSIM_EULER_SEQUENCE = "ZXY"
CONVERT_SIGN = ["elbow_flex_r_velocity", "elbow_flex_l_velocity", "arm_rot_r", "arm_rot_l"]

def export_combined_kinematics_csv(
    session_dir: Path,
    mot_path: Path,
    trc_path: Path,
    filter_config: dict | None = None,
    subject_metadata: dict | None = None,
    fps: float | int | None = None,
) -> Path:
    mot_df, in_degrees = _read_mot_dataframe(mot_path)
    trc_df = _read_trc_dataframe(trc_path)
    _validate_time_alignment(mot_df, trc_df, mot_path, trc_path)

    kinematics_df = _filter_angle_dataframe(mot_df.copy(), filter_config)
    trunk_global_angles = _trunk_global_angles(kinematics_df, in_degrees, mot_path)
    for index, column in enumerate(TRUNK_GLOBAL_COLUMNS):
        kinematics_df[column] = trunk_global_angles[:, index]
    kinematics_df = _filter_columns(kinematics_df, TRUNK_GLOBAL_COLUMNS, filter_config)
    velocity_df = _velocity_dataframe(kinematics_df, "time")
    
    combined_df = pd.concat([trc_df, kinematics_df.drop(columns=["time"]), velocity_df], axis=1)
    sign_columns = [column for column in CONVERT_SIGN if column in combined_df.columns]
    combined_df[sign_columns] = combined_df[sign_columns] * -1
    combined_df = _prepend_subject_metadata(combined_df, subject_metadata)
    combined_df = _append_pitching_parameters(combined_df, subject_metadata, fps)
    output_path = session_dir / f"{mot_path.stem}_keypoints_kinematics.csv"
    combined_df.to_csv(output_path, index=False)
    return output_path


def _prepend_subject_metadata(df: pd.DataFrame, subject_metadata: dict | None) -> pd.DataFrame:
    if not subject_metadata:
        return df
    metadata = {
        "name": subject_metadata.get("name"),
        "height": subject_metadata.get("height"),
        "weight": subject_metadata.get("weight"),
        "hand": subject_metadata.get("hand"),
    }
    metadata_df = pd.DataFrame({column: [value] * len(df) for column, value in metadata.items()})
    return pd.concat([metadata_df, df], axis=1)


def _append_pitching_parameters(
    df: pd.DataFrame,
    subject_metadata: dict | None,
    fps: float | int | None,
) -> pd.DataFrame:
    if not subject_metadata or fps is None:
        return df
    events = extract_pitching_events_from_dataframe(df, str(subject_metadata.get("hand") or ""), float(fps))
    for event_name, event in events.items():
        df[f"{event_name}_frame"] = event["frame"]
        df[f"{event_name}_time"] = event["time"]
    return df


def resolve_keypoint_trc(session_dir: Path, fallback_trc_path: Path) -> Path:
    pose3d_dir = session_dir / "pose-3d"
    candidates = [
        pose3d_dir / "butterworth.trc",
        pose3d_dir / "keypoints_3d_filt_butterworth.trc",
    ]
    candidates.extend(sorted(pose3d_dir.glob("*butterworth.trc")))
    candidates.append(fallback_trc_path)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"No butterworth TRC file found in {pose3d_dir}")


def _read_mot_dataframe(mot_path: Path) -> tuple[pd.DataFrame, bool]:
    headers = None
    data_start = 0
    in_degrees = False
    with mot_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_number, line in enumerate(handle):
            stripped = line.strip()
            if "inDegrees" in stripped:
                lowered = stripped.lower()
                in_degrees = "yes" in lowered or "true" in lowered
            if line.strip() == "endheader":
                headers = handle.readline().strip().split()
                data_start = line_number + 2
                break
    if not headers:
        raise ValueError(f"Invalid MOT header: {mot_path}")
    mot_df = pd.read_csv(mot_path, sep=r"\s+", skiprows=data_start, names=headers, engine="python")
    return mot_df, in_degrees


def _trunk_global_angles(mot_df: pd.DataFrame, in_degrees: bool, mot_path: Path) -> np.ndarray:
    required_columns = PELVIS_GLOBAL_COLUMNS + L5_S1_JOINT_COLUMNS
    missing_columns = [column for column in required_columns if column not in mot_df.columns]
    if missing_columns:
        raise ValueError(f"Cannot compute trunk global angles; missing {missing_columns} in {mot_path}")

    pelvis_global = mot_df.loc[:, PELVIS_GLOBAL_COLUMNS].to_numpy(dtype=float)
    l5_s1_joint = mot_df.loc[:, L5_S1_JOINT_COLUMNS].to_numpy(dtype=float)
    pelvis_rotation = Rotation.from_euler(OPENSIM_EULER_SEQUENCE, pelvis_global, degrees=in_degrees)
    l5_s1_rotation = Rotation.from_euler(OPENSIM_EULER_SEQUENCE, l5_s1_joint, degrees=in_degrees)
    trunk_rotation = pelvis_rotation * l5_s1_rotation
    return trunk_rotation.as_euler(OPENSIM_EULER_SEQUENCE, degrees=in_degrees)


def _filter_angle_dataframe(df: pd.DataFrame, filter_config: dict | None) -> pd.DataFrame:
    columns = [column for column in df.columns if column != "time"]
    return _filter_columns(df, columns, filter_config)


def _filter_columns(df: pd.DataFrame, columns: tuple[str, ...] | list[str], filter_config: dict | None) -> pd.DataFrame:
    if not filter_config:
        return df
    cutoff = float(filter_config.get("cut_off_frequency", 0) or 0)
    order = int(filter_config.get("order", 0) or 0)
    if cutoff <= 0 or order <= 0:
        return df
    if "time" not in df.columns:
        raise ValueError("Cannot filter kinematics without a time column")
    sample_frequency = _sample_frequency(df["time"].to_numpy(dtype=float))
    nyquist = sample_frequency / 2
    if cutoff >= nyquist:
        raise ValueError(f"Kinematics filter cutoff must be below Nyquist frequency ({nyquist:.3f} Hz)")
    b, a = butter(order, cutoff / nyquist, btype="low")
    for column in columns:
        if column not in df.columns:
            continue
        values = pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=float)
        df[column] = _butterworth_filter_series(values, b, a)
    return df


def _sample_frequency(time_values: np.ndarray) -> float:
    if len(time_values) < 2:
        raise ValueError("Cannot filter kinematics with fewer than two samples")
    dt = np.diff(time_values)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if len(dt) == 0:
        raise ValueError("Cannot filter kinematics with invalid time values")
    return 1.0 / float(np.median(dt))


def _butterworth_filter_series(values: np.ndarray, b: np.ndarray, a: np.ndarray) -> np.ndarray:
    valid = np.isfinite(values)
    if valid.sum() < max(len(a), len(b)) * 3:
        return values
    filled = pd.Series(values).interpolate(limit_direction="both").to_numpy(dtype=float)
    return filtfilt(b, a, filled)


def _read_trc_dataframe(trc_path: Path) -> pd.DataFrame:
    coordinates, frames_col, time_col, markers, _header = read_trc(trc_path)
    data = {
        "frame": frames_col.to_numpy(),
        "time": time_col.to_numpy(),
    }
    values = coordinates.to_numpy()
    axes = ("x", "y", "z")
    for marker_index, marker in enumerate(markers):
        for axis_index, axis in enumerate(axes):
            data[f"keypoint_{marker}_{axis}"] = values[:, marker_index * 3 + axis_index]
    return pd.DataFrame(data)


def _validate_time_alignment(mot_df: pd.DataFrame, trc_df: pd.DataFrame, mot_path: Path, trc_path: Path) -> None:
    if len(mot_df) != len(trc_df):
        raise ValueError(
            f"Cannot combine MOT/TRC with different row counts: {mot_path.name}={len(mot_df)}, "
            f"{trc_path.name}={len(trc_df)}"
        )
    if "time" not in mot_df.columns:
        raise ValueError(f"MOT file has no time column: {mot_path}")
    if not np.allclose(mot_df["time"].to_numpy(), trc_df["time"].to_numpy(), rtol=0, atol=1e-6):
        raise ValueError(f"Cannot combine MOT/TRC with different time columns: {mot_path.name}, {trc_path.name}")


def _velocity_dataframe(df: pd.DataFrame, time_column: str) -> pd.DataFrame:
    time_values = df[time_column].to_numpy(dtype=float)
    velocity_columns = {}
    for column in df.columns:
        if column == time_column:
            continue
        values = pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=float)
        velocity_columns[f"{column}_velocity"] = _three_point_derivative(values, time_values)
    return pd.DataFrame(velocity_columns)


def _three_point_derivative(values: np.ndarray, time_values: np.ndarray) -> np.ndarray:
    derivative = np.full(values.shape, np.nan, dtype=float)
    if len(values) < 2:
        return derivative

    derivative[0] = _slope(values[0], values[1], time_values[0], time_values[1])
    derivative[-1] = _slope(values[-2], values[-1], time_values[-2], time_values[-1])
    for index in range(1, len(values) - 1):
        derivative[index] = _slope(values[index - 1], values[index + 1], time_values[index - 1], time_values[index + 1])
    return derivative


def _slope(value_before: float, value_after: float, time_before: float, time_after: float) -> float:
    dt = time_after - time_before
    if not np.isfinite(dt) or dt == 0:
        return np.nan
    return (value_after - value_before) / dt

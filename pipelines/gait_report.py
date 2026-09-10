"""Reduce a walking trial to the gait parameters a report is built from.

Unlike the kinematics time series, this keeps only summary values: how many steps
were taken, how long and wide they were, and the joint excursions over the trial.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .kinematics_series import kinematics_dataframe, resolve_mot_file
from .parameters import (
    extract_walking_events_from_dataframe,
    heading_from_direction,
    project_ap,
    project_ml,
)
from .utilities import read_trc


METADATA_COLUMNS = ["name", "height", "weight", "hand", "motion", "walking_direction", "heading_deg"]

# A path that doubles back covers ground without going anywhere, so trust the
# detected heading only when the trial actually progresses along it.
MIN_PATH_PROGRESSION = 0.15
MIN_NET_TRAVEL_M = 0.20
PARAMETER_COLUMNS = ["parameter", "label", "side", "value", "unit"]


def export_gait_parameters_csv(
    session_dir: Path,
    walking_direction: str,
    filter_config: dict | None = None,
    subject_metadata: dict | None = None,
    motion: str = "Walking",
) -> tuple[Path, list[dict], dict]:
    session_dir = Path(session_dir)
    mot_path = resolve_mot_file(session_dir)
    if mot_path is None:
        raise ValueError("kinematics_mot_not_found")
    trc_path = resolve_keypoint_trc(session_dir)
    if trc_path is None:
        raise ValueError("keypoint_trc_not_found")

    angles_df = kinematics_dataframe(mot_path, filter_config)
    keypoints_df = read_trc_dataframe(trc_path)
    rows = min(len(angles_df), len(keypoints_df))
    angles_df = angles_df.iloc[:rows].reset_index(drop=True)
    keypoints_df = keypoints_df.iloc[:rows].reset_index(drop=True)

    heading = resolve_heading(keypoints_df, walking_direction)
    parameters = compute_gait_parameters(keypoints_df, angles_df, heading)
    output_path = session_dir / f"{mot_path.stem}_gait_parameters.csv"
    _write_parameters_csv(output_path, parameters, subject_metadata, motion, walking_direction, heading)
    return output_path, parameters, heading


def detect_walking_heading(keypoints_df: pd.DataFrame) -> dict | None:
    """Principal direction of the hip's horizontal path, oriented toward net travel.

    Subjects rarely walk along a coordinate axis, so the heading is measured from the
    motion itself instead of being declared up front.
    """
    x_column, z_column = "keypoint_Hip_x", "keypoint_Hip_z"
    if x_column not in keypoints_df.columns or z_column not in keypoints_df.columns:
        return None
    points = np.column_stack([
        pd.to_numeric(keypoints_df[x_column], errors="coerce").to_numpy(dtype=float),
        pd.to_numeric(keypoints_df[z_column], errors="coerce").to_numpy(dtype=float),
    ])
    points = points[np.isfinite(points).all(axis=1)]
    if len(points) < 5:
        return None

    centered = points - points.mean(axis=0)
    _u, singular, right_vectors = np.linalg.svd(centered, full_matrices=False)
    heading = right_vectors[0]
    net = points[-1] - points[0]
    if float(heading @ net) < 0:
        heading = -heading
    norm = float(np.linalg.norm(heading))
    if norm == 0:
        return None
    heading = heading / norm

    net_distance = float(np.linalg.norm(net))
    path_length = float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())
    spread = float(singular[0] + singular[1])
    return {
        "heading": (float(heading[0]), float(heading[1])),
        "net_distance": net_distance,
        "path_length": path_length,
        "straightness": float(singular[0] / spread) if spread > 0 else 0.0,
        "progression": net_distance / path_length if path_length > 0 else 0.0,
    }


def resolve_heading(keypoints_df: pd.DataFrame, walking_direction: str) -> dict:
    """Heading to project onto, plus how much the trial supports it."""
    requested = str(walking_direction or "auto").strip().lower()
    if requested != "auto":
        return {
            "heading": heading_from_direction(requested),
            "source": "manual",
            "confident": True,
            "net_distance": None,
            "progression": None,
            "straightness": None,
        }

    detected = detect_walking_heading(keypoints_df)
    if detected is None:
        return {
            "heading": heading_from_direction("-z"),
            "source": "fallback",
            "confident": False,
            "net_distance": None,
            "progression": None,
            "straightness": None,
        }
    confident = (
        detected["progression"] >= MIN_PATH_PROGRESSION
        and detected["net_distance"] >= MIN_NET_TRAVEL_M
    )
    return {**detected, "source": "auto", "confident": confident}


def heading_degrees(heading: tuple[float, float]) -> float:
    """Heading angle in the ground plane, measured from +x toward +z."""
    return float(np.degrees(np.arctan2(heading[1], heading[0])))


def compute_gait_parameters(
    keypoints_df: pd.DataFrame,
    angles_df: pd.DataFrame,
    heading: dict,
) -> list[dict]:
    unit = heading["heading"]
    time_values = keypoints_df["time"].to_numpy(dtype=float)
    duration = float(time_values[-1] - time_values[0]) if len(time_values) > 1 else 0.0

    # Everything spatial is measured along and across the direction of travel, so a
    # diagonal walk is handled the same way as one aligned to an axis.
    forward = {marker: project_ap(keypoints_df, marker, unit).to_numpy(dtype=float)
               for marker in ("Hip", "RHeel", "LHeel")}
    lateral = {marker: project_ml(keypoints_df, marker, unit).to_numpy(dtype=float)
               for marker in ("RHeel", "LHeel")}

    events = extract_walking_events_from_dataframe(
        pd.concat([keypoints_df, angles_df.drop(columns=["time"], errors="ignore")], axis=1),
        unit,
    )
    right_hc = _event_times(events.get("right_hc"))
    left_hc = _event_times(events.get("left_hc"))
    right_to = _event_times(events.get("right_to"))
    left_to = _event_times(events.get("left_to"))

    contacts = sorted(
        [(time, "right") for time in right_hc] + [(time, "left") for time in left_hc],
        key=lambda item: item[0],
    )
    step_count = max(0, len(contacts) - 1)

    parameters: list[dict] = [
        _parameter("duration", "Trial Duration", None, duration, "s"),
        _parameter("steps", "Total Steps", None, float(step_count), ""),
        _parameter("cadence", "Cadence", None, _cadence(step_count, duration), "steps/min"),
        _parameter("gait_speed", "Gait Speed", None, _gait_speed(forward["Hip"], duration), "m/s"),
    ]

    step_lengths = _step_lengths(forward, time_values, contacts)
    parameters.append(_parameter("step_length", "Step Length", None, _mean(step_lengths), "m"))
    parameters.append(_parameter("step_width", "Step Width", None, _step_width(lateral, time_values, contacts), "m"))

    for side, heel_contacts, marker in (("right", right_hc, "RHeel"), ("left", left_hc, "LHeel")):
        parameters.append(_parameter(
            "stride_length", "Stride Length", side,
            _mean(_stride_lengths(forward[marker], time_values, heel_contacts)), "m",
        ))
        parameters.append(_parameter(
            "stride_time", "Stride Time", side, _mean(_intervals(heel_contacts)), "s",
        ))

    for side, heel_contacts, toe_offs in (("right", right_hc, right_to), ("left", left_hc, left_to)):
        stance = _stance_percent(heel_contacts, toe_offs)
        parameters.append(_parameter("stance_phase", "Stance Phase", side, stance, "% cycle"))
        parameters.append(_parameter(
            "swing_phase", "Swing Phase", side,
            None if stance is None else 100.0 - stance, "% cycle",
        ))

    parameters.append(_parameter(
        "double_support", "Double Support", None,
        _double_support_percent(right_hc, right_to, left_hc, left_to), "% cycle",
    ))

    for side, suffix in (("right", "r"), ("left", "l")):
        parameters.append(_parameter(
            "peak_knee_flexion", "Peak Knee Flexion", side, _maximum(angles_df, f"knee_angle_{suffix}"), "deg",
        ))
        parameters.append(_parameter("knee_rom", "Knee ROM", side, _range_of_motion(angles_df, f"knee_angle_{suffix}"), "deg"))
        parameters.append(_parameter("hip_rom", "Hip ROM", side, _range_of_motion(angles_df, f"hip_flexion_{suffix}"), "deg"))
        parameters.append(_parameter("ankle_rom", "Ankle ROM", side, _range_of_motion(angles_df, f"ankle_angle_{suffix}"), "deg"))

    parameters.append(_parameter("trunk_lean", "Trunk Lean", None, _mean_absolute(angles_df, "trunk_tilt_global"), "deg"))
    parameters.append(_parameter("pelvic_obliquity", "Pelvic Obliquity", None, _range_of_motion(angles_df, "pelvis_list"), "deg"))

    # Surfaced so a detected heading can be sanity-checked against the recording.
    parameters.append(_parameter("path_heading", "Path Heading", None, heading_degrees(unit), "deg"))
    progression = heading.get("progression")
    parameters.append(_parameter(
        "path_confidence", "Path Confidence", None,
        None if progression is None else progression * 100.0, "%",
    ))
    parameters.append(_parameter("path_distance", "Path Distance", None, heading.get("net_distance"), "m"))
    return parameters


# ── file helpers ──────────────────────────────────────────────────────────────

def resolve_keypoint_trc(session_dir: Path) -> Path | None:
    pose3d_dir = Path(session_dir) / "pose-3d"
    candidates = [
        pose3d_dir / "butterworth.trc",
        pose3d_dir / "keypoints_3d_filt_butterworth.trc",
    ]
    candidates.extend(sorted(pose3d_dir.glob("*butterworth.trc")))
    candidates.extend(sorted(pose3d_dir.glob("*.trc")))
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def read_trc_dataframe(trc_path: Path) -> pd.DataFrame:
    coordinates, frames_col, time_col, markers, _header = read_trc(Path(trc_path))
    data = {"frame": frames_col.to_numpy(), "time": time_col.to_numpy()}
    values = coordinates.to_numpy()
    for marker_index, marker in enumerate(markers):
        for axis_index, axis in enumerate(("x", "y", "z")):
            data[f"keypoint_{marker}_{axis}"] = values[:, marker_index * 3 + axis_index]
    return pd.DataFrame(data)


def _write_parameters_csv(
    output_path: Path,
    parameters: list[dict],
    subject_metadata: dict | None,
    motion: str,
    walking_direction: str,
    heading: dict,
) -> None:
    metadata = {
        "name": (subject_metadata or {}).get("name"),
        "height": (subject_metadata or {}).get("height"),
        "weight": (subject_metadata or {}).get("weight"),
        "hand": (subject_metadata or {}).get("hand"),
        "motion": motion,
        "walking_direction": walking_direction,
        "heading_deg": round(heading_degrees(heading["heading"]), 2),
    }
    df = pd.DataFrame(parameters, columns=PARAMETER_COLUMNS)
    for column in reversed(METADATA_COLUMNS):
        df.insert(0, column, metadata.get(column))
    df.to_csv(output_path, index=False)


def read_gait_parameters_csv(csv_path: Path) -> list[dict]:
    df = pd.read_csv(csv_path)
    parameters: list[dict] = []
    for _index, row in df.iterrows():
        value = row.get("value")
        parameters.append({
            "parameter": _text(row.get("parameter")),
            "label": _text(row.get("label")),
            "side": _text(row.get("side")) or None,
            "value": None if pd.isna(value) else float(value),
            "unit": _text(row.get("unit")),
        })
    return parameters


# ── parameter maths ───────────────────────────────────────────────────────────

def _parameter(key: str, label: str, side: str | None, value: float | None, unit: str) -> dict:
    if value is not None and (not np.isfinite(value)):
        value = None
    return {
        "parameter": key,
        "label": label,
        "side": side or "",
        "value": None if value is None else round(float(value), 4),
        "unit": unit,
    }


def _event_times(events: list[dict] | None) -> list[float]:
    times = [event.get("time") for event in (events or []) if event.get("time") is not None]
    return sorted(float(time) for time in times)


def _cadence(step_count: int, duration: float) -> float | None:
    if duration <= 0 or step_count <= 0:
        return None
    return step_count / duration * 60.0


def _gait_speed(hip_forward: np.ndarray, duration: float) -> float | None:
    values = hip_forward[np.isfinite(hip_forward)]
    if duration <= 0 or len(values) < 2:
        return None
    return abs(float(values[-1] - values[0])) / duration


def _value_at_time(values: np.ndarray, time_values: np.ndarray, time: float) -> float | None:
    index = int(np.argmin(np.abs(time_values - time)))
    value = values[index]
    return float(value) if np.isfinite(value) else None


def _step_lengths(
    forward: dict[str, np.ndarray],
    time_values: np.ndarray,
    contacts: list[tuple[float, str]],
) -> list[float]:
    lengths: list[float] = []
    for (time_a, side_a), (time_b, side_b) in zip(contacts, contacts[1:]):
        if side_a == side_b:
            continue
        position_a = _value_at_time(forward["RHeel" if side_a == "right" else "LHeel"], time_values, time_a)
        position_b = _value_at_time(forward["RHeel" if side_b == "right" else "LHeel"], time_values, time_b)
        if position_a is None or position_b is None:
            continue
        lengths.append(abs(position_b - position_a))
    return lengths


def _stride_lengths(
    heel_forward: np.ndarray,
    time_values: np.ndarray,
    heel_contacts: list[float],
) -> list[float]:
    lengths: list[float] = []
    for time_a, time_b in zip(heel_contacts, heel_contacts[1:]):
        position_a = _value_at_time(heel_forward, time_values, time_a)
        position_b = _value_at_time(heel_forward, time_values, time_b)
        if position_a is None or position_b is None:
            continue
        lengths.append(abs(position_b - position_a))
    return lengths


def _step_width(
    lateral: dict[str, np.ndarray],
    time_values: np.ndarray,
    contacts: list[tuple[float, str]],
) -> float | None:
    widths: list[float] = []
    for time, _side in contacts:
        right = _value_at_time(lateral["RHeel"], time_values, time)
        left = _value_at_time(lateral["LHeel"], time_values, time)
        if right is None or left is None:
            continue
        widths.append(abs(right - left))
    return _mean(widths)


def _intervals(times: list[float]) -> list[float]:
    return [float(b - a) for a, b in zip(times, times[1:]) if b > a]


def _stance_percent(heel_contacts: list[float], toe_offs: list[float]) -> float | None:
    percents: list[float] = []
    for contact, next_contact in zip(heel_contacts, heel_contacts[1:]):
        cycle = next_contact - contact
        if cycle <= 0:
            continue
        toe_off = next((time for time in toe_offs if contact < time < next_contact), None)
        if toe_off is None:
            continue
        percents.append((toe_off - contact) / cycle * 100.0)
    return _mean(percents)


def _stance_intervals(heel_contacts: list[float], toe_offs: list[float]) -> list[tuple[float, float]]:
    """(heel contact, next toe off) windows — the times that foot is on the ground."""
    intervals: list[tuple[float, float]] = []
    for contact in heel_contacts:
        toe_off = next((time for time in toe_offs if time > contact), None)
        if toe_off is not None:
            intervals.append((contact, toe_off))
    return intervals


def _double_support_percent(
    right_hc: list[float],
    right_to: list[float],
    left_hc: list[float],
    left_to: list[float],
) -> float | None:
    """Share of a stride with both feet down, as the overlap of the two stance windows.

    Intersecting the intervals keeps this bounded by the cycle, which summing each
    foot's contribution separately does not.
    """
    right_stance = _stance_intervals(right_hc, right_to)
    left_stance = _stance_intervals(left_hc, left_to)
    percents: list[float] = []
    for contact, next_contact in zip(right_hc, right_hc[1:]):
        cycle = next_contact - contact
        if cycle <= 0:
            continue
        overlap = 0.0
        for right_start, right_end in right_stance:
            for left_start, left_end in left_stance:
                start = max(contact, right_start, left_start)
                end = min(next_contact, right_end, left_end)
                if end > start:
                    overlap += end - start
        percents.append(min(100.0, overlap / cycle * 100.0))
    return _mean(percents)


def _numeric_series(df: pd.DataFrame, column: str) -> np.ndarray | None:
    if column not in df.columns:
        return None
    values = pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    return values if len(values) > 0 else None


def _maximum(df: pd.DataFrame, column: str) -> float | None:
    values = _numeric_series(df, column)
    return None if values is None else float(np.max(values))


def _range_of_motion(df: pd.DataFrame, column: str) -> float | None:
    values = _numeric_series(df, column)
    return None if values is None else float(np.max(values) - np.min(values))


def _mean_absolute(df: pd.DataFrame, column: str) -> float | None:
    values = _numeric_series(df, column)
    return None if values is None else float(np.mean(np.abs(values)))


def _mean(values: list[float]) -> float | None:
    finite = [value for value in values if value is not None and np.isfinite(value)]
    return float(np.mean(finite)) if finite else None


def _text(value) -> str:
    return "" if value is None or (isinstance(value, float) and np.isnan(value)) else str(value)

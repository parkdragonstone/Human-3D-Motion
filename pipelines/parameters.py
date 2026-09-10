"""Extract gait event frames from 3D keypoints, along the direction of travel."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import find_peaks


# A heel contact every 0.35 s is already a 170 steps/min cadence, so anything closer
# together is ridge noise on the marker trace rather than a real event. Extrema also
# have to stand out from the signal's own range to count.
MIN_EVENT_INTERVAL_S = 0.35
MIN_EVENT_PROMINENCE_RATIO = 0.12


def extract_walking_events_from_dataframe(
    df: pd.DataFrame,
    heading: tuple[float, float] = (0.0, -1.0),
) -> dict[str, list[dict[str, float | int | None]]]:
    """Heel contacts and toe offs, from each foot's reach ahead of the hip."""
    df = df.reset_index(drop=True)
    hip_ap = project_ap(df, "Hip", heading)
    return {
        "right_hc": _events_from_local_extrema(df, project_ap(df, "RHeel", heading) - hip_ap, "max"),
        "right_to": _events_from_local_extrema(df, project_ap(df, "RBigToe", heading) - hip_ap, "min"),
        "left_hc": _events_from_local_extrema(df, project_ap(df, "LHeel", heading) - hip_ap, "max"),
        "left_to": _events_from_local_extrema(df, project_ap(df, "LBigToe", heading) - hip_ap, "min"),
    }


def _event_at(df: pd.DataFrame, index: int) -> dict[str, float | int | None]:
    event = {
        "index": int(index),
        "frame": _numeric_value(df, index, "frame"),
        "time": _numeric_value(df, index, "time"),
    }
    return event


def _numeric_value(df: pd.DataFrame, index: int, column: str) -> float | int | None:
    if column not in df.columns:
        return None
    value = df.at[index, column]
    if pd.isna(value):
        return None
    if column == "frame":
        return int(value)
    return float(value)


def _require_columns(df: pd.DataFrame, columns: list[str]) -> None:
    missing_columns = [column for column in columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")


def heading_from_direction(walking_direction: str) -> tuple[float, float]:
    """Map an axis label to a ground-plane unit vector.

    `-z` becomes (0, -1), which projects to `-z` exactly as the axis-indexed code did,
    so a manual direction reproduces the previous numbers.
    """
    normalized = str(walking_direction or "-z").strip().lower()
    headings = {"+x": (1.0, 0.0), "-x": (-1.0, 0.0), "+z": (0.0, 1.0), "-z": (0.0, -1.0)}
    if normalized not in headings:
        raise ValueError("walking_direction must be one of +x, -x, +z, -z")
    return headings[normalized]


def project_ap(df: pd.DataFrame, marker: str, heading: tuple[float, float]) -> pd.Series:
    """Marker position along the direction of travel."""
    return _projected(df, marker, heading[0], heading[1])


def project_ml(df: pd.DataFrame, marker: str, heading: tuple[float, float]) -> pd.Series:
    """Marker position across the direction of travel."""
    return _projected(df, marker, -heading[1], heading[0])


def _projected(df: pd.DataFrame, marker: str, unit_x: float, unit_z: float) -> pd.Series:
    x_column = f"keypoint_{marker}_x"
    z_column = f"keypoint_{marker}_z"
    _require_columns(df, [x_column, z_column])
    x_values = pd.to_numeric(df[x_column], errors="coerce")
    z_values = pd.to_numeric(df[z_column], errors="coerce")
    return x_values * unit_x + z_values * unit_z


def _events_from_local_extrema(
    df: pd.DataFrame,
    values: pd.Series,
    extrema_type: str,
) -> list[dict[str, float | int | None]]:
    numeric_values = values.to_numpy(dtype=float)
    if len(numeric_values) < 3:
        return []
    signal = numeric_values if extrema_type == "max" else -numeric_values
    finite = signal[np.isfinite(signal)]
    if len(finite) == 0:
        return []
    signal = pd.Series(signal).interpolate(limit_direction="both").to_numpy(dtype=float)

    span = float(np.max(finite) - np.min(finite))
    prominence = span * MIN_EVENT_PROMINENCE_RATIO if span > 0 else None
    distance = _minimum_peak_distance(df)
    peaks, _properties = find_peaks(signal, distance=distance, prominence=prominence)
    return [_event_at(df, int(index)) for index in peaks]


def _minimum_peak_distance(df: pd.DataFrame) -> int | None:
    if "time" not in df.columns:
        return None
    time_values = pd.to_numeric(df["time"], errors="coerce").to_numpy(dtype=float)
    steps = np.diff(time_values)
    steps = steps[np.isfinite(steps) & (steps > 0)]
    if len(steps) == 0:
        return None
    sample_interval = float(np.median(steps))
    return max(1, int(round(MIN_EVENT_INTERVAL_S / sample_interval)))

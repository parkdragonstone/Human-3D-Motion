"""Extract pitching event frames from the combined kinematics CSV."""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def extract_pitching_events(
    csv_path: str | Path,
    throwing_hand: str,
    fps: float | int,
) -> dict[str, dict[str, float | int | None]]:
    df = pd.read_csv(csv_path)
    return extract_pitching_events_from_dataframe(df, throwing_hand, fps)


def extract_pitching_events_from_dataframe(
    df: pd.DataFrame,
    throwing_hand: str,
    fps: float | int,
) -> dict[str, dict[str, float | int | None]]:
    df = df.reset_index(drop=True)
    side = _throwing_side(throwing_hand)
    lead_side = _opposite_side(side)
    br_offset = _br_offset(fps)

    knee_high_index = _max_value_index(df, f"keypoint_{lead_side}Knee_y")
    mer_index = _max_value_index(df, f"arm_rot_{side.lower()}")
    br_forward_index = _first_forward_wrist_index_after_elbow_passes_shoulder(
        df,
        mer_index,
        shoulder_column=f"keypoint_{side}Shoulder_z",
        elbow_column=f"keypoint_{side}Elbow_z",
        wrist_column=f"keypoint_{side}Wrist_z",
    )
    ball_release_index = br_forward_index + br_offset
    if ball_release_index >= len(df):
        raise ValueError("Ball release frame is outside the CSV range")

    return {
        "knee_high": _event_at(df, knee_high_index),
        "mer": _event_at(df, mer_index),
        "ball_release": _event_at(df, ball_release_index),
    }


def _throwing_side(throwing_hand: str) -> str:
    normalized = throwing_hand.strip().lower()
    if normalized in {"right", "r"}:
        return "R"
    if normalized in {"left", "l"}:
        return "L"
    raise ValueError("throwing_hand must be 'right' or 'left'")


def _opposite_side(side: str) -> str:
    if side == "R":
        return "L"
    if side == "L":
        return "R"
    raise ValueError("side must be 'R' or 'L'")


def _br_offset(fps: float | int) -> int:
    try:
        fps_value = float(fps)
    except (TypeError, ValueError) as exc:
        raise ValueError("fps must be a positive number") from exc
    if fps_value <= 0:
        raise ValueError("fps must be a positive number")
    if fps_value < 60:
        return 1
    if fps_value <= 120:
        return 2
    if fps_value <= 240:
        return 4
    return 5


def _max_value_index(df: pd.DataFrame, column: str) -> int:
    _require_columns(df, [column])
    values = pd.to_numeric(df[column], errors="coerce")
    valid_values = values.dropna()
    if valid_values.empty:
        raise ValueError(f"{column} has no numeric values")
    return int(valid_values.idxmax())


def _first_forward_wrist_index_after_elbow_passes_shoulder(
    df: pd.DataFrame,
    mer_index: int,
    shoulder_column: str,
    elbow_column: str,
    wrist_column: str,
) -> int:
    _require_columns(df, [shoulder_column, elbow_column, wrist_column])
    shoulder_z = pd.to_numeric(df[shoulder_column], errors="coerce")
    elbow_z = pd.to_numeric(df[elbow_column], errors="coerce")
    wrist_z = pd.to_numeric(df[wrist_column], errors="coerce")

    elbow_forward_mask = elbow_z.iloc[mer_index + 1 :] < shoulder_z.iloc[mer_index + 1 :]
    elbow_forward_indices = elbow_forward_mask[elbow_forward_mask].index
    if elbow_forward_indices.empty:
        raise ValueError("No elbow-forward frame found after MER")

    elbow_forward_index = int(elbow_forward_indices[0])
    wrist_forward_mask = wrist_z.iloc[elbow_forward_index + 1 :] < elbow_z.iloc[elbow_forward_index + 1 :]
    wrist_forward_indices = wrist_forward_mask[wrist_forward_mask].index
    if wrist_forward_indices.empty:
        raise ValueError("No wrist-forward frame found after elbow passed shoulder")
    return int(wrist_forward_indices[0])


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

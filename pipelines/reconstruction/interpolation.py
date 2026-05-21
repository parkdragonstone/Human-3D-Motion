from __future__ import annotations

import numpy as np
from scipy.interpolate import interp1d


def nan_runs(mask):
    runs = []
    start = None
    for index, is_missing in enumerate(mask):
        if is_missing and start is None:
            start = index
        elif not is_missing and start is not None:
            runs.append((start, index))
            start = None
    if start is not None:
        runs.append((start, len(mask)))
    return runs


def interpolate_series(values, max_gap, method):
    values = np.asarray(values, dtype=np.float64).copy()
    missing = ~np.isfinite(values)
    if not missing.any() or method == "none":
        return values, 0, 0

    finite_indices = np.flatnonzero(~missing)
    if finite_indices.size < 2:
        return values, 0, 1

    interpolated = 0
    x = finite_indices.astype(np.float64)
    y = values[finite_indices]

    kind = method if method in {"linear", "slinear", "quadratic", "cubic"} else "linear"
    min_points = {"linear": 2, "slinear": 2, "quadratic": 3, "cubic": 4}[kind]
    if finite_indices.size < min_points:
        return values, 0, 1

    interpolator = interp1d(x, y, kind=kind, bounds_error=False, fill_value=np.nan)
    for start, end in nan_runs(missing):
        gap = end - start
        bounded = start > finite_indices[0] and end - 1 < finite_indices[-1]
        if bounded and gap <= max_gap:
            gap_indices = np.arange(start, end, dtype=np.float64)
            values[start:end] = interpolator(gap_indices)
            interpolated += gap
    return values, interpolated, 0


def fill_remaining_gaps(values, fill_mode):
    values = np.asarray(values, dtype=np.float64).copy()
    missing = ~np.isfinite(values)
    if not missing.any() or fill_mode == "nan":
        return values
    if fill_mode == "zeros":
        values[missing] = 0.0
        return values
    if fill_mode != "last_value":
        return values

    finite_indices = np.flatnonzero(~missing)
    if finite_indices.size == 0:
        return values
    for index in range(1, len(values)):
        if not np.isfinite(values[index]) and np.isfinite(values[index - 1]):
            values[index] = values[index - 1]
    for index in range(len(values) - 2, -1, -1):
        if not np.isfinite(values[index]) and np.isfinite(values[index + 1]):
            values[index] = values[index + 1]
    return values


def interpolate_3d_keypoints(keypoints_3d, lifting_config):
    method = str(lifting_config.get("interpolation", "linear") or "none").lower()
    max_gap = int(lifting_config.get("interp_if_gap_smaller_than", 20))
    fill_mode = str(lifting_config.get("fill_large_gaps_with", "last_value") or "nan").lower()
    if method not in {"linear", "slinear", "quadratic", "cubic", "none"}:
        method = "linear"
    if method == "none" and fill_mode == "nan":
        return keypoints_3d, {
            "method": method,
            "fill_mode": fill_mode,
            "interpolated_values": 0,
            "skipped_series": 0,
            "remaining_nan_values": int(np.isnan(keypoints_3d).sum()),
        }

    result = np.asarray(keypoints_3d, dtype=np.float64).copy()
    interpolated_values = 0
    skipped_series = 0
    for keypoint_idx in range(result.shape[1]):
        for axis_idx in range(result.shape[2]):
            series, count, skipped = interpolate_series(result[:, keypoint_idx, axis_idx], max_gap, method)
            result[:, keypoint_idx, axis_idx] = fill_remaining_gaps(series, fill_mode)
            interpolated_values += count
            skipped_series += skipped

    return result, {
        "method": method,
        "fill_mode": fill_mode,
        "interpolated_values": int(interpolated_values),
        "skipped_series": int(skipped_series),
        "remaining_nan_values": int(np.isnan(result).sum()),
    }

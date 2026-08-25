"""Metric scaling and gravity alignment of a keypoint-calibrated scene.

Adapted from flodelaplace/lab-camera-dynamic-calibrator ``postprocessing/scale_scene.py``
(MIT).

The linear solve fixes the scene only up to a similarity transform: the reconstruction is
correct in shape but arbitrary in size, orientation and origin. Three cues from the
subject fix the remaining degrees of freedom:

* the head-to-foot axis, averaged over the sequence, is *up* (gravity);
* the lowest foot contact defines the floor plane, so the origin sits on the floor;
* the subject's known stature converts the arbitrary unit into metres.

The scene is emitted **Z-up**, matching the Object/CheckerBoard targets (a board lying
flat on the floor spans X/Y). The analysis pipeline remaps that to the Y-up convention
OpenSim expects, so all three calibration targets land in the same place.
"""

from __future__ import annotations

import numpy as np

from .skeleton import FOOT_JOINTS, HALPE26_KEY, HEAD_JOINTS, UP_AXIS_FOOT_JOINTS


def _valid_mean(points_NxJx3, joints) -> np.ndarray | None:
    selected = points_NxJx3[:, joints, :]
    finite = selected[~np.isnan(selected).any(axis=-1)]
    if finite.size == 0:
        return None
    return finite.reshape(-1, 3)


def estimate_up_axis(points_NxJx3) -> np.ndarray | None:
    """Mean foot-to-head direction over the sequence, normalized."""
    head_index = HALPE26_KEY["Head"]
    hip_index = HALPE26_KEY["MidHip"]

    vectors = []
    for frame in points_NxJx3:
        head = frame[head_index]
        feet = frame[UP_AXIS_FOOT_JOINTS]
        feet = feet[~np.isnan(feet).any(axis=-1)]
        if np.isnan(head).any() or feet.size == 0:
            hip = frame[hip_index]
            if np.isnan(hip).any() or feet.size == 0:
                continue
            vectors.append(hip - feet.mean(axis=0))
            continue
        vectors.append(head - feet.mean(axis=0))

    if not vectors:
        return None
    mean = np.mean(np.asarray(vectors), axis=0)
    norm = np.linalg.norm(mean)
    if norm < 1e-9:
        return None
    return mean / norm


def _rotation_between(source, target) -> np.ndarray:
    """Shortest rotation taking unit vector ``source`` onto unit vector ``target``."""
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    axis = np.cross(source, target)
    axis_norm = np.linalg.norm(axis)
    dot = float(np.clip(np.dot(source, target), -1.0, 1.0))
    if axis_norm < 1e-9:
        if dot > 0:
            return np.eye(3)
        # Anti-parallel: rotate half a turn about any perpendicular axis.
        fallback = np.array([1.0, 0.0, 0.0])
        if abs(source[0]) > 0.9:
            fallback = np.array([0.0, 1.0, 0.0])
        axis = np.cross(source, fallback)
        axis /= np.linalg.norm(axis)
        return -np.eye(3) + 2 * np.outer(axis, axis)

    axis = axis / axis_norm
    angle = np.arctan2(axis_norm, dot)
    skew = np.array([
        [0.0, -axis[2], axis[1]],
        [axis[2], 0.0, -axis[0]],
        [-axis[1], axis[0], 0.0],
    ])
    return np.eye(3) + np.sin(angle) * skew + (1 - np.cos(angle)) * (skew @ skew)


def measure_stature(points_NxJx3, up_axis) -> float | None:
    """Median head-above-floor height along ``up_axis``, in scene units."""
    heights = []
    for frame in points_NxJx3:
        head = frame[HEAD_JOINTS]
        head = head[~np.isnan(head).any(axis=-1)]
        feet = frame[FOOT_JOINTS]
        feet = feet[~np.isnan(feet).any(axis=-1)]
        if head.size == 0 or feet.size == 0:
            continue
        top = float(np.max(head @ up_axis))
        bottom = float(np.min(feet @ up_axis))
        if top > bottom:
            heights.append(top - bottom)
    if not heights:
        return None
    return float(np.median(heights))


def align_and_scale(R_w2c, t_w2c, points_NxJx3, subject_height_m: float | None):
    """Rotate the scene so up is +Z, drop the origin on the floor, and scale to metres.

    Returns ``(R_w2c, t_w2c, points, report)``. Camera rotations/translations are updated
    to express the same physical geometry in the new world frame.
    """
    points = np.asarray(points_NxJx3, dtype=np.float64)
    report: dict[str, object] = {"gravity_aligned": False, "scaled": False}

    up = estimate_up_axis(points)
    if up is None:
        return np.asarray(R_w2c), np.asarray(t_w2c).reshape(-1, 3, 1), points, report

    rotation = _rotation_between(up, np.array([0.0, 0.0, 1.0]))
    rotated = (rotation @ points.reshape(-1, 3).T).T.reshape(points.shape)
    report["gravity_aligned"] = True

    # Floor: the lowest foot contact seen over the sequence.
    feet = _valid_mean(rotated, FOOT_JOINTS)
    floor = float(np.percentile(feet[:, 2], 1.0)) if feet is not None else 0.0

    ground = _valid_mean(rotated, list(range(rotated.shape[1])))
    horizontal_origin = ground.mean(axis=0) if ground is not None else np.zeros(3)
    origin = np.array([horizontal_origin[0], horizontal_origin[1], floor])

    scale = 1.0
    measured = measure_stature(rotated, np.array([0.0, 0.0, 1.0]))
    if subject_height_m and measured and measured > 1e-6:
        scale = float(subject_height_m) / measured
        report["scaled"] = True
        report["measured_height_scene_units"] = measured
        report["scale_factor"] = scale

    transformed = (rotated - origin) * scale

    # World point x_new = scale * (rotation @ x_old - rotation @ origin_old)
    # so the camera pose that reproduces the same image is:
    #   R_new = R_old @ rotation^T,  t_new = t_old + R_new @ (scale * origin_in_rotated)
    R_old = np.asarray(R_w2c, dtype=np.float64)
    t_old = np.asarray(t_w2c, dtype=np.float64).reshape(-1, 3)
    R_new = np.array([R @ rotation.T for R in R_old])
    t_new = np.array([scale * t_old[c] + R_new[c] @ origin * scale for c in range(len(R_old))])

    report["floor_offset"] = float(floor)
    report["up_axis_scene"] = up.tolist()
    return R_new, t_new.reshape(-1, 3, 1), transformed, report

"""Multi-view geometry helpers used by the keypoint calibration.

Adapted from flodelaplace/lab-camera-dynamic-calibrator ``core/geometry.py`` (MIT).
"""

from __future__ import annotations

import numpy as np


def triangulate_dlt(points_Cx2, projections_Cx3x4) -> np.ndarray:
    """Direct linear transform triangulation from two or more views.

    ``points_Cx2`` may carry a third homogeneous column (bearing vectors); only the
    first two entries are used, so the same routine serves pixel coordinates with a
    ``K``-bearing projection matrix and normalized coordinates without one.
    """
    points = np.asarray(points_Cx2, dtype=np.float64)
    projections = np.asarray(projections_Cx3x4, dtype=np.float64)
    if len(points) != len(projections):
        raise ValueError("points and projection matrices must have the same length")

    AtA = np.zeros((4, 4), dtype=np.float64)
    row = np.zeros((2, 4), dtype=np.float64)
    for point, projection in zip(points, projections):
        row[0, :] = projection[0, :] - point[0] * projection[2, :]
        row[1, :] = projection[1, :] - point[1] * projection[2, :]
        AtA += row.T @ row

    _, vectors = np.linalg.eigh(AtA)
    solution = vectors[:, 0]
    if np.isclose(solution[3], 0.0):
        return solution
    return solution / solution[3]


def triangulate_points(p2d_CxNxJx2, s2d_CxNxJ, K_Cx3x3, R_w2c, t_w2c, conf_threshold: float) -> np.ndarray:
    """Triangulate every (frame, joint) seen by at least two confident cameras.

    Returns ``(N, J, 3)`` with NaN where fewer than two cameras contributed.
    """
    p2d = np.asarray(p2d_CxNxJx2, dtype=np.float64)
    s2d = np.asarray(s2d_CxNxJ, dtype=np.float64)
    C, N, J, _ = p2d.shape

    projections = np.array([
        K_Cx3x3[c] @ np.hstack([R_w2c[c], np.asarray(t_w2c[c], dtype=np.float64).reshape(3, 1)])
        for c in range(C)
    ])

    visible = s2d > conf_threshold
    points = np.full((N, J, 3), np.nan, dtype=np.float64)
    for frame in range(N):
        for joint in range(J):
            mask = visible[:, frame, joint]
            if mask.sum() < 2:
                continue
            points[frame, joint] = triangulate_dlt(p2d[mask, frame, joint, :], projections[mask])[:3]
    return points


def project(K, R_w2c, t_w2c, points_Nx3) -> np.ndarray:
    """Project world points into one camera. Returns ``(N, 2)`` pixel coordinates."""
    points = np.asarray(points_Nx3, dtype=np.float64)
    camera = K @ (R_w2c @ points.T + np.asarray(t_w2c, dtype=np.float64).reshape(3, 1))
    return (camera[:2, :] / camera[2, :]).T


def invert_rt(R_w2c, t_w2c) -> tuple[np.ndarray, np.ndarray]:
    """Convert a world-to-camera pose into the camera-to-world pose."""
    R = np.asarray(R_w2c, dtype=np.float64)
    t = np.asarray(t_w2c, dtype=np.float64).reshape(3)
    R_c2w = R.T
    return R_c2w, -R_c2w @ t


def cheirality_sign(R1, t1, R2, t2, n1, n2) -> int:
    """Resolve the global sign of the linear solution by a cheirality (z > 0) test."""
    import cv2

    def _triangulate(t_a, t_b):
        homogeneous = cv2.triangulatePoints(
            np.hstack([R1, np.asarray(t_a, dtype=np.float64).reshape(3, 1)]),
            np.hstack([R2, np.asarray(t_b, dtype=np.float64).reshape(3, 1)]),
            np.ascontiguousarray(n1[:, :2].T),
            np.ascontiguousarray(n2[:, :2].T),
        )
        homogeneous /= homogeneous[3, :]
        return homogeneous[:3, :].T

    def _in_front(R, t, points):
        camera = R @ points.T + np.asarray(t, dtype=np.float64).reshape(3, 1)
        return int(np.sum(camera[2, :] > 0))

    positive = _triangulate(t1, t2)
    negative = _triangulate(-np.asarray(t1), -np.asarray(t2))
    score_positive = _in_front(R1, t1, positive) + _in_front(R2, t2, positive)
    score_negative = _in_front(R1, -np.asarray(t1), negative) + _in_front(R2, -np.asarray(t2), negative)
    return 1 if score_positive >= score_negative else -1

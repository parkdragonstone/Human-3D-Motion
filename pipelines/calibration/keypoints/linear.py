"""Linear extrinsic calibration from human bone orientations and 2D projections.

Adapted from flodelaplace/lab-camera-dynamic-calibrator ``calibration/calib_linear.py``
(MIT). Two observations drive the solve:

* every bone direction is the *same* world vector seen by all cameras, so stacking the
  per-camera unit directions and taking a rank-3 SVD recovers all rotations at once;
* every joint is a single world point, giving collinearity constraints per camera and
  coplanarity constraints per camera pair, whose joint null space yields the
  translations up to one global scale and sign.
"""

from __future__ import annotations

import itertools

import numpy as np
import scipy as sp
import scipy.sparse

from .geometry import cheirality_sign, triangulate_dlt


def joints_to_orientations(p3d_CxNxJx3, mask_CxNxJ, bones_Bx2) -> np.ndarray:
    """Unit bone directions in each camera's own frame.

    Only bones whose two endpoints are valid in *every* camera are kept, because the
    rotation solve needs the same physical direction observed by all views.
    Returns ``(C, K, 3)``.
    """
    p3d = np.array(p3d_CxNxJx3, dtype=np.float64, copy=True)
    mask = np.asarray(mask_CxNxJ, dtype=bool)
    C, N = p3d.shape[0], p3d.shape[1]
    B = bones_Bx2.shape[0]

    p3d[~mask] = np.nan
    pairs = p3d[:, :, bones_Bx2, :]
    directions = (pairs[:, :, :, 1, :] - pairs[:, :, :, 0, :]).reshape((C, N * B, 3))

    keep = np.all(~np.isnan(directions), axis=(0, 2))
    directions = directions[:, keep, :]
    if directions.shape[1] == 0:
        return directions

    norms = np.linalg.norm(directions, axis=2)
    keep_nonzero = np.all(norms > 1e-12, axis=0)
    directions = directions[:, keep_nonzero, :]
    norms = norms[:, keep_nonzero]
    if directions.shape[1] == 0:
        return directions
    return directions / norms[:, :, None]


def joints_to_projections(p2d_CxNxJx2, mask_CxNxJ) -> np.ndarray:
    """2D joint observations visible in every camera. Returns ``(C, M, 2)``."""
    p2d = np.array(p2d_CxNxJx2, dtype=np.float64, copy=True)
    mask = np.asarray(mask_CxNxJ, dtype=bool)
    p2d[~mask] = np.nan

    flattened = p2d.reshape((p2d.shape[0], -1, 2))
    drop = np.isnan(flattened).any(axis=(0, 2))
    return flattened[:, ~drop, :]


def bearing_vectors(p2d_CxMx2, K_Cx3x3) -> np.ndarray:
    """Normalized homogeneous bearing vectors ``K^-1 [u, v, 1]``. Returns ``(C, M, 3)``."""
    p2d = np.asarray(p2d_CxMx2, dtype=np.float64)
    homogeneous = np.ones((p2d.shape[0], p2d.shape[1], 3), dtype=np.float64)
    homogeneous[:, :, :2] = p2d
    return np.array([
        homogeneous[c] @ np.linalg.inv(np.asarray(K_Cx3x3[c], dtype=np.float64)).T
        for c in range(p2d.shape[0])
    ])


def _collinearity_w2c(R_w2c, bearing, point_index, camera_index, num_points, num_cameras):
    """Row block stating that the bearing, the camera centre and the world point align."""
    skew = np.array([
        [0.0, -bearing[2], bearing[1]],
        [bearing[2], 0.0, -bearing[0]],
        [-bearing[1], bearing[0], 0.0],
    ])
    translation_offset = num_points * 3
    block = sp.sparse.lil_matrix((3, (num_points + num_cameras) * 3), dtype=np.float64)
    block[:, point_index * 3:point_index * 3 + 3] = skew @ R_w2c
    block[:, translation_offset + camera_index * 3:translation_offset + camera_index * 3 + 3] = skew
    return block


def _coplanarity_w2c(R_a, R_b, bearings_a, bearings_b, camera_a, camera_b, num_cameras):
    """Row block stating that both bearings and the baseline are coplanar."""
    normals = np.cross(bearings_a @ R_a, bearings_b @ R_b)
    block = sp.sparse.lil_matrix((bearings_a.shape[0], num_cameras * 3), dtype=np.float64)
    block[:, camera_a * 3:camera_a * 3 + 3] = normals @ R_a.T
    block[:, camera_b * 3:camera_b * 3 + 3] = -normals @ R_b.T
    return block


def calibrate_linear(v_CxKx3, n_CxMx3):
    """Estimate ``R_w2c`` and ``t_w2c`` for every camera plus the observed 3D points.

    ``v_CxKx3`` are unit bone directions per camera, ``n_CxMx3`` normalized bearing
    vectors of the covisible joints. Returns ``(R (C,3,3), t (C,3,1), X (M,3))`` or
    ``(None, None, None)`` when the configuration is degenerate.
    """
    v = np.asarray(v_CxKx3, dtype=np.float64)
    n = np.asarray(n_CxMx3, dtype=np.float64)
    C, M = v.shape[0], n.shape[1]
    if v.shape[1] < 3:
        return None, None, None

    # --- Rotations: one rank-3 factorization over all cameras at once -------------
    stacked = np.hstack(v)
    left, singular, right = np.linalg.svd(stacked)
    del left, singular
    rotations_3x3C = np.sqrt(C) * right[:3, :]
    # Gauge: make camera 0 the identity (this also fixes handedness).
    try:
        gauge = np.linalg.inv(rotations_3x3C[:3, :3])
    except np.linalg.LinAlgError:
        return None, None, None
    rotations_3x3C = gauge @ rotations_3x3C
    if np.linalg.det(rotations_3x3C[:3, :3]) <= 0:
        return None, None, None
    R_w2c = rotations_3x3C.T.reshape((-1, 3, 3))

    # --- Translations: null space of the collinearity + coplanarity system ---------
    collinearity = [
        _collinearity_w2c(R_w2c[camera], n[camera][point], point, camera, M, C)
        for camera in range(C)
        for point in range(M)
    ]
    if not collinearity:
        return None, None, None
    A = sp.sparse.vstack(collinearity)

    coplanarity = [
        _coplanarity_w2c(R_w2c[a], R_w2c[b], n[a], n[b], a, b, C)
        for a, b in itertools.combinations(range(C), 2)
    ]
    if not coplanarity:
        return None, None, None
    B = sp.sparse.vstack(coplanarity)

    system = sp.sparse.lil_matrix((A.shape[0] + B.shape[0], A.shape[1]), dtype=np.float64)
    system[:A.shape[0]] = A
    system[A.shape[0]:, -B.shape[1]:] = B

    try:
        eigenvalues, eigenvectors = sp.linalg.eigh(
            (system.T @ system).toarray(),
            subset_by_index=(0, 5),
            overwrite_a=True,
        )
    except (np.linalg.LinAlgError, ValueError):
        return None, None, None

    null_space = eigenvectors[:, :4]
    _, _, right_singular = np.linalg.svd(null_space[-B.shape[1]:-B.shape[1] + 3, :])
    solution = null_space @ right_singular[3, :].T

    points = solution[:-B.shape[1]].reshape((-1, 3))
    translations = solution[-B.shape[1]:]
    scale = np.linalg.norm(translations[3:6])
    if scale < 1e-9:
        return None, None, None
    translations = (translations / scale).reshape((-1, 3))
    points = points / scale

    sign = cheirality_sign(R_w2c[0], translations[0], R_w2c[1], translations[1], n[0], n[1])
    translations = sign * translations

    projections = np.concatenate((R_w2c, translations[:, :, None]), axis=2)
    triangulated = np.array([triangulate_dlt(n[:, i, :], projections)[:3] for i in range(M)])

    return R_w2c, translations.reshape((-1, 3, 1)), triangulated


def degenerate_eigenvalue_ratio(v_CxKx3) -> float:
    """Share of the bone-direction energy outside the top-3 singular values.

    A well-conditioned capture keeps this small; a person who barely changes pose makes
    the rotation factorization rank-deficient and the ratio approaches 1.
    """
    v = np.asarray(v_CxKx3, dtype=np.float64)
    if v.size == 0 or v.shape[1] < 3:
        return 1.0
    singular = np.linalg.svd(np.hstack(v), compute_uv=False)
    total = float(np.sum(singular))
    if total <= 0:
        return 1.0
    return float(1.0 - np.sum(singular[:3]) / total)

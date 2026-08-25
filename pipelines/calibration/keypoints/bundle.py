"""Bundle adjustment for keypoint-based extrinsic calibration.

Adapted from flodelaplace/lab-camera-dynamic-calibrator ``calibration/ba.py`` (MIT).

Three residual blocks are minimized jointly over the camera poses and the triangulated
joint cloud:

``nll``      confidence-weighted reprojection error of every visible joint;
``var3d``    disagreement between cameras about a bone direction in world space, which
             pins the rotations even where triangulation is weak (weight ``lambda1``);
``varbone``  per-bone length variance across frames -- human bones do not change length,
             so this removes the projective wobble the 2D term cannot see (weight
             ``lambda2``, auto-balanced to ~10% of the reprojection energy).
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import coo_matrix

from .geometry import triangulate_points


def pack_parameters(R_w2c, t_w2c, points_Nx3, focal_scales=None) -> np.ndarray:
    import cv2

    rotation_vectors = np.array([
        cv2.Rodrigues(np.asarray(R, dtype=np.float64))[0].ravel() for R in R_w2c
    ])
    blocks = [
        rotation_vectors.ravel(),
        np.asarray(t_w2c, dtype=np.float64).ravel(),
    ]
    if focal_scales is not None:
        blocks.append(np.asarray(focal_scales, dtype=np.float64).ravel())
    blocks.append(np.asarray(points_Nx3, dtype=np.float64).ravel())
    return np.hstack(blocks)


def unpack_parameters(theta, num_cameras: int, refine_focal: bool = False):
    """Split the parameter vector into rotations, translations, focal scales and points."""
    import cv2

    rotation_vectors = theta[0:3 * num_cameras].reshape((num_cameras, 3))
    translations = theta[3 * num_cameras:6 * num_cameras].reshape((num_cameras, 3, 1))
    offset = 6 * num_cameras
    if refine_focal:
        focal_scales = theta[offset:offset + num_cameras]
        offset += num_cameras
    else:
        focal_scales = np.ones(num_cameras, dtype=np.float64)
    points = theta[offset:].reshape((-1, 3))
    rotations = np.array([cv2.Rodrigues(vector)[0] for vector in rotation_vectors])
    return rotations, translations, focal_scales, points


def scale_focal(K_all, focal_scales) -> np.ndarray:
    """Apply a per-camera multiplier to fx/fy, leaving the principal point alone."""
    scaled = np.array(K_all, dtype=np.float64, copy=True)
    scaled[:, 0, 0] *= focal_scales
    scaled[:, 1, 1] *= focal_scales
    return scaled


def residual_reprojection(K_all, R_all, t_all, points, p2d_flat, visible_flat, conf_flat):
    """Confidence-weighted reprojection residuals, concatenated over cameras."""
    residuals = []
    for K, R, t, visible, observed, confidence in zip(
        K_all, R_all, t_all, visible_flat, p2d_flat, conf_flat
    ):
        if not np.any(visible):
            continue
        x = points[visible]
        y = observed[visible]
        weight = (np.sqrt(2.0) * confidence[visible]).reshape((-1, 1))
        projected = K @ (R @ x.T + np.asarray(t, dtype=np.float64).reshape((3, 1)))
        projected = (projected[:2, :] / projected[2, :]).T
        residuals.append(((y - projected) * weight).ravel())
    if not residuals:
        return np.zeros(0, dtype=np.float64)
    return np.concatenate(residuals)


def residual_direction_consistency(R_all, p3d_CxNxJx3, mask_CxNxJ, bones_Bx2):
    """1 - |mean unit bone direction| across cameras, once rotated into world space."""
    world = np.array([p3d @ R for R, p3d in zip(R_all, p3d_CxNxJx3)])
    world[~mask_CxNxJ] = np.nan

    pairs = world[:, :, bones_Bx2, :]
    directions = (pairs[:, :, :, 0, :] - pairs[:, :, :, 1, :]).reshape(world.shape[0], -1, 3)
    with np.errstate(invalid="ignore"):
        directions = directions / np.linalg.norm(directions, axis=2)[:, :, None]

    invalid = np.isnan(directions).any(axis=(0, 2))
    if np.all(invalid):
        return np.zeros(0, dtype=np.float64)
    return 1.0 - np.linalg.norm(np.nanmean(directions[:, ~invalid], axis=0), axis=1)


def residual_bone_length_variance(points_NxJx3, bones_Bx2, invalid_mask=None):
    """Variance of each bone length over the sequence."""
    points = np.asarray(points_NxJx3, dtype=np.float64)
    if invalid_mask is not None:
        flat = points.reshape(-1, 3).copy()
        flat[invalid_mask] = np.nan
        points = flat.reshape(points.shape)

    bone = points[:, bones_Bx2, :]
    lengths = np.linalg.norm(bone[:, :, 0, :] - bone[:, :, 1, :], axis=2)
    with np.errstate(invalid="ignore"):
        variance = np.nanvar(lengths, axis=0)
    variance[np.isnan(variance)] = 0.0
    return variance


def _objective(theta, K, p2d_flat, s2d, p3d, s3d, bones, C, N, J,
               lambda1, lambda2, invalid_mask, conf_threshold, refine_focal):
    R, t, focal_scales, points = unpack_parameters(theta, C, refine_focal)
    K_current = scale_focal(K, focal_scales) if refine_focal else K
    blocks = [
        residual_reprojection(
            K_current, R, t, points, p2d_flat,
            (s2d > conf_threshold).reshape((C, N * J)),
            s2d.reshape((C, N * J)),
        ),
        residual_direction_consistency(R, p3d, s3d > 0, bones) * lambda1,
        residual_bone_length_variance(points.reshape(N, J, 3), bones, invalid_mask) * lambda2,
    ]
    return np.concatenate(blocks)


def _jacobian_sparsity(C, N, J, s2d, s3d, bones, conf_threshold, refine_focal=False):
    """Which parameters can touch which residuals, so scipy skips dead finite differences.

    Parameter layout: rvecs, tvecs, [focal scales], then the 3D points.
    """
    pose_params = 6 * C
    num_camera_params = pose_params + (C if refine_focal else 0)
    num_params = num_camera_params + 3 * N * J

    visible = (s2d > conf_threshold).reshape(C, N * J)
    bone_visible = (s3d > 0)[:, :, bones[:, 0]] & (s3d > 0)[:, :, bones[:, 1]]
    num_direction = int(bone_visible.reshape(C, -1).all(axis=0).sum())
    num_reprojection = 2 * int(visible.sum())
    num_residuals = num_reprojection + num_direction + len(bones)

    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    row = 0

    for camera in range(C):
        indices = np.where(visible[camera])[0]
        if indices.size == 0:
            continue
        camera_columns = [
            np.arange(3 * camera, 3 * camera + 3),
            np.arange(3 * C + 3 * camera, 3 * C + 3 * camera + 3),
        ]
        if refine_focal:
            camera_columns.append(np.array([pose_params + camera]))
        camera_columns = np.concatenate(camera_columns)
        width = camera_columns.size + 3
        per_point = np.concatenate([
            np.broadcast_to(camera_columns, (indices.size, camera_columns.size)),
            num_camera_params + 3 * indices[:, None] + np.arange(3),
        ], axis=1)
        even = 2 * np.arange(indices.size) + row
        rows.append(np.repeat(even, width))
        rows.append(np.repeat(even + 1, width))
        cols.append(per_point.ravel())
        cols.append(per_point.ravel())
        row += 2 * indices.size

    if num_direction > 0:
        rows.append(np.repeat(np.arange(num_direction) + row, 3 * C))
        cols.append(np.tile(np.arange(3 * C), num_direction))
        row += num_direction

    frames = np.arange(N)
    for first, second in bones:
        point_columns = np.concatenate([
            (num_camera_params + 3 * (frames * J + first)[:, None] + np.arange(3)).ravel(),
            (num_camera_params + 3 * (frames * J + second)[:, None] + np.arange(3)).ravel(),
        ])
        rows.append(np.full(point_columns.size, row))
        cols.append(point_columns)
        row += 1

    if row != num_residuals:
        raise AssertionError(f"jacobian sparsity row mismatch: {row} vs {num_residuals}")

    row_index = np.concatenate(rows) if rows else np.zeros(0, dtype=int)
    col_index = np.concatenate(cols) if cols else np.zeros(0, dtype=int)
    pattern = coo_matrix(
        (np.ones(row_index.size, dtype=np.int8), (row_index, col_index)),
        shape=(num_residuals, num_params),
    )
    return pattern.tocsc()


def mean_reprojection_error(K_all, R_all, t_all, points_NxJx3, p2d, s2d, conf_threshold: float) -> float:
    """Mean pixel distance between observed and reprojected joints."""
    C, N, _J, _ = p2d.shape
    errors = []
    for camera in range(C):
        R = np.asarray(R_all[camera], dtype=np.float64)
        t = np.asarray(t_all[camera], dtype=np.float64).reshape(3)
        for frame in range(N):
            visible = s2d[camera, frame] > conf_threshold
            valid = visible & ~np.isnan(points_NxJx3[frame]).any(axis=1)
            if not np.any(valid):
                continue
            projected = K_all[camera] @ (R @ points_NxJx3[frame][valid].T + t[:, None])
            projected = (projected[:2, :] / projected[2, :]).T
            errors.append(np.linalg.norm(p2d[camera, frame][valid] - projected, axis=1))
    if not errors:
        return float("nan")
    return float(np.mean(np.concatenate(errors)))


def run_bundle_adjustment(
    K_all,
    R_w2c,
    t_w2c,
    p2d,
    s2d,
    p3d,
    s3d,
    bones,
    *,
    lambda1: float = 1.0,
    conf_threshold: float = 0.5,
    iterations: int = 2,
    max_evaluations: int | None = None,
    refine_focal: bool = False,
    focal_bounds: tuple[float, float] = (0.5, 2.0),
    progress=None,
):
    """Refine camera poses, optionally the focal lengths, and the joint cloud.

    ``refine_focal`` matters when the intrinsics were guessed from image resolution: a
    wrong focal would otherwise be absorbed by the extrinsics and bend every depth.
    Returns ``(R, t, points, K, report)``.
    """
    K_all = np.asarray(K_all, dtype=np.float64)
    p2d = np.asarray(p2d, dtype=np.float64)
    s2d_work = np.array(s2d, dtype=np.float64, copy=True)
    C, N, J, _ = p2d.shape

    R = np.asarray(R_w2c, dtype=np.float64)
    t = np.asarray(t_w2c, dtype=np.float64).reshape(C, 3, 1)
    focal_scales = np.ones(C, dtype=np.float64)
    K_current = np.array(K_all, copy=True)
    points = np.zeros((N * J, 3), dtype=np.float64)
    report: dict[str, object] = {"iterations": [], "refine_focal": bool(refine_focal)}

    for iteration in range(max(1, int(iterations))):
        points = triangulate_points(p2d, s2d_work, K_current, R, t, conf_threshold).reshape(N * J, 3)
        invalid = np.isnan(points).any(axis=1)
        points[invalid] = 0.0

        # A joint nobody could triangulate must not pull on the reprojection term.
        invalid_by_frame = invalid.reshape(N, J)
        s2d_work[:, invalid_by_frame] = 0.0

        p2d_flat = p2d.reshape((C, N * J, 2))
        visible_flat = (s2d_work > conf_threshold).reshape((C, N * J))

        reprojection = residual_reprojection(
            K_current, R, t, points, p2d_flat, visible_flat, s2d_work.reshape((C, N * J)),
        )
        bone_variance = residual_bone_length_variance(points.reshape(N, J, 3), bones, invalid)
        reprojection_energy = float(np.sum(reprojection ** 2))
        bone_energy = float(np.sum(bone_variance ** 2))

        # Keep the bone-length prior at ~10% of the reprojection energy: strong enough
        # to remove the projective wobble, weak enough not to bend the geometry.
        if bone_energy > 1e-3:
            lambda2 = min(float(np.sqrt(0.1 * reprojection_energy / bone_energy)), 1000.0)
        else:
            lambda2 = 0.0

        # The focal solve restarts from 1.0 each pass because K_current already carries
        # the previous pass's correction.
        theta0 = pack_parameters(R, t, points, np.ones(C) if refine_focal else None)
        sparsity = _jacobian_sparsity(C, N, J, s2d_work, s3d, bones, conf_threshold, refine_focal)
        evaluations = max_evaluations or min(max(60000, 4 * theta0.size), 80000)

        bounds = (-np.inf, np.inf)
        if refine_focal:
            lower = np.full(theta0.size, -np.inf)
            upper = np.full(theta0.size, np.inf)
            lower[6 * C:7 * C] = focal_bounds[0]
            upper[6 * C:7 * C] = focal_bounds[1]
            bounds = (lower, upper)

        result = least_squares(
            _objective,
            theta0,
            jac_sparsity=sparsity,
            bounds=bounds,
            method="trf",
            verbose=0,
            ftol=1e-7,
            xtol=1e-7,
            gtol=1e-7,
            max_nfev=evaluations,
            args=(K_current, p2d_flat, s2d_work, p3d, s3d, bones, C, N, J,
                  lambda1, lambda2, invalid, conf_threshold, refine_focal),
        )
        R, t, step_focal, points = unpack_parameters(result.x, C, refine_focal)
        if refine_focal:
            K_current = scale_focal(K_current, step_focal)
            focal_scales = focal_scales * step_focal

        error = mean_reprojection_error(
            K_current, R, t, points.reshape(N, J, 3), p2d, s2d_work, conf_threshold
        )
        report["iterations"].append({
            "iteration": iteration + 1,
            "lambda2": lambda2,
            "cost": float(result.cost),
            "mean_reprojection_px": error,
            "evaluations": int(result.nfev),
            "focal_scales": focal_scales.tolist() if refine_focal else None,
        })
        if progress:
            message = f"bundle adjustment {iteration + 1}/{iterations}: mean reprojection {error:.2f} px"
            if refine_focal:
                message += f" (focal x{np.mean(focal_scales):.3f})"
            progress(message)

    report["mean_reprojection_px"] = report["iterations"][-1]["mean_reprojection_px"]
    report["focal_scales"] = focal_scales.tolist()
    return R, t, points.reshape(N, J, 3), K_current, report

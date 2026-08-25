"""Camera intrinsics for automatic keypoint calibration.

When a session has no calibration upload the pipeline still needs a camera matrix to
start from. A resolution-derived guess is enough to seed the solve *because* the bundle
adjustment then refines the focal length against the observed reprojection error -- see
``refine_focal`` in :mod:`.bundle`. Without that refinement a wrong focal would be
absorbed by the extrinsics and quietly distort every reconstructed depth.
"""

from __future__ import annotations

from typing import Any

import numpy as np


# A typical machine-vision or phone lens sits near a 55-65 degree horizontal field of
# view, which puts the focal length around 0.9x the image width.
DEFAULT_FOCAL_RATIO = 0.9


def approximate_intrinsic(width: int, height: int, focal_ratio: float = DEFAULT_FOCAL_RATIO) -> dict[str, Any]:
    """Build a pinhole guess: principal point at the centre, focal from image width."""
    width = int(width)
    height = int(height)
    if width <= 0 or height <= 0:
        raise ValueError(f"bad_image_size: {width}x{height}")
    focal = float(width) * float(focal_ratio)
    return {
        "camera_matrix": [
            [focal, 0.0, width / 2.0],
            [0.0, focal, height / 2.0],
            [0.0, 0.0, 1.0],
        ],
        "dist_coeffs": [0.0, 0.0, 0.0, 0.0, 0.0],
        "image_size": [width, height],
        "approximate": True,
    }


def intrinsic_from_bundle(bundle: dict[str, Any] | None, camera_label: str) -> dict[str, Any] | None:
    """Find a camera matrix for ``camera_label`` in an uploaded calibration payload."""
    if not isinstance(bundle, dict):
        return None

    from ..metadata import intrinsic_for_label

    return intrinsic_for_label(bundle, camera_label)


def resolve_intrinsics(
    camera_labels: list[str],
    image_sizes_by_label: dict[str, tuple[int, int]],
    uploaded_bundle: dict[str, Any] | None = None,
    focal_ratio: float = DEFAULT_FOCAL_RATIO,
) -> tuple[dict[str, dict[str, Any]], bool]:
    """Prefer uploaded intrinsics, fall back to a resolution guess per camera.

    Returns ``(intrinsics_by_label, any_approximate)``. The flag tells the caller whether
    the focal length still has to be refined during bundle adjustment.
    """
    intrinsics: dict[str, dict[str, Any]] = {}
    approximate = False
    for label in camera_labels:
        found = intrinsic_from_bundle(uploaded_bundle, label)
        if found is not None:
            intrinsics[label] = found
            continue
        width, height = image_sizes_by_label[label]
        intrinsics[label] = approximate_intrinsic(width, height, focal_ratio)
        approximate = True
    return intrinsics, approximate


def camera_matrices(intrinsics_by_label: dict[str, dict[str, Any]], camera_labels: list[str]) -> np.ndarray:
    return np.array([
        np.asarray(intrinsics_by_label[label]["camera_matrix"], dtype=np.float64)
        for label in camera_labels
    ])

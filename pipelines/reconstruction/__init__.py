import os
import json
import itertools
import cv2
import numpy as np
import logging

from .interpolation import interpolate_3d_keypoints
from .keypoints import (
    camera_sort_key as _camera_sort_key,
    filter_frame_numbers as _filter_frame_numbers,
    get_frame_number,
    load_synchronized_kps,
    normalize_camera_label as _normalize_camera_label,
    person_to_keypoints as _person_to_keypoints,
    pose_json_dirs as _pose_json_dirs,
)
from ..utilities import export_to_trc

logger = logging.getLogger(__name__)

# =====================================================================
# 1. Body-segment proportions used for scale estimation.
#    Values are based on common anthropometric ratios such as David Winter's tables.
PROPORTIONS = {
    'upper_arm': 0.186,
    'lower_arm': 0.146,
    'thigh': 0.245,
    'calf': 0.246,
    'trunk': 0.288,
    'shoulder_width': 0.259,
    'pelvis_width': 0.191
}

TRC_MARKERS = [
    ("Hip", 19), ("RHip", 12), ("RKnee", 14), ("RAnkle", 16),
    ("RBigToe", 21), ("RSmallToe", 23), ("RHeel", 25),
    ("LHip", 11), ("LKnee", 13), ("LAnkle", 15),
    ("LBigToe", 20), ("LSmallToe", 22), ("LHeel", 24),
    ("Neck", 18), ("Head", 17), ("Nose", 0),
    ("RShoulder", 6), ("RElbow", 8), ("RWrist", 10),
    ("LShoulder", 5), ("LElbow", 7), ("LWrist", 9)
]

# Bone pairs used for scale estimation: ((joint_a, joint_b), proportion_name).
SCALE_BONES = [
    # Segment-length constraints.
    ((11, 13), 'thigh'), ((12, 14), 'thigh'),         # L/R Hip -> Knee
    ((13, 15), 'calf'), ((14, 16), 'calf'),           # L/R Knee -> Ankle
    ((18, 19), 'trunk'),                              # Neck(18) -> Pelvis_Center(19)
    ((5, 7), 'upper_arm'), ((6, 8), 'upper_arm'),     # L/R Shoulder -> Elbow
    ((7, 9), 'lower_arm'), ((8, 10), 'lower_arm'),    # L/R Elbow -> Wrist
    # Width constraints.
    ((5, 6), 'shoulder_width'),                       # L Shoulder(5) -> R Shoulder(6)
    ((11, 12), 'pelvis_width')                        # L Hip(11) -> R Hip(12)
]

## FUNCTIONS
def indices_of_first_last_non_nan_chunks(series, min_chunk_size=10, chunk_choice_method='largest', trim_output_chunk=True):
    '''
    Find indices of the chunks of at least min_chunk_size consecutive non-NaN values.

    INPUT:
    - series: pandas Series to trim
    - min_chunk_size: minimum size of consecutive non-NaN values to consider (default: 10)
    - chunk_choice_method: 'largest' to return the largest chunk, 'all' to return everything between the first and last non-nan chunk, 
                           'first' to return only the first one, 'last' to return only the last one
    - trim_output_chunk:   if True, the output chunk starts when all values are valid and ends at the first nan
                           else, it starts when at least on value is valid and ends when none is anymore

    OUTPUT:
    - tuple: (start_index, end_index) of the first and last valid chunks
    '''
    
    min_chunk_size = 10 if min_chunk_size == None else min_chunk_size
    non_nan_mask = ~np.isnan(series.values)
    
    # Find runs of consecutive non-NaN values (eg [(8, 15), (16, 17), (19, 26)])
    runs = []
    run_start = None
    for i, bool_val in enumerate(non_nan_mask):
        if bool_val and run_start is None:
            run_start = i
        elif not bool_val and run_start is not None:
            run_end = i
            runs.append((run_start, run_end))
            run_start = None
    if run_start is not None:
        runs.append((run_start, len(non_nan_mask)))
    
    # Find runs that have at least min_chunk_size consecutive non-NaN values
    valid_runs = [(start, end) for start, end in runs if end - start >= min_chunk_size]
    if not valid_runs:
        return(0,0)
    
    if chunk_choice_method not in ['largest', 'all', 'first', 'last']:
        chunk_choice_method = 'all'
    if chunk_choice_method == 'largest':
        # Choose the largest chunk
        valid_runs.sort(key=lambda x: x[1] - x[0], reverse=True)
        first_run_start, last_run_end = valid_runs[0]
    elif chunk_choice_method == 'all':
        # Get the start of the first valid run and the end of the last valid run
        first_run_start = valid_runs[0][0]
        last_run_end = valid_runs[-1][1]
    elif chunk_choice_method == 'first':
        # Get the start of the first valid run and the end of that run
        first_run_start, last_run_end = valid_runs[0]
    elif chunk_choice_method == 'last':
        # Get the start of the last valid run and the end of that run
        first_run_start, last_run_end = valid_runs[-1]
    
    # Return the trimmed series
    return first_run_start, last_run_end

# ---------------------------------------------------------------------
# 1. Resolve camera intrinsics.
#    - Prefer intrinsics from the calibration dict when present.
#    - Fall back to approximate intrinsics derived from camera resolution.
# ---------------------------------------------------------------------
def camera_intrinsic_from_config(calibration, camera_intrinsic_file, RESOLUTION_CAM1, RESOLUTION_CAM2):
    def estimate_intrinsics(width, height):
        """Estimate approximate intrinsics from image resolution."""
        cx, cy = width / 2.0, height / 2.0
        fx = fy = width * 0.8
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
        return K, np.zeros(5)

    # Source 1: calibration bundle.
    if isinstance(calibration, dict):
        # schema 1: {"intrinsic_cam1": {"camera_matrix": [...], "dist_coeffs":[...]}, "intrinsic_cam2": {...}}
        c1 = calibration.get('intrinsic_cam1') or calibration.get('cam1') or {}
        c2 = calibration.get('intrinsic_cam2') or calibration.get('cam2') or {}
        K1 = np.asarray(c1.get('camera_matrix'), dtype=np.float64) if c1.get('camera_matrix') is not None else None
        K2 = np.asarray(c2.get('camera_matrix'), dtype=np.float64) if c2.get('camera_matrix') is not None else None
        D1 = np.asarray(c1.get('dist_coeffs'), dtype=np.float64).reshape(-1) if c1.get('dist_coeffs') is not None else None
        D2 = np.asarray(c2.get('dist_coeffs'), dtype=np.float64).reshape(-1) if c2.get('dist_coeffs') is not None else None
        if K1 is not None and K2 is not None:
            if D1 is None: D1 = np.zeros(5, dtype=np.float64)
            if D2 is None: D2 = np.zeros(5, dtype=np.float64)
            logger.info("3D Lifting: Using calibration intrinsics from bundle.")
            return K1, D1, K2, D2, True

    # Source 2: reserved for legacy camera_intrinsic_file support.
    _ = camera_intrinsic_file

    # Source 3: approximate intrinsics from resolution.
    K1, dist1 = estimate_intrinsics(*RESOLUTION_CAM1)
    K2, dist2 = estimate_intrinsics(*RESOLUTION_CAM2)
    logger.info("3D Lifting: Using approximated intrinsics from resolution (no calibration intrinsics).")
    return K1, dist1, K2, dist2, False


def _score_person_combination(candidate_kps, camera_labels, projection_matrices, lifting_config):
    likelihood_threshold = float(lifting_config.get('likelihood_threshold_triangulation', 0.3))
    min_cameras = min(int(lifting_config.get('min_cameras_for_triangulation', 2)), len(camera_labels))
    errors = []
    valid_keypoints = 0

    for keypoint_idx in range(26):
        x_points = np.asarray([candidate_kps[label][keypoint_idx, 0] for label in camera_labels], dtype=np.float64)
        y_points = np.asarray([candidate_kps[label][keypoint_idx, 1] for label in camera_labels], dtype=np.float64)
        likelihoods = np.asarray([candidate_kps[label][keypoint_idx, 2] for label in camera_labels], dtype=np.float64)
        valid = (
            np.isfinite(x_points)
            & np.isfinite(y_points)
            & np.isfinite(likelihoods)
            & (likelihoods >= likelihood_threshold)
        )
        if int(np.count_nonzero(valid)) < min_cameras:
            continue

        projection_subset = [P for P, keep in zip(projection_matrices, valid) if keep]
        point_3d = _weighted_triangulation(projection_subset, x_points[valid], y_points[valid], likelihoods[valid])
        if not np.isfinite(point_3d).all():
            continue

        error = _reprojection_error(projection_subset, point_3d, x_points[valid], y_points[valid])
        if np.isfinite(error):
            errors.append(error)
            valid_keypoints += 1

    if not errors:
        return np.inf, 0
    return float(np.mean(errors)), valid_keypoints


def _select_people_by_reprojection(people_by_camera, camera_labels, projection_matrices, lifting_config):
    best = None
    best_score = (np.inf, 0)
    people_lists = [people_by_camera[label] for label in camera_labels]

    for combination in itertools.product(*people_lists):
        candidate_kps = {
            label: _person_to_keypoints(person)
            for label, person in zip(camera_labels, combination)
        }
        mean_error, valid_keypoints = _score_person_combination(
            candidate_kps,
            camera_labels,
            projection_matrices,
            lifting_config,
        )
        score = (mean_error, -valid_keypoints)
        if score < best_score:
            best_score = score
            best = candidate_kps

    return best, best_score


def load_synchronized_kps_multi_auto(camera_dirs, camera_labels, projection_matrices, lifting_config, frame_range=None):
    files_by_camera = {
        label: {get_frame_number(f): f for f in os.listdir(path) if f.endswith('.json')}
        for label, path in camera_dirs.items()
    }
    if not files_by_camera:
        return {}, [], {}

    common_frames = _filter_frame_numbers(
        set.intersection(*(set(files.keys()) for files in files_by_camera.values())),
        frame_range,
    )
    kps_by_camera = {label: [] for label in camera_labels}
    valid_frames = []
    selection_stats = {
        'frames_considered': len(common_frames),
        'frames_selected': 0,
        'mean_reprojection_error_px': None,
    }
    selected_errors = []

    for frame_idx in common_frames:
        people_by_camera = {}
        skip = False
        for label in camera_labels:
            directory = camera_dirs[label]
            with open(os.path.join(directory, files_by_camera[label][frame_idx])) as handle:
                data = json.load(handle)
            people = data.get('people', []) or []
            if len(people) == 0:
                skip = True
                break
            people_by_camera[label] = people
        if skip:
            continue

        selected, score = _select_people_by_reprojection(
            people_by_camera,
            camera_labels,
            projection_matrices,
            lifting_config,
        )
        if selected is None or not np.isfinite(score[0]):
            continue

        for label in camera_labels:
            kps_by_camera[label].append(selected[label])
        valid_frames.append(frame_idx)
        selected_errors.append(float(score[0]))

    selection_stats['frames_selected'] = len(valid_frames)
    if selected_errors:
        selection_stats['mean_reprojection_error_px'] = float(np.mean(selected_errors))
    return {label: np.asarray(values) for label, values in kps_by_camera.items()}, valid_frames, selection_stats

# =====================================================================
# 3. Build a rough 3D calibration skeleton for self-calibration.
# =====================================================================
def get_calibration_3d(kp1_all, kp2_all, target_height, calib_limit):
    actual_calib = min(calib_limit, len(kp1_all))
    p3d_raw = np.zeros((actual_calib, 26, 3))
    
    # Build an initial pseudo-3D skeleton from paired 2D keypoints.
    p3d_raw[:, :, 0] = kp1_all[:actual_calib, :, 0]
    p3d_raw[:, :, 2] = kp2_all[:actual_calib, :, 0]
    p3d_raw[:, :, 1] = -(kp1_all[:actual_calib, :, 1] + kp2_all[:actual_calib, :, 1]) / 2.0
    
    # Estimate a robust scale from segment-length constraints.
    scales = []
    for f in range(actual_calib):
        for (j1, j2), prop in SCALE_BONES:
            pixel_len = np.linalg.norm(p3d_raw[f, j1] - p3d_raw[f, j2])
            if pixel_len > 1.0:
                scales.append((target_height * PROPORTIONS[prop]) / pixel_len)
                
    robust_scale = np.median(scales)
    p3d_scaled = p3d_raw * robust_scale
    
    # Center coordinates around the pelvis in the first calibration frame.
    p3d_scaled -= p3d_scaled[0, 19]
    return p3d_scaled, actual_calib


def _has_calibration_intrinsics(calibration):
    if not isinstance(calibration, dict):
        return False
    cameras = calibration.get('cameras')
    if isinstance(cameras, dict) and len(cameras) >= 2:
        intrinsics_by_label = {
            _normalize_camera_label(label): intrinsic
            for label, intrinsic in calibration.items()
            if isinstance(intrinsic, dict)
        }
        return all(
            isinstance(intrinsics_by_label.get(_normalize_camera_label(label)), dict)
            and intrinsics_by_label[_normalize_camera_label(label)].get('camera_matrix') is not None
            for label in cameras.keys()
        )
    c1 = calibration.get('intrinsic_cam1') or calibration.get('cam1') or {}
    c2 = calibration.get('intrinsic_cam2') or calibration.get('cam2') or {}
    return c1.get('camera_matrix') is not None and c2.get('camera_matrix') is not None


def _has_calibration_extrinsics(calibration):
    if isinstance(calibration, dict):
        cameras = calibration.get('cameras')
        if isinstance(cameras, dict):
            solved = [
                camera for camera in cameras.values()
                if isinstance(camera, dict)
                and camera.get('rvec') is not None
                and camera.get('tvec') is not None
            ]
            return len(solved) >= 2
    return (
        isinstance(calibration, dict)
        and calibration.get('rvec_cam1') is not None
        and calibration.get('tvec_cam1') is not None
        and calibration.get('rvec_cam2') is not None
        and calibration.get('tvec_cam2') is not None
    )


def _has_full_calibration(calibration):
    return _has_calibration_intrinsics(calibration) and _has_calibration_extrinsics(calibration)


def _projection_matrices_from_calibration(calibration, camera_intrinsic_file, RESOLUTION_CAM1, RESOLUTION_CAM2):
    K1, dist1, K2, dist2, used_intr = camera_intrinsic_from_config(
        calibration,
        camera_intrinsic_file,
        RESOLUTION_CAM1,
        RESOLUTION_CAM2,
    )
    r1 = np.asarray(calibration.get('rvec_cam1'), dtype=np.float64).reshape(3, 1)
    t1 = np.asarray(calibration.get('tvec_cam1'), dtype=np.float64).reshape(3, 1)
    r2 = np.asarray(calibration.get('rvec_cam2'), dtype=np.float64).reshape(3, 1)
    t2 = np.asarray(calibration.get('tvec_cam2'), dtype=np.float64).reshape(3, 1)
    R1, _ = cv2.Rodrigues(r1)
    R2, _ = cv2.Rodrigues(r2)
    P1 = K1 @ np.hstack((R1, t1))
    P2 = K2 @ np.hstack((R2, t2))
    return P1, P2, used_intr


def _projection_matrices_from_calibration_multi(calibration, camera_labels):
    cameras = calibration.get('cameras') if isinstance(calibration, dict) else None
    if not isinstance(cameras, dict):
        return None
    cameras_by_label = {_normalize_camera_label(label): camera for label, camera in cameras.items()}
    intrinsics_by_label = {
        _normalize_camera_label(label): intrinsic
        for label, intrinsic in calibration.items()
        if isinstance(intrinsic, dict)
    }
    projection_matrices = []
    used_labels = []
    for label in camera_labels:
        label = _normalize_camera_label(label)
        intrinsic = intrinsics_by_label.get(label)
        camera = cameras_by_label.get(label)
        if not isinstance(intrinsic, dict) or not isinstance(camera, dict):
            continue
        if intrinsic.get('camera_matrix') is None or camera.get('rvec') is None or camera.get('tvec') is None:
            continue
        K = np.asarray(intrinsic.get('camera_matrix'), dtype=np.float64)
        rvec = np.asarray(camera.get('rvec'), dtype=np.float64).reshape(3, 1)
        tvec = np.asarray(camera.get('tvec'), dtype=np.float64).reshape(3, 1)
        R, _ = cv2.Rodrigues(rvec)
        projection_matrices.append(K @ np.hstack((R, tvec)))
        used_labels.append(label)
    if len(projection_matrices) < 2:
        return None
    return used_labels, projection_matrices


def _weighted_triangulation(projection_matrices, x_points, y_points, likelihoods):
    rows = []
    for P, x, y, likelihood in zip(projection_matrices, x_points, y_points, likelihoods):
        weight = float(likelihood)
        rows.append(weight * (x * P[2] - P[0]))
        rows.append(weight * (y * P[2] - P[1]))
    A = np.asarray(rows, dtype=np.float64)
    _, _, vt = np.linalg.svd(A)
    point_h = vt[-1]
    if abs(point_h[3]) < 1e-12:
        return np.array([np.nan, np.nan, np.nan], dtype=np.float64)
    return point_h[:3] / point_h[3]


def _reprojection_error(projection_matrices, point_3d, x_points, y_points):
    point_h = np.asarray([point_3d[0], point_3d[1], point_3d[2], 1.0], dtype=np.float64)
    errors = []
    for P, x, y in zip(projection_matrices, x_points, y_points):
        projected = P @ point_h
        if abs(projected[2]) < 1e-12:
            return np.inf
        px = projected[0] / projected[2]
        py = projected[1] / projected[2]
        errors.append(np.linalg.norm(np.array([x, y], dtype=np.float64) - np.array([px, py], dtype=np.float64)))
    return float(np.mean(errors)) if errors else np.inf


def _triangulate_pose2sim_style(kp1_all, kp2_all, projection_matrices, lifting_config):
    likelihood_threshold = float(lifting_config.get('likelihood_threshold_triangulation', 0.3))
    error_threshold = float(lifting_config.get('reproj_error_threshold_triangulation', 15))
    min_cameras = min(int(lifting_config.get('min_cameras_for_triangulation', 2)), 2)

    num_frames = len(kp1_all)
    final_3d = np.full((num_frames, 26, 3), np.nan, dtype=np.float64)
    for f in range(num_frames):
        for keypoint_idx in range(26):
            x_points = np.array([kp1_all[f, keypoint_idx, 0], kp2_all[f, keypoint_idx, 0]], dtype=np.float64)
            y_points = np.array([kp1_all[f, keypoint_idx, 1], kp2_all[f, keypoint_idx, 1]], dtype=np.float64)
            likelihoods = np.array([kp1_all[f, keypoint_idx, 2], kp2_all[f, keypoint_idx, 2]], dtype=np.float64)
            valid = (
                np.isfinite(x_points)
                & np.isfinite(y_points)
                & np.isfinite(likelihoods)
                & (likelihoods >= likelihood_threshold)
            )
            if int(np.count_nonzero(valid)) < min_cameras:
                continue
            projection_subset = [P for P, keep in zip(projection_matrices, valid) if keep]
            point_3d = _weighted_triangulation(projection_subset, x_points[valid], y_points[valid], likelihoods[valid])
            if not np.isfinite(point_3d).all():
                continue
            error = _reprojection_error(projection_subset, point_3d, x_points[valid], y_points[valid])
            if error <= error_threshold:
                final_3d[f, keypoint_idx] = point_3d
    return final_3d


def _triangulate_pose2sim_style_multi(kps_by_camera, camera_labels, projection_matrices, lifting_config):
    likelihood_threshold = float(lifting_config.get('likelihood_threshold_triangulation', 0.3))
    error_threshold = float(lifting_config.get('reproj_error_threshold_triangulation', 15))
    min_cameras = min(int(lifting_config.get('min_cameras_for_triangulation', 2)), len(camera_labels))

    num_frames = min(len(kps_by_camera[label]) for label in camera_labels)
    final_3d = np.full((num_frames, 26, 3), np.nan, dtype=np.float64)
    for f in range(num_frames):
        for keypoint_idx in range(26):
            x_points = np.asarray([kps_by_camera[label][f, keypoint_idx, 0] for label in camera_labels], dtype=np.float64)
            y_points = np.asarray([kps_by_camera[label][f, keypoint_idx, 1] for label in camera_labels], dtype=np.float64)
            likelihoods = np.asarray([kps_by_camera[label][f, keypoint_idx, 2] for label in camera_labels], dtype=np.float64)
            valid = (
                np.isfinite(x_points)
                & np.isfinite(y_points)
                & np.isfinite(likelihoods)
                & (likelihoods >= likelihood_threshold)
            )
            if int(np.count_nonzero(valid)) < min_cameras:
                continue
            projection_subset = [P for P, keep in zip(projection_matrices, valid) if keep]
            point_3d = _weighted_triangulation(projection_subset, x_points[valid], y_points[valid], likelihoods[valid])
            if not np.isfinite(point_3d).all():
                continue
            error = _reprojection_error(projection_subset, point_3d, x_points[valid], y_points[valid])
            if error <= error_threshold:
                final_3d[f, keypoint_idx] = point_3d
    return final_3d


def triangulate_all_frames(kp1_all, kp2_all, obj_points_3d, actual_calib,
                           camera_intrinsic_file, RESOLUTION_CAM1, RESOLUTION_CAM2,
                           calibration=None, lifting_config=None):
    # Resolve intrinsics and projection matrices.
    lifting_config = lifting_config or {}

    if _has_full_calibration(calibration):
        P1, P2, used_intr = _projection_matrices_from_calibration(
            calibration,
            camera_intrinsic_file,
            RESOLUTION_CAM1,
            RESOLUTION_CAM2,
        )
        logger.info(
            f"3D Lifting: Using Pose2Sim-style weighted triangulation with full calibration. "
            f"Intrinsics from bundle={used_intr}."
        )
        return _triangulate_pose2sim_style(kp1_all, kp2_all, [P1, P2], lifting_config)

    K1, dist1, K2, dist2, used_intr = camera_intrinsic_from_config(calibration, camera_intrinsic_file, RESOLUTION_CAM1, RESOLUTION_CAM2)

    # Resolve extrinsics: use calibration bundle when complete, otherwise self-PnP.
    use_calib_extr = False
    R1 = None
    t1 = None
    R2 = None
    t2 = None

    if _has_full_calibration(calibration):
        # Source A: camera rvec/tvec from the calibration bundle.
        r1 = calibration.get('rvec_cam1'); t1o = calibration.get('tvec_cam1')
        r2 = calibration.get('rvec_cam2'); t2o = calibration.get('tvec_cam2')
        
        r1 = np.asarray(r1, dtype=np.float64).reshape(3, 1)
        t1o = np.asarray(t1o, dtype=np.float64).reshape(3, 1)
        r2 = np.asarray(r2, dtype=np.float64).reshape(3, 1)
        t2o = np.asarray(t2o, dtype=np.float64).reshape(3, 1)
        R1, _ = cv2.Rodrigues(r1)
        R2, _ = cv2.Rodrigues(r2)
        t1, t2 = t1o, t2o
        use_calib_extr = True

    if use_calib_extr:
        if R1 is None or t1 is None or R2 is None or t2 is None:
            raise ValueError("calibration extrinsic incomplete")
        P1 = K1 @ np.hstack((R1, t1))
        P2 = K2 @ np.hstack((R2, t2))
        logger.info(
            f"3D Lifting: Using calibration extrinsics (bundle). "
            f"Intrinsics from bundle={used_intr}."
        )
    else:
        # Self-PnP: estimate camera poses from the rough 3D skeleton and 2D detections.
        obj_pts, img_pts1, img_pts2 = [], [], []
        for f in range(actual_calib):
            for i in range(26):
                if kp1_all[f, i, 2] > 0.1 and kp2_all[f, i, 2] > 0.1:
                    obj_pts.append(obj_points_3d[f, i])
                    img_pts1.append(kp1_all[f, i, :2])
                    img_pts2.append(kp2_all[f, i, :2])
        obj_pts = np.array(obj_pts, dtype=np.float32)
        img_pts1 = np.array(img_pts1, dtype=np.float32)
        img_pts2 = np.array(img_pts2, dtype=np.float32)

        ret1, rvec1, tvec1 = cv2.solvePnP(obj_pts, img_pts1, K1, dist1, flags=cv2.SOLVEPNP_ITERATIVE)
        R1, _ = cv2.Rodrigues(rvec1)
        P1 = K1 @ np.hstack((R1, tvec1))

        ret2, rvec2, tvec2 = cv2.solvePnP(obj_pts, img_pts2, K2, dist2, flags=cv2.SOLVEPNP_ITERATIVE)
        R2, _ = cv2.Rodrigues(rvec2)
        P2 = K2 @ np.hstack((R2, tvec2))
        logger.info(f"3D Lifting: Using key-based self-PnP extrinsics (Cam1={ret1}, Cam2={ret2}). Intrinsics from bundle={used_intr}.")

    # Triangulate every frame.
    num_frames = len(kp1_all)
    final_3d = np.zeros((num_frames, 26, 3))
    
    for f in range(num_frames):
        pts1 = kp1_all[f, :, :2].T # (2, 26)
        pts2 = kp2_all[f, :, :2].T # (2, 26)
        
        # OpenCV returns homogeneous 4D coordinates.
        pts4d = cv2.triangulatePoints(P1, P2, pts1, pts2)
        
        # Convert homogeneous coordinates (X, Y, Z, W) to 3D coordinates.
        pts3d = pts4d[:3, :] / pts4d[3, :] 
        final_3d[f] = pts3d.T
    return final_3d
            
# =====================================================================
# Pipeline entry point.
# =====================================================================
def run_3d_lifting(config, emit_log=None):
    def _log(text, level='info'):
        if callable(emit_log):
            emit_log(text, level)
        else:
            logger.info(text)

    project_dir = config.get('paths').get('project_dir')
    pose3d_dir = os.path.join(project_dir, 'pose-3d')
    os.makedirs(pose3d_dir, exist_ok=True)

    fps = config.get('base').get('fps')
    frame_range = config.get('base').get('frame_range')
    flip_left_right = config.get('lifting', {}).get('flip_left_right', True)
    calibration_bundle = config.get('calibration', None)
    use_bundle = _has_full_calibration(calibration_bundle)

    if use_bundle and isinstance(calibration_bundle.get('cameras'), dict):
        camera_dirs = _pose_json_dirs(project_dir)
        calibrated_labels = sorted(
            {_normalize_camera_label(label) for label in calibration_bundle.get('cameras', {}).keys()},
            key=_camera_sort_key,
        )
        camera_labels = [label for label in calibrated_labels if label in camera_dirs]
        projection_data = _projection_matrices_from_calibration_multi(calibration_bundle, camera_labels)
        if projection_data is None:
            logger.error("3D Lifting: full calibration was provided, but fewer than two calibrated pose cameras were found.")
            return False, None
        camera_labels, projection_matrices = projection_data
        _log(
            "3D Lifting: using calibrated cameras with automatic reprojection-error person selection: "
            + ", ".join(camera_labels)
        )
        kps_by_camera, valid_frames, selection_stats = load_synchronized_kps_multi_auto(
            {label: camera_dirs[label] for label in camera_labels},
            camera_labels,
            projection_matrices,
            config.get('lifting', {}),
            frame_range=frame_range,
        )
        if len(valid_frames) == 0:
            logger.error("3D Lifting: no synchronized pose frames found for calibrated cameras.")
            return False, None
        _log(
            "3D Lifting person selection: "
            f"selected={selection_stats['frames_selected']}/{selection_stats['frames_considered']} "
            f"mean_reprojection_error_px={selection_stats['mean_reprojection_error_px']}"
        )
        _log(
            f"3D Lifting triangulation: frames={len(valid_frames)} "
            f"cameras={len(camera_labels)} use_bundle=True"
        )
        final_3d_frames = _triangulate_pose2sim_style_multi(
            kps_by_camera,
            camera_labels,
            projection_matrices,
            config.get('lifting', {}),
        )
    else:
        cam1_json_dir = os.path.join(project_dir, 'pose', 'cam1_json')
        cam2_json_dir = os.path.join(project_dir, 'pose', 'cam2_json')
        camera_intrinsic_file = config.get('lifting').get('camera_intrinsic_file')
        calib_frames = int(config.get('lifting').get('calib_frames'))
        cam1_person_idx = int(config.get('lifting', {}).get('cam1_person_idx', 0))
        cam2_person_idx = int(config.get('lifting', {}).get('cam2_person_idx', 0))
        resolution_cam1 = config.get('base').get('resolution_cam1')
        resolution_cam2 = config.get('base').get('resolution_cam2')
        height = float(config.get('subject').get('height'))

        _log(
            "3D Lifting: selecting "
            f"cam1 people[{cam1_person_idx}] and cam2 people[{cam2_person_idx}] "
        )
        kp1_list, kp2_list, valid_frames = load_synchronized_kps(
            cam1_json_dir,
            cam2_json_dir,
            cam1_person_idx=cam1_person_idx,
            cam2_person_idx=cam2_person_idx,
            frame_range=frame_range,
        )
        if len(valid_frames) == 0:
            logger.error("3D Lifting: no synchronized pose frames found.")
            return False, None

        if use_bundle:
            _log("3D Lifting: calibration extrinsics detected; skipping self-calibration.")
            obj_points_calib = np.zeros((1, 26, 3), dtype=np.float64)
            actual_calib = 0
        else:
            _log(f"3D Lifting: estimating self-calibration from {calib_frames} frames.")
            obj_points_calib, actual_calib = get_calibration_3d(kp1_list, kp2_list, height, calib_frames)
            _log(f"3D Lifting self-calibration source frames: {valid_frames[:actual_calib]}")

        _log(f"3D Lifting triangulation: frames={len(valid_frames)} use_bundle={use_bundle}")
        final_3d_frames = triangulate_all_frames(
            kp1_list, kp2_list, obj_points_calib, actual_calib,
            camera_intrinsic_file, resolution_cam1, resolution_cam2,
            calibration=calibration_bundle,
            lifting_config=config.get('lifting', {}),
        )

    if use_bundle:
        try:
            old = final_3d_frames
            new_x = old[:, :, 0]
            new_y = old[:, :, 2]
            new_z = -old[:, :, 1]
            final_3d_frames = np.stack([new_x, new_y, new_z], axis=2)
        except Exception as e:
            logger.warning(f"3D Lifting: axis remap failed: {e}")

    if flip_left_right:
        final_3d_frames[:, :, 0] *= -1

    final_3d_frames, interpolation_stats = interpolate_3d_keypoints(
        final_3d_frames,
        config.get('lifting', {}),
    )
    _log(
        "3D Lifting interpolation: "
        f"method={interpolation_stats['method']} "
        f"fill_large_gaps_with={interpolation_stats['fill_mode']} "
        f"filled={interpolation_stats['interpolated_values']} "
        f"skipped_series={interpolation_stats['skipped_series']} "
        f"remaining_nan={interpolation_stats['remaining_nan_values']}"
    )

    trc_filename = os.path.join(pose3d_dir, "keypoints_3d.trc")
    out_unit = 'cm'
    if use_bundle:
        out_unit = calibration_bundle.get('object_points_unit_used') or 'm'
        if out_unit not in ('mm', 'cm', 'm'):
            out_unit = 'm'
    export_to_trc(trc_filename, final_3d_frames, valid_frames, fps, TRC_MARKERS, out_unit=out_unit)
    _log(f"3D Lifting complete. TRC saved: {trc_filename}")
    return True, trc_filename

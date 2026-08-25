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
    normalize_camera_label as _normalize_camera_label,
    person_to_keypoints as _person_to_keypoints,
    pose_json_dirs as _pose_json_dirs,
)
from ..utilities import export_to_trc

logger = logging.getLogger(__name__)

TRC_MARKERS = [
    ("Hip", 19), ("RHip", 12), ("RKnee", 14), ("RAnkle", 16),
    ("RBigToe", 21), ("RSmallToe", 23), ("RHeel", 25),
    ("LHip", 11), ("LKnee", 13), ("LAnkle", 15),
    ("LBigToe", 20), ("LSmallToe", 22), ("LHeel", 24),
    ("Neck", 18), ("Head", 17), ("Nose", 0),
    ("RShoulder", 6), ("RElbow", 8), ("RWrist", 10),
    ("LShoulder", 5), ("LElbow", 7), ("LWrist", 9)
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


def has_full_calibration(calibration) -> bool:
    """Public check: does this bundle carry usable intrinsics *and* extrinsics?"""
    return _has_full_calibration(_normalize_calibration_bundle(calibration))


def _normalize_calibration_bundle(calibration):
    """Return the bundle with a ``cameras`` dict, synthesizing one when needed.

    Older calibration files carry flat ``rvec_cam1`` / ``tvec_cam1`` / ``rvec_cam2`` /
    ``tvec_cam2`` keys instead of a per-label ``cameras`` map. Converting them up front
    means every calibrated scene, old or new, goes through the same multi-camera
    triangulation with automatic person selection.
    """
    if not isinstance(calibration, dict):
        return calibration
    if isinstance(calibration.get('cameras'), dict) and calibration['cameras']:
        return calibration

    cameras = {}
    for index in (1, 2):
        rvec = calibration.get(f'rvec_cam{index}')
        tvec = calibration.get(f'tvec_cam{index}')
        if rvec is None or tvec is None:
            continue
        cameras[f'cam{index}'] = {'rvec': rvec, 'tvec': tvec}
    if len(cameras) < 2:
        return calibration

    normalized = dict(calibration)
    normalized['cameras'] = cameras
    normalized['camera_labels'] = sorted(cameras.keys(), key=_camera_sort_key)
    logger.info("3D Lifting: converted a legacy two-camera calibration into the cameras schema.")
    return normalized


def _has_full_calibration(calibration):
    return _has_calibration_intrinsics(calibration) and _has_calibration_extrinsics(calibration)


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
    calibration_bundle = _normalize_calibration_bundle(config.get('calibration', None))
    if not _has_full_calibration(calibration_bundle):
        # The analysis runner calibrates automatically when a session has no file, so
        # reaching here means that step was skipped or produced an unusable bundle.
        logger.error(
            "3D Lifting: a complete camera calibration (intrinsics + extrinsics) is required."
        )
        _log(
            "3D Lifting: calibration missing or incomplete.",
            "error",
        )
        return False, None

    if not isinstance(calibration_bundle.get('cameras'), dict):
        logger.error("3D Lifting: the calibration bundle has no per-camera extrinsics.")
        _log("3D Lifting: the calibration bundle has no per-camera extrinsics.", "error")
        return False, None

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
        f"cameras={len(camera_labels)}"
    )
    final_3d_frames = _triangulate_pose2sim_style_multi(
        kps_by_camera,
        camera_labels,
        projection_matrices,
        config.get('lifting', {}),
    )

    # Calibration world frames are Z-up (a board lying flat on the floor spans X/Y);
    # OpenSim wants Y-up, so (x, y, z) -> (x, z, -y).
    try:
        remapped = final_3d_frames
        final_3d_frames = np.stack(
            [remapped[:, :, 0], remapped[:, :, 2], -remapped[:, :, 1]], axis=2
        )
    except Exception as exc:
        logger.warning(f"3D Lifting: axis remap failed: {exc}")

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
    out_unit = calibration_bundle.get('object_points_unit_used') or 'm'
    if out_unit not in ('mm', 'cm', 'm'):
        out_unit = 'm'
    export_to_trc(trc_filename, final_3d_frames, valid_frames, fps, TRC_MARKERS, out_unit=out_unit)
    _log(f"3D Lifting complete. TRC saved: {trc_filename}")
    return True, trc_filename

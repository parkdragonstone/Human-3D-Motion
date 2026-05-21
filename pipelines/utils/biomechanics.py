"""생체역학 수치 계산 유틸리티 (좌표 변환, 통계, 사람 추적)."""
import logging
import re

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

logger = logging.getLogger(__name__)


def pad_shape(arr, target_len, fill_value=np.nan):
    """배열을 target_len 길이로 패딩한다."""
    if len(arr) < target_len:
        pad = (target_len - len(arr),) + arr.shape[1:]
        padding = np.full(pad, fill_value)
        return np.concatenate((arr, padding))
    return arr


def trimmed_mean(arr, trimmed_extrema_percent=0.5):
    """양 끝 값을 제거한 평균(trimmed mean)을 반환한다."""
    sorted_arr = np.sort(arr)
    lower_idx = int(len(sorted_arr) * (trimmed_extrema_percent / 2))
    upper_idx = int(len(sorted_arr) * (1 - trimmed_extrema_percent / 2))
    trimmed_arr = sorted_arr[lower_idx:upper_idx]
    if len(trimmed_arr) == 0:
        return np.mean(arr)
    return np.mean(trimmed_arr)


def euclidean_distance(q1, q2):
    """두 점(N차원) 사이의 유클리드 거리를 반환한다."""
    q1 = np.array(q1)
    q2 = np.array(q2)
    dist = q2 - q1
    if np.isnan(dist).all():
        dist = np.empty_like(dist)
        dist[...] = np.inf
    if len(dist.shape) == 1:
        return np.sqrt(np.nansum([d**2 for d in dist]))
    return np.sqrt(np.nansum([d**2 for d in dist], axis=1))


def best_coords_for_measurements(
    Q_coords,
    keypoints_names,
    fastest_frames_to_remove_percent=0.2,
    close_to_zero_speed=0.2,
    large_hip_knee_angles=45,
):
    """
    측정에 적합한 최적 좌표를 선택한다.
    - 가장 빠른 프레임 20% 제거
    - 속도가 0에 가까운 프레임 제거
    - 고관절·슬관절 각도가 45° 미만인 프레임 제거
    """
    df_MidShoulder = pd.DataFrame((Q_coords['RShoulder'].values + Q_coords['LShoulder'].values) / 2)
    df_MidShoulder.columns = ['MidShoulder'] * 3
    Q_coords = pd.concat((Q_coords.reset_index(drop=True), df_MidShoulder), axis=1)

    n_markers_init = len(keypoints_names)
    if 'Hip' not in keypoints_names:
        df_Hip = pd.DataFrame((Q_coords['RHip'].values + Q_coords['LHip'].values) / 2)
        df_Hip.columns = ['Hip'] * 3
        Q_coords = pd.concat((Q_coords.reset_index(drop=True), df_Hip), axis=1)
    n_markers = len(keypoints_names)

    sum_speeds = pd.Series(np.nansum([np.linalg.norm(Q_coords[kpt].diff(), axis=1) for kpt in keypoints_names], axis=0))
    sum_speeds = sum_speeds[sum_speeds > close_to_zero_speed]
    if len(sum_speeds) == 0:
        logger.warning('All frames have speed close to zero. Not restricting the speeds above threshold.')
        Q_coords_low_speeds = Q_coords
    else:
        min_speed_indices = sum_speeds.abs().nsmallest(int(len(sum_speeds) * (1 - fastest_frames_to_remove_percent))).index
        Q_coords_low_speeds = Q_coords.iloc[min_speed_indices].reset_index(drop=True)

    try:
        from analysis.kinematics import mean_angles
        ang_mean = mean_angles(Q_coords_low_speeds, ang_to_consider=['right knee', 'left knee', 'right hip', 'left hip'])
        Q_coords_low_speeds_low_angles = Q_coords_low_speeds[ang_mean < large_hip_knee_angles]
        if len(Q_coords_low_speeds_low_angles) < 50:
            Q_coords_low_speeds_low_angles = Q_coords_low_speeds.iloc[pd.Series(ang_mean).nsmallest(50).index]
    except Exception:
        Q_coords_low_speeds_low_angles = Q_coords_low_speeds
        logger.warning(f"Knee/hip angle markers missing. Not restricting angles below {large_hip_knee_angles}°.")

    if Q_coords_low_speeds_low_angles.empty:
        logger.warning('Selected person might not move or is not well detected. Taking all data.')
        Q_coords_low_speeds_low_angles = Q_coords.copy()

    if n_markers_init < n_markers:
        Q_coords_low_speeds_low_angles = Q_coords_low_speeds_low_angles.iloc[:, :-3]

    return Q_coords_low_speeds_low_angles


def sort_people_sports2d(keyptpre, keypt, scores=None, max_dist=None):
    """프레임 간 사람 인덱스를 연속성 있게 정렬한다 (Hungarian algorithm)."""
    n_prev = len(keyptpre)
    n_curr = len(keypt)

    if n_prev == 0 and n_curr == 0:
        if scores is not None:
            return np.array([]), np.array([]), np.array([])
        return np.array([]), np.array([])

    keyptpre_expanded = keyptpre[:, np.newaxis, :, :]
    keypt_expanded = keypt[np.newaxis, :, :, :]
    diff = keypt_expanded - keyptpre_expanded
    distances_per_keypoint = np.sqrt(np.nansum(diff**2, axis=3))
    dist_matrix = np.nanmean(distances_per_keypoint, axis=2)
    dist_matrix = np.nan_to_num(dist_matrix, nan=1e10, posinf=1e10)

    pre_ids, curr_ids = linear_sum_assignment(dist_matrix)

    valid_associations = []
    if max_dist is not None:
        for pre_id, curr_id in zip(pre_ids, curr_ids):
            if dist_matrix[pre_id, curr_id] <= max_dist:
                valid_associations.append((pre_id, curr_id))
    else:
        valid_associations = list(zip(pre_ids, curr_ids))

    associated_curr_ids = {curr_id for _, curr_id in valid_associations}
    unassociated_curr_ids = [i for i in range(n_curr) if i not in associated_curr_ids]

    n_total = n_prev + len(unassociated_curr_ids)
    sorted_keypoints = np.full((n_total,) + keypt.shape[1:], np.nan)
    if scores is not None:
        sorted_scores = np.full((n_total,) + scores.shape[1:], np.nan)
    else:
        sorted_ids = np.full(n_total, -1)

    for prev_idx, curr_idx in valid_associations:
        sorted_keypoints[prev_idx] = keypt[curr_idx]
        if scores is not None:
            sorted_scores[prev_idx] = scores[curr_idx]
        else:
            sorted_ids[prev_idx] = curr_idx

    for new_idx, curr_idx in enumerate(unassociated_curr_ids):
        sorted_keypoints[n_prev + new_idx] = keypt[curr_idx]
        if scores is not None:
            sorted_scores[n_prev + new_idx] = scores[curr_idx]
        else:
            sorted_ids[n_prev + new_idx] = curr_idx

    keyptpre_padded = pad_shape(keyptpre, n_total, fill_value=np.nan)
    sorted_prev_keypoints = np.where(
        np.isnan(sorted_keypoints) & ~np.isnan(keyptpre_padded),
        keyptpre_padded,
        sorted_keypoints,
    )

    if scores is not None:
        return sorted_prev_keypoints, sorted_keypoints, sorted_scores
    return sorted_prev_keypoints, sorted_keypoints, sorted_ids


def natural_sort_key(s):
    """문자열을 숫자 순서로 자연 정렬하는 키를 반환한다."""
    s = str(s)
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', s)]

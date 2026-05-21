from __future__ import annotations

import logging
import os
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from ..utilities import (
    colors,
    draw_bounding_box,
    draw_keypts,
    draw_skel,
    setup_video,
    sort_people_sports2d,
    thickness,
    transcode_to_h264,
)
from .cameras import normalize_camera_label
from .openpose_io import save_to_openpose


logger = logging.getLogger(__name__)


def process_frame(
    video_path,
    project_dir,
    detect_model,
    pose_solver,
    output_format,
    save_video,
    frame_range,
    max_distance_px,
    keypoint_likelihood_threshold,
    average_likelihood_threshold,
    keypoint_number_threshold,
    progress_log=None,
    progress_step_percent: int = 5,
):
    try:
        cap = cv2.VideoCapture(video_path)
        cap.read()
        if cap.read()[0] is False:
            raise RuntimeError
    except Exception:
        logger.error(f"Error opening video file {video_path}")
        raise NameError(f"{video_path} is not a video file")

    video_path = Path(video_path)
    cam_id = normalize_camera_label(video_path.stem.split("_")[-1])
    pose_dir = Path(os.path.join(project_dir, "pose"))
    json_output_dir = pose_dir / f"{cam_id}_json"
    os.makedirs(json_output_dir, exist_ok=True)
    video_output_path = pose_dir / f"{cam_id}_pose.mp4"

    cap, out, _cam_width, _cam_height, _fps = setup_video(video_path, video_output_path, save_video)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_range = [0, total_frames] if frame_range in ("all", "auto", []) else frame_range
    frame_idx = frame_range[0]
    logger.info(f"Processing frames {frame_range[0]}-{frame_range[1]} of {total_frames} ({video_path.name})...")
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

    start_frame = int(frame_range[0])
    end_frame = int(frame_range[1])
    total_iters = max(1, end_frame - start_frame)
    next_log_threshold = 0

    with tqdm(iterable=range(*frame_range), desc=f"Processing {video_path.name}") as progress_bar:
        while cap.isOpened():
            if frame_idx in range(*frame_range):
                success, frame = cap.read()
                if not success:
                    break
                try:
                    bboxes = detect_model(frame)
                    if bboxes.shape[0] == 0:
                        keypoints = np.full((1, 26, 2), np.nan, dtype=np.float32)
                        scores = np.full((1, 26), np.nan, dtype=np.float32)
                    else:
                        keypoints, scores = pose_solver(frame, bboxes=bboxes)
                    if "prev_keypoints" not in locals():
                        prev_keypoints = keypoints
                    prev_keypoints, keypoints, scores = sort_people_sports2d(
                        prev_keypoints,
                        keypoints,
                        scores=scores,
                        max_dist=max_distance_px,
                    )
                except Exception as exc:
                    logger.exception(f"[Pose Estimation] frame={frame_idx} failed: {exc}")
                    keypoints = np.full((1, 26, 2), fill_value=np.nan, dtype=np.float32)
                    scores = np.full((1, 26), fill_value=np.nan, dtype=np.float32)

                if "openpose" in output_format:
                    json_file_path = os.path.join(json_output_dir, f"{cam_id}_{frame_idx:06d}.json")
                    save_to_openpose(json_file_path, keypoints, scores)

                valid_x, valid_y, valid_scores = [], [], []
                for person_idx in range(len(keypoints)):
                    person_x, person_y = np.where(
                        scores[person_idx][:, np.newaxis] < keypoint_likelihood_threshold,
                        np.nan,
                        keypoints[person_idx],
                    ).T
                    person_scores = np.where(
                        scores[person_idx] < keypoint_likelihood_threshold,
                        np.nan,
                        scores[person_idx],
                    )
                    enough_good_keypoints = len(person_scores[~np.isnan(person_scores)]) >= len(person_scores) * keypoint_number_threshold
                    scores_of_good_keypoints = person_scores[~np.isnan(person_scores)]
                    average_score_is_enough = (
                        np.nanmean(scores_of_good_keypoints) if len(scores_of_good_keypoints) > 0 else 0
                    ) >= average_likelihood_threshold
                    if not enough_good_keypoints or not average_score_is_enough:
                        person_x = np.full_like(person_x, np.nan)
                        person_y = np.full_like(person_y, np.nan)
                        person_scores = np.full_like(person_scores, np.nan)
                    valid_x.append(person_x)
                    valid_y.append(person_y)
                    valid_scores.append(person_scores)

                if save_video:
                    img_show = frame.copy()
                    img_show = draw_bounding_box(img_show, valid_x, valid_y, colors=colors, fontSize=2, thickness=thickness)
                    img_show = draw_keypts(img_show, valid_x, valid_y, valid_scores, cmap_str="RdYlGn")
                    img_show = draw_skel(img_show, valid_x, valid_y)
                    out.write(img_show)

                processed = int(frame_idx - start_frame + 1)
                percent = int(processed * 100 / total_iters)
                if percent >= next_log_threshold or percent == 100:
                    message = f"[Pose Estimation] {cam_id} progress: {processed}/{total_iters} ({percent}%)"
                    try:
                        if callable(progress_log):
                            progress_log(message, "info")
                        else:
                            logger.info(message)
                    except Exception:
                        logger.debug("progress_log failed", exc_info=True)
                    next_log_threshold = min(100, next_log_threshold + max(1, int(progress_step_percent)))

                frame_idx += 1
                progress_bar.update(1)
            if frame_idx >= frame_range[1]:
                break

    cap.release()
    if save_video:
        out.release()
        transcode_to_h264(video_output_path)
        logger.info(f"--> Output video  saved to {video_output_path}")

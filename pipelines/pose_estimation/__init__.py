import logging
import os

from .backend import setup_backend_device
from .cameras import (
    camera_sort_key as _camera_sort_key,
    configured_camera_videos as _configured_camera_videos,
    normalize_camera_label as _normalize_camera_label,
)
from .frame_processing import process_frame
from .models import (
    WrappingDetector as wrapping_detector,
    setup_detector,
    setup_pose_solver,
)
from .openpose_io import save_to_openpose


logger = logging.getLogger(__name__)

keypoints_names = [
    "Hip", "RHip", "RKnee", "RAnkle", "RBigToe", "RSmallToe", "RHeel",
    "LHip", "LKnee", "LAnkle", "LBigToe", "LSmallToe", "LHeel",
    "Neck", "Head", "Nose", "RShoulder", "RElbow", "RWrist",
    "LShoulder", "LElbow", "LWrist",
]
keypoints_ids = [
    19, 12, 14, 16, 21, 23, 25,
    11, 13, 15, 20, 22, 24,
    18, 17, 0, 6, 8, 10,
    5, 7, 9,
]


def run_poseEstimation(config, emit_log=None):
    frame_range = config.get("base").get("frame_range")

    camera_videos = _configured_camera_videos(config)
    if len(camera_videos) < 2:
        raise ValueError(f"pose_estimation_requires_at_least_two_cameras: {len(camera_videos)}")
    project_dir = config.get("paths").get("project_dir")

    pose_config = config.get("pose")
    mode = pose_config.get("mode")
    pose_dir = os.path.join(project_dir, "pose")
    det_score_threshold = pose_config.get("det_score_threshold")
    det_iou = pose_config.get("det_iou")
    det_nms = pose_config.get("det_nms")
    keypoint_likelihood_threshold = pose_config.get("keypoint_likelihood_threshold")
    average_likelihood_threshold = pose_config.get("average_likelihood_threshold")
    keypoint_number_threshold = pose_config.get("keypoint_number_threshold")
    overwrite_pose = pose_config.get("overwrite_pose")
    max_distance_px = pose_config.get("max_distance_px")
    output_format = pose_config.get("output_format")
    save_video = pose_config.get("save_video")

    device = pose_config.get("device")
    backend = pose_config.get("backend")
    requested_backend = backend
    requested_device = device
    backend, device = setup_backend_device(backend=backend, device=device)
    if callable(emit_log):
        emit_log(
            f"Pose backend/device resolved: backend={backend}, device={device} "
            f"(requested backend={requested_backend}, device={requested_device})"
        )

    logger.info("Pose Estimation...")
    json_dirs = [os.path.join(pose_dir, f"{label}_json") for label, _ in camera_videos]
    try:
        if not overwrite_pose and all(os.path.isdir(json_dir) for json_dir in json_dirs):
            counts = [
                len([filename for filename in os.listdir(json_dir) if filename.endswith(".json")])
                for json_dir in json_dirs
            ]
            if all(count > 0 for count in counts):
                logger.info("overwrite_pose=False: 湲곗〈 pose 寃곌낵 ?ъ슜, 3D lifting留?吏꾪뻾?⑸땲??")
                return
    except Exception:
        pass

    if overwrite_pose:
        for json_dir in json_dirs:
            if os.path.isdir(json_dir):
                removed = 0
                for filename in os.listdir(json_dir):
                    if filename.endswith(".json"):
                        try:
                            os.remove(os.path.join(json_dir, filename))
                            removed += 1
                        except Exception as exc:
                            logger.warning(f"湲곗〈 JSON ??젣 ?ㅽ뙣 {filename}: {exc}")
                if removed > 0:
                    logger.info(f"湲곗〈 pose JSON {removed}媛???젣?? {json_dir}")

    detector, detector_cfg = setup_detector(
        device=device,
        det_score_threshold=det_score_threshold,
        det_iou=det_iou,
        det_nms=det_nms,
        mode=mode,
    )
    detect_model = wrapping_detector(detector, detector_cfg)
    pose_solver = setup_pose_solver(mode=mode, backend=backend, device=device)

    for camera_label, video_path in camera_videos:
        logger.info(f"Pose Estimation: processing {camera_label} ({video_path})")
        process_frame(
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
            progress_log=emit_log,
        )

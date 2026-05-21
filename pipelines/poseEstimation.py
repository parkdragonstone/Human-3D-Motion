import os
import json
import re
import logging
import cv2

import numpy as np

from pathlib import Path
from ultralytics import YOLO
from tqdm import tqdm

from .utilities import (
    setup_video,
    transcode_to_h264,
    sort_people_sports2d,
    draw_bounding_box,
    draw_keypts,
    draw_skel,
    colors,
    thickness,
)

logger = logging.getLogger(__name__)

keypoints_names = ["Hip", "RHip", "RKnee", "RAnkle", "RBigToe", "RSmallToe", "RHeel",
                  "LHip", "LKnee", "LAnkle", "LBigToe", "LSmallToe", "LHeel",
                  "Neck", "Head", "Nose", "RShoulder", "RElbow", "RWrist",
                  "LShoulder", "LElbow", "LWrist"]

keypoints_ids = [19, 12, 14, 16, 21, 23, 25,
                 11, 13, 15, 20, 22, 24,
                 18, 17, 0, 6, 8, 10,
                 5, 7, 9]


def _normalize_camera_label(label: str) -> str:
    match = re.search(r"cam0*(\d+)$", str(label).lower())
    return f"cam{int(match.group(1))}" if match else str(label).lower()


def _camera_sort_key(label: str):
    normalized = _normalize_camera_label(label)
    match = re.search(r"cam(\d+)$", normalized)
    return (int(match.group(1)) if match else 9999, str(label).lower())


def _configured_camera_videos(config):
    paths = config.get('paths') or {}
    videos = []
    for key, value in paths.items():
        match = re.fullmatch(r'cam0*(\d+)', str(key).lower())
        if match and value:
            label = _normalize_camera_label(key)
            videos.append((label, value))
    return sorted(videos, key=lambda item: _camera_sort_key(item[0]))

def save_to_openpose(json_file_path, keypoints, scores):
    """Save keypoints/scores to OpenPose-like JSON."""
    nb_detections = len(keypoints)
    detections = []
    for i in range(nb_detections):  # nb of detected people
        keypoints_with_confidence_i = []
        for kp, score in zip(keypoints[i], scores[i]):
            x = float(kp[0]) if not np.isnan(kp[0]) else float('nan')
            y = float(kp[1]) if not np.isnan(kp[1]) else float('nan')
            s = float(score) if not np.isnan(score) else float('nan')
            keypoints_with_confidence_i.extend([x, y, s])

        detections.append({
            "person_id": [-1],
            "pose_keypoints_2d": keypoints_with_confidence_i,
            "face_keypoints_2d": [],
            "hand_left_keypoints_2d": [],
            "hand_right_keypoints_2d": [],
            "pose_keypoints_3d": [],
            "face_keypoints_3d": [],
            "hand_left_keypoints_3d": [],
            "hand_right_keypoints_3d": []
        })

    json_output = {"version": 1.3, "people": detections}

    json_output_dir = os.path.abspath(os.path.join(json_file_path, '..'))
    if not os.path.isdir(json_output_dir):
        os.makedirs(json_output_dir, exist_ok=True)
    with open(json_file_path, 'w', encoding='utf-8') as json_file:
        json.dump(json_output, json_file, ensure_ascii=False)

def setup_backend_device(backend='auto', device='auto'):
    """
    Set up backend & device.

    If device and backend are not specified, they are determined automatically:
      1) CUDA + onnxruntime
      2) ROCm + onnxruntime
      3) MPS/CoreML + onnxruntime
      4) CPU + openvino (fallback)
    """
    if device != 'auto' and backend != 'auto':
        device = device.lower()
        backend = backend.lower()

    if device == 'auto' or backend == 'auto':
        if (device == 'auto' and backend != 'auto') or (device != 'auto' and backend == 'auto'):
            logger.warning("If you set device or backend to 'auto', set the other to 'auto' too. Auto-detecting both.")

        try:
            import torch
            import onnxruntime as ort
            if torch.cuda.is_available() and 'CUDAExecutionProvider' in ort.get_available_providers():
                device = 'cuda'
                backend = 'onnxruntime'
                logger.info("\nValid CUDA installation found: using ONNXRuntime backend with GPU.")
            elif torch.cuda.is_available() and 'ROCMExecutionProvider' in ort.get_available_providers():
                device = 'rocm'
                backend = 'onnxruntime'
                logger.info("\nValid ROCM installation found: using ONNXRuntime backend with GPU.")
            else:
                raise RuntimeError("No CUDA/ROCM provider")
        except Exception:
            try:
                import onnxruntime as ort
                if 'MPSExecutionProvider' in ort.get_available_providers() or 'CoreMLExecutionProvider' in ort.get_available_providers():
                    device = 'mps'
                    backend = 'onnxruntime'
                    logger.info("\nValid MPS installation found: using ONNXRuntime backend with GPU.")
                else:
                    raise RuntimeError("No MPS/CoreML provider")
            except Exception:
                device = 'cpu'
                backend = 'openvino'
                logger.info("\nNo valid CUDA installation found: using OpenVINO backend with CPU.")
    return backend, device
      
def setup_detector(device, det_score_threshold, det_iou, det_nms, mode: str = "normal"):
    """Setup pose detector based on config."""
    model_ckpt = os.path.join(os.path.dirname(__file__), "models", mode, 'yolo_ckpt.pt')
    
    detector = YOLO(model_ckpt)
    detector_cfg = dict(
        device=device,
        conf = det_score_threshold,
        iou = det_iou,
        nms = det_nms,
        verbose = False,
    )
    return detector, detector_cfg

class wrapping_detector:
    def __init__(self, detector, detector_cfg):
        self.model = detector
        self.cfg = detector_cfg
    
    def __call__(self, frame_bgr: np.ndarray) -> np.ndarray:
        res = self.model.predict(frame_bgr, classes=[0], **self.cfg)[0]
        if res.boxes is None or len(res.boxes) == 0:
            return np.zeros((0, 4), dtype=np.float32)
        return res.boxes.xyxy.detach().cpu().numpy().astype(np.float32)

def setup_pose_solver(mode: str, backend: str, device: str):
    """Setup pose solver based on config."""
    pose_onnx = os.path.join(os.path.dirname(__file__), "models", mode, 'rtmpose_end2end.onnx')
    
    try:
        from rtmlib.tools.pose_estimation.rtmpose import RTMPose
    except Exception:
        try:
            from rtmlib.tools.pose_estimation import RTMPose
        except Exception as e:
            raise ImportError("Cannot import RTMPose from rtmlib. Please check your rtmlib version.") from e
    
    pose_solver = RTMPose(
        pose_onnx,
        model_input_size=(192, 256),
        to_openpose=False,
        backend=backend,
        device=device,
    )
    return pose_solver

def process_frame(video_path, 
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
            raise
    except:
        logger.error(f"Error opening video file {video_path}")
        raise NameError(f"{video_path} is not a video file")
    
    video_path = Path(video_path)
    cam_id = _normalize_camera_label(video_path.stem.split('_')[-1])
    pose_dir = Path(os.path.join(project_dir, 'pose'))
    json_output_dir = pose_dir / f'{cam_id}_json'
    os.makedirs(json_output_dir, exist_ok=True)
    # pose 렌더는 camN_pose.mp4 로 저장 (functions.setup_video 가 H.264 우선 시도)
    video_output_path = pose_dir / f'{cam_id}_pose.mp4'
    
    cap, out, cam_width, cam_height, fps = setup_video(video_path, video_output_path, save_video)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_range = [0, total_frames] if frame_range in ('all', 'auto', []) else frame_range
    frame_idx = frame_range[0]
    logger.info(f"Processing frames {frame_range[0]}-{frame_range[1]} of {total_frames} ({video_path.name})...")
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

    start_f = int(frame_range[0])
    end_f   = int(frame_range[1])
    total_iters = max(1, end_f - start_f)
    next_log_threshold = 0
    
    with tqdm(iterable=range(*frame_range), desc=f"Processing {video_path.name}") as pbar:
        while cap.isOpened():
            if frame_idx in range(*frame_range):
                success, frame = cap.read()
                if not success:
                    break
                try:
                    bboxes = detect_model(frame) # Nx4 xyxy
                    if bboxes.shape[0] == 0:
                        keypoints = np.full((1, 26, 2), np.nan, dtype=np.float32)
                        scores = np.full((1, 26), np.nan, dtype=np.float32)
                    else:
                        keypoints, scores = pose_solver(frame, bboxes=bboxes)
                    
                    if 'prev_keypoints' not in locals():
                        prev_keypoints = keypoints
                    prev_keypoints, keypoints, scores = sort_people_sports2d(
                        prev_keypoints, keypoints, scores=scores, max_dist=max_distance_px)
                
                except Exception as e:
                    logger.exception(f"[Pose Estimation] frame={frame_idx} failed: {e}")
                    keypoints = np.full((1, 26, 2), fill_value=np.nan, dtype=np.float32)
                    scores = np.full((1, 26), fill_value=np.nan, dtype=np.float32)
                
                if 'openpose' in output_format:
                    json_file_path = os.path.join(json_output_dir, f'{cam_id}_{frame_idx:06d}.json')
                    save_to_openpose(json_file_path, keypoints, scores)
                
                valid_X, valid_Y, valid_scores = [], [], []
                for person_idx in range(len(keypoints)):
                    person_X, person_Y = np.where(scores[person_idx][:, np.newaxis] < keypoint_likelihood_threshold, np.nan, keypoints[person_idx]).T
                    person_scores = np.where(scores[person_idx] < keypoint_likelihood_threshold, np.nan, scores[person_idx])
                    
                    enough_good_keypoints = len(person_scores[~np.isnan(person_scores)]) >= len(person_scores) * keypoint_number_threshold
                    scores_of_good_keypoints = person_scores[~np.isnan(person_scores)]
                    average_score_of_remaining_keypoints_is_enough = (np.nanmean(scores_of_good_keypoints) if len(scores_of_good_keypoints)>0 else 0) >= average_likelihood_threshold
                    if not enough_good_keypoints or not average_score_of_remaining_keypoints_is_enough:
                        person_X = np.full_like(person_X, np.nan)
                        person_Y = np.full_like(person_Y, np.nan)
                        person_scores = np.full_like(person_scores, np.nan)

                    valid_X.append(person_X)
                    valid_Y.append(person_Y)
                    valid_scores.append(person_scores)
                    
                if save_video:
                    img_show = frame.copy()
                    img_show = draw_bounding_box(img_show, valid_X, valid_Y, colors=colors, fontSize=2, thickness=thickness)
                    img_show = draw_keypts(img_show, valid_X, valid_Y, valid_scores, cmap_str='RdYlGn')
                    img_show = draw_skel(img_show, valid_X, valid_Y)
                    out.write(img_show)
                
                # tqdm 진행상황을 UI 로그로도 남긴다(너무 잦은 로그 방지 위해 percent step 사용).
                processed = int(frame_idx - start_f + 1)
                percent = int(processed * 100 / total_iters)
                if percent >= next_log_threshold or percent == 100:
                    msg = f"[Pose Estimation] {cam_id} progress: {processed}/{total_iters} ({percent}%)"
                    try:
                        if callable(progress_log):
                            progress_log(msg, "info")
                        else:
                            logger.info(msg)
                    except Exception:
                        # 중간 progress 로그 실패는 학습/분석 실패로 이어지지 않게 한다.
                        logger.debug("progress_log failed", exc_info=True)
                    next_log_threshold = min(100, next_log_threshold + max(1, int(progress_step_percent)))

                frame_idx += 1
                pbar.update(1)
            
            if frame_idx >= frame_range[1]:
                break
            
    cap.release()
    if save_video:
        out.release()
        transcode_to_h264(video_output_path)
        logger.info(f"--> Output video  saved to {video_output_path}")
    

def run_poseEstimation(config, emit_log=None):
    frame_range = config.get('base').get('frame_range')
    
    camera_videos = _configured_camera_videos(config)
    if len(camera_videos) < 2:
        raise ValueError(f"pose_estimation_requires_at_least_two_cameras: {len(camera_videos)}")
    project_dir = config.get('paths').get('project_dir')
    
    mode = config.get('pose').get('mode')
    pose_dir = os.path.join(project_dir, 'pose')
    
    det_score_threshold = config.get('pose').get('det_score_threshold')
    det_iou = config.get('pose').get('det_iou')
    det_nms = config.get('pose').get('det_nms')
    keypoint_likelihood_threshold = config.get('pose').get('keypoint_likelihood_threshold')
    average_likelihood_threshold = config.get('pose').get('average_likelihood_threshold')
    keypoint_number_threshold = config.get('pose').get('keypoint_number_threshold')
    overwrite_pose = config.get('pose').get('overwrite_pose')
    max_distance_px = config.get('pose').get('max_distance_px')
    output_format = config.get('pose').get('output_format')
    save_video = config.get('pose').get('save_video')
    
    
    device = config.get('pose').get('device')
    backend = config.get('pose').get('backend')
    requested_backend = backend
    requested_device = device
    backend, device = setup_backend_device(backend=backend, device=device)
    if callable(emit_log):
        emit_log(
            f"Pose backend/device resolved: backend={backend}, device={device} "
            f"(requested backend={requested_backend}, device={requested_device})"
        )
    
    logger.info('Pose Estimation...')
    json_dirs = [os.path.join(pose_dir, f'{label}_json') for label, _ in camera_videos]
    try:
        if not overwrite_pose and all(os.path.isdir(json_dir) for json_dir in json_dirs):
            counts = [
                len([f for f in os.listdir(json_dir) if f.endswith('.json')])
                for json_dir in json_dirs
            ]
            if all(count > 0 for count in counts):
                logger.info("overwrite_pose=False: 기존 pose 결과 사용, 3D lifting만 진행합니다.")
                return 
    except Exception:
        pass

    # overwrite_pose=True 이면 기존 json 폴더 내 .json 파일을 삭제한 뒤 새 결과 저장
    if overwrite_pose:
        for json_dir in json_dirs:
            if os.path.isdir(json_dir):
                removed = 0
                for f in os.listdir(json_dir):
                    if f.endswith('.json'):
                        try:
                            os.remove(os.path.join(json_dir, f))
                            removed += 1
                        except Exception as e:
                            logger.warning(f"기존 JSON 삭제 실패 {f}: {e}")
                if removed > 0:
                    logger.info(f"기존 pose JSON {removed}개 삭제됨: {json_dir}")

    detector, detector_cfg = setup_detector(device=device, 
                                            det_score_threshold=det_score_threshold, 
                                            det_iou=det_iou, det_nms=det_nms, mode=mode)
    detect_model = wrapping_detector(detector, detector_cfg)
    pose_solver = setup_pose_solver(mode=mode, backend=backend, device=device)
    
    for camera_label, video_path in camera_videos:
        logger.info(f"Pose Estimation: processing {camera_label} ({video_path})")
        process_frame(video_path,
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




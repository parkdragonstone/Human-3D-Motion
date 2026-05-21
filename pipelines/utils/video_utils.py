"""비디오 처리 유틸리티 (OpenCV VideoWriter, 스켈레톤 렌더링)."""
import itertools as it
import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)

_FFMPEG_FALLBACK_PATHS = (
    '/opt/homebrew/bin/ffmpeg',   # macOS Apple Silicon (Homebrew)
    '/usr/local/bin/ffmpeg',      # macOS Intel (Homebrew)
    '/usr/bin/ffmpeg',            # Linux 시스템
    r'C:\ffmpeg\bin\ffmpeg.exe',  # Windows 일반 설치 경로
)

_FFPROBE_FALLBACK_PATHS = (
    '/opt/homebrew/bin/ffprobe',
    '/usr/local/bin/ffprobe',
    '/usr/bin/ffprobe',
    r'C:\ffmpeg\bin\ffprobe.exe',
)


def _ffmpeg_executable() -> str | None:
    """PATH와 일반 설치 경로에서 ffmpeg 실행 파일을 찾는다."""
    exe = shutil.which('ffmpeg') or shutil.which('ffmpeg.exe')
    if exe:
        return exe
    for p in _FFMPEG_FALLBACK_PATHS:
        if os.path.isfile(p):
            return p
    return None


def _ffprobe_executable() -> str | None:
    exe = shutil.which('ffprobe') or shutil.which('ffprobe.exe')
    if exe:
        return exe
    for p in _FFPROBE_FALLBACK_PATHS:
        if os.path.isfile(p):
            return p
    return None


def _is_h264_mp4(path: str) -> bool:
    """ffprobe로 파일이 이미 H.264 코덱인지 확인한다. 확인 불가 시 False 반환."""
    ffprobe = _ffprobe_executable()
    if not ffprobe:
        return False
    try:
        import json as _json
        r = subprocess.run(
            [ffprobe, '-v', 'quiet', '-print_format', 'json',
             '-show_streams', str(path)],
            capture_output=True, text=True, timeout=10,
        )
        if r.returncode != 0:
            return False
        data = _json.loads(r.stdout or '{}')
        for s in data.get('streams', []):
            if s.get('codec_type') == 'video':
                return s.get('codec_name', '') in ('h264', 'avc1', 'avc')
    except Exception:
        pass
    return False


HALPE26_SKELETON_PAIRS = [
    (0, 17), (18, 17), (18, 19),
    (18, 5), (18, 6),
    (5, 7), (7, 9),
    (6, 8), (8, 10),
    (19, 11), (19, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
    (15, 20), (20, 22), (15, 24),
    (16, 21), (21, 23), (16, 25),
]

colors = [
    (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255),
    (0, 0, 0), (255, 255, 255), (125, 0, 0), (0, 125, 0), (0, 0, 125),
    (125, 125, 0), (125, 0, 125), (0, 125, 125), (255, 125, 125),
    (125, 255, 125), (125, 125, 255), (255, 255, 125), (255, 125, 255),
    (125, 255, 255), (125, 125, 125), (255, 0, 125), (255, 125, 0),
    (0, 125, 255), (0, 255, 125), (125, 0, 255), (125, 255, 0), (0, 255, 0),
]

KEYPOINT_COLORS = [
    (255, 230, 240), (255, 180, 200), (200, 255, 220),
    (255, 140, 160), (140, 255, 180), (255, 200, 120), (200, 120, 255),
    (120, 255, 255), (255, 100, 200), (100, 200, 255), (220, 255, 100),
    (255, 160, 80), (160, 80, 255), (80, 255, 200), (240, 180, 255),
    (180, 255, 140), (255, 220, 160), (220, 160, 255), (160, 255, 255),
    (255, 130, 210), (130, 210, 255), (210, 255, 130), (255, 190, 150),
    (190, 150, 255), (150, 255, 190), (255, 170, 230), (170, 230, 255),
]

thickness = 2


def _open_mp4_writer_browser_safe(vid_output_path, fps, frame_size):
    """브라우저 <video> 호환 우선으로 OpenCV VideoWriter fourcc를 선택한다."""
    path = Path(vid_output_path)
    path_str = str(path.absolute())
    w, h = frame_size
    ext = (path.suffix or '').lower()
    if ext == '.webm':
        candidates = ('VP80', 'VP90', 'mp4v')
    else:
        candidates = ('avc1', 'H264', 'X264', 'mp4v')
    out = None
    for i, tag in enumerate(candidates):
        if i > 0 and path.is_file():
            try:
                path.unlink()
            except OSError:
                pass
        fourcc = cv2.VideoWriter_fourcc(*tag)
        out = cv2.VideoWriter(path_str, fourcc, fps, (w, h))
        if out.isOpened():
            if tag == 'mp4v':
                logger.warning("VideoWriter: 'mp4v' fallback이 사용되었습니다.")
            else:
                logger.info("VideoWriter: fourcc '%s' (웹 재생 호환 우선).", tag)
            return out
        out.release()
    raise RuntimeError(f"VideoWriter를 열 수 없습니다: {path_str}")


def transcode_to_h264(path):
    """FFmpeg로 영상을 H.264 + yuv420p로 재인코딩하여 브라우저 호환성을 보장한다.

    macOS Chrome에서 mp4v(MPEG-4 Part 2) 코덱이 지원되지 않는 문제를 해결한다.
    FFmpeg이 없으면 경고 로그만 남기고 원본 파일을 유지한다.
    """
    path = Path(path)
    if not path.is_file():
        # logger.warning("transcode_to_h264: 파일 없음 %s", path)
        return

    # 이미 H.264면 재인코딩 불필요 (멱등 보장)
    if _is_h264_mp4(str(path)):
        # logger.info("transcode_to_h264: 이미 H.264 — 건너뜀 %s", path)
        return

    ffmpeg = _ffmpeg_executable()
    if not ffmpeg:
        # logger.warning("transcode_to_h264: ffmpeg를 찾을 수 없습니다. mp4v 코덱으로 Chrome 재생에 실패할 수 있습니다.")
        return

    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".mp4", dir=path.parent)
    os.close(tmp_fd)
    try:
        result = subprocess.run(
            [
                ffmpeg, "-y",
                "-i", str(path),
                "-c:v", "libx264",
                "-preset", "fast",
                "-crf", "23",
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                "-an",
                tmp_path,
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            # logger.warning("transcode_to_h264: ffmpeg 실패 (returncode=%d)\n%s", result.returncode, result.stderr[-1000:])
            return
        os.replace(tmp_path, str(path))
        # logger.info("transcode_to_h264: H.264 재인코딩 완료 → %s", path)
    # except Exception as e:
        # logger.warning("transcode_to_h264: 예외 발생 — %s", e)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


def setup_video(video_file_path, vid_output_path, save_vid):
    """비디오 캡처·라이터를 설정하고 반환한다."""
    if video_file_path.name == video_file_path.stem:
        raise ValueError("Please set video_input to 'webcam' or to a video file (with extension) in Config.toml")
    try:
        cap = cv2.VideoCapture(str(video_file_path.absolute()))
        if not cap.isOpened():
            raise RuntimeError()
    except Exception:
        raise NameError(f"{video_file_path} is not a video. Check video_dir and video_input in your Config.toml file.")

    cam_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cam_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = None
    fps = round(cap.get(cv2.CAP_PROP_FPS))
    if fps == 0:
        fps = 30
    if save_vid:
        out = _open_mp4_writer_browser_safe(vid_output_path, fps, (cam_width, cam_height))

    return cap, out, cam_width, cam_height, fps


def draw_bounding_box(img, X, Y, colors=colors, fontSize=0.3, thickness=1):
    """바운딩 박스와 사람 ID를 그린다."""
    color_cycle = it.cycle(colors)
    for i, (x, y) in enumerate(zip(X, Y)):
        color = next(color_cycle)
        if not np.isnan(x).all():
            x_min, y_min = np.nanmin(x).astype(int), np.nanmin(y).astype(int)
            x_max, y_max = np.nanmax(x).astype(int), np.nanmax(y).astype(int)
            x_min = max(x_min, 0); x_max = min(x_max, img.shape[1])
            y_min = max(y_min, 0); y_max = min(y_max, img.shape[0])
            cv2.rectangle(img, (x_min - 25, y_min - 25), (x_max + 25, y_max + 25), color, thickness)
            cv2.putText(img, str(i), (x_min - 30, y_min - 30), cv2.FONT_HERSHEY_SIMPLEX, fontSize, color, 2, cv2.LINE_AA)
    return img


def draw_skel(img, X, Y, skeleton_pairs=HALPE26_SKELETON_PAIRS):
    """각 사람의 스켈레톤을 그린다."""
    left_keypoints = {1, 3, 5, 7, 9, 11, 13, 15, 20, 22, 24}
    right_keypoints = {2, 4, 6, 8, 10, 12, 14, 16, 21, 23, 25}

    for (x, y) in zip(X, Y):
        if not np.isnan(x).all():
            for id1, id2 in skeleton_pairs:
                if not (np.isnan(x[id1]) or np.isnan(y[id1]) or np.isnan(x[id2]) or np.isnan(y[id2])):
                    if (id1 in right_keypoints or id2 in right_keypoints) and not (id1 in left_keypoints or id2 in left_keypoints):
                        c = (0, 140, 255)
                    elif (id1 in left_keypoints or id2 in left_keypoints) and not (id1 in right_keypoints or id2 in right_keypoints):
                        c = (0, 255, 100)
                    else:
                        c = (255, 100, 255)
                    cv2.line(img, (int(x[id1]), int(y[id1])), (int(x[id2]), int(y[id2])), c, thickness)
    return img


def draw_keypts(img, X, Y, scores, cmap_str='RdYlGn', use_keypoint_colors=True):
    """각 사람의 키포인트를 그린다."""
    n_kp = len(KEYPOINT_COLORS)
    for (x, y, s) in zip(X, Y, scores):
        for i in range(len(x)):
            if np.isnan(x[i]) or np.isnan(y[i]):
                continue
            if use_keypoint_colors:
                c = KEYPOINT_COLORS[i % n_kp]
            else:
                sc = 0.0 if np.isnan(s[i]) else max(0, min(0.99, float(s[i])))
                c_rgb = plt.get_cmap(cmap_str)(sc)[:-1]
                c = tuple(int(c_rgb[k] * 255) for k in (2, 1, 0))
            cv2.circle(img, (int(x[i]), int(y[i])), thickness + 4, c, -1)
    return img

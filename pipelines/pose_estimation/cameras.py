import re


def normalize_camera_label(label: str) -> str:
    match = re.search(r"cam0*(\d+)$", str(label).lower())
    return f"cam{int(match.group(1))}" if match else str(label).lower()


def camera_sort_key(label: str):
    normalized = normalize_camera_label(label)
    match = re.search(r"cam(\d+)$", normalized)
    return (int(match.group(1)) if match else 9999, str(label).lower())


def configured_camera_videos(config):
    paths = config.get("paths") or {}
    videos = []
    for key, value in paths.items():
        match = re.fullmatch(r"cam0*(\d+)", str(key).lower())
        if match and value:
            label = normalize_camera_label(key)
            videos.append((label, value))
    return sorted(videos, key=lambda item: camera_sort_key(item[0]))

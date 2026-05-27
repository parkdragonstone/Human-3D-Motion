from __future__ import annotations

from flask import url_for


def camera_to_dict(camera):
    return {
        "camera_id": camera.camera_id,
        "label": camera.label,
        "connected": camera.connected,
        "recording": camera.recording,
        "live_view_url": camera.live_view_url,
        "live_view_frame_rate": camera.live_view_frame_rate,
        "last_error": camera.last_error,
    }


def session_to_dict(session):
    return {
        "session_id": session.session_id,
        "subject": {
            "name": session.subject.name,
            "height_cm": session.subject.height_cm,
            "weight_kg": session.subject.weight_kg,
            "hand": session.subject.hand,
        },
        "timestamp": session.timestamp,
        "display_timestamp": display_timestamp(session.timestamp),
        "session_path": session.session_path,
        "status": session.status,
        "videos": [video_to_dict(video) for video in session.videos],
        "created_at": session.created_at.isoformat(timespec="seconds") if session.created_at else None,
        "updated_at": session.updated_at.isoformat(timespec="seconds") if session.updated_at else None,
    }


def calibration_to_dict(calibration, status: str):
    return {
        "calibration_id": calibration.calibration_id,
        "mode": calibration.mode,
        "project_name": calibration.project_name,
        "timestamp": calibration.timestamp,
        "status": status,
        "output_dir": str(calibration.output_dir),
        "record_camera_ids": calibration.record_camera_ids,
        "save_camera_labels": sorted(calibration.save_camera_labels),
    }


def calibration_record_to_dict(record):
    return {
        "mode": record.mode,
        "project_name": record.project_name,
        "folder_name": record.folder_name,
        "display_name": f"CALIB {record.project_name}",
        "output_dir": str(record.output_dir),
        "updated_at": record.updated_at.isoformat(timespec="seconds"),
        "display_updated_at": record.updated_at.strftime("%Y-%m-%d %H:%M:%S"),
        "videos": [calibration_video_to_dict(video) for video in record.videos],
    }


def calibration_video_to_dict(video):
    return {
        "camera_label": video.camera_label,
        "path": video.path,
        "filename": video.filename,
        "size_bytes": video.size_bytes,
        "size_label": format_size(video.size_bytes),
    }


def phone_draft_to_dict(draft):
    return {
        "token": draft.token,
        "slots": [
            {
                "camera_id": slot.camera_id,
                "camera_label": slot.camera_label,
                "join_url": slot.join_url,
                "qr_data_url": slot.qr_data_url,
            }
            for slot in draft.slots
        ],
    }


def video_to_dict(video):
    return {
        "camera_id": video.camera_id,
        "camera_label": video.camera_label,
        "path": video.path,
        "video_url": url_for("api_analysis_video", path=video.path),
        "pose_video_path": video.pose_video_path,
        "pose_video_url": (
            url_for("api_analysis_video", path=video.pose_video_path, v=video.pose_video_mtime)
            if video.pose_video_path else None
        ),
        "filename": video.filename,
        "fps": video.fps,
        "frame_count": video.frame_count,
        "size_bytes": video.size_bytes,
        "size_label": format_size(video.size_bytes),
    }


def display_timestamp(timestamp: str) -> str:
    if len(timestamp) == 15 and timestamp[8] == "_":
        return f"{timestamp[:4]}-{timestamp[4:6]}-{timestamp[6:8]} {timestamp[9:]}"
    return timestamp


def format_size(size_bytes: int) -> str:
    if size_bytes >= 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    if size_bytes >= 1024:
        return f"{size_bytes / 1024:.1f} KB"
    return f"{size_bytes} B"

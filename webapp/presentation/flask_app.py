from __future__ import annotations

import base64
import os
import threading
import uuid
from pathlib import Path

from flask import Flask, jsonify, redirect, render_template, request, send_file, url_for
from flask_socketio import SocketIO

from webapp.application.analysis_pipeline_service import AnalysisPipelineService
from webapp.application.calibration_service import CalibrationService
from webapp.application.capture_service import CaptureService
from webapp.application.phone_capture_service import PhoneCaptureService
from webapp.domain.entities import SubjectInfo
from webapp.infrastructure.camera.mode_camera_controller import ModeCameraController
from webapp.infrastructure.camera.phone_camera_controller import PhoneCameraController
from webapp.infrastructure.camera.url_camera_controller import UrlCameraController
from webapp.infrastructure.persistence.file_session_catalog import FileSessionCatalog
from webapp.infrastructure.persistence.file_settings import JsonSettingsRepository


def create_app():
    app = Flask(
        __name__,
        template_folder=str(Path(__file__).resolve().parent / "templates"),
        static_folder=str(Path(__file__).resolve().parent / "static"),
    )
    app.config["SECRET_KEY"] = os.environ.get("BASEBALL_MOTION_SECRET", "dev")
    app.config["TEMPLATES_AUTO_RELOAD"] = True
    app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0
    socketio = SocketIO(
        app,
        cors_allowed_origins="*",
        async_mode="threading",
        logger=False,
        engineio_logger=False,
    )

    settings = JsonSettingsRepository(os.environ.get("BASEBALL_MOTION_SETTINGS", "webapp_data/settings.json"))
    sessions = FileSessionCatalog()
    camera_controller = _camera_controller_from_env(settings)
    capture_service = CaptureService(camera_controller, sessions, settings)
    analysis_service = AnalysisPipelineService()
    phone_service = PhoneCaptureService(settings)
    calibration_service = CalibrationService(settings)
    analysis_jobs: dict[str, dict] = {}
    capture_service.configure_cameras(
        settings.get_camera_count(),
        settings.get_ccb_url(),
        settings.get_live_view_frame_rate(),
    )
    capture_service.configure_capture_mode(
        settings.get_capture_mode(),
        settings.get_phone_camera_count(),
        settings.get_phone_frame_rate(),
        "landscape",
    )
    phone_socket_labels: dict[str, str] = {}

    @app.context_processor
    def static_asset_helpers():
        return {"static_asset": _static_asset_url}

    @app.get("/")
    def index():
        return redirect(url_for("capture_page"))

    @app.get("/app-logo")
    def app_logo():
        return send_file(_app_logo_path(), conditional=True)

    @app.get("/capture")
    def capture_page():
        phone_draft = phone_service.current_or_create_draft(_base_url())
        sessions = capture_service.list_sessions()
        return render_template(
            "capture.html",
            storage_root=capture_service.get_storage_root(),
            cameras=capture_service.list_cameras(),
            camera_settings=capture_service.camera_settings(),
            phone_draft=phone_draft,
            active_capture=capture_service.active_capture(),
            sessions=sessions,
            recent_sessions=[_session_to_dict(session) for session in sessions[:6]],
        )

    @app.post("/settings/storage-root")
    def set_storage_root_form():
        storage_root = request.form.get("storage_root", "").strip()
        if storage_root:
            capture_service.set_storage_root(storage_root)
        return redirect(url_for("capture_page"))

    @app.post("/settings/cameras")
    def set_cameras_form():
        capture_service.configure_cameras(
            int(request.form.get("camera_count", "1")),
            request.form.get("ccb_url", "").strip(),
            request.form.get("live_view_frame_rate", "low"),
        )
        capture_service.configure_capture_mode(
            request.form.get("capture_mode", capture_service.camera_settings()["capture_mode"]),
            int(request.form.get("phone_camera_count", capture_service.camera_settings()["phone_camera_count"])),
            int(request.form.get("phone_frame_rate", capture_service.camera_settings()["phone_frame_rate"])),
            "landscape",
        )
        _emit_camera_status(socketio, capture_service)
        return redirect(url_for("capture_page"))

    @app.post("/capture/start")
    def start_capture_form():
        payload = _capture_payload_from_form()
        capture_service.start_capture(payload["subject"], payload["camera_ids"])
        _emit_camera_status(socketio, capture_service)
        return redirect(url_for("capture_page"))

    @app.post("/capture/stop")
    def stop_capture_form():
        capture_service.stop_capture()
        _emit_camera_status(socketio, capture_service)
        return redirect(url_for("capture_page"))

    @app.get("/calibration")
    def calibration_page():
        phone_draft = phone_service.current_or_create_draft(_base_url())
        return render_template(
            "calibration.html",
            storage_root=capture_service.get_storage_root(),
            cameras=capture_service.list_cameras(),
            camera_settings=capture_service.camera_settings(),
            phone_draft=phone_draft,
            active_calibration=calibration_service.active(),
            calibrations=[_calibration_record_to_dict(item) for item in calibration_service.list_calibrations()],
        )

    @app.get("/analysis")
    def analysis_page():
        analysis_root = request.args.get("root") or capture_service.get_storage_root()
        if request.args.get("root"):
            analysis_root = capture_service.set_storage_root(analysis_root)
        return render_template(
            "analysis.html",
            storage_root=capture_service.get_storage_root(),
            initial_root=analysis_root,
            initial_session_id=request.args.get("session_id", ""),
        )

    @app.get("/api/analysis/config")
    def api_analysis_config():
        return jsonify(analysis_service.default_config())

    @app.get("/api/analysis/sessions")
    def api_analysis_sessions():
        root = str(request.args.get("root") or capture_service.get_storage_root()).strip()
        if root:
            root = capture_service.set_storage_root(root)
        return jsonify([_session_to_dict(session) for session in sessions.list_sessions(root)])

    @app.post("/api/analysis/session-root/select")
    def api_select_analysis_session_root():
        data = request.get_json(silent=True) or {}
        requested_root = str(data.get("root") or "").strip()
        if data.get("manual") and requested_root:
            return jsonify({"root": capture_service.set_storage_root(requested_root), "cancelled": False})
        current_root = capture_service.get_storage_root()
        try:
            selected_path = _select_directory(current_root)
        except RuntimeError as exc:
            return jsonify({
                "root": current_root,
                "cancelled": True,
                "manual_required": True,
                "error": str(exc),
            })
        if selected_path is None:
            return jsonify({"root": current_root, "cancelled": True})
        return jsonify({"root": capture_service.set_storage_root(selected_path), "cancelled": False})

    @app.get("/api/analysis/video")
    def api_analysis_video():
        path = Path(str(request.args.get("path") or "")).expanduser().resolve()
        if not path.is_file() or path.suffix.lower() not in {".mp4", ".mov", ".webm", ".avi"}:
            return jsonify({"error": "video_not_found"}), 404
        return send_file(path, conditional=True)

    @app.post("/api/analysis/calibration/upload")
    def api_upload_analysis_calibration():
        try:
            session_path = Path(str(request.form.get("session_path") or "")).expanduser().resolve()
            if not session_path.is_dir():
                raise ValueError("session_path_not_found")
            session = _session_by_path(sessions.list_sessions(str(session_path.parent)), session_path)
            if session is None:
                raise ValueError("session_not_found")
            uploaded = request.files.get("calibration_file")
            if uploaded is None or not uploaded.filename:
                raise ValueError("calibration_file_required")
            if Path(uploaded.filename).suffix.lower() != ".json":
                raise ValueError("calibration_file_must_be_json")
            import json

            payload = json.load(uploaded.stream)
            if not isinstance(payload, dict):
                raise ValueError("calibration_json_must_be_object")
            mode = str(payload.get("mode") or ("EXTR" if payload.get("extrinsic") else "INTR")).upper()
            target = Path(session.session_path) / "analysis_calibration.json"
            target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            return jsonify({"filename": uploaded.filename, "mode": mode, "path": str(target)})
        except Exception as exc:
            return jsonify({"error": str(exc)}), 400

    @app.post("/api/analysis/run")
    def api_run_analysis():
        data = request.get_json(silent=True) or {}
        try:
            session_path = Path(str(data.get("session_path") or "")).expanduser().resolve()
            if not session_path.is_dir():
                raise ValueError("session_path_not_found")
            session = _session_by_path(sessions.list_sessions(str(session_path.parent)), session_path)
            if session is None:
                raise ValueError("session_not_found")
            job_id = uuid.uuid4().hex
            analysis_jobs[job_id] = {"status": "queued", "logs": [], "error": None}
            thread = threading.Thread(
                target=_run_analysis_job,
                args=(job_id, session, data.get("config") or {}, analysis_jobs, analysis_service),
                daemon=True,
            )
            thread.start()
            return jsonify({"job_id": job_id, **analysis_jobs[job_id]})
        except Exception as exc:
            return jsonify({"error": str(exc)}), 400

    @app.get("/api/analysis/jobs/<job_id>")
    def api_analysis_job(job_id):
        job = analysis_jobs.get(job_id)
        if job is None:
            return jsonify({"error": "job_not_found"}), 404
        return jsonify({"job_id": job_id, **job})

    @app.get("/api/analysis/pose3d/files")
    def api_analysis_pose3d_files():
        try:
            session = _analysis_session_from_request(sessions)
            pose3d_dir = Path(session.session_path) / "pose-3d"
            files = sorted(path.name for path in pose3d_dir.glob("*.trc") if path.is_file()) if pose3d_dir.is_dir() else []
            return jsonify({"files": files})
        except Exception as exc:
            return jsonify({"error": str(exc), "files": []}), 400

    @app.get("/api/analysis/pose3d/data")
    def api_analysis_pose3d_data():
        try:
            session = _analysis_session_from_request(sessions)
            filename = str(request.args.get("file") or "").strip()
            if not filename or "/" in filename or "\\" in filename:
                raise ValueError("invalid_trc_file")
            trc_path = Path(session.session_path) / "pose-3d" / filename
            return jsonify(_pose3d_data_from_trc(trc_path))
        except Exception as exc:
            return jsonify({"error": str(exc)}), 400

    @app.get("/api/analysis/keypoints/frame")
    def api_analysis_keypoints_frame():
        try:
            session = _analysis_session_from_request(sessions)
            camera_label = str(request.args.get("camera_label") or "")
            frame = int(request.args.get("frame") or 0)
            return jsonify(_keypoint_frame_from_json(session.session_path, camera_label, frame))
        except Exception as exc:
            return jsonify({"error": str(exc)}), 400

    @app.post("/api/analysis/keypoints/frame")
    def api_save_analysis_keypoints_frame():
        try:
            data = request.get_json(silent=True) or {}
            session_path = Path(str(data.get("session_path") or "")).expanduser().resolve()
            if not session_path.is_dir():
                raise ValueError("session_path_not_found")
            session = _session_by_path(sessions.list_sessions(str(session_path.parent)), session_path)
            if session is None:
                raise ValueError("session_not_found")
            camera_label = str(data.get("camera_label") or "")
            frame = int(data.get("frame") or 0)
            keypoints = data.get("keypoints")
            if keypoints is None:
                keypoints = data.get("people")
            if not isinstance(keypoints, list):
                raise ValueError("people_keypoints_required")
            _save_keypoint_frame_to_json(session.session_path, camera_label, frame, keypoints)
            return jsonify({"status": "saved"})
        except Exception as exc:
            return jsonify({"error": str(exc)}), 400

    @app.post("/api/analysis/keypoints/render-video")
    def api_render_analysis_keypoint_video():
        try:
            data = request.get_json(silent=True) or {}
            session_path = Path(str(data.get("session_path") or "")).expanduser().resolve()
            if not session_path.is_dir():
                raise ValueError("session_path_not_found")
            session = _session_by_path(sessions.list_sessions(str(session_path.parent)), session_path)
            if session is None:
                raise ValueError("session_not_found")
            camera_label = str(data.get("camera_label") or "")
            pose_video_path = _render_pose_video_from_keypoints(session, camera_label)
            return jsonify({"status": "rendered", "pose_video_path": str(pose_video_path)})
        except Exception as exc:
            return jsonify({"error": str(exc)}), 400

    @app.get("/api/analysis/kinematics")
    def api_analysis_kinematics():
        try:
            session = _analysis_session_from_request(sessions)
            csv_path = _latest_kinematics_csv_file(Path(session.session_path))
            if csv_path is None:
                return jsonify({"available": False, "signals": [], "unit": "deg"})
            columns = _read_csv_columns(csv_path)
            signals = [signal for signal in _kinematics_signals() if signal["key"] in columns]
            return jsonify({
                "available": True,
                "file": csv_path.name,
                "unit": "deg",
                "signals": signals,
                "events": _kinematics_event_markers(csv_path, columns),
            })
        except Exception as exc:
            return jsonify({"error": str(exc)}), 400

    @app.get("/api/analysis/kinematics/timeseries")
    def api_analysis_kinematics_timeseries():
        try:
            session = _analysis_session_from_request(sessions)
            signal = str(request.args.get("signal") or "").strip()
            signal_map = {item["key"]: item for item in _kinematics_signals()}
            if signal not in signal_map:
                raise ValueError("invalid_signal")
            csv_path = _latest_kinematics_csv_file(Path(session.session_path))
            if csv_path is None:
                raise ValueError("kinematics_csv_not_found")
            columns = _read_csv_columns(csv_path)
            if signal not in columns:
                raise ValueError("signal_not_found")
            return jsonify({
                "unit": signal_map[signal]["unit"],
                "time": _finite_or_null(columns.get("time", [])),
                "values": _finite_or_null(columns.get(signal, [])),
            })
        except Exception as exc:
            return jsonify({"error": str(exc)}), 400

    @app.get("/report")
    def report_page():
        return render_template("workflow_placeholder.html", title="Report")

    @app.get("/api/cameras")
    def api_cameras():
        return jsonify([_camera_to_dict(c) for c in capture_service.list_cameras()])

    @app.get("/api/settings/cameras")
    def api_get_camera_settings():
        return jsonify(capture_service.camera_settings())

    @app.post("/api/settings/cameras")
    def api_set_camera_settings():
        data = request.get_json(silent=True) or {}
        try:
            capture_service.configure_cameras(
                int(data.get("camera_count", 1)),
                str(data.get("ccb_url", "")).strip(),
                str(data.get("live_view_frame_rate", "low")).lower(),
            )
            capture_service.configure_capture_mode(
                str(data.get("capture_mode", capture_service.camera_settings()["capture_mode"])).lower(),
                int(data.get("phone_camera_count", capture_service.camera_settings()["phone_camera_count"])),
                int(data.get("phone_frame_rate", capture_service.camera_settings()["phone_frame_rate"])),
                "landscape",
            )
            _emit_camera_status(socketio, capture_service)
            return jsonify(capture_service.camera_settings())
        except Exception as exc:
            return jsonify({"error": str(exc)}), 400

    @app.get("/api/capture/mode")
    def api_get_capture_mode():
        return jsonify(capture_service.camera_settings())

    @app.post("/api/capture/mode")
    def api_set_capture_mode():
        data = request.get_json(silent=True) or {}
        capture_service.configure_capture_mode(
            str(data.get("capture_mode", "sony")).lower(),
            int(data.get("phone_camera_count", capture_service.camera_settings()["phone_camera_count"])),
            int(data.get("phone_frame_rate", capture_service.camera_settings()["phone_frame_rate"])),
            "landscape",
        )
        _emit_camera_status(socketio, capture_service)
        return jsonify(capture_service.camera_settings())

    @app.post("/api/phone-sessions")
    def api_create_phone_session():
        return jsonify(_phone_draft_to_dict(phone_service.create_draft(_base_url())))

    @app.get("/api/phone-sessions/<session_token>/qr")
    def api_get_phone_qr(session_token):
        draft = phone_service.current_or_create_draft(_base_url())
        if draft.token != session_token:
            draft = phone_service.create_draft(_base_url())
        return jsonify(_phone_draft_to_dict(draft))

    @app.post("/api/phone-sessions/<session_token>/finalize")
    def api_finalize_phone_session(session_token):
        session = phone_service.stop_session(session_token)
        return jsonify({"status": "finalized", "session": _session_to_dict(session) if session else None})

    @app.post("/api/phone-sessions/<session_token>/<camera_label>/upload")
    def api_upload_phone_video(session_token, camera_label):
        try:
            upload = request.files.get("video")
            result = phone_service.save_upload(
                session_token,
                camera_label,
                upload,
                upload.mimetype if upload else "video/mp4",
                request.headers.get("User-Agent", ""),
                request.form.get("actual_fps"),
                request.form.get("actual_width"),
                request.form.get("actual_height"),
            )
            socketio.emit("phone_upload_complete", {"token": session_token, "camera_label": camera_label, **result})
            return jsonify(result)
        except Exception as exc:
            return jsonify({"error": str(exc)}), 400

    @app.get("/api/sessions")
    def api_sessions():
        return jsonify([_session_to_dict(s) for s in capture_service.list_sessions()])

    @app.delete("/api/sessions/<session_id>")
    def api_delete_session(session_id):
        try:
            session = capture_service.delete_session(session_id)
            return jsonify({"status": "deleted", "session": _session_to_dict(session)})
        except Exception as exc:
            return jsonify({"error": str(exc)}), 400

    @app.post("/api/sessions/rescan")
    def api_rescan_sessions():
        return jsonify([_session_to_dict(s) for s in capture_service.list_sessions()])

    @app.get("/api/settings/storage-root")
    def api_get_storage_root():
        return jsonify({"storage_root": capture_service.get_storage_root()})

    @app.post("/api/settings/storage-root")
    def api_set_storage_root():
        data = request.get_json(silent=True) or {}
        storage_root = str(data.get("storage_root", "")).strip()
        if not storage_root:
            return jsonify({"error": "storage_root_required"}), 400
        return jsonify({"storage_root": capture_service.set_storage_root(storage_root)})

    @app.post("/api/settings/storage-root/select")
    def api_select_storage_root():
        data = request.get_json(silent=True) or {}
        requested_root = str(data.get("storage_root") or "").strip()
        if data.get("manual") and requested_root:
            return jsonify({"storage_root": capture_service.set_storage_root(requested_root), "cancelled": False})
        current_root = capture_service.get_storage_root()
        try:
            selected_path = _select_directory(current_root)
        except RuntimeError as exc:
            return jsonify({
                "storage_root": current_root,
                "cancelled": True,
                "manual_required": True,
                "error": str(exc),
            })
        if selected_path is None:
            return jsonify({"storage_root": capture_service.get_storage_root(), "cancelled": True})
        return jsonify({"storage_root": capture_service.set_storage_root(selected_path), "cancelled": False})

    @app.post("/api/capture/start")
    def api_start_capture():
        data = request.get_json(silent=True) or {}
        try:
            subject = SubjectInfo(
                name=str(data.get("name", "")).strip(),
                height_cm=int(data.get("height_cm")),
                weight_kg=int(data.get("weight_kg")),
                hand=_safe_hand(str(data.get("hand", "right"))),
            )
            camera_ids = [str(v) for v in data.get("camera_ids", [])]
            session = capture_service.start_capture(subject, camera_ids)
            if capture_service.camera_settings()["capture_mode"] == "phone":
                token = str(data.get("phone_session_token") or phone_service.current_or_create_draft(_base_url()).token)
                phone_service.start_session(token, session)
                socketio.emit(
                    "phone_recording_command",
                    {
                        "command": "start",
                        "token": token,
                        "session": _session_to_dict(session),
                        "settings": phone_service.settings_payload(),
                    },
                )
            _emit_camera_status(socketio, capture_service)
            socketio.emit("capture_status", {"status": "recording", "session": _session_to_dict(session)})
            return jsonify(_session_to_dict(session))
        except Exception as exc:
            return jsonify({"error": str(exc)}), 400

    @app.post("/api/capture/stop")
    def api_stop_capture():
        try:
            settings_payload = capture_service.camera_settings()
            token = phone_service.current_or_create_draft(_base_url()).token
            session = capture_service.stop_capture()
            if settings_payload["capture_mode"] == "phone":
                socketio.emit(
                    "phone_recording_command",
                    {
                        "command": "stop",
                        "token": token,
                        "session": _session_to_dict(session),
                        "settings": phone_service.settings_payload(),
                    },
                )
            _emit_camera_status(socketio, capture_service)
            socketio.emit("capture_status", {"status": "captured", "session": _session_to_dict(session)})
            return jsonify(_session_to_dict(session))
        except Exception as exc:
            return jsonify({"error": str(exc)}), 400

    @app.post("/api/calibration/start")
    def api_start_calibration():
        data = request.get_json(silent=True) or {}
        try:
            mode = _calibration_mode(str(data.get("calibration_mode", "intrinsic")))
            cameras = capture_service.list_cameras()
            connected_cameras = [camera for camera in cameras if camera.connected]
            record_camera_ids = [camera.camera_id for camera in connected_cameras]
            if mode == "INTR":
                selected_label = str(data.get("intrinsic_camera_label", "")).strip()
                if not selected_label:
                    raise ValueError("intrinsic_camera_required")
                save_camera_labels = {selected_label}
            else:
                save_camera_labels = {camera.label for camera in connected_cameras}

            calibration = calibration_service.start(
                mode,
                str(data.get("project_name", "")),
                record_camera_ids,
                save_camera_labels,
                {
                    "checker_board_type": data.get("checker_board_type"),
                    "aruco_dictionary": data.get("aruco_dictionary"),
                    "checker_board_size_mm": data.get("checker_board_size_mm"),
                    "marker_size_mm": data.get("marker_size_mm"),
                    "checker_board_columns": data.get("checker_board_columns"),
                    "checker_board_rows": data.get("checker_board_rows"),
                    "object_points": data.get("object_points"),
                },
            )
            camera_controller.start_recording(record_camera_ids)
            if capture_service.camera_settings()["capture_mode"] == "phone":
                token = str(data.get("phone_session_token") or phone_service.current_or_create_draft(_base_url()).token)
                phone_service.start_calibration(
                    token,
                    calibration.mode,
                    calibration.project_name,
                    str(calibration.output_dir),
                    save_camera_labels,
                )
                socketio.emit(
                    "phone_recording_command",
                    {
                        "command": "start",
                        "token": token,
                        "session": _calibration_to_dict(calibration, "recording"),
                        "settings": phone_service.settings_payload(),
                    },
                )
            _emit_camera_status(socketio, capture_service)
            return jsonify(_calibration_to_dict(calibration, "recording"))
        except Exception as exc:
            return jsonify({"error": str(exc)}), 400

    @app.post("/api/calibration/stop")
    def api_stop_calibration():
        data = request.get_json(silent=True) or {}
        try:
            calibration = calibration_service.stop()
            camera_controller.stop_recording(
                calibration.record_camera_ids,
                str(calibration.output_dir),
                _calibration_subject(calibration),
                calibration.timestamp,
            )
            if calibration.mode == "INTR":
                _remove_unselected_calibration_videos(calibration)
            if capture_service.camera_settings()["capture_mode"] == "phone":
                token = str(data.get("phone_session_token") or phone_service.current_or_create_draft(_base_url()).token)
                phone_service.stop_calibration(token)
                socketio.emit(
                    "phone_recording_command",
                    {
                        "command": "stop",
                        "token": token,
                        "session": _calibration_to_dict(calibration, "captured"),
                        "settings": phone_service.settings_payload(),
                    },
                )
            _emit_camera_status(socketio, capture_service)
            return jsonify(_calibration_to_dict(calibration, "captured"))
        except Exception as exc:
            return jsonify({"error": str(exc)}), 400

    @app.get("/api/calibrations")
    def api_calibrations():
        return jsonify([_calibration_record_to_dict(item) for item in calibration_service.list_calibrations()])

    @app.delete("/api/calibrations/<folder_name>")
    def api_delete_calibration(folder_name):
        try:
            calibration = calibration_service.delete_calibration(folder_name)
            return jsonify({"status": "deleted", "calibration": _calibration_record_to_dict(calibration)})
        except Exception as exc:
            return jsonify({"error": str(exc)}), 400

    @app.post("/api/calibrations/<folder_name>/run")
    def api_run_calibration(folder_name):
        data = request.get_json(silent=True) or {}
        try:
            result = calibration_service.run_calibration(folder_name, data)
            return jsonify(result)
        except Exception as exc:
            return jsonify({"ok": False, "error": str(exc), "folder_name": folder_name})

    @app.get("/api/calibrations/<folder_name>/frames")
    def api_calibration_frames(folder_name):
        try:
            record = next((item for item in calibration_service.list_calibrations() if item.folder_name == folder_name), None)
            if record is None:
                raise ValueError("calibration_not_found")
            if record.mode != "EXTR":
                raise ValueError("extrinsic_calibration_required")
            frames = []
            for video in record.videos[:4]:
                frames.append({
                    "camera_label": video.camera_label,
                    "image": _first_frame_data_url(Path(video.path)),
                })
            if len(frames) < 2:
                raise ValueError(f"need_at_least_2_extrinsic_videos: {len(frames)}")
            return jsonify({"folder_name": folder_name, "frames": frames})
        except Exception as exc:
            return jsonify({"error": str(exc)}), 400

    @app.get("/phone-capture/<session_token>/<camera_label>")
    def phone_capture_page(session_token, camera_label):
        return render_template(
            "phone_capture.html",
            session_token=session_token,
            camera_label=camera_label,
            phone_settings=phone_service.settings_payload(),
        )

    @socketio.on("phone_register")
    def on_phone_register(payload):
        camera_label = _payload_camera_label(payload)
        if camera_label and hasattr(camera_controller, "mark_phone_connected"):
            phone_socket_labels[request.sid] = camera_label
            camera_controller.mark_phone_connected(camera_label)
            _emit_camera_status(socketio, capture_service)
        socketio.emit("phone_registered", payload)

    @socketio.on("phone_preview_status")
    def on_phone_preview_status(payload):
        camera_label = _payload_camera_label(payload)
        if camera_label:
            phone_socket_labels[request.sid] = camera_label
            if str(payload.get("status")) == "blocked" and hasattr(camera_controller, "mark_phone_blocked"):
                camera_controller.mark_phone_blocked(camera_label, str(payload.get("message") or ""))
            elif hasattr(camera_controller, "mark_phone_connected"):
                camera_controller.mark_phone_connected(camera_label)
            _emit_camera_status(socketio, capture_service)
        socketio.emit("phone_preview_status", payload)

    @socketio.on("phone_preview_frame")
    def on_phone_preview_frame(payload):
        socketio.emit("phone_preview_frame", payload)

    @socketio.on("connect")
    def on_connect():
        _emit_camera_status(socketio, capture_service)

    @socketio.on("disconnect")
    def on_disconnect():
        camera_label = phone_socket_labels.pop(request.sid, "")
        if camera_label and hasattr(camera_controller, "mark_phone_disconnected"):
            camera_controller.mark_phone_disconnected(camera_label)
            _emit_camera_status(socketio, capture_service)

    return app, socketio


def _camera_controller_from_env(settings):
    sony_count = int(os.environ.get("BASEBALL_MOTION_CAMERA_COUNT", str(settings.get_camera_count())))
    return ModeCameraController(
        settings,
        UrlCameraController(camera_count=sony_count),
        PhoneCameraController(camera_count=settings.get_phone_camera_count()),
    )


def _capture_payload_from_form():
    subject = SubjectInfo(
        name=request.form.get("name", "").strip() or "subject",
        height_cm=int(request.form.get("height_cm", "170")),
        weight_kg=int(request.form.get("weight_kg", "70")),
        hand=_safe_hand(request.form.get("hand", "right")),
    )
    camera_ids = request.form.getlist("camera_ids")
    return {"subject": subject, "camera_ids": camera_ids}


def _emit_camera_status(socketio: SocketIO, capture_service: CaptureService) -> None:
    socketio.emit("camera_status", [_camera_to_dict(c) for c in capture_service.list_cameras()])


def _payload_camera_label(payload) -> str:
    if isinstance(payload, dict):
        return str(payload.get("camera_label") or "").strip()
    return ""


def _safe_hand(hand: str) -> str:
    value = str(hand or "right").strip().lower()
    return value if value in {"right", "left"} else "right"


def _camera_to_dict(camera):
    return {
        "camera_id": camera.camera_id,
        "label": camera.label,
        "connected": camera.connected,
        "recording": camera.recording,
        "live_view_url": camera.live_view_url,
        "live_view_frame_rate": camera.live_view_frame_rate,
        "last_error": camera.last_error,
    }


def _session_to_dict(session):
    return {
        "session_id": session.session_id,
        "subject": {
            "name": session.subject.name,
            "height_cm": session.subject.height_cm,
            "weight_kg": session.subject.weight_kg,
            "hand": session.subject.hand,
        },
        "timestamp": session.timestamp,
        "display_timestamp": _display_timestamp(session.timestamp),
        "session_path": session.session_path,
        "status": session.status,
        "videos": [_video_to_dict(v, session.session_path) for v in session.videos],
        "created_at": session.created_at.isoformat(timespec="seconds") if session.created_at else None,
        "updated_at": session.updated_at.isoformat(timespec="seconds") if session.updated_at else None,
    }


def _session_by_path(session_list, session_path: Path):
    resolved = session_path.resolve()
    return next((session for session in session_list if Path(session.session_path).resolve() == resolved), None)


def _analysis_session_from_request(sessions):
    session_path = Path(str(request.args.get("session_path") or "")).expanduser().resolve()
    if not session_path.is_dir():
        raise ValueError("session_path_not_found")
    session = _session_by_path(sessions.list_sessions(str(session_path.parent)), session_path)
    if session is None:
        raise ValueError("session_not_found")
    return session


def _pose3d_data_from_trc(trc_path: Path) -> dict:
    if not trc_path.is_file() or trc_path.suffix.lower() != ".trc":
        raise ValueError("trc_file_not_found")
    import math
    from pipelines.utilities import read_trc

    coordinates, _frames_col, time_col, markers, header = read_trc(trc_path)
    fps = 30.0
    try:
        fps = float(header[2].split("\t")[0])
    except Exception:
        pass

    def safe_float(value):
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        return round(number, 4) if math.isfinite(number) else None

    frames = []
    for _, row in coordinates.iterrows():
        values = row.tolist()
        frame_points = []
        for marker_index in range(len(markers)):
            offset = marker_index * 3
            frame_points.append([
                safe_float(values[offset] if offset < len(values) else None),
                safe_float(values[offset + 1] if offset + 1 < len(values) else None),
                safe_float(values[offset + 2] if offset + 2 < len(values) else None),
            ])
        frames.append(frame_points)

    return {
        "file": trc_path.name,
        "markers": markers,
        "fps": fps,
        "num_frames": len(frames),
        "time": [safe_float(value) for value in time_col.tolist()],
        "frames": frames,
    }


def _normalized_pose_camera_label(label: str) -> str:
    import re

    match = re.search(r"cam0*(\d+)$", str(label).lower())
    return f"cam{int(match.group(1))}" if match else str(label).strip().lower()


def _keypoint_json_path(session_path: str, camera_label: str, frame: int) -> Path:
    pose_label = _normalized_pose_camera_label(camera_label)
    if not pose_label:
        raise ValueError("camera_label_required")
    if frame < 0:
        raise ValueError("invalid_frame")
    path = Path(session_path) / "pose" / f"{pose_label}_json" / f"{pose_label}_{frame:06d}.json"
    if not path.is_file():
        raise ValueError("keypoint_json_not_found")
    return path


def _keypoint_names(count: int) -> list[str]:
    halpe26 = [
        "Nose", "LEye", "REye", "LEar", "REar", "LShoulder", "RShoulder", "LElbow", "RElbow",
        "LWrist", "RWrist", "LHip", "RHip", "LKnee", "RKnee", "LAnkle", "RAnkle", "Head",
        "Neck", "Hip", "LBigToe", "RBigToe", "LSmallToe", "RSmallToe", "LHeel", "RHeel",
    ]
    return [halpe26[index] if index < len(halpe26) else f"Keypoint {index}" for index in range(count)]


def _keypoint_frame_from_json(session_path: str, camera_label: str, frame: int) -> dict:
    import json
    import math

    path = _keypoint_json_path(session_path, camera_label, frame)
    payload = json.loads(path.read_text(encoding="utf-8"))
    people = payload.get("people") or []
    people_keypoints = []
    max_keypoints = 0
    for person_index, person in enumerate(people):
        if not isinstance(person, dict):
            continue
        raw_keypoints = person.get("pose_keypoints_2d") or []
        keypoints = []
        for index in range(0, len(raw_keypoints), 3):
            keypoints.append({
                "x": _json_number(raw_keypoints[index] if index < len(raw_keypoints) else None, math),
                "y": _json_number(raw_keypoints[index + 1] if index + 1 < len(raw_keypoints) else None, math),
                "score": _json_number(raw_keypoints[index + 2] if index + 2 < len(raw_keypoints) else None, math),
            })
        max_keypoints = max(max_keypoints, len(keypoints))
        people_keypoints.append({"person_index": person_index, "keypoints": keypoints})
    return {
        "camera_label": _normalized_pose_camera_label(camera_label),
        "frame": frame,
        "file": path.name,
        "keypoint_names": _keypoint_names(max_keypoints),
        "people": people_keypoints,
    }


def _json_number(value, math_module) -> float | None:
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math_module.isfinite(number) else None


def _save_keypoint_frame_to_json(session_path: str, camera_label: str, frame: int, people_keypoints: list) -> None:
    import json
    import math

    path = _keypoint_json_path(session_path, camera_label, frame)
    payload = json.loads(path.read_text(encoding="utf-8"))
    people = payload.setdefault("people", [])
    if not people:
        people.append({})
    for person_payload in people_keypoints:
        if not isinstance(person_payload, dict):
            raise ValueError("invalid_person_keypoints")
        person_index = int(person_payload.get("person_index", 0))
        while len(people) <= person_index:
            people.append({})
        flattened = []
        for point in person_payload.get("keypoints") or []:
            if not isinstance(point, dict):
                raise ValueError("invalid_keypoint")
            for key in ("x", "y", "score"):
                value = point.get(key)
                if value is None or value == "":
                    flattened.append(None)
                    continue
                number = float(value)
                flattened.append(number if math.isfinite(number) else None)
        people[person_index]["pose_keypoints_2d"] = flattened
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _render_pose_video_from_keypoints(session, camera_label: str) -> Path:
    import json
    import numpy as np
    from pipelines.utilities import draw_keypts, draw_skel, setup_video, transcode_to_h264

    pose_label = _normalized_pose_camera_label(camera_label)
    video = next((item for item in session.videos if _normalized_pose_camera_label(item.camera_label) == pose_label), None)
    if video is None:
        raise ValueError("video_not_found")

    session_path = Path(session.session_path)
    json_dir = session_path / "pose" / f"{pose_label}_json"
    if not json_dir.is_dir():
        raise ValueError("keypoint_json_dir_not_found")
    output_dir = session_path / "pose"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{pose_label}_pose.mp4"
    temp_path = output_dir / f"{pose_label}_pose_tmp.mp4"

    cap, writer, _width, _height, _fps = setup_video(Path(video.path), temp_path, True)
    frame_index = 0
    try:
        while True:
            success, frame = cap.read()
            if not success:
                break
            json_path = json_dir / f"{pose_label}_{frame_index:06d}.json"
            if json_path.is_file():
                payload = json.loads(json_path.read_text(encoding="utf-8"))
                keypoints = []
                scores = []
                for person in payload.get("people") or []:
                    if not isinstance(person, dict):
                        continue
                    raw = np.asarray(person.get("pose_keypoints_2d") or [], dtype=float)
                    if raw.size == 0 or raw.size % 3 != 0:
                        continue
                    points = raw.reshape((-1, 3))
                    keypoints.append(points[:, :2])
                    scores.append(points[:, 2])
                if keypoints:
                    keypoints_array = np.asarray(keypoints, dtype=float)
                    scores_array = np.asarray(scores, dtype=float)
                    frame = draw_keypts(frame, keypoints_array[:, :, 0], keypoints_array[:, :, 1], scores_array, cmap_str="RdYlGn")
                    frame = draw_skel(frame, keypoints_array[:, :, 0], keypoints_array[:, :, 1])
            writer.write(frame)
            frame_index += 1
    finally:
        cap.release()
        if writer is not None:
            writer.release()

    transcode_to_h264(temp_path)
    if output_path.exists():
        output_path.unlink()
    temp_path.replace(output_path)
    return output_path


def _latest_mot_file(session_path: Path) -> Path | None:
    kinematics_dir = session_path / "kinematics"
    if not kinematics_dir.is_dir():
        return None
    mot_files = [path for path in kinematics_dir.glob("*.mot") if path.is_file()]
    if not mot_files:
        return None
    return max(mot_files, key=lambda path: path.stat().st_mtime)


def _latest_kinematics_csv_file(session_path: Path) -> Path | None:
    csv_files = [path for path in session_path.glob("*_keypoints_kinematics.csv") if path.is_file()]
    if not csv_files:
        csv_files = [path for path in session_path.glob("*.csv") if path.is_file()]
    if not csv_files:
        return None
    return max(csv_files, key=lambda path: path.stat().st_mtime)


def _read_csv_columns(csv_path: Path) -> dict[str, list[float]]:
    if not csv_path.is_file():
        raise ValueError("kinematics_csv_not_found")
    import csv

    columns: dict[str, list[float]] = {}
    with csv_path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError("invalid_csv_header")
        columns = {field: [] for field in reader.fieldnames}
        for row in reader:
            for field in columns:
                try:
                    value = float(row.get(field, "nan"))
                except ValueError:
                    value = float("nan")
                columns[field].append(value)
    return columns


def _kinematics_event_markers(csv_path: Path, columns: dict[str, list[float]]) -> list[dict[str, float | int | str]]:
    if not all(f"{key}_time" in columns for key in ("knee_high", "mer", "ball_release")):
        return []
    recalculated = _recalculate_kinematics_event_markers(csv_path)
    if recalculated:
        return recalculated
    events = [
        ("knee_high", "KH", "Knee High"),
        ("mer", "MER", "Max Shoulder External Rotation"),
        ("ball_release", "BR", "Ball Release"),
    ]
    markers: list[dict[str, float | int | str]] = []
    for key, label, description in events:
        frame = _first_finite(columns.get(f"{key}_frame", []))
        time = _first_finite(columns.get(f"{key}_time", []))
        if time is None:
            continue
        marker: dict[str, float | int | str] = {
            "key": key,
            "label": label,
            "description": description,
            "time": time,
        }
        if frame is not None:
            marker["frame"] = int(frame)
        markers.append(marker)
    return markers


def _recalculate_kinematics_event_markers(csv_path: Path) -> list[dict[str, float | int | str]]:
    try:
        import pandas as pd
        from pipelines.parameters import extract_pitching_events_from_dataframe

        df = pd.read_csv(csv_path)
        if "hand" not in df.columns:
            return []
        hand_values = df["hand"].dropna()
        if hand_values.empty:
            return []
        events = extract_pitching_events_from_dataframe(df, str(hand_values.iloc[0]), _infer_csv_fps(df))
    except Exception:
        return []
    labels = {
        "knee_high": ("KH", "Knee High"),
        "mer": ("MER", "Max Shoulder External Rotation"),
        "ball_release": ("BR", "Ball Release"),
    }
    markers: list[dict[str, float | int | str]] = []
    for key, event in events.items():
        label, description = labels[key]
        time = event.get("time")
        if time is None:
            continue
        marker: dict[str, float | int | str] = {
            "key": key,
            "label": label,
            "description": description,
            "time": float(time),
        }
        frame = event.get("frame")
        if frame is not None:
            marker["frame"] = int(frame)
        markers.append(marker)
    return markers


def _infer_csv_fps(df) -> float:
    if "time" not in df.columns:
        return 60.0
    import numpy as np
    import pandas as pd

    time_values = pd.to_numeric(df["time"], errors="coerce").to_numpy(dtype=float)
    dt = np.diff(time_values)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if len(dt) == 0:
        return 60.0
    return 1.0 / float(np.median(dt))


def _first_finite(values: list[float]) -> float | None:
    import math

    for value in values:
        if isinstance(value, float) and math.isfinite(value):
            return value
    return None


def _read_mot_table(mot_path: Path) -> tuple[bool, dict[str, list[float]]]:
    if not mot_path.is_file():
        raise ValueError("mot_file_not_found")
    in_degrees = False
    columns: dict[str, list[float]] = {}
    headers: list[str] | None = None
    with mot_path.open("r", encoding="utf-8", errors="replace") as handle:
        iterator = iter(handle)
        for line in iterator:
            stripped = line.strip()
            if "inDegrees" in stripped:
                lowered = stripped.lower()
                in_degrees = "yes" in lowered or "true" in lowered
            if stripped == "endheader":
                headers = next(iterator, "").strip().split()
                break
        if not headers:
            raise ValueError("invalid_mot_header")
        columns = {header: [] for header in headers}
        for line in iterator:
            parts = line.strip().split()
            if len(parts) < len(headers):
                continue
            for index, header in enumerate(headers):
                try:
                    value = float(parts[index])
                except ValueError:
                    value = float("nan")
                columns[header].append(value)
    return in_degrees, columns


def _kinematics_signals() -> list[dict[str, str]]:
    angle_signals = [
        {"key": "hip_flexion_l", "label": "Hip Flexion", "side": "Left", "category": "hip"},
        {"key": "hip_flexion_r", "label": "Hip Flexion", "side": "Right", "category": "hip"},
        {"key": "hip_adduction_l", "label": "Hip Abduction", "side": "Left", "category": "hip"},
        {"key": "hip_adduction_r", "label": "Hip Abduction", "side": "Right", "category": "hip"},
        {"key": "hip_rotation_l", "label": "Hip Rotation", "side": "Left", "category": "hip"},
        {"key": "hip_rotation_r", "label": "Hip Rotation", "side": "Right", "category": "hip"},
        {"key": "pelvis_tilt", "label": "Pelvis Tilt", "side": "Center", "category": "pelvis"},
        {"key": "pelvis_list", "label": "Pelvis List", "side": "Center", "category": "pelvis"},
        {"key": "pelvis_rotation", "label": "Pelvis Rotation", "side": "Center", "category": "pelvis"},
        {"key": "L5_S1_Flex_Ext", "label": "Flexion", "side": "Center", "category": "hip_shoulder"},
        {"key": "L5_S1_Lat_Bending", "label": "Lateral Bend", "side": "Center", "category": "hip_shoulder"},
        {"key": "L5_S1_axial_rotation", "label": "Rotation", "side": "Center", "category": "hip_shoulder"},
        {"key": "trunk_tilt_global", "label": "Trunk Tilt", "side": "Global", "category": "trunk"},
        {"key": "trunk_list_global", "label": "Trunk List", "side": "Global", "category": "trunk"},
        {"key": "trunk_rotation_global", "label": "Trunk Rotation", "side": "Global", "category": "trunk"},
        {"key": "knee_angle_l", "label": "Knee Flexion", "side": "Left", "category": "knee"},
        {"key": "knee_angle_r", "label": "Knee Flexion", "side": "Right", "category": "knee"},
        {"key": "ankle_angle_l", "label": "Ankle Dorsiflexion", "side": "Left", "category": "ankle"},
        {"key": "ankle_angle_r", "label": "Ankle Dorsiflexion", "side": "Right", "category": "ankle"},
        {"key": "arm_flex_l", "label": "Shoulder Flexion", "side": "Left", "category": "shoulder"},
        {"key": "arm_flex_r", "label": "Shoulder Flexion", "side": "Right", "category": "shoulder"},
        {"key": "arm_add_l", "label": "Shoulder Adduction", "side": "Left", "category": "shoulder"},
        {"key": "arm_add_r", "label": "Shoulder Adduction", "side": "Right", "category": "shoulder"},
        {"key": "arm_rot_l", "label": "Shoulder Rotation", "side": "Left", "category": "shoulder"},
        {"key": "arm_rot_r", "label": "Shoulder Rotation", "side": "Right", "category": "shoulder"},
        {"key": "elbow_flex_l", "label": "Elbow Flexion", "side": "Left", "category": "elbow"},
        {"key": "elbow_flex_r", "label": "Elbow Flexion", "side": "Right", "category": "elbow"},
        {"key": "pro_sup_l", "label": "Pronation Supination", "side": "Left", "category": "elbow"},
        {"key": "pro_sup_r", "label": "Pronation Supination", "side": "Right", "category": "elbow"},
    ]
    signals = [{**signal, "kind": "angle", "unit": "deg"} for signal in angle_signals]
    signals.extend([
        {
            **signal,
            "key": f"{signal['key']}_velocity",
            "kind": "velocity",
            "unit": "deg/s",
        }
        for signal in angle_signals
    ])
    return signals


def _finite_or_null(values: list[float]) -> list[float | None]:
    import math

    return [round(value, 4) if isinstance(value, float) and math.isfinite(value) else None for value in values]


def _run_analysis_job(
    job_id: str,
    session,
    user_config: dict,
    jobs: dict[str, dict],
    analysis_service: AnalysisPipelineService,
) -> None:
    job = jobs[job_id]

    def emit_log(message, level="info"):
        job["logs"].append({"level": level, "message": str(message)})

    try:
        job["status"] = "running"
        analysis_service.run(session, user_config, emit_log)
        job["status"] = "completed"
    except Exception as exc:
        job["status"] = "failed"
        job["error"] = str(exc)
        emit_log(str(exc), "error")


def _calibration_to_dict(calibration, status: str):
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


def _calibration_record_to_dict(record):
    return {
        "mode": record.mode,
        "project_name": record.project_name,
        "folder_name": record.folder_name,
        "display_name": f"CALIB {record.project_name}",
        "output_dir": str(record.output_dir),
        "updated_at": record.updated_at.isoformat(timespec="seconds"),
        "display_updated_at": record.updated_at.strftime("%Y-%m-%d %H:%M:%S"),
        "videos": [_calibration_video_to_dict(video) for video in record.videos],
    }


def _calibration_video_to_dict(video):
    path = Path(video.path)
    size_bytes = path.stat().st_size if path.is_file() else 0
    return {
        "camera_label": video.camera_label,
        "path": str(path),
        "filename": path.name,
        "size_bytes": size_bytes,
        "size_label": _format_size(size_bytes),
    }


def _first_frame_data_url(video_path: Path) -> str:
    import cv2

    capture = cv2.VideoCapture(str(video_path))
    try:
        ok, frame = capture.read()
    finally:
        capture.release()
    if not ok or frame is None:
        raise ValueError(f"cannot_read_first_frame: {video_path.name}")
    ok, buffer = cv2.imencode(".jpg", frame)
    if not ok:
        raise ValueError(f"cannot_encode_first_frame: {video_path.name}")
    encoded = base64.b64encode(buffer.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def _calibration_subject(calibration) -> SubjectInfo:
    return SubjectInfo(
        name=calibration.mode,
        height_cm=0,
        weight_kg=0,
        hand=calibration.project_name,
    )


def _calibration_mode(value: str) -> str:
    normalized = str(value or "intrinsic").strip().lower()
    if normalized in {"intrinsic", "intr", "in"}:
        return "INTR"
    if normalized in {"extrinsic", "extr", "ex"}:
        return "EXTR"
    raise ValueError("unknown_calibration_mode")


def _remove_unselected_calibration_videos(calibration) -> None:
    for video in Path(calibration.output_dir).glob("*.mp4"):
        if not any(video.stem.lower().endswith(label.lower()) for label in calibration.save_camera_labels):
            video.unlink()
            metadata = video.with_suffix(video.suffix + ".json")
            if metadata.exists():
                metadata.unlink()


def _video_to_dict(video, session_path: str | None = None):
    import re
    import cv2

    match = re.search(r"cam0*(\d+)$", str(video.camera_label).lower())
    pose_label = f"cam{int(match.group(1))}" if match else video.camera_label
    path = Path(video.path)
    size_bytes = path.stat().st_size if path.is_file() else 0
    pose_path = Path(session_path or path.parent) / "pose" / f"{pose_label}_pose.mp4"
    fps = 0.0
    frame_count = 0
    if path.is_file():
        capture = cv2.VideoCapture(str(path))
        try:
            fps = float(capture.get(cv2.CAP_PROP_FPS) or 0)
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        finally:
            capture.release()
    return {
        "camera_id": video.camera_id,
        "camera_label": video.camera_label,
        "path": video.path,
        "video_url": url_for("api_analysis_video", path=str(path)),
        "pose_video_path": str(pose_path) if pose_path.is_file() else None,
        "pose_video_url": (
            url_for("api_analysis_video", path=str(pose_path), v=int(pose_path.stat().st_mtime))
            if pose_path.is_file() else None
        ),
        "filename": path.name,
        "fps": fps,
        "frame_count": frame_count,
        "size_bytes": size_bytes,
        "size_label": _format_size(size_bytes),
    }


def _display_timestamp(timestamp: str) -> str:
    if len(timestamp) == 15 and timestamp[8] == "_":
        return f"{timestamp[:4]}-{timestamp[4:6]}-{timestamp[6:8]} {timestamp[9:]}"
    return timestamp


def _format_size(size_bytes: int) -> str:
    if size_bytes >= 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    if size_bytes >= 1024:
        return f"{size_bytes / 1024:.1f} KB"
    return f"{size_bytes} B"


def _select_directory(initial_dir: str) -> str | None:
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        selected = filedialog.askdirectory(initialdir=initial_dir, title="Select storage folder")
        root.destroy()
        return selected or None
    except Exception as exc:
        raise RuntimeError(f"folder_dialog_unavailable: {exc}") from exc


def _phone_draft_to_dict(draft):
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


def _base_url() -> str:
    public_url = os.environ.get("BASEBALL_MOTION_PUBLIC_URL", "").strip()
    if public_url:
        return public_url.rstrip("/") + "/"
    return request.url_root


def _static_asset_url(filename: str) -> str:
    static_path = Path(__file__).resolve().parent / "static" / filename
    version = str(int(static_path.stat().st_mtime)) if static_path.is_file() else "0"
    return url_for("static", filename=filename, v=version)


def _app_logo_path() -> Path:
    return Path(__file__).resolve().parents[2] / "images" / "baseball-motion.png"

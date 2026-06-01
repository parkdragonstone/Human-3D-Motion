from __future__ import annotations

import os
from pathlib import Path

from flask import Flask, jsonify, redirect, render_template, request, send_file, url_for
from flask_socketio import SocketIO

from webapp.application.phone_capture_service import PhoneVideoUpload
from webapp.bootstrap import create_app_services
from webapp.presentation.request_parsers import (
    calibration_mode as _calibration_mode,
    capture_payload_from_form as _capture_payload_from_form,
    capture_subject_from_json as _capture_subject_from_json,
    payload_camera_label as _payload_camera_label,
)
from webapp.presentation.serializers import (
    calibration_record_to_dict as _calibration_record_to_dict,
    calibration_to_dict as _calibration_to_dict,
    camera_to_dict as _camera_to_dict,
    phone_draft_to_dict as _phone_draft_to_dict,
    session_to_dict as _session_to_dict,
)


def create_app():
    app = Flask(
        __name__,
        template_folder=str(Path(__file__).resolve().parent / "templates"),
        static_folder=str(Path(__file__).resolve().parent / "static"),
    )
    app.config["SECRET_KEY"] = os.environ.get("HUMAN_3D_MOTION_SECRET", "dev")
    app.config["TEMPLATES_AUTO_RELOAD"] = True
    app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0
    socketio = SocketIO(
        app,
        cors_allowed_origins="*",
        async_mode="threading",
        logger=False,
        engineio_logger=False,
    )

    services = create_app_services()
    media_view_service = services.media_view_service
    storage_root_service = services.storage_root_service
    capture_service = services.capture_service
    capture_recording_service = services.capture_recording_service
    analysis_workspace_service = services.analysis_workspace_service
    phone_service = services.phone_service
    calibration_service = services.calibration_service
    calibration_recording_service = services.calibration_recording_service
    phone_socket_sessions: dict[str, dict[str, str]] = {}

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
            storage_root=storage_root_service.get(),
            cameras=capture_service.list_cameras(),
            camera_settings=capture_service.camera_settings(),
            phone_draft=phone_draft,
            active_capture=capture_service.active_capture(),
            sessions=sessions,
            recent_sessions=[_session_to_dict(media_view_service.session_view(session)) for session in sessions[:6]],
        )

    @app.post("/settings/storage-root")
    def set_storage_root_form():
        storage_root = request.form.get("storage_root", "").strip()
        if storage_root:
            storage_root_service.set(storage_root)
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
            request.form.get("phone_resolution", capture_service.camera_settings()["phone_resolution"]),
        )
        _emit_camera_status(socketio, capture_service)
        return redirect(url_for("capture_page"))

    @app.post("/capture/start")
    def start_capture_form():
        payload = _capture_payload_from_form(request.form)
        result = capture_recording_service.start(payload["subject"], payload["camera_ids"], "", _base_url())
        _emit_phone_recording_command(
            socketio,
            result.phone_command,
        )
        _emit_camera_status(socketio, capture_service)
        return redirect(url_for("capture_page"))

    @app.post("/capture/stop")
    def stop_capture_form():
        result = capture_recording_service.stop("", _base_url())
        _emit_phone_recording_command(
            socketio,
            result.phone_command,
        )
        _emit_camera_status(socketio, capture_service)
        return redirect(url_for("capture_page"))

    @app.get("/calibration")
    def calibration_page():
        phone_draft = phone_service.current_or_create_draft(_base_url())
        return render_template(
            "calibration.html",
            storage_root=storage_root_service.get(),
            cameras=capture_service.list_cameras(),
            camera_settings=capture_service.camera_settings(),
            phone_draft=phone_draft,
            active_calibration=calibration_service.active(),
            calibrations=[
                _calibration_record_to_dict(media_view_service.calibration_record_view(item))
                for item in calibration_service.list_calibrations()
            ],
        )

    @app.get("/analysis")
    def analysis_page():
        analysis_root = analysis_workspace_service.page_root(request.args.get("root"))
        return render_template(
            "analysis.html",
            storage_root=storage_root_service.get(),
            initial_root=analysis_root,
            initial_session_id=request.args.get("session_id", ""),
        )

    @app.get("/api/analysis/config")
    def api_analysis_config():
        return jsonify(analysis_workspace_service.default_config())

    @app.get("/api/analysis/sessions")
    def api_analysis_sessions():
        return jsonify([
            _session_to_dict(media_view_service.session_view(session))
            for session in analysis_workspace_service.list_sessions(request.args.get("root"))
        ])

    @app.post("/api/analysis/session-root/select")
    def api_select_analysis_session_root():
        data = request.get_json(silent=True) or {}
        selection = storage_root_service.select(str(data.get("root") or ""), bool(data.get("manual")))
        return jsonify(_root_selection_to_dict(selection, "root"))

    @app.get("/api/analysis/video")
    def api_analysis_video():
        try:
            return send_file(analysis_workspace_service.video_file(request.args.get("path") or ""), conditional=True)
        except FileNotFoundError as exc:
            return jsonify({"error": str(exc)}), 404

    @app.post("/api/analysis/calibration/upload")
    def api_upload_analysis_calibration():
        try:
            uploaded = request.files.get("calibration_file")
            return jsonify(analysis_workspace_service.upload_calibration(
                request.form.get("session_path") or "",
                uploaded.filename if uploaded else "",
                uploaded.stream if uploaded else None,
            ))
        except Exception as exc:
            return jsonify({"error": str(exc)}), 400

    @app.post("/api/analysis/run")
    def api_run_analysis():
        data = request.get_json(silent=True) or {}
        try:
            return jsonify(analysis_workspace_service.start_job(data.get("session_path") or "", data.get("config") or {}))
        except Exception as exc:
            return jsonify({"error": str(exc)}), 400

    @app.get("/api/analysis/jobs/<job_id>")
    def api_analysis_job(job_id):
        job = analysis_workspace_service.get_job(job_id)
        if job is None:
            return jsonify({"error": "job_not_found"}), 404
        return jsonify(job)

    @app.get("/api/analysis/pose3d/files")
    def api_analysis_pose3d_files():
        try:
            files = analysis_workspace_service.list_pose3d_files(request.args.get("session_path") or "")
            return jsonify({"files": files})
        except Exception as exc:
            return jsonify({"error": str(exc), "files": []}), 400

    @app.get("/api/analysis/pose3d/data")
    def api_analysis_pose3d_data():
        try:
            return jsonify(analysis_workspace_service.pose3d_data(
                request.args.get("session_path") or "",
                request.args.get("file") or "",
            ))
        except Exception as exc:
            return jsonify({"error": str(exc)}), 400

    @app.get("/api/analysis/keypoints/frame")
    def api_analysis_keypoints_frame():
        try:
            camera_label = str(request.args.get("camera_label") or "")
            frame = int(request.args.get("frame") or 0)
            return jsonify(analysis_workspace_service.keypoint_frame(
                request.args.get("session_path") or "",
                camera_label,
                frame,
            ))
        except Exception as exc:
            return jsonify({"error": str(exc)}), 400

    @app.post("/api/analysis/keypoints/frame")
    def api_save_analysis_keypoints_frame():
        try:
            data = request.get_json(silent=True) or {}
            camera_label = str(data.get("camera_label") or "")
            frame = int(data.get("frame") or 0)
            keypoints = data.get("keypoints")
            if keypoints is None:
                keypoints = data.get("people")
            analysis_workspace_service.save_keypoint_frame(
                data.get("session_path") or "",
                camera_label,
                frame,
                keypoints,
            )
            return jsonify({"status": "saved"})
        except Exception as exc:
            return jsonify({"error": str(exc)}), 400

    @app.post("/api/analysis/keypoints/render-video")
    def api_render_analysis_keypoint_video():
        try:
            data = request.get_json(silent=True) or {}
            camera_label = str(data.get("camera_label") or "")
            pose_video_path = analysis_workspace_service.render_keypoint_video(
                data.get("session_path") or "",
                camera_label,
            )
            return jsonify({"status": "rendered", "pose_video_path": str(pose_video_path)})
        except Exception as exc:
            return jsonify({"error": str(exc)}), 400

    @app.get("/api/analysis/kinematics")
    def api_analysis_kinematics():
        try:
            return jsonify(analysis_workspace_service.kinematics_summary(request.args.get("session_path") or ""))
        except Exception as exc:
            return jsonify({"error": str(exc)}), 400

    @app.get("/api/analysis/kinematics/timeseries")
    def api_analysis_kinematics_timeseries():
        try:
            signal = str(request.args.get("signal") or "").strip()
            return jsonify(analysis_workspace_service.kinematics_timeseries(
                request.args.get("session_path") or "",
                signal,
            ))
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
                str(data.get("phone_resolution", capture_service.camera_settings()["phone_resolution"])).lower(),
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
            str(data.get("phone_resolution", capture_service.camera_settings()["phone_resolution"])).lower(),
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
        return jsonify({
            "status": "finalized",
            "session": _session_to_dict(media_view_service.session_view(session)) if session else None,
        })

    @app.post("/api/phone-sessions/<session_token>/<camera_label>/upload")
    def api_upload_phone_video(session_token, camera_label):
        try:
            upload = request.files.get("video")
            result = phone_service.save_upload(
                session_token,
                camera_label,
                PhoneVideoUpload(
                    stream=upload.stream if upload else None,
                    content_type=upload.mimetype if upload else "video/mp4",
                    user_agent=request.headers.get("User-Agent", ""),
                    actual_fps=request.form.get("actual_fps"),
                    actual_width=request.form.get("actual_width"),
                    actual_height=request.form.get("actual_height"),
                ),
            )
            socketio.emit("phone_upload_complete", {"token": session_token, "camera_label": camera_label, **result})
            return jsonify(result)
        except Exception as exc:
            return jsonify({"error": str(exc)}), 400

    @app.get("/api/sessions")
    def api_sessions():
        return jsonify([_session_to_dict(media_view_service.session_view(s)) for s in capture_service.list_sessions()])

    @app.delete("/api/sessions/<session_id>")
    def api_delete_session(session_id):
        try:
            session = capture_service.delete_session(session_id)
            return jsonify({"status": "deleted", "session": _session_to_dict(media_view_service.session_view(session))})
        except Exception as exc:
            return jsonify({"error": str(exc)}), 400

    @app.post("/api/sessions/rescan")
    def api_rescan_sessions():
        return jsonify([_session_to_dict(media_view_service.session_view(s)) for s in capture_service.list_sessions()])

    @app.get("/api/settings/storage-root")
    def api_get_storage_root():
        return jsonify({"storage_root": storage_root_service.get()})

    @app.post("/api/settings/storage-root")
    def api_set_storage_root():
        data = request.get_json(silent=True) or {}
        storage_root = str(data.get("storage_root", "")).strip()
        if not storage_root:
            return jsonify({"error": "storage_root_required"}), 400
        return jsonify({"storage_root": storage_root_service.set(storage_root)})

    @app.post("/api/settings/storage-root/select")
    def api_select_storage_root():
        data = request.get_json(silent=True) or {}
        selection = storage_root_service.select(str(data.get("storage_root") or ""), bool(data.get("manual")))
        return jsonify(_root_selection_to_dict(selection, "storage_root"))

    @app.post("/api/capture/start")
    def api_start_capture():
        data = request.get_json(silent=True) or {}
        try:
            subject = _capture_subject_from_json(data)
            camera_ids = [str(v) for v in data.get("camera_ids", [])]
            result = capture_recording_service.start(
                subject,
                camera_ids,
                str(data.get("phone_session_token") or ""),
                _base_url(),
            )
            session_payload = _session_to_dict(media_view_service.session_view(result.session))
            phone_commands = _phone_recording_commands_for_cameras(
                "start",
                camera_ids,
                capture_service,
                phone_socket_sessions,
            )
            for command in phone_commands:
                phone_service.start_session(command["token"], result.session)
            _emit_phone_recording_commands(socketio, phone_commands or [result.phone_command])
            _emit_camera_status(socketio, capture_service)
            socketio.emit("capture_status", {"status": result.status, "session": session_payload})
            return jsonify(session_payload)
        except Exception as exc:
            return jsonify({"error": str(exc)}), 400

    @app.post("/api/capture/stop")
    def api_stop_capture():
        try:
            data = request.get_json(silent=True) or {}
            active_capture = capture_service.active_capture()
            active_camera_ids = active_capture.camera_ids if active_capture else []
            phone_commands = _phone_recording_commands_for_cameras(
                "stop",
                active_camera_ids,
                capture_service,
                phone_socket_sessions,
            )
            result = capture_recording_service.stop(str(data.get("phone_session_token") or ""), _base_url())
            session_payload = _session_to_dict(media_view_service.session_view(result.session))
            _emit_phone_recording_commands(socketio, phone_commands or [result.phone_command])
            _emit_camera_status(socketio, capture_service)
            socketio.emit("capture_status", {"status": result.status, "session": session_payload})
            return jsonify(session_payload)
        except Exception as exc:
            return jsonify({"error": str(exc)}), 400

    @app.post("/api/calibration/start")
    def api_start_calibration():
        data = request.get_json(silent=True) or {}
        try:
            mode = _calibration_mode(str(data.get("calibration_mode", "intrinsic")))
            result = calibration_recording_service.start(
                mode,
                str(data.get("project_name", "")),
                str(data.get("intrinsic_camera_label", "")),
                str(data.get("phone_session_token") or ""),
                _base_url(),
            )
            calibration_payload = _calibration_to_dict(result.calibration, result.status)
            phone_commands = _phone_recording_commands_for_cameras(
                "start",
                result.calibration.record_camera_ids,
                capture_service,
                phone_socket_sessions,
            )
            for command in phone_commands:
                phone_service.start_calibration(
                    command["token"],
                    result.calibration.mode,
                    result.calibration.project_name,
                    str(result.calibration.output_dir),
                    result.calibration.save_camera_labels,
                )
            _emit_phone_recording_commands(socketio, phone_commands or [result.phone_command])
            _emit_camera_status(socketio, capture_service)
            return jsonify(calibration_payload)
        except Exception as exc:
            app.logger.exception("Failed to start calibration recording")
            return jsonify({"error": str(exc)}), 400

    @app.post("/api/calibration/stop")
    def api_stop_calibration():
        data = request.get_json(silent=True) or {}
        try:
            active_calibration = calibration_service.active()
            active_camera_ids = active_calibration.record_camera_ids if active_calibration else []
            phone_commands = _phone_recording_commands_for_cameras(
                "stop",
                active_camera_ids,
                capture_service,
                phone_socket_sessions,
            )
            result = calibration_recording_service.stop(str(data.get("phone_session_token") or ""), _base_url())
            calibration_payload = _calibration_to_dict(result.calibration, result.status)
            _emit_phone_recording_commands(socketio, phone_commands or [result.phone_command])
            _emit_camera_status(socketio, capture_service)
            return jsonify(calibration_payload)
        except Exception as exc:
            return jsonify({"error": str(exc)}), 400

    @app.get("/api/calibrations")
    def api_calibrations():
        return jsonify([
            _calibration_record_to_dict(media_view_service.calibration_record_view(item))
            for item in calibration_service.list_calibrations()
        ])

    @app.delete("/api/calibrations/<folder_name>")
    def api_delete_calibration(folder_name):
        try:
            calibration = calibration_service.delete_calibration(folder_name)
            return jsonify({
                "status": "deleted",
                "calibration": _calibration_record_to_dict(media_view_service.calibration_record_view(calibration)),
            })
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
            return jsonify(calibration_service.calibration_frames(folder_name))
        except Exception as exc:
            return jsonify({"error": str(exc)}), 400

    @app.post("/api/calibrations/<folder_name>/chessboard-corners")
    def api_calibration_chessboard_corners(folder_name):
        data = request.get_json(silent=True) or {}
        try:
            return jsonify(calibration_service.chessboard_corners(folder_name, data))
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
        if camera_label:
            phone_socket_sessions[request.sid] = {
                "camera_label": camera_label,
                "token": str(payload.get("token") or ""),
            }
            capture_service.mark_phone_connected(camera_label)
            _emit_camera_status(socketio, capture_service)
        socketio.emit("phone_registered", payload)

    @socketio.on("phone_preview_status")
    def on_phone_preview_status(payload):
        camera_label = _payload_camera_label(payload)
        if camera_label:
            phone_socket_sessions[request.sid] = {
                "camera_label": camera_label,
                "token": str(payload.get("token") or ""),
            }
            if str(payload.get("status")) == "blocked":
                capture_service.mark_phone_blocked(camera_label, str(payload.get("message") or ""))
            else:
                capture_service.mark_phone_connected(camera_label)
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
        phone_session = phone_socket_sessions.pop(request.sid, {})
        camera_label = phone_session.get("camera_label", "")
        if camera_label:
            capture_service.mark_phone_disconnected(camera_label)
            _emit_camera_status(socketio, capture_service)

    return app, socketio


def _emit_camera_status(socketio: SocketIO, capture_service) -> None:
    socketio.emit("camera_status", [_camera_to_dict(c) for c in capture_service.list_cameras()])


def _emit_phone_recording_command(socketio: SocketIO, command: dict | None) -> None:
    if command:
        socketio.emit("phone_recording_command", command)


def _emit_phone_recording_commands(
    socketio: SocketIO,
    commands: list[dict | None],
) -> None:
    seen_commands: set[tuple[str, str, str]] = set()
    for command in commands:
        if not command:
            continue
        token = str(command.get("token") or "")
        camera_label = str(command.get("camera_label") or "")
        command_name = str(command.get("command") or "")
        key = (token, camera_label, command_name)
        if not token or key in seen_commands:
            continue
        seen_commands.add(key)
        _emit_phone_recording_command(socketio, command)


def _phone_recording_commands_for_cameras(
    command: str,
    camera_ids: list[str],
    capture_service,
    phone_socket_sessions: dict[str, dict[str, str]],
) -> list[dict]:
    if capture_service.camera_settings()["capture_mode"] != "phone":
        return []
    cameras_by_id = {camera.camera_id: camera for camera in capture_service.list_cameras()}
    selected_labels = {
        cameras_by_id[camera_id].label
        for camera_id in camera_ids
        if camera_id in cameras_by_id
    }
    commands: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for item in phone_socket_sessions.values():
        token = item.get("token")
        camera_label = item.get("camera_label")
        if not token or camera_label not in selected_labels or (token, camera_label) in seen:
            continue
        seen.add((token, camera_label))
        commands.append({
            "command": command,
            "token": token,
            "camera_label": camera_label,
        })
    return sorted(commands, key=lambda item: (item["camera_label"], item["token"]))


def _root_selection_to_dict(selection, root_key: str) -> dict:
    payload = {root_key: selection.root, "cancelled": selection.cancelled}
    if selection.manual_required:
        payload["manual_required"] = True
    if selection.error:
        payload["error"] = selection.error
    return payload


def _base_url() -> str:
    public_url = os.environ.get("HUMAN_3D_MOTION_PUBLIC_URL", "").strip()
    if public_url:
        return public_url.rstrip("/") + "/"
    return request.url_root


def _static_asset_url(filename: str) -> str:
    static_path = Path(__file__).resolve().parent / "static" / filename
    version = str(int(static_path.stat().st_mtime)) if static_path.is_file() else "0"
    return url_for("static", filename=filename, v=version)


def _app_logo_path() -> Path:
    return Path(__file__).resolve().parents[2] / "images" / "human-3d-motion.png"

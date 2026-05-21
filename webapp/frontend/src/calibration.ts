import { fetchJson, postJson } from "./api.js";
import type { CameraSettings, CameraStatus, PhoneDraft } from "./types.js";

declare global {
  interface Window {
    io?: () => {
      emit: (event: string, payload: unknown) => void;
      on: (event: string, callback: (payload: unknown) => void) => void;
    };
  }
}

interface PhonePreviewFrame {
  token: string;
  camera_label: string;
  image: string;
}

interface CalibrationStatus {
  calibration_id: string;
  mode: "INTR" | "EXTR";
  project_name: string;
  timestamp: string;
  status: string;
  output_dir: string;
}

interface CalibrationRecord {
  mode: "INTR" | "EXTR";
  project_name: string;
  folder_name: string;
  display_name: string;
  output_dir: string;
  updated_at: string;
  display_updated_at: string;
  videos: CalibrationVideo[];
}

interface CalibrationFramesResponse {
  folder_name: string;
  frames: CalibrationFrame[];
}

interface CalibrationFrame {
  camera_label: string;
  image: string;
}

interface CalibrationVideo {
  camera_label: string;
  path: string;
  filename: string;
  size_bytes: number;
  size_label: string;
}

interface CalibrationRunResult {
  ok: boolean;
  mode?: "INTR" | "EXTR";
  project_name?: string;
  output_path?: string;
  error?: string;
  intrinsics?: Record<string, IntrinsicCalibrationResult>;
  extrinsic?: ExtrinsicCalibrationResult;
}

interface IntrinsicCalibrationResult {
  ok: boolean;
  error?: string | null;
  rms?: number | null;
  used_frames?: number;
  frames_found?: number;
}

interface ExtrinsicCalibrationResult {
  ok: boolean;
  error?: string;
  used_frames?: number;
  common_obs?: number;
  cameras?: Record<string, ExtrinsicCameraResult>;
  reproj_rms_cam1_px?: number;
  reproj_rms_cam2_px?: number;
  reproj_rms_cam1_cm?: number;
  reproj_rms_cam2_cm?: number;
}

interface ExtrinsicCameraResult {
  reproj_rms_px?: number;
  reproj_rms_cm?: number;
  matched_points?: number;
  inliers?: number;
}

interface ObjectPoint {
  id: string | number;
  x: number;
  y: number;
  z: number;
}

interface ImagePoint {
  id: string | number;
  u: number;
  v: number;
}

interface PointCanvasState {
  cameraLabel: string;
  canvas: HTMLCanvasElement;
  ctx: CanvasRenderingContext2D;
  image: HTMLImageElement;
  points: ImagePoint[];
  scale: number;
  offsetX: number;
  offsetY: number;
  dragging: boolean;
  moved: boolean;
  lastX: number;
  lastY: number;
}

const cameraList = document.querySelector<HTMLElement>("#cameraList");
const calibrationList = document.querySelector<HTMLElement>("#calibrationList");
const captureState = document.querySelector<HTMLElement>("#captureState");
const storageRootInput = document.querySelector<HTMLInputElement>("#storageRootInput");
const selectStorageRootButton = document.querySelector<HTMLButtonElement>("[data-select-storage-root]");
const cameraSettingsForm = document.querySelector<HTMLFormElement>("[data-camera-settings-form]");
const calibrationSetupForm = document.querySelector<HTMLFormElement>("[data-calibration-setup-form]");
const calibrationDetailForm = document.querySelector<HTMLFormElement>("[data-calibration-detail-form]");
const calibrationMode = document.querySelector<HTMLSelectElement>("[data-calibration-mode]");
const calibrationBoardType = document.querySelector<HTMLSelectElement>("[data-calibration-board-type]");
const calibrationTargetSelect = document.querySelector<HTMLSelectElement>("[data-calibration-target-select]");
const intrinsicFields = document.querySelector<HTMLElement>("[data-intrinsic-fields]");
const extrinsicFields = document.querySelector<HTMLElement>("[data-extrinsic-fields]");
const boardFields = Array.from(document.querySelectorAll<HTMLElement>("[data-board-field]"));
const calibrationActionButton = document.querySelector<HTMLButtonElement>("[data-calibration-action-button]");
const calibrationActionLabel = document.querySelector<HTMLElement>("[data-calibration-action-label]");
const calibrationRecordingTimer = document.querySelector<HTMLElement>("[data-calibration-recording-timer]");
const modeInputs = Array.from(document.querySelectorAll<HTMLInputElement>("input[name='capture_mode']"));
const sonySettings = Array.from(document.querySelectorAll<HTMLElement>("[data-sony-setting]"));
const phoneSettings = Array.from(document.querySelectorAll<HTMLElement>("[data-phone-setting]"));
const phonePanel = document.querySelector<HTMLElement>("[data-phone-panel]");
const phoneQrList = document.querySelector<HTMLElement>("#phoneQrList");
const phoneTokenInput = document.querySelector<HTMLInputElement>("[data-phone-session-token-input]");
const calibrationModal = document.querySelector<HTMLElement>("[data-calibration-modal]");
const calibrationModalTitle = document.querySelector<HTMLElement>("[data-calibration-modal-title]");
const calibrationModalBody = document.querySelector<HTMLElement>("[data-calibration-modal-body]");
const calibrationModalSpinner = document.querySelector<HTMLElement>("[data-calibration-modal-spinner]");
const calibrationModalClose = document.querySelector<HTMLButtonElement>("[data-calibration-modal-close]");

let lastCameras: CameraStatus[] = [];
const phonePreviewFrames = new Map<string, string>();
let calibrationRecordingStartedAt = 0;
let calibrationRecordingTimerId = 0;
let extrinsicPointSession: {
  folderName: string;
  objectPoints: ObjectPoint[];
  intrinsicCalibration: unknown;
  canvases: PointCanvasState[];
  history: PointCanvasState[];
  status: HTMLElement;
} | null = null;

function renderState(status: string, detail: string): void {
  if (!captureState) return;
  const dotClass = status === "Recording" ? "recording" : status === "Issue" ? "warning" : "ready";
  captureState.innerHTML = `
    <span class="status-dot ${dotClass}"></span>
    <strong>${status}</strong>
    <span>${detail}</span>
  `;
}

function showCalibrationProcessingModal(): void {
  if (!calibrationModal || !calibrationModalTitle || !calibrationModalBody) return;
  calibrationModal.hidden = false;
  calibrationModalTitle.textContent = "Processing Calibration";
  calibrationModalBody.innerHTML = `<p>Processing Calibration</p>`;
  if (calibrationModalSpinner) calibrationModalSpinner.hidden = false;
  if (calibrationModalClose) calibrationModalClose.hidden = true;
}

function showCalibrationResultModal(result: CalibrationRunResult): void {
  if (!calibrationModal || !calibrationModalTitle || !calibrationModalBody) return;
  calibrationModal.hidden = false;
  if (calibrationModalSpinner) calibrationModalSpinner.hidden = true;
  if (calibrationModalClose) calibrationModalClose.hidden = false;

  if (!result.ok) {
    calibrationModalTitle.textContent = result.mode === "EXTR" ? "Extrinsic Calibration Failed" : "Intrinsic Calibration Failed";
    const intrinsics = result.intrinsics || {};
    const rows = Object.entries(intrinsics)
      .map(([cameraLabel, cameraResult]) => {
        const detail = [
          cameraResult.error || "Calibration failed",
          typeof cameraResult.frames_found === "number" ? `${cameraResult.frames_found} frames found` : "",
        ].filter(Boolean).join(" - ");
        return `
          <div class="calibration-result-row">
            <strong>${cameraLabel.toUpperCase()}</strong>
            <span>${detail}</span>
          </div>
        `;
      })
      .join("");
    calibrationModalBody.innerHTML = rows || `<p>${result.error || "Calibration failed"}</p>`;
    return;
  }

  if (result.mode === "EXTR" && result.extrinsic) {
    calibrationModalTitle.textContent = "Extrinsic Calibration Result";
    const extrinsic = result.extrinsic;
    const cameraResults: Array<[string, ExtrinsicCameraResult]> = extrinsic.cameras
      ? Object.entries(extrinsic.cameras)
      : [
          ["cam1", {
            reproj_rms_px: extrinsic.reproj_rms_cam1_px,
            reproj_rms_cm: extrinsic.reproj_rms_cam1_cm,
            matched_points: extrinsic.used_frames,
            inliers: undefined,
          }],
          ["cam2", {
            reproj_rms_px: extrinsic.reproj_rms_cam2_px,
            reproj_rms_cm: extrinsic.reproj_rms_cam2_cm,
            matched_points: extrinsic.common_obs,
            inliers: undefined,
          }],
        ];
    calibrationModalBody.innerHTML = `
      ${cameraResults.map(([cameraLabel, cameraResult]) => {
        const rmsPx = typeof cameraResult.reproj_rms_px === "number" ? `${cameraResult.reproj_rms_px.toFixed(4)} px` : "-";
        const rmsCm = typeof cameraResult.reproj_rms_cm === "number" ? `${cameraResult.reproj_rms_cm.toFixed(4)} cm` : "-";
        const detail = typeof cameraResult.inliers === "number"
          ? `${cameraResult.inliers}/${cameraResult.matched_points || 0} inliers`
          : `${cameraResult.matched_points || 0} points`;
        return `
          <div class="calibration-result-row">
            <strong>${cameraLabel.toUpperCase()}</strong>
            <span>${rmsPx} / ${rmsCm}</span>
            <small>${detail}</small>
          </div>
        `;
      }).join("")}
    `;
    return;
  }

  calibrationModalTitle.textContent = "Intrinsic Calibration Result";
  const intrinsics = result.intrinsics || {};
  const rows = Object.entries(intrinsics)
    .map(([cameraLabel, cameraResult]) => {
      const value = cameraResult.ok && typeof cameraResult.rms === "number"
        ? `${cameraResult.rms.toFixed(4)} px`
        : cameraResult.error || "Failed";
      const detail = cameraResult.ok
        ? `${cameraResult.used_frames || 0} frames`
        : "No calibration result";
      return `
        <div class="calibration-result-row">
          <strong>${cameraLabel.toUpperCase()}</strong>
          <span>${value}</span>
          <small>${detail}</small>
        </div>
      `;
    })
    .join("");
  calibrationModalBody.innerHTML = rows || `<p>No camera RMSE results found.</p>`;
}

function hideCalibrationModal(): void {
  extrinsicPointSession = null;
  if (calibrationModal) calibrationModal.hidden = true;
}

function currentPhoneSessionToken(): string {
  return phoneTokenInput?.value || phoneQrList?.dataset.phoneSessionToken || "";
}

function renderCameras(cameras: CameraStatus[]): void {
  if (!cameraList) return;
  lastCameras = cameras;
  updateCalibrationTargetOptions(cameras);
  cameraList.innerHTML = cameras
    .map((camera) => {
      const state = camera.recording ? "Recording" : camera.connected ? "Ready" : "Offline";
      const recordingClass = camera.recording ? " is-recording" : "";
      const liveViewUrl = camera.live_view_url || "about:blank";
      const phonePreviewFrame = phonePreviewFrames.get(camera.label);
      const liveView = camera.connected && camera.live_view_url
        ? `<iframe src="${liveViewUrl}" title="${camera.label} live view"></iframe>`
        : phonePreviewFrame
          ? `<img class="live-view-phone-preview" data-camera-label="${camera.label}" src="${phonePreviewFrame}" alt="${camera.label} phone preview">`
          : `<span class="live-view-message"><strong>${camera.label}</strong><span>${camera.last_error || state}</span></span>`;
      const selectable = camera.connected ? "checked" : "disabled";
      return `
        <label class="camera-tile${recordingClass}">
          <input class="camera-select" type="checkbox" name="camera_ids" value="${camera.camera_id}" ${selectable}>
          <span class="live-view-frame">${liveView}</span>
          <span class="camera-meta"><span class="camera-index">${camera.label}</span></span>
        </label>
      `;
    })
    .join("");
}

function renderCalibrations(calibrations: CalibrationRecord[]): void {
  if (!calibrationList) return;
  if (calibrations.length === 0) {
    calibrationList.innerHTML = `<p class="empty">No calibration folders yet.</p>`;
    return;
  }
  calibrationList.innerHTML = calibrations
    .map(
      (calibration) => {
        const videos = calibration.videos.length > 0
          ? calibration.videos
              .map(
                (video) => `
                  <div class="session-video-item">
                    <span class="session-camera-tag">${video.camera_label.toUpperCase()}</span>
                    <div>
                      <strong>${video.filename}</strong>
                      <small>${video.size_label}</small>
                    </div>
                  </div>
                `,
              )
              .join("")
          : `
            <div class="session-video-item">
              <span class="session-camera-tag">${calibration.mode}</span>
              <div>
                <strong>No videos found</strong>
                <small>${calibration.output_dir}</small>
              </div>
            </div>
          `;
        return `
        <article class="session-card calibration-card">
          <header class="session-card-header">
            <div class="session-subject">
              <strong>${calibration.display_name}</strong>
              <span>${calibration.mode}</span>
            </div>
            <div class="session-actions">
              <time>${calibration.display_updated_at || calibration.updated_at}</time>
              <button class="session-analyze-button" type="button" data-run-calibration-folder="${calibration.folder_name}" data-calibration-record-mode="${calibration.mode}">Calibration</button>
              <button class="session-delete-button" type="button" data-delete-calibration-folder="${calibration.folder_name}">Delete</button>
            </div>
          </header>
          <div class="session-video-list">${videos}</div>
        </article>
      `;
      },
    )
    .join("");
}

function updateCalibrationTargetOptions(cameras: CameraStatus[]): void {
  if (!calibrationTargetSelect) return;
  const current = calibrationTargetSelect.value;
  calibrationTargetSelect.innerHTML = [
    ...cameras.map((camera) => `<option value="intrinsic:${camera.label}">intrinsic ${camera.label.toLowerCase()}</option>`),
    `<option value="extrinsic">extrinsic</option>`,
  ].join("");
  if (current && Array.from(calibrationTargetSelect.options).some((option) => option.value === current)) {
    calibrationTargetSelect.value = current;
  }
  syncCalibrationMode();
}

function renderPhonePreviewFrame(frame: PhonePreviewFrame): void {
  if (frame.token !== currentPhoneSessionToken()) return;
  phonePreviewFrames.set(frame.camera_label, frame.image);
  const image = cameraList?.querySelector<HTMLImageElement>(
    `.live-view-phone-preview[data-camera-label="${frame.camera_label}"]`,
  );
  if (image) {
    image.src = frame.image;
    return;
  }
  if (lastCameras.some((camera) => camera.label === frame.camera_label)) {
    renderCameras(lastCameras);
  }
}

function currentCaptureMode(): "sony" | "phone" {
  return modeInputs.find((input) => input.checked)?.value === "phone" ? "phone" : "sony";
}

function applyCaptureMode(mode: "sony" | "phone"): void {
  sonySettings.forEach((element) => {
    element.hidden = mode !== "sony";
  });
  phoneSettings.forEach((element) => {
    element.hidden = mode !== "phone";
  });
  if (phonePanel) phonePanel.hidden = mode !== "phone";
  modeInputs.forEach((input) => {
    input.checked = input.value === mode;
    input.closest("label")?.classList.toggle("is-active", input.value === mode);
  });
}

function renderPhoneDraft(draft: PhoneDraft): void {
  if (!phoneQrList) return;
  phoneQrList.dataset.phoneSessionToken = draft.token;
  if (phoneTokenInput) phoneTokenInput.value = draft.token;
  phoneQrList.innerHTML = draft.slots
    .map(
      (slot) => `
        <article class="phone-qr-card">
          <img src="${slot.qr_data_url}" alt="${slot.camera_label} QR code">
          <div>
            <strong>${slot.camera_label}</strong>
            <span>Scan to pair phone</span>
          </div>
        </article>
      `,
    )
    .join("");
}

async function refreshPhoneDraft(): Promise<void> {
  const draft = await postJson<PhoneDraft>("/api/phone-sessions", {});
  renderPhoneDraft(draft);
}

async function refreshCameras(): Promise<void> {
  const cameras = await fetchJson<CameraStatus[]>("/api/cameras");
  renderCameras(cameras);
}

async function refreshCalibrations(): Promise<void> {
  const calibrations = await fetchJson<CalibrationRecord[]>("/api/calibrations");
  renderCalibrations(calibrations);
}

async function deleteCalibration(folderName: string): Promise<void> {
  const response = await fetch(`/api/calibrations/${encodeURIComponent(folderName)}`, {
    method: "DELETE",
  });
  if (!response.ok) {
    throw new Error(await response.text());
  }
  await refreshCalibrations();
}

async function runCalibration(folderName: string, extraPayload: Record<string, unknown> = {}): Promise<CalibrationRunResult> {
  const payload = await calibrationDetailPayload();
  if (!payload) return { ok: false, error: "calibration_setup_required" };
  const result = await postJson<CalibrationRunResult>(
    `/api/calibrations/${encodeURIComponent(folderName)}/run`,
    { ...payload, ...extraPayload },
  );
  await refreshCalibrations();
  return result;
}

async function fetchCalibrationFrames(folderName: string): Promise<CalibrationFramesResponse> {
  return fetchJson<CalibrationFramesResponse>(`/api/calibrations/${encodeURIComponent(folderName)}/frames`);
}

async function refreshCameraSettings(): Promise<void> {
  const settings = await fetchJson<CameraSettings>("/api/settings/cameras");
  applyCaptureMode(settings.capture_mode);
}

async function selectStorageRoot(): Promise<void> {
  let response = await postJson<{ storage_root: string; cancelled: boolean; manual_required?: boolean }>(
    "/api/settings/storage-root/select",
    {},
  );
  if (response.manual_required) {
    const manualPath = window.prompt("Enter storage path", storageRootInput?.value || response.storage_root || "");
    if (!manualPath) return;
    response = await postJson<{ storage_root: string; cancelled: boolean }>("/api/settings/storage-root/select", {
      storage_root: manualPath,
      manual: true,
    });
  }
  if (storageRootInput) storageRootInput.value = response.storage_root;
  if (!response.cancelled) renderState("Ready", "Storage path updated");
}

async function applyCameraSettings(): Promise<void> {
  if (!cameraSettingsForm || !cameraSettingsForm.reportValidity()) return;
  const data = new FormData(cameraSettingsForm);
  const settings = await postJson<CameraSettings>("/api/settings/cameras", {
    capture_mode: String(data.get("capture_mode") || currentCaptureMode()),
    camera_count: Number(data.get("camera_count") || 1),
    ccb_url: String(data.get("ccb_url") || ""),
    live_view_frame_rate: String(data.get("live_view_frame_rate") || "low"),
    phone_camera_count: Number(data.get("phone_camera_count") || 1),
    phone_frame_rate: Number(data.get("phone_frame_rate") || 120),
  });
  applyCaptureMode(settings.capture_mode);
  await refreshCameras();
  await refreshPhoneDraft();
  renderState("Ready", "Camera settings applied");
}

function setCalibrationRecording(recording: boolean): void {
  if (!calibrationActionButton) return;
  calibrationActionButton.dataset.recording = recording ? "true" : "false";
  if (calibrationActionLabel) {
    calibrationActionLabel.textContent = recording ? "Stop" : "Record";
  }
  calibrationActionButton.classList.toggle("record", !recording);
  calibrationActionButton.classList.toggle("stop", recording);
  if (recording) {
    startCalibrationRecordingTimer();
  } else {
    stopCalibrationRecordingTimer();
  }
}

function formatCalibrationRecordingElapsed(elapsedMs: number): string {
  const totalSeconds = Math.max(0, Math.floor(elapsedMs / 1000));
  const hours = Math.floor(totalSeconds / 3600);
  const minutes = Math.floor((totalSeconds % 3600) / 60);
  const seconds = totalSeconds % 60;
  return [hours, minutes, seconds].map((value) => String(value).padStart(2, "0")).join(":");
}

function updateCalibrationRecordingTimer(): void {
  if (!calibrationRecordingTimer) return;
  const elapsed = calibrationRecordingStartedAt > 0 ? Date.now() - calibrationRecordingStartedAt : 0;
  calibrationRecordingTimer.textContent = formatCalibrationRecordingElapsed(elapsed);
}

function startCalibrationRecordingTimer(): void {
  if (calibrationRecordingStartedAt === 0) {
    calibrationRecordingStartedAt = Date.now();
  }
  updateCalibrationRecordingTimer();
  if (calibrationRecordingTimerId) return;
  calibrationRecordingTimerId = window.setInterval(updateCalibrationRecordingTimer, 1000);
}

function stopCalibrationRecordingTimer(): void {
  if (calibrationRecordingTimerId) {
    window.clearInterval(calibrationRecordingTimerId);
    calibrationRecordingTimerId = 0;
  }
  calibrationRecordingStartedAt = 0;
  updateCalibrationRecordingTimer();
}

function errorMessage(error: unknown): string {
  if (!(error instanceof Error)) return "Request failed";
  try {
    const payload = JSON.parse(error.message) as { error?: string };
    return payload.error || error.message;
  } catch {
    return error.message;
  }
}

async function calibrationPayload(): Promise<Record<string, unknown> | null> {
  if (!calibrationSetupForm || !calibrationSetupForm.reportValidity()) return null;
  const data = new FormData(calibrationSetupForm);
  const detailPayload = await calibrationDetailPayload();
  if (!detailPayload) return null;
  const target = calibrationTargetSelect?.value || "extrinsic";
  const intrinsic = target.startsWith("intrinsic:");
  const intrinsicCameraLabel = intrinsic ? target.slice("intrinsic:".length) : "";
  return {
    project_name: String(data.get("project_name") || ""),
    calibration_mode: intrinsic ? "intrinsic" : "extrinsic",
    checker_board_type: detailPayload.checker_board_type,
    aruco_dictionary: detailPayload.aruco_dictionary,
    checker_board_size_mm: detailPayload.checker_board_size_mm,
    marker_size_mm: detailPayload.marker_size_mm,
    checker_board_columns: detailPayload.checker_board_columns,
    checker_board_rows: detailPayload.checker_board_rows,
    intrinsic_camera_label: intrinsicCameraLabel,
    object_points: detailPayload.object_points,
    phone_session_token: currentPhoneSessionToken(),
  };
}

async function calibrationDetailPayload(): Promise<Record<string, unknown> | null> {
  if (!calibrationDetailForm || !calibrationDetailForm.reportValidity()) return null;
  const detailData = new FormData(calibrationDetailForm);
  const file = detailData.get("intrinsic_calibration_file");
  const intrinsicCalibration = file instanceof File && file.size > 0
    ? JSON.parse(await file.text()) as unknown
    : null;
  return {
    calibration_mode: String(detailData.get("calibration_mode") || "intrinsic"),
    checker_board_type: String(detailData.get("checker_board_type") || "chessboard"),
    aruco_dictionary: String(detailData.get("aruco_dictionary") || "DICT_4X4_50"),
    checker_board_size_mm: Number(detailData.get("checker_board_size_mm") || 0),
    marker_size_mm: Number(detailData.get("marker_size_mm") || 0),
    checker_board_columns: Number(detailData.get("checker_board_columns") || 0),
    checker_board_rows: Number(detailData.get("checker_board_rows") || 0),
    object_points: String(detailData.get("object_points") || ""),
    intrinsic_calibration: intrinsicCalibration,
  };
}

async function startCalibration(): Promise<void> {
  const payload = await calibrationPayload();
  if (!payload) return;
  const calibration = await postJson<CalibrationStatus>("/api/calibration/start", payload);
  setCalibrationRecording(true);
  renderState("Recording", `${calibration.mode} ${calibration.project_name}`);
  await refreshCameras();
}

async function stopCalibration(): Promise<void> {
  const calibration = await postJson<CalibrationStatus>("/api/calibration/stop", {
    phone_session_token: currentPhoneSessionToken(),
  });
  setCalibrationRecording(false);
  renderState("Ready", `Saved to ${calibration.output_dir}`);
  await refreshCameras();
  await refreshCalibrations();
}

function parseObjectPoints(value: string): ObjectPoint[] {
  const points: ObjectPoint[] = [];
  value.split(/\r?\n/).forEach((line, index) => {
    const parts = line.split(",").map((part) => part.trim()).filter(Boolean);
    const point = parts.length === 3
      ? { id: index, x: Number(parts[0]), y: Number(parts[1]), z: Number(parts[2]) }
      : parts.length >= 4
        ? { id: parts[0], x: Number(parts[1]), y: Number(parts[2]), z: Number(parts[3]) }
        : null;
    if (point && Number.isFinite(point.x) && Number.isFinite(point.y) && Number.isFinite(point.z)) {
      points.push(point);
    }
  });
  return points;
}

async function openExtrinsicPointModal(folderName: string): Promise<void> {
  const detailPayload = await calibrationDetailPayload();
  if (!detailPayload) return;
  const objectPoints = parseObjectPoints(String(detailPayload.object_points || ""));
  if (objectPoints.length < 6) {
    showCalibrationResultModal({ ok: false, mode: "EXTR", error: `need_at_least_6_object_points: ${objectPoints.length}` });
    return;
  }
  if (!detailPayload.intrinsic_calibration) {
    showCalibrationResultModal({ ok: false, mode: "EXTR", error: "intrinsic_calibration_upload_required" });
    return;
  }
  const frames = await fetchCalibrationFrames(folderName);
  showExtrinsicPointModal(folderName, frames.frames, objectPoints, detailPayload.intrinsic_calibration);
}

function showExtrinsicPointModal(folderName: string, frames: CalibrationFrame[], objectPoints: ObjectPoint[], intrinsicCalibration: unknown): void {
  if (!calibrationModal || !calibrationModalTitle || !calibrationModalBody) return;
  extrinsicPointSession = null;
  calibrationModal.hidden = false;
  calibrationModalTitle.textContent = "Extrinsic Point Selection";
  if (calibrationModalSpinner) calibrationModalSpinner.hidden = true;
  if (calibrationModalClose) calibrationModalClose.hidden = false;
  calibrationModalBody.innerHTML = `
    <div class="extrinsic-point-toolbar">
      <strong data-extrinsic-point-status></strong>
      <div>
        <button class="button secondary" type="button" data-extrinsic-reset>Reset View</button>
        <button class="button secondary" type="button" data-extrinsic-undo>Undo Point</button>
        <button class="button record" type="button" data-extrinsic-submit>Calibration</button>
      </div>
    </div>
    <div class="extrinsic-point-grid">
      ${frames.slice(0, 4).map((frame, index) => `
        <div class="extrinsic-point-panel">
          <strong>${frame.camera_label.toUpperCase()}</strong>
          <canvas data-extrinsic-canvas="${index}" width="640" height="420"></canvas>
        </div>
      `).join("")}
    </div>
  `;
  const status = calibrationModalBody.querySelector<HTMLElement>("[data-extrinsic-point-status]");
  if (!status) return;
  const canvases: PointCanvasState[] = [];
  extrinsicPointSession = { folderName, objectPoints, intrinsicCalibration, canvases, history: [], status };
  frames.slice(0, 4).forEach((frame, index) => {
    const canvas = calibrationModalBody.querySelector<HTMLCanvasElement>(`[data-extrinsic-canvas="${index}"]`);
    const ctx = canvas?.getContext("2d");
    if (!canvas || !ctx) return;
    const image = new Image();
    const state: PointCanvasState = {
      cameraLabel: frame.camera_label,
      canvas,
      ctx,
      image,
      points: [],
      scale: 1,
      offsetX: 0,
      offsetY: 0,
      dragging: false,
      moved: false,
      lastX: 0,
      lastY: 0,
    };
    canvases.push(state);
    bindPointCanvas(state);
    image.addEventListener("load", () => {
      fitPointCanvas(state);
      drawPointCanvas(state);
      updateExtrinsicPointStatus();
    });
    image.src = frame.image;
  });
  calibrationModalBody.querySelector<HTMLButtonElement>("[data-extrinsic-reset]")?.addEventListener("click", () => {
    extrinsicPointSession?.canvases.forEach((state) => {
      fitPointCanvas(state);
      drawPointCanvas(state);
    });
  });
  calibrationModalBody.querySelector<HTMLButtonElement>("[data-extrinsic-undo]")?.addEventListener("click", () => {
    const target = extrinsicPointSession?.history.pop();
    target?.points.pop();
    if (target) drawPointCanvas(target);
    updateExtrinsicPointStatus();
  });
  calibrationModalBody.querySelector<HTMLButtonElement>("[data-extrinsic-submit]")?.addEventListener("click", submitExtrinsicPointCalibration);
  updateExtrinsicPointStatus();
}

function fitPointCanvas(state: PointCanvasState): void {
  const scale = Math.min(state.canvas.width / state.image.width, state.canvas.height / state.image.height);
  state.scale = scale;
  state.offsetX = (state.canvas.width - state.image.width * scale) / 2;
  state.offsetY = (state.canvas.height - state.image.height * scale) / 2;
}

function bindPointCanvas(state: PointCanvasState): void {
  state.canvas.addEventListener("contextmenu", (event) => {
    event.preventDefault();
  });
  state.canvas.addEventListener("wheel", (event) => {
    event.preventDefault();
    const { x, y } = eventCanvasPoint(state, event);
    const before = canvasToImagePoint(state, x, y);
    const factor = event.deltaY < 0 ? 1.15 : 0.87;
    state.scale = Math.min(8, Math.max(0.1, state.scale * factor));
    state.offsetX = x - before.u * state.scale;
    state.offsetY = y - before.v * state.scale;
    drawPointCanvas(state);
  }, { passive: false });
  state.canvas.addEventListener("pointerdown", (event) => {
    event.preventDefault();
    state.dragging = true;
    state.moved = false;
    const point = eventCanvasPoint(state, event);
    state.lastX = point.x;
    state.lastY = point.y;
    state.canvas.setPointerCapture(event.pointerId);
  });
  state.canvas.addEventListener("pointermove", (event) => {
    if (!state.dragging) return;
    const point = eventCanvasPoint(state, event);
    const dx = point.x - state.lastX;
    const dy = point.y - state.lastY;
    if (Math.abs(dx) + Math.abs(dy) > 2) state.moved = true;
    state.offsetX += dx;
    state.offsetY += dy;
    state.lastX = point.x;
    state.lastY = point.y;
    drawPointCanvas(state);
  });
  state.canvas.addEventListener("pointerup", (event) => {
    event.preventDefault();
    state.dragging = false;
    state.canvas.releasePointerCapture(event.pointerId);
    if (state.moved || event.button !== 0) return;
    const session = extrinsicPointSession;
    if (!session || state.points.length >= session.objectPoints.length) return;
    const canvasPoint = eventCanvasPoint(state, event);
    const point = canvasToImagePoint(state, canvasPoint.x, canvasPoint.y);
    if (point.u < 0 || point.v < 0 || point.u > state.image.width || point.v > state.image.height) return;
    state.points.push({ id: session.objectPoints[state.points.length].id, u: point.u, v: point.v });
    session.history.push(state);
    drawPointCanvas(state);
    updateExtrinsicPointStatus();
  });
}

function eventCanvasPoint(state: PointCanvasState, event: PointerEvent | WheelEvent): { x: number; y: number } {
  const rect = state.canvas.getBoundingClientRect();
  return {
    x: (event.clientX - rect.left) * (state.canvas.width / rect.width),
    y: (event.clientY - rect.top) * (state.canvas.height / rect.height),
  };
}

function canvasToImagePoint(state: PointCanvasState, x: number, y: number): ImagePoint {
  return { id: "", u: (x - state.offsetX) / state.scale, v: (y - state.offsetY) / state.scale };
}

function drawPointCanvas(state: PointCanvasState): void {
  state.ctx.clearRect(0, 0, state.canvas.width, state.canvas.height);
  state.ctx.fillStyle = "#05080a";
  state.ctx.fillRect(0, 0, state.canvas.width, state.canvas.height);
  state.ctx.drawImage(state.image, state.offsetX, state.offsetY, state.image.width * state.scale, state.image.height * state.scale);
  state.points.forEach((point, index) => {
    const x = state.offsetX + point.u * state.scale;
    const y = state.offsetY + point.v * state.scale;
    state.ctx.beginPath();
    state.ctx.arc(x, y, 5, 0, Math.PI * 2);
    state.ctx.fillStyle = "#d7ff43";
    state.ctx.fill();
    state.ctx.lineWidth = 2;
    state.ctx.strokeStyle = "#05080a";
    state.ctx.stroke();
    state.ctx.fillStyle = "#ffffff";
    state.ctx.font = "700 13px system-ui";
    state.ctx.fillText(String(point.id || index), x + 8, y - 8);
  });
}

function updateExtrinsicPointStatus(): void {
  const session = extrinsicPointSession;
  if (!session) return;
  const counts = session.canvases.map((state) => state.points.length);
  const next = Math.min(...counts);
  const nextPoint = session.objectPoints[next];
  const cameraCounts = session.canvases
    .map((state) => `${state.cameraLabel.toUpperCase()} ${state.points.length}/${session.objectPoints.length}`)
    .join(", ");
  session.status.textContent = nextPoint
    ? `Next point: ${nextPoint.id} (${nextPoint.x}, ${nextPoint.y}, ${nextPoint.z}) - ${cameraCounts}`
    : "All points selected";
}

async function submitExtrinsicPointCalibration(): Promise<void> {
  const session = extrinsicPointSession;
  if (!session || session.canvases.length < 2) return;
  if (session.canvases.some((state) => state.points.length < session.objectPoints.length)) {
    updateExtrinsicPointStatus();
    return;
  }
  const imagePointsByCamera = Object.fromEntries(
    session.canvases.map((state) => [state.cameraLabel, state.points]),
  );
  const [cam1, cam2] = session.canvases;
  showCalibrationProcessingModal();
  const result = await runCalibration(session.folderName, {
    intrinsic_calibration: session.intrinsicCalibration,
    object_points: session.objectPoints,
    image_points_by_camera: imagePointsByCamera,
    image_points_cam1: cam1?.points || [],
    image_points_cam2: cam2?.points || [],
  });
  showCalibrationResultModal(result);
  if (!result.ok) {
    renderState("Issue", result.error || "Extrinsic calibration failed");
  } else {
    renderState("Ready", `Calibration saved to ${result.output_path || session.folderName}`);
  }
}

function syncCalibrationMode(): void {
  const intrinsic = calibrationMode?.value !== "extrinsic";
  if (intrinsicFields) intrinsicFields.hidden = !intrinsic;
  if (extrinsicFields) extrinsicFields.hidden = intrinsic;
  syncBoardFields();
}

function syncBoardFields(): void {
  const boardType = calibrationBoardType?.value || "chessboard";
  boardFields.forEach((field) => {
    const allowedTypes = String(field.dataset.boardField || "").split(/\s+/).filter(Boolean);
    field.hidden = calibrationMode?.value === "extrinsic" || !allowedTypes.includes(boardType);
  });
}

document.querySelector<HTMLFormElement>("[data-storage-form]")?.addEventListener("submit", (event) => {
  event.preventDefault();
});

selectStorageRootButton?.addEventListener("click", async () => {
  selectStorageRootButton.disabled = true;
  try {
    await selectStorageRoot();
  } catch (error) {
    renderState("Issue", errorMessage(error));
  } finally {
    selectStorageRootButton.disabled = false;
  }
});

cameraSettingsForm?.addEventListener("submit", async (event) => {
  event.preventDefault();
  const button = cameraSettingsForm.querySelector<HTMLButtonElement>("button[type='submit']");
  if (button) button.disabled = true;
  try {
    await applyCameraSettings();
  } catch (error) {
    renderState("Issue", errorMessage(error));
  } finally {
    if (button) button.disabled = false;
  }
});

calibrationMode?.addEventListener("change", syncCalibrationMode);
calibrationBoardType?.addEventListener("change", syncBoardFields);

calibrationActionButton?.addEventListener("click", async () => {
  calibrationActionButton.disabled = true;
  try {
    if (calibrationActionButton.dataset.recording === "true") {
      await stopCalibration();
    } else {
      await startCalibration();
    }
  } catch (error) {
    renderState("Issue", errorMessage(error));
  } finally {
    calibrationActionButton.disabled = false;
  }
});

calibrationList?.addEventListener("click", async (event) => {
  const target = event.target as HTMLElement;
  const runButton = target.closest<HTMLButtonElement>("[data-run-calibration-folder]");
  if (runButton) {
    const folderName = runButton.dataset.runCalibrationFolder;
    if (!folderName) return;
    runButton.disabled = true;
    runButton.textContent = "Running";
    try {
      if (runButton.dataset.calibrationRecordMode === "EXTR") {
        await openExtrinsicPointModal(folderName);
      } else {
        showCalibrationProcessingModal();
        const result = await runCalibration(folderName);
        showCalibrationResultModal(result);
        if (!result.ok) {
          renderState("Issue", result.error || "Calibration failed");
        } else {
          renderState("Ready", `Calibration saved to ${result.output_path || folderName}`);
        }
      }
    } catch (error) {
      renderState("Issue", errorMessage(error));
    } finally {
      runButton.disabled = false;
      runButton.textContent = "Calibration";
    }
    return;
  }

  const button = target.closest<HTMLButtonElement>("[data-delete-calibration-folder]");
  if (!button) return;
  const folderName = button.dataset.deleteCalibrationFolder;
  if (!folderName) return;
  button.disabled = true;
  try {
    await deleteCalibration(folderName);
    renderState("Ready", "Calibration deleted");
  } catch (error) {
    renderState("Issue", errorMessage(error));
    button.disabled = false;
  }
});

calibrationModalClose?.addEventListener("click", hideCalibrationModal);

modeInputs.forEach((input) => {
  input.addEventListener("change", () => {
    const mode = currentCaptureMode();
    applyCaptureMode(mode);
    postJson<CameraSettings>("/api/capture/mode", { capture_mode: mode })
      .then(() => Promise.all([refreshCameras(), mode === "phone" ? refreshPhoneDraft() : Promise.resolve()]))
      .catch((error) => renderState("Issue", errorMessage(error)));
  });
});

if (window.io) {
  const socket = window.io();
  socket.on("camera_status", (payload) => renderCameras(payload as CameraStatus[]));
  socket.on("phone_preview_frame", (payload) => renderPhonePreviewFrame(payload as PhonePreviewFrame));
}

syncCalibrationMode();
refreshCameras().catch((error) => renderState("Issue", errorMessage(error)));
refreshCalibrations().catch((error) => renderState("Issue", errorMessage(error)));
refreshCameraSettings().catch(() => applyCaptureMode(currentCaptureMode()));

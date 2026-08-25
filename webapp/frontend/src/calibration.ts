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

interface PhoneRegistration {
  token: string;
  camera_label: string;
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
  frame_index?: number;
  corners?: ImagePoint[];
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

type ExtrinsicPointMode = "object" | "chessboard";

interface PointHistoryEntry {
  snapshots: Array<{
    state: PointCanvasState;
    points: ImagePoint[];
  }>;
}

interface PointCanvasState {
  cameraLabel: string;
  frameIndex: number;
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
const extrinsicTargetInputs = Array.from(document.querySelectorAll<HTMLInputElement>("[data-extrinsic-target-option]"));
const extrinsicObjectFields = Array.from(document.querySelectorAll<HTMLElement>("[data-extrinsic-object-fields]"));
const extrinsicChessboardFields = Array.from(document.querySelectorAll<HTMLElement>("[data-extrinsic-chessboard-fields]"));
const extrinsicBoardType = document.querySelector<HTMLSelectElement>("[data-extrinsic-calibration-board-type]");
const extrinsicBoardFields = Array.from(document.querySelectorAll<HTMLElement>("[data-extrinsic-board-field]"));
const boardFields = Array.from(document.querySelectorAll<HTMLElement>("[data-board-field]"));
const boardCountHints = Array.from(document.querySelectorAll<HTMLElement>("[data-board-count-hint]"));
const calibrationActionButton = document.querySelector<HTMLButtonElement>("[data-calibration-action-button]");
const calibrationActionLabel = document.querySelector<HTMLElement>("[data-calibration-action-label]");
const calibrationRecordingTimer = document.querySelector<HTMLElement>("[data-calibration-recording-timer]");
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
  mode: ExtrinsicPointMode;
  boardType: string;
  chessboardOrientation: string;
  objectPoints: ObjectPoint[];
  intrinsicCalibration: unknown;
  canvases: PointCanvasState[];
  history: PointHistoryEntry[];
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

function adoptPhoneSessionToken(token: string): void {
  const normalized = String(token || "").trim();
  if (!normalized || normalized === currentPhoneSessionToken()) return;
  if (phoneQrList) phoneQrList.dataset.phoneSessionToken = normalized;
  if (phoneTokenInput) phoneTokenInput.value = normalized;
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
  adoptPhoneSessionToken(frame.token);
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

function confirmCalibrationDelete(): Promise<boolean> {
  return new Promise((resolve) => {
    const overlay = document.createElement("div");
    overlay.className = "confirm-modal-overlay";
    overlay.innerHTML = `
      <div class="confirm-modal" role="dialog" aria-modal="true" aria-labelledby="deleteCalibrationTitle">
        <h2 id="deleteCalibrationTitle">Delete calibration?</h2>
        <p>This action cannot be undone.</p>
        <div class="confirm-modal-actions">
          <button class="button secondary" type="button" data-confirm-cancel>Cancel</button>
          <button class="button confirm-danger" type="button" data-confirm-delete>Delete</button>
        </div>
      </div>
    `;

    const close = (confirmed: boolean) => {
      document.removeEventListener("keydown", handleKeydown);
      overlay.remove();
      resolve(confirmed);
    };
    const handleKeydown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        close(false);
      }
    };

    overlay.addEventListener("click", (event) => {
      if (event.target === overlay) {
        close(false);
      }
    });
    overlay.querySelector<HTMLButtonElement>("[data-confirm-cancel]")?.addEventListener("click", () => close(false));
    overlay.querySelector<HTMLButtonElement>("[data-confirm-delete]")?.addEventListener("click", () => close(true));

    document.addEventListener("keydown", handleKeydown);
    document.body.appendChild(overlay);
    overlay.querySelector<HTMLButtonElement>("[data-confirm-delete]")?.focus();
  });
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

async function fetchChessboardCorners(
  folderName: string,
  payload: Record<string, unknown>,
): Promise<CalibrationFramesResponse> {
  return postJson<CalibrationFramesResponse>(
    `/api/calibrations/${encodeURIComponent(folderName)}/chessboard-corners`,
    payload,
  );
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
  await postJson<CameraSettings>("/api/settings/cameras", {
    capture_mode: "phone",
    phone_camera_count: Number(data.get("phone_camera_count") || 1),
    phone_frame_rate: Number(data.get("phone_frame_rate") || 60),
    phone_resolution: String(data.get("phone_resolution") || "720"),
  });
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

function currentExtrinsicTarget(): ExtrinsicPointMode {
  return extrinsicTargetInputs.find((input) => input.checked)?.value === "chessboard" ? "chessboard" : "object";
}

function detailFormValue(data: FormData, name: string): FormDataEntryValue | null {
  if (calibrationMode?.value === "extrinsic" && currentExtrinsicTarget() === "chessboard") {
    const field = calibrationDetailForm?.querySelector<HTMLInputElement | HTMLSelectElement>(
      `[data-extrinsic-chessboard-fields] [name='${name}']`,
    );
    if (field) return field.value;
  }
  return data.get(name);
}

function copyIntrinsicBoardFieldsToExtrinsic(): void {
  [
    "checker_board_type",
    "aruco_dictionary",
    "checker_board_size_mm",
    "marker_size_mm",
    "checker_board_columns",
    "checker_board_rows",
  ].forEach((name) => {
    const source = intrinsicFields?.querySelector<HTMLInputElement | HTMLSelectElement>(`[name='${name}']`);
    const target = calibrationDetailForm?.querySelector<HTMLInputElement | HTMLSelectElement>(
      `[data-extrinsic-chessboard-fields] [name='${name}']`,
    );
    if (source && target) target.value = source.value;
  });
  syncExtrinsicBoardFields();
}

async function calibrationPayload(): Promise<Record<string, unknown> | null> {
  if (!calibrationSetupForm || !calibrationSetupForm.reportValidity()) return null;
  const data = new FormData(calibrationSetupForm);
  const target = calibrationTargetSelect?.value || "extrinsic";
  const intrinsic = target.startsWith("intrinsic:");
  const intrinsicCameraLabel = intrinsic ? target.slice("intrinsic:".length) : "";
  return {
    project_name: String(data.get("project_name") || ""),
    calibration_mode: intrinsic ? "intrinsic" : "extrinsic",
    intrinsic_camera_label: intrinsicCameraLabel,
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
  const extrinsicTarget = currentExtrinsicTarget();
  const checkerBoardType = extrinsicTarget === "chessboard"
    ? String(detailFormValue(detailData, "checker_board_type") || "chessboard")
    : String(detailData.get("checker_board_type") || "chessboard");
  return {
    calibration_mode: String(detailData.get("calibration_mode") || "intrinsic"),
    extrinsic_calibration_target: extrinsicTarget,
    checker_board_type: checkerBoardType,
    aruco_dictionary: String(detailFormValue(detailData, "aruco_dictionary") || "DICT_4X4_50"),
    checker_board_size_mm: Number(detailFormValue(detailData, "checker_board_size_mm") || 0),
    marker_size_mm: Number(detailFormValue(detailData, "marker_size_mm") || 0),
    checker_board_columns: Number(detailFormValue(detailData, "checker_board_columns") || 0),
    checker_board_rows: Number(detailFormValue(detailData, "checker_board_rows") || 0),
    chessboard_orientation: String(detailFormValue(detailData, "chessboard_orientation") || "horizontal"),
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

function chessboardObjectPoints(columns: number, rows: number, squareSizeMm: number, orientation: string): ObjectPoint[] {
  const squareSizeM = squareSizeMm / 1000;
  const vertical = orientation === "vertical";
  const points: ObjectPoint[] = [];
  for (let row = 0; row < rows; row += 1) {
    for (let column = 0; column < columns; column += 1) {
      const id = points.length + 1;
      points.push({
        id,
        x: vertical ? 0 : -column * squareSizeM,
        y: row * squareSizeM,
        z: vertical ? column * squareSizeM : 0,
      });
    }
  }
  return points;
}

async function openExtrinsicPointModal(folderName: string): Promise<void> {
  const detailPayload = await calibrationDetailPayload();
  if (!detailPayload) return;
  if (detailPayload.extrinsic_calibration_target === "chessboard") {
    await openExtrinsicChessboardModal(folderName, detailPayload);
    return;
  }
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
  showExtrinsicPointModal(folderName, frames.frames, objectPoints, detailPayload.intrinsic_calibration, "object", "", "");
}

async function openExtrinsicChessboardModal(folderName: string, detailPayload: Record<string, unknown>): Promise<void> {
  if (!detailPayload.intrinsic_calibration) {
    showCalibrationResultModal({ ok: false, mode: "EXTR", error: "intrinsic_calibration_upload_required" });
    return;
  }
  const columns = Number(detailPayload.checker_board_columns || 0);
  const rows = Number(detailPayload.checker_board_rows || 0);
  const squareSizeMm = Number(detailPayload.checker_board_size_mm || 0);
  const boardType = String(detailPayload.checker_board_type || "chessboard");
  const markerSizeMm = Number(detailPayload.marker_size_mm || 0);
  if (columns < 3 || rows < 3 || squareSizeMm <= 0) {
    showCalibrationResultModal({ ok: false, mode: "EXTR", error: "bad_chessboard_setup" });
    return;
  }
  if (boardType === "charuco" && (markerSizeMm <= 0 || markerSizeMm >= squareSizeMm)) {
    showCalibrationResultModal({ ok: false, mode: "EXTR", error: "bad_charuco_setup" });
    return;
  }

  const orientation = String(detailPayload.chessboard_orientation || "horizontal");
  // Column/Row count squares for ChArUco and inner corners for Chessboard, so the
  // object grid the detector's corner ids index into differs by one in each direction.
  const gridColumns = boardType === "charuco" ? columns - 1 : columns;
  const gridRows = boardType === "charuco" ? rows - 1 : rows;
  const objectPoints = chessboardObjectPoints(gridColumns, gridRows, squareSizeMm, orientation);
  const response = await fetchChessboardCorners(folderName, {
    checker_board_type: boardType,
    aruco_dictionary: detailPayload.aruco_dictionary,
    checker_board_columns: columns,
    checker_board_rows: rows,
    checker_board_size_mm: squareSizeMm,
    marker_size_mm: markerSizeMm,
  });
  const frames = response.frames.map((frame) => ({
    ...frame,
    corners: (frame.corners || []).filter((corner) => objectPoints.some((point) => point.id === corner.id)).map((corner, index) => ({
      id: boardType === "charuco" ? corner.id : objectPoints[index].id,
      u: corner.u,
      v: corner.v,
    })),
  }));
  showExtrinsicPointModal(
    folderName,
    frames,
    objectPoints,
    detailPayload.intrinsic_calibration,
    "chessboard",
    boardType,
    orientation,
  );
}

function showExtrinsicPointModal(
  folderName: string,
  frames: CalibrationFrame[],
  objectPoints: ObjectPoint[],
  intrinsicCalibration: unknown,
  mode: ExtrinsicPointMode,
  boardType: string,
  chessboardOrientation: string,
): void {
  if (!calibrationModal || !calibrationModalTitle || !calibrationModalBody) return;
  extrinsicPointSession = null;
  calibrationModal.hidden = false;
  calibrationModalTitle.textContent = mode === "chessboard" ? "Extrinsic Board Corners" : "Extrinsic Point Selection";
  if (calibrationModalSpinner) calibrationModalSpinner.hidden = true;
  if (calibrationModalClose) calibrationModalClose.hidden = false;
  calibrationModalBody.innerHTML = `
    <div class="extrinsic-point-toolbar">
      <strong data-extrinsic-point-status></strong>
      <div>
        <button class="button secondary" type="button" data-extrinsic-reset>Reset View</button>
        ${mode === "chessboard" ? `<button class="button secondary" type="button" data-extrinsic-swap>Swap First/Last</button>` : ""}
        <button class="button secondary" type="button" data-extrinsic-undo>Undo</button>
        <button class="button record" type="button" data-extrinsic-submit>Calibration</button>
      </div>
    </div>
    <div class="extrinsic-point-grid">
      ${frames.slice(0, 4).map((frame, index) => `
        <div class="extrinsic-point-panel">
          <strong>${frame.camera_label.toUpperCase()}${typeof frame.frame_index === "number" ? ` - frame ${frame.frame_index}` : ""}</strong>
          <canvas data-extrinsic-canvas="${index}" width="640" height="420"></canvas>
        </div>
      `).join("")}
    </div>
  `;
  const status = calibrationModalBody.querySelector<HTMLElement>("[data-extrinsic-point-status]");
  if (!status) return;
  const canvases: PointCanvasState[] = [];
  extrinsicPointSession = {
    folderName,
    mode,
    boardType,
    chessboardOrientation,
    objectPoints,
    intrinsicCalibration,
    canvases,
    history: [],
    status,
  };
  frames.slice(0, 4).forEach((frame, index) => {
    const canvas = calibrationModalBody.querySelector<HTMLCanvasElement>(`[data-extrinsic-canvas="${index}"]`);
    const ctx = canvas?.getContext("2d");
    if (!canvas || !ctx) return;
    const image = new Image();
    const state: PointCanvasState = {
      cameraLabel: frame.camera_label,
      frameIndex: typeof frame.frame_index === "number" ? frame.frame_index : 0,
      canvas,
      ctx,
      image,
      points: (frame.corners || []).map((point, pointIndex) => ({
        id: point.id ?? objectPoints[pointIndex]?.id,
        u: point.u,
        v: point.v,
      })),
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
    undoExtrinsicPointEdit();
  });
  calibrationModalBody.querySelector<HTMLButtonElement>("[data-extrinsic-swap]")?.addEventListener("click", swapExtrinsicChessboardOrder);
  calibrationModalBody.querySelector<HTMLButtonElement>("[data-extrinsic-submit]")?.addEventListener("click", submitExtrinsicPointCalibration);
  updateExtrinsicPointStatus();
}

function fitPointCanvas(state: PointCanvasState): void {
  const scale = Math.min(state.canvas.width / state.image.width, state.canvas.height / state.image.height);
  state.scale = scale;
  state.offsetX = (state.canvas.width - state.image.width * scale) / 2;
  state.offsetY = (state.canvas.height - state.image.height * scale) / 2;
}

function cloneImagePoints(points: ImagePoint[]): ImagePoint[] {
  return points.map((point) => ({ ...point }));
}

function pushExtrinsicPointHistory(states: PointCanvasState[]): void {
  const session = extrinsicPointSession;
  if (!session) return;
  session.history.push({
    snapshots: states.map((state) => ({
      state,
      points: cloneImagePoints(state.points),
    })),
  });
}

function undoExtrinsicPointEdit(): void {
  const entry = extrinsicPointSession?.history.pop();
  if (!entry) return;
  entry.snapshots.forEach((snapshot) => {
    snapshot.state.points = cloneImagePoints(snapshot.points);
    drawPointCanvas(snapshot.state);
  });
  updateExtrinsicPointStatus();
}

function pointIdKey(id: string | number | undefined): string {
  return String(id ?? "");
}

function nextMissingObjectPoint(
  session: NonNullable<typeof extrinsicPointSession>,
  state: PointCanvasState,
): ObjectPoint | undefined {
  const selected = new Set(state.points.map((point) => pointIdKey(point.id)));
  return session.objectPoints.find((point) => !selected.has(pointIdKey(point.id)));
}

function swapExtrinsicChessboardOrder(): void {
  const session = extrinsicPointSession;
  if (!session || session.mode !== "chessboard") return;
  const reversedPoints = [...session.objectPoints].reverse();
  const swappedById = new Map(
    session.objectPoints.map((point, index) => [pointIdKey(point.id), reversedPoints[index]?.id]),
  );
  pushExtrinsicPointHistory(session.canvases);
  session.canvases.forEach((state) => {
    state.points = cloneImagePoints(state.points).map((point) => ({
      ...point,
      id: swappedById.get(pointIdKey(point.id)) ?? point.id,
    }));
    drawPointCanvas(state);
  });
  updateExtrinsicPointStatus();
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
    if (!session) return;
    const canvasPoint = eventCanvasPoint(state, event);
    const point = canvasToImagePoint(state, canvasPoint.x, canvasPoint.y);
    if (point.u < 0 || point.v < 0 || point.u > state.image.width || point.v > state.image.height) return;
    const nextPoint = session.mode === "chessboard"
      ? nextMissingObjectPoint(session, state)
      : session.objectPoints[state.points.length];
    if (!nextPoint) {
      if (session.mode !== "chessboard") return;
      const targetIndex = nearestPointIndex(state, canvasPoint.x, canvasPoint.y);
      if (targetIndex < 0) return;
      pushExtrinsicPointHistory([state]);
      state.points[targetIndex] = {
        ...state.points[targetIndex],
        u: point.u,
        v: point.v,
      };
      drawPointCanvas(state);
      updateExtrinsicPointStatus();
      return;
    }
    pushExtrinsicPointHistory([state]);
    state.points.push({ id: nextPoint.id, u: point.u, v: point.v });
    drawPointCanvas(state);
    updateExtrinsicPointStatus();
  });
}

function nearestPointIndex(state: PointCanvasState, x: number, y: number): number {
  let nearest = -1;
  let nearestDistance = Number.POSITIVE_INFINITY;
  state.points.forEach((point, index) => {
    const pointX = state.offsetX + point.u * state.scale;
    const pointY = state.offsetY + point.v * state.scale;
    const distance = (pointX - x) ** 2 + (pointY - y) ** 2;
    if (distance < nearestDistance) {
      nearest = index;
      nearestDistance = distance;
    }
  });
  return nearest;
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
    const session = extrinsicPointSession;
    const firstPointId = session?.mode === "chessboard" ? pointIdKey(session.objectPoints[0]?.id) : "";
    const lastPointId = session?.mode === "chessboard" ? pointIdKey(session.objectPoints[session.objectPoints.length - 1]?.id) : "";
    const pointId = pointIdKey(point.id);
    const isFirst = session?.mode === "chessboard" ? pointId === firstPointId : index === 0;
    const isLast = session?.mode === "chessboard" ? pointId === lastPointId : index === state.points.length - 1;
    state.ctx.beginPath();
    state.ctx.arc(x, y, isFirst || isLast ? 6 : 5, 0, Math.PI * 2);
    state.ctx.fillStyle = isFirst ? "#d7ff43" : isLast ? "#ffb347" : "#ffffff";
    state.ctx.fill();
    state.ctx.lineWidth = 2;
    state.ctx.strokeStyle = "#05080a";
    state.ctx.stroke();
    state.ctx.fillStyle = "#ffffff";
    state.ctx.font = "700 13px system-ui";
    state.ctx.fillText(String(point.id ?? index), x + 8, y - 8);
  });
}

function updateExtrinsicPointStatus(): void {
  const session = extrinsicPointSession;
  if (!session) return;
  const cameraCounts = session.canvases
    .map((state) => `${state.cameraLabel.toUpperCase()} ${state.points.length}/${session.objectPoints.length}`)
    .join(", ");
  if (session.mode === "chessboard") {
    const missing = session.canvases
      .map((state) => ({ state, point: nextMissingObjectPoint(session, state) }))
      .filter((item): item is { state: PointCanvasState; point: ObjectPoint } => Boolean(item.point));
    if (missing.length === 0) {
      session.status.textContent = `Detected corners loaded. Click a corner to move it. ${cameraCounts}`;
      return;
    }
    const missingDetails = missing
      .map((item) => `${item.state.cameraLabel.toUpperCase()} next ${item.point.id}`)
      .join(", ");
    session.status.textContent = `Add missing board corners: ${missingDetails} - ${cameraCounts}`;
    return;
  }
  const counts = session.canvases.map((state) => state.points.length);
  const next = Math.min(...counts);
  const nextPoint = session.objectPoints[next];
  session.status.textContent = nextPoint
    ? `Next point: ${nextPoint.id} (${nextPoint.x}, ${nextPoint.y}, ${nextPoint.z}) - ${cameraCounts}`
    : "All points selected";
}

async function submitExtrinsicPointCalibration(): Promise<void> {
  const session = extrinsicPointSession;
  if (!session || session.canvases.length < 2) return;
  const requiredPoints = session.objectPoints.length;
  const hasMissingChessboardPoint = session.mode === "chessboard"
    && session.canvases.some((state) => Boolean(nextMissingObjectPoint(session, state)));
  if (hasMissingChessboardPoint || session.canvases.some((state) => state.points.length < requiredPoints)) {
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
    extrinsic_calibration_target: session.mode,
    checker_board_type: session.boardType || "chessboard",
    chessboard_orientation: session.chessboardOrientation,
    board_position: session.chessboardOrientation,
    object_points: session.objectPoints,
    image_points_by_camera: imagePointsByCamera,
    image_points_cam1: cam1?.points || [],
    image_points_cam2: cam2?.points || [],
    image_frame_index_by_camera: Object.fromEntries(
      session.canvases.map((state) => [state.cameraLabel, state.frameIndex]),
    ),
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
  syncExtrinsicTarget();
  syncBoardFields();
}

function syncExtrinsicTarget(): void {
  const target = currentExtrinsicTarget();
  if (target === "chessboard") {
    copyIntrinsicBoardFieldsToExtrinsic();
  }
  extrinsicTargetInputs.forEach((input) => {
    input.closest("label")?.classList.toggle("is-active", input.value === target);
  });
  extrinsicObjectFields.forEach((field) => {
    field.hidden = target !== "object";
  });
  extrinsicChessboardFields.forEach((field) => {
    field.hidden = target !== "chessboard";
  });
  syncExtrinsicBoardFields();
}

function syncExtrinsicBoardFields(): void {
  const target = currentExtrinsicTarget();
  const boardType = extrinsicBoardType?.value || "chessboard";
  extrinsicBoardFields.forEach((field) => {
    const allowedTypes = String(field.dataset.extrinsicBoardField || "").split(/\s+/).filter(Boolean);
    field.hidden = target !== "chessboard" || !allowedTypes.includes(boardType);
  });
  syncBoardCountHints();
}

function boardCountHintText(boardType: string): string {
  return boardType === "charuco"
    ? "ChArUco: Column and Row count squares — a 10 x 7 board is 10 by 7 squares."
    : "Chessboard: Column and Row count inner corners — a 10 x 7 board has 9 x 6 inner corners.";
}

function syncBoardCountHints(): void {
  const extrinsic = calibrationMode?.value === "extrinsic";
  const boardType = (extrinsic ? extrinsicBoardType?.value : calibrationBoardType?.value) || "chessboard";
  boardCountHints.forEach((hint) => {
    hint.textContent = boardCountHintText(boardType);
  });
}

function syncBoardFields(): void {
  const boardType = calibrationBoardType?.value || "chessboard";
  boardFields.forEach((field) => {
    const allowedTypes = String(field.dataset.boardField || "").split(/\s+/).filter(Boolean);
    field.hidden = calibrationMode?.value === "extrinsic" || !allowedTypes.includes(boardType);
  });
  syncBoardCountHints();
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
extrinsicBoardType?.addEventListener("change", syncExtrinsicBoardFields);
extrinsicTargetInputs.forEach((input) => {
  input.addEventListener("change", syncExtrinsicTarget);
});

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
  if (!(await confirmCalibrationDelete())) return;
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

if (window.io) {
  const socket = window.io();
  socket.on("camera_status", (payload) => renderCameras(payload as CameraStatus[]));
  socket.on("phone_registered", (payload) => {
    const registration = payload as PhoneRegistration;
    adoptPhoneSessionToken(registration.token);
  });
  socket.on("phone_preview_frame", (payload) => renderPhonePreviewFrame(payload as PhonePreviewFrame));
  socket.on("phone_upload_complete", () => {
    refreshCalibrations().catch((error) => renderState("Issue", errorMessage(error)));
  });
}

syncCalibrationMode();
refreshCameras().catch((error) => renderState("Issue", errorMessage(error)));
refreshCalibrations().catch((error) => renderState("Issue", errorMessage(error)));

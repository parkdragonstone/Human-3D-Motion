import { fetchJson, postJson } from "./api.js";
import type { CameraSettings, CameraStatus, CaptureSession, CaptureStatusPayload, PhoneDraft } from "./types.js";

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

const cameraList = document.querySelector<HTMLElement>("#cameraList");
const cameraCount = document.querySelector<HTMLElement>("#cameraCount");
const captureState = document.querySelector<HTMLElement>("#captureState");
const captureForm = document.querySelector<HTMLFormElement>("[data-capture-form]");
const storageForm = document.querySelector<HTMLFormElement>("[data-storage-form]");
const storageRootInput = document.querySelector<HTMLInputElement>("#storageRootInput");
const selectStorageRootButton = document.querySelector<HTMLButtonElement>("[data-select-storage-root]");
const cameraSettingsForm = document.querySelector<HTMLFormElement>("[data-camera-settings-form]");
const captureActionButton = document.querySelector<HTMLButtonElement>("[data-capture-action-button]");
const captureActionLabel = document.querySelector<HTMLElement>("[data-capture-action-label]");
const recordingTimer = document.querySelector<HTMLElement>("[data-capture-recording-timer]");
const sessionList = document.querySelector<HTMLElement>("#sessionList");
const modeInputs = Array.from(document.querySelectorAll<HTMLInputElement>("input[name='capture_mode']"));
const sonySettings = Array.from(document.querySelectorAll<HTMLElement>("[data-sony-setting]"));
const phoneSettings = Array.from(document.querySelectorAll<HTMLElement>("[data-phone-setting]"));
const phonePanel = document.querySelector<HTMLElement>("[data-phone-panel]");
const phoneQrList = document.querySelector<HTMLElement>("#phoneQrList");
const phoneTokenInput = document.querySelector<HTMLInputElement>("[data-phone-session-token-input]");
let lastCameras: CameraStatus[] = [];
const phonePreviewFrames = new Map<string, string>();
let recordingStartedAt = 0;
let recordingTimerId = 0;

function renderCameras(cameras: CameraStatus[]): void {
  if (!cameraList) return;
  lastCameras = cameras;
  if (cameraCount) {
    cameraCount.textContent = `${cameras.length} cameras detected`;
  }
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
          <span class="live-view-frame">
            ${liveView}
          </span>
          <span class="camera-meta">
            <span class="camera-index">${camera.label}</span>
          </span>
        </label>
      `;
    })
    .join("");
}

function currentPhoneSessionToken(): string {
  return phoneTokenInput?.value || phoneQrList?.dataset.phoneSessionToken || "";
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
  return (modeInputs.find((input) => input.checked)?.value === "phone" ? "phone" : "sony");
}

function applyCaptureMode(mode: "sony" | "phone"): void {
  sonySettings.forEach((element) => {
    element.hidden = mode !== "sony";
  });
  phoneSettings.forEach((element) => {
    element.hidden = mode !== "phone";
  });
  if (phonePanel) {
    phonePanel.hidden = mode !== "phone";
  }
  modeInputs.forEach((input) => {
    input.checked = input.value === mode;
    input.closest("label")?.classList.toggle("is-active", input.value === mode);
  });
}

function renderPhoneDraft(draft: PhoneDraft): void {
  if (!phoneQrList) return;
  phoneQrList.dataset.phoneSessionToken = draft.token;
  if (phoneTokenInput) {
    phoneTokenInput.value = draft.token;
  }
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

function renderSessions(sessions: CaptureSession[]): void {
  if (!sessionList) return;
  if (sessions.length === 0) {
    sessionList.innerHTML = `<p class="empty">No capture sessions yet.</p>`;
    return;
  }
  sessionList.innerHTML = sessions
    .slice(0, 8)
    .map(
      (session) => `
        <article class="session-card">
          <header class="session-card-header">
            <div class="session-subject">
              <strong>${session.subject.name}</strong>
              <span>${session.subject.height_cm}cm</span>
              <span>${session.subject.weight_kg}kg</span>
              <span>${session.subject.hand === "left" ? "Left" : "Right"}</span>
            </div>
            <div class="session-actions">
              <time>${session.display_timestamp || session.timestamp}</time>
              <a class="session-analyze-button" href="/analysis">분석하기</a>
              <button class="session-delete-button" type="button" data-delete-session-id="${session.session_id}">삭제</button>
            </div>
          </header>
          <div class="session-video-list">
            ${session.videos
              .map(
                (video) => `
                  <div class="session-video-item">
                    <span class="session-camera-tag">${video.camera_label.toUpperCase()}</span>
                    <div>
                      <strong>${video.filename || video.path.split(/[\\/]/).pop() || video.path}</strong>
                      <small>${video.size_label || ""}</small>
                    </div>
                  </div>
                `,
              )
              .join("")}
          </div>
        </article>
      `,
    )
    .join("");
}

function renderState(status: string, detail: string): void {
  if (!captureState) return;
  const dotClass = status === "Recording" ? "recording" : status === "Issue" ? "warning" : "ready";
  captureState.innerHTML = `
    <span class="status-dot ${dotClass}"></span>
    <strong>${status}</strong>
    <span>${detail}</span>
  `;
}

function isRecording(): boolean {
  return captureActionButton?.dataset.recording === "true";
}

function formatRecordingElapsed(elapsedMs: number): string {
  const totalSeconds = Math.max(0, Math.floor(elapsedMs / 1000));
  const hours = Math.floor(totalSeconds / 3600);
  const minutes = Math.floor((totalSeconds % 3600) / 60);
  const seconds = totalSeconds % 60;
  return [hours, minutes, seconds].map((value) => String(value).padStart(2, "0")).join(":");
}

function updateRecordingTimer(): void {
  if (!recordingTimer) return;
  const elapsed = recordingStartedAt > 0 ? Date.now() - recordingStartedAt : 0;
  recordingTimer.textContent = formatRecordingElapsed(elapsed);
}

function startRecordingTimer(): void {
  if (recordingStartedAt === 0) {
    recordingStartedAt = Date.now();
  }
  updateRecordingTimer();
  if (recordingTimerId) return;
  recordingTimerId = window.setInterval(updateRecordingTimer, 1000);
}

function stopRecordingTimer(): void {
  if (recordingTimerId) {
    window.clearInterval(recordingTimerId);
    recordingTimerId = 0;
  }
  recordingStartedAt = 0;
  updateRecordingTimer();
}

function setCaptureActionRecording(recording: boolean): void {
  if (!captureActionButton) return;
  captureActionButton.dataset.recording = recording ? "true" : "false";
  if (captureActionLabel) {
    captureActionLabel.textContent = recording ? "Stop" : "Record";
  }
  captureActionButton.classList.toggle("record", !recording);
  captureActionButton.classList.toggle("stop", recording);
  captureActionButton.disabled = false;
  if (recording) {
    startRecordingTimer();
  } else {
    stopRecordingTimer();
  }
}

async function refreshCameras(): Promise<void> {
  const cameras = await fetchJson<CameraStatus[]>("/api/cameras");
  renderCameras(cameras);
  const recording = cameras.some((camera) => camera.recording);
  renderState(recording ? "Recording" : "Ready", `${cameras.length} cameras available`);
  setCaptureActionRecording(recording);
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
  if (storageRootInput) {
    storageRootInput.value = response.storage_root;
  }
  if (!response.cancelled) {
    renderState("Ready", "Storage path updated");
    await refreshSessions();
  }
}

async function applyCameraSettings(): Promise<void> {
  if (!cameraSettingsForm || !cameraSettingsForm.reportValidity()) return;
  const data = new FormData(cameraSettingsForm);
  const payload = {
    capture_mode: String(data.get("capture_mode") || currentCaptureMode()),
    camera_count: Number(data.get("camera_count") || 1),
    ccb_url: String(data.get("ccb_url") || ""),
    live_view_frame_rate: String(data.get("live_view_frame_rate") || "low"),
    phone_camera_count: Number(data.get("phone_camera_count") || 1),
    phone_frame_rate: Number(data.get("phone_frame_rate") || 120),
  };
  const settings = await postJson<CameraSettings>("/api/settings/cameras", payload);
  applyCaptureMode(settings.capture_mode);
  await refreshCameras();
  await refreshPhoneDraft();
  renderState("Ready", "Camera settings applied");
}

async function refreshSessions(): Promise<void> {
  const sessions = await fetchJson<CaptureSession[]>("/api/sessions");
  renderSessions(sessions);
}

function selectedCameraIds(form: HTMLFormElement): string[] {
  return Array.from(form.querySelectorAll<HTMLInputElement>("input[name='camera_ids']:checked")).map(
    (input) => input.value,
  );
}

function errorMessage(error: unknown): string {
  if (!(error instanceof Error)) return "Capture request failed";
  try {
    const payload = JSON.parse(error.message) as { error?: string };
    return payload.error || error.message;
  } catch {
    return error.message;
  }
}

async function deleteSession(sessionId: string): Promise<void> {
  const response = await fetch(`/api/sessions/${encodeURIComponent(sessionId)}`, {
    method: "DELETE",
    headers: { Accept: "application/json" },
  });
  if (!response.ok) {
    throw new Error(await response.text());
  }
}

async function startCapture(): Promise<void> {
  if (!captureForm || !captureForm.reportValidity()) return;
  const data = new FormData(captureForm);
  const cameraIds = selectedCameraIds(captureForm);
  if (cameraIds.length === 0) {
    renderState("Issue", "Pair at least one phone before recording");
    return;
  }
  const payload = {
    name: String(data.get("name") || "subject"),
    height_cm: Number(data.get("height_cm") || 170),
    weight_kg: Number(data.get("weight_kg") || 70),
    hand: String(data.get("hand") || "right"),
    camera_ids: cameraIds,
    phone_session_token: String(data.get("phone_session_token") || phoneQrList?.dataset.phoneSessionToken || ""),
  };
  const session = await postJson<CaptureSession>("/api/capture/start", payload);
  renderState("Recording", session.timestamp);
  setCaptureActionRecording(true);
  await refreshCameras();
}

async function stopCapture(): Promise<void> {
  const session = await postJson<CaptureSession>("/api/capture/stop", {});
  renderState("Ready", "Recording saved");
  setCaptureActionRecording(false);
  await refreshCameras();
  await refreshSessions();
}

captureForm?.addEventListener("submit", (event) => {
  event.preventDefault();
});

storageForm?.addEventListener("submit", (event) => {
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

captureActionButton?.addEventListener("click", async () => {
  captureActionButton.disabled = true;
  try {
    if (isRecording()) {
      await stopCapture();
    } else {
      await startCapture();
    }
  } catch (error) {
    renderState("Issue", errorMessage(error));
  } finally {
    captureActionButton.disabled = false;
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

sessionList?.addEventListener("click", async (event) => {
  const button = (event.target as HTMLElement).closest<HTMLButtonElement>("[data-delete-session-id]");
  if (!button) return;
  const sessionId = button.dataset.deleteSessionId;
  if (!sessionId) return;
  if (!window.confirm("이 세션을 삭제할까요?")) return;

  button.disabled = true;
  try {
    await deleteSession(sessionId);
    await refreshSessions();
  } catch (error) {
    renderState("Issue", errorMessage(error));
    button.disabled = false;
  }
});

modeInputs.forEach((input) => {
  input.addEventListener("change", () => {
    const mode = currentCaptureMode();
    applyCaptureMode(mode);
    postJson<CameraSettings>("/api/capture/mode", { capture_mode: mode })
      .then(() => Promise.all([refreshCameras(), mode === "phone" ? refreshPhoneDraft() : Promise.resolve()]))
      .catch(() => undefined);
  });
});

if (window.io) {
  const socket = window.io();
  socket.on("camera_status", (payload) => renderCameras(payload as CameraStatus[]));
  socket.on("phone_preview_frame", (payload) => renderPhonePreviewFrame(payload as PhonePreviewFrame));
  socket.on("capture_status", (payload) => {
    const event = payload as CaptureStatusPayload;
    const recording = event.status === "recording";
    renderState(recording ? "Recording" : "Ready", event.session.timestamp);
    setCaptureActionRecording(recording);
    refreshSessions().catch(() => undefined);
  });
}

refreshCameras().catch(() => undefined);
refreshSessions().catch(() => undefined);
refreshCameraSettings().catch(() => applyCaptureMode(currentCaptureMode()));

import { postJson } from "./api.js";

declare global {
  interface Window {
    phoneCaptureConfig?: {
      token: string;
      cameraLabel: string;
      frameRate: number;
      resolution: string;
    };
    io?: () => PhoneSocket;
  }
}

interface PhoneSocket {
  emit: (event: string, payload: unknown) => void;
  on: (event: string, callback: (payload: unknown) => void) => void;
}

interface PhoneRecordingCommand {
  command: "start" | "stop";
  token: string;
  camera_label?: string;
}

const config = window.phoneCaptureConfig;
const preview = document.querySelector<HTMLVideoElement>("#phonePreview");
const connectionState = document.querySelector<HTMLElement>("#phoneConnectionState");
const timer = document.querySelector<HTMLElement>("#phoneTimer");
const clipCount = document.querySelector<HTMLElement>("#phoneClipCount");
const captureMeta = document.querySelector<HTMLElement>("#phoneCaptureMeta");

let stream: MediaStream | null = null;
let recorder: MediaRecorder | null = null;
let chunks: Blob[] = [];
let startedAt = 0;
let timerId = 0;
let clips = 0;
let actualFrameRate = "";
let actualWidth = "";
let actualHeight = "";
let socket: PhoneSocket | null = null;
let previewFrameTimerId = 0;

type VideoSize = { width: number; height: number };

function setConnection(value: string): void {
  if (connectionState) connectionState.textContent = value;
}

function setRecording(active: boolean): void {
  document.documentElement.classList.toggle("is-phone-recording", active);
}

function isPortraitViewport(): boolean {
  const viewport = window.visualViewport;
  const width = viewport?.width || window.innerWidth;
  const height = viewport?.height || window.innerHeight;
  return height > width;
}

function sizeForViewport(size: VideoSize): VideoSize {
  const sizeIsPortrait = size.height > size.width;
  if (sizeIsPortrait === isPortraitViewport()) {
    return size;
  }
  return { width: size.height, height: size.width };
}

function videoSize(resolution: string): VideoSize {
  const match = /^(\d+)x(\d+)$/.exec(resolution);
  if (!match) return { width: 1280, height: 720 };
  return {
    width: Number(match[1]),
    height: Number(match[2]),
  };
}

function exactVideoConstraint(value: number): ConstrainULong {
  return { exact: value };
}

function displayVideoSize(video: HTMLVideoElement): VideoSize {
  const size = { width: video.videoWidth, height: video.videoHeight };
  return sizeForViewport(size);
}

function drawPreviewSurface(
  context: CanvasRenderingContext2D,
  video: HTMLVideoElement,
  width: number,
  height: number,
): void {
  const videoRatio = video.videoWidth / video.videoHeight;
  const targetRatio = width / height;
  let sourceX = 0;
  let sourceY = 0;
  let sourceWidth = video.videoWidth;
  let sourceHeight = video.videoHeight;
  if (videoRatio > targetRatio) {
    sourceWidth = video.videoHeight * targetRatio;
    sourceX = (video.videoWidth - sourceWidth) / 2;
  } else if (videoRatio < targetRatio) {
    sourceHeight = video.videoWidth / targetRatio;
    sourceY = (video.videoHeight - sourceHeight) / 2;
  }
  context.clearRect(0, 0, width, height);
  context.drawImage(video, sourceX, sourceY, sourceWidth, sourceHeight, 0, 0, width, height);
}

function mediaConstraints(
  frameRateConstraint: ConstrainDouble,
): MediaStreamConstraints {
  const size = videoSize(config?.resolution || "");
  return {
    audio: false,
    video: {
      facingMode: { ideal: "environment" },
      frameRate: frameRateConstraint,
      width: exactVideoConstraint(size.width),
      height: exactVideoConstraint(size.height),
    },
  };
}

function highSpeedConstraintAttempts(frameRate: number): ConstrainDouble[] {
  const attempts: ConstrainDouble[] = [{ exact: frameRate }];
  if (frameRate > 120) {
    attempts.push({ min: 120, ideal: frameRate, max: frameRate });
    attempts.push({ exact: 120 });
  }
  if (frameRate > 60) {
    attempts.push({ min: 60, ideal: Math.min(frameRate, 120), max: frameRate });
  }
  attempts.push({ ideal: frameRate });
  attempts.push({ ideal: Math.min(frameRate, 60) });
  return attempts;
}

async function openCamera(): Promise<MediaStream> {
  if (!config) {
    throw new Error("Missing phone capture settings.");
  }
  if (!navigator.mediaDevices?.getUserMedia) {
    throw new Error("Camera requires HTTPS or localhost.");
  }
  let lastError: unknown = null;
  for (const frameRateConstraint of highSpeedConstraintAttempts(config.frameRate)) {
    try {
      return await navigator.mediaDevices.getUserMedia(
        mediaConstraints(frameRateConstraint),
      );
    } catch (error) {
      if (error instanceof DOMException && error.name === "NotAllowedError") {
        throw error;
      }
      lastError = error;
    }
  }
  if (lastError instanceof Error) {
    throw lastError;
  }
  throw new Error("Camera permission failed");
}

function actualFrameRateText(requestedFrameRate: number): string {
  if (!actualFrameRate) return `${requestedFrameRate} fps requested`;
  const actual = Number(actualFrameRate);
  const actualText = Number.isFinite(actual) ? String(Math.round(actual)) : actualFrameRate;
  return actualText === String(requestedFrameRate)
    ? `${actualText} fps`
    : `${actualText} fps actual / ${requestedFrameRate} requested`;
}

function emitPreviewStatus(status: "ready" | "blocked", message?: string): void {
  if (!config || !socket) return;
  socket.emit("phone_preview_status", {
    token: config.token,
    camera_label: config.cameraLabel,
    status,
    message: message || "",
    actual_fps: actualFrameRate,
    actual_width: actualWidth,
    actual_height: actualHeight,
  });
}

function startPreviewFrameStream(): void {
  if (!config || !preview || !socket || previewFrameTimerId) return;
  const canvas = document.createElement("canvas");
  const context = canvas.getContext("2d");
  if (!context) return;

  previewFrameTimerId = window.setInterval(() => {
    if (!preview.videoWidth || !preview.videoHeight) return;
    const maxEdge = 640;
    const displaySize = displayVideoSize(preview);
    const scale = Math.min(1, maxEdge / Math.max(displaySize.width, displaySize.height));
    canvas.width = Math.round(displaySize.width * scale);
    canvas.height = Math.round(displaySize.height * scale);
    drawPreviewSurface(context, preview, canvas.width, canvas.height);
    socket?.emit("phone_preview_frame", {
      token: config.token,
      camera_label: config.cameraLabel,
      image: canvas.toDataURL("image/jpeg", 0.62),
      width: displaySize.width,
      height: displaySize.height,
    });
  }, 200);
}

async function startPreview(): Promise<void> {
  if (!config || !preview) return;
  stream = await openCamera();
  preview.srcObject = stream;
  await preview.play().catch(() => undefined);
  const settings = stream.getVideoTracks()[0]?.getSettings();
  actualFrameRate = settings?.frameRate ? String(settings.frameRate) : "";
  const settingsWidth = settings?.width || preview.videoWidth;
  const settingsHeight = settings?.height || preview.videoHeight;
  actualWidth = settingsWidth ? String(settingsWidth) : "";
  actualHeight = settingsHeight ? String(settingsHeight) : "";
  if (captureMeta) {
    const resolution = actualWidth && actualHeight ? `${actualWidth}x${actualHeight}` : config.resolution;
    captureMeta.textContent = `${actualFrameRateText(config.frameRate)} - ${resolution}`;
  }
  setConnection("Ready");
  emitPreviewStatus("ready");
  startPreviewFrameStream();
}

function startTimer(): void {
  startedAt = Date.now();
  window.clearInterval(timerId);
  timerId = window.setInterval(() => {
    if (!timer) return;
    const elapsed = Math.floor((Date.now() - startedAt) / 1000);
    const hours = String(Math.floor(elapsed / 3600)).padStart(2, "0");
    const minutes = String(Math.floor((elapsed % 3600) / 60)).padStart(2, "0");
    const seconds = String(elapsed % 60).padStart(2, "0");
    timer.textContent = `${hours}:${minutes}:${seconds}`;
  }, 250);
}

function preferredMimeType(): string {
  const candidates = ["video/mp4;codecs=h264", "video/mp4", "video/webm;codecs=vp9", "video/webm"];
  return candidates.find((type) => MediaRecorder.isTypeSupported(type)) || "";
}

function startRecording(): void {
  if (!stream || recorder?.state === "recording") return;
  chunks = [];
  const mimeType = preferredMimeType();
  recorder = new MediaRecorder(stream, mimeType ? { mimeType } : undefined);
  recorder.ondataavailable = (event) => {
    if (event.data.size > 0) chunks.push(event.data);
  };
  recorder.start(1000);
  setRecording(true);
  startTimer();
}

async function stopRecording(command: PhoneRecordingCommand): Promise<void> {
  if (!recorder || recorder.state === "inactive" || !config) return;
  await new Promise<void>((resolve) => {
    if (!recorder) {
      resolve();
      return;
    }
    recorder.onstop = () => resolve();
    recorder.stop();
  });
  window.clearInterval(timerId);
  setRecording(false);
  clips += 1;
  if (clipCount) clipCount.textContent = `${clips} clips`;

  const blob = new Blob(chunks, { type: recorder.mimeType || "video/mp4" });
  if (blob.size === 0) {
    throw new Error("recorded_video_is_empty");
  }
  const form = new FormData();
  form.append("video", blob, `${config.cameraLabel}.mp4`);
  form.append("actual_fps", actualFrameRate);
  form.append("actual_width", actualWidth);
  form.append("actual_height", actualHeight);
  const uploadResponse = await fetch(`/api/phone-sessions/${command.token}/${config.cameraLabel}/upload`, {
    method: "POST",
    body: form,
  });
  if (!uploadResponse.ok) {
    throw new Error(await uploadResponse.text());
  }
  await postJson(`/api/phone-sessions/${command.token}/finalize`, {});
}

if (config && window.io) {
  socket = window.io();
  socket.emit("phone_register", {
    token: config.token,
    camera_label: config.cameraLabel,
    frame_rate: config.frameRate,
    resolution: config.resolution,
  });
  socket.on("phone_recording_command", (payload) => {
    const command = payload as PhoneRecordingCommand;
    if (command.token !== config.token) return;
    if (command.camera_label && command.camera_label !== config.cameraLabel) return;
    if (command.command === "start") startRecording();
    if (command.command === "stop") stopRecording(command).catch(() => setConnection("Upload failed"));
  });
}

document.querySelectorAll("[data-phone-leave]").forEach((button) => {
  button.addEventListener("click", () => window.close());
});

startPreview().catch((error) => {
  setConnection("Camera blocked");
  const message = error instanceof Error ? error.message : "Camera permission failed";
  if (captureMeta) captureMeta.textContent = message;
  emitPreviewStatus("blocked", message);
});

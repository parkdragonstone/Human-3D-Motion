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

function setConnection(value: string): void {
  if (connectionState) connectionState.textContent = value;
}

function setRecording(active: boolean): void {
  document.documentElement.classList.toggle("is-phone-recording", active);
}

function videoSize(resolution: string): { width: number; height: number } {
  const match = /^(\d+)x(\d+)$/.exec(resolution);
  if (!match) return { width: 1280, height: 720 };
  return {
    width: Number(match[1]),
    height: Number(match[2]),
  };
}

function mediaConstraints(
  frameRate: number,
  frameRateConstraint: ConstrainDouble,
): MediaStreamConstraints {
  const size = videoSize(config?.resolution || "");
  return {
    audio: false,
    video: {
      facingMode: { ideal: "environment" },
      frameRate: frameRateConstraint,
      width: { ideal: size.width },
      height: { ideal: size.height },
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
        mediaConstraints(config.frameRate, frameRateConstraint),
      );
    } catch (error) {
      if (error instanceof DOMException && error.name === "NotAllowedError") {
        throw error;
      }
      lastError = error;
    }
  }
  try {
    return await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: { facingMode: { ideal: "environment" } },
    });
  } catch (error) {
    if (lastError instanceof Error) {
      throw lastError;
    }
    if (error instanceof Error) {
      throw error;
    }
    throw new Error("Camera permission failed");
  }
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
    const scale = Math.min(1, maxEdge / Math.max(preview.videoWidth, preview.videoHeight));
    canvas.width = Math.round(preview.videoWidth * scale);
    canvas.height = Math.round(preview.videoHeight * scale);
    context.drawImage(preview, 0, 0, canvas.width, canvas.height);
    socket?.emit("phone_preview_frame", {
      token: config.token,
      camera_label: config.cameraLabel,
      image: canvas.toDataURL("image/jpeg", 0.62),
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
  actualWidth = settings?.width ? String(settings.width) : "";
  actualHeight = settings?.height ? String(settings.height) : "";
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

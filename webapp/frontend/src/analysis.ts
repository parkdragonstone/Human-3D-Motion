import { fetchJson, postJson } from "./api.js";

interface SubjectInfo {
  name: string;
  height_cm: number;
  weight_kg: number;
  hand: string;
}

interface SessionVideo {
  camera_label: string;
  filename: string;
  path: string;
  video_url: string;
  fps?: number;
  frame_count?: number;
  pose_video_url?: string | null;
}

interface CaptureSession {
  session_id: string;
  session_path: string;
  subject: SubjectInfo;
  videos: SessionVideo[];
}

interface AnalysisJob {
  job_id: string;
  status: string;
  error?: string | null;
  logs: Array<{ level: string; message: string }>;
}

interface CalibrationUploadResult {
  filename: string;
  mode: string;
  path: string;
}

interface Pose3DFilesResult {
  files: string[];
}

interface Pose3DData {
  file: string;
  markers: string[];
  fps: number;
  num_frames: number;
  time?: Array<number | null>;
  frames: Array<Array<[number | null, number | null, number | null]>>;
}

interface KinematicsSignal {
  key: string;
  label: string;
  side: string;
  category: string;
  kind: string;
  unit: string;
}

interface KinematicsSummary {
  available: boolean;
  file?: string;
  unit: string;
  signals: KinematicsSignal[];
  events?: KinematicsEventMarker[];
}

interface KinematicsTimeseries {
  unit: string;
  time: Array<number | null>;
  values: Array<number | null>;
}

interface KinematicsEventMarker {
  key: string;
  label: string;
  description: string;
  time: number;
  frame?: number;
}

type PreparedKinematicsSeries = {
  signal: KinematicsSignal;
  data: KinematicsTimeseries;
  values: Array<{ value: number; time: number }>;
};

interface Keypoint2D {
  x: number | null;
  y: number | null;
  score: number | null;
}

interface KeypointPerson {
  person_index: number;
  keypoints: Keypoint2D[];
}

interface KeypointFrame {
  camera_label: string;
  frame: number;
  file: string;
  keypoint_names: string[];
  people: KeypointPerson[];
}

const page = document.querySelector<HTMLElement>(".analysis-page");
const rootInput = document.querySelector<HTMLInputElement>("#analysisRootInput");
const selectRootButton = document.querySelector<HTMLButtonElement>("[data-select-analysis-root]");
const sessionSelect = document.querySelector<HTMLSelectElement>("[data-analysis-session-select]");
const videoGrid = document.querySelector<HTMLElement>("[data-analysis-video-grid]");
const configForm = document.querySelector<HTMLFormElement>("[data-analysis-config-form]");
const runButton = document.querySelector<HTMLButtonElement>("[data-run-analysis]");
const logPanel = document.querySelector<HTMLElement>("[data-analysis-log]");
const overlayToggle = document.querySelector<HTMLInputElement>("[data-overlay-toggle]");
const togglePlayButton = document.querySelector<HTMLButtonElement>("[data-toggle-play-videos]");
const videoSeek = document.querySelector<HTMLInputElement>("[data-video-seek]");
const videoTimecode = document.querySelector<HTMLElement>("[data-video-timecode]");
const videoSpeed = document.querySelector<HTMLSelectElement>("[data-video-speed]");
const keypointEditToggle = document.querySelector<HTMLButtonElement>("[data-keypoint-edit-toggle]");
const keypointModal = document.querySelector<HTMLElement>("[data-keypoint-modal]");
const keypointCloseButton = document.querySelector<HTMLButtonElement>("[data-keypoint-close]");
const keypointEditVideo = document.querySelector<HTMLVideoElement>("[data-keypoint-edit-video]");
const keypointFrameSeek = document.querySelector<HTMLInputElement>("[data-keypoint-frame-seek]");
const keypointFrameLabel = document.querySelector<HTMLElement>("[data-keypoint-frame-label]");
const keypointSaveButton = document.querySelector<HTMLButtonElement>("[data-keypoint-save]");
const keypointNextVideoButton = document.querySelector<HTMLButtonElement>("[data-keypoint-next-video]");
const keypointSwapLowerButton = document.querySelector<HTMLButtonElement>("[data-keypoint-swap-lower]");
const keypointSwapUpperButton = document.querySelector<HTMLButtonElement>("[data-keypoint-swap-upper]");
const keypointUndoButton = document.querySelector<HTMLButtonElement>("[data-keypoint-undo]");
const keypointSaveStatus = document.querySelector<HTMLElement>("[data-keypoint-save-status]");
const calibrationForm = document.querySelector<HTMLFormElement>("[data-analysis-calibration-form]");
const calibrationFile = document.querySelector<HTMLInputElement>("[data-analysis-calibration-file]");
const calibrationFileName = document.querySelector<HTMLElement>("[data-analysis-calibration-name]");
const clearCalibrationButton = document.querySelector<HTMLButtonElement>("[data-clear-analysis-calibration]");
const pose3dFileSelect = document.querySelector<HTMLSelectElement>("[data-pose3d-file-select]");
const pose3dCanvas = document.querySelector<HTMLCanvasElement>("[data-pose3d-canvas]");
const pose3dEmpty = document.querySelector<HTMLElement>("[data-pose3d-empty]");
const pose3dPlayButton = document.querySelector<HTMLButtonElement>("[data-pose3d-play]");
const pose3dPipButton = document.querySelector<HTMLButtonElement>("[data-pose3d-pip]");
const pose3dSeek = document.querySelector<HTMLInputElement>("[data-pose3d-seek]");
const pose3dFrameLabel = document.querySelector<HTMLElement>("[data-pose3d-frame-label]");
const kinematicsTabs = document.querySelector<HTMLElement>("[data-kinematics-tabs]");
const kinematicsGrid = document.querySelector<HTMLElement>("[data-kinematics-grid]");
const kinematicsChart = document.querySelector<HTMLCanvasElement>("[data-kinematics-chart]");
const kinematicsChartTitle = document.querySelector<HTMLElement>("[data-kinematics-chart-title]");
const clearKinematicsSelectionButton = document.querySelector<HTMLButtonElement>("[data-clear-kinematics-selection]");

let sessions: CaptureSession[] = [];
let config: Record<string, unknown> = {};
let videosPlaying = false;
let seeking = false;
let keypointEditEnabled = false;
let keypointActiveVideoIndex = 0;
let keypointFrameData: KeypointFrame | null = null;
let keypointFrameRequestKey = "";
let keypointHoverTarget: { personIndex: number; keypointIndex: number } | null = null;
let keypointDragTarget: { personIndex: number; keypointIndex: number } | null = null;
let keypointSelectedPersonIndex: number | null = null;
let keypointPanDrag: { x: number; y: number; panX: number; panY: number } | null = null;
let keypointView = { zoom: 1, panX: 0, panY: 0 };
let keypointUndoStack: KeypointPerson[][] = [];
let keypointDirtyFrames = new Map<string, KeypointPerson[]>();
let keypointFrameDirty = false;
let keypointSaving = false;
let pose3dData: Pose3DData | null = null;
let pose3dFrame = 0;
let pose3dPlaying = false;
let pose3dAnimationId = 0;
let pose3dLastTick = 0;
let pose3dCamera = { rotX: -0.35, rotY: 0.45, zoom: 1, panX: 0, panY: 0 };
let pose3dDrag: { mode: "rotate" | "pan"; startX: number; startY: number; camera: typeof pose3dCamera } | null = null;
let pose3dPipVideo: HTMLVideoElement | null = null;
let kinematicsSummary: KinematicsSummary | null = null;
let selectedKinematicsKind = "angle";
let selectedKinematicsCategory = "pelvis";
const selectedKinematicsSignals = new Set<string>();
const kinematicsTimeseriesCache = new Map<string, KinematicsTimeseries>();
const kinematicsChartColors = ["#d7ff43", "#4aa3ff", "#ff6b4a", "#f6d34a", "#b9f6a5", "#c879ff"];
const kinematicsChartHeight = 260;
const keypointSkeletonPairs = [
  [0, 17], [18, 17], [18, 19], [18, 5], [18, 6], [5, 7], [7, 9], [6, 8], [8, 10],
  [19, 11], [19, 12], [11, 13], [13, 15], [12, 14], [14, 16], [15, 20], [20, 22],
  [15, 24], [16, 21], [21, 23], [16, 25],
];
const lowerBodySwapPairs = [[11, 12], [13, 14], [15, 16], [20, 21], [22, 23], [24, 25]];
const upperBodySwapPairs = [[5, 6], [7, 8], [9, 10]];
let kinematicsHoverX: number | null = null;
let kinematicsChartScrubbing = false;
const configSectionOrder = ["base", "pose", "lifting", "filtering", "kinematics"];
const hiddenConfigPaths = new Set([
  "pose.backend",
  "pose.det_nms",
  "pose.device",
  "pose.output_format",
  "pose.save_video",
  "lifting.camera_intrinsic_file",
  "lifting.remove_incomplete_frames",
  "lifting.show_interp_indices",
  "kinematics.remove_individual_ik_setup",
  "kinematics.remove_individual_scaling_setup",
]);

const liftingConfigGroups = [
  {
    title: "Calibration File",
    paths: [
      "reproj_error_threshold_triangulation",
      "likelihood_threshold_triangulation",
      "min_cameras_for_triangulation",
    ],
  },
  {
    title: "No Calibration File",
    paths: [
      "calib_frames",
      "cam1_person_idx",
      "cam2_person_idx",
    ],
  },
  {
    title: "Common",
    paths: [
      "feet_on_floor",
      "flip_left_right",
      "max_distance_m",
      "max_unseen_frames",
      "interp_if_gap_smaller_than",
      "interpolation",
      "sections_to_keep",
      "min_chunk_size",
      "fill_large_gaps_with",
    ],
  },
];

const kinematicsConfigGroups = [
  {
    title: "Kinematics Filter",
    paths: [
      "filter.cut_off_frequency",
      "filter.order",
    ],
  },
];

const selectConfigOptions: Record<string, string[]> = {
  "lifting.interpolation": ["linear", "slinear", "quadratic", "cubic", "none"],
  "lifting.sections_to_keep": ["all", "largest", "first", "last"],
  "lifting.fill_large_gaps_with": ["last_value", "nan", "zeros"],
};

const configFieldOrder: Record<string, string[]> = {
  pose: [
    "mode",
    "overwrite_pose",
    "det_score_threshold",
    "det_iou",
    "keypoint_likelihood_threshold",
    "average_likelihood_threshold",
    "keypoint_number_threshold",
    "max_distance_px",
  ],
  kinematics: [
    "use_simple_model",
    "use_augmentation",
    "right_left_symmetry",
    "fastest_frames_to_remove_percent",
    "close_to_zero_speed_m",
    "large_hip_knee_angles",
    "trimmed_extrema_percent",
    "filter",
  ],
};

function log(message: string): void {
  if (!logPanel) return;
  logPanel.textContent = `${logPanel.textContent || ""}${message}\n`;
  logPanel.scrollTop = logPanel.scrollHeight;
}

function selectedSession(): CaptureSession | null {
  return sessions.find((session) => session.session_id === sessionSelect?.value) || null;
}

async function loadConfig(): Promise<void> {
  config = await fetchJson<Record<string, unknown>>("/api/analysis/config");
  renderConfigForm();
}

async function loadSessions(preferredSessionId = ""): Promise<void> {
  if (!rootInput || !sessionSelect) return;
  const params = new URLSearchParams({ root: rootInput.value });
  sessions = await fetchJson<CaptureSession[]>(`/api/analysis/sessions?${params.toString()}`);
  sessionSelect.innerHTML = [
    `<option value="">Select session</option>`,
    ...sessions.map((session) => `<option value="${session.session_id}">${session.subject.name} - ${session.session_id}</option>`),
  ].join("");
  if (preferredSessionId && sessions.some((session) => session.session_id === preferredSessionId)) {
    sessionSelect.value = preferredSessionId;
  } else if (sessions.length > 0) {
    sessionSelect.value = sessions[0].session_id;
  }
  renderVideos();
  renderConfigForm();
}

function renderVideos(): void {
  const session = selectedSession();
  if (!videoGrid) return;
  if (!session) {
    videoGrid.classList.remove("four-up");
    videoGrid.innerHTML = `<p class="empty">Select a session.</p>`;
    keypointFrameData = null;
    keypointFrameRequestKey = "";
    if (videoSeek) {
      videoSeek.value = "0";
      videoSeek.max = "0";
    }
    pauseAllVideos();
    updateKeypointEditorControls();
    return;
  }
  const useOverlay = Boolean(overlayToggle?.checked);
  const currentTime = Number(videoSeek?.value || "0");
  videoGrid.classList.toggle("four-up", session.videos.length >= 4);
  videoGrid.innerHTML = session.videos.map((video) => {
    const src = useOverlay && video.pose_video_url ? video.pose_video_url : video.video_url;
    return `
      <article class="analysis-video-card">
        <div class="analysis-video-frame">
          <video src="${src}" controls preload="metadata"></video>
        </div>
        <small>${video.filename}</small>
      </article>
    `;
  }).join("");
  bindVideoControls(currentTime);
  updateKeypointEditorControls();
  loadCurrentKeypointFrame().catch(() => undefined);
}

async function loadAnalysisResults(): Promise<void> {
  await Promise.all([loadPose3DResults(), loadKinematicsResults()]);
}

async function loadPose3DResults(): Promise<void> {
  const session = selectedSession();
  if (!pose3dFileSelect) return;
  stopPose3D();
  pose3dData = null;
  pose3dFrame = 0;
  pose3dFileSelect.innerHTML = `<option value="">No 3D keypoint file</option>`;
  updatePose3DControls();
  drawPose3D();
  if (!session) return;
  try {
    const params = new URLSearchParams({ session_path: session.session_path });
    const result = await fetchJson<Pose3DFilesResult>(`/api/analysis/pose3d/files?${params.toString()}`);
    pose3dFileSelect.innerHTML = [
      `<option value="">Select TRC file</option>`,
      ...result.files.map((file) => `<option value="${file}">${file}</option>`),
    ].join("");
    if (result.files.length > 0) {
      pose3dFileSelect.value = preferredPose3DFile(result.files);
      await loadSelectedPose3DFile();
    }
  } catch {
    drawPose3D();
  }
}

function preferredPose3DFile(files: string[]): string {
  return (
    files.find((file) => file.toLowerCase().endsWith("lstm.trc")) ||
    files.find((file) => file.toLowerCase().endsWith("butterworth.trc")) ||
    files.find((file) => file.toLowerCase().endsWith("3d.trc")) ||
    files[0] ||
    ""
  );
}

async function loadSelectedPose3DFile(): Promise<void> {
  const session = selectedSession();
  const filename = pose3dFileSelect?.value || "";
  if (!session || !filename) {
    pose3dData = null;
    pose3dFrame = 0;
    updatePose3DControls();
    drawPose3D();
    return;
  }
  const params = new URLSearchParams({ session_path: session.session_path, file: filename });
  pose3dData = await fetchJson<Pose3DData>(`/api/analysis/pose3d/data?${params.toString()}`);
  pose3dFrame = 0;
  pose3dCamera = { rotX: -0.35, rotY: 0.45, zoom: 1, panX: 0, panY: 0 };
  updatePose3DControls();
  drawPose3D();
}

function updatePose3DControls(): void {
  const maxFrame = Math.max(0, (pose3dData?.num_frames || 0) - 1);
  if (pose3dSeek) {
    pose3dSeek.max = String(maxFrame);
    pose3dSeek.value = String(Math.min(pose3dFrame, maxFrame));
  }
  if (pose3dFrameLabel) {
    pose3dFrameLabel.textContent = pose3dData ? `${pose3dFrame + 1} / ${pose3dData.num_frames}` : "0 / 0";
  }
  if (pose3dPlayButton) {
    pose3dPlayButton.textContent = pose3dPlaying ? "Pause" : "Play";
  }
  if (pose3dPipButton) {
    pose3dPipButton.disabled = !pose3dData || !canUsePose3DPictureInPicture();
  }
}

function pose3DCurrentTime(): number | null {
  if (!pose3dData) return null;
  const timeValue = pose3dData.time?.[pose3dFrame];
  if (typeof timeValue === "number" && Number.isFinite(timeValue)) return timeValue;
  return pose3dFrame / Math.max(1, pose3dData.fps || 30);
}

function setPose3DFrameFromTime(time: number, stopPlayback = true): void {
  if (!pose3dData || !Number.isFinite(time)) return;
  const timeValues = pose3dData.time || [];
  let frame = Math.round(time * Math.max(1, pose3dData.fps || 30));
  let bestDistance = Number.POSITIVE_INFINITY;
  timeValues.forEach((value, index) => {
    if (typeof value !== "number" || !Number.isFinite(value)) return;
    const distance = Math.abs(value - time);
    if (distance < bestDistance) {
      bestDistance = distance;
      frame = index;
    }
  });
  pose3dFrame = Math.max(0, Math.min(pose3dData.num_frames - 1, frame));
  if (stopPlayback) {
    stopPose3D();
  }
  updatePose3DControls();
  drawPose3D();
  drawSelectedKinematicsChart();
}

function syncVideosToPose3DTime(): void {
  const time = pose3DCurrentTime();
  if (time === null) return;
  videoElements().forEach((video) => seekVideo(video, time));
  if (videoSeek) videoSeek.value = String(time);
  updateVideoTimecode(time);
}

function isPose3DAtEnd(): boolean {
  return Boolean(pose3dData && pose3dFrame >= Math.max(0, pose3dData.num_frames - 1));
}

function isPrimaryVideoAtEnd(): boolean {
  const primaryVideo = videoElements()[0];
  const duration = Number(videoSeek?.max || primaryVideo?.duration || 0);
  return Boolean(primaryVideo && duration > 0 && duration - primaryVideo.currentTime < 0.08);
}

function resetSyncedPlaybackToStart(): void {
  if (videoSeek) videoSeek.value = "0";
  videoElements().forEach((video) => seekVideo(video, 0));
  updateVideoTimecode(0);
  setPose3DFrameFromTime(0, false);
}

function drawPose3D(): void {
  if (!pose3dCanvas) return;
  const ctx = pose3dCanvas.getContext("2d");
  const wrapper = pose3dCanvas.parentElement;
  if (!ctx || !wrapper) return;
  const dpr = window.devicePixelRatio || 1;
  const width = wrapper.clientWidth || 640;
  const height = Math.max(320, Math.min(520, Math.round(width * 0.55)));
  pose3dCanvas.width = Math.round(width * dpr);
  pose3dCanvas.height = Math.round(height * dpr);
  pose3dCanvas.style.width = `${width}px`;
  pose3dCanvas.style.height = `${height}px`;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#05080a";
  ctx.fillRect(0, 0, width, height);
  if (!pose3dData || pose3dData.frames.length === 0) {
    if (pose3dEmpty) pose3dEmpty.hidden = false;
    return;
  }
  if (pose3dEmpty) pose3dEmpty.hidden = true;
  const frame = pose3dData.frames[Math.min(pose3dFrame, pose3dData.frames.length - 1)] || [];
  const scene = pose3DScene(pose3dData);
  const points = frame.map((point) => projectPose3D(point, scene, width, height));
  drawPose3DAxes(ctx, scene, width, height);
  pose3DBones(pose3dData.markers).forEach(([a, b]) => {
    const pa = points[a];
    const pb = points[b];
    if (!pa || !pb) return;
    const markerA = pose3dData?.markers[a] || "";
    const markerB = pose3dData?.markers[b] || "";
    ctx.beginPath();
    ctx.moveTo(pa.x, pa.y);
    ctx.lineTo(pb.x, pb.y);
    ctx.strokeStyle = pose3DLineColor(markerA, markerB);
    ctx.lineWidth = 2;
    ctx.stroke();
  });
  points.forEach((point, index) => {
    if (!point) return;
    ctx.beginPath();
    ctx.arc(point.x, point.y, 4, 0, Math.PI * 2);
    ctx.fillStyle = pose3DMarkerColor(pose3dData?.markers[index] || "");
    ctx.fill();
  });
}

function pose3DMarkerColor(marker: string): string {
  const side = pose3DMarkerSide(marker);
  if (side === "augmented") return "rgba(185, 246, 165, 0.48)";
  if (side === "right") return "#ff6b4a";
  if (side === "left") return "#4aa3ff";
  if (side === "center") return "#f6d34a";
  return "#d7ff43";
}

function pose3DLineColor(markerA: string, markerB: string): string {
  const sideA = pose3DMarkerSide(markerA);
  const sideB = pose3DMarkerSide(markerB);
  if (sideA === "augmented" || sideB === "augmented") return "rgba(185, 246, 165, 0.36)";
  if (sideA === sideB && sideA === "right") return "rgba(255, 107, 74, 0.7)";
  if (sideA === sideB && sideA === "left") return "rgba(74, 163, 255, 0.7)";
  if (sideA === sideB && sideA === "center") return "rgba(246, 211, 74, 0.72)";
  return "rgba(246, 244, 233, 0.42)";
}

function pose3DMarkerSide(marker: string): "left" | "right" | "center" | "augmented" | "other" {
  const normalized = marker.trim();
  if (/_study$/i.test(normalized)) return "augmented";
  if (/^(Hip|Neck|Head|Nose)$/i.test(normalized)) return "center";
  if (/^(L|l[._])/.test(normalized)) return "left";
  if (/^(R|r[._])/.test(normalized)) return "right";
  return "other";
}

function pose3DScene(data: Pose3DData): { center: [number, number, number]; scale: number } {
  const bounds = { minX: Infinity, maxX: -Infinity, minY: Infinity, maxY: -Infinity, minZ: Infinity, maxZ: -Infinity };
  data.frames.forEach((frame) => frame.forEach(([x, y, z]) => {
    if (x === null || y === null || z === null) return;
    bounds.minX = Math.min(bounds.minX, x);
    bounds.maxX = Math.max(bounds.maxX, x);
    bounds.minY = Math.min(bounds.minY, y);
    bounds.maxY = Math.max(bounds.maxY, y);
    bounds.minZ = Math.min(bounds.minZ, z);
    bounds.maxZ = Math.max(bounds.maxZ, z);
  }));
  if (!Number.isFinite(bounds.minX)) return { center: [0, 0, 0], scale: 1 };
  const span = Math.max(bounds.maxX - bounds.minX, bounds.maxY - bounds.minY, bounds.maxZ - bounds.minZ, 0.001);
  return {
    center: [(bounds.minX + bounds.maxX) / 2, (bounds.minY + bounds.maxY) / 2, (bounds.minZ + bounds.maxZ) / 2],
    scale: 280 / span,
  };
}

function projectPose3D(
  point: [number | null, number | null, number | null],
  scene: { center: [number, number, number]; scale: number },
  width: number,
  height: number,
): { x: number; y: number; depth: number } | null {
  const [x, y, z] = point;
  if (x === null || y === null || z === null) return null;
  const dx = (x - scene.center[0]) * scene.scale * pose3dCamera.zoom;
  const dy = (y - scene.center[1]) * scene.scale * pose3dCamera.zoom;
  const dz = (z - scene.center[2]) * scene.scale * pose3dCamera.zoom;
  const cosY = Math.cos(pose3dCamera.rotY);
  const sinY = Math.sin(pose3dCamera.rotY);
  const rx = dx * cosY + dz * sinY;
  const rz = -dx * sinY + dz * cosY;
  const cosX = Math.cos(pose3dCamera.rotX);
  const sinX = Math.sin(pose3dCamera.rotX);
  const ry = dy * cosX - rz * sinX;
  const rz2 = dy * sinX + rz * cosX;
  return { x: rx + width / 2 + pose3dCamera.panX, y: -ry + height / 2 + pose3dCamera.panY, depth: rz2 };
}

function drawPose3DAxes(ctx: CanvasRenderingContext2D, scene: { center: [number, number, number]; scale: number }, width: number, height: number): void {
  const origin = projectPose3D([0, 0, 0], scene, width, height);
  if (!origin) return;
  const axisLength = 80 / Math.max(scene.scale, 0.001);
  [
    { point: [axisLength, 0, 0] as [number, number, number], color: "#ff4d57", label: "X" },
    { point: [0, axisLength, 0] as [number, number, number], color: "#63d87b", label: "Y" },
    { point: [0, 0, axisLength] as [number, number, number], color: "#50a0ff", label: "Z" },
  ].forEach((axis) => {
    const target = projectPose3D(axis.point, scene, width, height);
    if (!target) return;
    ctx.beginPath();
    ctx.moveTo(origin.x, origin.y);
    ctx.lineTo(target.x, target.y);
    ctx.strokeStyle = axis.color;
    ctx.lineWidth = 2;
    ctx.stroke();
    ctx.fillStyle = axis.color;
    ctx.fillText(axis.label, target.x + 5, target.y + 5);
  });
}

function pose3DBones(markers: string[]): Array<[number, number]> {
  const markerIndex = new Map(markers.map((marker, index) => [marker, index]));
  return [
    ["Hip", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["RAnkle", "RBigToe"],
    ["RBigToe", "RSmallToe"], ["RAnkle", "RHeel"], ["Hip", "LHip"], ["LHip", "LKnee"],
    ["LKnee", "LAnkle"], ["LAnkle", "LBigToe"], ["LBigToe", "LSmallToe"], ["LAnkle", "LHeel"],
    ["Hip", "Neck"], ["Neck", "Head"], ["Head", "Nose"], ["Neck", "RShoulder"],
    ["RShoulder", "RElbow"], ["RElbow", "RWrist"], ["Neck", "LShoulder"], ["LShoulder", "LElbow"],
    ["LElbow", "LWrist"],
  ].flatMap(([a, b]) => {
    const ai = markerIndex.get(a);
    const bi = markerIndex.get(b);
    return ai === undefined || bi === undefined ? [] : [[ai, bi] as [number, number]];
  });
}

function togglePose3DPlayback(): void {
  if (!pose3dData) return;
  if (pose3dPlaying) {
    stopPose3D();
    pauseAllVideos(false);
  } else {
    if (isPose3DAtEnd() || isPrimaryVideoAtEnd()) {
      resetSyncedPlaybackToStart();
    }
    syncVideosToPose3DTime();
    playAllVideos(false);
    startPose3DPlayback();
  }
}

function startPose3DPlayback(): void {
  if (!pose3dData || pose3dPlaying) return;
  if (isPose3DAtEnd() || isPrimaryVideoAtEnd()) {
    resetSyncedPlaybackToStart();
  }
  pose3dPlaying = true;
  pose3dLastTick = performance.now();
  updatePose3DControls();
  tickPose3D();
}

function tickPose3D(): void {
  if (!pose3dPlaying || !pose3dData) return;
  pose3dAnimationId = window.requestAnimationFrame((time) => {
    const frameDuration = 1000 / (pose3dData?.fps || 30);
    if (time - pose3dLastTick >= frameDuration) {
      pose3dLastTick = time;
      const primaryVideo = videoElements()[0];
      if (videosPlaying && primaryVideo) {
        setPose3DFrameFromTime(primaryVideo.currentTime, false);
      } else {
        pose3dFrame = (pose3dFrame + 1) % Math.max(1, pose3dData?.num_frames || 1);
        updatePose3DControls();
        drawPose3D();
        drawSelectedKinematicsChart();
      }
    }
    tickPose3D();
  });
}

function stopPose3D(): void {
  pose3dPlaying = false;
  if (pose3dAnimationId) {
    window.cancelAnimationFrame(pose3dAnimationId);
    pose3dAnimationId = 0;
  }
  updatePose3DControls();
}

function canUsePose3DPictureInPicture(): boolean {
  return Boolean(
    pose3dCanvas &&
    "captureStream" in pose3dCanvas &&
    document.pictureInPictureEnabled &&
    HTMLVideoElement.prototype.requestPictureInPicture,
  );
}

async function togglePose3DPictureInPicture(): Promise<void> {
  if (document.pictureInPictureElement) {
    await document.exitPictureInPicture();
    return;
  }
  if (!pose3dData || !pose3dCanvas) return;
  if (!canUsePose3DPictureInPicture()) {
    log("3D Keypoints Picture-in-Picture is not supported in this browser.");
    return;
  }
  drawPose3D();
  const stream = pose3dCanvas.captureStream(Math.max(1, Math.round(pose3dData.fps || 30)));
  const video = pose3dPipVideo || document.createElement("video");
  pose3dPipVideo = video;
  video.muted = true;
  video.playsInline = true;
  video.srcObject = stream;
  await video.play();
  await video.requestPictureInPicture();
}

async function loadKinematicsResults(): Promise<void> {
  const session = selectedSession();
  kinematicsSummary = null;
  selectedKinematicsSignals.clear();
  kinematicsTimeseriesCache.clear();
  if (!kinematicsGrid) return;
  kinematicsGrid.innerHTML = `<p class="empty">No kinematics results yet.</p>`;
  clearKinematicsChart();
  if (!session) return;
  try {
    const params = new URLSearchParams({ session_path: session.session_path });
    kinematicsSummary = await fetchJson<KinematicsSummary>(`/api/analysis/kinematics?${params.toString()}`);
    renderKinematicsCards();
  } catch {
    kinematicsGrid.innerHTML = `<p class="empty">No kinematics results yet.</p>`;
  }
}

function renderKinematicsCards(): void {
  if (!kinematicsGrid || !kinematicsSummary?.available) {
    if (kinematicsGrid) kinematicsGrid.innerHTML = `<p class="empty">No kinematics results yet.</p>`;
    return;
  }
  const signals = kinematicsSummary.signals.filter((signal) => (
    signal.kind === selectedKinematicsKind && signal.category === selectedKinematicsCategory
  ));
  if (signals.length === 0) {
    kinematicsGrid.innerHTML = `<p class="empty">No ${selectedKinematicsKind} signals for this category.</p>`;
    return;
  }
  kinematicsGrid.innerHTML = signals.map((signal) => {
    const active = selectedKinematicsSignals.has(signal.key);
    const color = active ? kinematicsSignalColor(signal.key) : "";
    return `
      <button class="kinematics-card ${active ? "is-selected" : ""}" type="button" data-kin-signal="${signal.key}" style="${active ? `--kinematics-card-color: ${color};` : ""}">
        <span>
          <strong>${signal.label}</strong>
          <small>${signal.side}</small>
        </span>
      </button>
    `;
  }).join("");
}

function kinematicsSignalColor(signalKey: string): string {
  const selected = Array.from(selectedKinematicsSignals);
  const index = Math.max(0, selected.indexOf(signalKey));
  return kinematicsChartColors[index % kinematicsChartColors.length];
}

async function toggleKinematicsTimeseries(signalKey: string): Promise<void> {
  const session = selectedSession();
  if (!session) return;
  if (selectedKinematicsSignals.has(signalKey)) {
    selectedKinematicsSignals.delete(signalKey);
    renderKinematicsCards();
    drawSelectedKinematicsChart();
    return;
  }
  selectedKinematicsSignals.add(signalKey);
  if (!kinematicsTimeseriesCache.has(signalKey)) {
    const params = new URLSearchParams({ session_path: session.session_path, signal: signalKey });
    const data = await fetchJson<KinematicsTimeseries>(`/api/analysis/kinematics/timeseries?${params.toString()}`);
    kinematicsTimeseriesCache.set(signalKey, data);
  }
  renderKinematicsCards();
  drawSelectedKinematicsChart();
}

function drawSelectedKinematicsChart(): void {
  const series = Array.from(selectedKinematicsSignals)
    .map((key) => {
      const signal = kinematicsSummary?.signals.find((item) => item.key === key);
      const data = kinematicsTimeseriesCache.get(key);
      return signal && data ? { signal, data } : null;
    })
    .filter((item): item is { signal: KinematicsSignal; data: KinematicsTimeseries } => item !== null);
  if (series.length === 0) {
    clearKinematicsChart();
    return;
  }
  if (kinematicsChartTitle) {
    kinematicsChartTitle.textContent = series.map((item) => `${item.signal.label} ${item.signal.side}`).join(", ");
  }
  drawKinematicsChart(series);
}

function drawKinematicsChart(series: Array<{ signal: KinematicsSignal; data: KinematicsTimeseries }>): void {
  if (!kinematicsChart) return;
  const ctx = kinematicsChart.getContext("2d");
  const wrapper = kinematicsChart.parentElement;
  if (!ctx || !wrapper) return;
  const dpr = window.devicePixelRatio || 1;
  const width = wrapper.clientWidth || 640;
  const height = kinematicsChartHeight;
  sizeKinematicsChartCanvas(width, height, dpr);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#05080a";
  ctx.fillRect(0, 0, width, height);
  const prepared = series.map((item) => ({
    ...item,
    values: item.data.values
      .map((value, index) => ({ value, time: item.data.time[index] ?? index }))
      .filter((point): point is { value: number; time: number } => point.value !== null),
  })).filter((item) => item.values.length >= 2);
  if (prepared.length === 0) {
    ctx.fillStyle = "#aeb7b8";
    ctx.fillText("No time series data", 18, 34);
    return;
  }
  const allValues = prepared.flatMap((item) => item.values);
  const minTime = Math.min(...allValues.map((point) => point.time));
  const maxTime = Math.max(...allValues.map((point) => point.time));
  const minValue = Math.min(...allValues.map((point) => point.value));
  const maxValue = Math.max(...allValues.map((point) => point.value));
  const pad = 28;
  const leftPad = 42;
  const valueSpan = Math.max(1, maxValue - minValue);
  const plotWidth = width - leftPad - pad;
  const plotHeight = height - pad * 2;
  ctx.strokeStyle = "rgba(246, 244, 233, 0.16)";
  ctx.lineWidth = 1;
  for (let index = 0; index <= 4; index += 1) {
    const y = pad + (plotHeight / 4) * index;
    ctx.beginPath();
    ctx.moveTo(leftPad, y);
    ctx.lineTo(width - pad, y);
    ctx.stroke();
  }
  ctx.beginPath();
  ctx.moveTo(leftPad, pad);
  ctx.lineTo(leftPad, height - pad);
  ctx.lineTo(width - pad, height - pad);
  ctx.stroke();
  ctx.save();
  ctx.translate(14, height / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillStyle = "#aeb7b8";
  ctx.font = "800 12px system-ui";
  ctx.textAlign = "center";
  ctx.fillText(`Y (${prepared[0]?.data.unit || ""})`, 0, 0);
  ctx.restore();
  prepared.forEach((item, seriesIndex) => {
    ctx.strokeStyle = kinematicsChartColors[seriesIndex % kinematicsChartColors.length];
    ctx.lineWidth = 2;
    ctx.beginPath();
    item.values.forEach((point, index) => {
      const x = leftPad + ((point.time - minTime) / Math.max(0.001, maxTime - minTime)) * plotWidth;
      const y = height - pad - ((point.value - minValue) / valueSpan) * plotHeight;
      if (index === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();
    ctx.fillStyle = kinematicsChartColors[seriesIndex % kinematicsChartColors.length];
    ctx.fillText(`${item.signal.label} ${item.signal.side}`, pad + 8, pad + 16 + seriesIndex * 16);
  });
  const currentTime = pose3DCurrentTime();
  if (currentTime !== null) {
    drawKinematicsTimeCursor(ctx, {
      time: currentTime,
      minTime,
      maxTime,
      leftPad,
      pad,
      plotWidth,
      height,
    });
  }
  if (kinematicsHoverX !== null && kinematicsHoverX >= leftPad && kinematicsHoverX <= width - pad) {
    drawKinematicsHover(ctx, prepared, {
      hoverX: kinematicsHoverX,
      minTime,
      maxTime,
      minValue,
      valueSpan,
      leftPad,
      pad,
      plotWidth,
      plotHeight,
      width,
      height,
    });
  }
  drawKinematicsEventMarkers(ctx, {
    series: prepared,
    events: kinematicsSummary?.events || [],
    hoverX: kinematicsHoverX,
    minTime,
    maxTime,
    minValue,
    valueSpan,
    leftPad,
    pad,
    plotWidth,
    plotHeight,
    width,
    height,
  });
}

function drawKinematicsEventMarkers(
  ctx: CanvasRenderingContext2D,
  bounds: {
    series: PreparedKinematicsSeries[];
    events: KinematicsEventMarker[];
    hoverX: number | null;
    minTime: number;
    maxTime: number;
    minValue: number;
    valueSpan: number;
    leftPad: number;
    pad: number;
    plotWidth: number;
    plotHeight: number;
    width: number;
    height: number;
  },
): void {
  const visibleEvents = bounds.events
    .filter((event) => Number.isFinite(event.time) && event.time >= bounds.minTime && event.time <= bounds.maxTime)
    .map((event) => ({
      event,
      x: bounds.leftPad + ((event.time - bounds.minTime) / Math.max(0.001, bounds.maxTime - bounds.minTime)) * bounds.plotWidth,
    }));
  visibleEvents.forEach(({ event, x }) => {
    ctx.strokeStyle = "rgba(255, 214, 102, 0.38)";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(x, bounds.pad);
    ctx.lineTo(x, bounds.height - bounds.pad);
    ctx.stroke();
    ctx.fillStyle = "rgba(255, 214, 102, 0.72)";
    ctx.font = "900 11px system-ui";
    ctx.fillText(event.label, x + 5, bounds.pad + 14);
  });
  if (bounds.hoverX === null) return;
  const hovered = visibleEvents.find(({ x }) => Math.abs(x - bounds.hoverX!) <= 8);
  if (!hovered) return;
  drawKinematicsEventTooltip(ctx, hovered.event, hovered.x, bounds);
}

function drawKinematicsEventTooltip(
  ctx: CanvasRenderingContext2D,
  event: KinematicsEventMarker,
  x: number,
  bounds: {
    series: PreparedKinematicsSeries[];
    minTime: number;
    maxTime: number;
    minValue: number;
    valueSpan: number;
    leftPad: number;
    pad: number;
    plotWidth: number;
    plotHeight: number;
    width: number;
    height: number;
  },
): void {
  const rows = kinematicsRowsAtTime(bounds.series, event.time, bounds);
  rows.forEach((row) => {
    ctx.beginPath();
    ctx.arc(row.x, row.y, 4, 0, Math.PI * 2);
    ctx.fillStyle = row.color;
    ctx.fill();
  });
  const tooltipWidth = 300;
  const tooltipHeight = 64 + rows.length * 18;
  const tooltipX = Math.min(bounds.width - tooltipWidth - 12, Math.max(bounds.leftPad + 8, x + 12));
  const tooltipY = bounds.pad + 10;
  ctx.fillStyle = "rgba(5, 8, 10, 0.94)";
  ctx.strokeStyle = "rgba(255, 214, 102, 0.42)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.roundRect(tooltipX, tooltipY, tooltipWidth, tooltipHeight, 8);
  ctx.fill();
  ctx.stroke();
  ctx.fillStyle = "#ffd666";
  ctx.font = "900 12px system-ui";
  ctx.fillText(`${event.label} - ${event.description}`, tooltipX + 10, tooltipY + 18);
  ctx.fillStyle = "#dbe2e3";
  ctx.font = "800 12px system-ui";
  const frameText = event.frame === undefined ? "" : ` frame ${event.frame}`;
  ctx.fillText(`t ${event.time.toFixed(3)}${frameText}`, tooltipX + 10, tooltipY + 42);
  rows.forEach((row, index) => {
    ctx.fillStyle = row.color;
    ctx.fillText(
      `${row.item.signal.label} ${row.item.signal.side}: ${row.nearest.value.toFixed(2)} ${row.item.data.unit}`,
      tooltipX + 10,
      tooltipY + 62 + index * 18,
    );
  });
}

function drawKinematicsTimeCursor(
  ctx: CanvasRenderingContext2D,
  bounds: {
    time: number;
    minTime: number;
    maxTime: number;
    leftPad: number;
    pad: number;
    plotWidth: number;
    height: number;
  },
): void {
  if (bounds.time < bounds.minTime || bounds.time > bounds.maxTime) return;
  const x = bounds.leftPad + ((bounds.time - bounds.minTime) / Math.max(0.001, bounds.maxTime - bounds.minTime)) * bounds.plotWidth;
  ctx.strokeStyle = "#b8bec0";
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(x, bounds.pad);
  ctx.lineTo(x, bounds.height - bounds.pad);
  ctx.stroke();
  ctx.fillStyle = "#b8bec0";
  ctx.font = "900 11px system-ui";
  ctx.fillText("3D", x + 6, bounds.pad + 12);
}

function clearKinematicsChart(): void {
  if (kinematicsChartTitle) kinematicsChartTitle.textContent = "Select a signal";
  if (!kinematicsChart) return;
  const ctx = kinematicsChart.getContext("2d");
  const wrapper = kinematicsChart.parentElement;
  if (!ctx || !wrapper) return;
  const dpr = window.devicePixelRatio || 1;
  const width = wrapper.clientWidth || 640;
  const height = kinematicsChartHeight;
  sizeKinematicsChartCanvas(width, height, dpr);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.fillStyle = "#05080a";
  ctx.fillRect(0, 0, width, height);
}

function sizeKinematicsChartCanvas(width: number, height: number, dpr: number): void {
  if (!kinematicsChart) return;
  kinematicsChart.width = Math.round(width * dpr);
  kinematicsChart.height = Math.round(height * dpr);
  kinematicsChart.style.width = `${width}px`;
  kinematicsChart.style.height = `${height}px`;
}

function drawKinematicsHover(
  ctx: CanvasRenderingContext2D,
  series: PreparedKinematicsSeries[],
  bounds: {
    hoverX: number;
    minTime: number;
    maxTime: number;
    minValue: number;
    valueSpan: number;
    leftPad: number;
    pad: number;
    plotWidth: number;
    plotHeight: number;
    width: number;
    height: number;
  },
): void {
  const hoverTime = bounds.minTime + ((bounds.hoverX - bounds.leftPad) / Math.max(1, bounds.plotWidth)) * (bounds.maxTime - bounds.minTime);
  const rows = kinematicsRowsAtTime(series, hoverTime, bounds);
  if (rows.length === 0) return;
  const x = rows[0].x;
  ctx.strokeStyle = "rgba(184, 190, 192, 0.55)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(x, bounds.pad);
  ctx.lineTo(x, bounds.height - bounds.pad);
  ctx.stroke();
  rows.forEach((row) => {
    ctx.beginPath();
    ctx.arc(row.x, row.y, 4, 0, Math.PI * 2);
    ctx.fillStyle = row.color;
    ctx.fill();
  });
  const tooltipWidth = 230;
  const tooltipHeight = 24 + rows.length * 18;
  const tooltipX = Math.min(bounds.width - tooltipWidth - 12, Math.max(bounds.leftPad + 8, x + 12));
  const tooltipY = bounds.pad + 10;
  ctx.fillStyle = "rgba(5, 8, 10, 0.92)";
  ctx.strokeStyle = "rgba(246, 244, 233, 0.18)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.roundRect(tooltipX, tooltipY, tooltipWidth, tooltipHeight, 8);
  ctx.fill();
  ctx.stroke();
  ctx.font = "800 12px system-ui";
  ctx.fillStyle = "#aeb7b8";
  ctx.fillText(`t ${rows[0].nearest.time.toFixed(3)}`, tooltipX + 10, tooltipY + 17);
  rows.forEach((row, index) => {
    ctx.fillStyle = row.color;
    ctx.fillText(
      `${row.item.signal.label} ${row.item.signal.side}: ${row.nearest.value.toFixed(2)} ${row.item.data.unit}`,
      tooltipX + 10,
      tooltipY + 37 + index * 18,
    );
  });
}

function kinematicsRowsAtTime(
  series: PreparedKinematicsSeries[],
  targetTime: number,
  bounds: {
    minTime: number;
    maxTime: number;
    minValue: number;
    valueSpan: number;
    leftPad: number;
    pad: number;
    plotWidth: number;
    plotHeight: number;
    height: number;
  },
): Array<{
  item: PreparedKinematicsSeries;
  nearest: { value: number; time: number };
  x: number;
  y: number;
  color: string;
}> {
  return series.map((item, index) => {
    const nearest = item.values.reduce((best, point) => (
      Math.abs(point.time - targetTime) < Math.abs(best.time - targetTime) ? point : best
    ), item.values[0]);
    const x = bounds.leftPad + ((nearest.time - bounds.minTime) / Math.max(0.001, bounds.maxTime - bounds.minTime)) * bounds.plotWidth;
    const y = bounds.height - bounds.pad - ((nearest.value - bounds.minValue) / bounds.valueSpan) * bounds.plotHeight;
    const color = kinematicsChartColors[index % kinematicsChartColors.length];
    return { item, nearest, x, y, color };
  });
}

function selectedKinematicsTimeRange(): { minTime: number; maxTime: number } | null {
  const times = Array.from(selectedKinematicsSignals).flatMap((key) => {
    const data = kinematicsTimeseriesCache.get(key);
    if (!data) return [];
    return data.values.flatMap((value, index) => {
      const time = data.time[index];
      return value !== null && typeof time === "number" && Number.isFinite(time) ? [time] : [];
    });
  });
  if (times.length < 2) return null;
  return { minTime: Math.min(...times), maxTime: Math.max(...times) };
}

function kinematicsChartTimeFromClientX(clientX: number): number | null {
  if (!kinematicsChart) return null;
  const range = selectedKinematicsTimeRange();
  if (!range) return null;
  const rect = kinematicsChart.getBoundingClientRect();
  const leftPad = 42;
  const pad = 28;
  const plotWidth = Math.max(1, rect.width - leftPad - pad);
  const x = Math.max(leftPad, Math.min(rect.width - pad, clientX - rect.left));
  return range.minTime + ((x - leftPad) / plotWidth) * (range.maxTime - range.minTime);
}

function syncPose3DFromKinematicsChart(clientX: number): void {
  const time = kinematicsChartTimeFromClientX(clientX);
  if (time === null) return;
  setPose3DFrameFromTime(time);
  syncVideosToPose3DTime();
}

function renderConfigForm(): void {
  if (!configForm) return;
  configForm.innerHTML = orderedConfigEntries().map(([section, value]) => {
    if (!isObject(value)) return "";
    return `
      <fieldset>
        <legend>${section}</legend>
        <div class="analysis-config-grid">
          ${renderConfigFields(section, value as Record<string, unknown>)}
        </div>
      </fieldset>
    `;
  }).join("");
}

function orderedConfigEntries(): Array<[string, unknown]> {
  const entries = Object.entries(config);
  const entryMap = new Map(entries);
  const ordered = configSectionOrder
    .filter((section) => entryMap.has(section))
    .map((section) => [section, entryMap.get(section)] as [string, unknown]);
  const remaining = entries.filter(([section]) => !configSectionOrder.includes(section));
  return [...ordered, ...remaining];
}

function renderConfigFields(prefix: string, values: Record<string, unknown>): string {
  if (prefix === "lifting") {
    return renderLiftingConfigFields(values);
  }
  if (prefix === "kinematics") {
    return renderKinematicsConfigFields(values);
  }
  return orderedConfigFieldEntries(prefix, values).map(([key, value]) => {
    const path = `${prefix}.${key}`;
    return renderConfigField(path, key, value);
  }).join("");
}

function orderedConfigFieldEntries(prefix: string, values: Record<string, unknown>): Array<[string, unknown]> {
  const entries = Object.entries(values);
  const preferred = configFieldOrder[prefix];
  if (!preferred) return entries;
  const entryMap = new Map(entries);
  const ordered = preferred
    .filter((key) => entryMap.has(key))
    .map((key) => [key, entryMap.get(key)] as [string, unknown]);
  const remaining = entries.filter(([key]) => !preferred.includes(key));
  return [...ordered, ...remaining];
}

function renderLiftingConfigFields(values: Record<string, unknown>): string {
  const groupedKeys = new Set(liftingConfigGroups.flatMap((group) => group.paths));
  const groups = liftingConfigGroups.map((group) => {
    const fields = group.paths
      .map((key) => renderConfigField(`lifting.${key}`, key, values[key]))
      .join("");
    if (!fields.trim()) return "";
    return `
      <div class="analysis-config-mode-group">
        <strong>${group.title}</strong>
        <div class="analysis-config-grid">${fields}</div>
      </div>
    `;
  });
  const remaining = Object.entries(values)
    .filter(([key]) => !groupedKeys.has(key))
    .map(([key, value]) => renderConfigField(`lifting.${key}`, key, value))
    .join("");
  return [...groups, remaining].join("");
}

function renderKinematicsConfigFields(values: Record<string, unknown>): string {
  const groupedKeys = new Set(kinematicsConfigGroups.flatMap((group) => group.paths.map((path) => path.split(".")[0])));
  const regularFields = orderedConfigFieldEntries("kinematics", values)
    .filter(([key]) => !groupedKeys.has(key))
    .map(([key, value]) => renderConfigField(`kinematics.${key}`, key, value))
    .join("");
  const groups = kinematicsConfigGroups.map((group) => {
    const fields = group.paths.map((path) => {
      const parts = path.split(".");
      let value: unknown = values;
      parts.forEach((part) => {
        value = isObject(value) ? value[part] : undefined;
      });
      return renderConfigField(`kinematics.${path}`, parts[parts.length - 1], value);
    }).join("");
    if (!fields.trim()) return "";
    return `
      <div class="analysis-config-mode-group">
        <strong>${group.title}</strong>
        <div class="analysis-config-grid">${fields}</div>
      </div>
    `;
  });
  return [regularFields, ...groups].join("");
}

function renderConfigField(path: string, key: string, value: unknown): string {
  if (hiddenConfigPaths.has(path) || value === undefined) {
    return "";
  }
  if (path === "base.frame_range") {
    return renderFrameRangeControl(value);
  }
  if (path === "pose.mode") {
    return renderSegmentedControl(path, key, String(value || "normal"), ["normal", "performance"]);
  }
  if (selectConfigOptions[path]) {
    return renderSelectControl(path, key, String(value ?? ""), selectConfigOptions[path]);
  }
  if (isObject(value)) {
    return `<div class="analysis-config-subgroup"><strong>${key}</strong>${renderConfigFields(path, value as Record<string, unknown>)}</div>`;
  }
  if (typeof value === "boolean") {
    return `<label class="toggle-row"><span>${key}</span><input type="checkbox" data-config-path="${path}" ${value ? "checked" : ""}></label>`;
  }
  const type = typeof value === "number" ? "number" : "text";
  const step = typeof value === "number" && !Number.isInteger(value) ? "0.01" : "1";
  return `<label>${key}<input data-config-path="${path}" type="${type}" step="${step}" value="${value ?? ""}"></label>`;
}

function frameRangeMax(): number {
  const counts = selectedSession()?.videos.map((video) => Number(video.frame_count || 0)).filter((count) => count > 0) || [];
  return Math.max(1, counts.length > 0 ? Math.min(...counts) - 1 : 320);
}

function frameRangeValue(value: unknown, maxFrame: number): [number, number] {
  if (Array.isArray(value) && value.length >= 2) {
    return [
      Math.max(0, Math.min(maxFrame, Number(value[0]) || 0)),
      Math.max(0, Math.min(maxFrame, Number(value[1]) || maxFrame)),
    ];
  }
  return [0, maxFrame];
}

function renderFrameRangeControl(value: unknown): string {
  const maxFrame = frameRangeMax();
  const [startRaw, endRaw] = frameRangeValue(value, maxFrame);
  const start = Math.min(startRaw, endRaw);
  const end = Math.max(startRaw, endRaw);
  return `
    <div class="frame-range-control" data-frame-range-control>
      <div class="frame-range-label-row">
        <span>frame_range</span>
      </div>
      <div class="frame-range-slider">
        <input type="range" min="0" max="${maxFrame}" step="1" value="${start}" data-frame-range-min>
        <input type="range" min="0" max="${maxFrame}" step="1" value="${end}" data-frame-range-max>
      </div>
      <div class="frame-range-values">
        <span data-frame-range-min-label>f${start}</span>
        <span data-frame-range-max-label>f${end}</span>
      </div>
    </div>
  `;
}

function renderSegmentedControl(path: string, key: string, value: string, options: string[]): string {
  return `
    <div class="segmented-config-control" data-segmented-control>
      <span>${key}</span>
      <input type="hidden" data-config-path="${path}" value="${value}">
      <div class="segmented-options">
        ${options.map((option) => `
          <button type="button" class="${option === value ? "active" : ""}" data-segment-value="${option}">
            ${option}
          </button>
        `).join("")}
      </div>
    </div>
  `;
}

function renderSelectControl(path: string, key: string, value: string, options: string[]): string {
  return `
    <label>${key}
      <select data-config-path="${path}">
        ${options.map((option) => `<option value="${option}" ${option === value ? "selected" : ""}>${option}</option>`).join("")}
      </select>
    </label>
  `;
}

function readConfigForm(): Record<string, unknown> {
  const next = structuredClone(config) as Record<string, unknown>;
  configForm?.querySelectorAll<HTMLInputElement | HTMLSelectElement>("[data-config-path]").forEach((input) => {
    setConfigPathValue(next, String(input.dataset.configPath || "").split("."), inputValue(input));
  });
  const frameRange = configForm?.querySelector<HTMLElement>("[data-frame-range-control]");
  if (frameRange) {
    const start = Number(frameRange.querySelector<HTMLInputElement>("[data-frame-range-min]")?.value || "0");
    const end = Number(frameRange.querySelector<HTMLInputElement>("[data-frame-range-max]")?.value || String(frameRangeMax()));
    setConfigPathValue(next, ["base", "frame_range"], [Math.min(start, end), Math.max(start, end)]);
  }
  return next;
}

function logAnalysisConfig(analysisConfig: Record<string, unknown>): void {
  log("Analysis settings");
  configSectionOrder.forEach((section) => {
    if (!(section in analysisConfig)) return;
    log(`[${section}]`);
    log(JSON.stringify(analysisConfig[section], null, 2));
  });
}

function setConfigPathValue(targetConfig: Record<string, unknown>, path: string[], value: unknown): void {
  let target: Record<string, unknown> = targetConfig;
  while (path.length > 1) {
    const key = path.shift() as string;
    target = target[key] as Record<string, unknown>;
  }
  const key = path[0];
  target[key] = value;
}

function inputValue(input: HTMLInputElement | HTMLSelectElement): unknown {
  if (input instanceof HTMLInputElement) {
    return input.type === "checkbox" ? input.checked : input.type === "number" ? Number(input.value) : input.value;
  }
  return input.value;
}

function syncFrameRangeControl(control: HTMLElement): void {
  const minInput = control.querySelector<HTMLInputElement>("[data-frame-range-min]");
  const maxInput = control.querySelector<HTMLInputElement>("[data-frame-range-max]");
  const minLabel = control.querySelector<HTMLElement>("[data-frame-range-min-label]");
  const maxLabel = control.querySelector<HTMLElement>("[data-frame-range-max-label]");
  if (!minInput || !maxInput) return;
  let start = Number(minInput.value || "0");
  let end = Number(maxInput.value || "0");
  if (start > end) {
    [start, end] = [end, start];
    minInput.value = String(start);
    maxInput.value = String(end);
  }
  if (minLabel) minLabel.textContent = `f${start}`;
  if (maxLabel) maxLabel.textContent = `f${end}`;
}

function handleConfigFormInput(event: Event): void {
  const target = event.target;
  if (!(target instanceof HTMLElement)) return;
  const frameRange = target.closest<HTMLElement>("[data-frame-range-control]");
  if (frameRange) {
    syncFrameRangeControl(frameRange);
  }
}

function handleConfigFormClick(event: MouseEvent): void {
  const button = (event.target as HTMLElement | null)?.closest<HTMLButtonElement>("[data-segment-value]");
  if (!button) return;
  const control = button.closest<HTMLElement>("[data-segmented-control]");
  const hiddenInput = control?.querySelector<HTMLInputElement>("input[type='hidden'][data-config-path]");
  if (!control || !hiddenInput) return;
  hiddenInput.value = String(button.dataset.segmentValue || "");
  control.querySelectorAll<HTMLButtonElement>("[data-segment-value]").forEach((option) => {
    option.classList.toggle("active", option === button);
  });
}

function isObject(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value);
}

function keypointCanvases(): HTMLCanvasElement[] {
  return Array.from(keypointModal?.querySelectorAll<HTMLCanvasElement>("[data-keypoint-canvas]") || []);
}

function activeVideo(): HTMLVideoElement | null {
  if (keypointEditEnabled && keypointEditVideo) return keypointEditVideo;
  return videoElements()[keypointActiveVideoIndex] || null;
}

function activeVideoInfo(): SessionVideo | null {
  return selectedSession()?.videos[keypointActiveVideoIndex] || null;
}

function currentKeypointFrameIndex(): number {
  const video = activeVideo();
  const info = activeVideoInfo();
  const fps = Math.max(1, Number(info?.fps || 30));
  const frameCount = Math.max(1, Number(info?.frame_count || 1));
  return Math.max(0, Math.min(frameCount - 1, Math.round((video?.currentTime || 0) * fps)));
}

function keypointFrameKey(frame = currentKeypointFrameIndex()): string | null {
  const session = selectedSession();
  const info = activeVideoInfo();
  if (!session || !info) return null;
  return `${session.session_path}|${info.camera_label}|${frame}`;
}

function cloneKeypointPeople(people: KeypointPerson[]): KeypointPerson[] {
  return people.map((person) => ({
    person_index: person.person_index,
    keypoints: person.keypoints.map((point) => ({ ...point })),
  }));
}

function cacheCurrentKeypointFrame(): void {
  if (!keypointFrameDirty || !keypointFrameData) return;
  const key = keypointFrameKey(keypointFrameData.frame);
  if (!key) return;
  keypointDirtyFrames.set(key, cloneKeypointPeople(keypointFrameData.people));
}

function markKeypointFrameDirty(): void {
  keypointFrameDirty = true;
  cacheCurrentKeypointFrame();
}

function updateKeypointFrameControls(): void {
  const info = activeVideoInfo();
  const frameCount = Math.max(0, Number(info?.frame_count || 0));
  const frame = currentKeypointFrameIndex();
  if (keypointFrameSeek) {
    keypointFrameSeek.max = String(Math.max(0, frameCount - 1));
    keypointFrameSeek.value = String(frame);
    keypointFrameSeek.disabled = !keypointEditEnabled || frameCount <= 0;
  }
  if (keypointFrameLabel) {
    keypointFrameLabel.textContent = frameCount > 0 ? `${frame + 1} / ${frameCount}` : "0 / 0";
  }
}

function updateKeypointEditorControls(): void {
  const session = selectedSession();
  const enabled = keypointEditEnabled && Boolean(session);
  if (keypointModal) keypointModal.hidden = !keypointEditEnabled;
  if (keypointEditToggle) keypointEditToggle.textContent = "Change keypoint";
  [keypointSaveButton, keypointNextVideoButton, keypointSwapLowerButton, keypointSwapUpperButton].forEach((button) => {
    if (button) button.disabled = !enabled || keypointSaving;
  });
  if (keypointCloseButton) keypointCloseButton.disabled = keypointSaving;
  if (keypointUndoButton) keypointUndoButton.disabled = !enabled || keypointSaving || keypointUndoStack.length === 0;
  if (keypointSaveStatus && !keypointSaving) keypointSaveStatus.textContent = "";
  keypointCanvases().forEach((canvas) => {
    canvas.classList.toggle("is-enabled", enabled);
  });
  updateKeypointFrameControls();
}

function openKeypointEditor(): void {
  const info = activeVideoInfo();
  if (!info || !keypointEditVideo) return;
  const currentTime = Number(videoSeek?.value || videoElements()[keypointActiveVideoIndex]?.currentTime || "0");
  keypointEditEnabled = true;
  keypointFrameData = null;
  keypointFrameRequestKey = "";
  keypointSelectedPersonIndex = null;
  keypointFrameDirty = false;
  keypointDirtyFrames.clear();
  keypointUndoStack = [];
  keypointView = { zoom: 1, panX: 0, panY: 0 };
  keypointEditVideo.src = info.video_url;
  keypointEditVideo.currentTime = currentTime;
  updateKeypointEditorControls();
  loadCurrentKeypointFrame().catch((error) => log(error instanceof Error ? error.message : "Keypoint load failed"));
}

function closeKeypointEditor(): void {
  keypointEditEnabled = false;
  keypointFrameData = null;
  keypointFrameRequestKey = "";
  keypointUndoStack = [];
  keypointHoverTarget = null;
  keypointDragTarget = null;
  keypointSelectedPersonIndex = null;
  keypointFrameDirty = false;
  keypointDirtyFrames.clear();
  keypointPanDrag = null;
  keypointEditVideo?.pause();
  if (keypointEditVideo) {
    keypointEditVideo.style.transform = "";
  }
  updateKeypointEditorControls();
  drawKeypointOverlays();
}

function moveKeypointFrame(delta: number): void {
  const info = activeVideoInfo();
  if (!keypointEditEnabled || !keypointEditVideo || !info) return;
  cacheCurrentKeypointFrame();
  const frameCount = Math.max(1, Number(info.frame_count || 1));
  const fps = Math.max(1, Number(info.fps || 30));
  const nextFrame = Math.max(0, Math.min(frameCount - 1, currentKeypointFrameIndex() + delta));
  keypointFrameData = null;
  keypointFrameRequestKey = "";
  keypointFrameDirty = false;
  keypointEditVideo.currentTime = nextFrame / fps;
  updateKeypointFrameControls();
  loadCurrentKeypointFrame().catch((error) => log(error instanceof Error ? error.message : "Keypoint load failed"));
}

async function loadCurrentKeypointFrame(): Promise<void> {
  const session = selectedSession();
  const info = activeVideoInfo();
  if (!keypointEditEnabled || !session || !info || keypointDragTarget !== null) {
    drawKeypointOverlays();
    return;
  }
  const frame = currentKeypointFrameIndex();
  const requestKey = `${session.session_path}|${info.camera_label}|${frame}`;
  if (requestKey === keypointFrameRequestKey) {
    drawKeypointOverlays();
    return;
  }
  cacheCurrentKeypointFrame();
  keypointFrameRequestKey = requestKey;
  const cachedPeople = keypointDirtyFrames.get(requestKey);
  if (cachedPeople) {
    keypointFrameData = {
      camera_label: info.camera_label,
      frame,
      file: "",
      keypoint_names: keypointFrameData?.keypoint_names || [],
      people: cloneKeypointPeople(cachedPeople),
    };
  } else {
    keypointFrameData = await fetchJson<KeypointFrame>(`/api/analysis/keypoints/frame?${new URLSearchParams({
      session_path: session.session_path,
      camera_label: info.camera_label,
      frame: String(frame),
    }).toString()}`);
  }
  if (!keypointFrameData.people.some((person) => person.person_index === keypointSelectedPersonIndex)) {
    keypointSelectedPersonIndex = keypointFrameData.people.length > 0 ? keypointFrameData.people[0].person_index : null;
  }
  keypointFrameDirty = Boolean(cachedPeople);
  keypointUndoStack = [];
  updateKeypointFrameControls();
  updateKeypointEditorControls();
  drawKeypointOverlays();
}

function bindKeypointCanvasControls(): void {
  keypointCanvases().forEach((canvas, index) => {
    canvas.addEventListener("mousedown", (event) => handleKeypointMouseDown(event, index));
    canvas.addEventListener("mousemove", (event) => handleKeypointMouseMove(event, index));
    canvas.addEventListener("mouseleave", () => {
      keypointHoverTarget = null;
      drawKeypointOverlays();
    });
    canvas.addEventListener("wheel", handleKeypointWheel, { passive: false });
    canvas.addEventListener("contextmenu", (event) => event.preventDefault());
  });
}

function pushKeypointUndo(): void {
  if (!keypointFrameData) return;
  keypointUndoStack.push(keypointFrameData.people.map((person) => ({
    person_index: person.person_index,
    keypoints: person.keypoints.map((point) => ({ ...point })),
  })));
  keypointUndoStack = keypointUndoStack.slice(-30);
  updateKeypointEditorControls();
}

function keypointCanvasPoint(event: MouseEvent, canvas: HTMLCanvasElement): { x: number; y: number } {
  const rect = canvas.getBoundingClientRect();
  return { x: event.clientX - rect.left, y: event.clientY - rect.top };
}

function keypointContentRect(canvas: HTMLCanvasElement): { x: number; y: number; width: number; height: number; scale: number } {
  const video = activeVideo();
  const canvasWidth = canvas.clientWidth || 1;
  const canvasHeight = canvas.clientHeight || 1;
  const videoWidth = video?.videoWidth || canvasWidth;
  const videoHeight = video?.videoHeight || canvasHeight;
  const scale = Math.min(canvasWidth / Math.max(1, videoWidth), canvasHeight / Math.max(1, videoHeight));
  const width = videoWidth * scale;
  const height = videoHeight * scale;
  return {
    x: (canvasWidth - width) / 2,
    y: (canvasHeight - height) / 2,
    width,
    height,
    scale,
  };
}

function keypointVideoToCanvas(point: Keypoint2D | undefined, canvas: HTMLCanvasElement): { x: number; y: number } | null {
  if (!point || point.x === null || point.y === null || !Number.isFinite(point.x) || !Number.isFinite(point.y)) return null;
  const content = keypointContentRect(canvas);
  return {
    x: keypointView.panX + (content.x + point.x * content.scale) * keypointView.zoom,
    y: keypointView.panY + (content.y + point.y * content.scale) * keypointView.zoom,
  };
}

function keypointCanvasToVideo(x: number, y: number, canvas: HTMLCanvasElement): { x: number; y: number } {
  const content = keypointContentRect(canvas);
  return {
    x: ((x - keypointView.panX) / Math.max(0.001, keypointView.zoom) - content.x) / Math.max(0.001, content.scale),
    y: ((y - keypointView.panY) / Math.max(0.001, keypointView.zoom) - content.y) / Math.max(0.001, content.scale),
  };
}

function nearestKeypointIndex(x: number, y: number, canvas: HTMLCanvasElement): { personIndex: number; keypointIndex: number } | null {
  if (!keypointFrameData) return null;
  let bestTarget: { personIndex: number; keypointIndex: number } | null = null;
  let bestDistance = 14;
  keypointFrameData.people.forEach((person, personIndex) => {
    person.keypoints.forEach((point, keypointIndex) => {
      const projected = keypointVideoToCanvas(point, canvas);
      if (!projected) return;
      const distance = Math.hypot(projected.x - x, projected.y - y);
      if (distance < bestDistance) {
        bestDistance = distance;
        bestTarget = { personIndex, keypointIndex };
      }
    });
  });
  return bestTarget;
}

function keypointPersonBounds(person: KeypointPerson, canvas: HTMLCanvasElement): { x: number; y: number; width: number; height: number } | null {
  const points = person.keypoints
    .map((point) => keypointVideoToCanvas(point, canvas))
    .filter((point): point is { x: number; y: number } => point !== null);
  if (points.length === 0) return null;
  const minX = Math.min(...points.map((point) => point.x)) - 18;
  const minY = Math.min(...points.map((point) => point.y)) - 18;
  const maxX = Math.max(...points.map((point) => point.x)) + 18;
  const maxY = Math.max(...points.map((point) => point.y)) + 18;
  return { x: minX, y: minY, width: maxX - minX, height: maxY - minY };
}

function personIndexAtCanvasPoint(x: number, y: number, canvas: HTMLCanvasElement): number | null {
  if (!keypointFrameData) return null;
  for (const person of keypointFrameData.people) {
    const bounds = keypointPersonBounds(person, canvas);
    if (!bounds) continue;
    if (x >= bounds.x && x <= bounds.x + bounds.width && y >= bounds.y && y <= bounds.y + bounds.height) {
      return person.person_index;
    }
  }
  return null;
}

function handleKeypointMouseDown(event: MouseEvent, index: number): void {
  if (!keypointEditEnabled || index !== 0) return;
  const canvas = event.currentTarget as HTMLCanvasElement;
  const point = keypointCanvasPoint(event, canvas);
  const nearest = nearestKeypointIndex(point.x, point.y, canvas);
  if (event.button === 0 && nearest !== null) {
    pushKeypointUndo();
    keypointDragTarget = nearest;
    keypointSelectedPersonIndex = keypointFrameData?.people[nearest.personIndex]?.person_index ?? null;
    drawKeypointOverlays();
    return;
  }
  const selectedPerson = personIndexAtCanvasPoint(point.x, point.y, canvas);
  if (selectedPerson !== null) {
    keypointSelectedPersonIndex = selectedPerson;
    drawKeypointOverlays();
    return;
  }
  keypointPanDrag = { x: event.clientX, y: event.clientY, panX: keypointView.panX, panY: keypointView.panY };
}

function handleKeypointMouseMove(event: MouseEvent, index: number): void {
  if (!keypointEditEnabled || index !== 0) return;
  const canvas = event.currentTarget as HTMLCanvasElement;
  const point = keypointCanvasPoint(event, canvas);
  if (keypointDragTarget !== null && keypointFrameData) {
    const person = keypointFrameData.people[keypointDragTarget.personIndex];
    const currentPoint = person?.keypoints[keypointDragTarget.keypointIndex];
    if (person && currentPoint) {
      person.keypoints[keypointDragTarget.keypointIndex] = { ...currentPoint, ...keypointCanvasToVideo(point.x, point.y, canvas) };
      markKeypointFrameDirty();
    }
    drawKeypointOverlays();
    return;
  }
  if (keypointPanDrag) {
    keypointView.panX = keypointPanDrag.panX + event.clientX - keypointPanDrag.x;
    keypointView.panY = keypointPanDrag.panY + event.clientY - keypointPanDrag.y;
    applyKeypointVideoTransform();
    drawKeypointOverlays();
    return;
  }
  keypointHoverTarget = nearestKeypointIndex(point.x, point.y, canvas);
  drawKeypointOverlays();
}

function handleKeypointWheel(event: WheelEvent): void {
  if (!keypointEditEnabled) return;
  event.preventDefault();
  const canvas = event.currentTarget as HTMLCanvasElement;
  const point = keypointCanvasPoint(event, canvas);
  const previousZoom = keypointView.zoom;
  const nextZoom = Math.max(0.25, Math.min(8, previousZoom * (event.deltaY < 0 ? 1.1 : 0.9)));
  keypointView.panX = point.x - ((point.x - keypointView.panX) / previousZoom) * nextZoom;
  keypointView.panY = point.y - ((point.y - keypointView.panY) / previousZoom) * nextZoom;
  keypointView.zoom = nextZoom;
  applyKeypointVideoTransform();
  drawKeypointOverlays();
}

function applyKeypointVideoTransform(): void {
  if (!keypointEditVideo) return;
  keypointEditVideo.style.transformOrigin = "0 0";
  keypointEditVideo.style.transform = `translate(${keypointView.panX}px, ${keypointView.panY}px) scale(${keypointView.zoom})`;
}

function drawKeypointOverlays(): void {
  keypointCanvases().forEach((canvas, index) => {
    const ctx = canvas.getContext("2d");
    const video = keypointEditVideo;
    if (!ctx || !video) return;
    const dpr = window.devicePixelRatio || 1;
    const width = video.clientWidth || canvas.clientWidth || 1;
    const height = video.clientHeight || canvas.clientHeight || 1;
    canvas.width = Math.round(width * dpr);
    canvas.height = Math.round(height * dpr);
    canvas.style.width = `${width}px`;
    canvas.style.height = `${height}px`;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, width, height);
    applyKeypointVideoTransform();
    if (!keypointEditEnabled || index !== 0 || !keypointFrameData) return;
    keypointFrameData.people.forEach((person, personIndex) => {
      const bounds = keypointPersonBounds(person, canvas);
      if (bounds) {
        const selected = keypointSelectedPersonIndex === person.person_index;
        ctx.strokeStyle = selected ? "#ffffff" : "rgba(246, 244, 233, 0.45)";
        ctx.lineWidth = selected ? 3 : 1.5;
        ctx.strokeRect(bounds.x, bounds.y, bounds.width, bounds.height);
        ctx.fillStyle = selected ? "rgba(255, 255, 255, 0.92)" : "rgba(5, 8, 10, 0.78)";
        ctx.fillRect(bounds.x, Math.max(0, bounds.y - 22), 76, 20);
        ctx.fillStyle = selected ? "#05080a" : "#f6f4e9";
        ctx.font = "12px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace";
        ctx.fillText(`person ${person.person_index}`, bounds.x + 6, Math.max(14, bounds.y - 8));
      }
      keypointSkeletonPairs.forEach(([a, b]) => {
        const pa = keypointVideoToCanvas(person.keypoints[a], canvas);
        const pb = keypointVideoToCanvas(person.keypoints[b], canvas);
        if (!pa || !pb) return;
        ctx.beginPath();
        ctx.moveTo(pa.x, pa.y);
        ctx.lineTo(pb.x, pb.y);
        ctx.strokeStyle = keypointBoneColor(a, b);
        ctx.lineWidth = 2;
        ctx.stroke();
      });
      person.keypoints.forEach((kp, kpIndex) => {
        const projected = keypointVideoToCanvas(kp, canvas);
        if (!projected) return;
        const isHover = keypointHoverTarget?.personIndex === personIndex && keypointHoverTarget.keypointIndex === kpIndex;
        ctx.beginPath();
        ctx.arc(projected.x, projected.y, isHover ? 7 : 5, 0, Math.PI * 2);
        ctx.fillStyle = isHover ? "#ffffff" : keypointColor(kpIndex);
        ctx.fill();
        ctx.strokeStyle = "#05080a";
        ctx.lineWidth = 2;
        ctx.stroke();
      });
    });
    if (keypointHoverTarget !== null) {
      const kp = keypointFrameData.people[keypointHoverTarget.personIndex]?.keypoints[keypointHoverTarget.keypointIndex];
      const projected = keypointVideoToCanvas(kp, canvas);
      const name = keypointFrameData.keypoint_names[keypointHoverTarget.keypointIndex] || `Keypoint ${keypointHoverTarget.keypointIndex}`;
      if (projected) drawKeypointTooltip(ctx, name, projected.x, projected.y, width);
    }
  });
}

function drawKeypointTooltip(ctx: CanvasRenderingContext2D, text: string, x: number, y: number, width: number): void {
  ctx.font = "12px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace";
  const textWidth = ctx.measureText(text).width;
  const boxX = Math.max(6, Math.min(width - textWidth - 18, x + 10));
  const boxY = Math.max(6, y - 28);
  ctx.fillStyle = "rgba(5, 8, 10, 0.88)";
  ctx.fillRect(boxX, boxY, textWidth + 12, 22);
  ctx.fillStyle = "#f6f4e9";
  ctx.fillText(text, boxX + 6, boxY + 15);
}

function keypointColor(index: number): string {
  if ([5, 7, 9, 11, 13, 15, 20, 22, 24].includes(index)) return "#4aa3ff";
  if ([6, 8, 10, 12, 14, 16, 21, 23, 25].includes(index)) return "#ff6b4a";
  return "#d7ff43";
}

function keypointBoneColor(a: number, b: number): string {
  const left = "#4aa3ff";
  const right = "#ff6b4a";
  const colorA = keypointColor(a);
  const colorB = keypointColor(b);
  if (colorA === left && colorB === left) return "rgba(74, 163, 255, 0.68)";
  if (colorA === right && colorB === right) return "rgba(255, 107, 74, 0.68)";
  return "rgba(215, 255, 67, 0.52)";
}

function swapKeypointPairs(pairs: number[][]): void {
  if (!keypointFrameData) return;
  const targetPerson = keypointFrameData.people.find((person) => person.person_index === keypointSelectedPersonIndex);
  if (!targetPerson) {
    log("Select a person's bounding box before changing upper/lower body.");
    return;
  }
  pushKeypointUndo();
  pairs.forEach(([left, right]) => {
    if (left >= targetPerson.keypoints.length || right >= targetPerson.keypoints.length) return;
    const currentLeft = targetPerson.keypoints[left];
    targetPerson.keypoints[left] = targetPerson.keypoints[right];
    targetPerson.keypoints[right] = currentLeft;
  });
  markKeypointFrameDirty();
  drawKeypointOverlays();
}

async function saveCurrentKeypointFrame(): Promise<void> {
  const session = selectedSession();
  if (!session) return;
  cacheCurrentKeypointFrame();
  if (keypointDirtyFrames.size === 0) {
    log("No keypoint changes to save.");
    return;
  }
  keypointSaving = true;
  updateKeypointEditorControls();
  const saved = [];
  const renderedCameraLabels = new Set<string>();
  try {
    for (const [key, people] of keypointDirtyFrames.entries()) {
      const parts = key.split("|");
      const frame = Number(parts.pop() || "0");
      const cameraLabel = parts.pop() || "";
      if (keypointSaveStatus) keypointSaveStatus.textContent = `Saving ${cameraLabel} frame ${frame}...`;
      await postJson("/api/analysis/keypoints/frame", {
        session_path: session.session_path,
        camera_label: cameraLabel,
        frame,
        people,
      });
      renderedCameraLabels.add(cameraLabel);
      saved.push(`${cameraLabel} frame ${frame}`);
    }
    for (const cameraLabel of renderedCameraLabels) {
      if (keypointSaveStatus) keypointSaveStatus.textContent = `Rendering ${cameraLabel}_pose.mp4...`;
      await postJson("/api/analysis/keypoints/render-video", {
        session_path: session.session_path,
        camera_label: cameraLabel,
      });
    }
    keypointDirtyFrames.clear();
    keypointFrameDirty = false;
    log(`Saved keypoints and rendered pose video: ${saved.join(", ")}`);
    await loadSessions(sessionSelect?.value || "");
  } finally {
    keypointSaving = false;
    updateKeypointEditorControls();
  }
}

async function saveAndCloseKeypointEditor(): Promise<void> {
  try {
    await saveCurrentKeypointFrame();
    closeKeypointEditor();
  } catch (error) {
    log(error instanceof Error ? error.message : "Keypoint save failed");
    keypointSaving = false;
    updateKeypointEditorControls();
  }
}

function videoElements(): HTMLVideoElement[] {
  return Array.from(videoGrid?.querySelectorAll<HTMLVideoElement>("video") || []);
}

function playAllVideos(syncPose3D = true): void {
  if (syncPose3D) {
    if (isPose3DAtEnd() || isPrimaryVideoAtEnd()) {
      resetSyncedPlaybackToStart();
    }
    setPose3DFrameFromTime(Number(videoSeek?.value || videoElements()[0]?.currentTime || "0"), false);
  }
  videoElements().forEach((video) => {
    video.play().catch(() => undefined);
  });
  videosPlaying = true;
  if (togglePlayButton) togglePlayButton.textContent = "Pause";
  if (syncPose3D) {
    startPose3DPlayback();
  }
}

function pauseAllVideos(syncPose3D = true): void {
  videoElements().forEach((video) => video.pause());
  videosPlaying = false;
  if (togglePlayButton) togglePlayButton.textContent = "Play";
  if (syncPose3D) {
    stopPose3D();
  }
}

function toggleAllVideos(): void {
  if (videosPlaying) {
    pauseAllVideos();
  } else {
    playAllVideos();
  }
}

function bindVideoControls(currentTime = 0): void {
  const videos = videoElements();
  const speed = Number(videoSpeed?.value || "1");
  videos.forEach((video, index) => {
    video.playbackRate = speed;
    if (currentTime > 0) {
      seekVideo(video, currentTime);
    }
    video.addEventListener("loadedmetadata", updateSeekBounds);
    video.addEventListener("loadedmetadata", drawKeypointOverlays);
    if (index === 0) {
      video.addEventListener("timeupdate", updateSeekProgress);
      video.addEventListener("ended", handlePrimaryVideoEnded);
    }
  });
  updateSeekBounds();
  if (videosPlaying) playAllVideos();
}

function updateSeekBounds(): void {
  if (!videoSeek) return;
  const primaryDuration = videoElements()[0]?.duration;
  videoSeek.max = Number.isFinite(primaryDuration) && primaryDuration > 0 ? String(primaryDuration) : "0";
  updateVideoTimecode(Number(videoSeek.value || "0"));
}

function updateSeekProgress(): void {
  if (!videoSeek || seeking) return;
  const firstVideo = videoElements()[0];
  if (!firstVideo) return;
  const duration = Number(videoSeek.max);
  const isAtEnd = firstVideo.ended || (duration > 0 && duration - firstVideo.currentTime < 0.08);
  const currentTime = isAtEnd ? duration : firstVideo.currentTime;
  videoSeek.value = String(currentTime);
  updateVideoTimecode(currentTime);
  setPose3DFrameFromTime(currentTime, false);
  loadCurrentKeypointFrame().catch(() => undefined);
}

function handlePrimaryVideoEnded(): void {
  updateSeekProgress();
  resetSyncedPlaybackToStart();
  if (videosPlaying || pose3dPlaying) {
    videoElements().forEach((video) => {
      video.play().catch(() => undefined);
    });
    videosPlaying = true;
    if (togglePlayButton) togglePlayButton.textContent = "Pause";
    startPose3DPlayback();
  }
}

function seekAllVideos(): void {
  const time = Number(videoSeek?.value || "0");
  videoElements().forEach((video) => {
    seekVideo(video, time);
  });
  updateVideoTimecode(time);
  setPose3DFrameFromTime(time, !videosPlaying);
  loadCurrentKeypointFrame().catch(() => undefined);
  if (videosPlaying) {
    startPose3DPlayback();
  }
}

function seekVideo(video: HTMLVideoElement, time: number): void {
  const targetTime = Math.min(time, video.duration || time);
  try {
    video.currentTime = targetTime;
  } catch {
    video.addEventListener("loadedmetadata", () => {
      video.currentTime = Math.min(time, video.duration || time);
    }, { once: true });
  }
}

function setPlaybackSpeed(): void {
  const speed = Number(videoSpeed?.value || "1");
  videoElements().forEach((video) => {
    video.playbackRate = speed;
  });
}

function updateVideoTimecode(currentTime: number): void {
  if (!videoTimecode) return;
  const session = selectedSession();
  const primaryVideo = videoElements()[0];
  const duration = Number(videoSeek?.max || primaryVideo?.duration || 0);
  const fps = Number(session?.videos[0]?.fps || 0);
  const totalFrames = Number(session?.videos[0]?.frame_count || 0);
  const currentFrame = fps > 0
    ? Math.min(totalFrames || Number.POSITIVE_INFINITY, Math.max(0, Math.round(currentTime * fps)))
    : 0;
  videoTimecode.textContent = `${formatVideoTime(currentTime)} / ${formatVideoTime(duration)} (${currentFrame} / ${totalFrames || 0})`;
}

function formatVideoTime(seconds: number): string {
  const safeSeconds = Math.max(0, Number.isFinite(seconds) ? seconds : 0);
  const wholeSeconds = Math.floor(safeSeconds);
  const minutes = Math.floor(wholeSeconds / 60);
  const remainder = wholeSeconds % 60;
  return `${minutes}:${String(remainder).padStart(2, "0")}`;
}

async function runAnalysis(): Promise<void> {
  const session = selectedSession();
  if (!session || !runButton) return;
  runButton.disabled = true;
  logPanel!.textContent = "";
  log("Analysis queued");
  const analysisConfig = readConfigForm();
  logAnalysisConfig(analysisConfig);
  try {
    const job = await postJson<AnalysisJob>("/api/analysis/run", {
      session_path: session.session_path,
      config: analysisConfig,
    });
    pollJob(job.job_id);
  } catch (error) {
    log(error instanceof Error ? error.message : "Failed to start analysis");
    runButton.disabled = false;
  }
}

async function uploadCalibrationFile(): Promise<void> {
  const session = selectedSession();
  const file = calibrationFile?.files?.[0];
  if (calibrationFileName) calibrationFileName.textContent = file?.name || "No file selected";
  if (!session) {
    log("Select a session before choosing a calibration file.");
    return;
  }
  if (!file) {
    return;
  }
  const formData = new FormData();
  formData.set("session_path", session.session_path);
  formData.set("calibration_file", file);
  try {
    const response = await fetch("/api/analysis/calibration/upload", {
      method: "POST",
      headers: { Accept: "application/json" },
      body: formData,
    });
    if (!response.ok) throw new Error(await response.text());
    await response.json() as CalibrationUploadResult;
  } catch (error) {
    log(error instanceof Error ? error.message : "Calibration upload failed.");
  }
}

function clearCalibrationFile(): void {
  if (calibrationFile) calibrationFile.value = "";
  if (calibrationFileName) calibrationFileName.textContent = "No file selected";
}

async function pollJob(jobId: string): Promise<void> {
  let lastLogCount = 0;
  const timer = window.setInterval(async () => {
    const job = await fetchJson<AnalysisJob>(`/api/analysis/jobs/${jobId}`);
    job.logs.slice(lastLogCount).forEach((entry) => log(`[${entry.level}] ${entry.message}`));
    lastLogCount = job.logs.length;
    if (job.status === "completed" || job.status === "failed") {
      window.clearInterval(timer);
      log(job.status === "completed" ? "Analysis completed" : `Analysis failed: ${job.error || ""}`);
      if (runButton) runButton.disabled = false;
      await loadSessions(sessionSelect?.value || "");
      await loadAnalysisResults();
    }
  }, 1000);
}

selectRootButton?.addEventListener("click", async () => {
  if (!rootInput) return;
  let result = await postJson<{ root: string; cancelled: boolean; manual_required?: boolean }>("/api/analysis/session-root/select", {});
  if (result.manual_required) {
    const manualPath = window.prompt("Enter analysis session root", rootInput.value || result.root || "");
    if (!manualPath) return;
    result = await postJson<{ root: string; cancelled: boolean }>("/api/analysis/session-root/select", {
      root: manualPath,
      manual: true,
    });
  }
  if (!result.cancelled) {
    rootInput.value = result.root;
    await loadSessions();
  }
});

sessionSelect?.addEventListener("change", () => {
  closeKeypointEditor();
  keypointFrameData = null;
  keypointFrameRequestKey = "";
  keypointUndoStack = [];
  renderVideos();
  renderConfigForm();
  loadAnalysisResults().catch(() => undefined);
});
configForm?.addEventListener("input", handleConfigFormInput);
configForm?.addEventListener("click", handleConfigFormClick);
calibrationForm?.addEventListener("submit", (event) => event.preventDefault());
calibrationFile?.addEventListener("change", uploadCalibrationFile);
clearCalibrationButton?.addEventListener("click", clearCalibrationFile);
pose3dFileSelect?.addEventListener("change", () => loadSelectedPose3DFile().catch(() => undefined));
pose3dPlayButton?.addEventListener("click", togglePose3DPlayback);
pose3dPipButton?.addEventListener("click", () => {
  togglePose3DPictureInPicture().catch((error) => {
    log(error instanceof Error ? error.message : "3D Keypoints Picture-in-Picture failed");
  });
});
pose3dSeek?.addEventListener("input", () => {
  pose3dFrame = Number(pose3dSeek.value || "0");
  stopPose3D();
  updatePose3DControls();
  drawPose3D();
  drawSelectedKinematicsChart();
  syncVideosToPose3DTime();
});
pose3dCanvas?.addEventListener("mousedown", (event) => {
  pose3dDrag = {
    mode: event.button === 2 ? "pan" : "rotate",
    startX: event.clientX,
    startY: event.clientY,
    camera: { ...pose3dCamera },
  };
});
pose3dCanvas?.addEventListener("wheel", (event) => {
  event.preventDefault();
  pose3dCamera.zoom = Math.max(0.1, Math.min(10, pose3dCamera.zoom * (event.deltaY < 0 ? 1.1 : 0.9)));
  drawPose3D();
}, { passive: false });
pose3dCanvas?.addEventListener("contextmenu", (event) => event.preventDefault());
window.addEventListener("mousemove", (event) => {
  if (!pose3dDrag) return;
  const dx = event.clientX - pose3dDrag.startX;
  const dy = event.clientY - pose3dDrag.startY;
  if (pose3dDrag.mode === "pan") {
    pose3dCamera.panX = pose3dDrag.camera.panX + dx;
    pose3dCamera.panY = pose3dDrag.camera.panY + dy;
  } else {
    pose3dCamera.rotY = pose3dDrag.camera.rotY + dx * 0.01;
    pose3dCamera.rotX = pose3dDrag.camera.rotX + dy * 0.01;
  }
  drawPose3D();
});
window.addEventListener("mouseup", () => {
  pose3dDrag = null;
  keypointDragTarget = null;
  keypointPanDrag = null;
});
window.addEventListener("resize", () => {
  drawPose3D();
  drawKeypointOverlays();
  if (selectedKinematicsSignals.size > 0) {
    drawSelectedKinematicsChart();
  } else {
    clearKinematicsChart();
  }
});
kinematicsTabs?.addEventListener("click", (event) => {
  const kindButton = (event.target as HTMLElement).closest<HTMLButtonElement>("[data-kin-kind]");
  if (kindButton) {
    selectedKinematicsKind = kindButton.dataset.kinKind || "angle";
    kinematicsTabs.querySelectorAll<HTMLButtonElement>("[data-kin-kind]").forEach((tab) => {
      tab.classList.toggle("is-active", tab === kindButton);
    });
    selectedKinematicsSignals.clear();
    kinematicsTimeseriesCache.clear();
    renderKinematicsCards();
    clearKinematicsChart();
    return;
  }
  const button = (event.target as HTMLElement).closest<HTMLButtonElement>("[data-kin-category]");
  if (!button) return;
  selectedKinematicsCategory = button.dataset.kinCategory || "pelvis";
  kinematicsTabs.querySelectorAll<HTMLButtonElement>("[data-kin-category]").forEach((tab) => {
    tab.classList.toggle("is-active", tab === button);
  });
  renderKinematicsCards();
  drawSelectedKinematicsChart();
});
clearKinematicsSelectionButton?.addEventListener("click", () => {
  selectedKinematicsSignals.clear();
  renderKinematicsCards();
  clearKinematicsChart();
});
kinematicsGrid?.addEventListener("click", (event) => {
  const button = (event.target as HTMLElement).closest<HTMLButtonElement>("[data-kin-signal]");
  const signal = button?.dataset.kinSignal;
  if (!signal) return;
  toggleKinematicsTimeseries(signal).catch((error) => log(error instanceof Error ? error.message : "Kinematics chart failed"));
});
kinematicsChart?.addEventListener("mousemove", (event) => {
  const rect = kinematicsChart.getBoundingClientRect();
  kinematicsHoverX = event.clientX - rect.left;
  if (kinematicsChartScrubbing) {
    syncPose3DFromKinematicsChart(event.clientX);
  }
  drawSelectedKinematicsChart();
});
kinematicsChart?.addEventListener("mouseleave", () => {
  kinematicsHoverX = null;
  kinematicsChartScrubbing = false;
  drawSelectedKinematicsChart();
});
kinematicsChart?.addEventListener("mousedown", (event) => {
  kinematicsChartScrubbing = true;
  syncPose3DFromKinematicsChart(event.clientX);
});
window.addEventListener("mouseup", () => {
  kinematicsChartScrubbing = false;
});
window.addEventListener("keydown", (event) => {
  if (!keypointEditEnabled || event.altKey || event.ctrlKey || event.metaKey || event.shiftKey) return;
  if (event.key === "ArrowLeft") {
    event.preventDefault();
    moveKeypointFrame(-1);
  } else if (event.key === "ArrowRight") {
    event.preventDefault();
    moveKeypointFrame(1);
  }
});
overlayToggle?.addEventListener("change", renderVideos);
bindKeypointCanvasControls();
keypointEditVideo?.addEventListener("loadedmetadata", () => {
  updateKeypointFrameControls();
  drawKeypointOverlays();
  loadCurrentKeypointFrame().catch(() => undefined);
});
keypointEditVideo?.addEventListener("timeupdate", () => {
  updateKeypointFrameControls();
  loadCurrentKeypointFrame().catch(() => undefined);
});
keypointFrameSeek?.addEventListener("input", () => {
  const info = activeVideoInfo();
  if (!keypointEditVideo || !info) return;
  cacheCurrentKeypointFrame();
  const fps = Math.max(1, Number(info.fps || 30));
  keypointFrameData = null;
  keypointFrameRequestKey = "";
  keypointUndoStack = [];
  keypointFrameDirty = false;
  keypointEditVideo.currentTime = Number(keypointFrameSeek.value || "0") / fps;
  updateKeypointFrameControls();
  loadCurrentKeypointFrame().catch((error) => log(error instanceof Error ? error.message : "Keypoint load failed"));
});
keypointEditToggle?.addEventListener("click", openKeypointEditor);
keypointCloseButton?.addEventListener("click", () => {
  saveAndCloseKeypointEditor().catch((error) => log(error instanceof Error ? error.message : "Keypoint close failed"));
});
keypointSaveButton?.addEventListener("click", () => {
  saveCurrentKeypointFrame().catch((error) => log(error instanceof Error ? error.message : "Keypoint save failed"));
});
keypointNextVideoButton?.addEventListener("click", () => {
  const count = selectedSession()?.videos.length || 0;
  if (count === 0) return;
  cacheCurrentKeypointFrame();
  keypointActiveVideoIndex = (keypointActiveVideoIndex + 1) % count;
  keypointFrameData = null;
  keypointFrameRequestKey = "";
  keypointUndoStack = [];
  keypointFrameDirty = false;
  keypointView = { zoom: 1, panX: 0, panY: 0 };
  applyKeypointVideoTransform();
  const info = activeVideoInfo();
  if (info && keypointEditVideo) {
    keypointEditVideo.src = info.video_url;
    keypointEditVideo.currentTime = Number(videoSeek?.value || "0");
  }
  updateKeypointEditorControls();
  loadCurrentKeypointFrame().catch((error) => log(error instanceof Error ? error.message : "Keypoint load failed"));
});
keypointSwapLowerButton?.addEventListener("click", () => swapKeypointPairs(lowerBodySwapPairs));
keypointSwapUpperButton?.addEventListener("click", () => swapKeypointPairs(upperBodySwapPairs));
keypointUndoButton?.addEventListener("click", () => {
  const previous = keypointUndoStack.pop();
  if (!previous || !keypointFrameData) return;
  keypointFrameData.people = previous;
  markKeypointFrameDirty();
  updateKeypointEditorControls();
  drawKeypointOverlays();
});
togglePlayButton?.addEventListener("click", toggleAllVideos);
videoSeek?.addEventListener("input", () => {
  seeking = true;
  seekAllVideos();
});
videoSeek?.addEventListener("change", () => {
  seekAllVideos();
  seeking = false;
});
videoSpeed?.addEventListener("change", setPlaybackSpeed);
runButton?.addEventListener("click", runAnalysis);

const initialSessionId = page?.dataset.initialSessionId || "";
loadConfig().catch((error) => log(error instanceof Error ? error.message : "Config load failed"));
loadSessions(initialSessionId)
  .then(() => loadAnalysisResults())
  .catch((error) => log(error instanceof Error ? error.message : "Session load failed"));

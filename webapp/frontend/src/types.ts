export interface CameraStatus {
  camera_id: string;
  label: string;
  connected: boolean;
  recording: boolean;
  live_view_url: string | null;
  live_view_frame_rate: string;
  last_error: string | null;
}

export interface CaptureSession {
  session_id: string;
  subject: {
    name: string;
    height_cm: number;
    weight_kg: number;
    hand: "right" | "left";
  };
  timestamp: string;
  display_timestamp: string;
  session_path: string;
  status: string;
  videos: Array<{
    camera_id: string;
    camera_label: string;
    path: string;
    filename: string;
    size_bytes: number;
    size_label: string;
  }>;
}

export interface CaptureStatusPayload {
  status: string;
  session: CaptureSession;
}

export interface CameraSettings {
  camera_count: number;
  ccb_url: string;
  live_view_frame_rate: string;
  capture_mode: "sony" | "phone";
  phone_camera_count: number;
  phone_frame_rate: number;
  phone_resolution: "720" | "1080";
  backend: string;
}

export interface PhoneSlot {
  camera_id: string;
  camera_label: string;
  join_url: string;
  qr_data_url: string;
}

export interface PhoneDraft {
  token: string;
  slots: PhoneSlot[];
}

"""VideoPose3D 2D-to-3D lifting for keypoint-based extrinsic calibration.

The linear calibration needs, per camera, the *direction* of every bone in that camera
frame. RTMPose only gives 2D, so a temporal lifter turns each camera view into a
root-relative 3D pose; the scale it returns is arbitrary, which is fine because only
directions are consumed downstream.

This module re-implements the dilated temporal convolution stack of VideoPose3D
(Pavllo et al., CVPR 2019) so it can load the released
``pretrained_h36m_detectron_coco.bin`` checkpoint. That checkpoint is published by
Meta under CC BY-NC 4.0 and is **not** redistributed with this project: download it
yourself and place it at ``pipelines/models/videopose3d/`` before running keypoint
calibration.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .skeleton import COCO17_FROM_HALPE26, scatter_h36m17_to_halpe26


MODEL_DIR = Path(__file__).resolve().parents[2] / "models" / "videopose3d"
CHECKPOINT_NAME = "pretrained_h36m_detectron_coco.bin"
CHECKPOINT_URL = "https://dl.fbaipublicfiles.com/video-pose-3d/pretrained_h36m_detectron_coco.bin"

# Architecture of the released checkpoint: 5 blocks of width 3 -> 243-frame receptive field.
FILTER_WIDTHS = (3, 3, 3, 3, 3)
CHANNELS = 1024
NUM_JOINTS_IN = 17
NUM_JOINTS_OUT = 17


class MissingLifterCheckpoint(RuntimeError):
    """Raised when the VideoPose3D weights are not available locally."""


def checkpoint_path(explicit: str | None = None) -> Path:
    return Path(explicit) if explicit else MODEL_DIR / CHECKPOINT_NAME


def _require_torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on the installed environment
        raise RuntimeError(
            "keypoint calibration needs PyTorch for 3D lifting. Install it into the "
            "human-3d-motion environment, for example: "
            "python -m pip install torch --index-url https://download.pytorch.org/whl/cpu"
        ) from exc
    return torch


def _build_model(torch):
    nn = torch.nn

    class TemporalModel(nn.Module):
        """Dilated temporal convolutions over a window of 2D poses."""

        def __init__(self, num_joints_in, in_features, num_joints_out, filter_widths, channels):
            super().__init__()
            self.num_joints_out = num_joints_out
            self.filter_widths = filter_widths
            self.pad = [filter_widths[0] // 2]

            self.expand_conv = nn.Conv1d(
                num_joints_in * in_features, channels, filter_widths[0], bias=False
            )
            self.expand_bn = nn.BatchNorm1d(channels, momentum=0.1)

            layers_conv = []
            layers_bn = []
            next_dilation = filter_widths[0]
            for width in filter_widths[1:]:
                self.pad.append((width - 1) * next_dilation // 2)
                layers_conv.append(
                    nn.Conv1d(channels, channels, width, dilation=next_dilation, bias=False)
                )
                layers_bn.append(nn.BatchNorm1d(channels, momentum=0.1))
                layers_conv.append(nn.Conv1d(channels, channels, 1, dilation=1, bias=False))
                layers_bn.append(nn.BatchNorm1d(channels, momentum=0.1))
                next_dilation *= width

            self.layers_conv = nn.ModuleList(layers_conv)
            self.layers_bn = nn.ModuleList(layers_bn)
            self.shrink = nn.Conv1d(channels, num_joints_out * 3, 1)
            self.relu = nn.ReLU(inplace=True)

        def receptive_field(self) -> int:
            field = 1
            for width in self.filter_widths:
                field *= width
            return field

        def forward(self, x):
            # x: (B, T, J, 2)
            batch, frames = x.shape[0], x.shape[1]
            x = x.view(batch, frames, -1).permute(0, 2, 1)

            x = self.relu(self.expand_bn(self.expand_conv(x)))
            for index in range(len(self.pad) - 1):
                pad = self.pad[index + 1]
                residual = x[:, :, pad:x.shape[2] - pad]
                x = self.relu(self.layers_bn[2 * index](self.layers_conv[2 * index](x)))
                x = residual + self.relu(
                    self.layers_bn[2 * index + 1](self.layers_conv[2 * index + 1](x))
                )
            x = self.shrink(x)
            return x.permute(0, 2, 1).view(batch, -1, self.num_joints_out, 3)

    return TemporalModel


def load_lifter(checkpoint: str | None = None, device: str = "cpu"):
    """Load the VideoPose3D model. Raises ``MissingLifterCheckpoint`` when absent."""
    torch = _require_torch()
    path = checkpoint_path(checkpoint)
    if not path.is_file():
        raise MissingLifterCheckpoint(
            f"videopose3d_checkpoint_not_found: {path}. Download {CHECKPOINT_NAME} from "
            f"{CHECKPOINT_URL} (Meta, CC BY-NC 4.0) and place it there."
        )

    model = _build_model(torch)(NUM_JOINTS_IN, 2, NUM_JOINTS_OUT, FILTER_WIDTHS, CHANNELS)
    payload = torch.load(str(path), map_location="cpu", weights_only=False)
    state = payload.get("model_pos", payload) if isinstance(payload, dict) else payload
    state = {key.replace("module.", "", 1): value for key, value in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        raise RuntimeError(f"videopose3d_checkpoint_incompatible: missing keys {sorted(missing)[:5]}")
    model.eval()
    model.to(device)
    return model, unexpected


def normalize_screen_coordinates(p2d, width: int, height: int) -> np.ndarray:
    """Map pixels to the [-1, 1] range VideoPose3D was trained on (aspect preserving)."""
    return np.asarray(p2d, dtype=np.float32) / width * 2.0 - np.array([1.0, height / width], dtype=np.float32)


def lift_sequence(model, torch, p2d_NxJx2, width: int, height: int, device: str = "cpu",
                  batch_frames: int = 256) -> np.ndarray:
    """Lift one camera sequence. ``p2d`` is Halpe-26 pixels; returns ``(N, 17, 3)``."""
    coco = np.asarray(p2d_NxJx2, dtype=np.float32)[:, COCO17_FROM_HALPE26, :]
    normalized = normalize_screen_coordinates(coco, width, height)

    receptive_field = model.receptive_field()
    pad = receptive_field // 2
    padded = np.pad(normalized, ((pad, pad), (0, 0), (0, 0)), mode="edge")

    outputs = []
    with torch.no_grad():
        # Each output frame needs `receptive_field` inputs, so consecutive chunks overlap.
        start = 0
        total = normalized.shape[0]
        while start < total:
            stop = min(start + batch_frames, total)
            window = padded[start:stop + 2 * pad]
            tensor = torch.from_numpy(window[None]).float().to(device)
            outputs.append(model(tensor).cpu().numpy()[0])
            start = stop

    lifted = np.concatenate(outputs, axis=0)
    if lifted.shape[0] != normalized.shape[0]:
        raise RuntimeError(f"lifter_length_mismatch: {lifted.shape[0]} vs {normalized.shape[0]}")
    return lifted


def lift_all_cameras(p2d_CxNxJx2, s2d_CxNxJ, image_sizes, *, checkpoint: str | None = None,
                     device: str = "cpu", progress=None):
    """Lift every camera sequence to root-relative 3D in Halpe-26 slots.

    Returns ``(p3d (C, N, J, 3), s3d (C, N, J))``. ``s3d`` is the 2D confidence masked to
    the joints the lifter actually produces.
    """
    torch = _require_torch()
    model, _ = load_lifter(checkpoint, device)

    p2d = np.asarray(p2d_CxNxJx2, dtype=np.float64)
    s2d = np.asarray(s2d_CxNxJ, dtype=np.float64)
    C, N, J, _ = p2d.shape

    p3d = np.zeros((C, N, J, 3), dtype=np.float64)
    s3d = np.zeros((C, N, J), dtype=np.float64)
    for camera in range(C):
        width, height = image_sizes[camera]
        if progress:
            progress(f"lifting camera {camera + 1}/{C} to 3D ({N} frames)")
        lifted = lift_sequence(model, torch, p2d[camera], int(width), int(height), device)
        expanded, filled = scatter_h36m17_to_halpe26(lifted.astype(np.float64))
        expanded[~filled] = 0.0
        p3d[camera] = expanded
        s3d[camera] = s2d[camera] * filled
    return p3d, s3d

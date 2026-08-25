"""Skeleton definitions for keypoint-based extrinsic calibration.

The pose solver bundled with this project (``pipelines/models/*/rtmpose_end2end.onnx``)
emits the Halpe-26 layout, whose first 17 joints are exactly the COCO-17 order that
VideoPose3D's ``pretrained_h36m_detectron_coco`` checkpoint expects. The lifted output
comes back in the Human3.6M-17 layout and is scattered back into Halpe-26 slots so the
2D and 3D arrays share one joint indexing throughout the pipeline.

Bone topology mirrors ``OP_BONE`` of flodelaplace/lab-camera-dynamic-calibrator (MIT):
only limb/torso segments whose length is genuinely constant are kept, because the linear
calibration derives camera rotations from bone *orientations* and the bundle adjustment
regularizes bone *length variance*.
"""

from __future__ import annotations

import numpy as np


HALPE26_KEY = {
    "Nose": 0,
    "LEye": 1,
    "REye": 2,
    "LEar": 3,
    "REar": 4,
    "LShoulder": 5,
    "RShoulder": 6,
    "LElbow": 7,
    "RElbow": 8,
    "LWrist": 9,
    "RWrist": 10,
    "LHip": 11,
    "RHip": 12,
    "LKnee": 13,
    "RKnee": 14,
    "LAnkle": 15,
    "RAnkle": 16,
    "Head": 17,
    "Neck": 18,
    "MidHip": 19,
    "LBigToe": 20,
    "RBigToe": 21,
    "LSmallToe": 22,
    "RSmallToe": 23,
    "LHeel": 24,
    "RHeel": 25,
}

NUM_JOINTS = 26

_K = HALPE26_KEY

BONES = np.array(
    [
        [_K["MidHip"], _K["Neck"]],
        [_K["RShoulder"], _K["LShoulder"]],
        [_K["RShoulder"], _K["RElbow"]],
        [_K["LShoulder"], _K["LElbow"]],
        [_K["RWrist"], _K["RElbow"]],
        [_K["LWrist"], _K["LElbow"]],
        [_K["RHip"], _K["MidHip"]],
        [_K["LHip"], _K["MidHip"]],
        [_K["RHip"], _K["RKnee"]],
        [_K["RKnee"], _K["RAnkle"]],
        [_K["LHip"], _K["LKnee"]],
        [_K["LKnee"], _K["LAnkle"]],
    ],
    dtype=int,
)

BONE_JOINTS = np.sort(np.unique(BONES.flatten()))

# Halpe-26 slots 0..16 already follow the COCO-17 order VideoPose3D was trained on.
COCO17_FROM_HALPE26 = np.arange(17, dtype=int)

# Human3.6M-17 joint index -> Halpe-26 slot it is written back into.
H36M17_TO_HALPE26 = {
    0: _K["MidHip"],
    1: _K["RHip"],
    2: _K["RKnee"],
    3: _K["RAnkle"],
    4: _K["LHip"],
    5: _K["LKnee"],
    6: _K["LAnkle"],
    8: _K["Neck"],
    9: _K["Nose"],
    10: _K["Head"],
    11: _K["LShoulder"],
    12: _K["LElbow"],
    13: _K["LWrist"],
    14: _K["RShoulder"],
    15: _K["RElbow"],
    16: _K["RWrist"],
}

# Joints used to put the scene on the floor / measure stature during metric scaling.
FOOT_JOINTS = [_K["LHeel"], _K["RHeel"], _K["LBigToe"], _K["RBigToe"], _K["LAnkle"], _K["RAnkle"]]
HEAD_JOINTS = [_K["Head"], _K["Nose"]]

# Toes stick out in front of the body, which tilts a head-minus-foot vector away from
# vertical; heels and ankles sit under it, so only those define the gravity axis.
UP_AXIS_FOOT_JOINTS = [_K["LHeel"], _K["RHeel"], _K["LAnkle"], _K["RAnkle"]]


def scatter_h36m17_to_halpe26(p3d_h36m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Expand ``(..., 17, 3)`` Human3.6M poses into ``(..., 26, 3)`` Halpe-26 slots.

    Returns the expanded array plus a boolean ``(..., 26)`` mask marking which slots
    actually received a lifted joint.
    """
    if p3d_h36m.shape[-2:] != (17, 3):
        raise ValueError(f"expected (..., 17, 3) poses, got {p3d_h36m.shape}")

    out = np.full(p3d_h36m.shape[:-2] + (NUM_JOINTS, 3), np.nan, dtype=np.float64)
    filled = np.zeros(p3d_h36m.shape[:-2] + (NUM_JOINTS,), dtype=bool)
    for h36m_index, halpe_index in H36M17_TO_HALPE26.items():
        out[..., halpe_index, :] = p3d_h36m[..., h36m_index, :]
        filled[..., halpe_index] = True
    return out, filled

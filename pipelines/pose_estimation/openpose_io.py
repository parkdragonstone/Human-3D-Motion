import json
import os

import numpy as np


def save_to_openpose(json_file_path, keypoints, scores):
    """Save keypoints/scores to OpenPose-like JSON."""
    detections = []
    for detection_index in range(len(keypoints)):
        keypoints_with_confidence = []
        for keypoint, score in zip(keypoints[detection_index], scores[detection_index]):
            x_coord = float(keypoint[0]) if not np.isnan(keypoint[0]) else float("nan")
            y_coord = float(keypoint[1]) if not np.isnan(keypoint[1]) else float("nan")
            confidence = float(score) if not np.isnan(score) else float("nan")
            keypoints_with_confidence.extend([x_coord, y_coord, confidence])

        detections.append({
            "person_id": [-1],
            "pose_keypoints_2d": keypoints_with_confidence,
            "face_keypoints_2d": [],
            "hand_left_keypoints_2d": [],
            "hand_right_keypoints_2d": [],
            "pose_keypoints_3d": [],
            "face_keypoints_3d": [],
            "hand_left_keypoints_3d": [],
            "hand_right_keypoints_3d": [],
        })

    json_output = {"version": 1.3, "people": detections}
    json_output_dir = os.path.abspath(os.path.join(json_file_path, ".."))
    if not os.path.isdir(json_output_dir):
        os.makedirs(json_output_dir, exist_ok=True)
    with open(json_file_path, "w", encoding="utf-8") as json_file:
        json.dump(json_output, json_file, ensure_ascii=False)

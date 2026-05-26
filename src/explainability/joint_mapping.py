"""Mapping from the 15 hand-crafted features to the MediaPipe joints
they consume, used to render feature-level attributions as a per-joint
skeleton heatmap.

Each feature in :mod:`src.features.extractor` is computed from a
specific set of keypoints. We project a feature-level attribution
vector back onto the 33 MediaPipe landmarks by distributing each
feature's contribution evenly across the joints it touches. The
resulting per-joint score answers: *"how much did this joint's
position drive the current fall prediction?"*
"""

from typing import Dict, List

import numpy as np

from src.pose.keypoints import MediaPipeKeypoint as MP


FEATURE_NAMES: List[str] = [
    "torso_inclination",        # 0
    "hip_shoulder_angle",       # 1
    "bbox_aspect_ratio",        # 2
    "com_velocity",             # 3
    "com_acceleration",         # 4
    "head_to_toe_distance",     # 5
    "shoulder_ankle_distance",  # 6
    "left_knee_angle",          # 7
    "right_knee_angle",         # 8
    "left_hip_angle",           # 9
    "right_hip_angle",          # 10
    "wrist_hip_distance",       # 11
    "body_spread",              # 12
    "mean_visibility",          # 13
    "min_core_visibility",      # 14
]


# Core joints used for the "global" features (bbox, body spread, visibility).
# These features depend on all visible landmarks; we distribute their
# attribution across the structural-torso joints to keep the visualization
# focused on the body rather than face/hand landmarks.
_CORE_BODY: List[int] = [
    int(MP.LEFT_SHOULDER), int(MP.RIGHT_SHOULDER),
    int(MP.LEFT_HIP),      int(MP.RIGHT_HIP),
    int(MP.LEFT_KNEE),     int(MP.RIGHT_KNEE),
    int(MP.LEFT_ANKLE),    int(MP.RIGHT_ANKLE),
]


# Per-feature joint sets. Index matches FEATURE_NAMES.
FEATURE_JOINTS: Dict[int, List[int]] = {
    0: [int(MP.LEFT_SHOULDER), int(MP.RIGHT_SHOULDER),
        int(MP.LEFT_HIP), int(MP.RIGHT_HIP)],            # torso_inclination
    1: [int(MP.LEFT_SHOULDER), int(MP.RIGHT_SHOULDER),
        int(MP.LEFT_HIP), int(MP.RIGHT_HIP)],            # hip_shoulder_angle
    2: list(_CORE_BODY),                                  # bbox_aspect_ratio
    3: [int(MP.LEFT_HIP), int(MP.RIGHT_HIP)],            # com_velocity
    4: [int(MP.LEFT_HIP), int(MP.RIGHT_HIP)],            # com_acceleration
    5: [int(MP.NOSE), int(MP.LEFT_ANKLE), int(MP.RIGHT_ANKLE)],   # head_to_toe
    6: [int(MP.LEFT_SHOULDER), int(MP.RIGHT_SHOULDER),
        int(MP.LEFT_ANKLE), int(MP.RIGHT_ANKLE)],        # shoulder_ankle
    7: [int(MP.LEFT_HIP), int(MP.LEFT_KNEE), int(MP.LEFT_ANKLE)],     # L knee
    8: [int(MP.RIGHT_HIP), int(MP.RIGHT_KNEE), int(MP.RIGHT_ANKLE)],  # R knee
    9: [int(MP.LEFT_SHOULDER), int(MP.LEFT_HIP), int(MP.LEFT_KNEE)],     # L hip
    10: [int(MP.RIGHT_SHOULDER), int(MP.RIGHT_HIP), int(MP.RIGHT_KNEE)], # R hip
    11: [int(MP.LEFT_WRIST), int(MP.RIGHT_WRIST),
         int(MP.LEFT_HIP), int(MP.RIGHT_HIP)],           # wrist_hip
    12: list(_CORE_BODY),                                 # body_spread
    13: list(_CORE_BODY),                                 # mean_visibility
    14: list(_CORE_BODY),                                 # min_core_visibility
}


NUM_MEDIAPIPE_JOINTS: int = 33


def joints_from_feature_attribution(
    feature_attribution: np.ndarray,
    num_joints: int = NUM_MEDIAPIPE_JOINTS,
) -> np.ndarray:
    """Project a feature-level attribution vector onto joint indices.

    For each feature, its signed attribution is distributed *evenly*
    across the joints that feature consumes. Joint scores are then
    summed across features. The result is a signed per-joint vector
    where positive values indicate the joint pushed the prediction
    toward "fall" and negative values pushed it toward "non-fall".

    Parameters
    ----------
    feature_attribution : np.ndarray
        Shape ``(num_features,)`` — signed contribution per feature.
    num_joints : int
        Number of landmarks in the keypoint format. Defaults to 33
        (MediaPipe). For COCO (17), the head, hand and foot landmarks
        beyond index 16 are simply unused.

    Returns
    -------
    np.ndarray
        Shape ``(num_joints,)`` — signed per-joint score.
    """
    scores = np.zeros(num_joints, dtype=np.float32)
    for feature_idx, attribution in enumerate(feature_attribution):
        joints = FEATURE_JOINTS.get(feature_idx, [])
        if not joints:
            continue
        share = float(attribution) / float(len(joints))
        for j in joints:
            if 0 <= j < num_joints:
                scores[j] += share
    return scores

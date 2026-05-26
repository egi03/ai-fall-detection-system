"""
Keypoint constants, index mappings, and utility functions.

Provides explicit named constants for all MediaPipe and COCO keypoint
indices. Never use magic numbers for keypoint access.

Reference: research/8.3 - Anatomical landmark index mappings table.
"""

from enum import IntEnum


class MediaPipeKeypoint(IntEnum):
    """MediaPipe BlazePose 33-landmark indices."""

    NOSE = 0
    LEFT_EYE_INNER = 1
    LEFT_EYE = 2
    LEFT_EYE_OUTER = 3
    RIGHT_EYE_INNER = 4
    RIGHT_EYE = 5
    RIGHT_EYE_OUTER = 6
    LEFT_EAR = 7
    RIGHT_EAR = 8
    MOUTH_LEFT = 9
    MOUTH_RIGHT = 10
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_PINKY = 17
    RIGHT_PINKY = 18
    LEFT_INDEX = 19
    RIGHT_INDEX = 20
    LEFT_THUMB = 21
    RIGHT_THUMB = 22
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28
    LEFT_HEEL = 29
    RIGHT_HEEL = 30
    LEFT_FOOT_INDEX = 31
    RIGHT_FOOT_INDEX = 32


class COCOKeypoint(IntEnum):
    """COCO 17-keypoint indices (used by YOLO-Pose)."""

    NOSE = 0
    LEFT_EYE = 1
    RIGHT_EYE = 2
    LEFT_EAR = 3
    RIGHT_EAR = 4
    LEFT_SHOULDER = 5
    RIGHT_SHOULDER = 6
    LEFT_ELBOW = 7
    RIGHT_ELBOW = 8
    LEFT_WRIST = 9
    RIGHT_WRIST = 10
    LEFT_HIP = 11
    RIGHT_HIP = 12
    LEFT_KNEE = 13
    RIGHT_KNEE = 14
    LEFT_ANKLE = 15
    RIGHT_ANKLE = 16


# Skeleton connection pairs for visualization
MEDIAPIPE_SKELETON_CONNECTIONS: list = [
    (MediaPipeKeypoint.LEFT_SHOULDER, MediaPipeKeypoint.RIGHT_SHOULDER),
    (MediaPipeKeypoint.LEFT_SHOULDER, MediaPipeKeypoint.LEFT_ELBOW),
    (MediaPipeKeypoint.LEFT_ELBOW, MediaPipeKeypoint.LEFT_WRIST),
    (MediaPipeKeypoint.RIGHT_SHOULDER, MediaPipeKeypoint.RIGHT_ELBOW),
    (MediaPipeKeypoint.RIGHT_ELBOW, MediaPipeKeypoint.RIGHT_WRIST),
    (MediaPipeKeypoint.LEFT_SHOULDER, MediaPipeKeypoint.LEFT_HIP),
    (MediaPipeKeypoint.RIGHT_SHOULDER, MediaPipeKeypoint.RIGHT_HIP),
    (MediaPipeKeypoint.LEFT_HIP, MediaPipeKeypoint.RIGHT_HIP),
    (MediaPipeKeypoint.LEFT_HIP, MediaPipeKeypoint.LEFT_KNEE),
    (MediaPipeKeypoint.LEFT_KNEE, MediaPipeKeypoint.LEFT_ANKLE),
    (MediaPipeKeypoint.RIGHT_HIP, MediaPipeKeypoint.RIGHT_KNEE),
    (MediaPipeKeypoint.RIGHT_KNEE, MediaPipeKeypoint.RIGHT_ANKLE),
]

COCO_SKELETON_CONNECTIONS: list = [
    (COCOKeypoint.LEFT_SHOULDER, COCOKeypoint.RIGHT_SHOULDER),
    (COCOKeypoint.LEFT_SHOULDER, COCOKeypoint.LEFT_ELBOW),
    (COCOKeypoint.LEFT_ELBOW, COCOKeypoint.LEFT_WRIST),
    (COCOKeypoint.RIGHT_SHOULDER, COCOKeypoint.RIGHT_ELBOW),
    (COCOKeypoint.RIGHT_ELBOW, COCOKeypoint.RIGHT_WRIST),
    (COCOKeypoint.LEFT_SHOULDER, COCOKeypoint.LEFT_HIP),
    (COCOKeypoint.RIGHT_SHOULDER, COCOKeypoint.RIGHT_HIP),
    (COCOKeypoint.LEFT_HIP, COCOKeypoint.RIGHT_HIP),
    (COCOKeypoint.LEFT_HIP, COCOKeypoint.LEFT_KNEE),
    (COCOKeypoint.LEFT_KNEE, COCOKeypoint.LEFT_ANKLE),
    (COCOKeypoint.RIGHT_HIP, COCOKeypoint.RIGHT_KNEE),
    (COCOKeypoint.RIGHT_KNEE, COCOKeypoint.RIGHT_ANKLE),
]


def get_index(joint_name: str, keypoint_format: str = "mediapipe") -> int:
    """
    Get the numeric index for a named joint.

    Parameters
    ----------
    joint_name : str
        Anatomical joint name, e.g. 'LEFT_SHOULDER'.
    keypoint_format : str
        Either 'mediapipe' (33 keypoints) or 'coco' (17 keypoints).

    Returns
    -------
    int
        Keypoint index.

    Raises
    ------
    ValueError
        If joint_name or keypoint_format is not recognized.
    """
    if keypoint_format == "mediapipe":
        return MediaPipeKeypoint[joint_name].value
    elif keypoint_format == "coco":
        return COCOKeypoint[joint_name].value
    else:
        raise ValueError(f"Unknown keypoint format: {keypoint_format}")

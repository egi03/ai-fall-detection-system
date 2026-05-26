"""
Skeleton overlay drawing on video frames.

Draws keypoint landmarks and bone connections directly onto
NumPy image arrays using OpenCV primitives.

Reference: research/8.1 - OpenCV drawing functions (cv2.line,
cv2.circle) for optimized C++ array manipulation.
Reference: research/9.1 - Color coding for alarm states.
"""

from typing import Optional, Tuple

import numpy as np

from src.pose.keypoints import (
    MEDIAPIPE_SKELETON_CONNECTIONS,
    COCO_SKELETON_CONNECTIONS,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Default colors (BGR)
COLOR_NORMAL = (0, 255, 0)       # Green — normal state
COLOR_IMMINENT = (0, 255, 255)   # Yellow — pre-fall warning (rising probability)
COLOR_WARNING = (0, 165, 255)    # Orange — impact detected
COLOR_ALARM = (0, 0, 255)        # Red — fall confirmed
COLOR_BBOX = (255, 255, 0)       # Cyan — bounding box
COLOR_TEXT = (255, 255, 255)     # White — text

# Attribution heatmap endpoints (BGR)
_ATTRIB_POSITIVE = (40, 40, 255)   # red — pushes toward fall
_ATTRIB_NEGATIVE = (255, 140, 40)  # blue — pushes away from fall


def _blend_attribution_color(
    base: Tuple[int, int, int],
    signed_intensity: float,
) -> Tuple[int, int, int]:
    """Blend a base color toward red (positive) or blue (negative).

    Parameters
    ----------
    base : tuple
        Neutral BGR color used at zero attribution.
    signed_intensity : float
        Value in [-1, 1]. Positive => blend toward red; negative => blue.

    Returns
    -------
    tuple
        BGR color suitable for cv2 drawing primitives.
    """
    s = max(-1.0, min(1.0, float(signed_intensity)))
    target = _ATTRIB_POSITIVE if s >= 0 else _ATTRIB_NEGATIVE
    alpha = abs(s)
    return tuple(
        int((1.0 - alpha) * b + alpha * t) for b, t in zip(base, target)
    )


class SkeletonDrawer:
    """
    Draws skeleton overlays on video frames.

    Supports both MediaPipe (33 keypoints) and COCO (17 keypoints)
    connection topologies. Keypoints below the confidence threshold
    are not drawn.

    Parameters
    ----------
    keypoint_format : str
        Either 'mediapipe' or 'coco' for connection topology.
    keypoint_radius : int
        Radius of keypoint circles in pixels.
    line_thickness : int
        Thickness of bone connection lines in pixels.
    confidence_threshold : float
        Only draw keypoints above this visibility score.
    """

    def __init__(
        self,
        keypoint_format: str = "mediapipe",
        keypoint_radius: int = 4,
        line_thickness: int = 2,
        confidence_threshold: float = 0.5,
    ) -> None:
        if keypoint_format == "mediapipe":
            self._connections = MEDIAPIPE_SKELETON_CONNECTIONS
            num_kps = 33
        elif keypoint_format == "coco":
            self._connections = COCO_SKELETON_CONNECTIONS
            num_kps = 17
        else:
            raise ValueError(
                f"Unknown keypoint format: {keypoint_format}. "
                "Must be 'mediapipe' or 'coco'."
            )

        self._format = keypoint_format
        self._radius = keypoint_radius
        self._thickness = line_thickness
        self._conf_threshold = confidence_threshold
        # Hysteresis: lower threshold to turn off (prevents flicker on
        # keypoints oscillating near the threshold)
        self._conf_threshold_low = max(0.0, confidence_threshold - 0.15)
        # Per-keypoint visibility state for hysteresis
        self._kp_visible = np.zeros(num_kps, dtype=bool)

    def draw(
        self,
        frame: np.ndarray,
        keypoints: np.ndarray,
        color: Tuple[int, int, int] = COLOR_NORMAL,
    ) -> np.ndarray:
        """
        Draw skeleton overlay on a frame.

        Parameters
        ----------
        frame : np.ndarray
            BGR image of shape (H, W, 3).
        keypoints : np.ndarray
            Keypoints of shape (N, 4) with (x, y, z, visibility).
            Coordinates must be in pixel space.
        color : tuple of int
            BGR color for the skeleton.

        Returns
        -------
        np.ndarray
            Frame with skeleton overlay drawn.
        """
        import cv2

        h, w = frame.shape[:2]

        if keypoints is None or len(keypoints) == 0:
            return frame

        # Determine confidence per keypoint
        if keypoints.shape[1] >= 4:
            conf = keypoints[:, 3]
        elif keypoints.shape[1] == 3:
            conf = keypoints[:, 2]
        else:
            conf = np.ones(len(keypoints), dtype=np.float32)

        # Hysteresis: once visible, stays visible until conf drops below
        # the lower threshold. Prevents flickering on borderline keypoints.
        n = min(len(conf), len(self._kp_visible))
        for idx in range(n):
            if self._kp_visible[idx]:
                # Currently visible: only hide if drops below low threshold
                self._kp_visible[idx] = conf[idx] >= self._conf_threshold_low
            else:
                # Currently hidden: only show if rises above high threshold
                self._kp_visible[idx] = conf[idx] >= self._conf_threshold

        valid = self._kp_visible[:n].copy()

        # Also filter out NaN coordinates
        valid = valid & ~np.isnan(keypoints[:n, 0]) & ~np.isnan(keypoints[:n, 1])

        # Draw bone connections first (underneath keypoints)
        for (i, j) in self._connections:
            if i < n and j < n and valid[i] and valid[j]:
                pt1 = (int(keypoints[i, 0]), int(keypoints[i, 1]))
                pt2 = (int(keypoints[j, 0]), int(keypoints[j, 1]))

                if (0 <= pt1[0] < w and 0 <= pt1[1] < h and
                        0 <= pt2[0] < w and 0 <= pt2[1] < h):
                    cv2.line(frame, pt1, pt2, color, self._thickness)

        # Draw keypoint circles
        for idx in range(n):
            if valid[idx]:
                x, y = int(keypoints[idx, 0]), int(keypoints[idx, 1])
                if 0 <= x < w and 0 <= y < h:
                    cv2.circle(frame, (x, y), self._radius, color, -1)

        return frame

    def draw_with_attribution(
        self,
        frame: np.ndarray,
        keypoints: np.ndarray,
        joint_attribution: np.ndarray,
        base_color: Tuple[int, int, int] = COLOR_NORMAL,
        max_radius: int = 16,
    ) -> np.ndarray:
        """Draw the skeleton with per-joint attribution heatmap.

        Each joint's circle is colored along a blue → base → red gradient
        proportional to its signed attribution (negative = pushes away from
        fall; positive = pushes toward fall). Bone connections use the base
        alarm color so the body remains visually coherent.

        Parameters
        ----------
        frame : np.ndarray
            BGR image of shape (H, W, 3).
        keypoints : np.ndarray
            Keypoints of shape (N, 4) in *pixel* coordinates.
        joint_attribution : np.ndarray
            Shape (N,) signed scores. Will be normalized internally by its
            absolute maximum so the visualization stays consistent across
            frames.
        base_color : tuple
            BGR color used for bones and as the neutral mid-point for joints
            with zero attribution.
        max_radius : int
            Maximum joint-circle radius (pixels) for the strongest absolute
            attribution. Other joints scale linearly down to ``self._radius``.

        Returns
        -------
        np.ndarray
            Frame with the attribution-tinted skeleton drawn on it.
        """
        import cv2

        h, w = frame.shape[:2]
        if keypoints is None or len(keypoints) == 0:
            return frame

        if keypoints.shape[1] >= 4:
            conf = keypoints[:, 3]
        elif keypoints.shape[1] == 3:
            conf = keypoints[:, 2]
        else:
            conf = np.ones(len(keypoints), dtype=np.float32)

        n = min(len(conf), len(self._kp_visible))
        for idx in range(n):
            if self._kp_visible[idx]:
                self._kp_visible[idx] = conf[idx] >= self._conf_threshold_low
            else:
                self._kp_visible[idx] = conf[idx] >= self._conf_threshold
        valid = self._kp_visible[:n].copy()
        valid = valid & ~np.isnan(keypoints[:n, 0]) & ~np.isnan(keypoints[:n, 1])

        # Bone connections in the base alarm color
        for (i, j) in self._connections:
            if i < n and j < n and valid[i] and valid[j]:
                pt1 = (int(keypoints[i, 0]), int(keypoints[i, 1]))
                pt2 = (int(keypoints[j, 0]), int(keypoints[j, 1]))
                if (0 <= pt1[0] < w and 0 <= pt1[1] < h
                        and 0 <= pt2[0] < w and 0 <= pt2[1] < h):
                    cv2.line(frame, pt1, pt2, base_color, self._thickness)

        # Normalize attribution by absolute max for consistent intensity
        attribution = np.asarray(joint_attribution, dtype=np.float32)[:n]
        max_abs = float(np.max(np.abs(attribution))) if attribution.size else 0.0
        if max_abs < 1e-8:
            scale = np.zeros_like(attribution)
        else:
            scale = attribution / max_abs  # signed, [-1, 1]

        for idx in range(n):
            if not valid[idx]:
                continue
            x, y = int(keypoints[idx, 0]), int(keypoints[idx, 1])
            if not (0 <= x < w and 0 <= y < h):
                continue

            s = float(scale[idx])
            magnitude = abs(s)
            radius = self._radius + int((max_radius - self._radius) * magnitude)
            joint_color = _blend_attribution_color(base_color, s)

            # Filled disk
            cv2.circle(frame, (x, y), radius, joint_color, -1)
            # Subtle dark outline for separability
            cv2.circle(frame, (x, y), radius, (20, 20, 20), 1)

        return frame

    def draw_bbox(
        self,
        frame: np.ndarray,
        bbox: Tuple[int, int, int, int],
        color: Tuple[int, int, int] = COLOR_BBOX,
        label: Optional[str] = None,
    ) -> np.ndarray:
        """
        Draw a bounding box on the frame.

        Parameters
        ----------
        frame : np.ndarray
            BGR image of shape (H, W, 3).
        bbox : tuple
            (x1, y1, x2, y2) bounding box coordinates.
        color : tuple of int
            BGR color for the bounding box.
        label : str, optional
            Text label to draw above the box.

        Returns
        -------
        np.ndarray
            Frame with bounding box drawn.
        """
        import cv2

        output = frame.copy()
        x1, y1, x2, y2 = bbox

        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)

        if label:
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 1
            (text_w, text_h), baseline = cv2.getTextSize(
                label, font, font_scale, thickness
            )

            # Draw text background
            cv2.rectangle(
                output,
                (x1, y1 - text_h - baseline - 4),
                (x1 + text_w, y1),
                color,
                -1,
            )

            # Draw text
            cv2.putText(
                output,
                label,
                (x1, y1 - baseline - 2),
                font,
                font_scale,
                (0, 0, 0),
                thickness,
            )

        return output


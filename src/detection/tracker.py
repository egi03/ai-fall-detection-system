"""
Person tracking across video frames.

Maintains consistent identity (track ID) for detected persons
across consecutive frames using centroid-based IoU matching.

DECISION: Using simple centroid+IoU tracker rather than full
ByteTrack for the initial implementation. This is sufficient
for the single-person primary use case (elderly monitoring).
ByteTrack's low-confidence retention can be added later if
multi-person support is needed (research/10.1).

Reference: research/3.3 - ByteTrack retains low-confidence detections
during falls, preventing ID loss at the critical moment.
Reference: research/10.1 - Multi-person tracking requirements.
"""

from collections import OrderedDict
from typing import Dict, List, Tuple

import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)


def _compute_iou(
    box_a: Tuple[int, int, int, int],
    box_b: Tuple[int, int, int, int],
) -> float:
    """
    Compute Intersection over Union between two bounding boxes.

    Parameters
    ----------
    box_a, box_b : tuple
        (x1, y1, x2, y2) bounding box coordinates.

    Returns
    -------
    float
        IoU value in [0, 1].
    """
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    if inter_area == 0:
        return 0.0

    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union_area = area_a + area_b - inter_area

    if union_area <= 0:
        return 0.0

    return inter_area / union_area


def _centroid(bbox: Tuple[int, int, int, int]) -> Tuple[float, float]:
    """Compute centroid of a bounding box."""
    return ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)


class PersonTracker:
    """
    Centroid + IoU based person tracker for single or few persons.

    Assigns persistent track IDs to detected bounding boxes across
    frames. Uses a combination of centroid distance and IoU for
    matching detections to existing tracks.

    Parameters
    ----------
    max_disappeared : int
        Maximum consecutive frames a track can be missing before removal.
    max_distance : float
        Maximum centroid distance (pixels) for matching detections.
    min_iou : float
        Minimum IoU for considering a detection as the same track.
    """

    def __init__(
        self,
        max_disappeared: int = 30,
        max_distance: float = 100.0,
        min_iou: float = 0.1,
    ) -> None:
        self._next_id = 0
        self._max_disappeared = max_disappeared
        self._max_distance = max_distance
        self._min_iou = min_iou

        # Active tracks: track_id -> detection dict
        self._tracks: OrderedDict[int, dict] = OrderedDict()
        # Disappearance counter per track
        self._disappeared: OrderedDict[int, int] = OrderedDict()

    @property
    def active_tracks(self) -> Dict[int, dict]:
        """Return currently active tracks."""
        return dict(self._tracks)

    @property
    def num_tracks(self) -> int:
        """Return number of active tracks."""
        return len(self._tracks)

    def update(self, detections: List[dict]) -> Dict[int, dict]:
        """
        Update tracks with new frame detections.

        Matching strategy:
        1. If no existing tracks, register all detections as new tracks
        2. If no detections, increment disappeared counter for all tracks
        3. Otherwise, match detections to tracks using centroid distance
           with IoU tiebreaking. Unmatched detections become new tracks.
           Unmatched tracks increment disappeared counter.

        Parameters
        ----------
        detections : list of dict
            Detection results from PersonDetector.detect().
            Each dict must have a 'bbox' key.

        Returns
        -------
        dict
            Mapping of track_id -> detection dict (with 'track_id' key added).
        """
        # No existing tracks — register all detections
        if len(self._tracks) == 0:
            for det in detections:
                self._register(det)
            return self._get_output()

        # No detections — increment disappeared for all tracks
        if len(detections) == 0:
            self._mark_all_disappeared()
            return self._get_output()

        # Match detections to existing tracks
        track_ids = list(self._tracks.keys())
        track_centroids = [_centroid(self._tracks[tid]["bbox"]) for tid in track_ids]
        det_centroids = [_centroid(d["bbox"]) for d in detections]

        # Compute distance matrix: (num_tracks, num_detections)
        num_tracks = len(track_ids)
        num_dets = len(detections)
        dist_matrix = np.zeros((num_tracks, num_dets), dtype=np.float64)

        for i, tc in enumerate(track_centroids):
            for j, dc in enumerate(det_centroids):
                dist_matrix[i, j] = np.sqrt(
                    (tc[0] - dc[0]) ** 2 + (tc[1] - dc[1]) ** 2
                )

        # Greedy matching: assign closest pairs
        matched_tracks = set()
        matched_dets = set()

        # Sort by distance (ascending) for greedy assignment
        flat_indices = np.argsort(dist_matrix, axis=None)

        for flat_idx in flat_indices:
            i = int(flat_idx // num_dets)
            j = int(flat_idx % num_dets)

            if i in matched_tracks or j in matched_dets:
                continue

            distance = dist_matrix[i, j]
            if distance > self._max_distance:
                continue

            # Optional IoU check
            iou = _compute_iou(
                self._tracks[track_ids[i]]["bbox"],
                detections[j]["bbox"],
            )
            if iou < self._min_iou and distance > self._max_distance * 0.5:
                continue

            # Match found
            tid = track_ids[i]
            det = detections[j].copy()
            det["track_id"] = tid
            self._tracks[tid] = det
            self._disappeared[tid] = 0
            matched_tracks.add(i)
            matched_dets.add(j)

        # Handle unmatched tracks
        for i in range(num_tracks):
            if i not in matched_tracks:
                tid = track_ids[i]
                self._disappeared[tid] += 1
                if self._disappeared[tid] > self._max_disappeared:
                    self._deregister(tid)

        # Handle unmatched detections — register as new tracks
        for j in range(num_dets):
            if j not in matched_dets:
                self._register(detections[j])

        return self._get_output()

    def _register(self, detection: dict) -> int:
        """Register a new track."""
        tid = self._next_id
        det = detection.copy()
        det["track_id"] = tid
        self._tracks[tid] = det
        self._disappeared[tid] = 0
        self._next_id += 1
        return tid

    def _deregister(self, track_id: int) -> None:
        """Remove a track."""
        del self._tracks[track_id]
        del self._disappeared[track_id]

    def _mark_all_disappeared(self) -> None:
        """Increment disappeared counter for all tracks."""
        to_remove = []
        for tid in list(self._disappeared.keys()):
            self._disappeared[tid] += 1
            if self._disappeared[tid] > self._max_disappeared:
                to_remove.append(tid)

        for tid in to_remove:
            self._deregister(tid)

    def _get_output(self) -> Dict[int, dict]:
        """Build output dict with track_id keys."""
        output = {}
        for tid, det in self._tracks.items():
            d = det.copy()
            d["track_id"] = tid
            output[tid] = d
        return output

    def reset(self) -> None:
        """Reset all active tracks."""
        self._tracks.clear()
        self._disappeared.clear()
        self._next_id = 0

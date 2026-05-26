"""
Unit tests for Phase 2: Detection, Pose, Tracker, and Visualization modules.

Tests core logic without requiring actual model weights (YOLOv8, MediaPipe).
Heavy ML backends are mocked; only pure-logic methods are tested directly.
"""

import numpy as np
import pytest

from src.detection.person_detector import PersonDetector
from src.detection.tracker import PersonTracker, _compute_iou, _centroid
from src.visualization.skeleton import SkeletonDrawer, COLOR_NORMAL, COLOR_ALARM
from src.visualization.display import DisplayManager, _STATE_COLORS
from src.pose.keypoints import (
    MediaPipeKeypoint,
    COCOKeypoint,
    get_index,
    MEDIAPIPE_SKELETON_CONNECTIONS,
    COCO_SKELETON_CONNECTIONS,
)


# ─── PersonDetector static methods ──────────────────────────────────

class TestPersonDetectorStatics:
    """Test pure-logic static methods on PersonDetector (no model needed)."""

    def test_aspect_ratio_standing(self) -> None:
        """Standing bbox (taller than wide) should have AR < 1."""
        bbox = (100, 50, 200, 400)  # w=100, h=350
        ar = PersonDetector.compute_aspect_ratio(bbox)
        assert ar < 1.0
        assert abs(ar - 100 / 350) < 1e-5

    def test_aspect_ratio_lying(self) -> None:
        """Lying bbox (wider than tall) should have AR > 1."""
        bbox = (50, 200, 400, 300)  # w=350, h=100
        ar = PersonDetector.compute_aspect_ratio(bbox)
        assert ar > 1.0
        assert abs(ar - 350 / 100) < 1e-5

    def test_aspect_ratio_square(self) -> None:
        """Square bbox should have AR = 1."""
        bbox = (0, 0, 100, 100)
        ar = PersonDetector.compute_aspect_ratio(bbox)
        assert abs(ar - 1.0) < 1e-5

    def test_aspect_ratio_zero_height(self) -> None:
        """Zero-height bbox should return NaN."""
        bbox = (0, 50, 100, 50)
        ar = PersonDetector.compute_aspect_ratio(bbox)
        assert np.isnan(ar)

    def test_centroid(self) -> None:
        """Centroid should be at the center of the bbox."""
        bbox = (100, 200, 300, 400)
        cx, cy = PersonDetector.compute_centroid(bbox)
        assert abs(cx - 200.0) < 1e-5
        assert abs(cy - 300.0) < 1e-5

    def test_area(self) -> None:
        """Area should be width * height."""
        bbox = (10, 20, 110, 220)  # w=100, h=200
        assert PersonDetector.compute_area(bbox) == 20000

    def test_area_degenerate(self) -> None:
        """Degenerate (zero or negative dims) bbox should have area 0."""
        assert PersonDetector.compute_area((100, 100, 50, 50)) == 0


# ─── IoU and centroid helpers ────────────────────────────────────────

class TestIoUHelper:
    """Test the _compute_iou helper function."""

    def test_identical_boxes(self) -> None:
        """Identical boxes should have IoU = 1.0."""
        box = (0, 0, 100, 100)
        assert abs(_compute_iou(box, box) - 1.0) < 1e-5

    def test_no_overlap(self) -> None:
        """Non-overlapping boxes should have IoU = 0.0."""
        assert _compute_iou((0, 0, 50, 50), (100, 100, 200, 200)) == 0.0

    def test_partial_overlap(self) -> None:
        """Partially overlapping boxes should have 0 < IoU < 1."""
        iou = _compute_iou((0, 0, 100, 100), (50, 50, 150, 150))
        assert 0.0 < iou < 1.0
        # intersection = 50*50 = 2500, union = 10000+10000-2500 = 17500
        assert abs(iou - 2500 / 17500) < 1e-5

    def test_contained_box(self) -> None:
        """A box fully inside another should have IoU = small_area/big_area."""
        iou = _compute_iou((0, 0, 100, 100), (25, 25, 75, 75))
        # intersection = 50*50 = 2500, union = 10000+2500-2500 = 10000
        assert abs(iou - 2500 / 10000) < 1e-5

    def test_zero_area_box(self) -> None:
        """Zero area box should return IoU = 0."""
        assert _compute_iou((0, 0, 0, 0), (0, 0, 100, 100)) == 0.0


class TestCentroidHelper:
    """Test the _centroid helper function."""

    def test_centroid_computation(self) -> None:
        cx, cy = _centroid((10, 20, 30, 40))
        assert abs(cx - 20.0) < 1e-5
        assert abs(cy - 30.0) < 1e-5


# ─── PersonTracker ───────────────────────────────────────────────────

class TestPersonTracker:
    """Test the centroid+IoU person tracker."""

    def setup_method(self) -> None:
        self.tracker = PersonTracker(
            max_disappeared=3,
            max_distance=200.0,
            min_iou=0.1,
        )

    def test_register_new_tracks(self) -> None:
        """First detections should register as new tracks."""
        dets = [
            {"bbox": (100, 100, 200, 300), "confidence": 0.9},
            {"bbox": (300, 100, 400, 300), "confidence": 0.8},
        ]
        result = self.tracker.update(dets)
        assert len(result) == 2
        assert self.tracker.num_tracks == 2

    def test_track_ids_are_sequential(self) -> None:
        """Track IDs should start at 0 and increment."""
        dets = [{"bbox": (0, 0, 50, 50), "confidence": 0.9}]
        result = self.tracker.update(dets)
        assert 0 in result

        dets2 = [
            {"bbox": (0, 0, 50, 50), "confidence": 0.9},
            {"bbox": (200, 200, 250, 250), "confidence": 0.8},
        ]
        result2 = self.tracker.update(dets2)
        assert 0 in result2  # matched existing
        assert 1 in result2  # new track

    def test_match_same_detection(self) -> None:
        """Same detection across frames should keep the same track ID."""
        det = [{"bbox": (100, 100, 200, 300), "confidence": 0.9}]
        r1 = self.tracker.update(det)
        tid = list(r1.keys())[0]

        # Slightly moved detection
        det2 = [{"bbox": (105, 102, 205, 302), "confidence": 0.85}]
        r2 = self.tracker.update(det2)
        assert tid in r2

    def test_disappearance_counter(self) -> None:
        """Track should survive max_disappeared frames without detection."""
        det = [{"bbox": (100, 100, 200, 300), "confidence": 0.9}]
        self.tracker.update(det)
        assert self.tracker.num_tracks == 1

        # 3 frames with no detections (max_disappeared=3)
        for _ in range(3):
            self.tracker.update([])
        assert self.tracker.num_tracks == 1  # still alive

        # One more empty frame should remove it
        self.tracker.update([])
        assert self.tracker.num_tracks == 0

    def test_reset(self) -> None:
        """Reset should clear all tracks."""
        det = [{"bbox": (100, 100, 200, 300), "confidence": 0.9}]
        self.tracker.update(det)
        assert self.tracker.num_tracks == 1
        self.tracker.reset()
        assert self.tracker.num_tracks == 0

    def test_active_tracks_returns_copy(self) -> None:
        """active_tracks should return a copy, not the internal dict."""
        det = [{"bbox": (100, 100, 200, 300), "confidence": 0.9}]
        self.tracker.update(det)
        tracks = self.tracker.active_tracks
        tracks.clear()
        assert self.tracker.num_tracks == 1  # internal unchanged

    def test_far_away_detection_new_track(self) -> None:
        """Detection far from existing track should create a new track."""
        self.tracker.update([{"bbox": (0, 0, 50, 50), "confidence": 0.9}])
        assert self.tracker.num_tracks == 1
        # Very far detection
        self.tracker.update([
            {"bbox": (0, 0, 50, 50), "confidence": 0.9},
            {"bbox": (500, 500, 600, 600), "confidence": 0.8},
        ])
        assert self.tracker.num_tracks == 2

    def test_output_has_track_id_key(self) -> None:
        """Each output detection should contain a 'track_id' key."""
        det = [{"bbox": (100, 100, 200, 300), "confidence": 0.9}]
        result = self.tracker.update(det)
        for tid, d in result.items():
            assert "track_id" in d
            assert d["track_id"] == tid


# ─── SkeletonDrawer ──────────────────────────────────────────────────

class TestSkeletonDrawer:
    """Test SkeletonDrawer initialization and drawing logic."""

    def test_mediapipe_format(self) -> None:
        """MediaPipe format should be accepted."""
        drawer = SkeletonDrawer(keypoint_format="mediapipe")
        assert drawer._format == "mediapipe"

    def test_coco_format(self) -> None:
        """COCO format should be accepted."""
        drawer = SkeletonDrawer(keypoint_format="coco")
        assert drawer._format == "coco"

    def test_invalid_format_raises(self) -> None:
        """Unknown keypoint format should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown keypoint format"):
            SkeletonDrawer(keypoint_format="openpose")

    def test_draw_empty_keypoints(self) -> None:
        """Drawing with empty keypoints should return frame unchanged."""
        drawer = SkeletonDrawer(keypoint_format="mediapipe")
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = drawer.draw(frame, np.array([]))
        np.testing.assert_array_equal(result, frame)

    def test_draw_none_keypoints(self) -> None:
        """Drawing with None keypoints should return frame unchanged."""
        drawer = SkeletonDrawer(keypoint_format="mediapipe")
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = drawer.draw(frame, None)
        np.testing.assert_array_equal(result, frame)

    def test_draw_modifies_in_place(self) -> None:
        """draw() draws in-place on the input frame for efficiency."""
        drawer = SkeletonDrawer(keypoint_format="mediapipe")
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        kps = np.zeros((33, 4), dtype=np.float32)
        kps[:, 3] = 0.9
        kps[:, 0] = 320
        kps[:, 1] = 240
        result = drawer.draw(frame, kps)
        # draw() returns the same array (in-place)
        assert result is frame
        # Skeleton was drawn — frame is no longer all zeros
        assert frame.sum() > 0

    def test_draw_bbox_returns_copy(self) -> None:
        """draw_bbox() should not modify the original frame."""
        drawer = SkeletonDrawer(keypoint_format="mediapipe")
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        original = frame.copy()
        drawer.draw_bbox(frame, (10, 10, 100, 200))
        np.testing.assert_array_equal(frame, original)

    def test_draw_low_confidence_skipped(self) -> None:
        """Keypoints below confidence threshold should not be drawn."""
        drawer = SkeletonDrawer(
            keypoint_format="mediapipe",
            confidence_threshold=0.5,
        )
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        kps = np.zeros((33, 4), dtype=np.float32)
        kps[:, 3] = 0.1  # all below threshold
        kps[:, 0] = 320
        kps[:, 1] = 240
        result = drawer.draw(frame, kps)
        # Frame should still be all black (nothing drawn)
        np.testing.assert_array_equal(result, frame)


# ─── DisplayManager ──────────────────────────────────────────────────

class TestDisplayManager:
    """Test DisplayManager initialization and state color mapping."""

    def test_default_init(self) -> None:
        """Default initialization should set correct attributes."""
        dm = DisplayManager()
        assert dm._window_name == "Fall Detection System"
        assert dm._resolution == (640, 480)
        assert dm._is_open is False

    def test_custom_init(self) -> None:
        """Custom initialization should override defaults."""
        dm = DisplayManager(window_name="Test", resolution=(1280, 720))
        assert dm._window_name == "Test"
        assert dm._resolution == (1280, 720)

    def test_state_color_mapping(self) -> None:
        """All expected alarm states should have color mappings."""
        assert "NORMAL" in _STATE_COLORS
        assert "IMPACT_DETECTED" in _STATE_COLORS
        assert "FALL_CONFIRMED" in _STATE_COLORS
        assert "RECOVERED" in _STATE_COLORS

    def test_normal_is_green(self) -> None:
        """NORMAL state should map to green."""
        assert _STATE_COLORS["NORMAL"] == COLOR_NORMAL

    def test_fall_confirmed_is_alarm(self) -> None:
        """FALL_CONFIRMED state should map to alarm (red) color."""
        assert _STATE_COLORS["FALL_CONFIRMED"] == COLOR_ALARM

    def test_destroy_when_not_open(self) -> None:
        """Destroying a non-open display should not raise."""
        dm = DisplayManager()
        dm.destroy()  # should be a no-op


# ─── Keypoints module ───────────────────────────────────────────────

class TestKeypointsModule:
    """Test keypoint constants and utility functions."""

    def test_mediapipe_has_33_keypoints(self) -> None:
        """MediaPipe should define exactly 33 keypoints."""
        assert len(MediaPipeKeypoint) == 33

    def test_coco_has_17_keypoints(self) -> None:
        """COCO should define exactly 17 keypoints."""
        assert len(COCOKeypoint) == 17

    def test_get_index_mediapipe(self) -> None:
        """get_index should return correct MediaPipe indices."""
        assert get_index("LEFT_SHOULDER", "mediapipe") == 11
        assert get_index("NOSE", "mediapipe") == 0

    def test_get_index_coco(self) -> None:
        """get_index should return correct COCO indices."""
        assert get_index("LEFT_SHOULDER", "coco") == 5
        assert get_index("NOSE", "coco") == 0

    def test_get_index_invalid_format(self) -> None:
        """Invalid format should raise ValueError."""
        with pytest.raises(ValueError):
            get_index("NOSE", "invalid")

    def test_get_index_invalid_joint(self) -> None:
        """Invalid joint name should raise KeyError."""
        with pytest.raises(KeyError):
            get_index("NONEXISTENT_JOINT", "mediapipe")

    def test_skeleton_connections_valid(self) -> None:
        """All skeleton connection indices should be within keypoint range."""
        for i, j in MEDIAPIPE_SKELETON_CONNECTIONS:
            assert 0 <= i < 33
            assert 0 <= j < 33
        for i, j in COCO_SKELETON_CONNECTIONS:
            assert 0 <= i < 17
            assert 0 <= j < 17

    def test_connections_are_pairs(self) -> None:
        """Connections should be 2-tuples."""
        for conn in MEDIAPIPE_SKELETON_CONNECTIONS:
            assert len(conn) == 2
        for conn in COCO_SKELETON_CONNECTIONS:
            assert len(conn) == 2

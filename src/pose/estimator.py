"""
Pose estimation wrapper for extracting human skeleton keypoints.

Supports three backends:
  - MediaPipe Pose (primary, 33 BlazePose landmarks, normalized coords)
  - YOLO-Pose (Ultralytics, 17 COCO keypoints, pixel coords)
  - RTMPose (rtmlib, 17 COCO keypoints, pixel coords)

DECISION: MediaPipe Pose is the primary backend per research/3.1.
It achieves >30 FPS on CPU with 33-point 3D topology, and its
BlazePose detector-tracker pipeline minimizes per-frame latency
after initialization.

Known limitation (research/3.1): MediaPipe struggles to initialize
tracking on subjects already lying down. The detector relies on
upright face/hip detection for the initial orientation vector.
Mitigation: Use continuous tracking mode; if tracking is lost and
the subject is horizontal, the alarm system uses the last known
keypoints and bounding box aspect ratio as fallback signals.

RTMPose alternative: top-down ONNX-based estimator robust to non-
canonical poses (lying down, occlusion) and supports multi-person
output. Uses 17 COCO keypoints — feature extractor must be
instantiated with `keypoint_format="coco"`.

Reference: research/3.1 - MediaPipe achieves >30 FPS on CPU with
33-point 3D topology, superior for CPU-bound fall detection.
Reference: research/8.1 - MediaPipe provides x, y, z, visibility
per landmark (132 raw features per frame).
"""

from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)


class PoseEstimator:
    """
    Wrapper around pose estimation backends (MediaPipe, YOLO-Pose, RTMPose).

    MediaPipe outputs normalized coordinates (0.0-1.0 range relative
    to frame dimensions). YOLO-Pose and RTMPose output pixel coordinates.

    Parameters
    ----------
    backend : str
        Pose estimation backend: 'mediapipe', 'yolo_pose', or 'rtmpose'.
    model_complexity : int
        MediaPipe model complexity: 0=Lite, 1=Full, 2=Heavy.
    min_detection_confidence : float
        Minimum confidence for initial detection.
    min_tracking_confidence : float
        Minimum confidence for temporal tracking.
    rtmpose_mode : str
        RTMPose model size: 'lightweight', 'balanced', or 'performance'.
        Maps to the rtmlib `mode` argument (smaller=faster, larger=accurate).
    rtmpose_device : str
        RTMPose ONNXRuntime device: 'cpu' or 'cuda'.
    """

    _VALID_BACKENDS = ("mediapipe", "yolo_pose", "rtmpose")
    _VALID_RTMPOSE_MODES = ("lightweight", "balanced", "performance")

    def __init__(
        self,
        backend: str = "mediapipe",
        model_complexity: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        rtmpose_mode: str = "balanced",
        rtmpose_device: str = "cpu",
    ) -> None:
        if backend not in self._VALID_BACKENDS:
            raise ValueError(
                f"Unknown pose backend: {backend}. "
                f"Must be one of {self._VALID_BACKENDS}."
            )

        self._backend = backend
        self._pose = None
        self._min_kp_confidence = min_detection_confidence

        if backend == "mediapipe":
            self._init_mediapipe(
                model_complexity=model_complexity,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )
        elif backend == "yolo_pose":
            self._init_yolo_pose()
        else:
            self._init_rtmpose(
                mode=rtmpose_mode,
                device=rtmpose_device,
            )

    # Map model_complexity to task model filenames
    _MODEL_FILES = {
        0: "pose_landmarker_lite.task",
        1: "pose_landmarker_full.task",
        2: "pose_landmarker_heavy.task",
    }

    def _init_mediapipe(
        self,
        model_complexity: int,
        min_detection_confidence: float,
        min_tracking_confidence: float,
    ) -> None:
        """
        Initialize MediaPipe Pose backend using the Tasks API.

        mediapipe >= 0.10.8 removed mp.solutions entirely.
        The new entry point is mp.tasks.vision.PoseLandmarker with a
        downloaded .task model file.
        """
        try:
            import mediapipe as mp
        except ImportError as e:
            raise ImportError(
                "mediapipe package is required. "
                "Install with: pip install mediapipe"
            ) from e

        # Resolve model file path
        model_filename = self._MODEL_FILES.get(
            model_complexity, self._MODEL_FILES[1]
        )
        model_dir = Path(__file__).resolve().parent.parent.parent / "models"
        model_path = model_dir / model_filename

        if not model_path.exists():
            # Fall back to lite variant if requested model is missing
            lite_path = model_dir / self._MODEL_FILES[0]
            if lite_path.exists():
                model_path = lite_path
                logger.warning(
                    f"{model_filename} not found, falling back to "
                    f"{lite_path.name}"
                )
            else:
                raise FileNotFoundError(
                    f"MediaPipe model not found at {model_path}. "
                    f"Download it with: python scripts/download_dataset.py "
                    f"--pose-model, or manually place a .task file in "
                    f"{model_dir}/"
                )

        self._mp = mp
        options = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(
                model_asset_path=str(model_path)
            ),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._pose = mp.tasks.vision.PoseLandmarker.create_from_options(
            options
        )
        self._timestamp_ms = 0
        self._num_keypoints = 33

        logger.info(
            "MediaPipe PoseLandmarker initialized",
            extra={
                "model": model_path.name,
                "det_conf": min_detection_confidence,
                "track_conf": min_tracking_confidence,
            },
        )

    def _init_yolo_pose(self) -> None:
        """Initialize YOLO-Pose backend."""
        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise ImportError(
                "ultralytics package is required for YOLO-Pose. "
                "Install with: pip install ultralytics"
            ) from e

        self._yolo_model = YOLO("yolov8n-pose.pt")
        self._num_keypoints = 17

        logger.info("YOLO-Pose initialized")

    def _init_rtmpose(self, mode: str, device: str) -> None:
        """
        Initialize RTMPose backend via rtmlib.

        rtmlib downloads ONNX weights on first use to ~/.cache/rtmlib/.
        Uses ONNXRuntime for inference; no PyTorch dependency at runtime.

        Parameters
        ----------
        mode : str
            One of 'lightweight', 'balanced', 'performance'.
        device : str
            ONNXRuntime device: 'cpu' or 'cuda'.
        """
        if mode not in self._VALID_RTMPOSE_MODES:
            raise ValueError(
                f"Unknown rtmpose_mode: {mode}. "
                f"Must be one of {self._VALID_RTMPOSE_MODES}."
            )

        try:
            from rtmlib import Body
        except ImportError as e:
            raise ImportError(
                "rtmlib package is required for RTMPose backend. "
                "Install with: pip install rtmlib"
            ) from e

        # Body() uses an internal detector + RTMPose head. Output is COCO 17.
        self._rtmpose_model = Body(
            mode=mode,
            to_openpose=False,
            backend="onnxruntime",
            device=device,
        )
        self._num_keypoints = 17

        logger.info(
            "RTMPose initialized",
            extra={"mode": mode, "device": device},
        )

    @property
    def backend(self) -> str:
        """Return the active backend name."""
        return self._backend

    @property
    def num_keypoints(self) -> int:
        """Return the number of keypoints produced."""
        return self._num_keypoints

    def estimate(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract pose keypoints from a single video frame.

        Parameters
        ----------
        frame : np.ndarray
            BGR image array of shape (H, W, 3).

        Returns
        -------
        np.ndarray or None
            Keypoints array of shape (N, 4) where columns are
            (x, y, z, visibility). For MediaPipe, coordinates are
            normalized (0.0-1.0). For YOLO-Pose, coordinates are
            in pixels. Returns None if no person detected.
        """
        if self._backend == "mediapipe":
            return self._estimate_mediapipe(frame)
        elif self._backend == "yolo_pose":
            return self._estimate_yolo(frame)
        else:
            return self._estimate_rtmpose(frame)

    def estimate_with_bbox(
        self,
        frame: np.ndarray,
        bbox: Tuple[int, int, int, int],
    ) -> Optional[np.ndarray]:
        """
        Extract pose from a cropped region defined by a bounding box.

        Useful when person detection is done separately (e.g., YOLOv8
        detects a lying person that MediaPipe's detector would miss).

        Parameters
        ----------
        frame : np.ndarray
            Full BGR image of shape (H, W, 3).
        bbox : tuple
            (x1, y1, x2, y2) bounding box of the detected person.

        Returns
        -------
        np.ndarray or None
            Keypoints in the coordinate space of the full frame.
            Shape (N, 4) with (x, y, z, visibility).
        """
        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]

        # Clamp to frame bounds
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        if x2 <= x1 or y2 <= y1:
            return None

        crop = frame[y1:y2, x1:x2]

        keypoints = self.estimate(crop)
        if keypoints is None:
            return None

        # Transform coordinates back to full frame space
        if self._backend == "mediapipe":
            # MediaPipe returns normalized coords relative to crop
            crop_h, crop_w = crop.shape[:2]
            result = keypoints.copy()
            # Convert from crop-normalized to full-frame-normalized
            result[:, 0] = (keypoints[:, 0] * crop_w + x1) / w
            result[:, 1] = (keypoints[:, 1] * crop_h + y1) / h
            # z is depth relative to hip, scale by crop height ratio
            result[:, 2] = keypoints[:, 2] * (crop_h / h)
            return result
        else:
            # YOLO-Pose / RTMPose return pixel coords relative to crop
            result = keypoints.copy()
            result[:, 0] += x1
            result[:, 1] += y1
            return result

    def _estimate_mediapipe(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Run MediaPipe PoseLandmarker (Tasks API).

        Parameters
        ----------
        frame : np.ndarray
            BGR image of shape (H, W, 3).

        Returns
        -------
        np.ndarray or None
            Shape (33, 4) with (x, y, z, visibility), normalized coords.
        """
        import cv2

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = self._mp.Image(
            image_format=self._mp.ImageFormat.SRGB, data=rgb_frame
        )

        # Timestamp must be strictly monotonically increasing.
        # Use real wall-clock time for accurate tracking behavior.
        import time as _time

        new_ts = int(_time.monotonic() * 1000)
        # Ensure strictly increasing (minimum +1ms increment)
        self._timestamp_ms = max(self._timestamp_ms + 1, new_ts)
        result = self._pose.detect_for_video(mp_image, self._timestamp_ms)

        if not result.pose_landmarks:
            return None

        landmarks = result.pose_landmarks[0]
        keypoints = np.array(
            [
                [lm.x, lm.y, lm.z, lm.visibility]
                for lm in landmarks
            ],
            dtype=np.float32,
        )

        return keypoints

    def _estimate_yolo(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Run YOLO-Pose estimation.

        Parameters
        ----------
        frame : np.ndarray
            BGR image of shape (H, W, 3).

        Returns
        -------
        np.ndarray or None
            Shape (17, 4) with (x, y, z=0, confidence), pixel coords.
        """
        results = self._yolo_model(frame, verbose=False)

        if not results or results[0].keypoints is None:
            return None

        kps_data = results[0].keypoints.data
        if kps_data.shape[0] == 0:
            return None

        # Take the first detected person
        kps = kps_data[0].cpu().numpy()  # shape: (17, 3) -> (x, y, conf)

        # Convert to (x, y, z=0, visibility) format
        keypoints = np.zeros((17, 4), dtype=np.float32)
        keypoints[:, 0] = kps[:, 0]  # x (pixels)
        keypoints[:, 1] = kps[:, 1]  # y (pixels)
        keypoints[:, 2] = 0.0        # z (not available in 2D)
        keypoints[:, 3] = kps[:, 2]  # confidence as visibility

        return keypoints

    def _estimate_rtmpose(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Run RTMPose estimation via rtmlib.

        rtmlib's `Body.__call__` returns (keypoints, scores) where:
          - keypoints: ndarray of shape (N, 17, 2) — (x, y) pixel coords
          - scores: ndarray of shape (N, 17) — per-keypoint confidence
        N is the number of detected persons (top-down). Returns None if
        no person is detected.

        Parameters
        ----------
        frame : np.ndarray
            BGR image of shape (H, W, 3).

        Returns
        -------
        np.ndarray or None
            Shape (17, 4) with (x, y, z=0, confidence), pixel coords.
            Picks the highest-mean-confidence detection if multiple persons.
        """
        try:
            keypoints, scores = self._rtmpose_model(frame)
        except Exception as e:
            logger.warning(f"RTMPose inference failed: {e}")
            return None

        if keypoints is None or len(keypoints) == 0:
            return None

        kps_arr = np.asarray(keypoints)
        scores_arr = np.asarray(scores)

        if kps_arr.ndim != 3 or kps_arr.shape[0] == 0:
            return None

        # Pick the most confident detection (top-down may return multiple)
        best_idx = int(np.argmax(scores_arr.mean(axis=1)))
        kps = kps_arr[best_idx]    # (17, 2)
        confs = scores_arr[best_idx]  # (17,)

        result = np.zeros((17, 4), dtype=np.float32)
        result[:, 0] = kps[:, 0]
        result[:, 1] = kps[:, 1]
        result[:, 2] = 0.0
        result[:, 3] = confs

        return result

    def to_pixel_coords(
        self, keypoints: np.ndarray, frame_width: int, frame_height: int
    ) -> np.ndarray:
        """
        Convert normalized keypoints to pixel coordinates.

        Only needed for MediaPipe (which returns 0.0-1.0 normalized).

        Parameters
        ----------
        keypoints : np.ndarray
            Normalized keypoints of shape (N, 4).
        frame_width : int
            Frame width in pixels.
        frame_height : int
            Frame height in pixels.

        Returns
        -------
        np.ndarray
            Pixel-space keypoints of shape (N, 4).
        """
        result = keypoints.copy()

        if self._backend == "mediapipe":
            result[:, 0] = keypoints[:, 0] * frame_width
            result[:, 1] = keypoints[:, 1] * frame_height
            # z is relative depth; scale by frame height for rough pixel scale
            result[:, 2] = keypoints[:, 2] * frame_height

        return result

    def release(self) -> None:
        """Release pose estimation resources."""
        if self._backend == "mediapipe" and self._pose is not None:
            self._pose.close()
            self._pose = None
            logger.info("MediaPipe Pose resources released")
        elif self._backend == "yolo_pose":
            self._yolo_model = None
            logger.info("YOLO-Pose resources released")
        elif self._backend == "rtmpose":
            self._rtmpose_model = None
            logger.info("RTMPose resources released")

"""
Application entry point for the fall detection system.

Orchestrates the complete pipeline: video capture -> detection ->
pose estimation -> feature extraction -> LSTM classification ->
alarm logic -> display. Uses producer-consumer threading.

Reference: research/6.1 - End-to-end pipeline architecture.
Reference: research/8.1 - Background thread for video capture,
main thread for ML processing and cv2.imshow display.
Reference: research/8.4 - Non-blocking multithreaded architecture.
"""

import time
import threading
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import yaml

from src.features.extractor import FeatureExtractor
from src.features.window import SlidingWindow
from src.models.classifier import FallClassifier
from src.alarm.detector import AlarmDetector, FallState
from src.alarm.logger import EventLogger
from src.visualization.skeleton import SkeletonDrawer, COLOR_NORMAL, COLOR_WARNING, COLOR_ALARM
from src.visualization.display import DisplayManager
from src.utils.logger import get_logger

logger = get_logger(__name__)


class VideoCapture:
    """
    Threaded video capture to decouple I/O from processing.

    Reads frames in a background thread and provides the latest
    frame via a thread-safe getter.

    Parameters
    ----------
    source : int or str
        Camera index or video file path.
    target_fps : float
        Target frame rate (controls capture interval).
    """

    def __init__(self, source, target_fps: float = 15.0) -> None:
        self._cap = cv2.VideoCapture(source)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {source}")

        self._target_fps = target_fps
        self._frame = None
        self._lock = threading.Lock()
        self._running = False
        self._thread = None

    def start(self) -> "VideoCapture":
        """Start the background capture thread."""
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        return self

    def _capture_loop(self) -> None:
        """Background loop that continuously reads frames."""
        interval = 1.0 / self._target_fps
        while self._running:
            ret, frame = self._cap.read()
            if not ret:
                self._running = False
                break
            with self._lock:
                self._frame = frame
            time.sleep(interval)

    def read(self) -> Optional[np.ndarray]:
        """Get the latest captured frame (thread-safe)."""
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    def stop(self) -> None:
        """Stop capture and release resources."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._cap is not None:
            self._cap.release()

    @property
    def is_running(self) -> bool:
        """Whether the capture thread is active."""
        return self._running


def _load_config(config_path: Optional[Path] = None) -> dict:
    """Load YAML configuration file."""
    if config_path is None:
        config_path = Path("config/config.yaml")
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run(config_path: Optional[Path] = None) -> None:
    """
    Launch the fall detection application.

    Parameters
    ----------
    config_path : Path, optional
        Path to the YAML configuration file.
        Defaults to config/config.yaml.
    """
    config = _load_config(config_path)

    video_cfg = config["video"]
    alarm_cfg = config["alarm"]
    model_cfg = config["model"]
    feature_cfg = config["features"]
    pose_cfg = config["pose"]

    # Initialize components
    target_fps = video_cfg["target_fps"]
    dt = 1.0 / target_fps

    feature_extractor = FeatureExtractor(
        keypoint_format=pose_cfg["backend"],
        confidence_threshold=pose_cfg.get("keypoint_confidence_threshold", 0.5),
    )

    window = SlidingWindow(
        window_size=feature_cfg["window_size"],
        num_features=FeatureExtractor.NUM_FEATURES,
    )

    # Load model if available
    model_dir = Path(config["paths"]["models"])
    model_path = model_dir / "best_model.pth"
    classifier = None
    if model_path.exists():
        classifier = FallClassifier(
            model_path=model_path,
            input_size=FeatureExtractor.NUM_FEATURES,
            hidden_size=model_cfg["hidden_size"],
            num_layers=model_cfg["num_layers"],
            bidirectional=model_cfg.get("bidirectional", False),
        )
        logger.info("Model loaded for inference")
    else:
        logger.warning(f"No model found at {model_path} — running in skeleton-only mode")

    alarm = AlarmDetector(
        confidence_threshold=alarm_cfg["confidence_threshold"],
        persistence_frames=alarm_cfg["persistence_frames"],
        cooldown_seconds=alarm_cfg["cooldown_seconds"],
        stillness_duration=alarm_cfg.get("stillness_duration_seconds", 5.0),
        ema_alpha=alarm_cfg.get("ema_alpha", 0.3),
        fps=target_fps,
    )

    event_logger = EventLogger(
        db_path=Path(config["paths"]["logs"]) / "events.db"
    )

    skeleton_drawer = SkeletonDrawer(
        keypoint_format="mediapipe" if pose_cfg["backend"] == "mediapipe" else "coco",
    )

    display = DisplayManager(
        resolution=tuple(video_cfg["resolution"]),
    )

    # Initialize pose estimator
    from src.pose.estimator import PoseEstimator

    pose_estimator = PoseEstimator(
        backend=pose_cfg["backend"],
        model_complexity=pose_cfg.get("model_complexity", 1),
        min_detection_confidence=pose_cfg["min_detection_confidence"],
        min_tracking_confidence=pose_cfg["min_tracking_confidence"],
    )

    # Start video capture
    source = video_cfg["source"]
    capture = VideoCapture(source, target_fps=target_fps).start()
    logger.info(f"Video capture started: source={source}, fps={target_fps}")

    prev_keypoints = None
    prev_velocity = float("nan")
    frame_count = 0
    # Rolling FPS: track last N frame times
    _fps_window_size = 30
    _frame_times: list = []

    try:
        while capture.is_running:
            frame_start = time.monotonic()

            frame = capture.read()
            if frame is None:
                time.sleep(0.01)
                continue

            frame_count += 1

            # Pose estimation
            keypoints = pose_estimator.estimate(frame)

            fall_conf = 0.0
            alarm_state_str = "NORMAL"
            skeleton_color = COLOR_NORMAL

            if keypoints is not None:
                # Convert to pixel coords for drawing
                h, w = frame.shape[:2]
                pixel_kps = pose_estimator.to_pixel_coords(keypoints, w, h)

                # Extract features
                features = feature_extractor.extract(
                    keypoints,
                    prev_keypoints=prev_keypoints,
                    dt=dt,
                    prev_velocity=prev_velocity,
                )
                features = np.nan_to_num(features, nan=0.0)
                prev_velocity = features[3]  # CoM velocity
                prev_keypoints = keypoints

                window.push(features)

                # Classify if window is ready and model is loaded
                if window.is_ready and classifier is not None:
                    seq = window.get_array()
                    pred_class, fall_conf = classifier.predict(seq)

                    # Determine if upright from torso inclination (feature 0)
                    is_upright = features[0] < 45.0  # degrees from vertical

                    state = alarm.process_frame(fall_conf, is_upright)
                    alarm_state_str = state.name

                    if state == FallState.IMPACT_DETECTED:
                        skeleton_color = COLOR_WARNING
                    elif state == FallState.FALL_CONFIRMED:
                        skeleton_color = COLOR_ALARM
                        event_logger.log_event(
                            severity="CRITICAL",
                            confidence=alarm.peak_confidence,
                        )
                elif not window.is_ready:
                    fill = window.current_size
                    total = window.window_size
                    alarm_state_str = f"INIT ({fill}/{total})"

                # Draw skeleton AFTER classification so color reflects current state
                frame = skeleton_drawer.draw(frame, pixel_kps, color=skeleton_color)
            else:
                prev_keypoints = None
                prev_velocity = float("nan")
                alarm_state_str = "NO PERSON"

            # Rolling FPS calculation
            frame_end = time.monotonic()
            _frame_times.append(frame_end - frame_start)
            if len(_frame_times) > _fps_window_size:
                _frame_times.pop(0)
            avg_frame_time = sum(_frame_times) / len(_frame_times)
            current_fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0.0

            # Display
            if not display.show(
                frame,
                fps=current_fps,
                alarm_state=alarm_state_str,
                confidence=fall_conf,
            ):
                break

    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
    finally:
        capture.stop()
        pose_estimator.release()
        display.destroy()
        event_logger.close()
        logger.info("Application shutdown complete")


if __name__ == "__main__":
    run()

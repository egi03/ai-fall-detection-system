"""Synchronous video-analysis pipeline used by the FastAPI uploader.

Mirrors the loop in ``scripts/demo.py`` but writes an annotated MP4 to disk
instead of opening an OpenCV window, and returns a structured result dict
(events, per-frame probabilities, optional Gemini narration). One call ==
one uploaded video.

The default model selection is the Run3 + Run5 BiLSTM ensemble — the best
configuration recorded in the project memory (AUC=0.897).
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional

import cv2
import numpy as np

from src.alarm.detector import AlarmDetector, FallState
from src.features.extractor import FeatureExtractor
from src.features.window import SlidingWindow
from src.models.classifier import FallClassifier
from src.utils.logger import get_logger
from src.visualization.skeleton import (
    COLOR_ALARM,
    COLOR_NORMAL,
    COLOR_WARNING,
    SkeletonDrawer,
)

logger = get_logger(__name__)


# Demo-mode FSM tuned for short uploaded clips, matching scripts/demo.py
_DEMO_ALARM_CFG = {
    "confidence_threshold": 0.65,
    "persistence_frames": 5,
    "cooldown_seconds": 15.0,
    "stillness_duration_seconds": 3.0,
    "ema_alpha": 0.3,
}

# Default ensemble per project memory: Run3 (models/urfd) + Run5 (models/run5_stride2_aug)
DEFAULT_MODEL_PATHS: tuple[str, ...] = ("models/urfd", "models/run5_stride2_aug")


@dataclass
class FrameProb:
    frame: int
    time_sec: float
    p_fall: float


@dataclass
class FallEvent:
    frame: int
    time_sec: float
    confidence: float


@dataclass
class PipelineResult:
    annotated_video: Path
    events: List[FallEvent] = field(default_factory=list)
    probabilities: List[FrameProb] = field(default_factory=list)
    narration: Optional[str] = None
    total_frames: int = 0
    processed_frames: int = 0
    fps: float = 15.0
    duration_sec: float = 0.0
    elapsed_sec: float = 0.0
    ensemble_models: List[str] = field(default_factory=list)
    threshold_used: float = 0.5

    def to_json(self) -> dict:
        return {
            "annotated_video": self.annotated_video.name,
            "events": [e.__dict__ for e in self.events],
            "probabilities": [p.__dict__ for p in self.probabilities],
            "narration": self.narration,
            "total_frames": self.total_frames,
            "processed_frames": self.processed_frames,
            "fps": self.fps,
            "duration_sec": self.duration_sec,
            "elapsed_sec": self.elapsed_sec,
            "ensemble_models": self.ensemble_models,
            "threshold_used": self.threshold_used,
        }


def _open_writer(path: Path, fps: float, size: tuple[int, int]) -> cv2.VideoWriter:
    """Open a VideoWriter, preferring browser-friendly H.264, falling back to mp4v.

    OpenCV's H.264 encoder needs ``openh264-*.dll`` (or ffmpeg) on the system
    path. If that isn't available the writer's ``isOpened()`` returns False and
    we fall back to ``mp4v``, which Chromium-based browsers on Windows still
    play but Firefox sometimes refuses.
    """
    for fourcc_str in ("avc1", "H264", "mp4v"):
        fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
        writer = cv2.VideoWriter(str(path), fourcc, fps, size)
        if writer.isOpened():
            logger.info(f"VideoWriter using fourcc={fourcc_str}")
            return writer
        writer.release()
    raise RuntimeError(f"Could not open VideoWriter for {path}")


def _ensemble_predict(classifiers: list, sequence: np.ndarray) -> float:
    probs = [float(clf.predict_proba(sequence)[1]) for clf in classifiers]
    return float(np.mean(probs))


def _load_classifiers(
    model_paths: List[str],
    input_size: int,
    hidden_size: int,
    num_layers: int,
    bidirectional: bool,
) -> list:
    """Load FallClassifier instances; LOSO directories resolve to fold_0/best_model.pth."""
    classifiers: list = []
    for mp in model_paths:
        p = Path(mp)
        if not p.exists():
            logger.warning(f"Model path missing, skipping: {mp}")
            continue
        if p.is_dir():
            for candidate in (p / "fold_0" / "best_model.pth", p / "best_model.pth"):
                if candidate.exists():
                    p = candidate
                    break
            else:
                logger.warning(f"No checkpoint found inside {mp}")
                continue
        classifiers.append(
            FallClassifier(
                model_path=p,
                architecture="lstm",
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                bidirectional=bidirectional,
            )
        )
        logger.info(f"Loaded classifier: {p}")
    return classifiers


def _request_vlm_narration(
    trigger_frame: np.ndarray,
    feature_history: np.ndarray,
    confidence: float,
    timeout_sec: float = 15.0,
) -> Optional[str]:
    """Synchronously fetch a Gemini narration. Returns None on any failure.

    Reuses :class:`src.alarm.vlm_narrator.VLMNarrator` (which runs the call on
    its own worker thread); we just block until ``poll()`` returns or the
    timeout fires.
    """
    try:
        from src.alarm.vlm_narrator import VLMNarrator
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"VLM narrator import failed: {exc}")
        return None

    try:
        narrator = VLMNarrator()
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"VLM narrator unavailable: {exc}")
        return None

    try:
        narrator.request_narration(trigger_frame, feature_history, confidence)
        deadline = time.monotonic() + timeout_sec
        while time.monotonic() < deadline:
            text = narrator.poll()
            if text:
                return text
            time.sleep(0.2)
        logger.warning("VLM narration timed out")
        return None
    finally:
        narrator.shutdown()


def process_video(
    video_path: Path,
    output_video_path: Path,
    config: dict,
    model_paths: Optional[List[str]] = None,
    use_vlm: bool = False,
    progress_cb: Optional[Callable[[float], None]] = None,
) -> PipelineResult:
    """Run pose -> features -> ensemble -> alarm FSM on a single video file.

    Parameters
    ----------
    video_path : Path
        Path to the uploaded video on disk.
    output_video_path : Path
        Where to write the annotated MP4. Parent directory must exist.
    config : dict
        Project config (the same dict ``scripts/demo.py`` loads).
    model_paths : list of str, optional
        Paths to ensemble checkpoints / LOSO dirs. Defaults to the Run3 + Run5
        ensemble that scored AUC=0.897 on URFD LOSO.
    use_vlm : bool
        If True, call Gemini once on the first confirmed fall and attach the
        returned narration to the result. Silently disabled if no API key.
    progress_cb : callable, optional
        Receives a float in [0, 1] every ~30 frames so the caller can report
        progress. Pipeline runs synchronously regardless.
    """
    start_wall = time.monotonic()

    pose_cfg = config["pose"]
    alarm_cfg = dict(_DEMO_ALARM_CFG)
    cfg_model = config["model"]
    window_size = config["features"]["window_size"]

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    if src_fps <= 0 or src_fps > 240:
        src_fps = float(config["video"].get("target_fps", 15))
    total_frames_meta = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
    dt = 1.0 / src_fps

    # Pose backend
    from src.pose.estimator import PoseEstimator

    pose_estimator = PoseEstimator(
        backend=pose_cfg["backend"],
        model_complexity=pose_cfg.get("model_complexity", 1),
        min_detection_confidence=pose_cfg["min_detection_confidence"],
        min_tracking_confidence=pose_cfg["min_tracking_confidence"],
        rtmpose_mode=pose_cfg.get("rtmpose_mode", "balanced"),
        rtmpose_device=pose_cfg.get("rtmpose_device", "cpu"),
    )
    kp_format = "mediapipe" if pose_cfg["backend"] == "mediapipe" else "coco"

    feature_extractor = FeatureExtractor(keypoint_format=kp_format)
    window = SlidingWindow(window_size=window_size, num_features=FeatureExtractor.NUM_FEATURES)
    skeleton_drawer = SkeletonDrawer(keypoint_format=kp_format, keypoint_radius=5, line_thickness=3)

    paths = list(model_paths) if model_paths else list(DEFAULT_MODEL_PATHS)
    classifiers = _load_classifiers(
        paths,
        input_size=FeatureExtractor.NUM_FEATURES,
        hidden_size=cfg_model["hidden_size"],
        num_layers=cfg_model["num_layers"],
        bidirectional=cfg_model.get("bidirectional", True),
    )
    if not classifiers:
        raise RuntimeError(
            "No classifier checkpoints could be loaded. Expected the Run3+Run5 ensemble at "
            f"{paths!r}."
        )

    alarm = AlarmDetector(
        confidence_threshold=alarm_cfg["confidence_threshold"],
        persistence_frames=alarm_cfg["persistence_frames"],
        cooldown_seconds=alarm_cfg["cooldown_seconds"],
        stillness_duration=alarm_cfg["stillness_duration_seconds"],
        ema_alpha=alarm_cfg["ema_alpha"],
        fps=src_fps,
    )

    writer = _open_writer(output_video_path, src_fps, (width, height))

    result = PipelineResult(
        annotated_video=output_video_path,
        total_frames=total_frames_meta,
        fps=src_fps,
        ensemble_models=paths,
        threshold_used=alarm_cfg["confidence_threshold"],
    )

    prev_keypoints = None
    prev_velocity = float("nan")
    prev_state = FallState.NORMAL
    feature_history: deque = deque(maxlen=60)
    trigger_payload: Optional[tuple] = None  # frame + features + confidence for VLM

    frame_idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1
            t_sec = frame_idx / src_fps

            keypoints = pose_estimator.estimate(frame)
            fall_conf = 0.0
            skeleton_color = COLOR_NORMAL
            state_label = "NORMAL"

            if keypoints is not None:
                h, w = frame.shape[:2]
                pixel_kps = pose_estimator.to_pixel_coords(keypoints, w, h)

                features = feature_extractor.extract(
                    keypoints,
                    prev_keypoints=prev_keypoints,
                    dt=dt,
                    prev_velocity=prev_velocity,
                )
                features = np.nan_to_num(features, nan=0.0)
                prev_velocity = features[3]
                prev_keypoints = keypoints

                window.push(features)
                feature_history.append(features.copy())

                if window.is_ready:
                    seq = window.get_array()
                    fall_conf = (
                        float(classifiers[0].predict(seq)[1])
                        if len(classifiers) == 1
                        else _ensemble_predict(classifiers, seq)
                    )

                    is_upright = features[0] < 45.0
                    state = alarm.process_frame(fall_conf, is_upright)
                    state_label = state.name

                    if state == FallState.IMPACT_DETECTED:
                        skeleton_color = COLOR_WARNING
                    elif state == FallState.FALL_CONFIRMED:
                        skeleton_color = COLOR_ALARM

                    if state == FallState.FALL_CONFIRMED and prev_state != FallState.FALL_CONFIRMED:
                        result.events.append(
                            FallEvent(
                                frame=frame_idx,
                                time_sec=round(t_sec, 3),
                                confidence=round(float(alarm.peak_confidence), 4),
                            )
                        )
                        if use_vlm and trigger_payload is None:
                            trigger_payload = (
                                frame.copy(),
                                np.array(feature_history),
                                float(fall_conf),
                            )
                    prev_state = state
                else:
                    state_label = f"INIT ({window.current_size}/{window.window_size})"

                frame = skeleton_drawer.draw(frame, pixel_kps, color=skeleton_color)
            else:
                prev_keypoints = None
                prev_velocity = float("nan")
                state_label = "NO PERSON"

            result.probabilities.append(
                FrameProb(frame=frame_idx, time_sec=round(t_sec, 3), p_fall=round(fall_conf, 4))
            )

            _draw_hud(frame, state_label, fall_conf, len(result.events))
            writer.write(frame)

            if progress_cb is not None and total_frames_meta > 0 and frame_idx % 30 == 0:
                progress_cb(min(1.0, frame_idx / total_frames_meta))
    finally:
        cap.release()
        writer.release()
        pose_estimator.release()

    result.processed_frames = frame_idx
    result.duration_sec = round(frame_idx / src_fps, 3)
    if progress_cb is not None:
        progress_cb(1.0)

    if use_vlm and trigger_payload is not None:
        frame_for_vlm, feats_for_vlm, conf_for_vlm = trigger_payload
        result.narration = _request_vlm_narration(frame_for_vlm, feats_for_vlm, conf_for_vlm)

    result.elapsed_sec = round(time.monotonic() - start_wall, 2)
    logger.info(
        f"Pipeline done: {result.processed_frames} frames in {result.elapsed_sec}s, "
        f"{len(result.events)} fall event(s)"
    )
    return result


def _draw_hud(frame: np.ndarray, state_label: str, conf: float, fall_count: int) -> None:
    """Draw a compact status banner so the annotated MP4 is interpretable on its own."""
    h = frame.shape[0]
    color = (0, 0, 255) if state_label == "FALL_CONFIRMED" else (
        (0, 165, 255) if state_label == "IMPACT_DETECTED" else (0, 255, 0)
    )
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 36), (0, 0, 0), -1)
    cv2.putText(
        frame,
        f"{state_label}   P(fall)={conf:.2f}   falls={fall_count}",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2,
        cv2.LINE_AA,
    )
    if state_label == "FALL_CONFIRMED":
        cv2.rectangle(frame, (0, h - 40), (frame.shape[1], h), (0, 0, 180), -1)
        cv2.putText(
            frame,
            "ALARM",
            (10, h - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

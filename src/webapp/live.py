"""Stateful single-frame pipeline for live webcam fall detection.

Mirrors the offline ``process_video`` loop but is shaped for a streaming
WebSocket: one ``LiveSession`` instance per browser connection, one
``process_frame`` call per JPEG the client uploads. Keypoints come back
to the browser in normalized [0, 1] space so the client draws the
skeleton on a canvas overlay — the server never returns annotated
pixels, which keeps the bandwidth budget tiny.

Defaults match the demo-mode FSM tuned in ``scripts/demo.py`` and the
Run3+Run5 ensemble recorded in project memory (AUC=0.897).
"""

from __future__ import annotations

import time
from collections import deque
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from src.alarm.detector import AlarmDetector, FallState
from src.features.extractor import FeatureExtractor
from src.features.window import SlidingWindow
from src.models.classifier import FallClassifier
from src.pose.estimator import PoseEstimator
from src.pose.keypoints import (
    COCO_SKELETON_CONNECTIONS,
    MEDIAPIPE_SKELETON_CONNECTIONS,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


# Demo-mode alarm FSM — matches scripts/demo.py and webapp/pipeline.py.
_DEMO_ALARM_CFG = {
    "confidence_threshold": 0.65,
    "persistence_frames": 5,
    "cooldown_seconds": 15.0,
    "stillness_duration_seconds": 3.0,
    "ema_alpha": 0.3,
}

# Run3 + Run5 ensemble — the AUC=0.897 configuration from project memory.
DEFAULT_MODEL_PATHS: tuple[str, ...] = ("models/urfd", "models/run5_stride2_aug")

# Browser cameras commonly stream at 30 FPS. The model was trained at 15 FPS
# (URFD / Le2i source rate) and our features include velocity/acceleration
# computed against ``dt``. Run inference at this rate to keep features
# in-distribution; the client decides whether to drop intermediate frames.
TARGET_FPS = 15.0


class LiveSession:
    """One browser ↔ one pipeline instance.

    Parameters
    ----------
    config : dict
        Project config (the same dict the FastAPI server loads from
        ``config/config.yaml``).
    model_paths : list of str, optional
        Ensemble checkpoint directories. Defaults to the Run3+Run5
        ensemble (``models/urfd`` + ``models/run5_stride2_aug``).
    fps : float, optional
        Effective processing FPS, used by the alarm FSM for its stillness
        timer and by the feature extractor for ``dt``. Defaults to
        :data:`TARGET_FPS`.
    """

    def __init__(
        self,
        config: dict,
        model_paths: Optional[List[str]] = None,
        fps: float = TARGET_FPS,
    ) -> None:
        pose_cfg = config["pose"]
        cfg_model = config["model"]
        window_size = config["features"]["window_size"]

        self._fps = float(fps)
        self._dt = 1.0 / self._fps

        self._pose = PoseEstimator(
            backend=pose_cfg["backend"],
            model_complexity=pose_cfg.get("model_complexity", 1),
            min_detection_confidence=pose_cfg["min_detection_confidence"],
            min_tracking_confidence=pose_cfg["min_tracking_confidence"],
            rtmpose_mode=pose_cfg.get("rtmpose_mode", "balanced"),
            rtmpose_device=pose_cfg.get("rtmpose_device", "cpu"),
        )
        self._kp_format = "mediapipe" if pose_cfg["backend"] == "mediapipe" else "coco"
        self._connections = (
            MEDIAPIPE_SKELETON_CONNECTIONS
            if self._kp_format == "mediapipe"
            else COCO_SKELETON_CONNECTIONS
        )

        self._features = FeatureExtractor(keypoint_format=self._kp_format)
        self._window = SlidingWindow(
            window_size=window_size,
            num_features=FeatureExtractor.NUM_FEATURES,
        )

        paths = list(model_paths) if model_paths else list(DEFAULT_MODEL_PATHS)
        self._model_paths = paths
        self._classifiers = self._load_classifiers(
            paths,
            input_size=FeatureExtractor.NUM_FEATURES,
            hidden_size=cfg_model["hidden_size"],
            num_layers=cfg_model["num_layers"],
            bidirectional=cfg_model.get("bidirectional", True),
        )
        if not self._classifiers:
            raise RuntimeError(
                "No classifier checkpoints could be loaded. Expected the Run3+Run5 ensemble at "
                f"{paths!r}."
            )

        self._alarm = AlarmDetector(
            confidence_threshold=_DEMO_ALARM_CFG["confidence_threshold"],
            persistence_frames=_DEMO_ALARM_CFG["persistence_frames"],
            cooldown_seconds=_DEMO_ALARM_CFG["cooldown_seconds"],
            stillness_duration=_DEMO_ALARM_CFG["stillness_duration_seconds"],
            ema_alpha=_DEMO_ALARM_CFG["ema_alpha"],
            fps=self._fps,
        )

        # Per-frame transient state
        self._prev_keypoints: Optional[np.ndarray] = None
        self._prev_velocity: float = float("nan")
        self._prev_state: FallState = FallState.NORMAL
        self._frame_idx: int = 0
        self._fall_count: int = 0
        # Wall-clock anchor — events report seconds since the session opened.
        self._t0 = time.monotonic()

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _load_classifiers(
        model_paths: List[str],
        input_size: int,
        hidden_size: int,
        num_layers: int,
        bidirectional: bool,
    ) -> list:
        """Resolve LOSO dirs to fold_0/best_model.pth and instantiate classifiers."""
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
            logger.info(f"LiveSession loaded classifier: {p}")
        return classifiers

    # ------------------------------------------------------------------
    # Per-session metadata used by the client to draw the overlay.
    # ------------------------------------------------------------------
    def config_payload(self) -> dict:
        """Static config the client receives once on connect."""
        return {
            "type": "config",
            "keypoint_format": self._kp_format,
            "num_keypoints": self._pose.num_keypoints,
            "bones": [[int(i), int(j)] for (i, j) in self._connections],
            "window_size": self._window.window_size,
            "fps": self._fps,
            "threshold": _DEMO_ALARM_CFG["confidence_threshold"],
            "models": list(self._model_paths),
        }

    # ------------------------------------------------------------------
    # Core per-frame entry point
    # ------------------------------------------------------------------
    def process_frame(self, jpeg_bytes: bytes) -> dict:
        """Decode one JPEG, run the pipeline, return a JSON-serializable payload.

        Parameters
        ----------
        jpeg_bytes : bytes
            Raw JPEG image from the browser ``<canvas>``.

        Returns
        -------
        dict
            ``type="frame"`` payload with keypoints (normalized), state,
            smoothed and raw probabilities, and optionally a newly fired
            fall event.
        """
        t_start = time.monotonic()
        frame = self._decode(jpeg_bytes)
        if frame is None:
            return {"type": "error", "detail": "Could not decode JPEG frame"}

        self._frame_idx += 1
        t_sec = round(time.monotonic() - self._t0, 3)

        keypoints = self._pose.estimate(frame)
        fall_conf = 0.0
        state_label = "NORMAL"
        kp_payload: Optional[list] = None
        event_payload: Optional[dict] = None
        window_progress = self._window.current_size / self._window.window_size

        if keypoints is None:
            self._prev_keypoints = None
            self._prev_velocity = float("nan")
            state_label = "NO_PERSON"
        else:
            features = self._features.extract(
                keypoints,
                prev_keypoints=self._prev_keypoints,
                dt=self._dt,
                prev_velocity=self._prev_velocity,
            )
            features = np.nan_to_num(features, nan=0.0)
            self._prev_velocity = float(features[3])
            self._prev_keypoints = keypoints
            self._window.push(features)
            window_progress = self._window.current_size / self._window.window_size

            if self._window.is_ready:
                seq = self._window.get_array()
                if len(self._classifiers) == 1:
                    fall_conf = float(self._classifiers[0].predict(seq)[1])
                else:
                    fall_conf = float(
                        np.mean(
                            [float(c.predict_proba(seq)[1]) for c in self._classifiers]
                        )
                    )
                is_upright = features[0] < 45.0
                state = self._alarm.process_frame(fall_conf, is_upright)
                state_label = state.name

                if (
                    state == FallState.FALL_CONFIRMED
                    and self._prev_state != FallState.FALL_CONFIRMED
                ):
                    self._fall_count += 1
                    event_payload = {
                        "frame": self._frame_idx,
                        "time_sec": t_sec,
                        "confidence": round(float(self._alarm.peak_confidence), 4),
                    }
                self._prev_state = state
            else:
                state_label = "INIT"

            kp_payload = self._to_normalized_keypoints(keypoints, frame.shape)

        elapsed_ms = round((time.monotonic() - t_start) * 1000.0, 1)

        return {
            "type": "frame",
            "frame": self._frame_idx,
            "time_sec": t_sec,
            "state": state_label,
            "p_fall": round(fall_conf, 4),
            "fall_count": self._fall_count,
            "window_progress": round(window_progress, 3),
            "keypoints": kp_payload,
            "event": event_payload,
            "elapsed_ms": elapsed_ms,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _decode(jpeg_bytes: bytes) -> Optional[np.ndarray]:
        """JPEG bytes → BGR uint8 image, or None on failure."""
        if not jpeg_bytes:
            return None
        try:
            buf = np.frombuffer(jpeg_bytes, dtype=np.uint8)
            return cv2.imdecode(buf, cv2.IMREAD_COLOR)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"JPEG decode failed: {exc}")
            return None

    def _to_normalized_keypoints(
        self, keypoints: np.ndarray, frame_shape: tuple
    ) -> list:
        """Return ``[[x_norm, y_norm, visibility], ...]`` in [0, 1] space.

        MediaPipe already gives normalized coords; COCO backends give
        pixel coords. Both are projected to [0, 1] so the client can scale
        them to whatever ``<video>`` size it ends up rendering at.
        """
        h, w = frame_shape[:2]
        out: list = []
        if self._kp_format == "mediapipe":
            # (x, y) already in [0, 1]; visibility in column 3.
            for row in keypoints:
                vis = float(row[3]) if keypoints.shape[1] >= 4 else 1.0
                out.append([float(row[0]), float(row[1]), round(vis, 3)])
        else:
            for row in keypoints:
                vis_col = 3 if keypoints.shape[1] >= 4 else 2
                vis = float(row[vis_col]) if keypoints.shape[1] > vis_col else 1.0
                out.append(
                    [
                        round(float(row[0]) / max(w, 1), 4),
                        round(float(row[1]) / max(h, 1), 4),
                        round(vis, 3),
                    ]
                )
        return out

    def release(self) -> None:
        """Drop the MediaPipe / RTMPose handles."""
        try:
            self._pose.release()
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Pose release failed: {exc}")

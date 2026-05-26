"""
Demo script for running fall detection on video file or webcam.

Supports single model or ensemble (averaging probabilities from
two models for improved accuracy, per ENSEMBLE_ANALYSIS.md).

Usage:
    python scripts/demo.py                              # webcam, skeleton-only
    python scripts/demo.py --source video.mp4           # video file
    python scripts/demo.py --model models/run5_stride2_aug  # single model (LOSO folds)
    python scripts/demo.py --model models/urfd --model2 models/run5_stride2_aug  # ensemble

Reference: research/8.4 - Real-time interface optimization.
"""

import argparse
import sys
import time
from collections import deque
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np
import yaml

from src.features.extractor import FeatureExtractor
from src.features.window import SlidingWindow
from src.alarm.detector import AlarmDetector, FallState
from src.visualization.skeleton import (
    SkeletonDrawer,
    COLOR_NORMAL,
    COLOR_IMMINENT,
    COLOR_WARNING,
    COLOR_ALARM,
)
from src.visualization.display import DisplayManager
from src.alarm.logger import EventLogger
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _load_dotenv(env_path: Path = Path(".env")) -> None:
    """Load ``KEY=VALUE`` pairs from a ``.env`` file into ``os.environ``.

    Tiny self-contained reader so the project doesn't need ``python-dotenv``.
    Lines starting with ``#`` and blank lines are skipped. Existing
    environment variables are not overwritten.
    """
    import os

    if not env_path.exists():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key and key not in os.environ:
            os.environ[key] = value


def _load_config() -> dict:
    """Load config or return fallback defaults."""
    config_path = Path("config/config.yaml")
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {
        "video": {"target_fps": 15, "resolution": [640, 480]},
        "pose": {
            "backend": "mediapipe",
            "model_complexity": 1,
            "min_detection_confidence": 0.5,
            "min_tracking_confidence": 0.5,
            "keypoint_confidence_threshold": 0.5,
        },
        "features": {"window_size": 30, "window_stride": 5},
        "model": {
            "hidden_size": 128,
            "num_layers": 2,
            "bidirectional": True,
        },
        "alarm": {
            "confidence_threshold": 0.75,
            "persistence_frames": 10,
            "cooldown_seconds": 30,
            "stillness_duration_seconds": 5.0,
            "ema_alpha": 0.3,
        },
    }


def _load_classifiers(
    model_paths: List[str],
    input_size: int,
    hidden_size: int,
    num_layers: int,
    bidirectional: bool,
    architecture: str = "lstm",
    transformer_cfg: Optional[dict] = None,
) -> list:
    """
    Load one or more FallClassifier instances.

    If a path points to a directory containing LOSO fold subdirectories
    (fold_0/, fold_1/, ...), loads fold_0/best_model.pth as the
    representative model.

    Parameters
    ----------
    model_paths : list of str
        Paths to model checkpoints (.pth) or LOSO directories.
    input_size : int
        Number of input features.
    hidden_size : int
        LSTM hidden dimension.
    num_layers : int
        Number of LSTM layers.
    bidirectional : bool
        Whether models are bidirectional.
    architecture : str
        Model architecture: "lstm" or "transformer_lstm".
    transformer_cfg : dict, optional
        Transformer-LSTM hyperparameters.

    Returns
    -------
    list
        List of FallClassifier instances.
    """
    from src.models.classifier import FallClassifier

    tcfg = transformer_cfg or {}
    classifiers = []
    for mp in model_paths:
        p = Path(mp)
        if not p.exists():
            logger.warning(f"Model path not found: {mp}")
            continue

        # Auto-detect architecture from run_meta.json if present
        model_arch = architecture
        model_dir = Path(mp) if Path(mp).is_dir() else Path(mp).parent
        meta_path = model_dir / "run_meta.json"
        if not meta_path.exists() and (model_dir / "fold_0").exists():
            meta_path = model_dir.parent / "run_meta.json"
        if meta_path.exists():
            import json
            with open(meta_path, "r") as f:
                meta = json.load(f)
            model_arch = meta.get("architecture", "lstm")
            if "transformer_cfg" in meta:
                tcfg = meta["transformer_cfg"]
            logger.info(f"Auto-detected architecture: {model_arch} from {meta_path}")

        # If directory, look for fold_0/best_model.pth or best_model.pth
        if p.is_dir():
            fold0 = p / "fold_0" / "best_model.pth"
            direct = p / "best_model.pth"
            if fold0.exists():
                p = fold0
            elif direct.exists():
                p = direct
            else:
                logger.warning(f"No model found in {mp}")
                continue

        classifier = FallClassifier(
            model_path=p,
            architecture=model_arch,
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            d_model=tcfg.get("d_model", 64),
            nhead=tcfg.get("nhead", 4),
            num_encoder_layers=tcfg.get("num_encoder_layers", 2),
            dim_feedforward=tcfg.get("dim_feedforward", 128),
            lstm_hidden_size=tcfg.get("lstm_hidden_size", 64),
            lstm_num_layers=tcfg.get("lstm_num_layers", 1),
        )
        classifiers.append(classifier)
        logger.info(f"Model loaded: {p} (arch={model_arch})")

    return classifiers


def _ensemble_predict(classifiers: list, sequence: np.ndarray) -> float:
    """
    Average fall probabilities across multiple classifiers.

    Parameters
    ----------
    classifiers : list
        List of FallClassifier instances.
    sequence : np.ndarray
        Feature array of shape (window_size, num_features).

    Returns
    -------
    float
        Averaged fall probability.
    """
    fall_probs = []
    for clf in classifiers:
        probs = clf.predict_proba(sequence)
        fall_probs.append(float(probs[1]))
    return float(np.mean(fall_probs))


def _load_stgcn_classifier(model_path: str):
    """Resolve a path (file or LOSO directory) to a STGCNClassifier instance."""
    from src.models.stgcn_classifier import STGCNClassifier

    p = Path(model_path)
    if not p.exists():
        logger.error(f"ST-GCN model path not found: {model_path}")
        return None
    if p.is_dir():
        fold0 = p / "fold_0" / "best_model.pth"
        direct = p / "best_model.pth"
        if fold0.exists():
            p = fold0
        elif direct.exists():
            p = direct
        else:
            logger.error(f"No ST-GCN checkpoint found in {model_path}")
            return None
    return STGCNClassifier(model_path=p)


def main(
    source: Optional[str] = None,
    model_path: Optional[str] = None,
    model_path2: Optional[str] = None,
    demo_mode: bool = False,
    explain: bool = False,
    explain_every: int = 3,
    vlm: bool = False,
    arch: str = "lstm",
) -> None:
    """
    Run the fall detection demo.

    Parameters
    ----------
    source : str or int, optional
        Video source: integer for webcam index, or path to video file.
        Defaults to webcam 0.
    model_path : str, optional
        Path to primary trained model checkpoint or LOSO directory.
    model_path2 : str, optional
        Path to secondary model for ensemble.
    demo_mode : bool
        If True, use faster alarm parameters for presentation.
    """
    _load_dotenv()  # populate GOOGLE_API_KEY etc. before constructing narrator
    config = _load_config()
    pose_cfg = config["pose"]
    alarm_cfg = config["alarm"]
    target_fps = config["video"]["target_fps"]
    dt = 1.0 / target_fps

    # Demo mode: faster alarm response for presentation
    if demo_mode:
        alarm_cfg = {
            "confidence_threshold": 0.65,
            "warning_threshold": 0.45,
            "persistence_frames": 5,
            "cooldown_seconds": 15,
            "stillness_duration_seconds": 3.0,
            "ema_alpha": 0.3,
        }
        logger.info("Demo mode: threshold=0.65, persistence=5")

    # Video source
    if source is None:
        source = 0
    elif source.isdigit():
        source = int(source)
    elif not Path(source).exists():
        logger.error(f"Video file not found: {source}")
        sys.exit(1)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        logger.error(f"Cannot open video source: {source}")
        sys.exit(1)

    # Initialize pose estimator
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

    # Optional: OneEuro keypoint smoothing — config-gated, default off.
    keypoint_smoother = None
    smoothing_cfg = pose_cfg.get("smoothing") or {}
    if smoothing_cfg.get("enabled", False):
        from src.pose.smoothing import KeypointSmoother

        num_kp = 33 if kp_format == "mediapipe" else 17
        keypoint_smoother = KeypointSmoother(
            num_keypoints=num_kp,
            fps=target_fps,
            min_cutoff=float(smoothing_cfg.get("min_cutoff", 1.0)),
            beta=float(smoothing_cfg.get("beta", 0.007)),
            d_cutoff=float(smoothing_cfg.get("d_cutoff", 1.0)),
        )
        logger.info(
            f"OneEuro smoothing enabled "
            f"(min_cutoff={smoothing_cfg.get('min_cutoff', 1.0)}, "
            f"beta={smoothing_cfg.get('beta', 0.007)})"
        )

    # Optional: per-camera baseline calibration — config-gated, default off.
    calibrator = None
    cal_cfg = (config.get("features") or {}).get("calibration") or {}
    if cal_cfg.get("enabled", False):
        from src.features.calibration import BaselineCalibrator

        warmup_frames = max(2, int(float(cal_cfg.get("warmup_seconds", 3.0)) * target_fps))
        calibrator = BaselineCalibrator(
            feature_indices=cal_cfg.get("indices", [3, 4]),
            warmup_frames=warmup_frames,
            k=float(cal_cfg.get("k", 1.5)),
            min_std=float(cal_cfg.get("min_std", 5e-4)),
        )
        logger.info(
            f"Per-camera calibration enabled "
            f"(warmup={warmup_frames} frames, k={cal_cfg.get('k', 1.5)}, "
            f"indices={cal_cfg.get('indices', [3, 4])})"
        )

    window = SlidingWindow(
        window_size=config["features"]["window_size"],
        num_features=FeatureExtractor.NUM_FEATURES,
    )

    skeleton_drawer = SkeletonDrawer(
        keypoint_format=kp_format,
        keypoint_radius=5 if demo_mode else 4,
        line_thickness=3 if demo_mode else 2,
    )

    display = DisplayManager(
        resolution=tuple(config["video"]["resolution"]),
    )

    alarm = AlarmDetector(
        confidence_threshold=alarm_cfg["confidence_threshold"],
        persistence_frames=alarm_cfg["persistence_frames"],
        cooldown_seconds=alarm_cfg["cooldown_seconds"],
        stillness_duration=alarm_cfg.get("stillness_duration_seconds", 5.0),
        ema_alpha=alarm_cfg.get("ema_alpha", 0.3),
        fps=target_fps,
        warning_threshold=alarm_cfg.get("warning_threshold"),
    )

    # Load classifiers — either the BiLSTM ensemble (default) or a single ST-GCN.
    cfg_model = config["model"]
    cfg_transformer = config.get("transformer_lstm", {})
    classifiers: list = []
    stgcn_clf = None
    if arch == "stgcn":
        if model_path is None:
            logger.error("--arch stgcn requires --model pointing at a trained ST-GCN")
            sys.exit(1)
        stgcn_clf = _load_stgcn_classifier(model_path)
        if stgcn_clf is None:
            sys.exit(1)
        logger.info(f"Inference mode: ST-GCN ({model_path})")
    else:
        model_paths = [p for p in [model_path, model_path2] if p is not None]
        classifiers = _load_classifiers(
            model_paths,
            input_size=FeatureExtractor.NUM_FEATURES,
            hidden_size=cfg_model["hidden_size"],
            num_layers=cfg_model["num_layers"],
            bidirectional=cfg_model.get("bidirectional", True),
            transformer_cfg=cfg_transformer,
        )
        if classifiers:
            mode_str = "ensemble" if len(classifiers) > 1 else "single model"
            logger.info(f"Inference mode: BiLSTM {mode_str} ({len(classifiers)} model(s))")
        else:
            logger.info("Running in skeleton-only mode (no model loaded)")

    # Optional: explainability via Integrated Gradients on the first model.
    # IG is wired for the BiLSTM feature input only; the ST-GCN path skips it.
    explainer = None
    if explain and arch == "stgcn":
        logger.warning(
            "--explain is not supported with --arch stgcn (the IG mapping "
            "targets the 15-feature LSTM input); disabling explainability."
        )
    elif explain and classifiers:
        from src.explainability import (
            IntegratedGradients,
            FEATURE_NAMES,
            joints_from_feature_attribution,
        )
        explainer = IntegratedGradients(classifiers[0].model, target_class=1, n_steps=8)
        logger.info("Explainability enabled (Integrated Gradients, n_steps=8)")
    elif explain:
        logger.warning("--explain requested but no model loaded; ignoring")

    # Optional: VLM narration
    narrator = None
    if vlm:
        try:
            from src.alarm.vlm_narrator import VLMNarrator
            narrator = VLMNarrator()
            logger.info(f"VLM narration enabled (model={narrator.model_name})")
        except Exception as exc:
            logger.warning(f"VLM narration disabled: {exc}")
            narrator = None

    # Persistent event log (always on per project spec)
    event_log = EventLogger(Path("logs/events.db"))
    last_event_id: Optional[int] = None

    # Optional: post-fall motion monitor + severity refinement
    motion_monitor = None
    severity_cfg = alarm_cfg.get("severity") or {}
    if severity_cfg.get("enabled", False):
        from src.alarm.motion_monitor import MotionMonitor, classify_severity

        motion_monitor = MotionMonitor(
            history_frames=int(severity_cfg.get("history_frames", 15)),
            motion_threshold=float(severity_cfg.get("motion_threshold", 0.003)),
        )
        SEVERE_SEC = float(severity_cfg.get("severe_seconds", 5.0))
        MINOR_SEC = float(severity_cfg.get("minor_seconds", 1.5))
        ASSESS_TIMEOUT = float(severity_cfg.get("assessment_timeout_seconds", 10.0))
        logger.info(
            f"Post-fall severity refinement enabled "
            f"(motion_threshold={severity_cfg.get('motion_threshold', 0.003)}, "
            f"severe={SEVERE_SEC}s, minor={MINOR_SEC}s)"
        )
    # Per-event severity bookkeeping
    fall_start_time: Optional[float] = None
    fall_still_seconds: float = 0.0
    severity_finalized: bool = True

    prev_keypoints = None
    prev_velocity = float("nan")
    frame_count = 0
    fall_count = 0
    prev_alarm_state = None

    # Explainability cache — recompute only every N frames; reuse between
    last_joint_attribution: Optional[np.ndarray] = None
    last_top_features: Optional[List[tuple]] = None
    explain_counter = 0
    # Pose feature buffer for VLM (one row of 15 features per frame)
    feature_history: deque = deque(maxlen=60)
    # Raw keypoint buffer used by the ST-GCN inference path
    kp_history: deque = deque(maxlen=config["features"]["window_size"])
    # Currently-displayed narration (set when API returns, cleared after duration)
    active_narration: Optional[str] = None
    narration_clear_at = 0.0
    NARRATION_DISPLAY_SECONDS = 6.0

    # Rolling FPS: track last N frame durations
    fps_window_size = 30
    frame_times: deque = deque(maxlen=fps_window_size)

    logger.info(f"Demo started: source={source}, press 'q' or ESC to quit")

    try:
        while True:
            frame_start = time.monotonic()

            ret, frame = cap.read()
            if not ret:
                # Video ended -- loop for video files
                if isinstance(source, str):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    window.reset()
                    kp_history.clear()
                    feature_history.clear()
                    alarm.reset()
                    if keypoint_smoother is not None:
                        keypoint_smoother.reset()
                    if calibrator is not None:
                        calibrator.reset()
                    if motion_monitor is not None:
                        motion_monitor.reset()
                    fall_start_time = None
                    fall_still_seconds = 0.0
                    severity_finalized = True
                    prev_keypoints = None
                    prev_velocity = float("nan")
                    continue
                break

            frame_count += 1

            # Pose estimation
            keypoints = pose_estimator.estimate(frame)
            if keypoints is not None and keypoint_smoother is not None:
                keypoints = keypoint_smoother.smooth(keypoints)
            if motion_monitor is not None:
                motion_monitor.update(keypoints)

            fall_conf = 0.0
            alarm_state_str = "NORMAL"
            skeleton_color = COLOR_NORMAL

            if keypoints is not None:
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
                prev_velocity = features[3]
                prev_keypoints = keypoints

                if calibrator is not None:
                    calibrator.observe(features)
                    features = calibrator.apply(features)

                window.push(features)
                feature_history.append(features.copy())
                kp_history.append(keypoints.copy())

                # Classify if ready — branch on architecture
                window_size_full = config["features"]["window_size"]
                model_active = (
                    stgcn_clf is not None and len(kp_history) >= window_size_full
                ) or (classifiers and window.is_ready)

                if model_active:
                    if stgcn_clf is not None:
                        kp_window = np.array(kp_history)  # (T, 33, 4)
                        _, fall_conf = stgcn_clf.predict_from_mediapipe(kp_window)
                        seq = None
                    else:
                        seq = window.get_array()
                        if len(classifiers) == 1:
                            _, fall_conf = classifiers[0].predict(seq)
                        else:
                            fall_conf = _ensemble_predict(classifiers, seq)

                    is_upright = features[0] < 45.0
                    state = alarm.process_frame(fall_conf, is_upright)
                    alarm_state_str = state.name

                    if state == FallState.FALL_CONFIRMED and prev_alarm_state != FallState.FALL_CONFIRMED:
                        fall_count += 1
                        last_event_id = event_log.log_event(
                            severity="CRITICAL",
                            confidence=alarm.peak_confidence,
                            metadata={
                                "source": str(source),
                                "frame_index": frame_count,
                                "pose_backend": pose_cfg["backend"],
                            },
                        )
                        if motion_monitor is not None:
                            fall_start_time = time.monotonic()
                            fall_still_seconds = 0.0
                            severity_finalized = False
                        if narrator is not None:
                            narrator.request_narration(
                                rgb_frame=frame.copy(),
                                feature_history=np.array(feature_history),
                                confidence=fall_conf,
                            )
                    prev_alarm_state = state

                    if state == FallState.IMMINENT:
                        skeleton_color = COLOR_IMMINENT
                    elif state == FallState.IMPACT_DETECTED:
                        skeleton_color = COLOR_WARNING
                    elif state == FallState.FALL_CONFIRMED:
                        skeleton_color = COLOR_ALARM

                    # Attribution (every N frames to stay cheap)
                    if explainer is not None:
                        explain_counter += 1
                        if explain_counter % max(1, explain_every) == 0:
                            result = explainer.explain(seq)
                            joint_scores = joints_from_feature_attribution(
                                result.feature_attribution
                            )
                            last_joint_attribution = joint_scores

                            attribs = result.feature_attribution
                            order = np.argsort(-np.abs(attribs))[:3]
                            max_abs = float(np.max(np.abs(attribs))) or 1.0
                            last_top_features = [
                                (FEATURE_NAMES[i], float(attribs[i] / max_abs))
                                for i in order
                            ]
                else:
                    if stgcn_clf is not None:
                        fill, total = len(kp_history), window_size_full
                    else:
                        fill, total = window.current_size, window.window_size
                    alarm_state_str = f"INIT ({fill}/{total})"

                if calibrator is not None and not calibrator.is_ready:
                    pct = int(round(calibrator.progress * 100))
                    alarm_state_str = f"CALIBRATING {pct}%"

                # Draw skeleton AFTER classification so color reflects current state
                if explainer is not None and last_joint_attribution is not None:
                    frame = skeleton_drawer.draw_with_attribution(
                        frame, pixel_kps, last_joint_attribution,
                        base_color=skeleton_color,
                    )
                else:
                    frame = skeleton_drawer.draw(frame, pixel_kps, color=skeleton_color)
            else:
                prev_keypoints = None
                prev_velocity = float("nan")
                alarm_state_str = "NO PERSON"

            # Rolling FPS
            frame_end = time.monotonic()
            frame_dur = frame_end - frame_start
            frame_times.append(frame_dur)
            avg_frame_time = sum(frame_times) / len(frame_times)
            current_fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0.0

            # Post-fall severity refinement (if enabled and a fall is being assessed)
            if (
                motion_monitor is not None
                and not severity_finalized
                and fall_start_time is not None
                and last_event_id is not None
            ):
                if motion_monitor.is_still:
                    fall_still_seconds += frame_dur
                else:
                    fall_still_seconds = 0.0
                elapsed_since_fall = time.monotonic() - fall_start_time
                recovered = alarm.current_state == FallState.RECOVERED
                hit_severe = fall_still_seconds >= SEVERE_SEC
                hit_timeout = elapsed_since_fall >= ASSESS_TIMEOUT
                if recovered or hit_severe or hit_timeout:
                    severity = classify_severity(
                        still_seconds=fall_still_seconds,
                        recovered=recovered,
                        severe_seconds=SEVERE_SEC,
                        minor_seconds=MINOR_SEC,
                    )
                    event_log.update_severity(last_event_id, severity)
                    event_log.update_metadata(
                        last_event_id,
                        {
                            "still_seconds": round(fall_still_seconds, 2),
                            "elapsed_seconds": round(elapsed_since_fall, 2),
                            "recovered": recovered,
                            "motion_score": round(motion_monitor.motion_score, 5),
                        },
                    )
                    logger.info(
                        f"Severity refined: event={last_event_id} → {severity} "
                        f"(still={fall_still_seconds:.1f}s, "
                        f"elapsed={elapsed_since_fall:.1f}s, recovered={recovered})"
                    )
                    severity_finalized = True

            # Pick up VLM result if it arrived this frame
            if narrator is not None:
                fresh = narrator.poll()
                if fresh is not None:
                    active_narration = fresh
                    narration_clear_at = time.monotonic() + NARRATION_DISPLAY_SECONDS
                    if last_event_id is not None:
                        event_log.update_metadata(last_event_id, {"narration": fresh})
                elif active_narration is not None and time.monotonic() >= narration_clear_at:
                    active_narration = None

            if not display.show(
                frame,
                fps=current_fps,
                alarm_state=alarm_state_str,
                confidence=fall_conf,
                fall_count=fall_count,
                top_features=last_top_features if explainer is not None else None,
                narration=active_narration,
            ):
                break

            # Frame rate limiting (per-frame)
            target_delay = 1.0 / target_fps
            elapsed = time.monotonic() - frame_start
            if elapsed < target_delay:
                time.sleep(target_delay - elapsed)

    except KeyboardInterrupt:
        logger.info("Demo interrupted")
    finally:
        cap.release()
        pose_estimator.release()
        display.destroy()
        if narrator is not None:
            narrator.shutdown()
        event_log.close()
        logger.info("Demo shutdown complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fall detection demo")
    parser.add_argument(
        "--source", type=str, default=None,
        help="Video source (0 for webcam, or path to video file)",
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Path to primary model checkpoint (.pth) or LOSO directory",
    )
    parser.add_argument(
        "--model2", type=str, default=None,
        help="Path to secondary model for ensemble (optional)",
    )
    parser.add_argument(
        "--demo-mode", action="store_true",
        help="Use faster alarm parameters for live presentation",
    )
    parser.add_argument(
        "--explain", action="store_true",
        help="Overlay Integrated-Gradients attribution on the live skeleton",
    )
    parser.add_argument(
        "--explain-every", type=int, default=3,
        help="Recompute attribution every N frames (default 3 for speed)",
    )
    parser.add_argument(
        "--vlm", action="store_true",
        help="Generate a natural-language narration via Gemini when a fall is confirmed",
    )
    parser.add_argument(
        "--arch", choices=["lstm", "stgcn"], default="lstm",
        help="Inference model family: 'lstm' (default, supports ensemble + --explain) "
             "or 'stgcn' (the 13-joint graph baseline).",
    )
    args = parser.parse_args()
    main(
        source=args.source,
        model_path=args.model,
        model_path2=args.model2,
        demo_mode=args.demo_mode,
        explain=args.explain,
        explain_every=args.explain_every,
        vlm=args.vlm,
        arch=args.arch,
    )

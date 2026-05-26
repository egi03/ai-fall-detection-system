"""
Integration tests verifying the full pipeline from keypoints to alarm.

These tests ensure all components work together correctly:
keypoints -> features -> sliding window -> LSTM -> alarm FSM.
"""

import tempfile
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.features.extractor import FeatureExtractor
from src.features.window import SlidingWindow
from src.models.lstm import FallDetectionLSTM
from src.models.classifier import FallClassifier
from src.training.dataset import FallDetectionDataset
from src.training.trainer import Trainer
from src.training.evaluate import Evaluator
from src.alarm.detector import AlarmDetector, FallState
from src.alarm.logger import EventLogger
from src.pose.keypoints import MediaPipeKeypoint


def _make_standing_kps() -> np.ndarray:
    """Standing person keypoints (MediaPipe 33-point)."""
    kps = np.zeros((33, 4), dtype=np.float32)
    kps[:, 3] = 0.9
    kps[MediaPipeKeypoint.NOSE] = [0.5, 0.1, 0.0, 0.95]
    kps[MediaPipeKeypoint.LEFT_SHOULDER] = [0.45, 0.25, 0.0, 0.9]
    kps[MediaPipeKeypoint.RIGHT_SHOULDER] = [0.55, 0.25, 0.0, 0.9]
    kps[MediaPipeKeypoint.LEFT_HIP] = [0.47, 0.50, 0.0, 0.9]
    kps[MediaPipeKeypoint.RIGHT_HIP] = [0.53, 0.50, 0.0, 0.9]
    kps[MediaPipeKeypoint.LEFT_KNEE] = [0.46, 0.70, 0.0, 0.85]
    kps[MediaPipeKeypoint.RIGHT_KNEE] = [0.54, 0.70, 0.0, 0.85]
    kps[MediaPipeKeypoint.LEFT_ANKLE] = [0.45, 0.90, 0.0, 0.8]
    kps[MediaPipeKeypoint.RIGHT_ANKLE] = [0.55, 0.90, 0.0, 0.8]
    kps[MediaPipeKeypoint.LEFT_WRIST] = [0.42, 0.50, 0.0, 0.8]
    kps[MediaPipeKeypoint.RIGHT_WRIST] = [0.58, 0.50, 0.0, 0.8]
    kps[MediaPipeKeypoint.LEFT_ELBOW] = [0.40, 0.40, 0.0, 0.85]
    kps[MediaPipeKeypoint.RIGHT_ELBOW] = [0.60, 0.40, 0.0, 0.85]
    return kps


class TestEndToEndPipeline:
    """Full pipeline integration tests."""

    def test_feature_to_window_to_model(self) -> None:
        """Features -> SlidingWindow -> LSTM should produce valid output."""
        fe = FeatureExtractor(keypoint_format="mediapipe")
        window = SlidingWindow(window_size=10, num_features=15)
        model = FallDetectionLSTM(input_size=15, hidden_size=16, num_layers=1)
        model.eval()

        kps = _make_standing_kps()
        prev_kps = None

        for _ in range(10):
            features = fe.extract(kps, prev_keypoints=prev_kps, dt=1.0 / 15)
            features = np.nan_to_num(features, nan=0.0)
            window.push(features)
            prev_kps = kps

        assert window.is_ready
        arr = window.get_tensor()
        tensor = torch.from_numpy(arr)
        with torch.no_grad():
            logits = model(tensor)
        assert logits.shape == (1, 2)

    def test_train_save_load_predict(self) -> None:
        """Train -> save -> load -> predict cycle."""
        # Create synthetic dataset
        n_samples = 30
        seq_len = 10
        n_features = 15
        seqs = np.random.randn(n_samples, seq_len, n_features).astype(np.float32)
        labels = np.random.randint(0, 2, n_samples).astype(np.int64)

        ds = FallDetectionDataset(seqs, labels)
        loader = DataLoader(ds, batch_size=8, shuffle=True)

        # Train
        model = FallDetectionLSTM(
            input_size=n_features, hidden_size=16, num_layers=1
        )
        trainer = Trainer(model, epochs=3, early_stopping_patience=100)
        history = trainer.train(loader, loader)
        assert len(history["train_loss"]) == 3

        # Save
        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as f:
            ckpt_path = Path(f.name)
        trainer.save_checkpoint(ckpt_path)

        # Load and predict
        classifier = FallClassifier(
            model_path=ckpt_path,
            input_size=n_features,
            hidden_size=16,
            num_layers=1,
        )
        pred_class, confidence = classifier.predict(seqs[0])
        assert pred_class in (0, 1)
        assert 0.0 <= confidence <= 1.0

        probs = classifier.predict_proba(seqs[0])
        assert probs.shape == (2,)
        assert abs(probs.sum() - 1.0) < 1e-5

        ckpt_path.unlink()

    def test_alarm_fsm_with_real_features(self) -> None:
        """Alarm FSM should stay NORMAL for standing poses."""
        fe = FeatureExtractor(keypoint_format="mediapipe")
        alarm = AlarmDetector(
            confidence_threshold=0.5,
            persistence_frames=3,
            ema_alpha=1.0,
            cooldown_seconds=0.0,
        )

        kps = _make_standing_kps()
        features = fe.extract(kps)
        torso_angle = features[0]

        # Standing should be upright (angle < 45)
        is_upright = torso_angle < 45.0
        assert is_upright

        # Low confidence should keep alarm in NORMAL
        for _ in range(20):
            state = alarm.process_frame(0.1, is_upright)
        assert state == FallState.NORMAL

    def test_evaluator_with_model_outputs(self) -> None:
        """Evaluator should handle realistic model outputs."""
        tmpdir = tempfile.mkdtemp()
        evaluator = Evaluator(output_dir=Path(tmpdir))

        # Simulate model predictions
        y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        y_pred = np.array([0, 0, 0, 1, 0, 1, 1, 0, 1, 1])
        y_prob = np.array([0.1, 0.2, 0.3, 0.6, 0.2, 0.8, 0.9, 0.4, 0.7, 0.85])

        metrics = evaluator.compute_metrics(y_true, y_pred, y_prob)
        assert 0.0 <= metrics["sensitivity"] <= 1.0
        assert 0.0 <= metrics["specificity"] <= 1.0
        assert metrics["tp"] + metrics["fn"] == 5  # 5 actual positives
        assert metrics["tn"] + metrics["fp"] == 5  # 5 actual negatives

    def test_event_logger_full_cycle(self) -> None:
        """Log event -> query -> mark reviewed cycle."""
        tmpdir = tempfile.mkdtemp()
        el = EventLogger(db_path=Path(tmpdir) / "test.db")

        eid = el.log_event(
            "CRITICAL",
            0.92,
            camera_id="cam_1",
            metadata={"torso_angle": 78.5},
        )

        events = el.get_events()
        assert len(events) == 1
        assert events[0]["confidence"] == 0.92

        el.mark_reviewed(eid, is_false_positive=False)
        events = el.get_events()
        assert events[0]["reviewed"] == 1
        assert events[0]["is_false_positive"] == 0

        el.close()

    def test_sequence_feature_extraction(self) -> None:
        """extract_sequence should produce features for a full sequence."""
        fe = FeatureExtractor(keypoint_format="mediapipe")
        kps = _make_standing_kps()
        sequence = np.stack([kps for _ in range(30)])

        features = fe.extract_sequence(sequence, dt=1.0 / 15)
        assert features.shape == (30, 15)
        # Velocity should be ~0 for stationary
        assert abs(features[5, 3]) < 0.01  # frame 5 velocity

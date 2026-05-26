"""
Unit tests for Phase 4-6: LSTM model, training, evaluation, and alarm modules.
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.models.lstm import FallDetectionLSTM
from src.training.trainer import Trainer
from src.training.evaluate import Evaluator
from src.training.dataset import FallDetectionDataset
from src.alarm.detector import AlarmDetector, FallState
from src.alarm.logger import EventLogger


# ─── FallDetectionLSTM ──────────────────────────────────────────────

class TestFallDetectionLSTM:
    """Test LSTM model architecture."""

    def test_output_shape(self) -> None:
        """Output should be (batch, num_classes)."""
        model = FallDetectionLSTM(input_size=15, hidden_size=64, num_layers=1)
        x = torch.randn(4, 30, 15)
        out = model(x)
        assert out.shape == (4, 2)

    def test_bidirectional_output_shape(self) -> None:
        """Bidirectional model should also produce (batch, num_classes)."""
        model = FallDetectionLSTM(
            input_size=15, hidden_size=64, num_layers=2, bidirectional=True
        )
        x = torch.randn(2, 30, 15)
        out = model(x)
        assert out.shape == (2, 2)

    def test_single_sample(self) -> None:
        """Should handle batch_size=1."""
        model = FallDetectionLSTM(input_size=15, hidden_size=32, num_layers=1)
        x = torch.randn(1, 30, 15)
        out = model(x)
        assert out.shape == (1, 2)

    def test_different_seq_len(self) -> None:
        """Should handle different sequence lengths."""
        model = FallDetectionLSTM(input_size=15, hidden_size=32, num_layers=1)
        for seq_len in [10, 30, 60]:
            x = torch.randn(2, seq_len, 15)
            out = model(x)
            assert out.shape == (2, 2)

    def test_gradient_flow(self) -> None:
        """Gradients should flow through the model."""
        model = FallDetectionLSTM(input_size=15, hidden_size=32, num_layers=1)
        x = torch.randn(2, 30, 15, requires_grad=True)
        out = model(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None

    def test_default_parameters(self) -> None:
        """Default initialization should match config values."""
        model = FallDetectionLSTM()
        assert model.input_size == 15
        assert model.hidden_size == 128
        assert model.num_layers == 2
        assert model.num_classes == 2
        assert model.bidirectional is False


# ─── Trainer ─────────────────────────────────────────────────────────

class TestTrainer:
    """Test the training loop."""

    def _make_data(self, n: int = 50):
        """Create a small synthetic dataset."""
        seqs = np.random.randn(n, 30, 15).astype(np.float32)
        labels = np.random.randint(0, 2, size=n).astype(np.int64)
        return seqs, labels

    def test_train_runs(self) -> None:
        """Training loop should complete without error."""
        model = FallDetectionLSTM(input_size=15, hidden_size=16, num_layers=1)
        trainer = Trainer(model, epochs=3, early_stopping_patience=100)

        seqs, labels = self._make_data(40)
        ds = FallDetectionDataset(seqs, labels)
        loader = DataLoader(ds, batch_size=8)

        history = trainer.train(loader, loader)
        assert len(history["train_loss"]) == 3
        assert len(history["val_loss"]) == 3

    def test_early_stopping(self) -> None:
        """Training should stop before max epochs if no improvement."""
        model = FallDetectionLSTM(input_size=15, hidden_size=8, num_layers=1)
        trainer = Trainer(
            model, epochs=100, early_stopping_patience=2, learning_rate=0.0
        )

        seqs, labels = self._make_data(20)
        ds = FallDetectionDataset(seqs, labels)
        loader = DataLoader(ds, batch_size=10)

        history = trainer.train(loader, loader)
        # Should stop early (lr=0 means no learning, so val_loss won't improve)
        assert len(history["train_loss"]) < 100

    def test_checkpoint_save_load(self) -> None:
        """Save and load checkpoint should preserve model weights."""
        model = FallDetectionLSTM(input_size=15, hidden_size=8, num_layers=1)
        trainer = Trainer(model, epochs=1)

        seqs, labels = self._make_data(20)
        ds = FallDetectionDataset(seqs, labels)
        loader = DataLoader(ds, batch_size=10)
        trainer.train(loader, loader)

        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as f:
            ckpt_path = Path(f.name)

        trainer.save_checkpoint(ckpt_path)

        model2 = FallDetectionLSTM(input_size=15, hidden_size=8, num_layers=1)
        trainer2 = Trainer(model2, epochs=1)
        trainer2.load_checkpoint(ckpt_path)

        # Compare parameters
        for p1, p2 in zip(model.parameters(), model2.parameters()):
            assert torch.allclose(p1, p2)

        ckpt_path.unlink()


# ─── Evaluator ───────────────────────────────────────────────────────

class TestEvaluator:
    """Test evaluation metrics computation."""

    def setup_method(self) -> None:
        self._tmpdir = tempfile.mkdtemp()
        self.evaluator = Evaluator(output_dir=Path(self._tmpdir))

    def test_perfect_predictions(self) -> None:
        """Perfect predictions should give 1.0 for all metrics."""
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_pred = np.array([0, 0, 0, 1, 1, 1])
        y_prob = np.array([0.1, 0.1, 0.1, 0.9, 0.9, 0.9])

        metrics = self.evaluator.compute_metrics(y_true, y_pred, y_prob)
        assert metrics["sensitivity"] == 1.0
        assert metrics["specificity"] == 1.0
        assert metrics["accuracy"] == 1.0
        assert metrics["f1"] == 1.0

    def test_all_wrong_predictions(self) -> None:
        """All-wrong predictions should give 0.0 sensitivity."""
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_pred = np.array([1, 1, 1, 0, 0, 0])
        y_prob = np.array([0.9, 0.9, 0.9, 0.1, 0.1, 0.1])

        metrics = self.evaluator.compute_metrics(y_true, y_pred, y_prob)
        assert metrics["sensitivity"] == 0.0
        assert metrics["specificity"] == 0.0

    def test_false_alarm_rate(self) -> None:
        """False alarm rate should be FP / total_hours."""
        y_true = np.array([0, 0, 0, 0, 1])
        y_pred = np.array([1, 1, 0, 0, 1])  # 2 FP

        far = self.evaluator.compute_false_alarm_rate(y_true, y_pred, 2.0)
        assert abs(far - 1.0) < 1e-5  # 2 FP / 2 hours = 1.0

    def test_false_alarm_rate_zero_hours(self) -> None:
        y_true = np.array([0, 1])
        y_pred = np.array([1, 1])
        far = self.evaluator.compute_false_alarm_rate(y_true, y_pred, 0.0)
        assert np.isnan(far)

    def test_save_results(self) -> None:
        """Metrics should be saved as valid JSON."""
        metrics = {"accuracy": 0.95, "f1": 0.92}
        path = Path(self._tmpdir) / "test_metrics.json"
        self.evaluator.save_results(metrics, path)
        assert path.exists()
        with open(path) as f:
            loaded = json.load(f)
        assert loaded["accuracy"] == 0.95

    def test_plot_confusion_matrix(self) -> None:
        """Confusion matrix plot should be saved as PNG."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 1, 0])
        path = Path(self._tmpdir) / "cm.png"
        self.evaluator.plot_confusion_matrix(y_true, y_pred, path)
        assert path.exists()

    def test_plot_roc_curve(self) -> None:
        """ROC curve plot should be saved as PNG."""
        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([0.2, 0.4, 0.6, 0.8])
        path = Path(self._tmpdir) / "roc.png"
        self.evaluator.plot_roc_curve(y_true, y_prob, path)
        assert path.exists()

    def test_plot_training_curves(self) -> None:
        """Training curves plot should be saved as PNG."""
        history = {
            "train_loss": [0.5, 0.3, 0.2],
            "val_loss": [0.6, 0.4, 0.35],
            "train_acc": [0.7, 0.8, 0.9],
            "val_acc": [0.65, 0.75, 0.8],
        }
        path = Path(self._tmpdir) / "curves.png"
        self.evaluator.plot_training_curves(history, path)
        assert path.exists()


# ─── AlarmDetector ───────────────────────────────────────────────────

class TestAlarmDetector:
    """Test the alarm FSM logic."""

    def test_initial_state_normal(self) -> None:
        """Initial state should be NORMAL."""
        ad = AlarmDetector()
        assert ad.current_state == FallState.NORMAL
        assert not ad.is_alarm_active

    def test_high_confidence_triggers_impact(self) -> None:
        """Sustained high confidence should transition to IMPACT_DETECTED."""
        ad = AlarmDetector(
            confidence_threshold=0.5,
            persistence_frames=3,
            ema_alpha=1.0,  # no smoothing
            cooldown_seconds=0.0,
        )
        for _ in range(3):
            state = ad.process_frame(0.9, is_subject_upright=False)
        assert state == FallState.IMPACT_DETECTED

    def test_low_confidence_stays_normal(self) -> None:
        """Low confidence should not trigger any transition."""
        ad = AlarmDetector(
            confidence_threshold=0.5,
            persistence_frames=3,
            ema_alpha=1.0,
        )
        for _ in range(10):
            state = ad.process_frame(0.2, is_subject_upright=True)
        assert state == FallState.NORMAL

    def test_impact_to_confirmed_after_stillness(self) -> None:
        """Sustained non-upright after impact should confirm fall."""
        ad = AlarmDetector(
            confidence_threshold=0.5,
            persistence_frames=2,
            stillness_duration=0.2,
            ema_alpha=1.0,
            fps=10.0,  # 0.2s * 10fps = 2 frames stillness
            cooldown_seconds=0.0,
        )
        # Trigger impact
        for _ in range(2):
            ad.process_frame(0.9, is_subject_upright=False)
        assert ad.current_state == FallState.IMPACT_DETECTED

        # Stillness period
        for _ in range(2):
            ad.process_frame(0.9, is_subject_upright=False)
        assert ad.current_state == FallState.FALL_CONFIRMED
        assert ad.is_alarm_active

    def test_impact_cleared_if_upright(self) -> None:
        """If subject becomes upright during impact, return to NORMAL."""
        ad = AlarmDetector(
            confidence_threshold=0.5,
            persistence_frames=2,
            ema_alpha=1.0,
            cooldown_seconds=0.0,
        )
        for _ in range(2):
            ad.process_frame(0.9, is_subject_upright=False)
        assert ad.current_state == FallState.IMPACT_DETECTED

        ad.process_frame(0.9, is_subject_upright=True)
        assert ad.current_state == FallState.NORMAL

    def test_recovery_from_confirmed(self) -> None:
        """Subject becoming upright after FALL_CONFIRMED -> RECOVERED."""
        ad = AlarmDetector(
            confidence_threshold=0.5,
            persistence_frames=1,
            stillness_duration=0.1,
            ema_alpha=1.0,
            fps=10.0,
            cooldown_seconds=0.0,
        )
        ad.process_frame(0.9, is_subject_upright=False)
        # Now IMPACT_DETECTED, stillness = 1 frame (0.1s * 10fps = 1)
        ad.process_frame(0.9, is_subject_upright=False)
        assert ad.current_state == FallState.FALL_CONFIRMED

        ad.process_frame(0.1, is_subject_upright=True)
        assert ad.current_state == FallState.RECOVERED

    def test_reset(self) -> None:
        """Reset should return to NORMAL."""
        ad = AlarmDetector(
            confidence_threshold=0.5,
            persistence_frames=1,
            ema_alpha=1.0,
            cooldown_seconds=0.0,
        )
        ad.process_frame(0.9, is_subject_upright=False)
        ad.reset()
        assert ad.current_state == FallState.NORMAL
        assert ad.smoothed_confidence == 0.0

    def test_ema_smoothing(self) -> None:
        """EMA should smooth out spiky confidence values."""
        ad = AlarmDetector(
            confidence_threshold=0.5,
            persistence_frames=100,
            ema_alpha=0.3,
            cooldown_seconds=0.0,
        )
        # Single high spike should be dampened
        ad.process_frame(1.0, is_subject_upright=True)
        assert ad.smoothed_confidence < 0.5  # 0.3 * 1.0 + 0.7 * 0 = 0.3


# ─── EventLogger ─────────────────────────────────────────────────────

class TestEventLogger:
    """Test SQLite event logging."""

    def setup_method(self) -> None:
        self._tmpdir = tempfile.mkdtemp()
        self._db_path = Path(self._tmpdir) / "test_events.db"
        self.el = EventLogger(self._db_path)

    def teardown_method(self) -> None:
        self.el.close()

    def test_log_event_returns_id(self) -> None:
        """log_event should return a positive event_id."""
        eid = self.el.log_event("CRITICAL", 0.95)
        assert eid > 0

    def test_sequential_ids(self) -> None:
        """Sequential events should get incrementing IDs."""
        e1 = self.el.log_event("WARNING", 0.8)
        e2 = self.el.log_event("CRITICAL", 0.95)
        assert e2 > e1

    def test_get_events(self) -> None:
        """get_events should return logged events."""
        self.el.log_event("CRITICAL", 0.95)
        self.el.log_event("WARNING", 0.7)
        events = self.el.get_events()
        assert len(events) == 2

    def test_get_events_filter_severity(self) -> None:
        """Severity filter should work."""
        self.el.log_event("CRITICAL", 0.95)
        self.el.log_event("WARNING", 0.7)
        events = self.el.get_events(severity="CRITICAL")
        assert len(events) == 1
        assert events[0]["severity"] == "CRITICAL"

    def test_mark_reviewed(self) -> None:
        """mark_reviewed should update the record."""
        eid = self.el.log_event("CRITICAL", 0.95)
        self.el.mark_reviewed(eid, is_false_positive=True)
        events = self.el.get_events()
        event = [e for e in events if e["event_id"] == eid][0]
        assert event["reviewed"] == 1
        assert event["is_false_positive"] == 1

    def test_update_severity(self) -> None:
        """update_severity should overwrite the severity column."""
        eid = self.el.log_event("CRITICAL", 0.95)
        self.el.update_severity(eid, "SEVERE")
        events = self.el.get_events()
        event = [e for e in events if e["event_id"] == eid][0]
        assert event["severity"] == "SEVERE"

    def test_metadata_stored(self) -> None:
        """Metadata dict should be stored as JSON."""
        meta = {"feature_0": 42.5, "note": "test"}
        eid = self.el.log_event("WARNING", 0.8, metadata=meta)
        events = self.el.get_events()
        event = [e for e in events if e["event_id"] == eid][0]
        loaded_meta = json.loads(event["metadata"])
        assert loaded_meta["feature_0"] == 42.5

    def test_limit_respected(self) -> None:
        """get_events limit parameter should cap results."""
        for _ in range(10):
            self.el.log_event("WARNING", 0.6)
        events = self.el.get_events(limit=3)
        assert len(events) == 3

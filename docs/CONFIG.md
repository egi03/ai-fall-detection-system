# Configuration

Field-by-field reference for [`config/config.yaml`](../config/config.yaml).
Defaults shipped with the repo are the values used to produce every
published result.

## `seed`

```yaml
seed: 42
```

Single global seed for NumPy, PyTorch, and Python's `random`. Used
by every script. Never change for published results.

## `video:`

```yaml
video:
  source: 0              # 0 = webcam, or path to a video file
  target_fps: 15
  resolution: [640, 480]
```

| Field | Effect |
|---|---|
| `source` | Default video source. `scripts/demo.py --source ...` overrides this. |
| `target_fps` | Used for sliding-window timing, alarm gates, OneEuro defaults. |
| `resolution` | Display window size only. Pose runs on the native frame. |

## `detection:` (YOLOv8 person detector — optional fallback)

```yaml
detection:
  model: yolov8n
  confidence_threshold: 0.5
  iou_threshold: 0.45
```

Used by `src/detection/person_detector.py` when explicitly enabled by
the pipeline (currently only by the multi-person tracking path).
The default demo loop relies on MediaPipe's internal detector.

## `pose:`

```yaml
pose:
  backend: mediapipe                  # mediapipe | yolo_pose | rtmpose
  model_complexity: 1                 # MediaPipe: 0 = Lite, 1 = Full, 2 = Heavy
  min_detection_confidence: 0.5
  min_tracking_confidence: 0.5
  smooth_landmarks: true              # built-in MediaPipe smoothing (separate from OneEuro)
  keypoint_confidence_threshold: 0.5  # discard keypoints below this in features
  rtmpose_mode: balanced              # lightweight | balanced | performance
  rtmpose_device: cpu                 # cpu | cuda
  smoothing:
    enabled: false                    # OneEuro filter, see docs/FEATURES.md
    min_cutoff: 1.0
    beta: 0.007
    d_cutoff: 1.0
```

### Choosing a backend

| Backend | Keypoints | Strength |
|---|---|---|
| `mediapipe` | 33 (BlazePose, 3D) | Default — fastest, integrates with the existing feature set. |
| `yolo_pose` | 17 (COCO, 2D) | Alternative when MediaPipe is unavailable. |
| `rtmpose` | 17 (COCO, 2D) | More robust on lying-down subjects; ONNX runtime, no PyTorch needed. |

Switching backends changes the `keypoint_format` passed to
`FeatureExtractor`; everything downstream adapts automatically.

## `features:`

```yaml
features:
  window_size: 30              # frames per classification window
  window_stride: 5             # training-time stride (inference is always 1)
  normalize: true              # person-centric normalization
  normalization_method: person_centric   # person_centric | per_sequence
  calibration:
    enabled: false             # see docs/FEATURES.md
    warmup_seconds: 3.0
    indices: [3, 4]            # com_velocity, com_acceleration
    k: 1.5
    min_std: 0.0005
```

`window_size: 30` corresponds to 2 s at 15 FPS. Empirically optimal
(`WINDOW_SIZE_SENSITIVITY.md`).

## `model:` (BiLSTM)

```yaml
model:
  hidden_size: 128
  num_layers: 2
  dropout: 0.5                 # increased from 0.3 to reduce overfitting
  bidirectional: true          # BiLSTM, not plain LSTM
  num_classes: 2               # fall / non-fall
```

## `transformer_lstm:` (alternative hybrid architecture)

```yaml
transformer_lstm:
  d_model: 64
  nhead: 4
  num_encoder_layers: 2
  dim_feedforward: 128
  lstm_hidden_size: 128
  lstm_num_layers: 1
```

Used only when a `run_meta.json` declares
`architecture: "transformer_lstm"`. See [docs/MODELS.md](MODELS.md).

## `training:`

```yaml
training:
  batch_size: 32
  learning_rate: 0.001
  weight_decay: 0.0001         # L2 regularization
  epochs: 100
  early_stopping_patience: 30  # increased from 15 — small val set needs patience
  class_weight_auto: true      # automatic class balancing
  optimizer: adam
  scheduler: reduce_on_plateau
  scheduler_patience: 5
  scheduler_factor: 0.5
  positive_label_threshold: 0.5  # ratio of fall frames to label a window as fall
  device: cpu                  # cpu | cuda | mps
```

## `alarm:`

```yaml
alarm:
  confidence_threshold: 0.75
  warning_threshold: 0.55      # IMMINENT gate (yellow). Pass to AlarmDetector(warning_threshold=...)
  persistence_frames: 10
  cooldown_seconds: 30
  stillness_duration_seconds: 5.0
  ema_alpha: 0.3
  severity:
    enabled: true              # ON by default — refines CRITICAL to SEVERE / MODERATE / MINOR
    motion_threshold: 0.003    # rolling-mean keypoint displacement floor
    history_frames: 15
    severe_seconds: 5.0
    minor_seconds: 1.5
    assessment_timeout_seconds: 10.0
```

See [docs/ALARM.md](ALARM.md) for the full FSM behaviour driven by
these knobs.

## `evaluation:`

```yaml
evaluation:
  loso: true
  metrics:
    - sensitivity
    - specificity
    - accuracy
    - f1
    - auc_roc
    - pr_auc
    - false_alarm_rate
```

Controls which metrics `scripts/evaluate.py` writes to
`results/<run>/metrics.json`.

## `paths:`

```yaml
paths:
  data_raw: data/data/raw/
  data_processed: data/data/processed/
  data_splits: data/data/splits/
  models: models/
  logs: logs/
  results: results/
```

All paths are relative to the repo root. The `data/data/` double
nesting is intentional — see [SETUP.md](SETUP.md).

## `logging:`

```yaml
logging:
  level: INFO
  format: json                 # json | text
  log_file: logs/app.log
  max_bytes: 10485760          # 10 MB
  backup_count: 5
```

Used by [`src/utils/logger.py`](../src/utils/logger.py). The JSON
formatter is what the test suite and production deployments expect;
switch to `text` for human-readable terminal output.

## Where overrides happen

- Most scripts read this file directly and respect every field.
- `scripts/demo.py --demo-mode` overrides only the `alarm:` block,
  for presentation-friendly thresholds. See [docs/DEMO.md](DEMO.md).
- `scripts/train_experiment.py` accepts CLI flags (`--stride`,
  `--augment`, `--dataset`, `--no-loso`) that override the
  corresponding YAML values for that run.

## Adding a new knob

1. Add the field to `config/config.yaml` under a sensible block.
2. Read it where it is consumed (typically near the top of a demo /
   training script). Default explicitly with `.get(key, default)`
   so older config files still work.
3. Document it here.

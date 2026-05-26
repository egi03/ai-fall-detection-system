# Architecture

End-to-end pipeline, module responsibilities, and key design decisions.

## Data flow

```
                    ┌─────────────────────────────────────────────┐
                    │             SCRIPTS/DEMO.PY                 │
                    └─────────────────────────────────────────────┘
                                       │
   ┌───────────────────────────────────┴───────────────────────────────────┐
   │                                                                       │
   ▼                                                                       │
[cv2.VideoCapture]                                                         │
   │  BGR frame                                                            │
   ▼                                                                       │
[src/pose/estimator.py]            backends: mediapipe | yolo_pose | rtmpose
   │  ndarray (N, 4) = (x, y, z, visibility)                               │
   ▼                                                                       │
[src/pose/smoothing.py]   ◄── optional, config-gated                       │
   │  smoothed keypoints                                                   │
   ▼                                                                       │
[src/features/extractor.py]                                                │
   │  ndarray (15,) per frame                                              │
   ▼                                                                       │
[src/features/calibration.py]   ◄── optional, config-gated                 │
   │  calibrated features                                                  │
   ▼                                                                       │
[src/features/window.py]                                                   │
   │  ndarray (30, 15) once warm                                           │
   ▼                                                                       │
[src/models/classifier.py]   single model OR ensemble averaging            │
   │  P(fall) ∈ [0, 1]                                                     │
   ▼                                                                       │
[src/alarm/detector.py]   FSM + EMA + persistence + cooldown               │
   │  FallState  ────────────────────────────► [src/visualization/skeleton.py]
   ▼                                                                       │ skeleton overlay
[src/alarm/motion_monitor.py] + classify_severity()                        │ with state-coloured joints
   │  severity label + motion metadata                                     │
   ▼                                                                       │
[src/alarm/logger.py]   SQLite (logs/events.db, WAL mode)                  │
                                                                           ▼
                                                                  [src/visualization/display.py]
                                                                    window with HUD
```

## Module responsibilities

| Module | Owns |
|---|---|
| `src/pose/estimator.py` | Wrapping MediaPipe / YOLO-Pose / RTMPose behind a single `estimate(frame) -> ndarray | None` API. |
| `src/pose/smoothing.py` | OneEuro filter applied per (joint, axis) before feature extraction. |
| `src/pose/keypoints.py` | Enum constants for MediaPipe (33) and COCO (17) joints; skeleton connection lists. |
| `src/detection/person_detector.py` | YOLOv8 person bounding boxes (used by tracker / fallback for lying poses). |
| `src/detection/tracker.py` | Centroid + IoU multi-person tracker (used when multiple people in frame). |
| `src/features/extractor.py` | The 15-feature vector per frame — geometric, kinematic, angular, reflexive, confidence. |
| `src/features/calibration.py` | Per-camera baseline learner + soft-threshold suppressor for velocity/accel. |
| `src/features/window.py` | Fixed-size rolling buffer over feature vectors, exposes `(W, F)` array. |
| `src/models/lstm.py` | `FallDetectionLSTM` (BiLSTM hidden=128, 2 layers, dropout=0.5). |
| `src/models/stgcn.py` | Skeleton-graph 13-joint baseline with motion channels and augmentation. |
| `src/models/transformer_lstm.py` | Hybrid Transformer encoder → LSTM decoder, alternative architecture. |
| `src/models/classifier.py` / `stgcn_classifier.py` | Inference wrappers; load checkpoints, expose `predict_proba`. |
| `src/training/trainer.py` | Adam + ReduceLROnPlateau + early stopping + checkpoint best by AUC. |
| `src/training/evaluate.py` | Metrics, confusion matrix, ROC, training curves, per-subject CSVs. |
| `src/training/dataset.py` | PyTorch Dataset / DataLoader builders, optional combined-dataset path. |
| `src/data_processing/loader.py` | Dataset-specific raw-video → keypoint pipelines (URFD, Le2i, UP-Fall). |
| `src/data_processing/preprocessor.py` | Normalization, augmentation (jitter, drop, scale, mirror). |
| `src/data_processing/splitter.py` | Subject-independent LOSO and train/val/test split generation. |
| `src/alarm/detector.py` | Five-state FSM with EMA smoothing and IMMINENT pre-warning. |
| `src/alarm/motion_monitor.py` | Per-frame keypoint displacement tracker; classifies post-fall severity. |
| `src/alarm/logger.py` | SQLite event log (WAL mode), severity + metadata writes. |
| `src/alarm/vlm_narrator.py` | Optional Gemini call to caption a confirmed fall. |
| `src/visualization/skeleton.py` | Draws joints, bones, colour-coded by FSM state; attribution heatmap blend. |
| `src/visualization/display.py` | OpenCV window with HUD (FPS, state, confidence, narration). |
| `src/explainability/attribution.py` | Integrated Gradients over the 15-feature input. |
| `src/explainability/joint_mapping.py` | Maps feature-level attribution back onto skeleton joints. |
| `src/webapp/` | Optional FastAPI server for browser-based video upload and review. |
| `src/utils/logger.py` | JSON-structured logging used everywhere. |

## Design decisions (locked unless re-justified)

These choices are grounded in the research survey under `research/` and
in empirical results documented under `DAILY LOGS/`.

| Decision | Reason |
|---|---|
| **MediaPipe as default pose backend** | Best CPU performance with 33 landmarks (3D), maintained by Google. RTMPose is offered as a more robust alternative for lying-down subjects. |
| **15 hand-designed features instead of raw keypoints** | Smaller models, faster inference, better generalization with a 5-subject training set. Raw-keypoint ST-GCN is shipped as a baseline. |
| **BiLSTM over Transformer** | At 5-subject scale, BiLSTM matches or beats transformer variants. Transformer-LSTM hybrid is available for experimentation but not the default. |
| **Window size 30 frames at 15 FPS (2 s)** | Fall dynamics fit comfortably; longer windows hurt latency without adding accuracy (see `WINDOW_SIZE_SENSITIVITY.md`). |
| **Window stride 2 frames at training time** | Sweet spot: stride 5 too sparse, stride 1 overfits (97 % window overlap). |
| **Subject-independent (LOSO) evaluation only** | Mixing subjects across train/test leaks identity; subject-independent is the only honest setup. |
| **AUC as primary metric, sens/spec at three thresholds** | A single threshold biases the comparison. Three operating points (sens-, spec-, balanced-optimal) capture the trade-off. |
| **SQLite (WAL) event log instead of files** | Atomic writes, queryable history, no race conditions when the demo and webapp share the same DB. |
| **Privacy: skeletons only** | No raw frames are written to disk by the alarm system. The VLM path sends a single frame to Gemini only on confirmed fall, and only with the user's API key. |

## Real-time alarm FSM

```
                       confidence below warning_threshold
   ┌───────────────────────────────────────────────────────┐
   │                                                       │
   │    ┌───────────┐                                      │
   └──▶ │  NORMAL   │ ◄────────────────────────────────────┤
        └─────┬─────┘                                      │
              │ smoothed_conf >= warning_threshold         │
              ▼                                            │
        ┌───────────┐  drop below warning_threshold        │
        │ IMMINENT  │ ─────────────────────────────────────┘
        └─────┬─────┘
              │ persistence_counter >= persistence_frames
              ▼
        ┌────────────────────┐
        │ IMPACT_DETECTED    │ ◄── subject upright? → NORMAL (cleared)
        └─────┬──────────────┘
              │ stillness_counter >= stillness_frames
              ▼
        ┌────────────────────┐
        │ FALL_CONFIRMED     │ ── EventLogger writes CRITICAL row
        └─────┬──────────────┘    MotionMonitor begins severity assessment
              │ subject upright
              ▼
        ┌────────────────────┐
        │ RECOVERED          │ ── 2-second hold, then NORMAL
        └────────────────────┘    severity refined and written back
```

`IMMINENT` was added in the recent batch — it is a transitional state
that fires while the smoothed probability is rising but the
high-confidence persistence gate has not yet triggered. Demo skeleton
renders this state yellow, giving a visible early signal before the
red alarm. See [docs/ALARM.md](ALARM.md).

## Inference budget (AMD Ryzen 7 7700, CPU)

| Stage | Time |
|---|---|
| MediaPipe pose | ~10 ms |
| Feature extraction | <1 ms |
| BiLSTM forward | 1.1 ms |
| Drawing / display | ~5 ms |
| **Total / frame** | **~17 ms (≈ 60 FPS budget)** |
| Real-time at 15 FPS | comfortably real-time on CPU |

ST-GCN is roughly 4× slower than the BiLSTM on the same machine; still
real-time at 15 FPS but eats more of the budget.

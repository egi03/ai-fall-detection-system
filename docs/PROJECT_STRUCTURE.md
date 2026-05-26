# Project Structure

```
ai-fall-detection-system/
├── README.md                  # Top-level index
├── requirements.txt           # Dependency 
├── Makefile                   # demo / training / analysis targets
├── pyproject.toml             # Python packaging configuration
├── .gitignore                 # Authorization for ignored artifacts
├── .gitattributes             # Git attributes configuration
├── .env                       # Local secrets (e.g. GOOGLE_API_KEY)
│
├── config/
│   └── config.yaml            # All tunable parameters (see docs/CONFIG.md)
│
├── docs/                      # ← this directory
│   ├── SETUP.md
│   ├── ARCHITECTURE.md
│   ├── FEATURES.md
│   ├── MODELS.md
│   ├── ALARM.md
│   ├── TRAINING.md
│   ├── EVALUATION.md
│   ├── DEMO.md
│   ├── EXPLAINABILITY.md
│   ├── CONFIG.md
│   └── PROJECT_STRUCTURE.md
│
├── src/
│   ├── __init__.py
│   ├── detection/
│   │   ├── person_detector.py     # YOLOv8 person detector
│   │   └── tracker.py             # Multi-person centroid + IoU tracker
│   ├── pose/
│   │   ├── estimator.py           # MediaPipe / YOLO-Pose / RTMPose unified API
│   │   ├── smoothing.py           # OneEuro keypoint filter
│   │   └── keypoints.py           # MediaPipe (33) + COCO (17) constants
│   ├── features/
│   │   ├── extractor.py           # The 15-feature vector
│   │   ├── calibration.py         # Per-camera baseline calibrator
│   │   └── window.py              # Rolling buffer for sliding window
│   ├── models/
│   │   ├── lstm.py                # FallDetectionLSTM (BiLSTM)
│   │   ├── stgcn.py               # ST-GCN graph baseline
│   │   ├── transformer_lstm.py    # Hybrid alternative
│   │   ├── classifier.py          # LSTM inference wrapper
│   │   └── stgcn_classifier.py    # ST-GCN inference wrapper
│   ├── training/
│   │   ├── trainer.py             # Adam / scheduler / early-stopping loop
│   │   ├── evaluate.py            # Metrics, plots, per-subject CSVs
│   │   └── dataset.py             # PyTorch Dataset + DataLoader builders
│   ├── data_processing/
│   │   ├── loader.py              # URFD / Le2i / UP-Fall raw → keypoints
│   │   ├── preprocessor.py        # Normalization + augmentation
│   │   └── splitter.py            # Subject-independent LOSO splits
│   ├── alarm/
│   │   ├── detector.py            # 5-state FSM (NORMAL → IMMINENT → ...)
│   │   ├── motion_monitor.py      # Post-fall keypoint stillness tracker
│   │   ├── logger.py              # SQLite event log (WAL mode)
│   │   └── vlm_narrator.py        # Optional Gemini narration (uses google-genai)
│   ├── visualization/
│   │   ├── skeleton.py            # Joint + bone drawing + attribution blend
│   │   └── display.py             # OpenCV window with HUD
│   ├── explainability/
│   │   ├── attribution.py         # Integrated Gradients
│   │   └── joint_mapping.py       # Feature → joint score projection
│   ├── webapp/
│   │   ├── server.py              # FastAPI app
│   │   ├── pipeline.py            # Server-side video processing
│   │   ├── templates/             # Jinja2 HTML
│   │   └── static/                # JS + CSS
│   ├── app/
│   │   └── main.py                # Application entry point (used by tests)
│   └── utils/
│       └── logger.py              # JSON structured logging
│
├── scripts/
│   ├── demo.py                    # Live demo (entry point)
│   ├── webapp.py                  # Webapp entry point
│   ├── preprocess.py              # Dataset preprocessing
│   ├── train.py                   # Run 3 training (URFD LOSO)
│   ├── train_experiment.py        # Configurable LOSO / full-dataset training
│   ├── train_combined.py          # Combined URFD + Le2i training
│   ├── train_stgcn.py             # ST-GCN training
│   ├── evaluate.py                # Standalone evaluation
│   ├── cross_dataset_eval.py      # URFD ↔ Le2i transfers
│   ├── cross_upfall_eval.py       # URFD → UP-Fall transfer
│   ├── eval_upfall_skeletons.py   # UP-Fall Zenodo skeleton evaluation
│   ├── ablation_study.py          # 21-condition feature ablation
│   ├── false_alarm_analysis.py    # Per-subject FP/FN breakdown
│   ├── bootstrap_ci.py            # Bootstrap CIs + ensemble vs single
│   ├── inference_benchmark.py     # Latency / FPS profiling
│   ├── v3_*.py                    # Various v3 paper experiments
│   ├── v4_*.py                    # Various v4 paper experiments
│   ├── compare_pose_backends.py   # MediaPipe vs RTMPose A/B
│   ├── activity_confusion.py      # Per-activity error breakdown
│   ├── attention_visualization.py # ST-GCN attention heatmaps
│   ├── classical_baselines.py     # Random Forest / SVM baselines
│   ├── tcn_baseline.py            # Temporal Convolutional Network baseline
│   ├── temperature_scaling.py     # Post-hoc calibration
│   ├── window_size_experiment.py  # Window-size sensitivity sweep
│   ├── dropout_experiment.py      # Dropout sensitivity sweep
│   ├── stgcn_threshold_sweep.py   # ST-GCN threshold sweep
│   ├── optimal_features.py        # Feature-subset comparison
│   ├── statistical_comparison.py  # Wilcoxon / paired tests
│   ├── regenerate_ablation_plot.py
│   ├── event_level_alarm_metrics.py
│   ├── run_optimal12.py           # Optimal-12 model training
│   ├── download_dataset.py        # Convenience download script
│   ├── download_models.py         # Pretrained-model fetcher
│   ├── download_upfall.py         # UP-Fall full bulk download
│   ├── download_upfall_targeted.py # UP-Fall targeted-clip download
│   ├── test_smoothing.py          # ← end-to-end test for OneEuro
│   ├── test_calibration.py        # ← end-to-end test for calibration
│   ├── test_imminent.py           # ← FSM scenarios (no video)
│   ├── test_post_fall.py          # ← motion monitor on a video
│   └── test_all_new.py            # ← one-shot validator
│
├── tests/                         # Unit + integration tests
│   ├── test_alarm.py
│   ├── test_calibration.py
│   ├── test_detection_pose.py
│   ├── test_extractor.py
│   ├── test_features.py
│   ├── test_imminent.py
│   ├── test_integration.py
│   ├── test_loader.py
│   ├── test_model_alarm.py
│   ├── test_motion_monitor.py
│   ├── test_pipeline.py
│   └── test_smoothing.py
│
├── data/data/                     # gitignored
│   ├── raw/                       # URFD, Le2i, UP-Fall original files
│   ├── processed/                 # MediaPipe keypoint sequences per dataset
│   └── splits/                    # LOSO and train/val/test split JSONs
│
├── models/                        # mostly gitignored; download_models.py fetches them
│   ├── pose_landmarker_full.task  # MediaPipe pose model (~9.4 MB)
│   ├── pose_landmarker_heavy.task # MediaPipe pose model (~30.7 MB)
│   ├── pose_landmarker_lite.task  # MediaPipe pose model (~5.8 MB)
│   ├── urfd/                      # Run 3 LOSO BiLSTM
│   └── run5_stride2_aug/          # Run 5 LOSO BiLSTM (best single)
│
├── results/                       # generated, gitignored
│   ├── urfd_loso_v3/              # Run 3 metrics, plots, CSVs
│   ├── run5_stride2_aug/          # Run 5 metrics
│   ├── ablation_study/
│   └── false_alarm_analysis/
│
├── notebooks/                     # Exploration notebooks
│   └── 01_dataset_exploration.ipynb
│
├── logs/                          # gitignored; live demo writes events.db here
│   ├── events.db                  # SQLite event log (WAL mode)
│   └── app.log                    # Rotating JSON log
│
└── samples/                       # Short example clips for the demo
    └── README.md
```

## Conventions

- **Tests live in `tests/`**, one file per `src/` module that has
  non-trivial logic. New modules **must** ship with their own
  `tests/test_<name>.py`.
- **Scripts live in `scripts/`**. They are CLI entry points only —
  any non-trivial logic belongs in `src/`. The `test_*.py` scripts
  under `scripts/` are end-to-end exercisers, not unit tests.
- **Constants live in `config/config.yaml`**, never in code. Magic
  numbers in functions are a bug.
- **Logging via `src/utils/logger.get_logger(__name__)`** —
  never `print()` in shipped modules. Scripts may print for CLI UX.

## What's gitignored

- `data/data/raw/`, `data/data/processed/`, `data/data/splits/`
- `models/` (except scripts and metadata; weights are
  fetched separately)
- `results/`
- `logs/`
- `*.zip` (such as download archives)
- `venv/`, `__pycache__/`
- `.env`
- `notebooks/.ipynb_checkpoints/`

See [`.gitignore`](../.gitignore) for the authoritative list.

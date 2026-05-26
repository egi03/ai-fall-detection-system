# AI Real Time Fall Detection System

![Python](https://img.shields.io/badge/python-3.10+-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Tests](https://img.shields.io/badge/tests-261%20passing-brightgreen)

<p align="center">
  <img src="demo.gif" alt="AI Fall Detection System Demo" width="640">
</p>

A real-time, skeleton-based human fall detection system. It ingests
video from a webcam or file, estimates 2D/3D pose with MediaPipe,
extracts 15 biomechanical features per frame, classifies fall vs.
non-fall with a BiLSTM over a sliding window, and runs a multi-state
alarm FSM with pre-fall warning, debouncing, and post-fall severity
assessment.

---

## Quick start

This branch bundles the Run 5 BiLSTM weights and the MediaPipe pose
task file via **Git LFS** so the demo runs out-of-the-box. Install
Git LFS once (`git lfs install`) before cloning.

```bash
git lfs install
git clone https://github.com/egi03/ai-fall-detection-system.git
cd ai-fall-detection-system
python -m venv venv
# Windows:  venv\Scripts\activate
# Linux/macOS:  source venv/bin/activate
pip install -r requirements.txt
python scripts/demo.py --source samples/fall_sample.mp4 --model models/run5_stride2_aug --demo-mode
```

Press `q` or `ESC` to quit.

> Bundled model: `models/run5_stride2_aug/` (best single-model BiLSTM,
> AUC 0.888). The full ensemble (Run 3 + Run 5, AUC 0.897) and other
> experimental checkpoints are kept on the `main` branch and can be
> rebuilt with `scripts/train_experiment.py`.

---

## Headline results (URFD LOSO, 5-fold)

| Model | Sens (t=0.5) | Spec (t=0.5) | AUC-ROC |
|---|---|---|---|
| BiLSTM Run 5 (stride 2, augment) | 0.806 | 0.842 | 0.888 |
| **Ensemble (Run 3 + Run 5)** | **0.820** | **0.838** | **0.897** |
| Optimal 12-feature BiLSTM | 0.757 | 0.871 | 0.884 |

**Cross-dataset (URFD → UP-Fall, unseen):** 93.9 % overall sensitivity, 100 % on front-view clips.

Full results table including ablations, stride sweep, ensemble bootstrap, and per-camera breakdowns: [docs/EVALUATION.md](docs/EVALUATION.md).

---

## Documentation index

| Document | Contents |
|---|---|
| [docs/SETUP.md](docs/SETUP.md) | Installation, dataset download, preprocessing, pretrained-model fetching. |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | End-to-end pipeline, module responsibilities, design decisions. |
| [docs/FEATURES.md](docs/FEATURES.md) | The 15 biomechanical features, OneEuro smoothing, per-camera calibration. |
| [docs/MODELS.md](docs/MODELS.md) | BiLSTM, ST-GCN, Transformer-LSTM hybrid, ensemble averaging. |
| [docs/ALARM.md](docs/ALARM.md) | FSM (NORMAL → IMMINENT → IMPACT → CONFIRMED → RECOVERED), motion monitor, severity refinement, SQLite event log. |
| [docs/TRAINING.md](docs/TRAINING.md) | Training scripts, LOSO loop, augmentation, reproducibility. |
| [docs/EVALUATION.md](docs/EVALUATION.md) | LOSO, cross-dataset, ablations, every numeric result. |
| [docs/DEMO.md](docs/DEMO.md) | Live demo CLI, controls, demo mode, all flags. |
| [docs/EXPLAINABILITY.md](docs/EXPLAINABILITY.md) | Integrated-Gradients attribution overlay, VLM (Gemini) narration. |
| [docs/CONFIG.md](docs/CONFIG.md) | `config/config.yaml` field-by-field reference. |
| [docs/PROJECT_STRUCTURE.md](docs/PROJECT_STRUCTURE.md) | Directory tree, what each module owns. |
| [docs/CHANGELOG.md](docs/CHANGELOG.md) | Recent features (smoothing, calibration, IMMINENT, severity, test scripts). |

---

## Key capabilities

- **Real-time inference** at ~12 FPS CPU-only (AMD Ryzen 7 7700; LSTM forward pass 1.1 ms).
- **Three pose backends**: MediaPipe (default, 33 landmarks), YOLO-Pose (17 COCO), RTMPose (ONNX, 17 COCO).
- **Two temporal models**: BiLSTM (production) and ST-GCN (skeleton-graph baseline).
- **Ensemble averaging** of two LOSO models for the best published AUC (0.897).
- **Pre-fall warning** (yellow skeleton) before the confirmed alarm.
- **Post-fall severity refinement**: events written to SQLite as `SEVERE` / `MODERATE` / `MINOR` based on stillness analysis.
- **Explainability**: per-joint Integrated-Gradients overlay live on the skeleton.
- **VLM narration** (optional, Gemini): natural-language description of each confirmed fall.
- **Privacy by design**: only skeletons and event metadata are persisted; raw video is never written to disk.

---

## At a glance

```
   ┌──────────┐  ┌──────────────┐  ┌────────────────┐  ┌──────────────┐
   │  Frame   │→ │ Pose         │→ │ FeatureExtractor│→│ BiLSTM /      │
   │  (BGR)   │  │ (MediaPipe / │  │ 15 features /   │  │ ST-GCN /      │
   │          │  │  RTMPose)    │  │ frame           │  │ Ensemble      │
   └──────────┘  └──────────────┘  └────────────────┘  └──────┬───────┘
                                                              │ P(fall)
                                                              ▼
                                                ┌─────────────────────────┐
                                                │  Alarm FSM              │
   ┌──────────────┐                              │  NORMAL → IMMINENT →    │
   │ Skeleton     │ ◄────────── color  ─────────│  IMPACT → CONFIRMED →   │
   │ overlay      │                              │  RECOVERED              │
   └──────────────┘                              └──────────────┬──────────┘
                                                                │
                                                                ▼
                                                ┌─────────────────────────┐
                                                │  EventLogger (SQLite)   │
                                                │  + MotionMonitor →      │
                                                │  severity refinement    │
                                                └─────────────────────────┘
```

---

## Tests

```bash
python -m pytest -q                  # 261 unit + integration tests
python scripts/test_all_new.py       # smoke-test recently added features
```

---

## License

The datasets (URFD, Le2i, UP-Fall) carry their own licenses. Comply with each dataset's terms when downloading.

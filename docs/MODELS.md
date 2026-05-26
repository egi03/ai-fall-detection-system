# Models

Three architectures ship with the project. Only the BiLSTM is wired
into the live demo by default; the others are research baselines /
alternatives.

## 1. BiLSTM (production)

[`src/models/lstm.py`](../src/models/lstm.py) — `FallDetectionLSTM`

```
Input:  (B, T=30, F=15)
   ↓
BiLSTM (hidden=128, num_layers=2, dropout=0.5, bidirectional=True)
   ↓                                            output shape (B, T, 256)
mean-pool over T
   ↓                                            (B, 256)
Dropout(0.5)
   ↓
Linear(256 → 2)
   ↓
log-softmax / argmax
```

Hyperparameters live in `model:` block of [`config/config.yaml`](../config/config.yaml).

### Training recipe

- Optimizer: Adam, lr=1e-3, weight_decay=1e-4
- Scheduler: ReduceLROnPlateau(patience=5, factor=0.5) on val AUC
- Early stopping: patience=30 on val AUC, restore best
- Class weights computed automatically from training labels
- Loss: cross-entropy
- Seed: 42 everywhere

### Pretrained checkpoints

| Path | Run | Notes |
|---|---|---|
| `models/urfd/` | **Run 3** | URFD LOSO, stride=5, no augmentation. AUC 0.878. |
| `models/run5_stride2_aug/` | **Run 5** | URFD LOSO, stride=2, with augmentation. AUC 0.888 — best single. |
| `models/cross_urfd_full_r5/` | Cross-eval | Trained on **all** URFD subjects (no LOSO) with Run 5 hyperparameters. Used for URFD→Le2i / URFD→UP-Fall. |
| `models/cross_le2i_full_r5/` | Cross-eval | Trained on all Le2i, used for Le2i→URFD. |

Each LOSO directory contains `fold_0/best_model.pth` through
`fold_4/best_model.pth` and a `run_meta.json` that records the
training hyperparameters (architecture, stride, augmentation flags,
final metrics). Inference scripts auto-load `fold_0` as the
representative model.

## 2. Ensemble (best AUC)

The ensemble averages fall probabilities from two LOSO models:

```python
P_fall_ensemble = (P_fall_model_A + P_fall_model_B) / 2
```

In the demo this is wired by passing both `--model` and `--model2`:

```bash
python scripts/demo.py --model models/urfd --model2 models/run5_stride2_aug
```

Results on URFD LOSO (5-fold):

| Threshold | Sensitivity | Specificity | AUC |
|---|---|---|---|
| 0.30 (sens-optimal) | **0.901** | 0.755 | 0.897 |
| 0.50 (balanced) | 0.820 | 0.838 | 0.897 |
| 0.54 (spec-optimal) | 0.803 | **0.851** | 0.897 |

Adding a third model (`models/optimal12`) does not improve AUC; the
two-model ensemble is the production choice. Details in
`DAILY LOGS/15.3/ENSEMBLE_ANALYSIS.md`.

## 3. ST-GCN (graph baseline)

[`src/models/stgcn.py`](../src/models/stgcn.py) +
[`src/models/stgcn_classifier.py`](../src/models/stgcn_classifier.py)

A skeleton-graph convolution network that operates directly on
13 anatomically-meaningful joints (a subset of MediaPipe's 33) plus
motion channels (first-difference between consecutive frames).

```
Input:  (B, C=6, T=30, V=13)   # 3D coords + 3D velocity per joint
   ↓
ST-GCN blocks (spatial graph conv ⊕ temporal conv) × N
   ↓
global average pool
   ↓
Linear → 2 classes
```

Use it in the demo:

```bash
python scripts/demo.py --arch stgcn --model models/stgcn_<run_id>
```

ST-GCN matches the BiLSTM ensemble on URFD AUC but is **~4× slower**
on CPU, so it is not the default. The graph baseline is included to
document that the gain from hand-designed features is not free — the
larger end-to-end model can recover similar accuracy, but at a
runtime cost.

## 4. Transformer-LSTM hybrid (alternative)

[`src/models/transformer_lstm.py`](../src/models/transformer_lstm.py)

```
Input → Linear projection → TransformerEncoder (n heads, n layers) → LSTM decoder → Linear
```

Available via `--arch lstm` once the `run_meta.json` declares
`architecture: "transformer_lstm"`. With a 5-subject training set the
hybrid does not beat the plain BiLSTM; it is included for
completeness and as a future experimentation surface.

Hyperparameters in `config/config.yaml` under `transformer_lstm:`.

## Inference wrappers

[`src/models/classifier.py`](../src/models/classifier.py) —
`FallClassifier`. Loads a checkpoint, exposes:

```python
clf = FallClassifier(model_path="models/run5_stride2_aug/fold_0/best_model.pth", ...)
label, prob = clf.predict(sequence)           # sequence shape (30, 15)
probs = clf.predict_proba(sequence)           # [P(no_fall), P(fall)]
```

The demo script picks the wrapper based on `--arch`:

| `--arch` | Wrapper | Input |
|---|---|---|
| `lstm` (default) | `FallClassifier` | `(30, 15)` feature window |
| `stgcn` | `STGCNClassifier` | `(30, 33, 4)` raw keypoint window |

## Choosing a model

| Need | Use |
|---|---|
| Best accuracy | Ensemble: `--model models/urfd --model2 models/run5_stride2_aug` |
| Best single model | `--model models/run5_stride2_aug` |
| Best specificity | Optimal 12-feature model (custom training, see `scripts/run_optimal12.py`) |
| Graph-based, no hand features | `--arch stgcn --model models/stgcn_<run>` |
| Cross-dataset deployment | `models/cross_urfd_full_r5` (trained on all subjects, no LOSO held out) |

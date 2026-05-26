# Evaluation

LOSO baselines, cross-dataset transfers, and ablations. Every number
here is reproducible with `seed: 42` and the scripts in `scripts/`.

## How to reproduce

```bash
# Re-run the headline LOSO numbers
python scripts/evaluate.py --model models/run5_stride2_aug --dataset urfd

# Ensemble bootstrap (Run 3 + Run 5)
python scripts/bootstrap_ci.py

# Cross-dataset transfers
python scripts/cross_dataset_eval.py --train-model models/cross_urfd_full_r5 --test-dataset le2i
python scripts/cross_dataset_eval.py --train-model models/cross_le2i_full_r5 --test-dataset urfd
python scripts/cross_upfall_eval.py  --model models/cross_urfd_full_r5

# Sequence-level (non-overlapping) AUC for honest bias bound
python scripts/v3_sequence_level_auc.py --model models/run5_stride2_aug
```

## Primary metric: AUC-ROC

The single number we lead with is **window-level AUC-ROC** because it
is threshold-independent. Sensitivity and specificity are reported at
three operating points so the trade-off is visible.

## URFD LOSO — all runs

| Run | Hyperparameters | Sens@0.5 | Spec@0.5 | AUC | Valid? |
|---|---|---|---|---|---|
| Run 1 | BiLSTM, stride 5 | 0.683 | 0.735 | 0.716 | ❌ dead features |
| Run 2 | BiLSTM, stride 5 | 0.673 | 0.818 | 0.796 | ❌ wrong normalization order |
| Run 3 | BiLSTM, stride 5 (fixed) | 0.805 | 0.821 | 0.878 | ✅ |
| Run 5 | BiLSTM, stride 2, augment | **0.806** | **0.842** | **0.888** | ✅ best single |
| Run 6 | BiLSTM, stride 2, hidden 64 | 0.810 | 0.787 | 0.875 | ✅ |
| Run 8 | BiLSTM, stride 1, augment | 0.775 | 0.799 | 0.840 | ✅ overfits |
| Run 9 | BiLSTM, stride 2, threshold 0.3 | 0.806 | 0.842 | 0.888 | ✅ no-op threshold variant |
| Run 10 | Optimal 12 features, stride 2 | 0.757 | **0.871** | 0.879 | ✅ best specificity |
| **Ensemble (3 + 5)** | avg probs | **0.820** | 0.838 | **0.897** | ✅ best overall |

Run 4 (combined URFD + Le2i) is reported in the combined-training
section below.

## Three operating points for the best ensemble

| Threshold | Sens | Spec | Use case |
|---|---|---|---|
| **0.30** | **0.901** | 0.755 | Sensitivity-critical: never miss a fall, accept more false alarms. |
| **0.50** | 0.820 | 0.838 | Balanced default. |
| **0.54** | 0.803 | **0.851** | Specificity-critical: keep operators happy, fewer nuisance alarms. |

No single threshold satisfies the literature target of sens ≥ 0.90
**and** spec ≥ 0.85 simultaneously on a 5-subject dataset — the curve
shape is the limit, not the model.

## Bootstrap confidence intervals

[`scripts/bootstrap_ci.py`](../scripts/bootstrap_ci.py) — 10 000 fold
re-samples.

| Comparison | Δ AUC | 95 % CI | p-value |
|---|---|---|---|
| Run 3 vs Run 5 | -0.010 | [-0.025, +0.005] | n.s. |
| Run 5 vs Ensemble | -0.009 | [-0.020, +0.001] | **0.030** |
| Optimal-12 vs Run 5 | -0.009 | [-0.024, +0.006] | n.s. |

The ensemble's improvement over the best single model is small (Δ ≈ 0.009)
but statistically significant.

## Sequence-level (non-overlapping) AUC

[`scripts/v3_sequence_level_auc.py`](../scripts/v3_sequence_level_auc.py)

Window-level AUC slightly over-estimates the deployment metric because
adjacent windows share frames. The non-overlapping variant (stride 30,
i.e. one window per 2 s of video) gives:

| Model | Window-level AUC | Sequence-level AUC (stride 30) | Bias bound |
|---|---|---|---|
| Ensemble | 0.897 | 0.800 | Δ = -0.040 |

This is the honest "AUC ceiling under any deployment policy" number.

## Cross-dataset transfers

### URFD → Le2i

| Threshold | Sens | Spec | AUC |
|---|---|---|---|
| 0.50 | 0.433 | 0.719 | 0.645 |

Large domain gap. Le2i's diverse camera angles confuse a URFD-only
model. Per-room breakdown in `results/cross_dataset/per_room_results.csv`.

### Le2i → URFD

| Threshold | Sens | Spec | AUC |
|---|---|---|---|
| 0.50 | 0.852 | 0.545 | 0.827 |

Much smaller gap. Training on the more diverse dataset (Le2i)
generalizes well to the cleaner one (URFD). **Asymmetric domain shift**
is one of the novel findings of this project.

### URFD → UP-Fall (Zenodo skeleton dataset)

| View | Sens@0.5 | Sens@0.3 | Notes |
|---|---|---|---|
| Both cameras | **93.9 %** | 97.6 % | 77/82 fall sequences detected. |
| cam1 (side) | 81.5 % | 92.6 % | Hardest: A2 "forward fall onto knees" = 28.6 %. |
| cam2 (front) | **100 %** | 100 % | Perfect on every fall type. |

#### Per-fall-type (both cameras)

| Activity | Sens@0.5 | Notes |
|---|---|---|
| A1 forward (hands) | 100 % | |
| A2 forward (knees) | 75 % | Slow descent resembles kneeling. |
| A3 backward | 100 % | |
| A4 sideways | 100 % | |
| A5 sitting empty chair | 100 % | |

UP-Fall has no ADL skeleton data on Zenodo, so we report sensitivity
only — there is no specificity to measure.

Operational takeaway: **camera angle matters more than fall type**.
Front-view deployment is strongly preferred.

## Feature ablation

[`scripts/ablation_study.py`](../scripts/ablation_study.py) — 21
conditions × 5 folds.

### Most-critical features (removing hurts)

| Removed feature | Δ AUC |
|---|---|
| `hip_shoulder_angle` | **-0.025** |
| `left_knee_angle` | -0.021 |
| `mean_visibility` | -0.011 |

### Features that *hurt* the model

| Removed feature | Δ AUC |
|---|---|
| `wrist_hip_distance` | **+0.023** |
| `com_acceleration` | +0.018 |
| `shoulder_ankle_distance` | +0.016 |

Removing all three yields the **optimal-12** model: AUC 0.884, spec
0.871. The default 15-feature set is kept for completeness; the
12-feature variant is the recommended choice when specificity
matters more than sensitivity.

## False-alarm analysis

[`scripts/false_alarm_analysis.py`](../scripts/false_alarm_analysis.py)

Per-subject FP/FN breakdown:

| Subject | ADL FP rate | Fall miss rate | Notes |
|---|---|---|---|
| S01 | 14.2 % | 12.5 % | |
| S02 | 22.1 % | 16.7 % | |
| S03 | **5.6 %** | **4.2 %** | Best. |
| S04 | **31.6 %** | **37.5 %** | Outlier — pose detection struggles. |
| S05 | 18.4 % | 8.3 % | |

17 of 40 ADL sequences trigger zero false alarms (42.5 %). The model
is mildly overconfident in calibration plots — predicted probability
exceeds actual fall rate by 0.09–0.25 in the high-confidence bin.
Temperature scaling brings this within ±0.05.

## Other experiments

| Question | Answer | Source |
|---|---|---|
| Optimal window size? | 30 frames (AUC 0.880). 15 underfits, 50 starts to hurt latency without gain. | `WINDOW_SIZE_SENSITIVITY.md` |
| Optimal stride? | 2 (AUC 0.888). 5 is too sparse, 1 overfits. | Run 5 vs Run 8 |
| Dropout sensitivity? | 0.5 optimal; very flat (spread 0.024 AUC across 0.1–0.7). | `DROPOUT_SENSITIVITY.md` |
| Learning-curve plateau? | At ~50 % of training data. Subject diversity, not data volume, is the bottleneck. | `LEARNING_CURVE.md` |
| Inference speed? | LSTM 1.1 ms; full pipeline ~12.3 FPS on AMD Ryzen 7 7700. | `INFERENCE_BENCHMARK.md` |
| Feature correlation? | 4 clusters; `head_to_toe` and `shoulder_ankle` r = 0.993. | `FEATURE_CORRELATION.md` |
| Activity confusion? | cam0 90 % vs cam1 100 % per-sequence detection. | `ACTIVITY_CONFUSION.md` |

Full per-experiment reports live in `DAILY LOGS/15.3/`.

## Combined training

| Run | Setup | AUC |
|---|---|---|
| Run 5 | URFD only | 0.888 |
| Run 4 | URFD + Le2i (joint training) | 0.808 |

Combined training **loses 0.07 AUC**. The two datasets are different
enough that the model spends capacity reconciling them rather than
learning falls. This is a documented finding — the recommended
practice is to train on the deployment-target dataset alone, or to
use the wider-camera-coverage dataset (Le2i) as the trainer and
deploy on the narrower one (URFD), exploiting the asymmetric domain
shift.

## Failure modes

[`docs/research/10.3 Error Analysis and Common Failure Modes.md`](../research/10.%20ADVANCED%20TOPICS/10.3%20Error%20Analysis%20and%20Common%20Failure%20Modes.md)
+ in-project investigation in `DAILY LOGS/15.3/ERROR_ANALYSIS.md`:

- **Slow descents** (e.g. UP-Fall A2 forward-onto-knees) are mis-classified
  as sit-down activities. ~25 % of A2 falls missed.
- **Subject lies down voluntarily** (e.g. into bed) — without a depth
  cue these can trigger false positives. Mitigated by the stillness
  duration gate.
- **Subject already lying when MediaPipe initializes** — the BlazePose
  detector struggles to find an orientation vector. RTMPose is the
  recommended alternative when this is a deployment concern.
- **Multi-person scenes** — the BiLSTM is single-person; the tracker
  picks the highest-confidence person and only that track is fed
  forward. Per-person FSM is on the roadmap.

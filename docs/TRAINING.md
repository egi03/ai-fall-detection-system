# Training

How to train the models from scratch, and how to reproduce every
published run.

## Prerequisites

- Datasets preprocessed (`python scripts/preprocess.py --dataset urfd`,
  same for `le2i`). See [SETUP.md](SETUP.md).
- `seed: 42` everywhere — guaranteed by `config/config.yaml`.

## Run 5 — best single model (recommended baseline)

```bash
python scripts/train_experiment.py \
    --dataset urfd \
    --stride 2 \
    --augment \
    --output-dir models/run5_stride2_aug
```

Trains five LOSO folds. Each fold logs to
`results/run5_stride2_aug/fold_<i>/` and writes its best checkpoint
to `models/run5_stride2_aug/fold_<i>/best_model.pth`. A `run_meta.json`
in the parent directory records every hyperparameter so the inference
wrappers can auto-detect the architecture.

Expected result: AUC ≈ 0.888 (5-fold mean).

## Run 3 — second member of the production ensemble

```bash
python scripts/train.py
```

Trains five LOSO folds with the project's "classic" hyperparameters
(stride=5, no augmentation). Output: `models/urfd/`.

Expected result: AUC ≈ 0.878.

## Ensemble at inference

There is no separate "train the ensemble" step. The ensemble is built
purely at inference by averaging probabilities from two LOSO models.
After the two training runs above, the demo and evaluation scripts
accept both checkpoints:

```bash
python scripts/demo.py     --model models/urfd --model2 models/run5_stride2_aug
python scripts/bootstrap_ci.py                # writes ensemble metrics to results/
```

## Full-dataset training (cross-dataset eval)

For URFD→Le2i and similar transfers, we need a model trained on every
URFD subject — no held-out fold:

```bash
python scripts/train_experiment.py \
    --dataset urfd --stride 2 --augment --no-loso \
    --output-dir models/cross_urfd_full_r5

python scripts/train_experiment.py \
    --dataset le2i --stride 2 --augment --no-loso \
    --output-dir models/cross_le2i_full_r5
```

These are evaluated via `scripts/cross_dataset_eval.py` and
`scripts/cross_upfall_eval.py`.

## Combined URFD + Le2i training

```bash
python scripts/train_combined.py
```

> **Warning**: combined training degrades AUC by ~0.07 due to domain
> shift between the two datasets. This is a documented finding, not a
> bug — see [EVALUATION.md](EVALUATION.md#combined-training).

## ST-GCN graph baseline

```bash
python scripts/train_stgcn.py --output-dir models/stgcn_run1
```

Threshold sweeping for ST-GCN:

```bash
python scripts/stgcn_threshold_sweep.py --model models/stgcn_run1
```

Results land in `results/stgcn_run1/`.

## Hyperparameter experiments

Every experiment script under `scripts/` is named after what it
varies:

| Script | Varies |
|---|---|
| `ablation_study.py` | Removes feature groups one at a time (21 conditions × 5 folds). |
| `window_size_experiment.py` | Window sizes 15 / 20 / 30 / 40 / 50 frames. |
| `dropout_experiment.py` | Dropout 0.1 → 0.7. |
| `temperature_scaling.py` | Post-hoc calibration. |
| `optimal_features.py` | Compares feature subsets including the 12-feature best. |
| `v3_fsm_sensitivity.py` | 75 alarm-FSM parameter combinations. |
| `v3_fold_bootstrap.py` | Bootstrap CI over fold metrics. |
| `bootstrap_ci.py` | Statistical-significance test ensemble vs Run 5. |
| `classical_baselines.py` | Random Forest / SVM baselines on the same features. |
| `tcn_baseline.py` | Temporal Convolutional Network baseline. |

All write CSV / JSON / PNG outputs under `results/<experiment>/`.

## Training loop internals

[`src/training/trainer.py`](../src/training/trainer.py) — `Trainer` class.

- Optimizer: Adam, lr 1e-3, weight_decay 1e-4
- Scheduler: ReduceLROnPlateau (patience=5, factor=0.5) on val AUC
- Early stopping: patience=30 on val AUC; restore best weights
- Class weights: auto-computed from training labels
- Loss: cross-entropy
- Checkpoint policy: save best by validation AUC
- Logging: every epoch dumps train/val loss and metrics to
  `results/<run>/fold_<i>/history.json`

The training curves PNGs are emitted by `src/training/evaluate.py`'s
plotting helpers (called at the end of each fold).

## Reproducibility checklist

1. Identical `requirements.txt` lockfile.
2. `seed: 42` set in NumPy, PyTorch, Python's `random`.
3. CPU deterministic algorithms (`torch.use_deterministic_algorithms(True)`).
4. Subject-independent LOSO splits regenerated from `splits.json`.
5. Identical preprocessing pipeline (always re-run `scripts/preprocess.py`).
6. Same dataset versions (URFD 2014 release, Le2i 2013 release, UP-Fall 2019).

Re-running `python scripts/train.py` and `python scripts/train_experiment.py
--dataset urfd --stride 2 --augment --output-dir models/run5_stride2_aug`
on a fresh machine should reproduce AUC 0.878 and AUC 0.888
respectively within ±0.005 (driven mostly by non-determinism in
MediaPipe pose inference, which we cannot seed).

## Where things land

| Output | Path |
|---|---|
| Model checkpoints | `models/<run>/fold_<i>/best_model.pth` |
| Run metadata | `models/<run>/run_meta.json` |
| Training history | `results/<run>/fold_<i>/history.json` |
| Confusion / ROC plots | `results/<run>/fold_<i>/*.png` |
| Per-subject CSV | `results/<run>/per_subject_results.csv` |
| Bootstrap CIs | `results/<run>/bootstrap_ci.json` |

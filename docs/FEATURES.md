# Features

The 15-feature vector fed to the temporal model, plus the optional
pre- and post-processing that wraps it.

## The 15 features

Implemented in [`src/features/extractor.py`](../src/features/extractor.py).
All features are computed per frame and are scale-invariant once
person-centric normalization is applied upstream.

| Index | Name | Group | What it measures |
|---|---|---|---|
| 0 | `torso_inclination` | Geometric | Angle of the shoulder→hip vector from vertical, in degrees [0, 90]. >60° suggests a fall. |
| 1 | `hip_shoulder_angle` | Geometric | Angle between hip-line and shoulder-line. Becomes non-trivial when the body twists during a fall. |
| 2 | `bbox_aspect_ratio` | Geometric | Width / height of the keypoint bounding box. Rises sharply when lying down. |
| 3 | `com_velocity` | Kinematic | Vertical velocity of the centre of mass (hips midpoint), in normalized units/sec. |
| 4 | `com_acceleration` | Kinematic | Vertical acceleration of the COM. Peaks at impact. |
| 5 | `head_to_toe_distance` | Distance | Euclidean distance between nose and ankle midpoint. Shrinks as the body collapses. |
| 6 | `shoulder_ankle_distance` | Distance | Like (5) but using shoulder midpoint, more robust when the head exits the frame. |
| 7 | `left_knee_angle` | Angular | Angle at the left knee, in degrees. |
| 8 | `right_knee_angle` | Angular | Symmetric to (7). |
| 9 | `left_hip_angle` | Angular | Angle at the left hip. |
| 10 | `right_hip_angle` | Angular | Symmetric to (9). |
| 11 | `wrist_hip_distance` | Reflexive | Mean distance from wrists to hip midpoint. Spikes during a "catch yourself" reflex. |
| 12 | `body_spread` | Reflexive | Horizontal extent of the keypoint cloud, normalized by height. |
| 13 | `mean_visibility` | Confidence | Mean MediaPipe visibility across all 33 landmarks. |
| 14 | `min_core_visibility` | Confidence | Minimum visibility over the torso/limb joints. |

The feature constant is exposed as `FeatureExtractor.NUM_FEATURES = 15`
so it can never drift out of sync with the model input.

### Ablation findings

Documented in `DAILY LOGS/15.3/ABLATION_STUDY.md`. Headlines:

- **Most critical**: `hip_shoulder_angle` (-0.025 AUC if removed), `left_knee_angle` (-0.021), `mean_visibility` (-0.011).
- **Three features hurt**: removing `wrist_hip_distance` (+0.023), `com_acceleration` (+0.018) and `shoulder_ankle_distance` (+0.016) **improves** AUC to 0.884 and specificity to 0.871. These three add noise rather than signal at this dataset scale. They are kept in the default 15-feature set for completeness; the optimal-12 model is shipped as an alternative.

## Missing keypoint handling

The extractor never crashes on missing or low-confidence keypoints. It
returns `NaN` for any feature whose required joints fall below
`confidence_threshold`. In the live pipeline, `np.nan_to_num(features,
nan=0.0)` neutralizes those slots before they reach the LSTM; the
visibility features (13, 14) themselves carry the "we don't trust this
frame" signal forward.

## OneEuro keypoint smoothing (optional)

Implemented in [`src/pose/smoothing.py`](../src/pose/smoothing.py).

The OneEuro filter (Casiez et al., 2012) is an adaptive low-pass:

```
cutoff = min_cutoff + beta * |derivative|
```

High derivative (fast motion → impact) → higher cutoff → less smoothing.
Low derivative (idle jitter) → lower cutoff → more smoothing.

Applied independently to `(x, y)` of every keypoint; z and visibility
are passed through unchanged. The smoother is a deterministic,
dependency-free NumPy implementation with no learnable parameters.

### Config

```yaml
pose:
  smoothing:
    enabled: false               # default off
    min_cutoff: 1.0              # Hz at zero velocity
    beta: 0.007                  # speed coefficient
    d_cutoff: 1.0                # derivative low-pass cutoff
```

### When to enable

- Webcam streams where MediaPipe jitter is visible.
- Phone footage at variable frame rate.
- Any time the live yellow IMMINENT signal is flickering during idle scenes.

### When to leave disabled

- Offline benchmark runs (we want to compare apples-to-apples with the
  trained distribution).
- High-quality, well-lit footage where jitter is already low.

## Per-camera baseline calibration (optional)

Implemented in [`src/features/calibration.py`](../src/features/calibration.py).

The calibrator buffers the COM velocity and acceleration features
during a warmup period, then computes their per-camera mean and
standard deviation. At inference, every value within `k · σ` of the
baseline mean is "soft-thresholded" toward the mean; values outside
the band are monotonically shifted but preserved in sign and direction.

The effect is to suppress the camera-specific idle noise floor on the
two scale-sensitive kinematic features, without distorting the larger
spikes that correspond to actual falls.

### Config

```yaml
features:
  calibration:
    enabled: false
    warmup_seconds: 3.0
    indices: [3, 4]              # com_velocity, com_acceleration
    k: 1.5
    min_std: 0.0005
```

The HUD shows `CALIBRATING N%` during warmup so the operator knows the
calibrator isn't ready yet.

### Soft-threshold formula

For each calibrated index, given baseline mean `μ` and std `σ`:

```
band = k * σ
dev  = x - μ
if |dev| <= band:
    output = μ                            # fully suppress
else:
    output = μ + sign(dev) * (|dev| - band)   # shift toward the band edge
```

A value of `k = 0` reduces the calibrator to a strict pass-through.

## Feature normalization

Person-centric normalization is the default (`features.normalize: true`).
The body bounding box is rescaled so the keypoint cloud spans a unit
square; this removes camera-distance and person-size effects.

> **Important**: an earlier iteration applied normalization *after*
> computing velocities, which destroyed the kinematic signal. Run 3
> onwards normalizes the keypoints first, then derives velocity/accel
> from the normalized coordinates. Tests in `tests/test_extractor.py`
> guard against this regression.

## Sliding window

`src/features/window.py` maintains a fixed-size rolling buffer of the
last `window_size` feature vectors. The classifier is fed
`window.get_array()` once `window.is_ready` returns True; that array
has shape `(window_size, num_features)`.

The training-time stride (`features.window_stride`) does not apply at
inference — every frame produces one new probability.

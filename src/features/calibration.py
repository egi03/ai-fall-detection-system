"""
Per-camera baseline calibration for velocity / acceleration features.

During the first N frames of a stream, samples of selected features are
collected (typically COM vertical velocity and acceleration). After
warmup, the per-feature baseline mean and standard deviation define a
soft-threshold band: any feature deviation smaller than ``k * std``
around the baseline mean is suppressed at inference.

The intent is to neutralize camera-specific jitter (different frame
rates, person scale, sensor noise) without altering the overall scale
or polarity of the features the model expects. A real fall produces
velocity/accel spikes well outside the baseline band, so it is
unaffected; idle jitter from MediaPipe sub-pixel wobble is removed.

Disabled by default; enabled via ``features.calibration.enabled`` in
config.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np


class BaselineCalibrator:
    """Soft-threshold denoising of designated features around a baseline.

    The calibrator buffers samples of the target features during a
    warmup period, then computes per-feature ``mean`` and ``std`` once
    enough samples are available. At inference, each target feature is
    pulled toward the baseline mean if its deviation falls within
    ``k * std``; values outside the band are kept but shifted toward
    the band edge ("soft thresholding").

    Parameters
    ----------
    feature_indices : sequence of int
        Indices into the per-frame feature vector to calibrate.
        For the project's 15-feature layout, indices 3 and 4 correspond
        to COM vertical velocity and acceleration — the two scale-sensitive
        kinematic features.
    warmup_frames : int
        Number of samples to collect before the calibrator becomes
        active. Should correspond to a few seconds of video at the
        target FPS (e.g. 45 frames at 15 FPS = 3 s).
    k : float
        Width of the suppression band in units of baseline std. Larger
        ``k`` removes more idle motion but risks suppressing slow falls.
    min_std : float
        Minimum baseline std clamp. Avoids huge effective bands when the
        camera is perfectly still during warmup.

    Notes
    -----
    The calibrator never introduces new signal: an input within the
    band is mapped to the baseline mean, an input outside the band is
    monotonically mapped to the same side of the band. This is a strict
    no-op in the limit ``k = 0``.
    """

    def __init__(
        self,
        feature_indices: Sequence[int],
        warmup_frames: int,
        k: float = 1.5,
        min_std: float = 1e-4,
    ) -> None:
        if warmup_frames <= 1:
            raise ValueError(
                f"warmup_frames must be > 1, got {warmup_frames}"
            )
        if k < 0:
            raise ValueError(f"k must be >= 0, got {k}")

        self._indices = tuple(int(i) for i in feature_indices)
        if len(self._indices) == 0:
            raise ValueError("feature_indices must be non-empty")
        self._warmup = int(warmup_frames)
        self._k = float(k)
        self._min_std = float(min_std)
        self._samples: list[np.ndarray] = []
        self._baseline_mean: Optional[np.ndarray] = None
        self._baseline_std: Optional[np.ndarray] = None
        self._ready = False

    @property
    def is_ready(self) -> bool:
        """True once warmup is complete and the baseline is fitted."""
        return self._ready

    @property
    def progress(self) -> float:
        """Warmup progress in [0, 1]."""
        if self._ready:
            return 1.0
        return min(1.0, len(self._samples) / self._warmup)

    @property
    def baseline_mean(self) -> Optional[np.ndarray]:
        """Fitted baseline means (one value per calibrated feature) or None."""
        return self._baseline_mean

    @property
    def baseline_std(self) -> Optional[np.ndarray]:
        """Fitted baseline stds (one value per calibrated feature) or None."""
        return self._baseline_std

    def observe(self, features: np.ndarray) -> None:
        """Buffer one frame of features during warmup.

        After the buffer reaches ``warmup_frames`` samples, the baseline
        mean / std are computed and the calibrator becomes active.
        Calls after warmup is complete are no-ops.

        Parameters
        ----------
        features : np.ndarray
            1D array of features for the current frame. Must contain
            valid (non-NaN) values at every index in ``feature_indices``.
            Frames with NaN at any calibrated index are silently skipped.
        """
        if self._ready:
            return
        if features.ndim != 1:
            raise ValueError(
                f"features must be 1D, got shape {features.shape}"
            )

        try:
            sample = features[list(self._indices)].astype(np.float64).copy()
        except IndexError as exc:
            raise ValueError(
                f"feature indices {self._indices} out of bounds "
                f"for vector of length {len(features)}"
            ) from exc

        if not np.all(np.isfinite(sample)):
            return

        self._samples.append(sample)
        if len(self._samples) >= self._warmup:
            arr = np.stack(self._samples, axis=0)
            self._baseline_mean = arr.mean(axis=0)
            std = arr.std(axis=0)
            self._baseline_std = np.maximum(std, self._min_std)
            self._ready = True

    def apply(self, features: np.ndarray) -> np.ndarray:
        """Soft-threshold the calibrated features around the baseline.

        Returns a copy with the targeted indices denoised. Returns the
        input unchanged if warmup is incomplete.

        Parameters
        ----------
        features : np.ndarray
            1D feature vector for the current frame.

        Returns
        -------
        np.ndarray
            Soft-thresholded copy of the feature vector.
        """
        if features.ndim != 1:
            raise ValueError(
                f"features must be 1D, got shape {features.shape}"
            )
        if not self._ready:
            return features

        assert self._baseline_mean is not None
        assert self._baseline_std is not None

        out = features.copy()
        for i, idx in enumerate(self._indices):
            x = float(features[idx])
            if not np.isfinite(x):
                continue
            mean = float(self._baseline_mean[i])
            band = self._k * float(self._baseline_std[i])
            dev = x - mean
            if abs(dev) <= band:
                out[idx] = mean
            else:
                out[idx] = mean + np.sign(dev) * (abs(dev) - band)
        return out

    def reset(self) -> None:
        """Clear all buffered samples and the fitted baseline."""
        self._samples.clear()
        self._baseline_mean = None
        self._baseline_std = None
        self._ready = False

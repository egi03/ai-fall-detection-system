"""Integrated-Gradients attribution for the fall-detection classifier.

Computes per-feature contributions to P(fall=1) by approximating the path
integral of gradients between a zero baseline and the actual input
sequence. The 15 hand-crafted features are z-score normalized upstream,
so the zero baseline corresponds to the dataset mean — a meaningful
"average behavior" reference for attribution.

References
----------
Sundararajan, Taly & Yan (2017). "Axiomatic Attribution for Deep Networks."
ICML 2017. https://arxiv.org/abs/1703.01365
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class AttributionResult:
    """Attribution output for a single classified window.

    Attributes
    ----------
    fall_probability : float
        Softmax probability of the fall class.
    feature_attribution : np.ndarray
        Shape ``(num_features,)`` — signed contribution of each feature to
        the *fall* class logit, summed over the temporal window. Positive
        values push toward a fall prediction; negative values away from it.
    per_frame_attribution : np.ndarray
        Shape ``(window_size, num_features)`` — the raw per-frame, per-feature
        attribution before temporal aggregation. Useful for debugging.
    """

    fall_probability: float
    feature_attribution: np.ndarray
    per_frame_attribution: np.ndarray


class IntegratedGradients:
    """Integrated-Gradients explainer for a PyTorch sequence classifier.

    Wraps a fall-detection classifier (LSTM, Transformer-LSTM, or any
    other ``nn.Module`` returning class logits) and produces feature-level
    attributions for a single window via the IG approximation:

    ``IG_i = (x_i - x'_i) * (1/n) * sum_{k=1..n} d softmax(F(x' + k/n (x - x')))/d x_i``

    with baseline ``x' = 0`` and ``n = n_steps`` Riemann steps.

    Parameters
    ----------
    model : torch.nn.Module
        The trained classifier. Must accept tensors of shape
        ``(B, window_size, num_features)`` and return class logits.
    target_class : int
        Class index whose probability is being explained (1 = fall).
    n_steps : int
        Number of Riemann steps in the path integral. 8 is a good
        cost/accuracy trade-off for real-time inference on CPU.
    device : str
        Torch device on which to run attribution.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        target_class: int = 1,
        n_steps: int = 8,
        device: str = "cpu",
    ) -> None:
        if n_steps < 2:
            raise ValueError("n_steps must be at least 2")

        self._model = model
        self._target = target_class
        self._n_steps = n_steps
        self._device = torch.device(device)

    def explain(
        self,
        sequence: np.ndarray,
        baseline: Optional[np.ndarray] = None,
    ) -> AttributionResult:
        """Compute Integrated-Gradients attribution for one window.

        Parameters
        ----------
        sequence : np.ndarray
            Feature window of shape ``(window_size, num_features)`` or
            ``(1, window_size, num_features)``. Must be in the same
            normalized space the model was trained on.
        baseline : np.ndarray, optional
            Reference input. Defaults to zeros, which under z-score
            normalization corresponds to the per-feature training mean.

        Returns
        -------
        AttributionResult
            Probability and per-feature contributions.
        """
        if sequence.ndim == 2:
            sequence = sequence[np.newaxis, :]
        if sequence.shape[0] != 1:
            raise ValueError(
                "IntegratedGradients.explain() expects a single window; "
                f"got batch size {sequence.shape[0]}."
            )

        x = torch.from_numpy(sequence.astype(np.float32)).to(self._device)
        if baseline is None:
            x_prime = torch.zeros_like(x)
        else:
            if baseline.ndim == 2:
                baseline = baseline[np.newaxis, :]
            x_prime = torch.from_numpy(baseline.astype(np.float32)).to(self._device)

        was_training = self._model.training
        self._model.eval()

        # Build interpolated batch: shape (n_steps, T, F)
        alphas = torch.linspace(
            1.0 / self._n_steps, 1.0, steps=self._n_steps, device=self._device,
        ).view(-1, 1, 1)
        interpolated = x_prime + alphas * (x - x_prime)  # (n_steps, T, F)
        interpolated.requires_grad_(True)

        logits = self._model(interpolated)  # (n_steps, num_classes)
        probs = F.softmax(logits, dim=1)
        target_probs = probs[:, self._target]
        # Sum so each interpolated input gets its own gradient
        grads = torch.autograd.grad(
            outputs=target_probs.sum(), inputs=interpolated,
        )[0]  # (n_steps, T, F)

        avg_grads = grads.mean(dim=0)  # (T, F)
        ig = (x[0] - x_prime[0]) * avg_grads  # (T, F)

        per_frame = ig.detach().cpu().numpy().astype(np.float32)
        per_feature = per_frame.sum(axis=0)  # (F,)

        # Compute the model probability at the actual input
        with torch.no_grad():
            actual_logits = self._model(x)
            actual_probs = F.softmax(actual_logits, dim=1)
            fall_prob = float(actual_probs[0, self._target].item())

        if was_training:
            self._model.train()

        return AttributionResult(
            fall_probability=fall_prob,
            feature_attribution=per_feature,
            per_frame_attribution=per_frame,
        )

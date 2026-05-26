"""Explainability for fall detection models.

Provides Integrated-Gradients attribution over the 15-feature LSTM input,
plus a mapping from each feature to the MediaPipe joints it consumes so
attributions can be rendered as a per-joint heatmap on the live skeleton.
"""

from src.explainability.attribution import IntegratedGradients, AttributionResult
from src.explainability.joint_mapping import (
    FEATURE_NAMES,
    FEATURE_JOINTS,
    joints_from_feature_attribution,
)

__all__ = [
    "IntegratedGradients",
    "AttributionResult",
    "FEATURE_NAMES",
    "FEATURE_JOINTS",
    "joints_from_feature_attribution",
]

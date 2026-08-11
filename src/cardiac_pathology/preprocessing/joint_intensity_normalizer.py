"""Joint ED/ES percentile clipping and z-score normalization."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from cardiac_pathology.preprocessing.preprocessing_config import PreprocessingConfig


@dataclass(frozen=True, slots=True)
class NormalizationMetadata:
    """Joint normalization metadata."""

    clip_lower: float
    clip_upper: float
    mean: float
    std: float


class JointIntensityNormalizer:
    """Normalize ED and ES jointly over valid pre-padding voxels."""

    def __init__(self, config: PreprocessingConfig) -> None:
        self.config = config

    def normalize(
        self,
        ed_array: NDArray[np.floating],
        es_array: NDArray[np.floating],
    ) -> tuple[NDArray[np.float32], NDArray[np.float32], NormalizationMetadata]:
        """Clip and z-score ED/ES with one shared set of statistics."""
        joint_values = np.concatenate(
            [ed_array.astype(np.float64).ravel(), es_array.astype(np.float64).ravel()]
        )
        clip_lower = float(np.percentile(joint_values, self.config.lower_percentile))
        clip_upper = float(np.percentile(joint_values, self.config.upper_percentile))
        ed_clipped = np.clip(ed_array.astype(np.float64), clip_lower, clip_upper)
        es_clipped = np.clip(es_array.astype(np.float64), clip_lower, clip_upper)
        clipped_joint = np.concatenate([ed_clipped.ravel(), es_clipped.ravel()])
        mean = float(clipped_joint.mean())
        std = float(clipped_joint.std())
        if std <= self.config.epsilon:
            raise ValueError(f"Normalization std <= epsilon: {std} <= {self.config.epsilon}")

        denominator = max(std, self.config.epsilon)
        ed_normalized = ((ed_clipped - mean) / denominator).astype(np.float32)
        es_normalized = ((es_clipped - mean) / denominator).astype(np.float32)
        return (
            ed_normalized,
            es_normalized,
            NormalizationMetadata(
                clip_lower=clip_lower,
                clip_upper=clip_upper,
                mean=mean,
                std=std,
            ),
        )

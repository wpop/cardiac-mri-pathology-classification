"""Preprocessing result contract."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class PreprocessingResult:
    """Final preprocessed patient tensor and deterministic metadata."""

    tensor: NDArray[np.float32]
    patient_id: str
    source_shape_xyz: tuple[int, int, int]
    source_spacing_xyz: tuple[float, float, float]
    resampled_shape_xyz: tuple[int, int, int]
    target_spacing_xyz: tuple[float, float, float]
    crop_start_xy: tuple[int, int]
    crop_end_xy: tuple[int, int]
    valid_depth: int
    z_padding_lower: int
    z_padding_upper: int
    clip_lower: float
    clip_upper: float
    normalization_mean: float
    normalization_std: float

    def __post_init__(self) -> None:
        """Validate final tensor invariants."""
        if self.tensor.shape != (2, 14, 144, 144):
            raise ValueError(f"Unexpected tensor shape: {self.tensor.shape}")
        if self.tensor.dtype != np.float32:
            raise ValueError(f"Unexpected tensor dtype: {self.tensor.dtype}")
        if not self.tensor.flags.c_contiguous:
            raise ValueError("Preprocessed tensor must be C-contiguous")

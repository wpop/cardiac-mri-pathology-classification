"""Post-normalization center Z padding."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class ZPadding:
    """Z padding metadata."""

    lower: int
    upper: int
    total: int


class ZPadder:
    """Center-pad XYZ arrays in Z to the frozen target depth."""

    def __init__(self, target_depth: int, padding_value: float) -> None:
        if target_depth <= 0:
            raise ValueError("target_depth must be positive")
        self.target_depth = target_depth
        self.padding_value = padding_value

    def pad(self, array_xyz: NDArray[np.float32]) -> tuple[NDArray[np.float32], ZPadding]:
        """Center-pad Z after normalization."""
        depth = array_xyz.shape[2]
        if depth > self.target_depth:
            raise ValueError(f"Z cropping would be required: depth={depth}")
        total = self.target_depth - depth
        lower = total // 2
        upper = total - lower
        padded = np.pad(
            array_xyz,
            ((0, 0), (0, 0), (lower, upper)),
            mode="constant",
            constant_values=self.padding_value,
        ).astype(np.float32, copy=False)
        return padded, ZPadding(lower=lower, upper=upper, total=total)

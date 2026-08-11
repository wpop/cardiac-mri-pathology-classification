"""Physical destination grid for deterministic resampling."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class PhysicalGrid:
    """Target voxel grid geometry in source XYZ axis order."""

    shape_xyz: tuple[int, int, int]
    affine: NDArray[np.float64]
    spacing_xyz: tuple[float, float, float]

    def __post_init__(self) -> None:
        """Validate grid geometry."""
        if len(self.shape_xyz) != 3 or any(value < 1 for value in self.shape_xyz):
            raise ValueError("shape_xyz must contain three positive sizes")
        if len(self.spacing_xyz) != 3 or any(value <= 0 for value in self.spacing_xyz):
            raise ValueError("spacing_xyz must contain three positive spacings")
        if self.affine.shape != (4, 4):
            raise ValueError("affine must have shape (4, 4)")

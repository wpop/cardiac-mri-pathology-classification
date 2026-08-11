"""Physical resampling onto the frozen shared ED-derived grid."""

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import map_coordinates  # type: ignore[import-untyped]

from cardiac_pathology.preprocessing.orientation_normalizer import OrientedImage
from cardiac_pathology.preprocessing.physical_grid import PhysicalGrid
from cardiac_pathology.preprocessing.preprocessing_config import PreprocessingConfig


class PhysicalResampler:
    """Derive one center-preserving target grid and linearly resample volumes."""

    def __init__(self, config: PreprocessingConfig) -> None:
        self.config = config

    def derive_grid(self, reference_image: OrientedImage) -> PhysicalGrid:
        """Derive the shared target grid from the oriented ED image."""
        source_shape = reference_image.array.shape[:3]
        source_spacing = reference_image.spacing_xyz
        target_spacing = self.config.target_spacing_xyz
        target_shape = (
            self._target_size(source_shape[0], source_spacing[0], target_spacing[0]),
            self._target_size(source_shape[1], source_spacing[1], target_spacing[1]),
            self._target_size(source_shape[2], source_spacing[2], target_spacing[2]),
        )
        direction = np.asarray(reference_image.affine[:3, :3], dtype=np.float64).copy()
        for axis, spacing in enumerate(self._spacing_from_affine(reference_image.affine)):
            direction[:, axis] /= spacing
        target_matrix = direction * np.asarray(target_spacing, dtype=np.float64)
        source_center = (np.asarray(source_shape, dtype=np.float64) - 1.0) / 2.0
        target_center = (np.asarray(target_shape, dtype=np.float64) - 1.0) / 2.0
        source_center_physical = (
            reference_image.affine[:3, :3] @ source_center + reference_image.affine[:3, 3]
        )

        affine = np.eye(4, dtype=np.float64)
        affine[:3, :3] = target_matrix
        affine[:3, 3] = source_center_physical - target_matrix @ target_center
        return PhysicalGrid(
            shape_xyz=target_shape,
            affine=affine,
            spacing_xyz=target_spacing,
        )

    def resample(self, image: OrientedImage, grid: PhysicalGrid) -> NDArray[np.float64]:
        """Linearly resample one oriented image to the shared target grid.

        The coordinate convention reproduces Phase 1: first-to-last voxel-center
        physical span is preserved as closely as possible. Sub-voxel edge samples
        use deterministic nearest-boundary extension and do not create artificial
        pre-normalization padding.
        """
        source_spacing = image.spacing_xyz
        coordinate_axes = []
        for axis, target_size in enumerate(grid.shape_xyz):
            source_center = (image.array.shape[axis] - 1) / 2.0
            target_center = (target_size - 1) / 2.0
            spacing_ratio = grid.spacing_xyz[axis] / source_spacing[axis]
            coordinates = (
                np.arange(target_size, dtype=np.float64) - target_center
            ) * spacing_ratio + source_center
            coordinate_axes.append(coordinates)

        coordinate_grid = np.meshgrid(*coordinate_axes, indexing="ij")
        resampled = map_coordinates(
            image.array.astype(np.float64),
            coordinate_grid,
            order=1,
            mode="nearest",
            prefilter=False,
        )
        return np.asarray(resampled, dtype=np.float64)

    def _spacing_from_affine(self, affine: NDArray[np.float64]) -> tuple[float, float, float]:
        """Calculate voxel spacing from affine column norms."""
        return (
            float(np.linalg.norm(affine[:3, 0])),
            float(np.linalg.norm(affine[:3, 1])),
            float(np.linalg.norm(affine[:3, 2])),
        )

    def _target_size(
        self,
        source_size: int,
        source_spacing: float,
        target_spacing: float,
    ) -> int:
        """Calculate target size from first-to-last voxel-center span."""
        return max(
            1,
            int(round(((source_size - 1) * source_spacing) / target_spacing)) + 1,
        )

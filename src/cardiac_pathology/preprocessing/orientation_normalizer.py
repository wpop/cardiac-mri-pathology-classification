"""Explicit orientation normalization to LPS."""

from dataclasses import dataclass
from typing import cast

import nibabel as nib
import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class OrientedImage:
    """Array and affine after orientation-only normalization."""

    array: NDArray[np.floating]
    affine: NDArray[np.float64]
    spacing_xyz: tuple[float, float, float]
    orientation: tuple[str, str, str]


class OrientationNormalizer:
    """Apply permutation/flipping transforms to reach the target orientation."""

    def __init__(self, target_orientation: str) -> None:
        if target_orientation != "LPS":
            raise ValueError("Only the frozen LPS target orientation is supported")
        self.target_orientation = target_orientation

    def normalize(self, image: nib.spatialimages.SpatialImage) -> OrientedImage:
        """Return image data reoriented to LPS without interpolation."""
        source_orientation = nib.orientations.io_orientation(image.affine)  # type: ignore[no-untyped-call]
        target_orientation = nib.orientations.axcodes2ornt(  # type: ignore[no-untyped-call]
            tuple(self.target_orientation)
        )
        transform = nib.orientations.ornt_transform(  # type: ignore[no-untyped-call]
            source_orientation,
            target_orientation,
        )
        source_array = np.asanyarray(image.dataobj)
        oriented_array = nib.orientations.apply_orientation(  # type: ignore[no-untyped-call]
            source_array,
            transform,
        )
        affine_transform = nib.orientations.inv_ornt_aff(  # type: ignore[no-untyped-call]
            transform,
            source_array.shape[:3],
        )
        oriented_affine = np.asarray(image.affine @ affine_transform, dtype=np.float64)
        orientation_codes = cast(
            tuple[str, str, str],
            nib.aff2axcodes(oriented_affine),  # type: ignore[no-untyped-call]
        )
        source_spacing = tuple(float(value) for value in image.header.get_zooms()[:3])
        transformed_axes = np.argsort(transform[:, 0].astype(int))
        oriented_spacing = (
            float(source_spacing[transformed_axes[0]]),
            float(source_spacing[transformed_axes[1]]),
            float(source_spacing[transformed_axes[2]]),
        )

        if orientation_codes != tuple(self.target_orientation):
            raise ValueError(
                f"Orientation normalization failed: {orientation_codes} != "
                f"{tuple(self.target_orientation)}"
            )

        return OrientedImage(
            array=np.asarray(oriented_array),
            affine=oriented_affine,
            spacing_xyz=oriented_spacing,
            orientation=orientation_codes,
        )

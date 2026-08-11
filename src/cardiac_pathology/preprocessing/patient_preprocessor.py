"""Patient-level deterministic preprocessing pipeline."""

from typing import cast

import nibabel as nib
import numpy as np

from cardiac_pathology.data import AcdcPatient
from cardiac_pathology.preprocessing.center_cropper import CenterCropper
from cardiac_pathology.preprocessing.joint_intensity_normalizer import JointIntensityNormalizer
from cardiac_pathology.preprocessing.orientation_normalizer import (
    OrientationNormalizer,
    OrientedImage,
)
from cardiac_pathology.preprocessing.physical_resampler import PhysicalResampler
from cardiac_pathology.preprocessing.preprocessing_config import PreprocessingConfig
from cardiac_pathology.preprocessing.preprocessing_result import PreprocessingResult
from cardiac_pathology.preprocessing.z_padder import ZPadder


class PatientPreprocessor:
    """Compose the frozen deterministic patient preprocessing pipeline."""

    def __init__(self, config: PreprocessingConfig) -> None:
        self.config = config
        self.orientation_normalizer = OrientationNormalizer(config.target_orientation)
        self.resampler = PhysicalResampler(config)
        _target_depth, target_h, target_w = config.target_shape_dhw
        self.cropper = CenterCropper(target_h=target_h, target_w=target_w)
        self.normalizer = JointIntensityNormalizer(config)
        self.z_padder = ZPadder(
            target_depth=config.target_shape_dhw[0],
            padding_value=config.z_padding_value,
        )

    def preprocess(self, patient: AcdcPatient) -> PreprocessingResult:
        """Preprocess one indexed ACDC patient into [2, 14, 144, 144]."""
        ed_image = cast(nib.spatialimages.SpatialImage, nib.load(patient.ed_path))
        es_image = cast(nib.spatialimages.SpatialImage, nib.load(patient.es_path))
        ed_oriented = self.orientation_normalizer.normalize(ed_image)
        es_oriented = self.orientation_normalizer.normalize(es_image)
        self._validate_oriented_pair(patient.patient_id, ed_oriented, es_oriented)

        grid = self.resampler.derive_grid(ed_oriented)
        ed_resampled = self.resampler.resample(ed_oriented, grid)
        es_resampled = self.resampler.resample(es_oriented, grid)
        if ed_resampled.shape != es_resampled.shape:
            raise ValueError(f"{patient.patient_id}: ED/ES resampled shape mismatch")

        crop_window = self.cropper.derive_window(ed_resampled)
        ed_cropped = self.cropper.crop(ed_resampled, crop_window)
        es_cropped = self.cropper.crop(es_resampled, crop_window)
        valid_depth = ed_cropped.shape[2]

        ed_normalized, es_normalized, normalization = self.normalizer.normalize(
            ed_cropped,
            es_cropped,
        )
        ed_padded, z_padding = self.z_padder.pad(ed_normalized)
        es_padded, es_z_padding = self.z_padder.pad(es_normalized)
        if z_padding != es_z_padding:
            raise ValueError(f"{patient.patient_id}: ED/ES Z padding mismatch")

        ed_dhw = np.transpose(ed_padded, (2, 1, 0))
        es_dhw = np.transpose(es_padded, (2, 1, 0))
        tensor = np.ascontiguousarray(np.stack([ed_dhw, es_dhw], axis=0).astype(np.float32))
        if not np.isfinite(tensor).all():
            raise ValueError(f"{patient.patient_id}: preprocessed tensor is not finite")

        return PreprocessingResult(
            tensor=tensor,
            patient_id=patient.patient_id,
            source_shape_xyz=ed_oriented.array.shape,
            source_spacing_xyz=ed_oriented.spacing_xyz,
            resampled_shape_xyz=ed_resampled.shape,
            target_spacing_xyz=grid.spacing_xyz,
            crop_start_xy=crop_window.start_xy,
            crop_end_xy=crop_window.end_xy,
            valid_depth=valid_depth,
            z_padding_lower=z_padding.lower,
            z_padding_upper=z_padding.upper,
            clip_lower=normalization.clip_lower,
            clip_upper=normalization.clip_upper,
            normalization_mean=normalization.mean,
            normalization_std=normalization.std,
        )

    def _validate_oriented_pair(
        self,
        patient_id: str,
        ed_image: OrientedImage,
        es_image: OrientedImage,
    ) -> None:
        """Validate ED/ES geometry after explicit orientation normalization."""
        if ed_image.orientation != tuple(self.config.target_orientation):
            raise ValueError(f"{patient_id}: ED orientation is not LPS")
        if es_image.orientation != tuple(self.config.target_orientation):
            raise ValueError(f"{patient_id}: ES orientation is not LPS")
        if ed_image.array.shape != es_image.array.shape:
            raise ValueError(f"{patient_id}: ED/ES oriented shape mismatch")
        if ed_image.spacing_xyz != es_image.spacing_xyz:
            raise ValueError(f"{patient_id}: ED/ES oriented spacing mismatch")
        if not np.allclose(ed_image.affine[:3, :3], es_image.affine[:3, :3]):
            raise ValueError(f"{patient_id}: ED/ES oriented affine matrix mismatch")

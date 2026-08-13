"""Training dataset adapters built from existing ACDC preprocessing components."""

from collections.abc import Callable, Sequence

import torch
from torch import Tensor
from torch.utils.data import Dataset

from cardiac_pathology.data import AcdcPatient
from cardiac_pathology.preprocessing import PatientPreprocessor


class PreprocessedPatientDataset(Dataset[tuple[Tensor, Tensor]]):
    """Dataset that preprocesses real indexed ACDC patients on demand."""

    def __init__(
        self,
        patients: Sequence[AcdcPatient],
        preprocessor: PatientPreprocessor,
        augmentation: Callable[[Tensor], Tensor] | None = None,
    ) -> None:
        """Initialize an on-demand preprocessed patient dataset.

        Args:
            patients: Real indexed ACDC patient records to expose as supervised
                samples.
            preprocessor: Existing frozen preprocessing pipeline that converts
                one patient to a tensor with shape ``[2, 14, 144, 144]``.
            augmentation: Optional training-only transform applied after
                deterministic preprocessing. Validation and inference datasets
                must leave this as ``None``.

        Returns:
            None.
        """
        if not patients:
            raise ValueError("patients must be non-empty")

        self.patients = tuple(patients)
        self.preprocessor = preprocessor
        self.augmentation = augmentation

    def __len__(self) -> int:
        """Return the number of patient-level samples in the dataset."""
        return len(self.patients)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        """Load and preprocess one patient-level sample.

        Args:
            index: Integer patient index.

        Returns:
            Tuple containing an input tensor with shape ``[2, 14, 144, 144]`` and
            a scalar class-index tensor.
        """
        patient = self.patients[index]
        result = self.preprocessor.preprocess(patient)
        image = torch.from_numpy(result.tensor)

        if self.augmentation is not None:
            image = self.augmentation(image)

        target = torch.tensor(patient.class_index, dtype=torch.long)
        return image, target

"""Deterministic indexer for real ACDC patient metadata."""

import json
import re
from collections.abc import Mapping
from pathlib import Path

from cardiac_pathology.data.acdc_metadata_parser import AcdcMetadataParser
from cardiac_pathology.data.acdc_patient import AcdcPatient

PATIENT_ID_PATTERN = re.compile(r"^patient(?P<number>\d+)$")


class AcdcDatasetIndexer:
    """Build immutable patient records from an ACDC training directory."""

    def __init__(self, dataset_dir: Path, class_mapping_path: Path) -> None:
        self.dataset_dir = dataset_dir
        self.class_mapping_path = class_mapping_path
        self.metadata_parser = AcdcMetadataParser()

    def index_patients(self) -> tuple[AcdcPatient, ...]:
        """Index all patient directories deterministically by numeric patient ID."""
        if not self.dataset_dir.is_dir():
            raise FileNotFoundError(f"Dataset directory does not exist: {self.dataset_dir}")

        class_name_to_index = self._load_class_name_to_index()
        patient_dirs = sorted(
            self.dataset_dir.glob("patient*"),
            key=lambda path: self._patient_number(path.name),
        )
        if not patient_dirs:
            raise ValueError(f"No patient directories found in {self.dataset_dir}")

        patients: list[AcdcPatient] = []
        seen_patient_ids: set[str] = set()
        for patient_dir in patient_dirs:
            if not patient_dir.is_dir():
                continue

            patient_id = patient_dir.name
            if patient_id in seen_patient_ids:
                raise ValueError(f"Duplicate patient ID: {patient_id}")
            seen_patient_ids.add(patient_id)

            info_path = patient_dir / "Info.cfg"
            metadata = self.metadata_parser.parse(info_path)
            if metadata.group not in class_name_to_index:
                raise ValueError(f"{patient_id}: unknown diagnostic group {metadata.group!r}")

            ed_path = patient_dir / f"{patient_id}_frame{metadata.ed_frame:02d}.nii.gz"
            es_path = patient_dir / f"{patient_id}_frame{metadata.es_frame:02d}.nii.gz"
            cine_4d_path = patient_dir / f"{patient_id}_4d.nii.gz"
            patients.append(
                AcdcPatient(
                    patient_id=patient_id,
                    class_index=class_name_to_index[metadata.group],
                    class_name=metadata.group,
                    ed_frame=metadata.ed_frame,
                    es_frame=metadata.es_frame,
                    ed_path=ed_path,
                    es_path=es_path,
                    cine_4d_path=cine_4d_path,
                    info_path=info_path,
                )
            )

        return tuple(patients)

    def load_class_mapping(self) -> dict[int, str]:
        """Load and validate class mapping as integer index to class name."""
        with self.class_mapping_path.open("r", encoding="utf-8") as file:
            raw_mapping = json.load(file)

        if not isinstance(raw_mapping, dict) or not raw_mapping:
            raise ValueError(f"Malformed class mapping: {self.class_mapping_path}")

        mapping: dict[int, str] = {}
        seen_names: set[str] = set()
        for raw_index, raw_name in raw_mapping.items():
            try:
                class_index = int(raw_index)
            except (TypeError, ValueError) as error:
                raise ValueError(f"Class mapping key must be an integer: {raw_index!r}") from error

            if not isinstance(raw_name, str) or not raw_name.strip():
                raise ValueError(f"Class name for index {raw_index!r} must be non-empty")

            class_name = raw_name.strip()
            if class_name in seen_names:
                raise ValueError(f"Duplicate class mapping value: {class_name}")
            if class_index in mapping:
                raise ValueError(f"Duplicate class mapping index: {class_index}")

            seen_names.add(class_name)
            mapping[class_index] = class_name

        expected_indices = set(range(len(mapping)))
        if set(mapping) != expected_indices:
            raise ValueError(
                "Class mapping indices must be contiguous from 0 to "
                f"{len(mapping) - 1}: {mapping}"
            )

        return dict(sorted(mapping.items()))

    def _load_class_name_to_index(self) -> Mapping[str, int]:
        """Return validated class-name to class-index mapping."""
        return {
            class_name: class_index
            for class_index, class_name in self.load_class_mapping().items()
        }

    def _patient_number(self, patient_id: str) -> int:
        """Extract numeric patient identifier for deterministic sorting."""
        match = PATIENT_ID_PATTERN.match(patient_id)
        if match is None:
            raise ValueError(f"Malformed patient directory name: {patient_id}")
        return int(match.group("number"))
